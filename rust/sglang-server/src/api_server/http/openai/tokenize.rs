//! `/v1/tokenize` and `/v1/detokenize` (+ the unversioned aliases), Python's
//! `OpenAIServingTokenize` / `OpenAIServingDetokenize`: the same request and
//! response shapes and the same flat error body. Encoding runs on the
//! dedicated HTTP utility workers, including rendering, JSON parsing and serialization.

use super::chat::{parse_chat, prepare_prompt};
use super::{AppState, contains_media};
use crate::api_server::http::work::{RESPONSE_BYTES, leased_body};
use crate::message::types::{OneOrMany, TokenIds};
use axum::{
    Json, Router,
    extract::{Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::io::{self, Write};
use std::sync::Arc;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/tokenize", post(tokenize))
        .route("/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/detokenize", post(detokenize))
}

fn default_true() -> bool {
    true
}

/// Python's `TokenizeRequest`. Unknown keys are kept for the chat-template
/// path, as its `extra="allow"` does.
#[derive(Deserialize)]
struct TokenizeBody {
    #[serde(default)]
    prompt: Option<OneOrMany<String>>,
    #[serde(default)]
    messages: Option<Value>,
    #[serde(default = "default_true")]
    add_special_tokens: bool,
    #[serde(flatten)]
    rest: Map<String, Value>,
}

/// Python's `DetokenizeRequest`.
#[derive(Deserialize)]
struct DetokenizeBody {
    tokens: OneOrMany<TokenIds>,
    #[serde(default = "default_true")]
    skip_special_tokens: bool,
}

/// Python's `ErrorResponse`, which these endpoints return flat rather than
/// under an `error` key.
fn error_body(status: StatusCode, error_type: &str, message: impl Into<String>) -> Response {
    let body = json!({
        "object": "error",
        "message": message.into(),
        "type": error_type,
        "param": null,
        "code": status.as_u16(),
    });
    (status, Json(body)).into_response()
}

struct Failure(StatusCode, &'static str, String);
impl Failure {
    fn invalid(e: impl ToString) -> Self {
        Self(StatusCode::BAD_REQUEST, "Bad Request", e.to_string())
    }
    fn bad(e: impl ToString) -> Self {
        Self(StatusCode::BAD_REQUEST, "BadRequestError", e.to_string())
    }
    fn large() -> Self {
        Self(
            StatusCode::PAYLOAD_TOO_LARGE,
            "PayloadTooLarge",
            "utility batch or result exceeds its limit".into(),
        )
    }
    fn internal(e: impl ToString) -> Self {
        Self(
            StatusCode::INTERNAL_SERVER_ERROR,
            "InternalServerError",
            e.to_string(),
        )
    }
}

/// Serialize one item at a time so a batch of expanding decodes never builds
/// an unbounded intermediate array of strings or JSON values.
#[derive(Default)]
struct Output(Vec<u8>);
impl Write for Output {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() > RESPONSE_BYTES - self.0.len() {
            return Err(io::Error::other("response too large"));
        }
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Output {
    fn json(&mut self, v: &impl Serialize) -> Result<(), Failure> {
        serde_json::to_writer(self, v).map_err(|_| Failure::large())
    }
    fn push(&mut self, bytes: &[u8]) -> Result<(), Failure> {
        self.write_all(bytes).map_err(|_| Failure::large())
    }
}

async fn tokenize(State(state): State<Arc<AppState>>, req: Request) -> Response {
    utility(state, req, true).await
}
async fn detokenize(State(state): State<Arc<AppState>>, req: Request) -> Response {
    utility(state, req, false).await
}
async fn utility(state: Arc<AppState>, req: Request, encode: bool) -> Response {
    let worker_state = state.clone();
    let result = state.workers.utility(&req).run(req, move |bytes, lease, reply| {
        let result = if encode { encode_body(&worker_state, &bytes, || reply.is_closed()) }
                     else { decode_body(&worker_state, &bytes, || reply.is_closed()) };
        let (status, data) = match result {
            Ok(out) => (StatusCode::OK, out.0),
            Err(Failure(status, typ, mut msg)) => {
                // Some serde errors contain the entire offending string.
                if msg.len() > 2048 { msg.truncate(msg.floor_char_boundary(2048)); }
                (status, serde_json::to_vec(&json!({"object":"error","message":msg,"type":typ,"param":null,"code":status.as_u16()})).unwrap())
            }
        };
        (status, [(header::CONTENT_TYPE, "application/json")], leased_body(data, lease)).into_response()
    }).await;
    match result {
        Ok(response) => response,
        Err(error) => error_body(
            error.0,
            error.0.canonical_reason().unwrap_or("Error"),
            error.1,
        ),
    }
}

fn encode_body(
    state: &AppState,
    bytes: &[u8],
    cancelled: impl Fn() -> bool,
) -> Result<Output, Failure> {
    let body: TokenizeBody = serde_json::from_slice(bytes).map_err(Failure::invalid)?;
    let codec = || {
        state.codec.as_ref().ok_or_else(|| {
            Failure::bad("tokenization is unavailable when skip_tokenizer_init=True")
        })
    };
    let encode = |text: &str, specials| codec()?.encode(text, specials).map_err(Failure::internal);
    let mut out = Output::default();
    out.push(br#"{"tokens":"#)?;
    match (body.prompt, body.messages) {
        (Some(OneOrMany::Many(texts)), None) => {
            if texts.len() > 1024 {
                return Err(Failure::large());
            }
            let mut counts = Vec::with_capacity(texts.len());
            out.push(b"[")?;
            for text in texts {
                if cancelled() {
                    return Err(Failure::bad("request cancelled"));
                }
                let ids = encode(&text, body.add_special_tokens)?;
                if !counts.is_empty() {
                    out.push(b",")?;
                }
                counts.push(ids.len());
                out.json(&ids)?;
            }
            out.push(br#"],"count":"#)?;
            out.json(&counts)?;
        }
        (prompt, messages) => {
            let ids = match (prompt, messages) {
                (Some(OneOrMany::One(text)), None) => encode(&text, body.add_special_tokens)?,
                (None, Some(messages)) => {
                    let mut fields = body.rest;
                    fields.insert("messages".into(), messages);
                    fields
                        .entry("model")
                        .or_insert_with(|| Value::from("default"));
                    let raw = Value::Object(fields);
                    let chat = parse_chat(state, &raw).map_err(Failure::invalid)?;
                    if chat.request.messages.is_empty() {
                        return Err(Failure::bad("messages cannot be empty"));
                    }
                    if contains_media(&raw["messages"]) {
                        return Err(Failure::bad(
                            "image, audio, video, and file message content is not supported",
                        ));
                    }
                    let (text, ids) =
                        prepare_prompt(state, raw, &chat.request, chat.chat_template_kwargs)
                            .map_err(Failure::bad)?;
                    match ids {
                        Some(ids) => ids,
                        None => encode(&text, false)?,
                    }
                }
                _ => {
                    return Err(Failure::invalid(
                        "Exactly one of 'prompt' or 'messages' must be provided.",
                    ));
                }
            };
            out.json(&ids)?;
            out.push(br#","count":"#)?;
            out.json(&ids.len())?;
        }
    }
    out.push(br#","max_model_len":"#)?;
    out.json(&state.server_args.tokenizer_model_max_length)?;
    out.push(b"}")?;
    Ok(out)
}

fn decode_body(
    state: &AppState,
    bytes: &[u8],
    cancelled: impl Fn() -> bool,
) -> Result<Output, Failure> {
    let body: DetokenizeBody = serde_json::from_slice(bytes).map_err(Failure::invalid)?;
    let decode = |tokens: TokenIds| {
        let ids = tokens
            .into_iter()
            .map(|id| {
                u32::try_from(id)
                    .map_err(|_| Failure::bad(format!("Token ID {id} is out of range")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        state
            .codec
            .as_ref()
            .ok_or_else(|| {
                Failure::bad("detokenization is unavailable when skip_tokenizer_init=True")
            })?
            .decode(&ids, body.skip_special_tokens)
            .map_err(Failure::internal)
    };
    let mut out = Output::default();
    out.push(br#"{"text":"#)?;
    match body.tokens {
        OneOrMany::One(tokens) if tokens.is_empty() => out.json(&"")?,
        OneOrMany::One(tokens) => out.json(&decode(tokens)?)?,
        OneOrMany::Many(lists) => {
            if lists.len() > 1024 {
                return Err(Failure::large());
            }
            out.push(b"[")?;
            for (i, list) in lists.into_iter().enumerate() {
                if cancelled() {
                    return Err(Failure::bad("request cancelled"));
                }
                if i > 0 {
                    out.push(b",")?;
                }
                out.json(&decode(list)?)?;
            }
            out.push(b"]")?;
        }
    }
    out.push(b"}")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::super::template_loader::load_chat_formatter;
    use super::super::test_utils::{body_json, post_json};
    use super::*;
    use crate::api_server::core::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::api_server::http::work::Codec;
    use crate::message::config::ServerArgs;
    struct Words;
    impl Codec for Words {
        fn encode(&self, text: &str, specials: bool) -> Result<Vec<i64>, String> {
            Ok(specials
                .then_some(0)
                .into_iter()
                .chain(text.split_whitespace().map(|w| w.len() as i64))
                .collect())
        }
        fn decode(&self, ids: &[u32], skip: bool) -> Result<String, String> {
            Ok(format!("{ids:?} {skip}"))
        }
    }
    fn answering_state() -> Arc<AppState> {
        // No intake receiver: routing either utility through generation fails this test.
        let intake_tx = flume::unbounded().0;
        let server_args = Arc::new(ServerArgs {
            tokenizer_model_max_length: 131072,
            ..Default::default()
        });
        Arc::new(AppState {
            core: CoreHandle::new(
                intake_tx,
                flume::unbounded().0,
                CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: CoreMetadata::from(server_args.as_ref()),
                },
            ),
            generation_policy: crate::message::request::GenerationPolicy::from(
                server_args.as_ref(),
            ),
            server_args,
            workers: Default::default(),
            codec: Some(Arc::new(Words)),
            chat_formatter: Some(
                load_chat_formatter(
                    None,           /* config_file */
                    None,           /* model_path */
                    None,           /* model_type */
                    Some("chatml"), /* chat_template_arg */
                )
                .unwrap(),
            ),
        })
    }

    async fn call(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let app = routes().with_state(answering_state());
        let response = post_json(app, path, body).await;
        (response.status(), body_json(response).await)
    }

    #[tokio::test]
    async fn tokenize_returns_python_response_shape() {
        for path in ["/v1/tokenize", "/tokenize"] {
            let (status, body) = call(path, json!({"model": "m", "prompt": "a bcd"})).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                body.to_string(),
                r#"{"tokens":[0,1,3],"count":3,"max_model_len":131072}"#
            );
        }
        let (_, body) = call(
            "/v1/tokenize",
            json!({"prompt": ["a", "bb cc"], "add_special_tokens": false}),
        )
        .await;
        assert_eq!(body["tokens"], json!([[1], [2, 2]]));
        assert_eq!(body["count"], json!([1, 2]));
        let (_, body) = call("/v1/tokenize", json!({"prompt": []})).await;
        assert_eq!((&body["tokens"], &body["count"]), (&json!([]), &json!([])));

        // `messages` render through the chat template, whose output keeps
        // its own specials: encoded without auto-added ones.
        let (status, body) = call(
            "/v1/tokenize",
            json!({"messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_ne!(body["tokens"][0], 0, "no auto BOS on a rendered template");
        assert_eq!(body["count"], body["tokens"].as_array().unwrap().len());
    }

    #[tokio::test]
    async fn detokenize_returns_python_response_shape() {
        for path in ["/v1/detokenize", "/detokenize"] {
            let (status, body) = call(path, json!({"tokens": [1, 2]})).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, json!({"text": "[1, 2] true"}));
        }
        let (_, body) = call(
            "/v1/detokenize",
            json!({"tokens": [[1], []], "skip_special_tokens": false}),
        )
        .await;
        assert_eq!(body, json!({"text": ["[1] false", "[] false"]}));
        let (_, body) = call("/v1/detokenize", json!({"tokens": []})).await;
        assert_eq!(body, json!({"text": ""}));
    }

    /// Bad bodies get Python's flat `ErrorResponse` with a 400.
    #[tokio::test]
    async fn invalid_requests_get_the_flat_python_error() {
        for (path, body) in [
            ("/v1/tokenize", json!({"model": "m"})),
            (
                "/v1/tokenize",
                json!({"prompt": "a", "messages": [{"role": "user", "content": "hi"}]}),
            ),
            ("/v1/tokenize", json!({"prompt": 3})),
            ("/v1/detokenize", json!({"tokens": [1, [2]]})),
            ("/v1/detokenize", json!({})),
        ] {
            let (status, error) = call(path, body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(error["object"], "error", "{body}");
            assert_eq!(error["type"], "Bad Request", "{body}");
            assert_eq!(error["code"], 400, "{body}");
            assert!(error["param"].is_null(), "{body}");
        }
        let (status, error) = call("/v1/tokenize", json!({"messages": []})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["type"], "BadRequestError");
    }
    #[tokio::test]
    async fn aigate_count_and_pd_chat_use_identical_native_prompt_ids() {
        use super::super::chat::prepare_chat;
        let mut state = answering_state();
        let inner = Arc::get_mut(&mut state).unwrap();
        inner.chat_formatter = Some(super::super::ChatFormatter::DeepSeek(
            sglang_processor::ChatFormatter::DeepSeekV41,
        ));
        let args = Arc::get_mut(&mut inner.server_args);
        // The immutable model config is also held by the generation policy on some builds.
        if let Some(args) = args {
            args.bos_token_id = Some(0);
        }
        let body = json!({"model":"m", "messages":[{"role":"user","content":"中文 test"}],
            "rid":"client-17","bootstrap_host":"prefill","bootstrap_port":8998,"bootstrap_room":9007199254740993_i64,
            "bootstrap_pair_key":"pair-17","decode_tp_size":16,"data_parallel_rank":0,"disagg_prefill_dp_rank":0,
            "min_tokens":2,"max_tokens":4,"ignore_eos":true});
        let response = post_json(
            routes().with_state(state.clone()),
            "/v1/tokenize",
            body.clone(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let counted = body_json(response).await;
        let (requests, _) = prepare_chat(&state, body.clone()).unwrap();
        let req = &requests[0];
        assert_eq!(req.rid.client_facing(), "client-17");
        assert_eq!(json!(req.input_ids), counted["tokens"]);
        assert_eq!(counted["count"], req.input_ids.as_ref().unwrap().len());
        assert_eq!(req.bootstrap_pair_key.as_deref(), Some("pair-17"));
        assert_eq!(
            (req.decode_tp_size, req.routed_dp_rank, req.bootstrap_room),
            (Some(16), Some(0), Some(9007199254740993))
        );
        assert_eq!(req.sampling_params.min_new_tokens, 2);
        assert!(req.sampling_params.ignore_eos);
        // DeepSeek's native body accepts Python tool history shapes and numeric effort.
        let mut history = body.clone();
        history["reasoning_effort"] = json!(0.33);
        history["messages"] = json!([
            {"role":"user","content":"Hi"},
            {"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"a","content":[{"type":"text","text":"snow"}]}
        ]);
        let counted = post_json(
            routes().with_state(state.clone()),
            "/v1/tokenize",
            history.clone(),
        )
        .await;
        assert_eq!(counted.status(), StatusCode::OK);
        let (requests, _) = prepare_chat(&state, history).unwrap();
        assert_eq!(
            body_json(counted).await["tokens"],
            json!(requests[0].input_ids)
        );
        for (field, value) in [("data_parallel_rank", json!(1)), ("n", json!(2))] {
            let mut invalid = body.clone();
            invalid[field] = value;
            assert!(prepare_chat(&state, invalid).is_err());
        }
    }
}
