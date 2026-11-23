//! OpenAI Chat Completions endpoint and chat-template preparation.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::Request;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::post,
};
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatChoice, ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionResponseMessage, ChatCompletionTokenLogprob, ChatCompletionToolChoiceOption,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason, ResponseFormat, Role, ServiceTier as ChatServiceTier, Stop,
    TopLogprobs,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use sglang_processor::{
    ReasoningOptions, ReasoningStreamSplitter, dynamo_tool_choice, split_reasoning,
    tool_call_stream,
};

use super::completions::completion_usage;
use super::pd_routing::PDRoutingFields;
use super::tools::{apply_tool_constraint, chat_delta, chat_finish_reason, parse_chat_tool_calls};
use super::{
    AppState, ChatFormatter, ChatTemplateKwargs, collect_output, contains_media, error_payload,
    indexed_decode_stream, model_error, openai_error, submit_generation, unix_seconds_u32,
};
use crate::api_server::core::{CoreCall, CoreEvent};
use crate::message::config::{DefaultSamplingParams, ServerArgs};
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;
use crate::message::response::ChunkExtras;
use crate::message::sampling::SamplingParams;
use crate::message::types::OneOrMany;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

#[derive(Deserialize)]
pub(super) struct ChatRequest {
    #[serde(flatten)]
    pub(super) request: CreateChatCompletionRequest,
    pub(super) chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(default)]
    min_tokens: i64,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(alias = "request_id")]
    rid: Option<String>,
    bootstrap_pair_key: Option<String>,
    decode_tp_size: Option<i64>,
    #[serde(flatten)]
    routing: PDRoutingFields,
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: Request) -> Response {
    let prepare_state = state.clone();
    let prepared = state
        .workers
        .generation
        .run(body, move |bytes, _lease, _| {
            let raw: Value = serde_json::from_slice(&bytes)
                .map_err(|e| openai_error(StatusCode::BAD_REQUEST, e.to_string(), false))?;
            prepare_chat(&prepare_state, raw)
        })
        .await;
    let (requests, reply) = match prepared {
        Ok(Ok(result)) => result,
        Ok(Err(response)) => return response,
        Err(error) => return openai_error(error.0, error.1, false),
    };
    let mut submitted = Vec::with_capacity(requests.len());
    for (index, native) in requests.into_iter().enumerate() {
        match submit_generation(&state, native, reply.stream).await {
            Ok(call) => submitted.push((index, call)),
            Err(response) => return response,
        }
    }
    reply.respond(submitted).await
}

/// The native encoder consumes the original body. Adapt only the copy Dynamo
/// uses for sampling, response options and tool definitions: Python accepts
/// additional tool-history and effort shapes that its OpenAI structs cannot read.
pub(super) fn parse_chat(state: &AppState, body: &Value) -> Result<ChatRequest, String> {
    let mut options = body.clone();
    if matches!(state.chat_formatter, Some(ChatFormatter::DeepSeek(_))) {
        if options["reasoning_effort"].is_number()
            || options["reasoning_effort"]
                .as_str()
                .is_some_and(|s| s.parse::<f64>().is_ok())
        {
            options.as_object_mut().unwrap().remove("reasoning_effort");
        }
        if let Some(tools) = options.get_mut("tools").and_then(Value::as_array_mut) {
            for tool in tools {
                if tool.get("type").is_none_or(|kind| kind == "function") {
                    *tool = sglang_processor::deepseek_tool_payload(tool)?;
                }
            }
        }
        if let Some(messages) = options.get_mut("messages").and_then(Value::as_array_mut) {
            for message in messages {
                if message["role"] == "tool"
                    && (message["content"].is_null() || message["content"].is_array())
                {
                    // Only the native renderer consumes tool-result content.
                    message["content"] = Value::String(String::new());
                }
                if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    for call in calls {
                        if let Some(function) =
                            call.get_mut("function").and_then(Value::as_object_mut)
                            && function.get("arguments").is_none_or(Value::is_null)
                        {
                            function.insert("arguments".into(), Value::String("{}".into()));
                        }
                    }
                }
            }
        }
    }
    serde_json::from_value(options).map_err(|e| e.to_string())
}

pub(super) fn prepare_chat(
    state: &AppState,
    raw: Value,
) -> Result<(Vec<GenerateRequest>, ChatReply), Response> {
    let ChatRequest {
        request,
        chat_template_kwargs,
        routing,
        min_tokens,
        ignore_eos,
        rid,
        bootstrap_pair_key,
        decode_tp_size,
    } = parse_chat(state, &raw).map_err(|e| openai_error(StatusCode::BAD_REQUEST, e, false))?;
    if let Some(message) = model_error(&request.model, &state.server_args.served_model_name) {
        return Err(openai_error(StatusCode::BAD_REQUEST, message, false));
    }
    if request.messages.is_empty() {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "messages cannot be empty",
            false,
        ));
    }
    if contains_media(&raw["messages"]) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "image, audio, video, and file message content is not supported",
            false,
        ));
    }
    if request.n == Some(0) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "n must be at least 1",
            false,
        ));
    }
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "max_completion_tokens must be positive",
            false,
        ));
    }
    if request.modalities.as_ref().is_some_and(|modalities| {
        serde_json::to_value(modalities).is_ok_and(|value| value.to_string().contains("\"audio\""))
    }) || request.audio.is_some()
        || request.prediction.is_some()
        || request.web_search_options.is_some()
        || request.mm_processor_kwargs.is_some()
    {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "audio, prediction, web search, and multimodal inputs are not supported",
            false,
        ));
    }
    #[allow(deprecated)]
    if request.function_call.is_some() || request.functions.is_some() {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "deprecated function_call/functions are not supported; use tools and tool_choice",
            false,
        ));
    }

    let tool_choice = dynamo_tool_choice(&request.tool_choice);
    let tools = match sglang_processor::chat_tool_definitions(
        request.tools.as_deref(),
        &request.messages,
    ) {
        Ok(tools) => tools,
        Err(error) => {
            return Err(openai_error(
                StatusCode::BAD_REQUEST,
                error.to_string(),
                false,
            ));
        }
    };
    let tools_enabled = !tools.is_empty() && tool_choice != DynamoToolChoice::None;
    let parser = tools_enabled
        .then(|| state.server_args.tool_call_parser.clone())
        .flatten();
    if tools_enabled && parser.is_none() {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "tool calls require --tool-call-parser",
            false,
        ));
    }
    // Python gates the split on `request.separate_reasoning` (default true);
    // the Dynamo request type has no such field, so it is always on when the
    // server was launched with `--reasoning-parser`.
    let reasoning_parser = state.server_args.reasoning_parser.clone();
    let tools = (!tools.is_empty()).then_some(tools);
    let tools_slice = tools.as_deref().unwrap_or_default();

    let mut request = request;
    let formatter = state.chat_formatter.as_ref().ok_or_else(|| {
        openai_error(
            StatusCode::BAD_REQUEST,
            "this model has no usable chat template",
            false,
        )
    })?;
    merge_template_stops(&mut request, formatter);
    let (prompt, ids) = prepare_prompt(state, raw, &request, chat_template_kwargs)
        .map_err(|e| openai_error(StatusCode::BAD_REQUEST, e, false))?;

    let mut sampling = match chat_sampling(
        &request,
        SamplingDefaults::CHAT,
        parser.as_deref(),
        &tool_choice,
        tools_slice,
        request.parallel_tool_calls,
        &state.server_args,
        min_tokens,
        ignore_eos,
    ) {
        Ok(sampling) => sampling,
        Err(message) => {
            return Err(openai_error(StatusCode::BAD_REQUEST, message, false));
        }
    };

    if ids.is_some()
        && let Some(codec) = &state.codec
        && let Some(window) = sampling
            .stop_strs
            .iter()
            .map(|s| codec.encode(s, true).map_or(s.len(), |ids| ids.len()))
            .max()
    {
        sampling.stop_str_max_len = window;
    }
    let stream = request.stream.unwrap_or(false);
    let prompt_count = 1;
    let choices_per_prompt = request.n.unwrap_or(1) as usize;
    let routing = match routing.normalize(prompt_count, choices_per_prompt) {
        Ok(routing) => routing,
        Err(error) => {
            return Err(openai_error(
                StatusCode::BAD_REQUEST,
                error.to_string(),
                false,
            ));
        }
    };
    // Chat normalizes one prompt, so each bootstrap column has one entry.
    let bootstrap_host = &routing.bootstrap.bootstrap_hosts[0];
    let bootstrap_port = routing.bootstrap.bootstrap_ports[0];
    let bootstrap_room = routing.bootstrap.bootstrap_rooms[0];
    let pd = state.server_args.is_disaggregation()
        || bootstrap_host.is_some()
        || bootstrap_room.is_some()
        || bootstrap_pair_key.is_some()
        || bootstrap_port.is_some();
    if pd
        && (choices_per_prompt != 1
            || bootstrap_host.as_ref().is_none_or(|s| s.trim().is_empty())
            || bootstrap_room.is_none_or(|r| r < 0))
    {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "PD chat requires one choice, bootstrap_host and a nonnegative bootstrap_room",
            false,
        ));
    }
    if bootstrap_pair_key
        .as_ref()
        .is_some_and(|s| s.is_empty() || s.len() > 128 || !s.is_ascii())
        || decode_tp_size.is_some_and(|n| n <= 0)
        || bootstrap_port.is_some_and(|n| !(1..=65535).contains(&n))
    {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "invalid PD bootstrap metadata",
            false,
        ));
    }
    state
        .server_args
        .check_ingress_rank(routing.routed_dp_rank)
        .map_err(|e| openai_error(StatusCode::BAD_REQUEST, e, false))?;
    crate::message::request::check_broadcast_budget(
        ids.as_ref().map_or(prompt.len(), |v| v.len() * 8),
        choices_per_prompt,
        "chat prompt",
    )
    .map_err(|e| openai_error(StatusCode::BAD_REQUEST, e.to_string(), false))?;
    let want_logprobs = request.logprobs.unwrap_or(false);
    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let stream_tool_choice = request.tool_choice.clone();
    let uses_tool_call_structural_tag = sampling.structural_tag.is_some();
    let service_tier = request.service_tier;
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let model = request.model;
    let include_usage = request
        .stream_options
        .is_some_and(|options| options.include_usage)
        || state.server_args.stream_response_default_include_usage;
    let mut requests = Vec::with_capacity(choices_per_prompt);

    let starts_in_reasoning = starts_in_reasoning(reasoning_parser.as_deref(), &prompt);
    let mut input = Some((ids.is_none().then_some(prompt), ids));
    for index in 0..choices_per_prompt {
        let rid = match &rid {
            Some(rid) if choices_per_prompt == 1 => Rid::from_client(rid),
            _ => Rid::from_client(&format!(
                "{}-{index}",
                rid.as_deref().unwrap_or(&response_id)
            )),
        };
        let (text, input_ids) = if index + 1 == choices_per_prompt {
            input.take().unwrap()
        } else {
            input.as_ref().unwrap().clone()
        };
        let native = GenerateRequest {
            rid,
            text,
            input_ids,
            // Rendered templates own their special tokens — the pool must not
            // add another BOS/EOS (Python's `add_special_tokens=False`).
            skip_special_tokens: true,
            sampling_params: sampling.clone(),
            stream,
            return_logprob: want_logprobs,
            logprob_start_len: -1,
            top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
            return_text_in_logprobs: want_logprobs.then_some(true),
            bootstrap_host: bootstrap_host.clone(),
            bootstrap_pair_key: bootstrap_pair_key.clone(),
            decode_tp_size,
            bootstrap_port,
            bootstrap_room,
            routed_dp_rank: routing.routed_dp_rank,
            disagg_prefill_dp_rank: routing.disagg_prefill_dp_rank,
            ..Default::default()
        };
        requests.push(native);
    }
    Ok((
        requests,
        ChatReply {
            stream,
            response_id,
            model,
            created,
            want_logprobs,
            include_usage,
            parser,
            reasoning_parser,
            starts_in_reasoning,
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            parallel_tool_calls,
            service_tier,
        },
    ))
}

pub(super) struct ChatReply {
    stream: bool,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    include_usage: bool,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    starts_in_reasoning: bool,
    tools: Option<Vec<ToolDefinition>>,
    stream_tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
}

impl ChatReply {
    async fn respond(self, submitted: Vec<(usize, CoreCall)>) -> Response {
        let Self {
            stream,
            response_id,
            model,
            created,
            want_logprobs,
            include_usage,
            parser,
            reasoning_parser,
            starts_in_reasoning,
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            parallel_tool_calls,
            service_tier,
        } = self;
        if stream {
            let event_stream = chat_event_stream(
                submitted,
                response_id,
                model,
                created,
                want_logprobs,
                include_usage,
                parser,
                reasoning_parser,
                starts_in_reasoning,
                tools,
                stream_tool_choice,
                uses_tool_call_structural_tag,
                parallel_tool_calls,
                service_tier,
            )
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
            Sse::new(event_stream).into_response()
        } else {
            unary_chat(
                submitted,
                response_id,
                model,
                created,
                want_logprobs,
                parser,
                reasoning_parser,
                starts_in_reasoning,
                tools,
                parallel_tool_calls,
                service_tier,
            )
            .await
        }
    }
}

/// Both generation and AIgate's count endpoint use this exact prompt preparation.
/// Native DeepSeek renders the original request, including message-level tools.
/// Python appends a continuation encoded separately, removing at most one BOS.
pub(super) fn prepare_prompt(
    state: &AppState,
    raw: Value,
    request: &CreateChatCompletionRequest,
    kwargs: Option<ChatTemplateKwargs>,
) -> Result<(String, Option<Vec<i64>>), String> {
    let formatter = state
        .chat_formatter
        .as_ref()
        .ok_or("this model has no usable chat template")?;
    let defaults = &state.server_args.default_chat_template_kwargs;
    if let ChatFormatter::DeepSeek(native) = formatter {
        let (prompt, suffix) = native
            .render_request(raw, defaults)
            .map_err(|e| e.to_string())?;
        let text = prompt.into_text();
        let codec = state
            .codec
            .as_ref()
            .ok_or("tokenization is unavailable when skip_tokenizer_init=True")?;
        let mut ids = codec.encode(&text, true)?;
        if !suffix.is_empty() {
            let extra = codec.encode(&suffix, true)?;
            let skip = usize::from(
                extra
                    .first()
                    .is_some_and(|id| Some(*id) == state.server_args.bos_token_id),
            );
            ids.extend_from_slice(&extra[skip..]);
        }
        return Ok((text, Some(ids)));
    }
    let mut merged = defaults.clone();
    merged.extend(kwargs.into_iter().flatten());
    let text = formatter
        .render(request, (!merged.is_empty()).then_some(&merged))
        .map_err(|e| format!("chat template render failed: {e}"))?;
    let ids = state
        .codec
        .as_ref()
        .map(|codec| codec.encode(&text, false))
        .transpose()?;
    Ok((text, ids))
}

/// Full sampling resolution for an OpenAI request, mirroring the Python
/// handler: endpoint defaults → tool-choice validation + constraint → clamp.
/// The tool-choice checks run regardless of whether a parser is configured
/// (see `apply_tool_constraint`).
pub(super) fn chat_sampling(
    request: &CreateChatCompletionRequest,
    defaults: SamplingDefaults,
    parser: Option<&str>,
    tool_choice: &DynamoToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    server_args: &ServerArgs,
    min_tokens: i64,
    ignore_eos: bool,
) -> Result<SamplingParams, String> {
    let mut sampling = chat_sampling_params(
        request,
        &defaults.with_model_defaults(&server_args.model_config.default_sampling_params),
    )?;
    sampling.min_new_tokens = min_tokens;
    sampling.ignore_eos = ignore_eos;
    apply_tool_constraint(
        &mut sampling,
        parser,
        tool_choice,
        tools,
        parallel_tool_calls,
    )?;
    sampling
        .normalize(
            server_args.skip_tokenizer_init,
            server_args.model_config.vocab_size,
        )
        .map_err(|error| error.to_string())?;
    Ok(sampling)
}

/// Merge the formatter's template stops into the request's `stop`.
///
/// Python `_apply_conversation_template`: `stop = copy.copy(conv.stop_str or [])
/// + request.stop` (a string request stop appends as one entry). Without this,
/// generation with a legacy/builtin template would run past the template's own
/// delimiters (e.g. chatml's `<|im_end|>`) whenever they are not model EOS ids.
fn merge_template_stops(request: &mut CreateChatCompletionRequest, formatter: &ChatFormatter) {
    let Some(template_stops) = formatter.stop_strs() else {
        return;
    };
    let mut stops = match template_stops {
        OneOrMany::One(one) => vec![one],
        OneOrMany::Many(many) => many,
    };
    if let Some(request_stop) = &request.stop {
        let Some(request_stops) = request_stop.strings() else {
            return;
        };
        stops.extend(request_stops);
    }
    request.stop = Some(Stop::StringArray(stops));
}

/// Where an omitted `temperature` / `top_p` gets its value. Mirrors Python's
/// `to_sampling_params` priority: user value > model generation_config (when
/// `--sampling-defaults model`) > OpenAI terminal default
/// (`_DEFAULT_SAMPLING_PARAMS`: chat uses 1.0/1.0).
pub(super) struct SamplingDefaults {
    /// Model defaults; `None` when the model config doesn't set them or when
    /// `--sampling-defaults openai` (the Python dump is then empty).
    temperature: Option<f64>,
    top_p: Option<f64>,
    /// OpenAI terminal defaults for chat completions.
    fallback_temperature: f64,
    fallback_top_p: f64,
}

impl SamplingDefaults {
    /// `protocol.py` chat `_DEFAULT_SAMPLING_PARAMS`: temperature 1.0, top_p 1.0.
    pub(super) const CHAT: SamplingDefaults = SamplingDefaults {
        temperature: None,
        top_p: None,
        fallback_temperature: 1.0,
        fallback_top_p: 1.0,
    };
    /// The resolved model defaults (empty in `--sampling-defaults openai`
    /// mode), which slot between the user's values and the OpenAI terminals.
    pub(super) fn with_model_defaults(mut self, model: &DefaultSamplingParams) -> SamplingDefaults {
        self.temperature = model.temperature;
        self.top_p = model.top_p;
        self
    }
}

#[allow(deprecated)]
pub(super) fn chat_sampling_params(
    request: &CreateChatCompletionRequest,
    defaults: &SamplingDefaults,
) -> Result<SamplingParams, String> {
    let mut stop = None;
    let mut stop_token_ids = None;
    match request.stop.as_ref() {
        Some(Stop::String(value)) => stop = Some(OneOrMany::One(value.clone())),
        Some(Stop::StringArray(values)) => stop = Some(OneOrMany::Many(values.clone())),
        Some(Stop::TokenIdArray(values)) => {
            stop_token_ids = Some(values.iter().map(|&id| id as i64).collect())
        }
        None => {}
    }
    let mut logit_bias = BTreeMap::new();
    if let Some(values) = request.logit_bias.as_ref() {
        for (token, bias) in values {
            let bias = bias
                .as_f64()
                .ok_or_else(|| format!("logit_bias[{token:?}] must be a number"))?;
            logit_bias.insert(token.clone(), bias);
        }
    }
    let json_schema = match request.response_format.as_ref() {
        Some(ResponseFormat::JsonSchema { json_schema }) => Some(json_schema.schema.to_string()),
        Some(ResponseFormat::JsonObject) => Some(r#"{"type":"object"}"#.into()),
        _ => None,
    };

    Ok(SamplingParams {
        max_new_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .map(i64::from),
        stop,
        stop_token_ids,
        temperature: request
            .temperature
            .map(f64::from)
            .or(defaults.temperature)
            .unwrap_or(defaults.fallback_temperature),
        top_p: request
            .top_p
            .map(f64::from)
            .or(defaults.top_p)
            .unwrap_or(defaults.fallback_top_p),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: request.presence_penalty.unwrap_or(0.0) as f64,
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        json_schema,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn unary_chat(
    submitted: Vec<(usize, CoreCall)>,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    starts_in_reasoning: bool,
    tools: Option<Vec<ToolDefinition>>,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> Response {
    let reasoning_options = ReasoningOptions {
        force_reasoning: starts_in_reasoning.then_some(true),
        ..Default::default()
    };
    let mut choices = Vec::with_capacity(submitted.len());
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0u64;

    for (index, call) in submitted {
        let output = match collect_output(call).await {
            Ok(output) => output,
            Err((status, message)) => {
                return openai_error(status, message, false);
            }
        };

        if prompt_tokens == 0 {
            prompt_tokens = output.prompt_tokens;
        }
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
        let finish_reason = chat_finish_reason(&output);
        // Split reasoning markers out of the content first (Python splits
        // before tool-call parsing too), then parse tool calls on the clean
        // normal text.
        let (reasoning_text, text) = split_reasoning(
            reasoning_parser.as_deref(),
            &reasoning_options,
            &output.text,
            output.stop_trimmed_token_ids(),
        );
        let (content, tool_calls) = parse_chat_tool_calls(
            text,
            parser.as_deref(),
            tools.as_deref(),
            parallel_tool_calls,
        )
        .await;
        let finish_reason = if tool_calls.is_some() {
            Some(OpenAIFinishReason::ToolCalls)
        } else {
            finish_reason
        };
        #[allow(deprecated)]
        let message = ChatCompletionResponseMessage {
            content: (!content.is_empty()).then_some(ChatCompletionMessageContent::Text(content)),
            refusal: None,
            tool_calls,
            role: Role::Assistant,
            function_call: None,
            audio: None,
            // Python: `reasoning_text if reasoning_text else None`.
            reasoning_content: (!reasoning_text.is_empty()).then_some(reasoning_text),
        };
        choices.push(ChatChoice {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            message,
            finish_reason,
            logprobs,
        });
    }

    Json(CreateChatCompletionResponse {
        id: response_id,
        choices,
        created,
        model,
        service_tier,
        system_fingerprint: None,
        object: "chat.completion".into(),
        usage: Some(completion_usage(
            prompt_tokens,
            u32::try_from(completion_tokens).unwrap_or(u32::MAX),
        )),
    })
    .into_response()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn chat_event_stream(
    submitted: Vec<(usize, CoreCall)>,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    include_usage: bool,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    starts_in_reasoning: bool,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> impl futures::Stream<Item = String> {
    let count = submitted.len();
    let raw = async_stream::stream! {
        let count = submitted.len();
        let mut streams = Vec::with_capacity(count);
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u64;
        // One stateful reasoning splitter per choice (Python keeps a
        // `reasoning_parser_dict` per index).
        let mut reasoning_splitters: Vec<ReasoningStreamSplitter> =
            if reasoning_parser.is_some() {
                (0..count)
                    .map(|_| ReasoningStreamSplitter::new(reasoning_parser.as_deref(), ReasoningOptions {
                        force_reasoning: starts_in_reasoning.then_some(true),
                        ..Default::default()
                    }))
                    .collect()
            } else {
                vec![]
            };
        let reasoning_enabled = !reasoning_splitters.is_empty();

        for (index, call) in submitted {
            streams.push(indexed_decode_stream(index, call));
            yield Annotated {
                data: Some(CreateChatCompletionStreamResponse {
                    id: response_id.clone(),
                    choices: vec![ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, Some(Role::Assistant), None, None),
                        finish_reason: None,
                        logprobs: None,
                    }],
                    created,
                    model: model.clone(),
                    service_tier: service_tier.clone(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".into(),
                    usage: None,
                }),
                id: None,
                event: None,
                comment: None,
                error: None,
            };
        }

        let mut events = futures::stream::select_all(streams);
        while let Some((index, event)) = events.next().await {
            let output = match event {
                CoreEvent::Delta(output) | CoreEvent::Finished(output) => output,
                CoreEvent::Failed(error) => {
                    yield Annotated {
                        data: None,
                        id: None,
                        event: None,
                        comment: None,
                        error: Some(error_payload(error.http_status(), error.to_string()).to_string()),
                    };
                    continue;
                }
            };
            if prompt_tokens == 0 {
                prompt_tokens = output.prompt_tokens;
            }
            completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
            let finish_reason = chat_finish_reason(&output);
            // Split the step's text into (reasoning, normal) deltas when
            // `--reasoning-parser` is set. Mirrors Python's per-step emission:
            // reasoning chunk first (logprobs ride it), then the content chunk.
            let mut emitted = Vec::with_capacity(2);
            if reasoning_enabled {
                let (reasoning_text, normal_text) =
                    reasoning_splitters[index].split(&output.text, output.stop_trimmed_token_ids());
                let mut remaining_logprobs =
                    want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
                if !reasoning_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(reasoning_text)),
                        finish_reason: None,
                        logprobs: remaining_logprobs.clone(),
                    });
                    remaining_logprobs = None;
                }
                if !normal_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(Some(normal_text), None, None, None),
                        finish_reason: None,
                        logprobs: remaining_logprobs,
                    });
                }
            } else {
                emitted.push(ChatChoiceStream {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    delta: chat_delta(
                        (!output.text.is_empty()).then_some(output.text),
                        None,
                        None,
                        None,
                    ),
                    finish_reason: None,
                    logprobs: want_logprobs.then(|| chat_logprobs(output.extras.as_deref())),
                });
            };
            // Flush the choice's buffered reasoning tail before its terminal
            // frame (Python `parse_stream_end`, which skips aborts — abort
            // frames already became error chunks above). Both columns flush:
            // some parsers buffer the answer text until EOF.
            if reasoning_enabled && finish_reason.is_some() {
                let (reasoning_tail, normal_tail) = reasoning_splitters[index].finish();
                if !reasoning_tail.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(reasoning_tail)),
                        finish_reason: None,
                        logprobs: None,
                    });
                }
                if !normal_tail.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(Some(normal_tail), None, None, None),
                        finish_reason: None,
                        logprobs: None,
                    });
                }
            }
            // The finish reason rides the last emitted chunk (the wire format
            // the equivalence tests pin); a step whose text was entirely
            // buffered inside the parser still gets a finish-only frame.
            match emitted.last_mut() {
                Some(last) => last.finish_reason = finish_reason,
                None => emitted.push(ChatChoiceStream {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    delta: chat_delta(None, None, None, None),
                    finish_reason,
                    logprobs: None,
                }),
            }
            for choice in emitted {
                yield Annotated {
                    data: Some(CreateChatCompletionStreamResponse {
                        id: response_id.clone(),
                        choices: vec![choice],
                        created,
                        model: model.clone(),
                        service_tier: service_tier.clone(),
                        system_fingerprint: None,
                        object: "chat.completion.chunk".into(),
                        usage: None,
                    }),
                    id: None,
                    event: None,
                    comment: None,
                    error: None,
                };
            }
        }

        if include_usage {
            yield Annotated {
                data: Some(CreateChatCompletionStreamResponse {
                    id: response_id,
                    choices: vec![],
                    created,
                    model,
                    service_tier,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".into(),
                    usage: Some(completion_usage(
                        prompt_tokens,
                        u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                    )),
                }),
                id: None,
                event: None,
                comment: None,
                error: None,
            };
        }
    };

    let parsed: std::pin::Pin<
        Box<dyn futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send>,
    > = if let Some(parser) = parser {
        tool_call_stream(
            &parser,
            tool_choice,
            tools,
            uses_tool_call_structural_tag,
            raw,
        )
    } else {
        Box::pin(raw)
    };

    async_stream::stream! {
        // Each choice's first tool call, whose deltas (a name and argument
        // fragments, for some parsers) alone pass without parallel tool calls.
        let mut first_tool_call: Vec<Option<u32>> = vec![None; count];
        futures::pin_mut!(parsed);
        while let Some(mut item) = parsed.next().await {
            if let Some(response) = item.data.as_mut() {
                if !parallel_tool_calls {
                    for choice in &mut response.choices {
                        let index = choice.index as usize;
                        if let Some(calls) = choice.delta.tool_calls.as_mut() {
                            let seen = first_tool_call.get(index).copied().flatten();
                            let first = seen.or(calls.first().map(|call| call.index));
                            calls.retain(|call| Some(call.index) == first);
                            if let (Some(first), Some(slot)) = (first, first_tool_call.get_mut(index))
                                && !calls.is_empty()
                            {
                                *slot = Some(first);
                            }
                            if calls.is_empty() {
                                choice.delta.tool_calls = None;
                            }
                        }
                    }
                }
                yield serialize_chat_stream_response(response.clone());
            } else if let Some(error) = item.error {
                yield error;
            }
        }
        yield "[DONE]".to_string();
    }
}

fn serialize_chat_stream_response(response: CreateChatCompletionStreamResponse) -> String {
    let mut response = serde_json::to_value(response).expect("OpenAI response must serialize");
    if let Some(delta) = response
        .pointer_mut("/choices/0/delta")
        .and_then(serde_json::Value::as_object_mut)
    {
        delta
            .entry("reasoning_content")
            .or_insert(serde_json::Value::Null);
    }
    response.to_string()
}

#[allow(deprecated)]
pub(super) fn chat_logprobs(extras: Option<&ChunkExtras>) -> ChatChoiceLogprobs {
    let mut content = Vec::new();
    let Some(extras) = extras else {
        return ChatChoiceLogprobs {
            content: Some(content),
            refusal: None,
        };
    };
    let mut top_offset = 0usize;
    for (position, (&logprob, &token_id)) in
        extras.out_lp_val.iter().zip(&extras.out_lp_idx).enumerate()
    {
        let token = extras
            .out_lp_txt
            .get(position)
            .cloned()
            .unwrap_or_else(|| format!("token_id:{token_id}"));
        let top_len = extras.out_top_lens.get(position).copied().unwrap_or(0) as usize;
        let top_logprobs = extras.out_top_val[top_offset..]
            .iter()
            .zip(&extras.out_top_idx[top_offset..])
            .take(top_len)
            .enumerate()
            .map(|(offset, (&logprob, &id))| {
                let text = extras
                    .out_top_txt
                    .get(top_offset + offset)
                    .cloned()
                    .unwrap_or_else(|| format!("token_id:{id}"));
                TopLogprobs {
                    bytes: Some(text.as_bytes().to_vec()),
                    token: text,
                    logprob,
                }
            })
            .collect();
        top_offset = top_offset.saturating_add(top_len);
        content.push(ChatCompletionTokenLogprob {
            bytes: Some(token.as_bytes().to_vec()),
            token,
            logprob,
            token_id: u32::try_from(token_id).ok(),
            top_logprobs,
        });
    }
    ChatChoiceLogprobs {
        content: Some(content),
        refusal: None,
    }
}

/// V4, GLM-5.3 and Qwen3.8 prefill `<think>` (Qwen3.8 with a newline) when
/// thinking, so the generated stream has no opening marker.
fn starts_in_reasoning(reasoning_parser: Option<&str>, prompt: &str) -> bool {
    matches!(
        reasoning_parser,
        Some("deepseek-v4" | "deepseek_v4" | "deepseekv4" | "glm45" | "qwen3")
    ) && prompt.trim_end_matches('\n').ends_with("<think>")
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{chat_submitted, chunk};
    use super::{
        SamplingDefaults, chat_event_stream, chat_logprobs, chat_sampling_params,
        merge_template_stops, starts_in_reasoning, unary_chat,
    };
    use crate::message::config::DefaultSamplingParams;
    use crate::message::response::ChunkExtras;
    use axum::http::StatusCode;
    use dynamo_protocols::types::{CreateChatCompletionRequest, Stop};
    use futures::StreamExt;

    fn request() -> CreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    #[test]
    fn prefilled_think_starts_in_reasoning() {
        let glm = Some("glm45");
        assert!(starts_in_reasoning(glm, "<|user|>hi<|assistant|><think>"));
        assert!(!starts_in_reasoning(
            glm,
            "<|user|>hi<|assistant|><think></think>"
        ));
        let qwen = Some("qwen3");
        assert!(starts_in_reasoning(
            qwen,
            "<|im_start|>assistant\n<think>\n"
        ));
        assert!(!starts_in_reasoning(
            qwen,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        assert!(starts_in_reasoning(
            Some("deepseek-v4"),
            "<｜Assistant｜><think>"
        ));
        assert!(!starts_in_reasoning(None, "<think>\n"));
    }

    /// Python `to_sampling_params` priority: user value > model generation
    /// config (`--sampling-defaults model`) > OpenAI terminal default.
    #[test]
    fn sampling_defaults_follow_python_priority_chain() {
        let model = DefaultSamplingParams {
            temperature: Some(0.6),
            top_p: Some(0.9),
            ..Default::default()
        };
        // Omitted → model defaults, not the 1.0 OpenAI terminals.
        let sampling = chat_sampling_params(
            &request(),
            &SamplingDefaults::CHAT.with_model_defaults(&model),
        )
        .unwrap();
        assert_eq!(sampling.temperature, 0.6);
        assert_eq!(sampling.top_p, 0.9);
        // Explicit request values win. `Option<f32>` loses precision in f64 —
        // compare with tolerance.
        let mut request = request();
        request.temperature = Some(0.2);
        request.top_p = Some(0.5);
        let sampling = chat_sampling_params(
            &request,
            &SamplingDefaults::CHAT.with_model_defaults(&model),
        )
        .unwrap();
        assert!((sampling.temperature - 0.2).abs() < 1e-6);
        assert!((sampling.top_p - 0.5).abs() < 1e-6);
    }

    /// `--sampling-defaults openai` resolves an empty model-config slice, so the
    /// conversion falls back to the OpenAI terminal defaults.
    #[test]
    fn sampling_defaults_fall_back_to_openai_terminals_in_openai_mode() {
        let openai_mode = DefaultSamplingParams::default();
        let sampling = chat_sampling_params(
            &request(),
            &SamplingDefaults::CHAT.with_model_defaults(&openai_mode),
        )
        .unwrap();
        assert_eq!(sampling.temperature, 1.0);
        assert_eq!(sampling.top_p, 1.0);
    }

    /// Python `_apply_conversation_template`: template `stop_str` first, then
    /// the request's own stops.
    #[test]
    fn template_stops_merge_before_request_stops() {
        let chatml = super::super::template::builtin_template("chatml").unwrap();
        let formatter = super::super::ChatFormatter::Legacy(Box::new(
            super::super::template::LegacyFormatter { spec: chatml },
        ));
        assert_eq!(
            formatter.stop_strs(),
            Some(crate::message::types::OneOrMany::Many(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into()
            ]))
        );
        // No request stop → the template's delimiters alone.
        let mut req = request();
        merge_template_stops(&mut req, &formatter);
        assert_eq!(
            req.stop,
            Some(Stop::StringArray(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into()
            ]))
        );
        // A string request stop appends as one entry.
        let mut req = request();
        req.stop = Some(Stop::String("<stop>".into()));
        merge_template_stops(&mut req, &formatter);
        assert_eq!(
            req.stop,
            Some(Stop::StringArray(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into(),
                "<stop>".into()
            ]))
        );
        // A list request stop extends the list.
        let mut req = request();
        req.stop = Some(Stop::StringArray(vec!["a".into(), "b".into()]));
        merge_template_stops(&mut req, &formatter);
        assert_eq!(
            req.stop,
            Some(Stop::StringArray(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into(),
                "a".into(),
                "b".into()
            ]))
        );
        // Token-id stops cannot be merged (Python has no such field) — kept alone.
        let mut req = request();
        req.stop = Some(Stop::TokenIdArray(vec![2, 3]));
        merge_template_stops(&mut req, &formatter);
        assert_eq!(req.stop, Some(Stop::TokenIdArray(vec![2, 3])));
    }

    /// The HuggingFace renderer carries no template stops (Python's jinja path
    /// keeps only the request's stops), so the request is left unchanged.
    #[test]
    fn huggingface_formatter_leaves_request_stops_alone() {
        let mut req = request();
        req.stop = Some(Stop::String("x".into()));
        // A prompt formatter is not constructible here without a tokenizer; the
        // empty-legacy-spec twin proves the merge is formatter-gated, and the
        // `HuggingFace` arm returns `None` by construction (see `stop_strs`).
        let legacy = super::super::ChatFormatter::Legacy(Box::new(
            super::super::template::LegacyFormatter {
                spec: super::super::template::LegacySpec::default(),
            },
        ));
        assert!(legacy.stop_strs().is_none());
        merge_template_stops(&mut req, &legacy);
        assert_eq!(req.stop, Some(Stop::String("x".into())));
    }

    /// A request with no `max_tokens`/`max_completion_tokens` stays unbounded —
    /// no terminal default is imposed.
    #[test]
    fn chat_without_a_token_limit_stays_unbounded() {
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        assert_eq!(
            chat_sampling_params(&request, &SamplingDefaults::CHAT)
                .unwrap()
                .max_new_tokens,
            None
        );
    }

    #[test]
    fn chat_logprobs_use_dynamo_wire_types() {
        let extras = ChunkExtras {
            out_lp_val: vec![-0.25],
            out_lp_idx: vec![7],
            out_lp_txt: vec!["x".into()],
            out_top_val: vec![-0.25, -1.0],
            out_top_idx: vec![7, 8],
            out_top_lens: vec![2],
            out_top_txt: vec!["x".into(), "y".into()],
            ..Default::default()
        };
        let logprobs = chat_logprobs(Some(&extras));
        let token = &logprobs.content.unwrap()[0];
        assert_eq!(token.token, "x");
        assert_eq!(token.token_id, Some(7));
        assert_eq!(token.top_logprobs.len(), 2);
        assert_eq!(token.top_logprobs[1].token, "y");
    }

    #[tokio::test]
    async fn unary_chat_fans_in_choices_and_usage() {
        let (choice0, tx0) = chat_submitted(0, "r0");
        let (choice1, tx1) = chat_submitted(1, "r1");
        tx0.send(chunk("r0", "Paris", true)).await.unwrap();
        tx1.send(chunk("r1", "Paris", true)).await.unwrap();

        let response = unary_chat(
            vec![choice0, choice1],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            None,
            None,
            false,
            None,
            true,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert_eq!(value["choices"][1]["index"], 1);
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn unary_chat_separates_reasoning_content_with_parser_configured() {
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk(
            "r0",
            "<think>because Paris is famous</think>Paris",
            true,
        ))
        .await
        .unwrap();

        let response = unary_chat(
            vec![choice],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            None,
            Some("deepseek-r1".into()),
            false,
            None,
            true,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "because Paris is famous"
        );
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert!(value["choices"][0]["message"]["reasoning_content"].is_string());
    }

    #[tokio::test]
    async fn streaming_chat_separates_reasoning_into_own_deltas() {
        let (choice, tx) = chat_submitted(0, "r0");
        // Force mode starts in reasoning, so the opener is stripped and the first
        // reasoning fragment streams immediately.
        tx.send(chunk("r0", "<think>be", false)).await.unwrap();
        tx.send(chunk("r0", "cause</think>Par", false))
            .await
            .unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            true,
            None,
            Some("deepseek-r1".into()),
            false,
            None,
            None,
            false,
            true,
            None,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let first_reasoning: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let second_reasoning: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let content: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[4]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            first_reasoning["choices"][0]["delta"]["reasoning_content"],
            "be"
        );
        assert!(first_reasoning["choices"][0]["delta"]["content"].is_null());
        assert_eq!(
            second_reasoning["choices"][0]["delta"]["reasoning_content"],
            "cause"
        );
        assert_eq!(content["choices"][0]["delta"]["content"], "Par");
        assert!(content["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(frames.len(), 7);
    }

    #[tokio::test]
    async fn streaming_chat_emits_role_deltas_usage_and_done() {
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            true,
            None,
            None,
            false,
            None,
            None,
            false,
            true,
            None,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        assert_eq!(frames.len(), 5);
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let delta: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let usage: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert!(role["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(delta["choices"][0]["delta"]["content"], "Par");
        assert!(delta["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[4], "[DONE]");
    }

    // A named dynamic tool must pass request validation and reach both wire formats.
    #[tokio::test]
    async fn system_message_tools_reach_http_responses() {
        use super::super::test_utils::{body_json, post_json};
        use crate::api_server::core::{CoreConfig, CoreHandle, CoreMetadata};
        use crate::message::config::ServerArgs;
        use crate::message::request::RequestKind;
        use crate::tokenizer_manager::wiring::TmEvent;
        use std::sync::Arc;

        for stream in [false, true] {
            let args = Arc::new(ServerArgs {
                served_model_name: "model".into(),
                tool_call_parser: Some("kimi_k3".into()),
                ..Default::default()
            });
            let (intake_tx, intake_rx) = flume::unbounded();
            let core = CoreHandle::new(
                intake_tx,
                flume::unbounded().0,
                CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: CoreMetadata::from(args.as_ref()),
                },
            );
            let formatter =
                dynamo_renderer::kimi_k3_formatter_for(&Some("kimi_k3".into()), "kimi_k3", false)
                    .unwrap();
            let state = Arc::new(super::super::AppState {
                core,
                generation_policy: crate::message::request::GenerationPolicy::from(args.as_ref()),
                server_args: args,
                workers: Default::default(),
                codec: None,
                chat_formatter: Some(super::super::ChatFormatter::HuggingFace(formatter)),
            });
            let responder = tokio::spawn(async move {
                let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
                else {
                    panic!("expected generation intake");
                };
                assert!(admission.try_accept());
                let RequestKind::Generate(generate) = &request.kind else {
                    panic!("expected generation");
                };
                assert!(generate.text.as_ref().unwrap().contains("lookup"));
                assert!(
                    generate
                        .sampling_params
                        .structural_tag
                        .as_ref()
                        .unwrap()
                        .contains("lookup")
                );
                request.sink.try_send(chunk(request.rid.as_str(),
                    "<|open|>tools<|sep|><|open|>call tool=\"lookup\" index=\"1\"<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>", true)).unwrap();
            });
            let response = post_json(super::super::routes().with_state(state), "/v1/chat/completions", serde_json::json!({
                "model":"model", "stream":stream,
                "messages":[{"role":"system","tools":[{"name":"lookup","parameters":{"type":"object","properties":{}}}]},
                            {"role":"user","content":"Call lookup"}],
                "tool_choice":{"type":"function","function":{"name":"lookup"}},
                "chat_template_kwargs":{"thinking":false}
            })).await;
            assert_eq!(response.status(), StatusCode::OK);
            if stream {
                let bytes = axum::body::to_bytes(response.into_body(), 65536)
                    .await
                    .unwrap();
                let text = std::str::from_utf8(&bytes).unwrap();
                let frames: Vec<serde_json::Value> = text
                    .lines()
                    .filter_map(|line| {
                        line.strip_prefix("data: ")
                            .filter(|data| *data != "[DONE]")
                            .map(|data| serde_json::from_str(data).unwrap())
                    })
                    .collect();
                assert!(
                    frames.iter().any(
                        |frame| frame["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
                            == "lookup"
                    ),
                    "{text}"
                );
                assert!(
                    frames
                        .iter()
                        .any(|frame| frame["choices"][0]["finish_reason"] == "tool_calls"),
                    "{text}"
                );
            } else {
                let body = body_json(response).await;
                assert_eq!(
                    body["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "lookup",
                    "{body}"
                );
                assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
            }
            responder.await.unwrap();
        }
    }
}
