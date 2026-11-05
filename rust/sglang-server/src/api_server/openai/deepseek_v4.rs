// Adapted from SGLang PR #42664 (45abea0269), Apache-2.0.
//! DeepSeek-V4 prompts as SGLang's `serving_chat.py` builds them, on Dynamo's V4 encoder.

use dynamo_renderer::deepseek::v4::{self, ReasoningEffort, ThinkingMode};
use serde::de::IgnoredAny;
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4Profile {
    Preview,
    Official,
}

/// Adapt only the copy used for OpenAI sampling/response validation. Rendering
/// still consumes the complete original body, including fields Dynamo drops.
pub(super) fn openai_request(request: &Value) -> Result<Value, String> {
    let mut typed = request.clone();
    let object = typed.as_object_mut().ok_or("request must be an object")?;
    // SGLang also accepts numeric effort. Dynamo's enum is narrower; the native
    // encoder owns effort, but malformed values must still fail as in pydantic.
    if let Some(effort) = requested_effort(request) {
        let valid = match effort {
            Value::String(s)
                if ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
                    .contains(&s.as_str()) =>
            {
                true
            }
            Value::String(s) => s.parse::<f64>().is_ok_and(|v| (0.0..1.0).contains(&v)),
            Value::Number(n) => n.as_f64().is_some_and(|v| (0.0..1.0).contains(&v)),
            _ => false,
        };
        if !valid {
            return Err("invalid reasoning_effort".into());
        }
    }
    object.remove("reasoning_effort");
    if let Some(tools) = object.get("tools").and_then(Value::as_array) {
        let tools = normalize_tools(tools)?;
        object.insert("tools".into(), tools.into());
    }
    if let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if !message.is_object() {
                return Err("message must be an object".into());
            }
            if message["content"].is_null() {
                message["content"] = "".into();
            }
            for arguments in tool_arguments(message) {
                if !arguments.is_string() {
                    // The native path validates these after final-turn handling:
                    // Python discards a final assistant turn's tool calls.
                    *arguments = if arguments.is_object() {
                        arguments.to_string()
                    } else {
                        "{}".into()
                    }
                    .into();
                }
            }
        }
    }
    Ok(typed)
}

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    profile: DeepSeekV4Profile,
    mut request: Value,
    default_thinking: bool,
    default_effort: Option<String>,
) -> Result<(String, String), String> {
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    let mut messages = messages.into_iter().map(engine_message).collect::<Vec<_>>();
    for message in &mut messages {
        flatten_content(message);
        check_tool_arguments(message)?;
    }
    let request = &request;
    let mut prefix = String::new();
    if let Some(last) = messages.last_mut().filter(|m| m["role"] == "assistant")
        && let Some(content) = last["content"].as_str().map(str::to_owned)
    {
        if request
            .get("continue_final_message")
            .map_or(Ok(false), pydantic_bool)?
        {
            prefix = content;
            messages.pop();
        } else {
            *last = json!({"role": "user", "content": content});
        }
    }
    // SGLang fails on `messages[0]` here.
    if messages.is_empty() {
        return Err("no messages to render".into());
    }
    if let Some(task) = request.get("task").filter(|task| !task.is_null()) {
        let message = messages
            .iter_mut()
            .rev()
            .find(|m| matches!(m["role"].as_str(), Some("user" | "developer")))
            .ok_or("task requires a user or developer message")?;
        message["task"] = task.clone();
    }
    // SGLang drops a later user's task when merging it into a user or tool-result turn.
    // Workaround for ai-dynamo/frontend-crates#366; drop once dynamo-renderer includes it.
    for index in 1..messages.len() {
        if messages[index]["role"] == "user"
            && matches!(messages[index - 1]["role"].as_str(), Some("user" | "tool"))
            && let Some(message) = messages[index].as_object_mut()
        {
            message.remove("task");
        }
    }
    normalize_messages(&mut messages)?;
    if messages.first().is_none_or(|m| m["role"] != "system") {
        messages.insert(0, json!({"role": "system", "content": ""}));
    }
    // Unlike the Jinja path, SGLang passes every request tool, whatever the tool_choice.
    if let Some(tools) = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty())
    {
        messages[0]["tools"] = normalize_tools(tools)?.into();
    }
    let mode = if thinking(request, default_thinking) {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Chat
    };
    let effort = effort(profile, request, default_effort);
    let prompt = v4::encode_messages_with_options(&messages, mode, true, true, effort)
        .map_err(|error| error.to_string())?;
    Ok((prompt, prefix))
}

/// `serving_chat.py`: kwargs `thinking` wins, then what `reasoning` and the
/// effort imply, then `SGLANG_DEFAULT_THINKING`.
pub(crate) fn thinking(request: &Value, default: bool) -> bool {
    if let Some(thinking) = request["chat_template_kwargs"].get("thinking") {
        return minijinja::Value::from_serialize(thinking).is_true();
    }
    requested_thinking(request).unwrap_or(default)
}

/// `encoding_dsv4.REASONING_EFFORT_PROFILES`: kwargs effort replaces the request
/// effort, `fallback` (`SGLANG_DSV4_REASONING_EFFORT`) fills in, and other tiers add no prefix.
fn effort(
    profile: DeepSeekV4Profile,
    request: &Value,
    fallback: Option<String>,
) -> Option<ReasoningEffort> {
    let requested = Some(&request["chat_template_kwargs"]["reasoning_effort"])
        .filter(|effort| !effort.is_null())
        .or_else(|| requested_effort(request))
        .map(|effort| effort.as_str().map(str::to_owned))
        .unwrap_or(fallback);
    match (profile, requested.as_deref()) {
        (DeepSeekV4Profile::Official, Some("high")) | (DeepSeekV4Profile::Preview, Some("max")) => {
            Some(ReasoningEffort::High)
        }
        (DeepSeekV4Profile::Official, Some("max")) => Some(ReasoningEffort::Max),
        _ => None,
    }
}

/// The pydantic dump SGLang renders: roles lowercased, unknown and null fields
/// dropped, `user` reduced to role and content, null content blanked.
fn engine_message(message: Value) -> Value {
    let mut message = match message {
        Value::Object(message) => message,
        _ => Map::new(),
    };
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_lowercase();
    let kept: &[&str] = if role == "user" {
        &["content"]
    } else {
        &[
            "content",
            "tool_call_id",
            "name",
            "reasoning_content",
            "tool_calls",
            "tools",
        ]
    };
    message.retain(|key, value| kept.contains(&key.as_str()) && !value.is_null());
    message.insert("role".into(), role.into());
    message.entry("content").or_insert_with(|| "".into());
    message.into()
}

/// `process_content_for_template_format`: text parts joined with spaces.
fn flatten_content(message: &mut Value) {
    if let Some(parts) = message["content"].as_array() {
        message["content"] = parts
            .iter()
            .filter(|part| matches!(part["type"].as_str(), Some("text" | "input_text")))
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .into();
    }
}

/// `serving_chat.normalize_assistant_tool_call_arguments`: string arguments must
/// be a JSON object; other values wait for the final-turn handling.
fn check_tool_arguments(message: &mut Value) -> Result<(), String> {
    for arguments in tool_arguments(message) {
        if let Some(text) = arguments.as_str()
            && (serde_json::from_str::<IgnoredAny>(text).is_err()
                || !text.trim_start().starts_with('{'))
        {
            return Err("assistant tool arguments must be a JSON object".into());
        }
    }
    Ok(())
}

/// The messages the encoder sees: empty tool lists dropped, message tools dumped,
/// and tool arguments as JSON object text, which Dynamo parses itself.
fn normalize_messages(messages: &mut [Value]) -> Result<(), String> {
    for message in messages {
        for arguments in tool_arguments(message) {
            match arguments {
                Value::String(_) => {}
                Value::Object(_) => *arguments = arguments.to_string().into(),
                _ => return Err("assistant tool arguments must be a JSON object".into()),
            }
        }
        let message = message.as_object_mut().ok_or("message must be an object")?;
        // Empty lists: workaround for ai-dynamo/frontend-crates#366; drop once
        // dynamo-renderer includes it.
        match message.get("tools").and_then(Value::as_array) {
            Some(tools) if tools.is_empty() => _ = message.remove("tools"),
            Some(tools) => _ = message.insert("tools".into(), normalize_tools(tools)?.into()),
            None => {}
        }
        if message
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            message.remove("tool_calls");
        }
    }
    Ok(())
}

/// Each assistant tool call's `function.arguments`, absent ones as `null`.
fn tool_arguments(message: &mut Value) -> impl Iterator<Item = &mut Value> {
    let is_assistant = message["role"] == "assistant";
    message
        .get_mut("tool_calls")
        .filter(|_| is_assistant)
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|call| call.get_mut("function")?.as_object_mut())
        .map(|function| function.entry("arguments").or_insert(Value::Null))
}

/// `protocol.py::Function.model_dump()`, in declared field order; the encoder
/// serializes it verbatim.
fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            let f = &tool["function"];
            let name = f["name"].as_str().ok_or("tool function requires name")?;
            let mut function = Map::new();
            function.insert("description".into(), f["description"].clone());
            function.insert("name".into(), name.into());
            function.insert("parameters".into(), f["parameters"].clone());
            let strict = f.get("strict").map_or(Ok(false), pydantic_bool)?;
            function.insert("strict".into(), strict.into());
            if let Some(defer) = [f.get("defer_loading"), tool.get("defer_loading")]
                .into_iter()
                .flatten()
                .find(|v| !v.is_null())
            {
                function.insert("defer_loading".into(), pydantic_bool(defer)?.into());
            }
            Ok(json!({"type": "function", "function": function}))
        })
        .collect()
}

/// pydantic's lax `bool`: booleans, 0 and 1, and the yes/no words in any case.
fn pydantic_bool(value: &Value) -> Result<bool, String> {
    let parsed = match value {
        Value::Bool(value) => Some(*value),
        Value::Number(number) => match number.as_f64() {
            Some(0.0) => Some(false),
            Some(1.0) => Some(true),
            _ => None,
        },
        Value::String(text) => match text.to_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        _ => None,
    };
    parsed.ok_or_else(|| format!("expected a boolean, got {value}"))
}

/// The request's effort: `reasoning.effort`, then `reasoning.reasoning_effort`,
/// then the top-level `reasoning_effort`.
pub fn requested_effort(request: &Value) -> Option<&Value> {
    let reasoning = &request["reasoning"];
    [
        &reasoning["effort"],
        &reasoning["reasoning_effort"],
        &request["reasoning_effort"],
    ]
    .into_iter()
    .find(|effort| !effort.is_null())
}

/// The thinking toggle the request implies: an effort turns it off only when it
/// is `"none"`; without one, a truthy `reasoning.enabled` turns it on.
pub fn requested_thinking(request: &Value) -> Option<bool> {
    if let Some(effort) = requested_effort(request) {
        return Some(effort != "none");
    }
    let reasoning = &request["reasoning"];
    let enabled = match reasoning
        .get("enabled")
        .filter(|v| !v.is_null())
        .or_else(|| reasoning.get("enable"))?
    {
        Value::String(enabled) => {
            ["1", "true", "yes", "y", "on"].contains(&enabled.trim().to_lowercase().as_str())
        }
        enabled => minijinja::Value::from_serialize(enabled).is_true(),
    };
    enabled.then_some(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn check_fixture(fixture: &Value, token_ids: bool) {
        let profile = match fixture["config"]["dsv4_reasoning_effort_profile"]
            .as_str()
            .unwrap()
        {
            "official" => DeepSeekV4Profile::Official,
            "preview" => DeepSeekV4Profile::Preview,
            other => panic!("unknown profile: {other}"),
        };
        let tokenizer = token_ids.then(|| {
            let model = fixture["local_model_path"]
                .as_str()
                .expect("local_model_path is required");
            crate::tokenizer_manager::tokenizer::load_tokenizer(Some(model), None, false)
                .unwrap()
                .unwrap()
        });
        let mut failures = Vec::new();
        let mut known_gaps = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let outcome = openai_request(&case["request"])
                .and_then(|typed| {
                    serde_json::from_value::<dynamo_protocols::types::CreateChatCompletionRequest>(
                        typed,
                    )
                    .map_err(|e| e.to_string())
                })
                .and_then(|_| render(profile, case["request"].clone(), false, None));
            let matches = match outcome {
                Err(_) => case.get("error").is_some(),
                Ok((prompt, prefix)) => {
                    let text_matches =
                        case.get("error").is_none() && prompt.clone() + &prefix == case["prompt"];
                    if let Some(tokenizer) = &tokenizer {
                        let tokenizer = crate::tokenizer_manager::tokenizer::DynamoTokenizer::new(
                            tokenizer.clone(),
                        );
                        let request = crate::message::request::GenerateRequest {
                            text: Some(prompt),
                            assistant_prefix: (!prefix.is_empty()).then_some((
                                prefix,
                                fixture["bos_token_id"].as_i64().map(|id| id as i32),
                            )),
                            ..Default::default()
                        };
                        let ids = crate::tokenizer_manager::tokenizer::encode_request(
                            &tokenizer,
                            &request,
                            &[],
                        )
                        .unwrap();
                        let digest = ids.iter().fold(Sha256::new(), |hash, id| {
                            hash.chain_update(id.to_le_bytes())
                        });
                        text_matches
                            && ids.len() as u64 == case["token_count"]
                            && format!("{:x}", digest.finalize()) == case["token_sha256"]
                    } else {
                        text_matches
                    }
                }
            };
            if let Some(gap) = case["known_gap"].as_str() {
                if matches {
                    failures.push(format!("{} now matches; remove known_gap", case["name"]));
                } else {
                    known_gaps += 1;
                    eprintln!("{}: known gap: {gap}", case["name"]);
                }
            } else if !matches {
                failures.push(case["name"].to_string());
            }
        }
        assert!(failures.is_empty(), "DSV4 parity failures: {failures:?}");
        eprintln!(
            "{} cases checked, {known_gaps} known gaps; token IDs checked: {token_ids}",
            fixture["cases"].as_array().unwrap().len()
        );
    }

    #[test]
    fn dsv4_python_prompt_fixtures() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/parity/deepseek-v4-flash-0731.json"
        ))
        .unwrap();
        check_fixture(&fixture, false);
    }

    #[test]
    #[ignore = "Generate tests/fixtures/parity/deployment.json with the deployment tokenizer first"]
    fn dsv4_deployment_token_parity() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/parity/deployment.json");
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        check_fixture(&fixture, true);
    }

    #[test]
    fn dsv4_profile_and_default_precedence() {
        use v4::ReasoningEffort::{High, Max};
        assert_eq!(
            effort(
                DeepSeekV4Profile::Preview,
                &json!({"reasoning_effort":"max"}),
                None
            ),
            Some(High)
        );
        assert_eq!(
            effort(
                DeepSeekV4Profile::Official,
                &json!({"reasoning_effort":"max"}),
                None
            ),
            Some(Max)
        );
        assert_eq!(
            effort(
                DeepSeekV4Profile::Official,
                &json!({"reasoning_effort":"low"}),
                Some("max".into())
            ),
            None
        );
        assert!(!thinking(&json!({}), false));
        assert!(thinking(&json!({}), true));
        assert!(!thinking(
            &json!({"chat_template_kwargs":{"thinking":false},"reasoning_effort":"max"}),
            true
        ));
    }
}
