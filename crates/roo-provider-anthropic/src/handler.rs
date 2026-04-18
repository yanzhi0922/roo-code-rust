//! Anthropic provider handler.
//!
//! Implements the Provider trait for the Anthropic Messages API.
//! Handles SSE streaming with Anthropic-specific event types:
//! - message_start, content_block_start, content_block_delta, message_delta
//! Supports extended thinking, prompt caching, and tool use.

use std::pin::Pin;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use roo_provider::error::{ProviderError, Result};
use roo_provider::handler::{ApiStream, CreateMessageMetadata, Provider};
use roo_provider::transform::anthropic_filter::filter_non_anthropic_blocks;
use roo_types::api::{
    ApiMessage, ApiStreamChunk, ContentBlock, ProviderName,
};
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::{
    AnthropicConfig, AnthropicDelta, AnthropicSseEvent, AnthropicUsage,
};

/// Anthropic API provider handler.
pub struct AnthropicHandler {
    http_client: reqwest::Client,
    api_key: String,
    base_url: String,
    model_id: String,
    model_info: ModelInfo,
    temperature: f64,
    use_extended_thinking: bool,
    max_thinking_tokens: Option<u64>,
}

impl AnthropicHandler {
    /// Create a new Anthropic handler from configuration.
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(8192),
                max_input_tokens: Some(200000),
                supports_images: true,
                supports_prompt_cache: true,
                input_price: Some(3.0),
                output_price: Some(15.0),
                cache_writes_price: Some(3.75),
                cache_reads_price: Some(0.3),
                description: Some("Anthropic Claude model (unknown variant)".to_string()),
                ..Default::default()
            });

        let mut client_builder = reqwest::Client::builder();
        if let Some(timeout) = config.request_timeout {
            client_builder =
                client_builder.timeout(std::time::Duration::from_millis(timeout));
        }
        let http_client = client_builder.build().map_err(ProviderError::Reqwest)?;

        let use_extended_thinking = config
            .use_extended_thinking
            .unwrap_or(false)
            && model_info.thinking.unwrap_or(false);

        Ok(Self {
            http_client,
            api_key: config.api_key,
            base_url: config.base_url,
            model_id,
            model_info,
            temperature: config.temperature.unwrap_or(0.0),
            use_extended_thinking,
            max_thinking_tokens: config.max_thinking_tokens,
        })
    }

    /// Create a new Anthropic handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self> {
        let config =
            AnthropicConfig::from_settings(settings).ok_or(ProviderError::ApiKeyRequired)?;
        Self::new(config)
    }

    /// Build the request body for the Anthropic Messages API.
    fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[ApiMessage],
        tools: Option<&Vec<Value>>,
    ) -> Value {
        let max_tokens = self.model_info.max_tokens.unwrap_or(8192);

        // Filter messages to only include Anthropic-compatible blocks
        let filtered_messages = filter_non_anthropic_blocks(messages.to_vec());

        // Convert messages to Anthropic format
        let anthropic_messages = convert_to_anthropic_messages(&filtered_messages);

        let mut body = json!({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": anthropic_messages,
            "system": system_prompt,
            "stream": true,
        });

        // Add extended thinking configuration
        if self.use_extended_thinking {
            let budget_tokens = self
                .max_thinking_tokens
                .unwrap_or(10000)
                .min(max_tokens.saturating_sub(1));
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": budget_tokens,
            });
            // Extended thinking requires temperature to be unset (or 1.0)
            // Remove temperature from body when thinking is enabled
            if let Some(obj) = body.as_object_mut() {
                obj.remove("temperature");
            }
        }

        // Add tools if provided
        if let Some(tools) = tools {
            if !tools.is_empty() {
                let anthropic_tools: Vec<Value> = tools
                    .iter()
                    .map(|tool| convert_tool_for_anthropic(tool))
                    .collect();
                body["tools"] = json!(anthropic_tools);
            }
        }

        body
    }

    /// Parse the SSE stream from the Anthropic API.
    fn parse_sse_stream(
        stream: Pin<Box<dyn Stream<Item = Result<AnthropicSseEvent>> + Send>>,
        model_info: ModelInfo,
    ) -> ApiStream {
        let mut tool_call_index: u64 = 0;
        let mut current_tool_id: Option<String> = Option::None;
        let mut current_tool_name: Option<String> = Option::None;
        let mut current_tool_args: String = String::new();
        let mut usage_info: Option<AnthropicUsage> = Option::None;

        let processed = stream.flat_map(move |event_result| {
            let model_info = model_info.clone();
            let mut idx = tool_call_index;
            let mut tool_id = current_tool_id.clone();
            let mut tool_name = current_tool_name.clone();
            let mut tool_args = current_tool_args.clone();
            let mut usage = usage_info.clone();

            let chunks: Vec<Result<ApiStreamChunk>> = match event_result {
                Ok(event) => {
                    let mut results = Vec::new();

                    match event {
                        AnthropicSseEvent::ContentBlockStart {
                            content_block, ..
                        } => {
                            match content_block {
                                crate::types::AnthropicContentBlock::ToolUse {
                                    id,
                                    name,
                                    ..
                                } => {
                                    tool_id = Some(id.clone());
                                    tool_name = Some(name.clone());
                                    tool_args = String::new();
                                    results.push(Ok(ApiStreamChunk::ToolCallStart {
                                        id: id.clone(),
                                        name,
                                    }));
                                }
                                crate::types::AnthropicContentBlock::Text { .. } => {}
                                crate::types::AnthropicContentBlock::Thinking { .. } => {}
                            }
                        }
                        AnthropicSseEvent::ContentBlockDelta { delta, .. } => match delta {
                            AnthropicDelta::TextDelta { text } => {
                                results.push(Ok(ApiStreamChunk::Text { text }));
                            }
                            AnthropicDelta::ThinkingDelta { thinking } => {
                                results.push(Ok(ApiStreamChunk::Reasoning {
                                    text: thinking,
                                    signature: None,
                                }));
                            }
                            AnthropicDelta::InputJsonDelta { partial_json } => {
                                tool_args.push_str(&partial_json);
                                if let Some(ref id) = tool_id {
                                    results.push(Ok(ApiStreamChunk::ToolCallDelta {
                                        id: id.clone(),
                                        delta: partial_json,
                                    }));
                                }
                            }
                            AnthropicDelta::SignatureDelta { signature } => {
                                results.push(Ok(ApiStreamChunk::ThinkingComplete { signature }));
                            }
                        },
                        AnthropicSseEvent::ContentBlockStop { .. } => {
                            if tool_id.is_some() {
                                results.push(Ok(ApiStreamChunk::ToolCall {
                                    id: tool_id.clone().unwrap_or_default(),
                                    name: tool_name.clone().unwrap_or_default(),
                                    arguments: tool_args.clone(),
                                }));
                                results.push(Ok(ApiStreamChunk::ToolCallEnd {
                                    id: tool_id.clone().unwrap_or_default(),
                                }));
                                tool_id = None;
                                tool_name = None;
                                tool_args = String::new();
                                idx += 1;
                            }
                        }
                        AnthropicSseEvent::MessageDelta { delta, usage: msg_usage } => {
                            let _ = delta;
                            if let Some(u) = msg_usage {
                                usage = Some(u);
                            }
                        }
                        AnthropicSseEvent::MessageStart { message } => {
                            if let Some(u) = message.usage {
                                usage = Some(u);
                            }
                        }
                        AnthropicSseEvent::Error { error } => {
                            results.push(Ok(ApiStreamChunk::Error {
                                error: error.error_type.unwrap_or_else(|| "api_error".to_string()),
                                message: error
                                    .message
                                    .unwrap_or_else(|| "Unknown Anthropic error".to_string()),
                            }));
                        }
                        AnthropicSseEvent::MessageStop => {
                            // Emit usage if we have it
                            if let Some(ref u) = usage {
                                results.push(Ok(calculate_anthropic_usage(u, &model_info)));
                            }
                        }
                        AnthropicSseEvent::Ping => {}
                    }

                    results
                }
                Err(e) => vec![Err(e)],
            };

            tool_call_index = idx;
            current_tool_id = tool_id;
            current_tool_name = tool_name;
            current_tool_args = tool_args;
            usage_info = usage;

            futures::stream::iter(chunks)
        });

        Box::pin(processed)
    }
}

/// Convert a tool definition to Anthropic format.
fn convert_tool_for_anthropic(tool: &Value) -> Value {
    let tool_type = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if tool_type != "function" {
        return tool.clone();
    }

    let function = tool.get("function").cloned().unwrap_or(json!({}));
    let name = function.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let description = function
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("");
    let parameters = function.get("parameters").cloned().unwrap_or(json!({}));

    json!({
        "name": name,
        "description": description,
        "input_schema": parameters,
    })
}

/// Convert ApiMessages to Anthropic message format.
fn convert_to_anthropic_messages(messages: &[ApiMessage]) -> Vec<Value> {
    let mut result = Vec::new();

    for msg in messages {
        let role = match msg.role {
            roo_types::api::MessageRole::User => "user",
            roo_types::api::MessageRole::Assistant => "assistant",
        };

        let mut content_parts: Vec<Value> = Vec::new();

        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    content_parts.push(json!({
                        "type": "text",
                        "text": text,
                    }));
                }
                ContentBlock::Image { source } => {
                    let source_json = match source {
                        roo_types::api::ImageSource::Base64 { media_type, data } => json!({
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }),
                        roo_types::api::ImageSource::Url { url } => {
                            // Anthropic doesn't support URL images directly,
                            // but we include the URL as a text description
                            json!({
                                "type": "text",
                                "text": format!("[Image: {}]", url),
                            })
                        }
                    };
                    content_parts.push(json!({
                        "type": "image",
                        "source": source_json,
                    }));
                }
                ContentBlock::ToolUse { id, name, input } => {
                    content_parts.push(json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": input,
                    }));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let tool_content: Vec<Value> = content
                        .iter()
                        .map(|c| match c {
                            roo_types::api::ToolResultContent::Text { text } => json!({
                                "type": "text",
                                "text": text,
                            }),
                            roo_types::api::ToolResultContent::Image { source } => {
                                let source_json = match source {
                                    roo_types::api::ImageSource::Base64 { media_type, data } => {
                                        json!({
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data,
                                        })
                                    }
                                    roo_types::api::ImageSource::Url { url } => json!({
                                        "type": "url",
                                        "url": url,
                                    }),
                                };
                                json!({
                                    "type": "image",
                                    "source": source_json,
                                })
                            }
                        })
                        .collect();

                    let mut result_json = json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_content,
                    });
                    if is_error.unwrap_or(false) {
                        result_json["is_error"] = json!(true);
                    }
                    content_parts.push(result_json);
                }
                ContentBlock::Thinking { thinking, signature } => {
                    content_parts.push(json!({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": signature,
                    }));
                }
                ContentBlock::RedactedThinking { data } => {
                    content_parts.push(json!({
                        "type": "redacted_thinking",
                        "data": data,
                    }));
                }
            }
        }

        if !content_parts.is_empty() {
            result.push(json!({
                "role": role,
                "content": content_parts,
            }));
        }
    }

    result
}

/// Calculate usage metrics for Anthropic.
fn calculate_anthropic_usage(usage: &AnthropicUsage, model_info: &ModelInfo) -> ApiStreamChunk {
    let input_tokens = usage.input_tokens.unwrap_or(0);
    let output_tokens = usage.output_tokens.unwrap_or(0);
    let cache_write_tokens = usage.cache_creation_input_tokens.unwrap_or(0);
    let cache_read_tokens = usage.cache_read_input_tokens.unwrap_or(0);

    let input_cost = model_info.input_price.unwrap_or(0.0) * input_tokens as f64 / 1_000_000.0;
    let output_cost = model_info.output_price.unwrap_or(0.0) * output_tokens as f64 / 1_000_000.0;
    let cache_write_cost =
        model_info.cache_writes_price.unwrap_or(0.0) * cache_write_tokens as f64 / 1_000_000.0;
    let cache_read_cost =
        model_info.cache_reads_price.unwrap_or(0.0) * cache_read_tokens as f64 / 1_000_000.0;
    let total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost;

    ApiStreamChunk::Usage {
        input_tokens,
        output_tokens,
        cache_write_tokens: if cache_write_tokens > 0 {
            Some(cache_write_tokens)
        } else {
            None
        },
        cache_read_tokens: if cache_read_tokens > 0 {
            Some(cache_read_tokens)
        } else {
            None
        },
        reasoning_tokens: None,
        total_cost: Some(total_cost),
    }
}

#[async_trait]
impl Provider for AnthropicHandler {
    async fn create_message(
        &self,
        system_prompt: &str,
        messages: Vec<ApiMessage>,
        tools: Option<Vec<Value>>,
        _metadata: CreateMessageMetadata,
    ) -> Result<ApiStream> {
        let body = self.build_request_body(system_prompt, &messages, tools.as_ref());
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let response = self
            .http_client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::api_error("anthropic", e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::api_error_response(
                "anthropic", status, text,
            ));
        }

        let model_info = self.model_info.clone();

        // Parse SSE stream
        let sse_stream = response
            .bytes_stream()
            .eventsource()
            .map(move |event| {
                match event {
                    Ok(event) => {
                        match serde_json::from_str::<AnthropicSseEvent>(&event.data) {
                            Ok(sse_event) => Ok(sse_event),
                            Err(e) => Err(ProviderError::ParseError(format!(
                                "Failed to parse Anthropic SSE event: {e}"
                            ))),
                        }
                    }
                    Err(e) => Err(ProviderError::StreamError(format!(
                        "SSE error: {e}"
                    ))),
                }
            });

        let stream: Pin<Box<dyn Stream<Item = Result<AnthropicSseEvent>> + Send>> =
            Box::pin(sse_stream);

        Ok(Self::parse_sse_stream(stream, model_info))
    }

    fn get_model(&self) -> (String, ModelInfo) {
        (self.model_id.clone(), self.model_info.clone())
    }

    async fn count_tokens(
        &self,
        content: &[ContentBlock],
    ) -> Result<u64> {
        let _ = content;
        Ok(0)
    }

    async fn complete_prompt(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let max_tokens = self.model_info.max_tokens.unwrap_or(8192);

        let body = json!({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{ "role": "user", "content": prompt }]
        });

        let response = self
            .http_client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::api_error("anthropic", e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::api_error_response(
                "anthropic", status, text,
            ));
        }

        let resp: Value = response.json().await.map_err(ProviderError::Reqwest)?;

        // Extract text from content blocks
        if let Some(content) = resp.get("content").and_then(|c| c.as_array()) {
            let text: String = content
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        block.get("text").and_then(|t| t.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            return Ok(text);
        }

        Ok(String::new())
    }

    fn provider_name(&self) -> ProviderName {
        ProviderName::Anthropic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models;

    #[test]
    fn test_default_model_exists() {
        let all_models = models::models();
        assert!(
            all_models.contains_key(models::DEFAULT_MODEL_ID),
            "Default model '{}' should exist",
            models::DEFAULT_MODEL_ID
        );
    }

    #[test]
    fn test_all_models_have_required_fields() {
        for (id, info) in models::models() {
            assert!(info.max_tokens.is_some(), "Model '{}' missing max_tokens", id);
            assert!(info.input_price.is_some(), "Model '{}' missing input_price", id);
            assert!(info.output_price.is_some(), "Model '{}' missing output_price", id);
        }
    }

    #[test]
    fn test_config_default_url() {
        assert_eq!(
            AnthropicConfig::DEFAULT_BASE_URL,
            "https://api.anthropic.com/v1"
        );
    }

    #[test]
    fn test_handler_creation_requires_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = AnthropicHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = AnthropicConfig {
            api_key: "sk-ant-test".to_string(),
            base_url: AnthropicConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_extended_thinking: None,
            max_thinking_tokens: None,
            request_timeout: None,
        };
        let handler = AnthropicHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = AnthropicConfig {
            api_key: "sk-ant-test".to_string(),
            base_url: AnthropicConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_extended_thinking: None,
            max_thinking_tokens: None,
            request_timeout: None,
        };
        let handler = AnthropicHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = AnthropicConfig {
            api_key: "sk-ant-test".to_string(),
            base_url: AnthropicConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("claude-opus-4-20250514".to_string()),
            temperature: None,
            use_extended_thinking: None,
            max_thinking_tokens: None,
            request_timeout: None,
        };
        let handler = AnthropicHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "claude-opus-4-20250514");
    }

    #[test]
    fn test_handler_provider_name() {
        let config = AnthropicConfig {
            api_key: "sk-ant-test".to_string(),
            base_url: AnthropicConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_extended_thinking: None,
            max_thinking_tokens: None,
            request_timeout: None,
        };
        let handler = AnthropicHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::Anthropic);
    }

    #[test]
    fn test_config_from_settings() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.api_key = Some("sk-ant-test".to_string());
        settings.model_id = Some("claude-3-5-haiku-20241022".to_string());

        let config = AnthropicConfig::from_settings(&settings).unwrap();
        assert_eq!(config.api_key, "sk-ant-test");
        assert_eq!(config.model_id, Some("claude-3-5-haiku-20241022".to_string()));
    }

    #[test]
    fn test_config_from_settings_no_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        assert!(AnthropicConfig::from_settings(&settings).is_none());
    }

    #[test]
    fn test_models_count() {
        let all_models = models::models();
        assert!(all_models.len() >= 5, "Should have at least 5 Anthropic models");
    }

    #[test]
    fn test_all_models_support_images() {
        for (id, info) in models::models() {
            assert!(info.supports_images, "Model '{}' should support images", id);
        }
    }

    #[test]
    fn test_all_models_support_cache() {
        for (id, info) in models::models() {
            assert!(info.supports_prompt_cache, "Model '{}' should support prompt cache", id);
        }
    }

    #[test]
    fn test_sonnet_4_has_thinking() {
        let all_models = models::models();
        let sonnet4 = all_models
            .get("claude-sonnet-4-20250514")
            .expect("claude-sonnet-4 should exist");
        assert_eq!(sonnet4.thinking, Some(true));
    }

    #[test]
    fn test_extended_thinking_config() {
        let config = AnthropicConfig {
            api_key: "sk-ant-test".to_string(),
            base_url: AnthropicConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("claude-sonnet-4-20250514".to_string()),
            temperature: None,
            use_extended_thinking: Some(true),
            max_thinking_tokens: Some(5000),
            request_timeout: None,
        };
        let handler = AnthropicHandler::new(config).unwrap();
        assert!(handler.use_extended_thinking);
    }

    #[test]
    fn test_convert_tool_for_anthropic() {
        let tool = json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": { "type": "object", "properties": { "path": { "type": "string" } } }
            }
        });
        let result = convert_tool_for_anthropic(&tool);
        assert_eq!(result["name"], "read_file");
        assert_eq!(result["description"], "Read a file");
        assert!(result.get("input_schema").is_some());
    }

    #[test]
    fn test_convert_to_anthropic_messages() {
        let messages = vec![ApiMessage {
            role: roo_types::api::MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        }];

        let result = convert_to_anthropic_messages(&messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
    }

    #[test]
    fn test_calculate_anthropic_usage() {
        let usage = AnthropicUsage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            cache_creation_input_tokens: Some(20),
            cache_read_input_tokens: Some(10),
        };

        let model_info = ModelInfo {
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.3),
            ..Default::default()
        };

        let result = calculate_anthropic_usage(&usage, &model_info);
        match result {
            ApiStreamChunk::Usage {
                input_tokens,
                output_tokens,
                cache_write_tokens,
                cache_read_tokens,
                total_cost,
                ..
            } => {
                assert_eq!(input_tokens, 100);
                assert_eq!(output_tokens, 50);
                assert_eq!(cache_write_tokens, Some(20));
                assert_eq!(cache_read_tokens, Some(10));
                assert!(total_cost.unwrap() > 0.0);
            }
            _ => panic!("Expected Usage chunk"),
        }
    }
}
