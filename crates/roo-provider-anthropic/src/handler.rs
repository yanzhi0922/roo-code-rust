//! Anthropic provider handler.
//!
//! Implements the Provider trait for the Anthropic Messages API.
//! Handles SSE streaming with Anthropic-specific event types:
//! - message_start, content_block_start, content_block_delta, message_delta
//! Supports extended thinking, prompt caching, and tool use.
//!
//! Also includes [`AnthropicVertexHandler`] for running Claude models
//! through Vertex AI's Anthropic publisher endpoint.

use std::pin::Pin;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use roo_provider::error::{ProviderError, Result};
use roo_provider::handler::{ApiStream, CreateMessageMetadata, Provider};
use roo_provider::transform::anthropic_filter::filter_non_anthropic_blocks;
use roo_provider::transform::caching::apply_vertex_caching;
use roo_types::api::{
    ApiMessage, ApiStreamChunk, ContentBlock, ProviderName,
};
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::{
    AnthropicConfig, AnthropicDelta, AnthropicSseEvent, AnthropicUsage,
    AnthropicVertexConfig, anthropic_vertex_models, anthropic_vertex_default_model_id,
};

// =========================================================================
// AnthropicHandler
// =========================================================================

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
                context_window: 200000,
                supports_images: Some(true),
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
            && model_info.supports_reasoning_budget.unwrap_or(false);

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
    ///
    /// This is the core SSE parsing logic shared between [`AnthropicHandler`]
    /// and [`AnthropicVertexHandler`]. Both handlers use the same Anthropic
    /// Messages API response format.
    pub(crate) fn parse_sse_stream_impl(
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
///
/// Handles two input formats:
/// 1. **OpenAI format**: `{ "type": "function", "function": { "name", "description", "parameters" } }`
/// 2. **Direct format** (from `ToolDefinition` serde): `{ "name", "description", "parameters" }`
///
/// Both are converted to Anthropic's `{ "name", "description", "input_schema" }` format.
fn convert_tool_for_anthropic(tool: &Value) -> Value {
    let tool_type = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");

    // OpenAI function-calling format
    if tool_type == "function" {
        let function = tool.get("function").cloned().unwrap_or(json!({}));
        let name = function.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let description = function
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("");
        let parameters = function.get("parameters").cloned().unwrap_or(json!({}));

        return json!({
            "name": name,
            "description": description,
            "input_schema": parameters,
        });
    }

    // Direct format (ToolDefinition serde output): { name, description, parameters }
    if tool.get("name").is_some() && tool.get("parameters").is_some() {
        let name = tool.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let description = tool
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("");
        let parameters = tool.get("parameters").cloned().unwrap_or(json!({}));

        return json!({
            "name": name,
            "description": description,
            "input_schema": parameters,
        });
    }

    // Fallback: return as-is
    tool.clone()
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
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

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

        Ok(Self::parse_sse_stream_impl(stream, model_info))
    }

    fn get_model(&self) -> (String, ModelInfo) {
        (self.model_id.clone(), self.model_info.clone())
    }

    async fn count_tokens(
        &self,
        content: &[ContentBlock],
    ) -> Result<u64> {
        // Simple estimation: ~4 characters per token (rough approximation)
        let mut total_chars: usize = 0;
        for block in content {
            match block {
                ContentBlock::Text { text } => total_chars += text.len(),
                ContentBlock::ToolUse { input, name, .. } => {
                    total_chars += name.len();
                    total_chars += input.to_string().len();
                }
                ContentBlock::ToolResult { content: inner, .. } => {
                    for c in inner {
                        match c {
                            roo_types::api::ToolResultContent::Text { text } => {
                                total_chars += text.len();
                            }
                            roo_types::api::ToolResultContent::Image { .. } => {
                                total_chars += 256; // rough estimate for image token count
                            }
                        }
                    }
                }
                ContentBlock::Thinking { thinking, .. } => total_chars += thinking.len(),
                ContentBlock::Image { .. } => total_chars += 256,
                ContentBlock::RedactedThinking { data } => total_chars += data.len(),
            }
        }
        Ok(((total_chars as f64) / 4.0).ceil() as u64)
    }

    async fn complete_prompt(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
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

// =========================================================================
// AnthropicVertexHandler
// =========================================================================

/// Anthropic Vertex AI provider handler.
///
/// Uses the Anthropic Messages API format through Vertex AI's
/// `streamRawPredict` endpoint with OAuth2 bearer token authentication.
///
/// Key differences from [`AnthropicHandler`]:
/// - Authentication via `Authorization: Bearer {access_token}` instead of `x-api-key`
/// - Endpoint: `{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict`
/// - Uses Vertex-style caching (`apply_vertex_caching`) instead of Anthropic caching
/// - Supports 1M context beta header (`context-1m-2025-08-07`)
pub struct AnthropicVertexHandler {
    http_client: reqwest::Client,
    config: AnthropicVertexConfig,
    model_id: String,
    model_info: ModelInfo,
    temperature: f64,
    use_extended_thinking: bool,
    max_thinking_tokens: Option<u64>,
    betas: Vec<String>,
}

impl AnthropicVertexHandler {
    /// Create a new Anthropic Vertex handler from configuration.
    pub fn new(config: AnthropicVertexConfig) -> Result<Self> {
        let model_id = config
            .model_id
            .clone()
            .unwrap_or_else(|| anthropic_vertex_default_model_id());

        let mut model_info = anthropic_vertex_models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(8192),
                context_window: 200_000,
                supports_images: Some(true),
                supports_prompt_cache: true,
                input_price: Some(3.0),
                output_price: Some(15.0),
                cache_writes_price: Some(3.75),
                cache_reads_price: Some(0.3),
                description: Some("Anthropic Vertex model (unknown variant)".to_string()),
                ..Default::default()
            });

        // Build betas array for request headers
        let mut betas = Vec::new();

        // If 1M context beta is enabled AND the model supports it, update model info
        let supports_1m = crate::types::VERTEX_1M_CONTEXT_MODEL_IDS
            .contains(&model_id.as_str());
        if config.enable_1m_context && supports_1m {
            if let Some(tier) = model_info.tiers.as_ref().and_then(|t| t.first()) {
                model_info.context_window = tier.context_window;
                if let Some(p) = tier.input_price {
                    model_info.input_price = Some(p);
                }
                if let Some(p) = tier.output_price {
                    model_info.output_price = Some(p);
                }
                if let Some(p) = tier.cache_writes_price {
                    model_info.cache_writes_price = Some(p);
                }
                if let Some(p) = tier.cache_reads_price {
                    model_info.cache_reads_price = Some(p);
                }
            }
            betas.push("context-1m-2025-08-07".to_string());
        }

        let use_extended_thinking = config
            .use_extended_thinking
            .unwrap_or(false)
            && model_info.supports_reasoning_budget.unwrap_or(false);

        let temperature = config.temperature.unwrap_or(0.0);

        let mut client_builder = reqwest::Client::builder();
        if let Some(timeout) = config.request_timeout {
            client_builder =
                client_builder.timeout(std::time::Duration::from_millis(timeout));
        }
        let http_client = client_builder.build().map_err(ProviderError::Reqwest)?;

        Ok(Self {
            http_client,
            config,
            model_id,
            model_info,
            temperature,
            use_extended_thinking,
            max_thinking_tokens: None,
            betas,
        })
    }

    /// Create a new Anthropic Vertex handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self> {
        let config = AnthropicVertexConfig::from_settings(settings)
            .ok_or(ProviderError::ApiKeyRequired)?;
        Self::new(config)
    }

    /// Build the request body for the Anthropic Messages API via Vertex.
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
        let mut anthropic_messages = convert_to_anthropic_messages(&filtered_messages);

        // Apply Vertex caching strategy (last 2 user messages)
        if self.model_info.supports_prompt_cache {
            apply_vertex_caching(&mut anthropic_messages);
        }

        let mut body = json!({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": anthropic_messages,
            "stream": true,
        });

        // System prompt with cache_control for Vertex
        if self.model_info.supports_prompt_cache {
            body["system"] = json!([{
                "type": "text",
                "text": system_prompt,
                "cache_control": { "type": "ephemeral" }
            }]);
        } else {
            body["system"] = json!(system_prompt);
        }

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
            // Extended thinking requires temperature to be unset
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
}

#[async_trait]
impl Provider for AnthropicVertexHandler {
    async fn create_message(
        &self,
        system_prompt: &str,
        messages: Vec<ApiMessage>,
        tools: Option<Vec<Value>>,
        _metadata: CreateMessageMetadata,
    ) -> Result<ApiStream> {
        let body = self.build_request_body(system_prompt, &messages, tools.as_ref());
        let url = self.config.stream_url(&self.model_id);

        let mut request = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");

        // Add beta headers if any
        if !self.betas.is_empty() {
            request = request.header("anthropic-beta", self.betas.join(","));
        }

        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::api_error("anthropic-vertex", e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::api_error_response(
                "anthropic-vertex", status, text,
            ));
        }

        let model_info = self.model_info.clone();

        // Parse SSE stream — reuse AnthropicHandler's parser
        let sse_stream = response
            .bytes_stream()
            .eventsource()
            .map(move |event| {
                match event {
                    Ok(event) => {
                        match serde_json::from_str::<AnthropicSseEvent>(&event.data) {
                            Ok(sse_event) => Ok(sse_event),
                            Err(e) => Err(ProviderError::ParseError(format!(
                                "Failed to parse Anthropic Vertex SSE event: {e}"
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

        Ok(AnthropicHandler::parse_sse_stream_impl(stream, model_info))
    }

    fn get_model(&self) -> (String, ModelInfo) {
        // Strip :thinking suffix for the returned model ID
        let clean_id = if self.model_id.ends_with(":thinking") {
            self.model_id[..self.model_id.len() - ":thinking".len()].to_string()
        } else {
            self.model_id.clone()
        };
        (clean_id, self.model_info.clone())
    }

    async fn count_tokens(
        &self,
        content: &[ContentBlock],
    ) -> Result<u64> {
        // Simple estimation: ~4 characters per token (rough approximation)
        let mut total_chars: usize = 0;
        for block in content {
            match block {
                ContentBlock::Text { text } => total_chars += text.len(),
                ContentBlock::ToolUse { input, name, .. } => {
                    total_chars += name.len();
                    total_chars += input.to_string().len();
                }
                ContentBlock::ToolResult { content: inner, .. } => {
                    for c in inner {
                        match c {
                            roo_types::api::ToolResultContent::Text { text } => {
                                total_chars += text.len();
                            }
                            roo_types::api::ToolResultContent::Image { .. } => {
                                total_chars += 256;
                            }
                        }
                    }
                }
                ContentBlock::Thinking { thinking, .. } => total_chars += thinking.len(),
                ContentBlock::Image { .. } => total_chars += 256,
                ContentBlock::RedactedThinking { data } => total_chars += data.len(),
            }
        }
        Ok(((total_chars as f64) / 4.0).ceil() as u64)
    }

    async fn complete_prompt(&self, prompt: &str) -> Result<String> {
        let url = self.config.predict_url(&self.model_id);
        let max_tokens = self.model_info.max_tokens.unwrap_or(8192);

        let body = json!({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": if self.model_info.supports_prompt_cache {
                    json!([{ "type": "text", "text": prompt, "cache_control": { "type": "ephemeral" } }])
                } else {
                    json!(prompt)
                }
            }],
            "stream": false,
        });

        let mut request = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.access_token))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if !self.betas.is_empty() {
            request = request.header("anthropic-beta", self.betas.join(","));
        }

        let response = request
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::api_error("anthropic-vertex", e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::api_error_response(
                "anthropic-vertex", status, text,
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
        ProviderName::Vertex
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models;

    // -----------------------------------------------------------------------
    // AnthropicHandler tests
    // -----------------------------------------------------------------------

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
            "https://api.anthropic.com"
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
        settings.api_model_id = Some("claude-3-5-haiku-20241022".to_string());

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
            assert!(info.supports_images.unwrap_or(false), "Model '{}' should support images", id);
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
        assert_eq!(sonnet4.supports_reasoning_budget, Some(true));
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

    #[tokio::test]
    async fn test_count_tokens_estimation() {
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

        let content = vec![ContentBlock::Text {
            text: "Hello, world!".to_string(),
        }];
        let tokens = handler.count_tokens(&content).await.unwrap();
        // "Hello, world!" = 13 chars → ceil(13/4) = 4 tokens
        assert_eq!(tokens, 4);

        // Empty content
        let empty: Vec<ContentBlock> = vec![];
        let tokens = handler.count_tokens(&empty).await.unwrap();
        assert_eq!(tokens, 0);

        // Mixed content blocks
        let mixed = vec![
            ContentBlock::Text {
                text: "abc".to_string(),
            },
            ContentBlock::Image {
                source: roo_types::api::ImageSource::Url {
                    url: "http://example.com/img.png".to_string(),
                },
            },
        ];
        let tokens = handler.count_tokens(&mixed).await.unwrap();
        // "abc" = 3 chars + image estimate 256 = 259 → ceil(259/4) = 65
        assert_eq!(tokens, 65);
    }

    #[test]
    fn test_build_request_body_url() {
        // Verify the URL construction uses /v1/messages
        let config = AnthropicConfig {
            api_key: "sk-ant-test".to_string(),
            base_url: "https://api.minimaxi.com/anthropic".to_string(),
            model_id: Some("minimax-m2.7".to_string()),
            temperature: None,
            use_extended_thinking: None,
            max_thinking_tokens: None,
            request_timeout: None,
        };
        let handler = AnthropicHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "minimax-m2.7");
        // The handler uses unknown model, so it gets fallback ModelInfo
        assert!(handler.model_info.max_tokens.unwrap_or(0) > 0);
    }

    // -----------------------------------------------------------------------
    // AnthropicVertexHandler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_config_default_region() {
        assert_eq!(AnthropicVertexConfig::DEFAULT_REGION, "us-east5");
    }

    #[test]
    fn test_vertex_config_default_project_id() {
        assert_eq!(AnthropicVertexConfig::DEFAULT_PROJECT_ID, "not-provided");
    }

    #[test]
    fn test_vertex_config_base_url() {
        let config = AnthropicVertexConfig {
            project_id: "my-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let expected = "https://us-east5-aiplatform.googleapis.com/v1/projects/my-project/locations/us-east5/publishers/anthropic/models";
        assert_eq!(config.base_url(), expected);
    }

    #[test]
    fn test_vertex_config_stream_url() {
        let config = AnthropicVertexConfig {
            project_id: "my-project".to_string(),
            region: "europe-west1".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let url = config.stream_url("claude-sonnet-4@20250514");
        assert!(url.contains("streamRawPredict"));
        assert!(url.contains("claude-sonnet-4@20250514"));
        assert!(url.contains("europe-west1"));
    }

    #[test]
    fn test_vertex_config_stream_url_strips_thinking() {
        let config = AnthropicVertexConfig {
            project_id: "my-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let url = config.stream_url("claude-3-7-sonnet@20250219:thinking");
        assert!(url.contains("claude-3-7-sonnet@20250219"));
        assert!(!url.contains(":thinking"));
        assert!(url.contains("streamRawPredict"));
    }

    #[test]
    fn test_vertex_config_predict_url() {
        let config = AnthropicVertexConfig {
            project_id: "my-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let url = config.predict_url("claude-sonnet-4@20250514");
        assert!(url.contains("rawPredict"));
        assert!(url.contains("claude-sonnet-4@20250514"));
    }

    #[test]
    fn test_vertex_handler_creation() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_vertex_handler_default_model() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, crate::types::ANTHROPIC_VERTEX_DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_vertex_handler_custom_model() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: Some("claude-opus-4@20250514".to_string()),
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "claude-opus-4@20250514");
    }

    #[test]
    fn test_vertex_handler_provider_name() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::Vertex);
    }

    #[test]
    fn test_vertex_handler_1m_context_beta() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: Some("claude-sonnet-4@20250514".to_string()),
            temperature: None,
            request_timeout: None,
            enable_1m_context: true,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();
        assert!(handler.betas.contains(&"context-1m-2025-08-07".to_string()));
        // Model info should be updated with tier pricing
        assert_eq!(handler.model_info.context_window, 1_000_000);
    }

    #[test]
    fn test_vertex_handler_1m_context_unsupported_model() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: Some("claude-3-opus@20240229".to_string()),
            temperature: None,
            request_timeout: None,
            enable_1m_context: true,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();
        // claude-3-opus is not in VERTEX_1M_CONTEXT_MODEL_IDS, so no beta
        assert!(handler.betas.is_empty());
        // Context window should remain at default 200_000
        assert_eq!(handler.model_info.context_window, 200_000);
    }

    #[test]
    fn test_vertex_config_from_settings() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.vertex_project_id = Some("my-project".to_string());
        settings.vertex_json_credentials = Some("my-creds".to_string());
        settings.vertex_region = Some("europe-west1".to_string());
        settings.api_model_id = Some("claude-sonnet-4@20250514".to_string());

        let config = AnthropicVertexConfig::from_settings(&settings).unwrap();
        assert_eq!(config.project_id, "my-project");
        assert_eq!(config.access_token, "my-creds");
        assert_eq!(config.region, "europe-west1");
        assert_eq!(config.model_id, Some("claude-sonnet-4@20250514".to_string()));
    }

    #[test]
    fn test_vertex_config_from_settings_no_credentials() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        // No credentials → should return None
        assert!(AnthropicVertexConfig::from_settings(&settings).is_none());
    }

    #[test]
    fn test_vertex_config_from_settings_defaults() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.vertex_json_credentials = Some("token".to_string());
        // No project_id → should use default
        // No region → should use default

        let config = AnthropicVertexConfig::from_settings(&settings).unwrap();
        assert_eq!(config.project_id, "not-provided");
        assert_eq!(config.region, "us-east5");
    }

    #[test]
    fn test_vertex_models_count() {
        let vertex_models = anthropic_vertex_models();
        assert!(
            vertex_models.len() >= 10,
            "Should have at least 10 Anthropic Vertex models, got {}",
            vertex_models.len()
        );
    }

    #[test]
    fn test_vertex_models_all_have_required_fields() {
        for (id, info) in anthropic_vertex_models() {
            assert!(
                info.max_tokens.is_some(),
                "Vertex model '{}' missing max_tokens",
                id
            );
            assert!(
                info.input_price.is_some(),
                "Vertex model '{}' missing input_price",
                id
            );
            assert!(
                info.output_price.is_some(),
                "Vertex model '{}' missing output_price",
                id
            );
        }
    }

    #[test]
    fn test_vertex_handler_thinking_suffix_stripped() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: Some("claude-3-7-sonnet@20250219:thinking".to_string()),
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        // :thinking should be stripped in get_model()
        assert_eq!(model_id, "claude-3-7-sonnet@20250219");
    }

    #[test]
    fn test_vertex_handler_build_request_body() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: Some("claude-sonnet-4@20250514".to_string()),
            temperature: Some(0.5),
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();

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

        let body = handler.build_request_body("You are helpful", &messages, None);

        // Verify basic structure
        assert_eq!(body["model"], "claude-sonnet-4@20250514");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0.5);

        // System should be array with cache_control
        let system = body["system"].as_array().expect("system should be array");
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");

        // Messages should be present
        let msgs = body["messages"].as_array().expect("messages should be array");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_vertex_handler_build_request_body_with_tools() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: Some("claude-sonnet-4@20250514".to_string()),
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();

        let messages = vec![];
        let tools = Some(vec![json!({
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": { "type": "object" }
            }
        })]);

        let body = handler.build_request_body("system", &messages, tools.as_ref());

        let tools_arr = body["tools"].as_array().expect("tools should be array");
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["name"], "test_tool");
        assert!(tools_arr[0].get("input_schema").is_some());
    }

    #[test]
    fn test_vertex_handler_from_settings_no_credentials() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = AnthropicVertexHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_vertex_handler_count_tokens() {
        let config = AnthropicVertexConfig {
            project_id: "test-project".to_string(),
            region: "us-east5".to_string(),
            access_token: "test-token".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            enable_1m_context: false,
            use_extended_thinking: None,
            max_thinking_tokens: None,
        };
        let handler = AnthropicVertexHandler::new(config).unwrap();

        let content = vec![ContentBlock::Text {
            text: "Hello, world!".to_string(),
        }];
        let tokens = handler.count_tokens(&content).await.unwrap();
        assert_eq!(tokens, 4);
    }
}
