//! Google Gemini provider handler.
//!
//! Uses the Gemini generateContent API with SSE streaming.
//! Converts messages from Anthropic format to Gemini format.

use std::pin::Pin;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

use roo_provider::error::{ProviderError, Result};
use roo_provider::handler::{ApiStream, CreateMessageMetadata, Provider};
use roo_provider::transform::gemini_format::{
    build_tool_id_to_name_map, convert_anthropic_message_to_gemini, GeminiConversionOptions,
};
use roo_types::api::{ApiMessage, ApiStreamChunk, GroundingSource, ProviderName};
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::{GeminiStreamResponse, GoogleConfig};

/// Google Gemini API provider handler.
pub struct GoogleHandler {
    http_client: reqwest::Client,
    api_key: String,
    base_url: String,
    model_id: String,
    model_info: ModelInfo,
    temperature: f64,
}

impl GoogleHandler {
    /// Create a new Google Gemini handler from configuration.
    pub fn new(config: GoogleConfig) -> Result<Self> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(65536),
                context_window: 1048576,
                supports_images: Some(true),
                supports_prompt_cache: true,
                input_price: Some(1.25),
                output_price: Some(10.0),
                description: Some("Google Gemini model (unknown variant)".to_string()),
                ..Default::default()
            });

        let mut client_builder = reqwest::Client::builder();
        if let Some(timeout) = config.request_timeout {
            client_builder =
                client_builder.timeout(std::time::Duration::from_millis(timeout));
        }
        let http_client = client_builder.build().map_err(ProviderError::Reqwest)?;

        Ok(Self {
            http_client,
            api_key: config.api_key,
            base_url: config.base_url,
            model_id,
            model_info,
            temperature: config.temperature.unwrap_or(0.0),
        })
    }

    /// Create a new Google Gemini handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self> {
        let config =
            GoogleConfig::from_settings(settings).ok_or(ProviderError::ApiKeyRequired)?;
        Self::new(config)
    }

    /// Build the request body for the Gemini generateContent API.
    fn build_request_body(
        &self,
        system_prompt: &str,
        messages: &[ApiMessage],
        tools: Option<&Vec<Value>>,
    ) -> Value {
        let tool_id_to_name = build_tool_id_to_name_map(messages);
        let conversion_opts = GeminiConversionOptions {
            include_thought_signatures: true,
            tool_id_to_name,
        };

        // Convert messages to Gemini format
        let mut gemini_contents: Vec<Value> = Vec::new();

        for msg in messages {
            let gemini_messages =
                convert_anthropic_message_to_gemini(msg, &conversion_opts);
            for gemini_msg in gemini_messages {
                gemini_contents.push(serde_json::to_value(gemini_msg).unwrap_or_default());
            }
        }

        let mut body = json!({
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.model_info.max_tokens.unwrap_or(8192),
            },
        });

        // Add system instruction
        if !system_prompt.is_empty() {
            body["systemInstruction"] = json!({
                "parts": [{ "text": system_prompt }]
            });
        }

        // Add tools if provided (Gemini function declarations)
        if let Some(tools) = tools {
            if !tools.is_empty() {
                let function_declarations: Vec<Value> = tools
                    .iter()
                    .filter_map(|tool| {
                        let tool_type = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if tool_type != "function" {
                            return None;
                        }
                        let function = tool.get("function")?;
                        Some(json!({
                            "name": function.get("name"),
                            "description": function.get("description"),
                            "parameters": function.get("parameters"),
                        }))
                    })
                    .collect();

                if !function_declarations.is_empty() {
                    body["tools"] = json!([{ "function_declarations": function_declarations }]);
                }
            }
        }

        body
    }

    /// Parse the SSE stream from the Gemini API.
    fn parse_sse_stream(
        stream: Pin<Box<dyn Stream<Item = Result<GeminiStreamResponse>> + Send>>,
        model_info: ModelInfo,
    ) -> ApiStream {
        let mut usage_emitted = false;

        let processed = stream.flat_map(move |chunk_result| {
            let model_info = model_info.clone();
            let mut emitted = usage_emitted;

            let chunks: Vec<Result<ApiStreamChunk>> = match chunk_result {
                Ok(response) => {
                    let mut results = Vec::new();

                    if let Some(candidates) = &response.candidates {
                        for candidate in candidates {
                            if let Some(content) = &candidate.content {
                                if let Some(parts) = &content.parts {
                                    for part in parts {
                                        if let Some(ref text) = part.text {
                                            if !text.is_empty() {
                                                results.push(Ok(ApiStreamChunk::Text {
                                                    text: text.clone(),
                                                }));
                                            }
                                        }
                                        if let Some(ref fc) = part.function_call {
                                            let id = format!("call_{}", simple_hash(&fc.name));
                                            results.push(Ok(ApiStreamChunk::ToolCallStart {
                                                id: id.clone(),
                                                name: fc.name.clone(),
                                            }));
                                            results.push(Ok(ApiStreamChunk::ToolCall {
                                                id: id.clone(),
                                                name: fc.name.clone(),
                                                arguments: serde_json::to_string(&fc.args)
                                                    .unwrap_or_default(),
                                            }));
                                            results.push(Ok(ApiStreamChunk::ToolCallEnd {
                                                id,
                                            }));
                                        }
                                        if let Some(ref thought) = part.thought {
                                            if !thought.is_empty() {
                                                results.push(Ok(ApiStreamChunk::Reasoning {
                                                    text: thought.clone(),
                                                    signature: None,
                                                }));
                                            }
                                        }
                                    }
                                }
                            }

                            // Handle grounding
                            if let Some(grounding) = &candidate.grounding_metadata {
                                if let Some(g_chunks) = &grounding.grounding_chunks {
                                    let sources: Vec<GroundingSource> = g_chunks
                                        .iter()
                                        .filter_map(|chunk| {
                                            chunk.web.as_ref().map(|web| GroundingSource {
                                                title: web.title.clone(),
                                                url: web.uri.clone(),
                                                snippet: None,
                                            })
                                        })
                                        .collect();
                                    if !sources.is_empty() {
                                        results.push(Ok(ApiStreamChunk::Grounding { sources }));
                                    }
                                }
                            }
                        }
                    }

                    // Handle usage
                    if let Some(usage) = &response.usage_metadata {
                        if !emitted {
                            let input_tokens = usage.prompt_token_count.unwrap_or(0);
                            let output_tokens = usage.candidates_token_count.unwrap_or(0);
                            let cache_read_tokens = usage.cached_content_token_count.unwrap_or(0);

                            let input_cost = model_info.input_price.unwrap_or(0.0)
                                * input_tokens as f64
                                / 1_000_000.0;
                            let output_cost = model_info.output_price.unwrap_or(0.0)
                                * output_tokens as f64
                                / 1_000_000.0;
                            let cache_read_cost =
                                model_info.cache_reads_price.unwrap_or(0.0) * cache_read_tokens as f64
                                    / 1_000_000.0;

                            results.push(Ok(ApiStreamChunk::Usage {
                                input_tokens,
                                output_tokens,
                                cache_write_tokens: None,
                                cache_read_tokens: if cache_read_tokens > 0 {
                                    Some(cache_read_tokens)
                                } else {
                                    None
                                },
                                reasoning_tokens: None,
                                total_cost: Some(input_cost + output_cost + cache_read_cost),
                            }));
                            emitted = true;
                        }
                    }

                    results
                }
                Err(e) => vec![Err(e)],
            };

            usage_emitted = emitted;
            futures::stream::iter(chunks)
        });

        Box::pin(processed)
    }
}

/// Simple hash function for generating tool call IDs.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

#[async_trait]
impl Provider for GoogleHandler {
    async fn create_message(
        &self,
        system_prompt: &str,
        messages: Vec<ApiMessage>,
        tools: Option<Vec<Value>>,
        _metadata: CreateMessageMetadata,
    ) -> Result<ApiStream> {
        let body = self.build_request_body(system_prompt, &messages, tools.as_ref());
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url.trim_end_matches('/'),
            self.model_id,
            self.api_key
        );

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::api_error("gemini", e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::api_error_response("gemini", status, text));
        }

        let model_info = self.model_info.clone();

        let sse_stream = response
            .bytes_stream()
            .eventsource()
            .map(move |event| {
                match event {
                    Ok(event) => {
                        match serde_json::from_str::<GeminiStreamResponse>(&event.data) {
                            Ok(chunk) => Ok(chunk),
                            Err(e) => Err(ProviderError::ParseError(format!(
                                "Failed to parse Gemini SSE event: {e}"
                            ))),
                        }
                    }
                    Err(e) => Err(ProviderError::StreamError(format!("SSE error: {e}"))),
                }
            });

        let stream: Pin<Box<dyn Stream<Item = Result<GeminiStreamResponse>> + Send>> =
            Box::pin(sse_stream);

        Ok(Self::parse_sse_stream(stream, model_info))
    }

    fn get_model(&self) -> (String, ModelInfo) {
        (self.model_id.clone(), self.model_info.clone())
    }


    async fn complete_prompt(&self, prompt: &str) -> Result<String> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url.trim_end_matches('/'),
            self.model_id,
            self.api_key
        );

        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [{ "text": prompt }]
            }],
            "generationConfig": {
                "maxOutputTokens": self.model_info.max_tokens.unwrap_or(8192),
            }
        });

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::api_error("gemini", e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::api_error_response("gemini", status, text));
        }

        let resp: Value = response.json().await.map_err(ProviderError::Reqwest)?;

        // Extract text from candidates
        if let Some(text) = resp
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
        {
            return Ok(text.to_string());
        }

        Ok(String::new())
    }

    fn provider_name(&self) -> ProviderName {
        ProviderName::Gemini
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
            GoogleConfig::DEFAULT_BASE_URL,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn test_handler_creation_requires_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = GoogleHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = GoogleConfig {
            api_key: "test-api-key".to_string(),
            base_url: GoogleConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = GoogleHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = GoogleConfig {
            api_key: "test-api-key".to_string(),
            base_url: GoogleConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = GoogleHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = GoogleConfig {
            api_key: "test-api-key".to_string(),
            base_url: GoogleConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("gemini-2.5-flash".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = GoogleHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "gemini-2.5-flash");
    }

    #[test]
    fn test_handler_provider_name() {
        let config = GoogleConfig {
            api_key: "test-api-key".to_string(),
            base_url: GoogleConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = GoogleHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::Gemini);
    }

    #[test]
    fn test_config_from_settings() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.google_api_key = Some("test-key".to_string());
        settings.api_model_id = Some("gemini-2.5-flash".to_string());

        let config = GoogleConfig::from_settings(&settings).unwrap();
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model_id, Some("gemini-2.5-flash".to_string()));
    }

    #[test]
    fn test_config_from_settings_fallback_to_api_key() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.api_key = Some("fallback-key".to_string());

        let config = GoogleConfig::from_settings(&settings).unwrap();
        assert_eq!(config.api_key, "fallback-key");
    }

    #[test]
    fn test_config_from_settings_no_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        assert!(GoogleConfig::from_settings(&settings).is_none());
    }

    #[test]
    fn test_models_count() {
        let all_models = models::models();
        assert!(all_models.len() >= 5, "Should have at least 5 Gemini models");
    }

    #[test]
    fn test_all_models_support_images() {
        for (id, info) in models::models() {
            assert!(info.supports_images.unwrap_or(false), "Model '{}' should support images", id);
        }
    }

    #[test]
    fn test_pro_model_has_thinking() {
        let all_models = models::models();
        let pro = all_models.get("gemini-2.5-pro").expect("gemini-2.5-pro should exist");
        assert_eq!(pro.supports_reasoning_budget, Some(true));
    }

    #[test]
    fn test_flash_model_cheaper() {
        let all_models = models::models();
        let pro = all_models.get("gemini-2.5-pro").unwrap();
        let flash = all_models.get("gemini-2.5-flash").unwrap();
        assert!(
            flash.input_price.unwrap() < pro.input_price.unwrap(),
            "Flash should be cheaper than Pro"
        );
    }

    #[test]
    fn test_handler_unknown_model_fallback() {
        let config = GoogleConfig {
            api_key: "test-api-key".to_string(),
            base_url: GoogleConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("gemini-future".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = GoogleHandler::new(config).unwrap();
        let (model_id, info) = handler.get_model();
        assert_eq!(model_id, "gemini-future");
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_simple_hash_deterministic() {
        let h1 = simple_hash("test_function");
        let h2 = simple_hash("test_function");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_handler_with_timeout() {
        let config = GoogleConfig {
            api_key: "test-api-key".to_string(),
            base_url: GoogleConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: Some(60000),
        };
        let handler = GoogleHandler::new(config);
        assert!(handler.is_ok());
    }
}
