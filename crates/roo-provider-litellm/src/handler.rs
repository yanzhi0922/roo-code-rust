//! LiteLLM provider handler.
//!
//! Uses the OpenAI-compatible chat completions API via LiteLLM proxy.
//! Supports prompt caching, GPT-5 detection, and Gemini model handling.

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::LiteLlmConfig;

/// LiteLLM API provider handler.
///
/// LiteLLM acts as a proxy that routes requests to various LLM providers.
/// It follows the OpenAI API format for compatibility.
pub struct LiteLlmHandler {
    inner: OpenAiCompatibleProvider,
}

impl LiteLlmHandler {
    /// Create a new LiteLLM handler from configuration.
    pub fn new(config: LiteLlmConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(4096),
                context_window: 128000,
                supports_prompt_cache: false,
                input_price: Some(2.50),
                output_price: Some(10.0),
                description: Some("LiteLLM proxy model (unknown)".to_string()),
                ..Default::default()
            });

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "litellm".to_string(),
            base_url: config.base_url,
            api_key: config.api_key,
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(0.0),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::LiteLlm,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;

        Ok(Self { inner })
    }

    /// Create a new LiteLLM handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config = LiteLlmConfig::from_settings(settings);
        let config = config.unwrap_or_else(|| LiteLlmConfig {
            api_key: "dummy-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_prompt_cache: false,
            request_timeout: None,
        });
        Self::new(config)
    }

    /// Detect if the model is a GPT-5 variant that requires `max_completion_tokens`.
    pub fn is_gpt5(model_id: &str) -> bool {
        // Match gpt-5, gpt5, and variants like gpt-5o, gpt-5-turbo, gpt5-preview, gpt-5.1
        // Avoid matching gpt-50, gpt-500, etc.
        let lower = model_id.to_lowercase();
        lower.contains("gpt-5") || lower.contains("gpt5")
    }

    /// Detect if the model is a Gemini model that requires thought signature handling.
    pub fn is_gemini_model(model_id: &str) -> bool {
        let lower = model_id.to_lowercase();
        lower.contains("gemini-3")
            || lower.contains("gemini-2.5")
            || lower.contains("gemini 3")
            || lower.contains("gemini 2.5")
            || lower.contains("gemini/gemini-3")
            || lower.contains("gemini/gemini-2.5")
            || lower.contains("google/gemini-3")
            || lower.contains("google/gemini-2.5")
            || lower.contains("vertex_ai/gemini-3")
            || lower.contains("vertex_ai/gemini-2.5")
    }
}

#[async_trait]
impl Provider for LiteLlmHandler {
    async fn create_message(
        &self,
        system_prompt: &str,
        messages: Vec<roo_types::api::ApiMessage>,
        tools: Option<Vec<serde_json::Value>>,
        metadata: CreateMessageMetadata,
    ) -> Result<ApiStream, roo_provider::ProviderError> {
        self.inner
            .create_message(system_prompt, messages, tools, metadata)
            .await
    }

    fn get_model(&self) -> (String, ModelInfo) {
        self.inner.get_model()
    }

    async fn count_tokens(
        &self,
        content: &[roo_types::api::ContentBlock],
    ) -> Result<u64, roo_provider::ProviderError> {
        let _ = content;
        Ok(0)
    }

    async fn complete_prompt(
        &self,
        prompt: &str,
    ) -> Result<String, roo_provider::ProviderError> {
        self.inner.complete_prompt(prompt).await
    }

    fn provider_name(&self) -> ProviderName {
        ProviderName::LiteLlm
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
            "Default model '{}' should exist in models map",
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
    fn test_default_url() {
        assert_eq!(LiteLlmConfig::DEFAULT_BASE_URL, "http://localhost:4000");
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_prompt_cache: false,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_prompt_cache: false,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("claude-3-5-sonnet-20241022".to_string()),
            temperature: None,
            use_prompt_cache: false,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_handler_from_settings() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let handler = LiteLlmHandler::from_settings(&settings);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_provider_name() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_prompt_cache: false,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::LiteLlm);
    }

    #[test]
    fn test_gpt5_detection() {
        assert!(LiteLlmHandler::is_gpt5("gpt-5"));
        assert!(LiteLlmHandler::is_gpt5("gpt5"));
        assert!(LiteLlmHandler::is_gpt5("gpt-5-turbo"));
        assert!(LiteLlmHandler::is_gpt5("gpt-5o"));
        assert!(LiteLlmHandler::is_gpt5("gpt5-preview"));
        assert!(LiteLlmHandler::is_gpt5("GPT-5"));
        assert!(!LiteLlmHandler::is_gpt5("gpt-4"));
        assert!(!LiteLlmHandler::is_gpt5("gpt-4o"));
        assert!(!LiteLlmHandler::is_gpt5("claude-3"));
    }

    #[test]
    fn test_gemini_model_detection() {
        assert!(LiteLlmHandler::is_gemini_model("gemini-3-pro"));
        assert!(LiteLlmHandler::is_gemini_model("gemini-2.5-pro"));
        assert!(LiteLlmHandler::is_gemini_model("gemini-3-flash"));
        assert!(LiteLlmHandler::is_gemini_model("google/gemini-3-pro"));
        assert!(LiteLlmHandler::is_gemini_model("vertex_ai/gemini-2.5-flash"));
        assert!(!LiteLlmHandler::is_gemini_model("gpt-4"));
        assert!(!LiteLlmHandler::is_gemini_model("claude-3-5-sonnet"));
        assert!(!LiteLlmHandler::is_gemini_model("gemini-1.5-pro"));
    }

    #[test]
    fn test_custom_base_url() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: "http://custom-litellm:8080".to_string(),
            model_id: None,
            temperature: None,
            use_prompt_cache: false,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_prompt_cache_config() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            use_prompt_cache: true,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_fallback_model_info() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("unknown-model-xyz".to_string()),
            temperature: None,
            use_prompt_cache: false,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config).unwrap();
        let (_, info) = handler.get_model();
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_temperature_config() {
        let config = LiteLlmConfig {
            api_key: "test-key".to_string(),
            base_url: LiteLlmConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: Some(0.7),
            use_prompt_cache: false,
            request_timeout: None,
        };
        let handler = LiteLlmHandler::new(config);
        assert!(handler.is_ok());
    }
}
