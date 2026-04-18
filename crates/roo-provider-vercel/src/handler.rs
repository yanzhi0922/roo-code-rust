//! Vercel AI Gateway provider handler.
//!
//! Uses the OpenAI-compatible chat completions API via Vercel AI Gateway.
//! Supports prompt caching and has a default temperature of 0.5.
//! Source: `src/api/providers/vercel-ai-gateway.ts`

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::VercelConfig;

/// Default temperature for Vercel AI Gateway.
const DEFAULT_TEMPERATURE: f64 = 0.5;

/// Vercel AI Gateway provider handler.
///
/// Vercel AI Gateway provides a unified API for accessing various LLM models.
/// It follows the OpenAI API format for compatibility.
pub struct VercelHandler {
    inner: OpenAiCompatibleProvider,
}

impl VercelHandler {
    /// Create a new Vercel AI Gateway handler from configuration.
    pub fn new(config: VercelConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config
            .model_id
            .clone()
            .unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(4096),
                context_window: 128000,
                supports_prompt_cache: false,
                input_price: Some(2.50),
                output_price: Some(10.0),
                description: Some("Vercel AI Gateway model (unknown)".to_string()),
                ..Default::default()
            });

        let base_url = config
            .base_url
            .unwrap_or_else(|| VercelConfig::DEFAULT_BASE_URL.to_string());

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "vercel".to_string(),
            base_url,
            api_key: config.api_key,
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::VercelAiGateway,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;

        Ok(Self { inner })
    }

    /// Create a new Vercel handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config = VercelConfig::from_settings(settings);
        let config = config.unwrap_or_else(|| VercelConfig {
            api_key: "dummy-key".to_string(),
            base_url: None,
            model_id: None,
            temperature: None,
            request_timeout: None,
        });
        Self::new(config)
    }
}

#[async_trait]
impl Provider for VercelHandler {
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
        ProviderName::VercelAiGateway
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
            assert!(
                info.input_price.is_some(),
                "Model '{}' missing input_price",
                id
            );
            assert!(
                info.output_price.is_some(),
                "Model '{}' missing output_price",
                id
            );
        }
    }

    #[test]
    fn test_default_url() {
        assert_eq!(
            VercelConfig::DEFAULT_BASE_URL,
            "https://sdk.vercel.ai/api/v1/ai-gateway-gateway"
        );
    }

    #[test]
    fn test_default_temperature() {
        assert_eq!(DEFAULT_TEMPERATURE, 0.5);
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = VercelConfig {
            api_key: "test-key".to_string(),
            base_url: None,
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = VercelHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = VercelConfig {
            api_key: "test-key".to_string(),
            base_url: None,
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = VercelHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = VercelConfig {
            api_key: "test-key".to_string(),
            base_url: None,
            model_id: Some("openai/gpt-4o".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = VercelHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "openai/gpt-4o");
    }

    #[test]
    fn test_handler_from_settings() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let handler = VercelHandler::from_settings(&settings);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_provider_name() {
        let config = VercelConfig {
            api_key: "test-key".to_string(),
            base_url: None,
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = VercelHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::VercelAiGateway);
    }

    #[test]
    fn test_custom_base_url() {
        let config = VercelConfig {
            api_key: "test-key".to_string(),
            base_url: Some("https://custom-vercel.example.com/v1".to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = VercelHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_fallback_model_info() {
        let config = VercelConfig {
            api_key: "test-key".to_string(),
            base_url: None,
            model_id: Some("unknown-vercel-model".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = VercelHandler::new(config).unwrap();
        let (_, info) = handler.get_model();
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_temperature_config() {
        let config = VercelConfig {
            api_key: "test-key".to_string(),
            base_url: None,
            model_id: None,
            temperature: Some(0.7),
            request_timeout: None,
        };
        let handler = VercelHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_models_count() {
        let all_models = models::models();
        assert_eq!(all_models.len(), 3);
    }

    #[test]
    fn test_all_models_support_images() {
        for (id, info) in models::models() {
            assert!(
                info.supports_images.unwrap_or(false),
                "Vercel model '{}' should support images",
                id
            );
        }
    }

    #[test]
    fn test_model_ids_include_provider_prefix() {
        for (id, _) in models::models() {
            assert!(
                id.contains('/'),
                "Vercel model ID '{}' should include provider prefix (e.g., 'anthropic/')",
                id
            );
        }
    }
}
