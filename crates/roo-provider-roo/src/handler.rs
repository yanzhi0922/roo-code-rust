//! Roo Code Cloud provider handler.
//!
//! Uses the OpenAI-compatible chat completions API via Roo Code Cloud.
//! Supports session token authentication, dynamic model loading,
//! reasoning details, and image generation.
//! Source: `src/api/providers/roo.ts`

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::RooConfig;

/// Roo Code Cloud API provider handler.
///
/// Roo Code Cloud provides access to various LLM models through
/// Roo's own infrastructure. It follows the OpenAI API format.
pub struct RooHandler {
    inner: OpenAiCompatibleProvider,
}

impl RooHandler {
    /// Create a new Roo handler from configuration.
    pub fn new(config: RooConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config
            .model_id
            .clone()
            .unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(8192),
                context_window: 200000,
                supports_prompt_cache: true,
                input_price: Some(3.0),
                output_price: Some(15.0),
                description: Some("Roo Code Cloud model (unknown)".to_string()),
                ..Default::default()
            });

        let base_url = config
            .base_url
            .unwrap_or_else(|| RooConfig::DEFAULT_BASE_URL.to_string());

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "roo".to_string(),
            base_url,
            api_key: config.api_key.unwrap_or_default(),
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(0.0),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::Roo,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;

        Ok(Self { inner })
    }

    /// Create a new Roo handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config = RooConfig::from_settings(settings);
        let config = config.unwrap_or_else(|| RooConfig {
            api_key: None,
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        });
        Self::new(config)
    }
}

#[async_trait]
impl Provider for RooHandler {
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

    async fn complete_prompt(
        &self,
        prompt: &str,
    ) -> Result<String, roo_provider::ProviderError> {
        self.inner.complete_prompt(prompt).await
    }

    fn provider_name(&self) -> ProviderName {
        ProviderName::Roo
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
        assert_eq!(RooConfig::DEFAULT_BASE_URL, "https://api.roocode.com/v1");
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = RooConfig {
            api_key: Some("test-key".to_string()),
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RooHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = RooConfig {
            api_key: Some("test-key".to_string()),
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RooHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = RooConfig {
            api_key: Some("test-key".to_string()),
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: Some("roo-gpt-4o".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = RooHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "roo-gpt-4o");
    }

    #[test]
    fn test_handler_from_settings() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let handler = RooHandler::from_settings(&settings);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_from_settings_with_api_key() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.roo_api_key = Some("test-roo-key".to_string());
        let handler = RooHandler::from_settings(&settings);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_provider_name() {
        let config = RooConfig {
            api_key: Some("test-key".to_string()),
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RooHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::Roo);
    }

    #[test]
    fn test_custom_base_url() {
        let config = RooConfig {
            api_key: Some("test-key".to_string()),
            base_url: Some("https://custom-roo.example.com/v1".to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RooHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_fallback_model_info() {
        let config = RooConfig {
            api_key: Some("test-key".to_string()),
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: Some("unknown-roo-model".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = RooHandler::new(config).unwrap();
        let (_, info) = handler.get_model();
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_temperature_config() {
        let config = RooConfig {
            api_key: Some("test-key".to_string()),
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: None,
            temperature: Some(0.5),
            request_timeout: None,
        };
        let handler = RooHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_models_count() {
        let all_models = models::models();
        assert_eq!(all_models.len(), 4);
    }

    #[test]
    fn test_all_models_support_images() {
        for (id, info) in models::models() {
            assert!(
                info.supports_images.unwrap_or(false),
                "Roo model '{}' should support images",
                id
            );
        }
    }

    #[test]
    fn test_handler_without_api_key() {
        // Roo can work without an API key (session token auth)
        let config = RooConfig {
            api_key: None,
            base_url: Some(RooConfig::DEFAULT_BASE_URL.to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RooHandler::new(config);
        assert!(handler.is_ok());
    }
}
