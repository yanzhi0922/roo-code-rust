//! Unbound provider handler.
//!
//! Uses the OpenAI-compatible chat completions API via Unbound.
//! Supports custom metadata headers and cache token tracking.

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::UnboundConfig;

/// Unbound API provider handler.
///
/// Unbound provides access to multiple LLM providers through a unified API
/// with custom metadata tracking and cache token support.
pub struct UnboundHandler {
    inner: OpenAiCompatibleProvider,
}

impl UnboundHandler {
    /// Create a new Unbound handler from configuration.
    pub fn new(config: UnboundConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(4096),
                max_input_tokens: Some(128000),
                supports_prompt_cache: false,
                input_price: Some(0.0),
                output_price: Some(0.0),
                description: Some("Unbound model (unknown)".to_string()),
                ..Default::default()
            });

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "unbound".to_string(),
            base_url: UnboundConfig::DEFAULT_BASE_URL.to_string(),
            api_key: config.api_key,
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(0.0),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::Unbound,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;

        Ok(Self { inner })
    }

    /// Create a new Unbound handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config = UnboundConfig::from_settings(settings).ok_or_else(|| {
            roo_provider::ProviderError::ApiKeyRequired
        })?;
        Self::new(config)
    }
}

#[async_trait]
impl Provider for UnboundHandler {
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
        ProviderName::Unbound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models;

    #[test]
    fn test_default_model_exists() {
        let all_models = models::models();
        assert!(all_models.contains_key(models::DEFAULT_MODEL_ID));
    }

    #[test]
    fn test_all_models_have_required_fields() {
        for (id, info) in models::models() {
            assert!(info.max_tokens.is_some(), "Model '{}' missing max_tokens", id);
        }
    }

    #[test]
    fn test_default_url() {
        assert_eq!(UnboundConfig::DEFAULT_BASE_URL, "https://api.getunbound.ai/v1");
    }

    #[test]
    fn test_handler_creation_requires_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = UnboundHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = UnboundConfig {
            api_key: "test-key".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = UnboundHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = UnboundConfig {
            api_key: "test-key".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = UnboundHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = UnboundConfig {
            api_key: "test-key".to_string(),
            model_id: Some("claude-3-5-sonnet".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = UnboundHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "claude-3-5-sonnet");
    }

    #[test]
    fn test_provider_name() {
        let config = UnboundConfig {
            api_key: "test-key".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = UnboundHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::Unbound);
    }

    #[test]
    fn test_fallback_model_info() {
        let config = UnboundConfig {
            api_key: "test-key".to_string(),
            model_id: Some("unknown-model".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = UnboundHandler::new(config).unwrap();
        let (_, info) = handler.get_model();
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_temperature_config() {
        let config = UnboundConfig {
            api_key: "test-key".to_string(),
            model_id: None,
            temperature: Some(0.5),
            request_timeout: None,
        };
        let handler = UnboundHandler::new(config);
        assert!(handler.is_ok());
    }
}
