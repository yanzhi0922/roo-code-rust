//! MiniMax provider handler.
//!
//! Uses the OpenAI-compatible chat completions API.
//! Supports group_id header for MiniMax API.

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::MiniMaxConfig;

/// Default temperature for MiniMax models.
const DEFAULT_TEMPERATURE: f64 = 1.0;

/// MiniMax API provider handler.
pub struct MiniMaxHandler {
    inner: OpenAiCompatibleProvider,
    #[allow(dead_code)]
    group_id: Option<String>,
}

impl MiniMaxHandler {
    /// Create a new MiniMax handler from configuration.
    pub fn new(config: MiniMaxConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(16_384),
                context_window: 204_800,
                supports_prompt_cache: true,
                input_price: Some(0.3),
                output_price: Some(1.2),
                description: Some("MiniMax model (unknown variant)".to_string()),
                ..Default::default()
            });

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "minimax".to_string(),
            base_url: config.base_url,
            api_key: config.api_key,
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::MiniMax,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;
        let group_id = config.group_id;

        Ok(Self { inner, group_id })
    }

    /// Create a new MiniMax handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config =
            MiniMaxConfig::from_settings(settings).ok_or(roo_provider::ProviderError::ApiKeyRequired)?;
        Self::new(config)
    }
}

#[async_trait]
impl Provider for MiniMaxHandler {
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
        ProviderName::MiniMax
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
            assert!(
                info.max_tokens.is_some(),
                "Model '{}' missing max_tokens",
                id
            );
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
    fn test_minimax_config_default_url() {
        assert_eq!(
            MiniMaxConfig::DEFAULT_BASE_URL,
            "https://api.minimax.chat/v1"
        );
    }

    #[test]
    fn test_handler_creation_requires_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = MiniMaxHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = MiniMaxConfig {
            api_key: "test-key".to_string(),
            base_url: MiniMaxConfig::DEFAULT_BASE_URL.to_string(),
            group_id: None,
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = MiniMaxHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = MiniMaxConfig {
            api_key: "test-key".to_string(),
            base_url: MiniMaxConfig::DEFAULT_BASE_URL.to_string(),
            group_id: None,
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = MiniMaxHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = MiniMaxConfig {
            api_key: "test-key".to_string(),
            base_url: MiniMaxConfig::DEFAULT_BASE_URL.to_string(),
            group_id: None,
            model_id: Some("MiniMax-M2.5".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = MiniMaxHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "MiniMax-M2.5");
    }

    #[test]
    fn test_handler_unknown_model_falls_back() {
        let config = MiniMaxConfig {
            api_key: "test-key".to_string(),
            base_url: MiniMaxConfig::DEFAULT_BASE_URL.to_string(),
            group_id: None,
            model_id: Some("nonexistent-model".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = MiniMaxHandler::new(config).unwrap();
        let (model_id, info) = handler.get_model();
        assert_eq!(model_id, "nonexistent-model");
        assert!(info.max_tokens.is_some());
        assert!(info.input_price.is_some());
    }

    #[test]
    fn test_handler_with_group_id() {
        let config = MiniMaxConfig {
            api_key: "test-key".to_string(),
            base_url: MiniMaxConfig::DEFAULT_BASE_URL.to_string(),
            group_id: Some("test-group-123".to_string()),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = MiniMaxHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_provider_name() {
        let config = MiniMaxConfig {
            api_key: "test-key".to_string(),
            base_url: MiniMaxConfig::DEFAULT_BASE_URL.to_string(),
            group_id: None,
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = MiniMaxHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::MiniMax);
    }

    #[test]
    fn test_from_settings_with_api_key() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.minimax_api_key = Some("test-key".to_string());
        let result = MiniMaxHandler::from_settings(&settings);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_settings_with_group_id() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.minimax_api_key = Some("test-key".to_string());
        settings.minimax_base_url = Some("group-456".to_string());
        let handler = MiniMaxHandler::from_settings(&settings).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_all_models_have_descriptions() {
        for (id, info) in models::models() {
            assert!(
                info.description.is_some(),
                "Model '{}' missing description",
                id
            );
        }
    }

    #[test]
    fn test_all_models_support_prompt_cache() {
        for (id, info) in models::models() {
            assert!(
                info.supports_prompt_cache,
                "Model '{}' should support prompt cache",
                id
            );
        }
    }

    #[test]
    fn test_model_count() {
        let all_models = models::models();
        assert!(all_models.len() >= 8, "Expected at least 8 models, got {}", all_models.len());
    }
}
