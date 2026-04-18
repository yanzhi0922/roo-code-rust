//! DeepSeek provider handler.
//!
//! Uses the OpenAI-compatible chat completions API.
//! Supports extended thinking mode via `deepseek-reasoner`.

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::DeepSeekConfig;

/// DeepSeek API provider handler.
pub struct DeepSeekHandler {
    inner: OpenAiCompatibleProvider,
}

impl DeepSeekHandler {
    /// Create a new DeepSeek handler from configuration.
    pub fn new(config: DeepSeekConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(8192),
                max_input_tokens: Some(65536),
                supports_prompt_cache: true,
                input_price: Some(0.27),
                output_price: Some(1.10),
                description: Some("DeepSeek model (unknown variant)".to_string()),
                ..Default::default()
            });

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "deepseek".to_string(),
            base_url: config.base_url,
            api_key: config.api_key,
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(0.0),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::DeepSeek,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;

        Ok(Self { inner })
    }

    /// Create a new DeepSeek handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config = DeepSeekConfig::from_settings(settings).ok_or_else(|| {
            roo_provider::ProviderError::ApiKeyRequired
        })?;
        Self::new(config)
    }
}

#[async_trait]
impl Provider for DeepSeekHandler {
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
        ProviderName::DeepSeek
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
    fn test_reasoner_has_thinking_enabled() {
        let all_models = models::models();
        let reasoner = all_models.get("deepseek-reasoner").expect("reasoner should exist");
        assert_eq!(reasoner.thinking, Some(true));
    }

    #[test]
    fn test_deepseek_config_default_url() {
        assert_eq!(
            DeepSeekConfig::DEFAULT_BASE_URL,
            "https://api.deepseek.com"
        );
    }

    #[test]
    fn test_handler_creation_requires_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = DeepSeekHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = DeepSeekConfig {
            api_key: "test-key".to_string(),
            base_url: DeepSeekConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = DeepSeekHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = DeepSeekConfig {
            api_key: "test-key".to_string(),
            base_url: DeepSeekConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = DeepSeekHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = DeepSeekConfig {
            api_key: "test-key".to_string(),
            base_url: DeepSeekConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("deepseek-reasoner".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = DeepSeekHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "deepseek-reasoner");
    }

    #[test]
    fn test_handler_provider_name() {
        let config = DeepSeekConfig {
            api_key: "test-key".to_string(),
            base_url: DeepSeekConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = DeepSeekHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::DeepSeek);
    }

    #[test]
    fn test_config_from_settings() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.api_key = Some("sk-test".to_string());
        settings.model_id = Some("deepseek-reasoner".to_string());

        let config = DeepSeekConfig::from_settings(&settings).unwrap();
        assert_eq!(config.api_key, "sk-test");
        assert_eq!(config.model_id, Some("deepseek-reasoner".to_string()));
    }

    #[test]
    fn test_config_from_settings_custom_base_url() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.api_key = Some("sk-test".to_string());
        settings.deepseek_base_url = Some("https://custom.deepseek.api".to_string());

        let config = DeepSeekConfig::from_settings(&settings).unwrap();
        assert_eq!(config.base_url, "https://custom.deepseek.api");
    }

    #[test]
    fn test_config_from_settings_no_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        assert!(DeepSeekConfig::from_settings(&settings).is_none());
    }

    #[test]
    fn test_models_count() {
        let all_models = models::models();
        assert!(all_models.len() >= 4, "Should have at least 4 DeepSeek models");
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
    fn test_chat_model_supports_cache() {
        let all_models = models::models();
        let chat = all_models.get("deepseek-chat").expect("chat model should exist");
        assert!(chat.supports_prompt_cache);
    }

    #[test]
    fn test_handler_unknown_model_fallback() {
        let config = DeepSeekConfig {
            api_key: "test-key".to_string(),
            base_url: DeepSeekConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("deepseek-unknown-model".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = DeepSeekHandler::new(config).unwrap();
        let (model_id, info) = handler.get_model();
        assert_eq!(model_id, "deepseek-unknown-model");
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_handler_with_temperature() {
        let config = DeepSeekConfig {
            api_key: "test-key".to_string(),
            base_url: DeepSeekConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: Some(0.5),
            request_timeout: None,
        };
        let handler = DeepSeekHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_with_timeout() {
        let config = DeepSeekConfig {
            api_key: "test-key".to_string(),
            base_url: DeepSeekConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: Some(30000),
        };
        let handler = DeepSeekHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_r1_model_has_higher_pricing() {
        let all_models = models::models();
        let chat = all_models.get("deepseek-chat").unwrap();
        let reasoner = all_models.get("deepseek-reasoner").unwrap();
        assert!(
            reasoner.input_price.unwrap() > chat.input_price.unwrap(),
            "Reasoner should cost more than chat"
        );
    }
}
