//! Requesty provider handler.
//!
//! Uses the OpenAI-compatible chat completions API via Requesty router.
//! Supports trace_id and mode tracking for observability.

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::RequestyConfig;

/// Requesty API provider handler.
///
/// Requesty is an LLM router that provides observability features
/// like trace_id tracking and mode identification.
pub struct RequestyHandler {
    inner: OpenAiCompatibleProvider,
}

impl RequestyHandler {
    /// Create a new Requesty handler from configuration.
    pub fn new(config: RequestyConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(8192),
                context_window: 200000,
                supports_prompt_cache: true,
                input_price: Some(3.0),
                output_price: Some(15.0),
                description: Some("Requesty model (unknown variant)".to_string()),
                ..Default::default()
            });

        // Ensure base URL ends with /v1 for OpenAI compatibility
        let base_url = if config.base_url.ends_with("/v1") {
            config.base_url
        } else {
            format!("{}/v1", config.base_url.trim_end_matches('/'))
        };

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "requesty".to_string(),
            base_url,
            api_key: config.api_key,
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(0.0),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::Requesty,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;

        Ok(Self { inner })
    }

    /// Create a new Requesty handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config = RequestyConfig::from_settings(settings).ok_or_else(|| {
            roo_provider::ProviderError::ApiKeyRequired
        })?;
        Self::new(config)
    }
}

#[async_trait]
impl Provider for RequestyHandler {
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
        ProviderName::Requesty
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
    fn test_default_url() {
        assert_eq!(RequestyConfig::DEFAULT_BASE_URL, "https://api.requesty.ai");
    }

    #[test]
    fn test_handler_creation_requires_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = RequestyHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: RequestyConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: RequestyConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: RequestyConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("gpt-4o".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "gpt-4o");
    }

    #[test]
    fn test_provider_name() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: RequestyConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::Requesty);
    }

    #[test]
    fn test_base_url_v1_suffix() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: "https://custom.requesty.ai".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_base_url_already_has_v1() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: "https://custom.requesty.ai/v1".to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_fallback_model_info() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: RequestyConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("unknown-model".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config).unwrap();
        let (_, info) = handler.get_model();
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_temperature_config() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: RequestyConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: Some(0.7),
            request_timeout: None,
        };
        let handler = RequestyHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_request_timeout_config() {
        let config = RequestyConfig {
            api_key: "test-key".to_string(),
            base_url: RequestyConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: Some(60000),
        };
        let handler = RequestyHandler::new(config);
        assert!(handler.is_ok());
    }
}
