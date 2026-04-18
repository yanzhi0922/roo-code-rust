//! Ollama provider handler.
//!
//! Uses the OpenAI-compatible chat completions API provided by Ollama.
//! Ollama does not require an API key for local instances.

use async_trait::async_trait;
use roo_provider::{
    ApiStream, CreateMessageMetadata, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Provider,
};
use roo_types::api::ProviderName;
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::OllamaConfig;

/// Ollama API provider handler.
pub struct OllamaHandler {
    inner: OpenAiCompatibleProvider,
}

impl OllamaHandler {
    /// Create a new Ollama handler from configuration.
    pub fn new(config: OllamaConfig) -> Result<Self, roo_provider::ProviderError> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(8192),
                context_window: 131072,
                description: Some("Ollama model (unknown variant)".to_string()),
                ..Default::default()
            });

        let compatible_config = OpenAiCompatibleConfig {
            provider_name: "ollama".to_string(),
            base_url: config.base_url,
            // Ollama doesn't require an API key, use a placeholder
            api_key: "ollama".to_string(),
            default_model_id: models::default_model_id(),
            default_temperature: config.temperature.unwrap_or(0.0),
            model_id: Some(model_id),
            model_info,
            provider_name_enum: ProviderName::Ollama,
            request_timeout: config.request_timeout,
        };

        let inner = OpenAiCompatibleProvider::new(compatible_config)?;

        Ok(Self { inner })
    }

    /// Create a new Ollama handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self, roo_provider::ProviderError> {
        let config = OllamaConfig::from_settings(settings);
        Self::new(config)
    }
}

#[async_trait]
impl Provider for OllamaHandler {
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
        ProviderName::Ollama
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
    fn test_all_models_have_max_tokens() {
        for (id, info) in models::models() {
            assert!(
                info.max_tokens.is_some(),
                "Model '{}' missing max_tokens",
                id
            );
        }
    }

    #[test]
    fn test_config_default_url() {
        assert_eq!(
            OllamaConfig::DEFAULT_BASE_URL,
            "http://localhost:11434/v1"
        );
    }

    #[test]
    fn test_handler_creation_no_api_key_required() {
        // Ollama doesn't require an API key
        let config = OllamaConfig {
            base_url: OllamaConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            api_options: None,
        };
        let handler = OllamaHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = OllamaConfig {
            base_url: OllamaConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            api_options: None,
        };
        let handler = OllamaHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = OllamaConfig {
            base_url: OllamaConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("mistral".to_string()),
            temperature: None,
            request_timeout: None,
            api_options: None,
        };
        let handler = OllamaHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "mistral");
    }

    #[test]
    fn test_handler_provider_name() {
        let config = OllamaConfig {
            base_url: OllamaConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
            api_options: None,
        };
        let handler = OllamaHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::Ollama);
    }

    #[test]
    fn test_config_from_settings() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.api_model_id = Some("codellama".to_string());

        let config = OllamaConfig::from_settings(&settings);
        assert_eq!(config.model_id, Some("codellama".to_string()));
        assert_eq!(config.base_url, OllamaConfig::DEFAULT_BASE_URL);
    }

    #[test]
    fn test_config_from_settings_custom_url() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.ollama_base_url = Some("http://192.168.1.100:11434/v1".to_string());

        let config = OllamaConfig::from_settings(&settings);
        assert_eq!(config.base_url, "http://192.168.1.100:11434/v1");
    }

    #[test]
    fn test_config_from_settings_with_options() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.ollama_api_options = Some(serde_json::json!({"num_ctx": 4096}));

        let config = OllamaConfig::from_settings(&settings);
        assert!(config.api_options.is_some());
    }

    #[test]
    fn test_models_count() {
        let all_models = models::models();
        assert!(all_models.len() >= 5, "Should have at least 5 Ollama models");
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
    fn test_llama_supports_images() {
        let all_models = models::models();
        let llama = all_models.get("llama3.2").expect("llama3.2 should exist");
        assert!(llama.supports_images.unwrap_or(false));
    }

    #[test]
    fn test_handler_unknown_model_fallback() {
        let config = OllamaConfig {
            base_url: OllamaConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("custom-model".to_string()),
            temperature: None,
            request_timeout: None,
            api_options: None,
        };
        let handler = OllamaHandler::new(config).unwrap();
        let (model_id, info) = handler.get_model();
        assert_eq!(model_id, "custom-model");
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_handler_from_settings() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        // Ollama always works since no API key is required
        let handler = OllamaHandler::from_settings(&settings);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_with_timeout() {
        let config = OllamaConfig {
            base_url: OllamaConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: Some(120000),
            api_options: None,
        };
        let handler = OllamaHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_no_pricing_for_local_models() {
        // Ollama models are local, so pricing should be None
        for (_, info) in models::models() {
            assert!(
                info.input_price.is_none(),
                "Local Ollama models should not have pricing"
            );
        }
    }
}
