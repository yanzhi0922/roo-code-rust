//! OpenRouter provider handler.
//!
//! Uses the OpenAI-compatible chat completions API via OpenRouter's gateway.
//! OpenRouter adds extra headers for site URL and ranking preferences.

use async_trait::async_trait;
use roo_provider::{
    ApiStream, BaseProvider, CreateMessageMetadata, Provider,
};
use roo_provider::error::{ProviderError, Result};
use roo_types::api::{ApiMessage, ProviderName};
use roo_types::model::ModelInfo;

use crate::models;
use crate::types::OpenRouterConfig;

/// OpenRouter API provider handler.
pub struct OpenRouterHandler {
    base: BaseProvider,
    http_client: reqwest::Client,
    api_key: String,
    base_url: String,
    temperature: f64,
}

impl OpenRouterHandler {
    /// Create a new OpenRouter handler from configuration.
    pub fn new(config: OpenRouterConfig) -> Result<Self> {
        let model_id = config.model_id.unwrap_or_else(|| models::default_model_id());
        let model_info = models::models()
            .get(&model_id)
            .cloned()
            .unwrap_or_else(|| ModelInfo {
                max_tokens: Some(8192),
                context_window: 128000,
                supports_images: Some(true),
                description: Some("OpenRouter model (unknown variant)".to_string()),
                ..Default::default()
            });

        let base = BaseProvider::new(model_id, model_info, ProviderName::OpenRouter);

        let mut client_builder = reqwest::Client::builder();
        if let Some(timeout) = config.request_timeout {
            client_builder =
                client_builder.timeout(std::time::Duration::from_millis(timeout));
        }
        let http_client = client_builder.build().map_err(ProviderError::Reqwest)?;

        Ok(Self {
            base,
            http_client,
            api_key: config.api_key,
            base_url: config.base_url,
            temperature: config.temperature.unwrap_or(0.0),
        })
    }

    /// Create a new OpenRouter handler from provider settings.
    pub fn from_settings(
        settings: &roo_types::provider_settings::ProviderSettings,
    ) -> Result<Self> {
        let config = OpenRouterConfig::from_settings(settings)
            .ok_or(ProviderError::ApiKeyRequired)?;
        Self::new(config)
    }

}

#[async_trait]
impl Provider for OpenRouterHandler {
    async fn create_message(
        &self,
        system_prompt: &str,
        messages: Vec<ApiMessage>,
        tools: Option<Vec<serde_json::Value>>,
        metadata: CreateMessageMetadata,
    ) -> Result<ApiStream> {
        // Delegate to OpenAiCompatibleProvider via a temporary inner provider
        let config = roo_provider::OpenAiCompatibleConfig {
            provider_name: "openrouter".to_string(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            default_model_id: models::default_model_id(),
            default_temperature: self.temperature,
            model_id: Some(self.base.model_id.clone()),
            model_info: self.base.model_info.clone(),
            provider_name_enum: ProviderName::OpenRouter,
            request_timeout: None,
        };

        let inner = roo_provider::OpenAiCompatibleProvider::new(config)?;
        inner
            .create_message(system_prompt, messages, tools, metadata)
            .await
    }

    fn get_model(&self) -> (String, ModelInfo) {
        self.base.get_model()
    }

    async fn complete_prompt(&self, prompt: &str) -> Result<String> {
        let (model, _) = self.base.get_model();

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }]
        });

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://roocode.com")
            .header("X-Title", "Roo Code")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::api_error("openrouter", e))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ProviderError::api_error_response(
                "openrouter", status, text,
            ));
        }

        let resp: serde_json::Value = response.json().await.map_err(ProviderError::Reqwest)?;
        Ok(resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    fn provider_name(&self) -> ProviderName {
        ProviderName::OpenRouter
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
    fn test_all_models_have_pricing() {
        for (id, info) in models::models() {
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
    fn test_config_default_url() {
        assert_eq!(
            OpenRouterConfig::DEFAULT_BASE_URL,
            "https://openrouter.ai/api/v1"
        );
    }

    #[test]
    fn test_handler_creation_requires_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        let result = OpenRouterHandler::from_settings(&settings);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_creation_with_config() {
        let config = OpenRouterConfig {
            api_key: "test-key".to_string(),
            base_url: OpenRouterConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = OpenRouterHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_handler_uses_default_model() {
        let config = OpenRouterConfig {
            api_key: "test-key".to_string(),
            base_url: OpenRouterConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = OpenRouterHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, models::DEFAULT_MODEL_ID);
    }

    #[test]
    fn test_handler_custom_model() {
        let config = OpenRouterConfig {
            api_key: "test-key".to_string(),
            base_url: OpenRouterConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("openai/gpt-4o".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = OpenRouterHandler::new(config).unwrap();
        let (model_id, _) = handler.get_model();
        assert_eq!(model_id, "openai/gpt-4o");
    }

    #[test]
    fn test_handler_provider_name() {
        let config = OpenRouterConfig {
            api_key: "test-key".to_string(),
            base_url: OpenRouterConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: None,
        };
        let handler = OpenRouterHandler::new(config).unwrap();
        assert_eq!(handler.provider_name(), ProviderName::OpenRouter);
    }

    #[test]
    fn test_config_from_settings() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.api_key = Some("sk-or-test".to_string());
        settings.open_router_model_id = Some("openai/gpt-4o".to_string());

        let config = OpenRouterConfig::from_settings(&settings).unwrap();
        assert_eq!(config.api_key, "sk-or-test");
        assert_eq!(config.model_id, Some("openai/gpt-4o".to_string()));
    }

    #[test]
    fn test_config_from_settings_custom_base_url() {
        let mut settings = roo_types::provider_settings::ProviderSettings::default();
        settings.api_key = Some("sk-or-test".to_string());
        settings.open_router_base_url = Some("https://custom.openrouter.api".to_string());

        let config = OpenRouterConfig::from_settings(&settings).unwrap();
        assert_eq!(config.base_url, "https://custom.openrouter.api");
    }

    #[test]
    fn test_config_from_settings_no_api_key() {
        let settings = roo_types::provider_settings::ProviderSettings::default();
        assert!(OpenRouterConfig::from_settings(&settings).is_none());
    }

    #[test]
    fn test_models_count() {
        let all_models = models::models();
        assert!(
            all_models.len() >= 5,
            "Should have at least 5 OpenRouter models"
        );
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
    fn test_claude_model_supports_images() {
        let all_models = models::models();
        let claude = all_models
            .get("anthropic/claude-sonnet-4")
            .expect("claude model should exist");
        assert!(claude.supports_images.unwrap_or(false));
    }

    #[test]
    fn test_handler_unknown_model_fallback() {
        let config = OpenRouterConfig {
            api_key: "test-key".to_string(),
            base_url: OpenRouterConfig::DEFAULT_BASE_URL.to_string(),
            model_id: Some("vendor/unknown-model".to_string()),
            temperature: None,
            request_timeout: None,
        };
        let handler = OpenRouterHandler::new(config).unwrap();
        let (model_id, info) = handler.get_model();
        assert_eq!(model_id, "vendor/unknown-model");
        assert!(info.max_tokens.is_some());
    }

    #[test]
    fn test_handler_with_timeout() {
        let config = OpenRouterConfig {
            api_key: "test-key".to_string(),
            base_url: OpenRouterConfig::DEFAULT_BASE_URL.to_string(),
            model_id: None,
            temperature: None,
            request_timeout: Some(60000),
        };
        let handler = OpenRouterHandler::new(config);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_gemini_model_has_large_context() {
        let all_models = models::models();
        let gemini = all_models
            .get("google/gemini-2.5-pro-preview")
            .expect("gemini model should exist");
        assert!(gemini.context_window > 500000);
    }
}
