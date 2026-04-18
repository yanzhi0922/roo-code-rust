//! Provider trait and factory function.
//!
//! Derived from `src/api/index.ts`.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use roo_types::api::{ApiMessage, ApiStreamChunk, ProviderName};
use roo_types::model::ModelInfo;
use roo_types::provider_settings::ProviderSettings;

use crate::error::{ProviderError, Result};

/// A stream of API response chunks.
pub type ApiStream = Pin<Box<dyn Stream<Item = Result<ApiStreamChunk>> + Send>>;

/// Metadata passed to `create_message`.
///
/// Source: `src/api/index.ts` — `ApiHandlerCreateMessageMetadata`
#[derive(Debug, Clone, Default)]
pub struct CreateMessageMetadata {
    pub task_id: Option<String>,
    pub mode: Option<String>,
    pub suppress_previous_response_id: Option<bool>,
    pub store: Option<bool>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: Option<serde_json::Value>,
    pub parallel_tool_calls: Option<bool>,
    pub allowed_function_names: Option<Vec<String>>,
}

/// Core trait for API providers.
///
/// Source: `src/api/index.ts` — `ApiHandler` + `SingleCompletionHandler`
#[async_trait]
pub trait Provider: Send + Sync {
    /// Create a streaming message response.
    async fn create_message(
        &self,
        system_prompt: &str,
        messages: Vec<ApiMessage>,
        tools: Option<Vec<serde_json::Value>>,
        metadata: CreateMessageMetadata,
    ) -> Result<ApiStream>;

    /// Get the model ID and info.
    fn get_model(&self) -> (String, ModelInfo);

    /// Count tokens for content blocks.
    async fn count_tokens(&self, content: &[roo_types::api::ContentBlock]) -> Result<u64> {
        let _ = content;
        Ok(0)
    }

    /// Complete a simple prompt (non-streaming).
    async fn complete_prompt(&self, prompt: &str) -> Result<String>;

    /// Get the provider name.
    fn provider_name(&self) -> ProviderName;
}

/// Builds an API handler for the given configuration.
///
/// Source: `src/api/index.ts` — `buildApiHandler`
///
/// Individual provider implementations live in their own crates.
/// This factory validates configuration and delegates construction.
pub fn build_api_handler(
    configuration: &ProviderSettings,
) -> std::result::Result<Box<dyn Provider>, ProviderError> {
    let provider_name = configuration
        .api_provider
        .ok_or_else(|| ProviderError::Other("No API provider specified".to_string()))?;

    Err(ProviderError::Other(format!(
        "Provider '{}' must be constructed through its dedicated crate",
        provider_name.as_str()
    )))
}
