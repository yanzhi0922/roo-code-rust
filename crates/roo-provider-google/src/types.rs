//! Google Gemini-specific configuration and response types.

use serde::Deserialize;
use roo_types::provider_settings::ProviderSettings;

/// Configuration for the Google Gemini provider.
#[derive(Debug, Clone)]
pub struct GoogleConfig {
    /// API key for Google Gemini.
    pub api_key: String,
    /// Base URL for the Gemini API.
    pub base_url: String,
    /// Model ID to use.
    pub model_id: Option<String>,
    /// Temperature for generation.
    pub temperature: Option<f64>,
    /// Request timeout in milliseconds.
    pub request_timeout: Option<u64>,
}

impl GoogleConfig {
    /// Default Google Gemini API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://generativelanguage.googleapis.com/v1beta";

    /// Create configuration from provider settings.
    pub fn from_settings(settings: &ProviderSettings) -> Option<Self> {
        let api_key = settings
            .google_api_key
            .clone()
            .or(settings.api_key.clone())?;
        let base_url = settings
            .google_gemini_base_url
            .clone()
            .or(settings.gemini_base_url.clone())
            .unwrap_or_else(|| Self::DEFAULT_BASE_URL.to_string());

        Some(Self {
            api_key,
            base_url,
            model_id: settings.api_model_id.clone(),
            temperature: settings.model_temperature.flatten(),
            request_timeout: settings.request_timeout,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemini SSE response types (for parsing streaming responses)
// ---------------------------------------------------------------------------

/// A chunk from the Gemini streaming API.
#[derive(Debug, Deserialize)]
pub struct GeminiStreamResponse {
    pub candidates: Option<Vec<GeminiCandidate>>,
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct GeminiCandidate {
    pub content: Option<GeminiResponseContent>,
    pub finish_reason: Option<String>,
    pub grounding_metadata: Option<GeminiGroundingMetadata>,
}

/// Response content from Gemini (uses flat part structure).
#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct GeminiResponseContent {
    pub parts: Option<Vec<GeminiResponsePart>>,
    pub role: Option<String>,
}

/// A single part in a Gemini response.
#[derive(Debug, Deserialize, Clone)]
pub struct GeminiResponsePart {
    pub text: Option<String>,
    pub function_call: Option<GeminiFunctionCall>,
    pub thought: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct GeminiUsageMetadata {
    pub prompt_token_count: Option<u64>,
    pub candidates_token_count: Option<u64>,
    pub cached_content_token_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct GeminiGroundingMetadata {
    pub grounding_chunks: Option<Vec<GeminiGroundingChunk>>,
    pub search_entry_point: Option<GeminiSearchEntryPoint>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiGroundingChunk {
    pub web: Option<GeminiWebChunk>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiWebChunk {
    pub uri: Option<String>,
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct GeminiSearchEntryPoint {
    pub rendered_content: Option<String>,
}
