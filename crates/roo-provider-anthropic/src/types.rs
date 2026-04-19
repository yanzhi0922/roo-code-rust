//! Anthropic-specific configuration and SSE response types.

use serde::Deserialize;
use roo_types::provider_settings::ProviderSettings;

/// Configuration for the Anthropic provider.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for Anthropic.
    pub api_key: String,
    /// Base URL for the Anthropic API.
    pub base_url: String,
    /// Model ID to use.
    pub model_id: Option<String>,
    /// Temperature for generation.
    pub temperature: Option<f64>,
    /// Whether to use extended thinking.
    pub use_extended_thinking: Option<bool>,
    /// Max thinking tokens for extended thinking.
    pub max_thinking_tokens: Option<u64>,
    /// Request timeout in milliseconds.
    pub request_timeout: Option<u64>,
}

impl AnthropicConfig {
    /// Default Anthropic API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.anthropic.com";

    /// Create configuration from provider settings.
    pub fn from_settings(settings: &ProviderSettings) -> Option<Self> {
        let api_key = settings.api_key.clone()?;
        let base_url = settings
            .anthropic_base_url
            .clone()
            .unwrap_or_else(|| Self::DEFAULT_BASE_URL.to_string());

        Some(Self {
            api_key,
            base_url,
            model_id: settings.api_model_id.clone(),
            temperature: settings.model_temperature.flatten(),
            use_extended_thinking: settings.anthropic_use_extended_thinking,
            max_thinking_tokens: settings.model_max_thinking_tokens,
            request_timeout: settings.request_timeout,
        })
    }
}

// ---------------------------------------------------------------------------
// Anthropic SSE response types
// ---------------------------------------------------------------------------

/// Anthropic SSE event types.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(tag = "type")]
pub enum AnthropicSseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessage },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u64,
        content_block: AnthropicContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u64,
        delta: AnthropicDelta,
    },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: Option<AnthropicUsage>,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u64 },

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "error")]
    Error { error: AnthropicErrorBody },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AnthropicMessage {
    pub id: Option<String>,
    pub model: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(rename = "signature_delta")]
    SignatureDelta {
        signature: String,
    },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AnthropicMessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AnthropicUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub message: Option<String>,
}
