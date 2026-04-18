//! Model type definitions.
//!
//! Derived from `packages/types/src/model.ts`.
//! Defines model information schema and related types.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------

/// Information about an AI model.
///
/// Source: `packages/types/src/model.ts` — `modelInfoSchema`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// Maximum tokens the model can output in a single response.
    pub max_tokens: Option<u64>,
    /// Maximum input tokens the model can accept.
    pub max_input_tokens: Option<u64>,
    /// Whether the model supports image inputs.
    #[serde(default)]
    pub supports_images: bool,
    /// Whether the model supports computer use.
    #[serde(default)]
    pub supports_computer_use: bool,
    /// Whether the model supports prompt caching.
    #[serde(default)]
    pub supports_prompt_cache: bool,
    /// Whether the model uses the OpenAI responses API.
    #[serde(default)]
    pub supports_responses_api: Option<bool>,
    /// Cost per million input tokens (in USD).
    pub input_price: Option<f64>,
    /// Cost per million output tokens (in USD).
    pub output_price: Option<f64>,
    /// Cost per million cache write tokens (in USD).
    pub cache_writes_price: Option<f64>,
    /// Cost per million cache read tokens (in USD).
    pub cache_reads_price: Option<f64>,
    /// Description of the model.
    pub description: Option<String>,
    /// Thinking/reasoning token budget.
    pub thinking: Option<bool>,
    /// Minimum thinking token budget.
    pub min_thinking_tokens: Option<u64>,
    /// Whether the model supports tool use.
    #[serde(default = "default_true")]
    pub supports_tool_use: Option<bool>,
}

const fn default_true() -> Option<bool> {
    Some(true)
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            max_tokens: None,
            max_input_tokens: None,
            supports_images: false,
            supports_computer_use: false,
            supports_prompt_cache: false,
            supports_responses_api: None,
            input_price: None,
            output_price: None,
            cache_writes_price: None,
            cache_reads_price: None,
            description: None,
            thinking: None,
            min_thinking_tokens: None,
            supports_tool_use: Some(true),
        }
    }
}

// ---------------------------------------------------------------------------
// ModelRecord
// ---------------------------------------------------------------------------

/// A map of model ID to model info.
///
/// Source: `packages/types/src/model.ts`
pub type ModelRecord = std::collections::HashMap<String, ModelInfo>;
