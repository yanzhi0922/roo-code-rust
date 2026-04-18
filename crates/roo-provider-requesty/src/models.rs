//! Requesty model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default Requesty model ID.
pub const DEFAULT_MODEL_ID: &str = "claude-3-5-sonnet-20241022";

/// Returns the supported Requesty fallback models.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "claude-3-5-sonnet-20241022".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.30),
            description: Some("Claude 3.5 Sonnet via Requesty".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "gpt-4o".to_string(),
        ModelInfo {
            max_tokens: Some(4096),
            max_input_tokens: Some(128000),
            supports_images: true,
            supports_prompt_cache: false,
            input_price: Some(2.50),
            output_price: Some(10.0),
            description: Some("GPT-4o via Requesty".to_string()),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
