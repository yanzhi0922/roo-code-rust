//! Roo Code Cloud model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default Roo model ID.
pub const DEFAULT_MODEL_ID: &str = "roo-claude-3.5-sonnet";

/// Returns the supported Roo Code Cloud models.
///
/// Roo Code Cloud provides access to various models through its infrastructure.
/// These are the known model configurations.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "roo-claude-3.5-sonnet".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            description: Some("Claude 3.5 Sonnet via Roo Code Cloud".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "roo-claude-3.5-haiku".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_prompt_cache: true,
            input_price: Some(0.80),
            output_price: Some(4.0),
            description: Some("Claude 3.5 Haiku via Roo Code Cloud".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "roo-gpt-4o".to_string(),
        ModelInfo {
            max_tokens: Some(4096),
            max_input_tokens: Some(128000),
            supports_images: true,
            supports_prompt_cache: false,
            input_price: Some(2.50),
            output_price: Some(10.0),
            description: Some("GPT-4o via Roo Code Cloud".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "roo-gemini-2.5-pro".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            max_input_tokens: Some(1048576),
            supports_images: true,
            supports_prompt_cache: true,
            input_price: Some(1.25),
            output_price: Some(10.0),
            description: Some("Gemini 2.5 Pro via Roo Code Cloud".to_string()),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
