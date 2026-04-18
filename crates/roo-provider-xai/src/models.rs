//! xAI/Grok model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default xAI model ID.
pub const DEFAULT_MODEL_ID: &str = "grok-3";

/// Returns the supported xAI models.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "grok-3".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(131072),
            supports_images: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            description: Some("xAI Grok 3".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "grok-3-mini".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(131072),
            supports_images: true,
            thinking: Some(true),
            input_price: Some(0.30),
            output_price: Some(0.50),
            description: Some("xAI Grok 3 Mini (reasoning)".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "grok-3-fast".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(131072),
            supports_images: true,
            input_price: Some(5.0),
            output_price: Some(25.0),
            description: Some("xAI Grok 3 Fast".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "grok-2".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(131072),
            supports_images: true,
            input_price: Some(2.0),
            output_price: Some(10.0),
            description: Some("xAI Grok 2".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "grok-2-mini".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(131072),
            input_price: Some(0.20),
            output_price: Some(0.30),
            description: Some("xAI Grok 2 Mini".to_string()),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
