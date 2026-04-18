//! Unbound model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default Unbound model ID.
pub const DEFAULT_MODEL_ID: &str = "default";

/// Returns the supported Unbound fallback models.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "default".to_string(),
        ModelInfo {
            max_tokens: Some(4096),
            max_input_tokens: Some(128000),
            supports_images: false,
            supports_prompt_cache: false,
            input_price: Some(0.0),
            output_price: Some(0.0),
            description: Some("Unbound default model".to_string()),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
