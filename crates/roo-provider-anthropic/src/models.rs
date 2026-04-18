//! Anthropic model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default Anthropic model ID.
pub const DEFAULT_MODEL_ID: &str = "claude-sonnet-4-20250514";

/// Returns the supported Anthropic models.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "claude-sonnet-4-20250514".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_computer_use: true,
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.3),
            thinking: Some(true),
            min_thinking_tokens: Some(1024),
            description: Some("Anthropic Claude Sonnet 4".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "claude-opus-4-20250514".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_computer_use: true,
            supports_prompt_cache: true,
            input_price: Some(15.0),
            output_price: Some(75.0),
            cache_writes_price: Some(18.75),
            cache_reads_price: Some(1.5),
            thinking: Some(true),
            min_thinking_tokens: Some(1024),
            description: Some("Anthropic Claude Opus 4".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "claude-3-7-sonnet-20250219".to_string(),
        ModelInfo {
            max_tokens: Some(16384),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_computer_use: true,
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.3),
            thinking: Some(true),
            min_thinking_tokens: Some(1024),
            description: Some("Anthropic Claude 3.7 Sonnet".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "claude-3-5-sonnet-20241022".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_computer_use: true,
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.3),
            description: Some("Anthropic Claude 3.5 Sonnet (v2)".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "claude-3-5-haiku-20241022".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_prompt_cache: true,
            input_price: Some(0.80),
            output_price: Some(4.0),
            cache_writes_price: Some(1.0),
            cache_reads_price: Some(0.08),
            description: Some("Anthropic Claude 3.5 Haiku".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "claude-3-opus-20240229".to_string(),
        ModelInfo {
            max_tokens: Some(4096),
            max_input_tokens: Some(200000),
            supports_images: true,
            supports_prompt_cache: true,
            input_price: Some(15.0),
            output_price: Some(75.0),
            cache_writes_price: Some(18.75),
            cache_reads_price: Some(1.5),
            description: Some("Anthropic Claude 3 Opus".to_string()),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
