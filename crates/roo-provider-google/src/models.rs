//! Google Gemini model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default Gemini model ID.
pub const DEFAULT_MODEL_ID: &str = "gemini-2.5-pro";

/// Returns the supported Google Gemini models.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "gemini-2.5-pro".to_string(),
        ModelInfo {
            max_tokens: Some(65536),
            context_window: 1048576,
            supports_images: Some(true),
            supports_prompt_cache: true,
            supports_reasoning_budget: Some(true),
            input_price: Some(1.25),
            output_price: Some(10.0),
            cache_writes_price: Some(4.50),
            cache_reads_price: Some(1.25),
            description: Some("Google Gemini 2.5 Pro".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "gemini-2.5-flash".to_string(),
        ModelInfo {
            max_tokens: Some(65536),
            context_window: 1048576,
            supports_images: Some(true),
            supports_prompt_cache: true,
            supports_reasoning_budget: Some(true),
            input_price: Some(0.15),
            output_price: Some(3.50),
            cache_writes_price: Some(1.0),
            cache_reads_price: Some(0.15),
            description: Some("Google Gemini 2.5 Flash".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "gemini-2.0-flash".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 1048576,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(0.10),
            output_price: Some(0.40),
            cache_writes_price: Some(0.50),
            cache_reads_price: Some(0.025),
            description: Some("Google Gemini 2.0 Flash".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "gemini-2.0-flash-lite".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 1048576,
            supports_images: Some(true),
            input_price: Some(0.075),
            output_price: Some(0.30),
            description: Some("Google Gemini 2.0 Flash Lite".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "gemini-1.5-pro".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 2097152,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(1.25),
            output_price: Some(5.0),
            cache_writes_price: Some(4.50),
            cache_reads_price: Some(1.25),
            description: Some("Google Gemini 1.5 Pro".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "gemini-1.5-flash".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 1048576,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(0.075),
            output_price: Some(0.30),
            cache_writes_price: Some(0.50),
            cache_reads_price: Some(0.025),
            description: Some("Google Gemini 1.5 Flash".to_string()),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
