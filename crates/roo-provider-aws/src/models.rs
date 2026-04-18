//! AWS Bedrock model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default Bedrock model ID.
pub const DEFAULT_MODEL_ID: &str = "anthropic.claude-sonnet-4-20250514-v1:0";

/// Returns the supported AWS Bedrock models.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 200000,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.3),
            description: Some("Anthropic Claude Sonnet 4 on Bedrock".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "anthropic.claude-opus-4-20250514-v1:0".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 200000,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(15.0),
            output_price: Some(75.0),
            cache_writes_price: Some(18.75),
            cache_reads_price: Some(1.5),
            description: Some("Anthropic Claude Opus 4 on Bedrock".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "anthropic.claude-3-7-sonnet-20250219-v1:0".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 200000,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.3),
            description: Some("Anthropic Claude 3.7 Sonnet on Bedrock".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 200000,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(3.0),
            output_price: Some(15.0),
            cache_writes_price: Some(3.75),
            cache_reads_price: Some(0.3),
            description: Some("Anthropic Claude 3.5 Sonnet v2 on Bedrock".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 200000,
            supports_images: Some(true),
            supports_prompt_cache: true,
            input_price: Some(0.80),
            output_price: Some(4.0),
            cache_writes_price: Some(1.0),
            cache_reads_price: Some(0.08),
            description: Some("Anthropic Claude 3.5 Haiku on Bedrock".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "us.amazon.nova-pro-v1:0".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 300000,
            supports_images: Some(true),
            input_price: Some(0.80),
            output_price: Some(3.20),
            description: Some("Amazon Nova Pro on Bedrock".to_string()),
            ..Default::default()
        },
    );

    m.insert(
        "us.amazon.nova-lite-v1:0".to_string(),
        ModelInfo {
            max_tokens: Some(8192),
            context_window: 300000,
            supports_images: Some(true),
            input_price: Some(0.06),
            output_price: Some(0.24),
            description: Some("Amazon Nova Lite on Bedrock".to_string()),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
