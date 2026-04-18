//! Fireworks AI model definitions.

use std::collections::HashMap;
use roo_types::model::ModelInfo;

/// Default Fireworks model ID.
pub const DEFAULT_MODEL_ID: &str = "accounts/fireworks/models/kimi-k2-instruct-0905";

/// Returns the supported Fireworks models.
pub fn models() -> HashMap<String, ModelInfo> {
    let mut m = HashMap::new();

    m.insert(
        "accounts/fireworks/models/kimi-k2-instruct-0905".to_string(),
        ModelInfo {
            max_tokens: Some(16_384),
            context_window: 262_144,
            supports_images: Some(false),
            supports_prompt_cache: true,
            input_price: Some(0.6),
            output_price: Some(2.5),
            cache_reads_price: Some(0.15),
            description: Some(
                "Kimi K2 model gets a new version update: Agentic coding: more accurate, \
                 better generalization across scaffolds."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/kimi-k2-instruct".to_string(),
        ModelInfo {
            max_tokens: Some(16_384),
            context_window: 128_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.6),
            output_price: Some(2.5),
            description: Some(
                "Kimi K2 is a state-of-the-art mixture-of-experts (MoE) language model \
                 with 32 billion activated parameters and 1 trillion total parameters."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/kimi-k2-thinking".to_string(),
        ModelInfo {
            max_tokens: Some(16_000),
            context_window: 256_000,
            supports_images: Some(false),
            supports_prompt_cache: true,
            supports_reasoning_budget: Some(true),
            input_price: Some(0.6),
            output_price: Some(2.5),
            cache_reads_price: Some(0.15),
            description: Some(
                "The kimi-k2-thinking model is a general-purpose agentic reasoning model \
                 developed by Moonshot AI."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/deepseek-r1-0528".to_string(),
        ModelInfo {
            max_tokens: Some(20_480),
            context_window: 160_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(3.0),
            output_price: Some(8.0),
            description: Some(
                "05/28 updated checkpoint of Deepseek R1. Its overall performance is now \
                 approaching that of leading models."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/deepseek-v3".to_string(),
        ModelInfo {
            max_tokens: Some(16_384),
            context_window: 128_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.9),
            output_price: Some(0.9),
            description: Some(
                "A strong Mixture-of-Experts (MoE) language model with 671B total parameters \
                 with 37B activated for each token from Deepseek."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/deepseek-v3p1".to_string(),
        ModelInfo {
            max_tokens: Some(16_384),
            context_window: 163_840,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.56),
            output_price: Some(1.68),
            description: Some(
                "DeepSeek v3.1 is an improved version of the v3 model with enhanced performance, \
                 better reasoning capabilities, and improved code generation."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/llama-v3p3-70b-instruct".to_string(),
        ModelInfo {
            max_tokens: Some(16_384),
            context_window: 131_072,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.9),
            output_price: Some(0.9),
            description: Some(
                "Meta Llama 3.3 70B Instruct model hosted on Fireworks."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507".to_string(),
        ModelInfo {
            max_tokens: Some(32_768),
            context_window: 256_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.22),
            output_price: Some(0.88),
            description: Some(
                "Latest Qwen3 thinking model, competitive against the best closed source models."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct".to_string(),
        ModelInfo {
            max_tokens: Some(32_768),
            context_window: 256_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.45),
            output_price: Some(1.8),
            description: Some(
                "Qwen3's most agentic code model to date."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/glm-4p6".to_string(),
        ModelInfo {
            max_tokens: Some(25_344),
            context_window: 198_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.55),
            output_price: Some(2.19),
            description: Some(
                "Z.ai GLM-4.6 is an advanced coding model with exceptional performance on \
                 complex programming tasks."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/gpt-oss-20b".to_string(),
        ModelInfo {
            max_tokens: Some(16_384),
            context_window: 128_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.07),
            output_price: Some(0.3),
            description: Some(
                "OpenAI gpt-oss-20b: Compact model for local/edge deployments."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m.insert(
        "accounts/fireworks/models/gpt-oss-120b".to_string(),
        ModelInfo {
            max_tokens: Some(16_384),
            context_window: 128_000,
            supports_images: Some(false),
            supports_prompt_cache: false,
            input_price: Some(0.15),
            output_price: Some(0.6),
            description: Some(
                "OpenAI gpt-oss-120b: Production-grade, general-purpose model."
                    .to_string(),
            ),
            ..Default::default()
        },
    );

    m
}

/// Returns the default model ID.
pub fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}
