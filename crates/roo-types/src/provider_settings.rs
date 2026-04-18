//! Provider settings type definitions.
//!
//! Derived from `packages/types/src/provider-settings.ts` (662 lines).
//! Defines configuration for all AI providers.

use serde::{Deserialize, Serialize};

use crate::api::ProviderName;

// ---------------------------------------------------------------------------
// ProviderSettings
// ---------------------------------------------------------------------------

/// Configuration for an AI provider.
///
/// Source: `packages/types/src/provider-settings.ts`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderSettings {
    /// Which provider to use.
    pub api_provider: Option<ProviderName>,

    // --- Common fields ---
    pub api_key: Option<String>,
    pub model_id: Option<String>,
    pub model_max_tokens: Option<u64>,
    pub model_max_thinking_tokens: Option<u64>,
    pub model_reasoning_effort: Option<String>,
    pub model_temperature: Option<f64>,
    pub model_top_p: Option<f64>,

    // --- Anthropic-specific ---
    pub anthropic_base_url: Option<String>,
    pub anthropic_use_extended_thinking: Option<bool>,

    // --- OpenAI-specific ---
    pub openai_base_url: Option<String>,
    pub openai_org_id: Option<String>,
    pub openai_use_legacy_completion: Option<bool>,

    // --- OpenAI Native-specific ---
    pub openai_native_model_max_tokens: Option<u64>,
    pub openai_native_reasoning_effort: Option<String>,

    // --- OpenAI Codex-specific ---
    pub openai_codex_base_url: Option<String>,

    // --- Google / Gemini-specific ---
    pub google_gemini_base_url: Option<String>,
    pub google_api_key: Option<String>,
    pub gemini_base_url: Option<String>,

    // --- AWS Bedrock-specific ---
    pub aws_access_key: Option<String>,
    pub aws_secret_key: Option<String>,
    pub aws_session_token: Option<String>,
    pub aws_region: Option<String>,
    pub aws_use_cross_region_inference: Option<bool>,
    pub aws_bedrock_custom_model_id: Option<String>,
    pub aws_bedrock_endpoint_url: Option<String>,

    // --- Vertex-specific ---
    pub vertex_project_id: Option<String>,
    pub vertex_region: Option<String>,

    // --- Azure-specific ---
    pub azure_api_version: Option<String>,
    pub azure_base_url: Option<String>,
    pub azure_deployment_name: Option<String>,
    pub azure_endpoint: Option<String>,

    // --- OpenRouter-specific ---
    pub openrouter_base_url: Option<String>,
    pub openrouter_model_id: Option<String>,
    pub openrouter_provider_rankings: Option<Vec<String>>,

    // --- Ollama-specific ---
    pub ollama_base_url: Option<String>,
    pub ollama_api_options: Option<serde_json::Value>,

    // --- LM Studio-specific ---
    pub lmstudio_base_url: Option<String>,
    pub lmstudio_model_id: Option<String>,

    // --- DeepSeek-specific ---
    pub deepseek_base_url: Option<String>,

    // --- xAI-specific ---
    pub xai_base_url: Option<String>,

    // --- MiniMax-specific ---
    pub minimax_base_url: Option<String>,
    pub minimax_api_key: Option<String>,
    pub minimax_group_id: Option<String>,

    // --- Moonshot-specific ---
    pub moonshot_base_url: Option<String>,
    pub moonshot_api_key: Option<String>,

    // --- Qwen-specific ---
    pub qwen_base_url: Option<String>,
    pub qwen_api_key: Option<String>,

    // --- ZAI-specific ---
    pub zai_base_url: Option<String>,
    pub zai_api_key: Option<String>,

    // --- Mistral-specific ---
    pub mistral_base_url: Option<String>,
    pub mistral_codestral_url: Option<String>,

    // --- Fireworks-specific ---
    pub fireworks_base_url: Option<String>,

    // --- SambaNova-specific ---
    pub sambanova_base_url: Option<String>,

    // --- Baseten-specific ---
    pub baseten_base_url: Option<String>,
    pub baseten_model_id: Option<String>,
    pub baseten_api_key: Option<String>,

    // --- VS Code LM-specific ---
    pub vscode_lm_model_selector: Option<serde_json::Value>,

    // --- Poe-specific ---
    pub poe_base_url: Option<String>,
    pub poe_api_key: Option<String>,
    pub poe_model_id: Option<String>,

    // --- LiteLLM-specific ---
    pub litellm_base_url: Option<String>,
    pub litellm_api_key: Option<String>,
    pub litellm_model_id: Option<String>,

    // --- Requesty-specific ---
    pub requesty_base_url: Option<String>,
    pub requesty_api_key: Option<String>,
    pub requesty_model_id: Option<String>,

    // --- Unbound-specific ---
    pub unbound_base_url: Option<String>,
    pub unbound_api_key: Option<String>,
    pub unbound_model_id: Option<String>,

    // --- Roo-specific ---
    pub roo_api_key: Option<String>,
    pub roo_base_url: Option<String>,

    // --- Vercel-specific ---
    pub vercel_base_url: Option<String>,
    pub vercel_api_key: Option<String>,
    pub vercel_model_id: Option<String>,

    // --- Router ---
    pub router_model_id: Option<String>,

    // --- Request settings ---
    pub include_developer_docs: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub request_timeout: Option<u64>,
}
