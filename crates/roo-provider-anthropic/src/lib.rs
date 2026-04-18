//! # Roo Provider: Anthropic
//!
//! Anthropic API provider for Roo Code Rust.
//! Uses the native Anthropic Messages API with SSE streaming.
//! Supports extended thinking, prompt caching, and tool use.

mod handler;
mod models;
mod types;

pub use handler::AnthropicHandler;
pub use models::{default_model_id, models};
pub use types::AnthropicConfig;
