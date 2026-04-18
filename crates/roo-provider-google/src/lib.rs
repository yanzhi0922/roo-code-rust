//! # Roo Provider: Google Gemini
//!
//! Google Gemini API provider for Roo Code Rust.
//! Uses the Gemini generateContent API with SSE streaming.
//! Supports grounding, safety settings, and thinking mode.

mod handler;
mod models;
mod types;

pub use handler::GoogleHandler;
pub use models::{default_model_id, models};
pub use types::GoogleConfig;
