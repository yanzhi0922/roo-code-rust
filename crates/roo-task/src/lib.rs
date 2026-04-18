//! # Roo Task Engine
//!
//! Task engine for the Roo Code Rust project.
//!
//! This crate provides:
//! - **Types**: [`TaskState`], [`TaskConfig`], [`TaskResult`], [`TaskError`]
//! - **State machine**: [`StateMachine`] for managing task lifecycle
//! - **Events**: [`TaskEvent`], [`TaskEventEmitter`] for event-driven communication
//! - **Loop control**: [`LoopControl`] for iteration and mistake limits
//! - **Engine**: [`TaskEngine`] orchestrating the full task lifecycle
//! - **Config**: [`validate_config`], [`default_config`] for configuration management

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod types;
pub mod state;
pub mod events;
pub mod loop_control;
pub mod config;
pub mod engine;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use types::{TaskConfig, TaskError, TaskResult, TaskState};
pub use state::StateMachine;
pub use events::{TaskEvent, TaskEventEmitter};
pub use loop_control::LoopControl;
pub use engine::TaskEngine;
pub use config::{validate_config, default_config, DEFAULT_MAX_MISTAKES, DEFAULT_MODE};
