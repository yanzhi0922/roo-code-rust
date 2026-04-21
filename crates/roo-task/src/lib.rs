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
//! - **Stream parser**: [`StreamParser`] for parsing API streaming responses
//! - **Tool dispatcher**: [`ToolDispatcher`] for routing tool calls to handlers
//! - **Message builder**: [`MessageBuilder`] for constructing API messages
//! - **Agent loop**: [`AgentLoop`] for the core agent execution loop

// ---------------------------------------------------------------------------
// Module declarations
// ---------------------------------------------------------------------------

pub mod types;
pub mod state;
pub mod events;
pub mod loop_control;
pub mod config;
pub mod engine;
pub mod stream_parser;
pub mod tool_dispatcher;
pub mod message_builder;
pub mod agent_loop;
pub mod task_manager;
pub mod ask_say;
pub mod native_tool_call_parser;
pub mod task_lifecycle;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use types::{
    TaskConfig, TaskError, TaskResult, TaskState, StreamingState,
    AssistantMessageContent, TextContent, ToolUse, McpToolUse,
    ToolCallStreamEvent, StreamingToolCallState, RawChunkTrackerEntry,
    StackItem, AttemptResult, DiffStrategy,
    is_mcp_tool_name, parse_mcp_tool_name, normalize_mcp_tool_name,
    TOOL_PARAM_NAMES, is_valid_tool_param,
};
pub use state::StateMachine;
pub use events::{TaskEvent, TaskEventEmitter};
pub use loop_control::LoopControl;
pub use engine::TaskEngine;
pub use config::{validate_config, default_config, DEFAULT_MAX_MISTAKES, DEFAULT_MODE};
pub use stream_parser::{StreamParser, ParsedStreamContent, ParsedToolCall, StreamUsage};
pub use tool_dispatcher::{ToolDispatcher, ToolExecutionResult, ToolContext, ToolHandler};
pub use message_builder::MessageBuilder;
pub use agent_loop::{AgentLoop, AgentLoopConfig};
pub use task_manager::TaskManager;
pub use ask_say::{AskSayHandler, AskResponse, AskResult, AskIgnoredError, SayOptions};
pub use task_lifecycle::{TaskLifecycle, ServiceRefs};