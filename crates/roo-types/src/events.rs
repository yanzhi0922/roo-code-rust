//! Event type definitions.
//!
//! Derived from `packages/types/src/events.ts`.
//! Defines all RooCode event names and their payload types.

use serde::{Deserialize, Serialize};

use crate::message::{ClineMessage, QueuedMessage, TokenUsage};
use crate::tool::ToolUsage;

// ---------------------------------------------------------------------------
// RooCodeEventName
// ---------------------------------------------------------------------------

/// All event names used in the Roo Code event system.
///
/// Source: `packages/types/src/events.ts` — `RooCodeEventName`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RooCodeEventName {
    // Task Provider Lifecycle
    TaskCreated,

    // Task Lifecycle
    TaskStarted,
    TaskCompleted,
    TaskAborted,
    TaskFocused,
    TaskUnfocused,
    TaskActive,
    TaskInteractive,
    TaskResumable,
    TaskIdle,

    // Subtask Lifecycle
    TaskPaused,
    TaskUnpaused,
    TaskSpawned,
    TaskDelegated,
    TaskDelegationCompleted,
    TaskDelegationResumed,

    // Task Execution
    Message,
    TaskModeSwitched,
    TaskAskResponded,
    TaskUserMessage,
    QueuedMessagesUpdated,

    // Task Analytics
    TaskTokenUsageUpdated,
    TaskToolFailed,

    // Configuration Changes
    ModeChanged,
    ProviderProfileChanged,

    // Query Responses
    CommandsResponse,
    ModesResponse,
    ModelsResponse,

    // Evals
    EvalPass,
    EvalFail,
}

// ---------------------------------------------------------------------------
// Event Payloads
// ---------------------------------------------------------------------------

/// Payload for TaskCompleted event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskCompletedPayload {
    pub task_id: String,
    pub token_usage: TokenUsage,
    pub tool_usage: ToolUsage,
    pub is_subtask: bool,
}

/// Payload for TaskDelegated event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDelegatedPayload {
    pub parent_task_id: String,
    pub child_task_id: String,
}

/// Payload for TaskDelegationCompleted event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDelegationCompletedPayload {
    pub parent_task_id: String,
    pub child_task_id: String,
    pub completion_result_summary: String,
}

/// Payload for TaskDelegationResumed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDelegationResumedPayload {
    pub parent_task_id: String,
    pub child_task_id: String,
}

/// Payload for Message event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePayload {
    pub task_id: String,
    pub action: String,
    pub message: Option<ClineMessage>,
}

/// Payload for TaskModeSwitched event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskModeSwitchedPayload {
    pub task_id: String,
    pub mode: String,
}

/// Payload for TaskAskResponded event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAskRespondedPayload {
    pub task_id: String,
    pub ask: String,
    pub response: Option<String>,
    pub images: Option<Vec<String>>,
    pub text: Option<String>,
    pub files: Option<Vec<String>>,
}

/// Payload for TaskUserMessage event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskUserMessagePayload {
    pub task_id: String,
    pub message: String,
    pub images: Option<Vec<String>>,
}

/// Payload for QueuedMessagesUpdated event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedMessagesUpdatedPayload {
    pub task_id: String,
    pub queued_messages: Vec<QueuedMessage>,
}

/// Payload for TaskTokenUsageUpdated event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTokenUsageUpdatedPayload {
    pub task_id: String,
    pub token_usage: TokenUsage,
}

/// Payload for TaskToolFailed event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskToolFailedPayload {
    pub task_id: String,
    pub tool: String,
    pub error: String,
}

/// Payload for ModeChanged event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeChangedPayload {
    pub task_id: Option<String>,
    pub mode: String,
}

/// Payload for ProviderProfileChanged event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderProfileChangedPayload {
    pub task_id: Option<String>,
    pub profile: Option<String>,
}

/// Payload for CommandsResponse event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandsResponsePayload {
    pub commands: Vec<serde_json::Value>,
}

/// Payload for ModesResponse event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModesResponsePayload {
    pub modes: Vec<serde_json::Value>,
}

/// Payload for ModelsResponse event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponsePayload {
    pub models: Vec<serde_json::Value>,
}

/// Payload for EvalPass / EvalFail events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalPayload {
    pub eval_id: String,
    pub task_id: String,
    pub result: Option<serde_json::Value>,
}
