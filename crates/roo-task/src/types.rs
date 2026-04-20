//! Task engine type definitions.
//!
//! Defines TaskState, TaskConfig, TaskResult, TaskEvent, TaskError,
//! StreamingState, and related types for the task engine.
//!
//! Source: `src/core/task/Task.ts` — Task class properties and types

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use roo_types::message::TokenUsage;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default checkpoint timeout in seconds.
///
/// Source: `packages/types/src/provider-settings.ts` — `DEFAULT_CHECKPOINT_TIMEOUT_SECONDS`
pub const DEFAULT_CHECKPOINT_TIMEOUT_SECONDS: usize = 30;

/// Maximum checkpoint timeout in seconds.
pub const MAX_CHECKPOINT_TIMEOUT_SECONDS: usize = 600;

/// Minimum checkpoint timeout in seconds.
pub const MIN_CHECKPOINT_TIMEOUT_SECONDS: usize = 1;

/// Maximum exponential backoff in seconds for API retries.
///
/// Source: `src/core/task/Task.ts` — `MAX_EXPONENTIAL_BACKOFF_SECONDS`
pub const MAX_EXPONENTIAL_BACKOFF_SECONDS: u64 = 600;

/// Forced context reduction percent (keep 75%, remove 25%).
///
/// Source: `src/core/task/Task.ts` — `FORCED_CONTEXT_REDUCTION_PERCENT`
pub const FORCED_CONTEXT_REDUCTION_PERCENT: u64 = 75;

/// Maximum retries for context window errors.
///
/// Source: `src/core/task/Task.ts` — `MAX_CONTEXT_WINDOW_RETRIES`
pub const MAX_CONTEXT_WINDOW_RETRIES: usize = 3;

// ---------------------------------------------------------------------------
// TaskState
// ---------------------------------------------------------------------------

/// The state of a task in its lifecycle.
///
/// State transitions follow a strict state machine:
/// - `Idle` → `Running`
/// - `Running` → `Paused` | `Completed` | `Aborted` | `Delegated`
/// - `Paused` → `Running` | `Aborted`
/// - `Completed` | `Aborted` | `Delegated` → (terminal, no transitions)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Idle,
    Running,
    Paused,
    Completed,
    Aborted,
    Delegated,
}

impl TaskState {
    /// Check whether a transition from `self` to `target` is valid.
    pub fn can_transition_to(&self, target: &TaskState) -> bool {
        match self {
            TaskState::Idle => matches!(target, TaskState::Running),
            TaskState::Running => matches!(
                target,
                TaskState::Paused | TaskState::Completed | TaskState::Aborted | TaskState::Delegated
            ),
            TaskState::Paused => matches!(target, TaskState::Running | TaskState::Aborted),
            TaskState::Delegated => matches!(target, TaskState::Running),
            TaskState::Completed | TaskState::Aborted => false,
        }
    }

    /// Returns `true` if this state is terminal (no further transitions).
    ///
    /// Note: `Delegated` is NOT terminal because it can transition back to
    /// `Running` via `resumeAfterDelegation()`. This matches the TS behavior.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Aborted
        )
    }

    /// Returns `true` if the task is currently active.
    pub fn is_active(&self) -> bool {
        matches!(self, TaskState::Running | TaskState::Paused)
    }
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskState::Idle => write!(f, "idle"),
            TaskState::Running => write!(f, "running"),
            TaskState::Paused => write!(f, "paused"),
            TaskState::Completed => write!(f, "completed"),
            TaskState::Aborted => write!(f, "aborted"),
            TaskState::Delegated => write!(f, "delegated"),
        }
    }
}

impl Default for TaskState {
    fn default() -> Self {
        Self::Idle
    }
}

// ---------------------------------------------------------------------------
// TaskConfig
// ---------------------------------------------------------------------------

/// Configuration for a task.
///
/// Source: `src/core/task/Task.ts` — `TaskOptions` and constructor parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// Unique task identifier.
    pub task_id: String,
    /// Root task ID (for subtasks).
    pub root_task_id: Option<String>,
    /// Parent task ID (for subtasks).
    pub parent_task_id: Option<String>,
    /// Task number (for subtask ordering).
    pub task_number: usize,
    /// Current working directory.
    pub cwd: String,
    /// Task mode (e.g., "code", "architect", "ask").
    pub mode: String,
    /// API configuration name (provider profile).
    pub api_config_name: Option<String>,
    /// Workspace path.
    pub workspace: String,
    /// Maximum iterations (None = unlimited).
    pub max_iterations: Option<usize>,
    /// Whether auto-approval is enabled.
    pub auto_approval: bool,
    /// Whether checkpoints are enabled.
    ///
    /// Source: `src/core/task/Task.ts` — `enableCheckpoints`
    pub enable_checkpoints: bool,
    /// Checkpoint timeout in seconds.
    ///
    /// Source: `src/core/task/Task.ts` — `checkpointTimeout`
    pub checkpoint_timeout: usize,
    /// Consecutive mistake limit (overrides global default).
    ///
    /// Source: `src/core/task/Task.ts` — `consecutiveMistakeLimit`
    pub consecutive_mistake_limit: usize,
    /// Initial task text (user message).
    ///
    /// Source: `src/core/task/Task.ts` — `task` option
    pub task_text: Option<String>,
    /// Initial images.
    ///
    /// Source: `src/core/task/Task.ts` — `images` option
    pub images: Vec<String>,
    /// History item ID to resume from.
    ///
    /// Source: `src/core/task/Task.ts` — `historyItem` option
    pub history_item_id: Option<String>,
    /// Base storage path for task persistence.
    ///
    /// When set, the engine will persist messages and metadata to disk.
    /// When `None`, persistence methods are no-ops.
    pub storage_path: Option<String>,
}

impl TaskConfig {
    /// Create a new task config with the given task ID and working directory.
    pub fn new(task_id: impl Into<String>, cwd: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            root_task_id: None,
            parent_task_id: None,
            task_number: 0,
            cwd: cwd.into(),
            mode: "code".to_string(),
            api_config_name: None,
            workspace: String::new(),
            max_iterations: None,
            auto_approval: false,
            enable_checkpoints: true,
            checkpoint_timeout: DEFAULT_CHECKPOINT_TIMEOUT_SECONDS,
            consecutive_mistake_limit: crate::config::DEFAULT_MAX_MISTAKES,
            task_text: None,
            images: Vec::new(),
            history_item_id: None,
            storage_path: None,
        }
    }

    /// Set the mode for this task.
    pub fn with_mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = mode.into();
        self
    }

    /// Set the API config name.
    pub fn with_api_config(mut self, name: impl Into<String>) -> Self {
        self.api_config_name = Some(name.into());
        self
    }

    /// Set the workspace.
    pub fn with_workspace(mut self, workspace: impl Into<String>) -> Self {
        self.workspace = workspace.into();
        self
    }

    /// Set max iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Set auto-approval.
    pub fn with_auto_approval(mut self, auto: bool) -> Self {
        self.auto_approval = auto;
        self
    }

    /// Set the root task ID.
    pub fn with_root_task_id(mut self, id: impl Into<String>) -> Self {
        self.root_task_id = Some(id.into());
        self
    }

    /// Set the parent task ID.
    pub fn with_parent_task_id(mut self, id: impl Into<String>) -> Self {
        self.parent_task_id = Some(id.into());
        self
    }

    /// Set the task number.
    pub fn with_task_number(mut self, num: usize) -> Self {
        self.task_number = num;
        self
    }

    /// Set whether checkpoints are enabled.
    pub fn with_checkpoints(mut self, enabled: bool) -> Self {
        self.enable_checkpoints = enabled;
        self
    }

    /// Set the checkpoint timeout in seconds.
    pub fn with_checkpoint_timeout(mut self, timeout: usize) -> Self {
        self.checkpoint_timeout = timeout;
        self
    }

    /// Set the consecutive mistake limit.
    pub fn with_consecutive_mistake_limit(mut self, limit: usize) -> Self {
        self.consecutive_mistake_limit = limit;
        self
    }

    /// Set the initial task text.
    pub fn with_task_text(mut self, text: impl Into<String>) -> Self {
        self.task_text = Some(text.into());
        self
    }

    /// Set the initial images.
    pub fn with_images(mut self, images: Vec<String>) -> Self {
        self.images = images;
        self
    }

    /// Set the history item ID to resume from.
    pub fn with_history_item_id(mut self, id: impl Into<String>) -> Self {
        self.history_item_id = Some(id.into());
        self
    }

    /// Set the base storage path for task persistence.
    pub fn with_storage_path(mut self, path: impl Into<String>) -> Self {
        self.storage_path = Some(path.into());
        self
    }
}

// ---------------------------------------------------------------------------
// TaskResult
// ---------------------------------------------------------------------------

/// The result of a completed task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub status: TaskState,
    pub final_message: Option<String>,
    pub token_usage: TokenUsage,
    pub iterations: usize,
    pub tool_usage: HashMap<String, usize>,
}

impl TaskResult {
    /// Create a new task result.
    pub fn new(task_id: impl Into<String>, status: TaskState) -> Self {
        Self {
            task_id: task_id.into(),
            status,
            final_message: None,
            token_usage: TokenUsage::default(),
            iterations: 0,
            tool_usage: HashMap::new(),
        }
    }

    /// Record a tool usage.
    pub fn record_tool_usage(&mut self, tool_name: &str) {
        *self.tool_usage.entry(tool_name.to_string()).or_insert(0) += 1;
    }
}

// ---------------------------------------------------------------------------
// StreamingState
// ---------------------------------------------------------------------------

/// Streaming state for the current API request.
///
/// Tracks the state of the current streaming response from the API,
/// including content parsing, tool call processing, and checkpoint status.
///
/// Source: `src/core/task/Task.ts` — streaming-related properties:
/// `isStreaming`, `isWaitingForFirstChunk`, `currentStreamingContentIndex`,
/// `didCompleteReadingStream`, `assistantMessageSavedToHistory`, etc.
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Whether we are currently streaming an API response.
    pub is_streaming: bool,
    /// Whether we are waiting for the first chunk of the response.
    pub is_waiting_for_first_chunk: bool,
    /// Current content index within the streaming response.
    pub current_streaming_content_index: usize,
    /// Whether a checkpoint was done for the current streaming session.
    pub current_streaming_did_checkpoint: bool,
    /// Whether the stream has been fully read.
    pub did_complete_reading_stream: bool,
    /// Whether the assistant message has been saved to API conversation history.
    ///
    /// This is critical for parallel tool calling: tools should NOT execute until
    /// the assistant message is saved.
    pub assistant_message_saved_to_history: bool,
    /// Whether a tool was rejected in the current streaming session.
    pub did_reject_tool: bool,
    /// Whether a tool was already used in the current streaming session.
    pub did_already_use_tool: bool,
    /// Whether a tool failed in the current turn.
    pub did_tool_fail_in_current_turn: bool,
    /// Whether the present-assistant-message handler is locked.
    pub present_assistant_message_locked: bool,
    /// Whether there are pending updates for the assistant message handler.
    pub present_assistant_message_has_pending_updates: bool,
    /// Whether the stream finished aborting.
    ///
    /// Source: `src/core/task/Task.ts` — `didFinishAbortingStream`
    pub did_finish_aborting_stream: bool,
}

impl StreamingState {
    /// Create a new default streaming state.
    pub fn new() -> Self {
        Self {
            is_streaming: false,
            is_waiting_for_first_chunk: false,
            current_streaming_content_index: 0,
            current_streaming_did_checkpoint: false,
            did_complete_reading_stream: false,
            assistant_message_saved_to_history: false,
            did_reject_tool: false,
            did_already_use_tool: false,
            did_tool_fail_in_current_turn: false,
            present_assistant_message_locked: false,
            present_assistant_message_has_pending_updates: false,
            did_finish_aborting_stream: false,
        }
    }

    /// Reset all streaming state for a new API request.
    ///
    /// Source: `src/core/task/Task.ts` — streaming state reset block in
    /// `recursivelyMakeClineRequests`
    pub fn reset_for_new_request(&mut self) {
        self.current_streaming_content_index = 0;
        self.current_streaming_did_checkpoint = false;
        self.did_complete_reading_stream = false;
        self.assistant_message_saved_to_history = false;
        self.did_reject_tool = false;
        self.did_already_use_tool = false;
        self.did_tool_fail_in_current_turn = false;
        self.present_assistant_message_locked = false;
        self.present_assistant_message_has_pending_updates = false;
        self.did_finish_aborting_stream = false;
    }
}

impl Default for StreamingState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TaskError
// ---------------------------------------------------------------------------

/// Errors that can occur during task execution.
#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    /// Invalid state transition.
    #[error("invalid state transition from {from} to {to}")]
    InvalidTransition { from: TaskState, to: TaskState },

    /// Task has been cancelled.
    #[error("task cancelled")]
    Cancelled,

    /// Maximum iterations exceeded.
    #[error("maximum iterations exceeded: {0}")]
    MaxIterationsExceeded(usize),

    /// Maximum consecutive mistakes exceeded.
    #[error("maximum consecutive mistakes exceeded: {0}")]
    MaxMistakesExceeded(usize),

    /// A general task error.
    #[error("task error: {0}")]
    General(String),

    /// An I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A serialization error.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A persistence error.
    #[error("persistence error: {0}")]
    Persistence(#[from] roo_task_persistence::TaskPersistenceError),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- TaskState tests ---

    #[test]
    fn test_task_state_default() {
        assert_eq!(TaskState::default(), TaskState::Idle);
    }

    #[test]
    fn test_task_state_display() {
        assert_eq!(format!("{}", TaskState::Idle), "idle");
        assert_eq!(format!("{}", TaskState::Running), "running");
        assert_eq!(format!("{}", TaskState::Paused), "paused");
        assert_eq!(format!("{}", TaskState::Completed), "completed");
        assert_eq!(format!("{}", TaskState::Aborted), "aborted");
        assert_eq!(format!("{}", TaskState::Delegated), "delegated");
    }

    #[test]
    fn test_task_state_is_terminal() {
        assert!(!TaskState::Idle.is_terminal());
        assert!(!TaskState::Running.is_terminal());
        assert!(!TaskState::Paused.is_terminal());
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Aborted.is_terminal());
        // Delegated is NOT terminal — can resume after delegation
        assert!(!TaskState::Delegated.is_terminal());
    }

    #[test]
    fn test_task_state_is_active() {
        assert!(!TaskState::Idle.is_active());
        assert!(TaskState::Running.is_active());
        assert!(TaskState::Paused.is_active());
        assert!(!TaskState::Completed.is_active());
        assert!(!TaskState::Aborted.is_active());
        assert!(!TaskState::Delegated.is_active());
    }

    #[test]
    fn test_valid_transitions_from_idle() {
        assert!(TaskState::Idle.can_transition_to(&TaskState::Running));
        assert!(!TaskState::Idle.can_transition_to(&TaskState::Paused));
        assert!(!TaskState::Idle.can_transition_to(&TaskState::Completed));
        assert!(!TaskState::Idle.can_transition_to(&TaskState::Aborted));
        assert!(!TaskState::Idle.can_transition_to(&TaskState::Idle));
    }

    #[test]
    fn test_valid_transitions_from_running() {
        assert!(TaskState::Running.can_transition_to(&TaskState::Paused));
        assert!(TaskState::Running.can_transition_to(&TaskState::Completed));
        assert!(TaskState::Running.can_transition_to(&TaskState::Aborted));
        assert!(TaskState::Running.can_transition_to(&TaskState::Delegated));
        assert!(!TaskState::Running.can_transition_to(&TaskState::Idle));
        assert!(!TaskState::Running.can_transition_to(&TaskState::Running));
    }

    #[test]
    fn test_valid_transitions_from_paused() {
        assert!(TaskState::Paused.can_transition_to(&TaskState::Running));
        assert!(TaskState::Paused.can_transition_to(&TaskState::Aborted));
        assert!(!TaskState::Paused.can_transition_to(&TaskState::Idle));
        assert!(!TaskState::Paused.can_transition_to(&TaskState::Paused));
        assert!(!TaskState::Paused.can_transition_to(&TaskState::Completed));
    }

    #[test]
    fn test_no_transitions_from_terminal_states() {
        // Completed and Aborted are truly terminal — no transitions allowed
        for terminal in [TaskState::Completed, TaskState::Aborted] {
            for target in [
                TaskState::Idle,
                TaskState::Running,
                TaskState::Paused,
                TaskState::Completed,
                TaskState::Aborted,
                TaskState::Delegated,
            ] {
                assert!(!terminal.can_transition_to(&target), "{terminal:?} should not transition to {target:?}");
            }
        }
        // Delegated can transition back to Running (resumeAfterDelegation)
        assert!(TaskState::Delegated.can_transition_to(&TaskState::Running));
        assert!(!TaskState::Delegated.can_transition_to(&TaskState::Idle));
        assert!(!TaskState::Delegated.can_transition_to(&TaskState::Paused));
        assert!(!TaskState::Delegated.can_transition_to(&TaskState::Completed));
        assert!(!TaskState::Delegated.can_transition_to(&TaskState::Aborted));
    }

    #[test]
    fn test_task_state_serde_roundtrip() {
        for state in [
            TaskState::Idle,
            TaskState::Running,
            TaskState::Paused,
            TaskState::Completed,
            TaskState::Aborted,
            TaskState::Delegated,
        ] {
            let json = serde_json::to_string(&state).unwrap();
            let back: TaskState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
        }
    }

    // --- TaskConfig tests ---

    #[test]
    fn test_task_config_new() {
        let config = TaskConfig::new("task-1", "/tmp/work");
        assert_eq!(config.task_id, "task-1");
        assert_eq!(config.cwd, "/tmp/work");
        assert_eq!(config.mode, "code");
        assert!(config.api_config_name.is_none());
        assert!(!config.auto_approval);
        assert!(config.max_iterations.is_none());
        assert!(config.enable_checkpoints);
        assert_eq!(config.checkpoint_timeout, DEFAULT_CHECKPOINT_TIMEOUT_SECONDS);
        assert_eq!(config.consecutive_mistake_limit, crate::config::DEFAULT_MAX_MISTAKES);
        assert!(config.task_text.is_none());
        assert!(config.images.is_empty());
        assert!(config.history_item_id.is_none());
    }

    #[test]
    fn test_task_config_builder() {
        let config = TaskConfig::new("task-2", "/tmp/work")
            .with_mode("architect")
            .with_api_config("gpt4")
            .with_workspace("/tmp/ws")
            .with_max_iterations(100)
            .with_auto_approval(true)
            .with_root_task_id("root-1")
            .with_parent_task_id("parent-1")
            .with_task_number(3)
            .with_checkpoints(false)
            .with_checkpoint_timeout(60)
            .with_consecutive_mistake_limit(5)
            .with_task_text("Fix the bug")
            .with_images(vec!["img1.png".to_string()])
            .with_history_item_id("hist-1");

        assert_eq!(config.task_id, "task-2");
        assert_eq!(config.mode, "architect");
        assert_eq!(config.api_config_name, Some("gpt4".to_string()));
        assert_eq!(config.workspace, "/tmp/ws");
        assert_eq!(config.max_iterations, Some(100));
        assert!(config.auto_approval);
        assert_eq!(config.root_task_id, Some("root-1".to_string()));
        assert_eq!(config.parent_task_id, Some("parent-1".to_string()));
        assert_eq!(config.task_number, 3);
        assert!(!config.enable_checkpoints);
        assert_eq!(config.checkpoint_timeout, 60);
        assert_eq!(config.consecutive_mistake_limit, 5);
        assert_eq!(config.task_text, Some("Fix the bug".to_string()));
        assert_eq!(config.images, vec!["img1.png".to_string()]);
        assert_eq!(config.history_item_id, Some("hist-1".to_string()));
    }

    #[test]
    fn test_task_config_serialization() {
        let config = TaskConfig::new("task-3", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(50);

        let json = serde_json::to_string(&config).unwrap();
        let back: TaskConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "task-3");
        assert_eq!(back.max_iterations, Some(50));
    }

    // --- TaskResult tests ---

    #[test]
    fn test_task_result_new() {
        let result = TaskResult::new("task-1", TaskState::Completed);
        assert_eq!(result.task_id, "task-1");
        assert_eq!(result.status, TaskState::Completed);
        assert!(result.final_message.is_none());
        assert_eq!(result.iterations, 0);
        assert!(result.tool_usage.is_empty());
    }

    #[test]
    fn test_task_result_record_tool_usage() {
        let mut result = TaskResult::new("task-1", TaskState::Completed);
        result.record_tool_usage("read_file");
        result.record_tool_usage("read_file");
        result.record_tool_usage("write_to_file");

        assert_eq!(result.tool_usage["read_file"], 2);
        assert_eq!(result.tool_usage["write_to_file"], 1);
    }

    #[test]
    fn test_task_result_serialization() {
        let mut result = TaskResult::new("task-1", TaskState::Completed);
        result.final_message = Some("Done!".to_string());
        result.iterations = 10;
        result.record_tool_usage("read_file");

        let json = serde_json::to_string(&result).unwrap();
        let back: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "task-1");
        assert_eq!(back.final_message, Some("Done!".to_string()));
        assert_eq!(back.iterations, 10);
        assert_eq!(back.tool_usage["read_file"], 1);
    }

    // --- StreamingState tests ---

    #[test]
    fn test_streaming_state_new() {
        let state = StreamingState::new();
        assert!(!state.is_streaming);
        assert!(!state.is_waiting_for_first_chunk);
        assert_eq!(state.current_streaming_content_index, 0);
        assert!(!state.current_streaming_did_checkpoint);
        assert!(!state.did_complete_reading_stream);
        assert!(!state.assistant_message_saved_to_history);
        assert!(!state.did_reject_tool);
        assert!(!state.did_already_use_tool);
        assert!(!state.did_tool_fail_in_current_turn);
        assert!(!state.present_assistant_message_locked);
        assert!(!state.present_assistant_message_has_pending_updates);
        assert!(!state.did_finish_aborting_stream);
    }

    #[test]
    fn test_streaming_state_default() {
        assert_eq!(StreamingState::default().is_streaming, false);
    }

    #[test]
    fn test_streaming_state_reset_for_new_request() {
        let mut state = StreamingState::new();
        state.is_streaming = true;
        state.current_streaming_content_index = 5;
        state.assistant_message_saved_to_history = true;
        state.did_tool_fail_in_current_turn = true;
        state.did_finish_aborting_stream = true;

        state.reset_for_new_request();

        // These should be reset
        assert_eq!(state.current_streaming_content_index, 0);
        assert!(!state.current_streaming_did_checkpoint);
        assert!(!state.did_complete_reading_stream);
        assert!(!state.assistant_message_saved_to_history);
        assert!(!state.did_reject_tool);
        assert!(!state.did_already_use_tool);
        assert!(!state.did_tool_fail_in_current_turn);
        assert!(!state.present_assistant_message_locked);
        assert!(!state.present_assistant_message_has_pending_updates);
        assert!(!state.did_finish_aborting_stream);

        // is_streaming should NOT be reset by reset_for_new_request
        assert!(state.is_streaming);
    }

    // --- TaskError tests ---

    #[test]
    fn test_task_error_invalid_transition() {
        let err = TaskError::InvalidTransition {
            from: TaskState::Idle,
            to: TaskState::Completed,
        };
        assert!(err.to_string().contains("invalid state transition"));
    }

    #[test]
    fn test_task_error_cancelled() {
        let err = TaskError::Cancelled;
        assert_eq!(err.to_string(), "task cancelled");
    }

    #[test]
    fn test_task_error_max_iterations() {
        let err = TaskError::MaxIterationsExceeded(100);
        assert!(err.to_string().contains("100"));
    }

    // --- Constants tests ---

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_CHECKPOINT_TIMEOUT_SECONDS, 30);
        assert_eq!(MAX_CHECKPOINT_TIMEOUT_SECONDS, 600);
        assert_eq!(MIN_CHECKPOINT_TIMEOUT_SECONDS, 1);
        assert_eq!(MAX_EXPONENTIAL_BACKOFF_SECONDS, 600);
        assert_eq!(FORCED_CONTEXT_REDUCTION_PERCENT, 75);
        assert_eq!(MAX_CONTEXT_WINDOW_RETRIES, 3);
    }
}
