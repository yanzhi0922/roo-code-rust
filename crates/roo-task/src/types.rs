//! Task engine type definitions.
//!
//! Defines TaskState, TaskConfig, TaskResult, TaskEvent, TaskError,
//! and related types for the task engine.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use roo_types::message::TokenUsage;

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
            TaskState::Completed | TaskState::Aborted | TaskState::Delegated => false,
        }
    }

    /// Returns `true` if this state is terminal (no further transitions).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Aborted | TaskState::Delegated
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub task_id: String,
    pub root_task_id: Option<String>,
    pub parent_task_id: Option<String>,
    pub task_number: usize,
    pub cwd: String,
    pub mode: String,
    pub api_config_name: Option<String>,
    pub workspace: String,
    pub max_iterations: Option<usize>,
    pub auto_approval: bool,
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
        assert!(TaskState::Delegated.is_terminal());
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
        for terminal in [TaskState::Completed, TaskState::Aborted, TaskState::Delegated] {
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
            .with_task_number(3);

        assert_eq!(config.task_id, "task-2");
        assert_eq!(config.mode, "architect");
        assert_eq!(config.api_config_name, Some("gpt4".to_string()));
        assert_eq!(config.workspace, "/tmp/ws");
        assert_eq!(config.max_iterations, Some(100));
        assert!(config.auto_approval);
        assert_eq!(config.root_task_id, Some("root-1".to_string()));
        assert_eq!(config.parent_task_id, Some("parent-1".to_string()));
        assert_eq!(config.task_number, 3);
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
}
