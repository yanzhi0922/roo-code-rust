//! Types for miscellaneous tool results and errors.

use serde::{Deserialize, Serialize};

/// Result of an attempt_completion operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResult {
    pub result: String,
    pub has_command: bool,
}

/// Result of an ask_followup_question operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowupResult {
    pub question: String,
    pub suggestions: Vec<String>,
}

/// Result of a skill operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillResult {
    pub skill_name: String,
    pub args: Option<String>,
    pub is_valid: bool,
}

/// A parsed todo item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub status: TodoStatus,
    pub text: String,
}

/// Status of a todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl TodoStatus {
    /// Parse from markdown checkbox syntax.
    pub fn from_checkbox(checked: bool) -> Self {
        if checked {
            TodoStatus::Completed
        } else {
            TodoStatus::Pending
        }
    }

    /// Convert to markdown checkbox string.
    pub fn to_checkbox(&self) -> &'static str {
        match self {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[-]",
            TodoStatus::Completed => "[x]",
        }
    }
}

/// Error type for miscellaneous tool operations.
#[derive(Debug, thiserror::Error)]
pub enum MiscToolError {
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Invalid skill: {0}")]
    InvalidSkill(String),

    #[error("Parse error: {0}")]
    Parse(String),
}
