//! attempt_completion tool implementation.
//!
//! Aligned with TS `AttemptCompletionTool.ts`:
//! - Validates the `result` parameter is non-empty.
//! - Checks for incomplete todos and emits a warning (todo guard).
//! - Populates `attempt_completion_result` with text and images.
//! - Emits a telemetry event stub for task completion.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::AttemptCompletionParams;

/// Validate attempt_completion parameters.
pub fn validate_attempt_completion_params(params: &AttemptCompletionParams) -> Result<(), MiscToolError> {
    validate_completion_result(&params.result)
}

/// Check for incomplete todos and return a warning message if any are found.
///
/// In the TS source, `preventCompletionWithOpenTodos` can block completion entirely.
/// Here we emit a warning but do NOT block — the caller decides whether to proceed.
///
/// Returns `Some(warning_message)` if there are incomplete todos, `None` otherwise.
pub fn check_todo_guard(todos: &[TodoItem]) -> Option<String> {
    let incomplete: Vec<&TodoItem> = todos
        .iter()
        .filter(|t| t.status != TodoStatus::Completed)
        .collect();

    if incomplete.is_empty() {
        return None;
    }

    let pending = incomplete
        .iter()
        .filter(|t| t.status == TodoStatus::Pending)
        .count();
    let in_progress = incomplete
        .iter()
        .filter(|t| t.status == TodoStatus::InProgress)
        .count();

    Some(format!(
        "Warning: There are incomplete todo items ({} pending, {} in progress). \
         Consider completing all todos before finishing.",
        pending, in_progress
    ))
}

/// Emit a telemetry event for task completion.
///
/// In the TS source, this calls `TelemetryService.instance.captureTaskCompleted(task.taskId)`.
/// In Rust, this is a stub that records the event. When `roo-telemetry` is integrated,
/// this should forward to the telemetry service.
///
/// TODO: Integrate with `roo-telemetry` crate when available.
pub fn emit_task_completed_event(_task_id: &str) {
    // Stub: no-op. Replace with actual telemetry integration when roo-telemetry is available.
    // In production, this would call something like:
    //   TelemetryService::instance().capture_task_completed(task_id);
}

/// Process an attempt_completion request.
///
/// Validates parameters, checks the todo guard, and builds the result.
/// The `attempt_completion_result` field contains the text and optional images.
pub fn process_attempt_completion(
    params: &AttemptCompletionParams,
    todos: &[TodoItem],
) -> Result<CompletionResult, MiscToolError> {
    validate_attempt_completion_params(params)?;

    let todo_warning = check_todo_guard(todos);

    let attempt_completion_result = Some(CompletionResultData {
        text: params.result.clone(),
        images: Vec::new(),
    });

    Ok(CompletionResult {
        result: params.result.clone(),
        has_command: params.command.is_some(),
        attempt_completion_result,
        todo_warning,
    })
}

/// Process an attempt_completion request with images.
///
/// This variant allows passing images that were collected during the
/// completion interaction (e.g., from user feedback).
pub fn process_attempt_completion_with_images(
    params: &AttemptCompletionParams,
    todos: &[TodoItem],
    images: Vec<String>,
) -> Result<CompletionResult, MiscToolError> {
    validate_attempt_completion_params(params)?;

    let todo_warning = check_todo_guard(todos);

    let attempt_completion_result = Some(CompletionResultData {
        text: params.result.clone(),
        images,
    });

    Ok(CompletionResult {
        result: params.result.clone(),
        has_command: params.command.is_some(),
        attempt_completion_result,
        todo_warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_result() {
        let params = AttemptCompletionParams {
            result: "".to_string(),
            command: None,
        };
        assert!(validate_attempt_completion_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid_result() {
        let params = AttemptCompletionParams {
            result: "Task done!".to_string(),
            command: None,
        };
        assert!(validate_attempt_completion_params(&params).is_ok());
    }

    #[test]
    fn test_process_completion() {
        let params = AttemptCompletionParams {
            result: "All good".to_string(),
            command: None,
        };
        let result = process_attempt_completion(&params, &[]).unwrap();
        assert_eq!(result.result, "All good");
        assert!(!result.has_command);
        assert!(result.attempt_completion_result.is_some());
        assert!(result.todo_warning.is_none());
    }

    #[test]
    fn test_process_completion_with_command() {
        let params = AttemptCompletionParams {
            result: "Deployed".to_string(),
            command: Some("npm start".to_string()),
        };
        let result = process_attempt_completion(&params, &[]).unwrap();
        assert!(result.has_command);
    }

    // --- New tests for todo guard, result data, and telemetry ---

    #[test]
    fn test_todo_guard_no_todos() {
        let warning = check_todo_guard(&[]);
        assert!(warning.is_none());
    }

    #[test]
    fn test_todo_guard_all_complete() {
        let todos = vec![
            TodoItem {
                status: TodoStatus::Completed,
                text: "task 1".to_string(),
            },
            TodoItem {
                status: TodoStatus::Completed,
                text: "task 2".to_string(),
            },
        ];
        let warning = check_todo_guard(&todos);
        assert!(warning.is_none());
    }

    #[test]
    fn test_todo_guard_with_incomplete_todos() {
        let todos = vec![
            TodoItem {
                status: TodoStatus::Completed,
                text: "done task".to_string(),
            },
            TodoItem {
                status: TodoStatus::Pending,
                text: "pending task".to_string(),
            },
            TodoItem {
                status: TodoStatus::InProgress,
                text: "in-progress task".to_string(),
            },
        ];
        let warning = check_todo_guard(&todos);
        assert!(warning.is_some());
        let msg = warning.unwrap();
        assert!(msg.contains("1 pending"));
        assert!(msg.contains("1 in progress"));
        assert!(msg.contains("incomplete todo items"));
    }

    #[test]
    fn test_todo_guard_only_pending() {
        let todos = vec![TodoItem {
            status: TodoStatus::Pending,
            text: "not done".to_string(),
        }];
        let warning = check_todo_guard(&todos).unwrap();
        assert!(msg_contains(&warning, "1 pending"));
        assert!(msg_contains(&warning, "0 in progress"));
    }

    #[test]
    fn test_attempt_completion_result_data() {
        let params = AttemptCompletionParams {
            result: "Finished!".to_string(),
            command: None,
        };
        let result = process_attempt_completion(&params, &[]).unwrap();
        let data = result.attempt_completion_result.unwrap();
        assert_eq!(data.text, "Finished!");
        assert!(data.images.is_empty());
    }

    #[test]
    fn test_attempt_completion_with_images() {
        let params = AttemptCompletionParams {
            result: "See screenshot".to_string(),
            command: None,
        };
        let images = vec!["data:image/png;base64,abc".to_string()];
        let result =
            process_attempt_completion_with_images(&params, &[], images.clone()).unwrap();
        let data = result.attempt_completion_result.unwrap();
        assert_eq!(data.images, images);
    }

    #[test]
    fn test_attempt_completion_with_todos_and_images() {
        let params = AttemptCompletionParams {
            result: "Done with warnings".to_string(),
            command: Some("echo done".to_string()),
        };
        let todos = vec![TodoItem {
            status: TodoStatus::InProgress,
            text: "still working".to_string(),
        }];
        let images = vec!["data:image/png;base64,xyz".to_string()];
        let result =
            process_attempt_completion_with_images(&params, &todos, images).unwrap();

        assert!(result.has_command);
        assert!(result.todo_warning.is_some());
        assert!(result.todo_warning.as_ref().unwrap().contains("0 pending"));
        assert!(result.todo_warning.as_ref().unwrap().contains("1 in progress"));
        let data = result.attempt_completion_result.unwrap();
        assert_eq!(data.images.len(), 1);
    }

    #[test]
    fn test_emit_task_completed_event() {
        // Should not panic; just a stub that logs.
        emit_task_completed_event("test-task-123");
    }

    /// Helper to check if a string contains a substring.
    fn msg_contains(s: &str, sub: &str) -> bool {
        s.contains(sub)
    }
}
