//! Task configuration validation and defaults.
//!
//! Provides validation logic for [`TaskConfig`] and default configuration helpers.

use crate::types::{TaskConfig, TaskError};

/// Default maximum consecutive mistakes.
pub const DEFAULT_MAX_MISTAKES: usize = 3;

/// Default mode.
pub const DEFAULT_MODE: &str = "code";

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate a task configuration.
///
/// Returns `Ok(())` if the configuration is valid, `Err` otherwise.
pub fn validate_config(config: &TaskConfig) -> Result<(), TaskError> {
    if config.task_id.is_empty() {
        return Err(TaskError::General("task_id cannot be empty".to_string()));
    }

    if config.cwd.is_empty() {
        return Err(TaskError::General("cwd cannot be empty".to_string()));
    }

    if config.mode.is_empty() {
        return Err(TaskError::General("mode cannot be empty".to_string()));
    }

    if let Some(max) = config.max_iterations {
        if max == 0 {
            return Err(TaskError::General("max_iterations must be greater than 0".to_string()));
        }
    }

    Ok(())
}

/// Create a default task config for the given task ID.
pub fn default_config(task_id: impl Into<String>) -> TaskConfig {
    TaskConfig::new(task_id, ".").with_mode(DEFAULT_MODE)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_config() {
        let config = TaskConfig::new("task-1", "/tmp/work").with_mode("code");
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_empty_task_id() {
        let config = TaskConfig::new("", "/tmp/work").with_mode("code");
        let result = validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("task_id"));
    }

    #[test]
    fn test_validate_empty_cwd() {
        let mut config = TaskConfig::new("task-1", "").with_mode("code");
        config.cwd = "".to_string();
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_empty_mode() {
        let mut config = TaskConfig::new("task-1", "/tmp/work");
        config.mode = "".to_string();
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_zero_max_iterations() {
        let config = TaskConfig::new("task-1", "/tmp/work").with_max_iterations(0);
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_none_max_iterations() {
        let config = TaskConfig::new("task-1", "/tmp/work");
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_default_config() {
        let config = default_config("task-1");
        assert_eq!(config.task_id, "task-1");
        assert_eq!(config.mode, DEFAULT_MODE);
        assert_eq!(config.cwd, ".");
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_MAX_MISTAKES, 3);
        assert_eq!(DEFAULT_MODE, "code");
    }
}
