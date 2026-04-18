//! execute_command tool implementation.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::ExecuteCommandParams;

/// Validate execute_command parameters.
pub fn validate_execute_command_params(params: &ExecuteCommandParams) -> Result<(), CommandToolError> {
    if params.command.trim().is_empty() {
        return Err(CommandToolError::InvalidCommand(
            "command must not be empty".to_string(),
        ));
    }

    Ok(())
}

/// Generate an artifact ID for a command execution.
pub fn generate_artifact_id(command: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    command.hash(&mut hasher);
    let hash = hasher.finish();
    format!("cmd-{hash:x}.txt")
}

/// Process command execution parameters and return a structured result.
///
/// Note: The actual command execution is handled by the task engine.
/// This function only validates parameters and prepares the result structure.
pub fn prepare_command_execution(
    params: &ExecuteCommandParams,
    timeout: Option<u64>,
) -> Result<PreparedCommand, CommandToolError> {
    validate_execute_command_params(params)?;
    let resolved_timeout = resolve_timeout(timeout)?;
    let artifact_id = generate_artifact_id(&params.command);

    Ok(PreparedCommand {
        command: params.command.clone(),
        timeout_secs: resolved_timeout,
        artifact_id,
    })
}

/// A prepared command ready for execution.
#[derive(Debug, Clone)]
pub struct PreparedCommand {
    pub command: String,
    pub timeout_secs: u64,
    pub artifact_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_command() {
        let params = ExecuteCommandParams {
            command: "".to_string(),
        };
        assert!(validate_execute_command_params(&params).is_err());
    }

    #[test]
    fn test_validate_whitespace_command() {
        let params = ExecuteCommandParams {
            command: "   ".to_string(),
        };
        assert!(validate_execute_command_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid_command() {
        let params = ExecuteCommandParams {
            command: "echo hello".to_string(),
        };
        assert!(validate_execute_command_params(&params).is_ok());
    }

    #[test]
    fn test_generate_artifact_id() {
        let id1 = generate_artifact_id("echo hello");
        let id2 = generate_artifact_id("echo hello");
        let id3 = generate_artifact_id("echo world");

        assert!(id1.starts_with("cmd-"));
        assert!(id1.ends_with(".txt"));
        assert_eq!(id1, id2); // Same command = same ID
        assert_ne!(id1, id3); // Different command = different ID
    }

    #[test]
    fn test_prepare_command() {
        let params = ExecuteCommandParams {
            command: "cargo build".to_string(),
        };
        let prepared = prepare_command_execution(&params, Some(60)).unwrap();
        assert_eq!(prepared.command, "cargo build");
        assert_eq!(prepared.timeout_secs, 60);
        assert!(!prepared.artifact_id.is_empty());
    }

    #[test]
    fn test_prepare_command_default_timeout() {
        let params = ExecuteCommandParams {
            command: "echo hi".to_string(),
        };
        let prepared = prepare_command_execution(&params, None).unwrap();
        assert_eq!(prepared.timeout_secs, DEFAULT_TIMEOUT_SECS);
    }
}
