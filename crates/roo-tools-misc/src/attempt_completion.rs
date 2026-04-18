//! attempt_completion tool implementation.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::AttemptCompletionParams;

/// Validate attempt_completion parameters.
pub fn validate_attempt_completion_params(params: &AttemptCompletionParams) -> Result<(), MiscToolError> {
    validate_completion_result(&params.result)
}

/// Process an attempt_completion request.
pub fn process_attempt_completion(params: &AttemptCompletionParams) -> Result<CompletionResult, MiscToolError> {
    validate_attempt_completion_params(params)?;

    Ok(CompletionResult {
        result: params.result.clone(),
        has_command: params.command.is_some(),
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
        let result = process_attempt_completion(&params).unwrap();
        assert_eq!(result.result, "All good");
        assert!(!result.has_command);
    }

    #[test]
    fn test_process_completion_with_command() {
        let params = AttemptCompletionParams {
            result: "Deployed".to_string(),
            command: Some("npm start".to_string()),
        };
        let result = process_attempt_completion(&params).unwrap();
        assert!(result.has_command);
    }
}
