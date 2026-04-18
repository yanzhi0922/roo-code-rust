//! skill tool implementation.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::SkillParams;

/// Validate skill parameters.
pub fn validate_skill_params(params: &SkillParams) -> Result<(), MiscToolError> {
    validate_skill_name(&params.skill)
}

/// Process a skill request.
pub fn process_skill(params: &SkillParams) -> Result<SkillResult, MiscToolError> {
    validate_skill_params(params)?;

    Ok(SkillResult {
        skill_name: params.skill.clone(),
        args: params.args.clone(),
        is_valid: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_skill_name() {
        let params = SkillParams {
            skill: "".to_string(),
            args: None,
        };
        assert!(validate_skill_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid_skill() {
        let params = SkillParams {
            skill: "my-skill".to_string(),
            args: None,
        };
        assert!(validate_skill_params(&params).is_ok());
    }

    #[test]
    fn test_process_skill() {
        let params = SkillParams {
            skill: "react-dev".to_string(),
            args: Some("create component".to_string()),
        };
        let result = process_skill(&params).unwrap();
        assert_eq!(result.skill_name, "react-dev");
        assert_eq!(result.args, Some("create component".to_string()));
        assert!(result.is_valid);
    }

    #[test]
    fn test_process_skill_no_args() {
        let params = SkillParams {
            skill: "flutter-dev".to_string(),
            args: None,
        };
        let result = process_skill(&params).unwrap();
        assert!(result.args.is_none());
    }
}
