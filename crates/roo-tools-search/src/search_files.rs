//! search_files tool implementation.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::SearchFilesParams;

/// Validate search_files parameters.
pub fn validate_search_files_params(params: &SearchFilesParams) -> Result<(), SearchToolError> {
    if params.path.trim().is_empty() {
        return Err(SearchToolError::Validation(
            "path must not be empty".to_string(),
        ));
    }

    if params.regex.trim().is_empty() {
        return Err(SearchToolError::Validation(
            "regex pattern must not be empty".to_string(),
        ));
    }

    // Validate regex
    validate_regex(&params.regex)?;

    // Validate file pattern if provided
    if let Some(ref pattern) = params.file_pattern {
        if !pattern.is_empty() {
            validate_file_pattern(pattern)?;
        }
    }

    Ok(())
}

/// Check if a file path matches a glob pattern.
///
/// Simple glob matching: supports `*` (any chars) and `**` (recursive).
pub fn matches_file_pattern(file_path: &str, pattern: &str) -> bool {
    if pattern.is_empty() || pattern == "*" {
        return true;
    }

    // Simple glob matching
    let pattern = pattern.replace("**", "⟶"); // Temporarily replace **
    let parts: Vec<&str> = pattern.split('*').collect();

    let mut remaining = file_path;
    for (i, part) in parts.iter().enumerate() {
        let part = part.replace("⟶", "");
        if part.is_empty() {
            continue;
        }
        if let Some(pos) = remaining.find(&part) {
            remaining = &remaining[pos + part.len()..];
        } else if i == 0 {
            return remaining.starts_with(part.as_str());
        } else {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_path() {
        let params = SearchFilesParams {
            path: "".to_string(),
            regex: "pattern".to_string(),
            file_pattern: None,
        };
        assert!(validate_search_files_params(&params).is_err());
    }

    #[test]
    fn test_validate_empty_regex() {
        let params = SearchFilesParams {
            path: "src".to_string(),
            regex: "".to_string(),
            file_pattern: None,
        };
        assert!(validate_search_files_params(&params).is_err());
    }

    #[test]
    fn test_validate_invalid_regex() {
        let params = SearchFilesParams {
            path: "src".to_string(),
            regex: "[invalid".to_string(),
            file_pattern: None,
        };
        assert!(validate_search_files_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid() {
        let params = SearchFilesParams {
            path: "src".to_string(),
            regex: r"fn\s+\w+".to_string(),
            file_pattern: Some("*.rs".to_string()),
        };
        assert!(validate_search_files_params(&params).is_ok());
    }

    #[test]
    fn test_validate_no_file_pattern() {
        let params = SearchFilesParams {
            path: "src".to_string(),
            regex: "pattern".to_string(),
            file_pattern: None,
        };
        assert!(validate_search_files_params(&params).is_ok());
    }

    #[test]
    fn test_matches_file_pattern_rs() {
        assert!(matches_file_pattern("src/main.rs", "*.rs"));
    }

    #[test]
    fn test_matches_file_pattern_any() {
        assert!(matches_file_pattern("any/path.txt", "*"));
    }

    #[test]
    fn test_matches_file_pattern_no_match() {
        assert!(!matches_file_pattern("main.py", "*.rs"));
    }

    #[test]
    fn test_matches_file_pattern_empty() {
        assert!(matches_file_pattern("anything", ""));
    }
}
