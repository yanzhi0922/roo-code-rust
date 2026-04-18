//! list_files tool implementation.

use crate::types::*;
use roo_types::tool::ListFilesParams;

/// Validate list_files parameters.
pub fn validate_list_files_params(params: &ListFilesParams) -> Result<(), SearchToolError> {
    if params.path.trim().is_empty() {
        return Err(SearchToolError::Validation(
            "path must not be empty".to_string(),
        ));
    }

    Ok(())
}

/// Build a FileListResult from a list of paths.
///
/// Separates files from directories and applies truncation.
pub fn build_file_list_result(
    path: &str,
    recursive: bool,
    entries: Vec<String>,
    max_entries: usize,
) -> FileListResult {
    let mut files = Vec::new();
    let mut directories = Vec::new();

    for entry in &entries {
        // Simple heuristic: entries ending with '/' or '\' are directories
        if entry.ends_with('/') || entry.ends_with('\\') {
            directories.push(entry.clone());
        } else {
            files.push(entry.clone());
        }
    }

    let total_count = files.len() + directories.len();
    let truncated = total_count > max_entries;

    // Apply truncation
    if truncated {
        files.truncate(max_entries / 2);
        directories.truncate(max_entries / 2);
    }

    FileListResult {
        path: path.to_string(),
        recursive,
        files,
        directories,
        total_count,
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_path() {
        let params = ListFilesParams {
            path: "".to_string(),
            recursive: true,
        };
        assert!(validate_list_files_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid() {
        let params = ListFilesParams {
            path: "src".to_string(),
            recursive: true,
        };
        assert!(validate_list_files_params(&params).is_ok());
    }

    #[test]
    fn test_build_file_list_basic() {
        let entries = vec![
            "src/".to_string(),
            "main.rs".to_string(),
            "lib.rs".to_string(),
        ];
        let result = build_file_list_result("src", true, entries, 100);
        assert_eq!(result.directories.len(), 1);
        assert_eq!(result.files.len(), 2);
        assert!(!result.truncated);
    }

    #[test]
    fn test_build_file_list_truncated() {
        let entries: Vec<String> = (0..20)
            .flat_map(|i| vec![format!("dir{i}/"), format!("file{i}.rs")])
            .collect();
        let result = build_file_list_result(".", false, entries, 5);
        assert!(result.truncated);
        assert_eq!(result.total_count, 40);
    }

    #[test]
    fn test_build_file_list_empty() {
        let result = build_file_list_result("empty", true, vec![], 100);
        assert!(result.files.is_empty());
        assert!(result.directories.is_empty());
        assert!(!result.truncated);
    }
}
