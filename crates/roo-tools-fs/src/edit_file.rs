//! edit_file tool implementation.
//!
//! Performs literal string replacement in files. Corresponds to
//! `src/core/tools/EditFileTool.ts`.
//!
//! Supports:
//! - Modifying existing files (provide `old_string` and `new_string`)
//! - Creating new files (set `old_string` to empty string)
//! - Multiple replacements via `expected_replacements`

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::EditFileParams;

/// Validate edit_file parameters.
pub fn validate_edit_file_params(params: &EditFileParams) -> Result<(), FsToolError> {
    if params.file_path.trim().is_empty() {
        return Err(FsToolError::Validation("file_path must not be empty".to_string()));
    }

    if params.file_path.contains("..") {
        return Err(FsToolError::InvalidPath(
            "file_path must not contain '..'".to_string(),
        ));
    }

    // new_string must be provided (but can be empty to delete text)
    // old_string can be empty only for new file creation
    Ok(())
}

/// Count non-overlapping occurrences of a substring.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut pos = 0;
    while let Some(idx) = haystack[pos..].find(needle) {
        count += 1;
        pos += idx + needle.len();
    }
    count
}

/// Safely replace all occurrences of a literal string.
/// Handles the case where the replacement string contains `$` which
/// would be interpreted specially by regex-like replace functions.
fn safe_literal_replace(haystack: &str, old: &str, new: &str) -> String {
    if old.is_empty() || !haystack.contains(old) {
        return haystack.to_string();
    }
    haystack.replace(old, new)
}

/// Detect the dominant line ending in content.
fn detect_line_ending(content: &str) -> &str {
    if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

/// Normalize line endings to LF.
fn normalize_to_lf(content: &str) -> String {
    content.replace("\r\n", "\n")
}

/// Restore line endings from LF to the detected format.
fn restore_line_ending(content_lf: &str, eol: &str) -> String {
    if eol == "\n" {
        content_lf.to_string()
    } else {
        content_lf.replace('\n', "\r\n")
    }
}

/// Process an edit_file operation.
///
/// Implements the logic from `EditFileTool.ts`:
/// 1. If `old_string` is empty → create new file
/// 2. If file doesn't exist and `old_string` is not empty → error
/// 3. If file exists and `old_string` is empty → error (file already exists)
/// 4. Otherwise → perform literal string replacement
pub fn process_edit_file(
    params: &EditFileParams,
    cwd: &std::path::Path,
) -> Result<EditFileResult, FsToolError> {
    validate_edit_file_params(params)?;

    // Resolve file path (can be absolute or relative)
    let file_path = if std::path::Path::new(&params.file_path).is_absolute() {
        std::path::PathBuf::from(&params.file_path)
    } else {
        cwd.join(&params.file_path)
    };

    let file_exists = file_path.exists();
    let old_string = &params.old_string;
    let new_string = &params.new_string;
    let expected_replacements = params.expected_replacements.unwrap_or(1).max(1) as usize;

    // Case: Creating a new file (old_string is empty)
    if old_string.is_empty() {
        if file_exists {
            return Err(FsToolError::Validation(format!(
                "File already exists: {}. To modify an existing file, provide a non-empty old_string.",
                params.file_path
            )));
        }

        // Create parent directories
        create_directories_for_file(&file_path)?;

        // Write the new file
        std::fs::write(&file_path, new_string)?;

        return Ok(EditFileResult {
            path: params.file_path.clone(),
            success: true,
            message: Some(format!(
                "Created new file: {} ({} lines)",
                params.file_path,
                new_string.lines().count()
            )),
        });
    }

    // Case: Modifying an existing file
    if !file_exists {
        return Err(FsToolError::FileNotFound(format!(
            "File does not exist: {}. If you intended to create a new file, set old_string to an empty string.",
            params.file_path
        )));
    }

    // Read the current content
    let current_content = std::fs::read_to_string(&file_path)?;

    // Detect and normalize line endings
    let eol = detect_line_ending(&current_content);
    let current_lf = normalize_to_lf(&current_content);
    let old_lf = normalize_to_lf(old_string);
    let new_lf = normalize_to_lf(new_string);

    // Validate that old_string and new_string are different
    if old_lf == new_lf {
        return Err(FsToolError::Validation(
            "old_string and new_string are identical after normalizing line endings. No changes to apply.".to_string(),
        ));
    }

    // Count occurrences
    let actual_count = count_occurrences(&current_lf, &old_lf);

    if actual_count == 0 {
        return Err(FsToolError::Validation(format!(
            "old_string not found in file: {}. Please verify the exact content using read_file.",
            params.file_path
        )));
    }

    // Validate against expected count
    if actual_count != expected_replacements {
        return Err(FsToolError::Validation(format!(
            "Expected {} replacement(s) but found {} occurrence(s) of old_string in file: {}. \
             If you want to replace all occurrences, set expected_replacements to {}.",
            expected_replacements, actual_count, params.file_path, actual_count
        )));
    }

    // Perform the replacement
    let modified_lf = if expected_replacements == 1 {
        // Single replacement: use find + replace_range for precision
        if let Some(pos) = current_lf.find(&old_lf) {
            let mut result = current_lf.clone();
            result.replace_range(pos..pos + old_lf.len(), &new_lf);
            result
        } else {
            // Should not happen since we checked count above
            current_lf
        }
    } else {
        // Multiple replacements
        safe_literal_replace(&current_lf, &old_lf, &new_lf)
    };

    // Restore original line endings
    let modified = restore_line_ending(&modified_lf, eol);

    // Write the modified content
    std::fs::write(&file_path, &modified)?;

    Ok(EditFileResult {
        path: params.file_path.clone(),
        success: true,
        message: Some(format!(
            "Successfully applied {} replacement(s) in {}",
            actual_count, params.file_path
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_path() {
        let params = EditFileParams {
            file_path: "".to_string(),
            old_string: "old".to_string(),
            new_string: "new".to_string(),
            expected_replacements: None,
        };
        assert!(validate_edit_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_path_traversal() {
        let params = EditFileParams {
            file_path: "../secret".to_string(),
            old_string: "old".to_string(),
            new_string: "new".to_string(),
            expected_replacements: None,
        };
        assert!(validate_edit_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid() {
        let params = EditFileParams {
            file_path: "test.txt".to_string(),
            old_string: "old".to_string(),
            new_string: "new".to_string(),
            expected_replacements: None,
        };
        assert!(validate_edit_file_params(&params).is_ok());
    }

    #[test]
    fn test_process_edit_file_not_found() {
        let params = EditFileParams {
            file_path: "nonexistent.txt".to_string(),
            old_string: "foo".to_string(),
            new_string: "bar".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new("."));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_edit_file_create_new() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "".to_string(),
            new_string: "hello world\n".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.success);
        assert!(result.message.unwrap().contains("Created new file"));

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "hello world\n");
    }

    #[test]
    fn test_process_edit_file_create_already_exists() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");
        std::fs::write(&file_path, "existing content").unwrap();

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "".to_string(),
            new_string: "new content".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new("."));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_edit_file_replace_success() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world\nfoo bar\n").unwrap();

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "hello world".to_string(),
            new_string: "HELLO WORLD".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.success);

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("HELLO WORLD"));
        assert!(!content.contains("hello world"));
    }

    #[test]
    fn test_process_edit_file_not_found_string() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world\n").unwrap();

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "nonexistent string".to_string(),
            new_string: "replacement".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new("."));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_edit_file_wrong_count() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "foo foo foo\n").unwrap();

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "foo".to_string(),
            new_string: "bar".to_string(),
            expected_replacements: Some(1),
        };
        let result = process_edit_file(&params, std::path::Path::new("."));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_edit_file_multiple_replacements() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "foo and foo and foo\n").unwrap();

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "foo".to_string(),
            new_string: "bar".to_string(),
            expected_replacements: Some(3),
        };
        let result = process_edit_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.success);

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "bar and bar and bar\n");
    }

    #[test]
    fn test_process_edit_file_identical_strings() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello\n").unwrap();

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "hello".to_string(),
            new_string: "hello".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new("."));
        assert!(result.is_err());
    }

    #[test]
    fn test_count_occurrences() {
        assert_eq!(count_occurrences("foo foo foo", "foo"), 3);
        assert_eq!(count_occurrences("hello world", "foo"), 0);
        assert_eq!(count_occurrences("aaa", "aa"), 1); // non-overlapping
        assert_eq!(count_occurrences("any text", ""), 0);
    }

    #[test]
    fn test_process_edit_file_crlf_handling() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("crlf.txt");
        std::fs::write(&file_path, "hello\r\nworld\r\n").unwrap();

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "hello\nworld".to_string(),
            new_string: "HELLO\nWORLD".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.success);

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("HELLO\r\nWORLD"));
    }

    #[test]
    fn test_process_edit_file_creates_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("a").join("b").join("deep.txt");

        let params = EditFileParams {
            file_path: file_path.to_str().unwrap().to_string(),
            old_string: "".to_string(),
            new_string: "deep content".to_string(),
            expected_replacements: None,
        };
        let result = process_edit_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.success);
        assert!(file_path.exists());
    }
}
