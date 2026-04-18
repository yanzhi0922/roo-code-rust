//! edit_file tool implementation.
//!
//! A simpler alternative to apply_diff that validates edit parameters.

use crate::types::*;
use roo_types::tool::EditFileParams;

/// Validate edit_file parameters.
pub fn validate_edit_file_params(params: &EditFileParams) -> Result<(), FsToolError> {
    if params.path.trim().is_empty() {
        return Err(FsToolError::Validation("path must not be empty".to_string()));
    }

    if params.path.contains("..") {
        return Err(FsToolError::InvalidPath(
            "path must not contain '..'".to_string(),
        ));
    }

    if params.diff.trim().is_empty() {
        return Err(FsToolError::Validation(
            "diff content must not be empty".to_string(),
        ));
    }

    Ok(())
}

/// Process an edit_file operation.
///
/// This is a simplified version that validates parameters and
/// returns the result. The actual diff application is delegated
/// to the apply_diff module.
pub fn process_edit_file(
    params: &EditFileParams,
    cwd: &std::path::Path,
) -> Result<EditFileResult, FsToolError> {
    validate_edit_file_params(params)?;

    let file_path = if std::path::Path::new(&params.path).is_absolute() {
        std::path::PathBuf::from(&params.path)
    } else {
        cwd.join(&params.path)
    };

    if !file_path.exists() {
        return Err(FsToolError::FileNotFound(params.path.clone()));
    }

    // Parse and apply diff blocks using apply_diff logic
    let original_content = std::fs::read_to_string(&file_path)?;

    let blocks = crate::apply_diff::parse_diff_blocks(&params.diff)?;

    if blocks.is_empty() {
        return Ok(EditFileResult {
            path: params.path.clone(),
            success: true,
            message: Some("No diff blocks found to apply".to_string()),
        });
    }

    let result = crate::apply_diff::apply_diff_blocks(&original_content, &blocks)?;

    if result.blocks_applied == 0 {
        return Ok(EditFileResult {
            path: params.path.clone(),
            success: false,
            message: Some(format!(
                "None of the {} diff blocks could be applied",
                blocks.len()
            )),
        });
    }

    // Write the modified content
    // Re-apply to get the final content
    let mut content = original_content.clone();
    for (search, replace) in &blocks {
        if let Some(pos) = content.find(search.as_str()) {
            content.replace_range(pos..pos + search.len(), replace);
        }
    }

    std::fs::write(&file_path, &content)?;

    let msg = if result.warnings.is_empty() {
        format!(
            "Successfully applied {} diff block(s)",
            result.blocks_applied
        )
    } else {
        format!(
            "Applied {} of {} blocks. Warnings: {}",
            result.blocks_applied,
            blocks.len(),
            result.warnings.join("; ")
        )
    };

    Ok(EditFileResult {
        path: params.path.clone(),
        success: true,
        message: Some(msg),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_path() {
        let params = EditFileParams {
            path: "".to_string(),
            diff: "some diff".to_string(),
        };
        assert!(validate_edit_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_path_traversal() {
        let params = EditFileParams {
            path: "../secret".to_string(),
            diff: "some diff".to_string(),
        };
        assert!(validate_edit_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_empty_diff() {
        let params = EditFileParams {
            path: "test.txt".to_string(),
            diff: "".to_string(),
        };
        assert!(validate_edit_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid() {
        let params = EditFileParams {
            path: "test.txt".to_string(),
            diff: "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE".to_string(),
        };
        assert!(validate_edit_file_params(&params).is_ok());
    }

    #[test]
    fn test_process_edit_file_not_found() {
        let params = EditFileParams {
            path: "nonexistent.txt".to_string(),
            diff: "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE".to_string(),
        };
        let result = process_edit_file(&params, std::path::Path::new("."));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_edit_file_success() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world\nfoo bar\n").unwrap();

        let params = EditFileParams {
            path: file_path.to_str().unwrap().to_string(),
            diff: "<<<<<<< SEARCH\nhello world\n=======\nHELLO WORLD\n>>>>>>> REPLACE".to_string(),
        };
        let result = process_edit_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.success);

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("HELLO WORLD"));
    }

    #[test]
    fn test_process_edit_file_no_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello\n").unwrap();

        // Valid params but no SEARCH markers in diff
        let params = EditFileParams {
            path: file_path.to_str().unwrap().to_string(),
            diff: "just some text".to_string(),
        };

        // parse_diff_blocks will return empty vec
        let result = process_edit_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.success);
        assert!(result.message.unwrap().contains("No diff blocks"));
    }
}
