//! write_to_file tool implementation.
//!
//! Handles content cleaning (markdown fence stripping), directory creation,
//! and detecting whether a file is new or being modified.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::WriteToFileParams;

/// Validate write_to_file parameters.
pub fn validate_write_to_file_params(params: &WriteToFileParams) -> Result<(), FsToolError> {
    if params.path.trim().is_empty() {
        return Err(FsToolError::Validation("path must not be empty".to_string()));
    }

    if params.path.contains("..") {
        return Err(FsToolError::InvalidPath(
            "path must not contain '..'".to_string(),
        ));
    }

    if params.content.is_empty() {
        return Err(FsToolError::Validation(
            "content must not be empty".to_string(),
        ));
    }

    Ok(())
}

/// Clean content for writing: strip markdown fences if present.
pub fn clean_write_content(content: &str) -> String {
    strip_markdown_fences(content)
}

/// Process a write_to_file operation.
pub fn process_write_to_file(
    params: &WriteToFileParams,
    cwd: &std::path::Path,
) -> Result<WriteResult, FsToolError> {
    validate_write_to_file_params(params)?;

    let file_path = resolve_path(&params.path, cwd)?;
    let is_new_file = !file_path.exists();

    // Create parent directories if needed
    create_directories_for_file(&file_path)?;

    // Clean content
    let cleaned_content = clean_write_content(&params.content);

    // Count lines
    let lines_written = cleaned_content.lines().count();

    // Write the file
    std::fs::write(&file_path, &cleaned_content)?;

    Ok(WriteResult {
        path: params.path.clone(),
        is_new_file,
        lines_written,
    })
}

/// Resolve a relative path against the current working directory.
fn resolve_path(path: &str, cwd: &std::path::Path) -> Result<std::path::PathBuf, FsToolError> {
    let p = std::path::Path::new(path);
    if p.is_absolute() {
        Ok(p.to_path_buf())
    } else {
        Ok(cwd.join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_path() {
        let params = WriteToFileParams {
            path: "".to_string(),
            content: "hello".to_string(),
        };
        assert!(validate_write_to_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_path_traversal() {
        let params = WriteToFileParams {
            path: "../etc/passwd".to_string(),
            content: "hello".to_string(),
        };
        assert!(validate_write_to_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_empty_content() {
        let params = WriteToFileParams {
            path: "test.txt".to_string(),
            content: "".to_string(),
        };
        assert!(validate_write_to_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid() {
        let params = WriteToFileParams {
            path: "test.txt".to_string(),
            content: "hello world".to_string(),
        };
        assert!(validate_write_to_file_params(&params).is_ok());
    }

    #[test]
    fn test_clean_content_no_fence() {
        let content = "fn main() { println!(\"hello\"); }";
        assert_eq!(clean_write_content(content), content);
    }

    #[test]
    fn test_clean_content_with_fence() {
        let content = "```rust\nfn main() {}\n```";
        assert_eq!(clean_write_content(content), "fn main() {}\n");
    }

    #[test]
    fn test_process_write_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");

        let params = WriteToFileParams {
            path: file_path.to_str().unwrap().to_string(),
            content: "hello\nworld".to_string(),
        };
        let result = process_write_to_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.is_new_file);
        assert_eq!(result.lines_written, 2);

        let written = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(written, "hello\nworld");
    }

    #[test]
    fn test_process_write_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");
        std::fs::write(&file_path, "old content").unwrap();

        let params = WriteToFileParams {
            path: file_path.to_str().unwrap().to_string(),
            content: "new content".to_string(),
        };
        let result = process_write_to_file(&params, std::path::Path::new(".")).unwrap();
        assert!(!result.is_new_file);

        let written = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(written, "new content");
    }

    #[test]
    fn test_process_write_creates_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("a").join("b").join("c").join("deep.txt");

        let params = WriteToFileParams {
            path: file_path.to_str().unwrap().to_string(),
            content: "deep content".to_string(),
        };
        let result = process_write_to_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.is_new_file);
        assert!(file_path.exists());
    }

    #[test]
    fn test_process_write_with_fence() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("fenced.txt");

        let params = WriteToFileParams {
            path: file_path.to_str().unwrap().to_string(),
            content: "```javascript\nconsole.log(\"hello\");\n```".to_string(),
        };
        process_write_to_file(&params, std::path::Path::new(".")).unwrap();

        let written = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(written, "console.log(\"hello\");\n");
    }
}
