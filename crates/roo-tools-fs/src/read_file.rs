//! read_file tool implementation.
//!
//! Supports slice mode (offset/limit) and provides line numbering,
//! binary detection, and long line truncation.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::ReadFileParams;

/// Validate read_file parameters.
pub fn validate_read_file_params(params: &ReadFileParams) -> Result<(), FsToolError> {
    if params.path.trim().is_empty() {
        return Err(FsToolError::Validation("path must not be empty".to_string()));
    }

    // Check for path traversal
    if params.path.contains("..") {
        return Err(FsToolError::InvalidPath(
            "path must not contain '..'".to_string(),
        ));
    }

    if let Some(offset) = params.offset {
        if offset == 0 {
            return Err(FsToolError::Validation(
                "offset must be >= 1 (1-based line number)".to_string(),
            ));
        }
    }

    if let Some(limit) = params.limit {
        if limit == 0 {
            return Err(FsToolError::Validation("limit must be >= 1".to_string()));
        }
    }

    Ok(())
}

/// Process a file read operation.
///
/// This function handles the core logic of reading a file:
/// 1. Reading raw bytes from disk
/// 2. Detecting binary content
/// 3. Decoding as UTF-8
/// 4. Applying offset/limit slicing
/// 5. Truncating long lines
/// 6. Adding line numbers
pub fn process_read_file(
    params: &ReadFileParams,
    cwd: &std::path::Path,
) -> Result<ReadResult, FsToolError> {
    validate_read_file_params(params)?;

    let file_path = resolve_path(&params.path, cwd)?;

    if !file_path.exists() {
        return Err(FsToolError::FileNotFound(params.path.clone()));
    }

    let metadata = std::fs::metadata(&file_path)?;
    if metadata.is_dir() {
        return Err(FsToolError::InvalidPath(format!(
            "{} is a directory, not a file",
            params.path
        )));
    }

    // Check file size
    if metadata.len() as usize > MAX_FILE_SIZE {
        return Err(FsToolError::ContentTooLarge(metadata.len() as usize, MAX_FILE_SIZE));
    }

    // Read raw bytes for binary detection
    let raw_data = std::fs::read(&file_path)?;

    if is_binary_content(&raw_data) {
        return Ok(ReadResult {
            content: format!(
                "(Binary file: {} bytes, not displaying)",
                raw_data.len()
            ),
            path: params.path.clone(),
            total_lines: 0,
            truncated: false,
            is_binary: true,
        });
    }

    // Decode as UTF-8
    let content = String::from_utf8_lossy(&raw_data).into_owned();

    build_read_result(content, &params.path, params.offset, params.limit)
}

/// Build a ReadResult from string content with optional offset/limit.
pub fn build_read_result(
    content: String,
    path: &str,
    offset: Option<u64>,
    limit: Option<u64>,
) -> Result<ReadResult, FsToolError> {
    let total_lines = content.lines().count();

    let (sliced_content, total) = if let Some(start) = offset {
        let max_lines = limit.unwrap_or(DEFAULT_READ_LIMIT as u64) as usize;
        slice_lines(&content, start as usize, max_lines)
    } else if let Some(max) = limit {
        slice_lines(&content, 1, max as usize)
    } else {
        (content.clone(), total_lines)
    };

    let truncated = total_lines > 0
        && (offset.is_some() || limit.is_some())
        && sliced_content.lines().count() < total_lines;

    // Truncate long lines
    let final_content = truncate_long_lines(&sliced_content, DEFAULT_MAX_LINE_LENGTH);

    Ok(ReadResult {
        content: final_content,
        path: path.to_string(),
        total_lines: total,
        truncated,
        is_binary: false,
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
        let params = ReadFileParams {
            path: "".to_string(),
            offset: None,
            limit: None,
        };
        assert!(validate_read_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_path_traversal() {
        let params = ReadFileParams {
            path: "../etc/passwd".to_string(),
            offset: None,
            limit: None,
        };
        assert!(validate_read_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_zero_offset() {
        let params = ReadFileParams {
            path: "test.txt".to_string(),
            offset: Some(0),
            limit: None,
        };
        assert!(validate_read_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_zero_limit() {
        let params = ReadFileParams {
            path: "test.txt".to_string(),
            offset: None,
            limit: Some(0),
        };
        assert!(validate_read_file_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid_params() {
        let params = ReadFileParams {
            path: "test.txt".to_string(),
            offset: Some(1),
            limit: Some(100),
        };
        assert!(validate_read_file_params(&params).is_ok());
    }

    #[test]
    fn test_validate_valid_no_optional() {
        let params = ReadFileParams {
            path: "test.txt".to_string(),
            offset: None,
            limit: None,
        };
        assert!(validate_read_file_params(&params).is_ok());
    }

    #[test]
    fn test_build_read_result_full() {
        let content = "line1\nline2\nline3".to_string();
        let result = build_read_result(content, "test.txt", None, None).unwrap();
        assert_eq!(result.total_lines, 3);
        assert!(!result.truncated);
        assert!(!result.is_binary);
    }

    #[test]
    fn test_build_read_result_with_offset() {
        let content = "line1\nline2\nline3\nline4\nline5".to_string();
        let result = build_read_result(content, "test.txt", Some(2), Some(2)).unwrap();
        assert_eq!(result.total_lines, 5);
        assert!(result.truncated);
        assert!(result.content.contains("line2"));
        assert!(result.content.contains("line3"));
    }

    #[test]
    fn test_build_read_result_with_limit() {
        let content = "line1\nline2\nline3".to_string();
        let result = build_read_result(content, "test.txt", None, Some(2)).unwrap();
        assert_eq!(result.total_lines, 3);
        assert!(result.truncated);
    }

    #[test]
    fn test_process_read_file_not_found() {
        let params = ReadFileParams {
            path: "nonexistent.txt".to_string(),
            offset: None,
            limit: None,
        };
        let cwd = std::env::current_dir().unwrap();
        let result = process_read_file(&params, &cwd);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_read_file_directory() {
        let dir = tempfile::tempdir().unwrap();
        let params = ReadFileParams {
            path: dir.path().to_str().unwrap().to_string(),
            offset: None,
            limit: None,
        };
        let result = process_read_file(&params, std::path::Path::new("."));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_read_file_actual() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello\nworld\nfoo\nbar\n").unwrap();

        let params = ReadFileParams {
            path: file_path.to_str().unwrap().to_string(),
            offset: None,
            limit: None,
        };
        let result = process_read_file(&params, std::path::Path::new(".")).unwrap();
        assert_eq!(result.total_lines, 4);
        assert!(!result.is_binary);
        assert!(result.content.contains("hello"));
    }

    #[test]
    fn test_process_read_file_binary() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("binary.bin");
        std::fs::write(&file_path, b"hello\x00world").unwrap();

        let params = ReadFileParams {
            path: file_path.to_str().unwrap().to_string(),
            offset: None,
            limit: None,
        };
        let result = process_read_file(&params, std::path::Path::new(".")).unwrap();
        assert!(result.is_binary);
    }

    #[test]
    fn test_process_read_file_with_offset() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\nline4\nline5\n").unwrap();

        let params = ReadFileParams {
            path: file_path.to_str().unwrap().to_string(),
            offset: Some(3),
            limit: Some(2),
        };
        let result = process_read_file(&params, std::path::Path::new(".")).unwrap();
        assert_eq!(result.total_lines, 5);
        assert!(result.truncated);
        assert!(result.content.contains("line3"));
        assert!(result.content.contains("line4"));
    }
}
