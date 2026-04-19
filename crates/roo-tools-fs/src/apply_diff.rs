//! apply_diff tool implementation.
//!
//! Handles parsing and validation of diff parameters, coordinating
//! the actual diff application (delegated to roo-diff).

use crate::types::*;
use roo_types::tool::ApplyDiffParams;

/// Validate apply_diff parameters.
pub fn validate_apply_diff_params(params: &ApplyDiffParams) -> Result<(), FsToolError> {
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

    // Validate diff format: must contain SEARCH/REPLACE markers
    validate_diff_format(&params.diff)?;

    Ok(())
}

/// Validate that the diff string contains properly formatted search/replace blocks.
pub fn validate_diff_format(diff: &str) -> Result<(), FsToolError> {
    let has_search = diff.contains("<<<<<<< SEARCH");
    let has_separator = diff.contains("=======");
    let has_end = diff.contains(">>>>>>> REPLACE");

    if !has_search {
        return Err(FsToolError::InvalidDiff(
            "diff must contain '<<<<<<< SEARCH' markers".to_string(),
        ));
    }

    if !has_separator {
        return Err(FsToolError::InvalidDiff(
            "diff must contain '=======' separator".to_string(),
        ));
    }

    if !has_end {
        return Err(FsToolError::InvalidDiff(
            "diff must contain '>>>>>>> REPLACE' markers".to_string(),
        ));
    }

    Ok(())
}

/// Count the number of diff blocks in a diff string.
pub fn count_diff_blocks(diff: &str) -> usize {
    diff.matches("<<<<<<< SEARCH").count()
}

/// Parse diff blocks from the diff string.
/// Returns a vector of (search_content, replace_content) tuples.
///
/// The diff format is:
/// ```text
/// <<<<<<< SEARCH
/// :start_line:N
/// -------
/// search content
/// =======
/// replace content
/// >>>>>>> REPLACE
/// ```
///
/// The `:start_line:N` line is optional and is skipped if present.
/// The `-------` separator line after `:start_line:` is also optional.
/// Legacy format `:N:` is also supported.
pub fn parse_diff_blocks(diff: &str) -> Result<Vec<(String, String)>, FsToolError> {
    let mut blocks = Vec::new();
    let mut remaining = diff;

    while let Some(search_start) = remaining.find("<<<<<<< SEARCH") {
        // Skip past the SEARCH marker line
        let header_end = remaining[search_start..]
            .find('\n')
            .map(|pos| search_start + pos + 1)
            .unwrap_or(remaining.len());

        let mut after_header = &remaining[header_end..];

        // Skip optional :start_line: header
        // The format is: `:start_line:N` or `:N:` (legacy)
        if let Some(line_end) = after_header.find('\n') {
            let first_line = after_header[..line_end].trim();
            // Match `:start_line:N` or `:N:` patterns
            if first_line.starts_with(':') {
                // Skip this line
                after_header = &after_header[line_end + 1..];

                // Skip the `-------` separator line that follows :start_line:
                if let Some(next_line_end) = after_header.find('\n') {
                    let second_line = after_header[..next_line_end].trim();
                    if second_line.starts_with('-') && second_line.chars().all(|c| c == '-') {
                        after_header = &after_header[next_line_end + 1..];
                    }
                }
            }
        }

        // Find the separator
        let separator_pos = after_header
            .find("=======")
            .ok_or_else(|| FsToolError::InvalidDiff("missing '=======' separator".to_string()))?;

        let search_content = after_header[..separator_pos].trim_end().to_string();

        let after_separator_raw = &after_header[separator_pos + "=======".len()..];
        // Skip the newline immediately after the separator
        let after_separator = after_separator_raw.strip_prefix('\n').unwrap_or(after_separator_raw);

        // Find the end marker
        let end_pos = after_separator
            .find(">>>>>>> REPLACE")
            .ok_or_else(|| FsToolError::InvalidDiff("missing '>>>>>>> REPLACE' marker".to_string()))?;

        let replace_content = after_separator[..end_pos].trim_end().to_string();

        blocks.push((search_content, replace_content));

        remaining = &after_separator[end_pos + ">>>>>>> REPLACE".len()..];
    }

    Ok(blocks)
}

/// Apply parsed diff blocks to content.
pub fn apply_diff_blocks(
    original: &str,
    blocks: &[(String, String)],
) -> Result<DiffApplyResult, FsToolError> {
    let mut content = original.to_string();
    let mut applied = 0;
    let mut warnings = Vec::new();

    for (i, (search, replace)) in blocks.iter().enumerate() {
        if let Some(pos) = content.find(search.as_str()) {
            content.replace_range(pos..pos + search.len(), replace);
            applied += 1;
        } else {
            warnings.push(format!(
                "Block {}: search content not found in file",
                i + 1
            ));
        }
    }

    Ok(DiffApplyResult {
        path: String::new(), // Caller sets this
        blocks_applied: applied,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_path() {
        let params = ApplyDiffParams {
            path: "".to_string(),
            diff: "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE".to_string(),
        };
        assert!(validate_apply_diff_params(&params).is_err());
    }

    #[test]
    fn test_validate_path_traversal() {
        let params = ApplyDiffParams {
            path: "../etc/passwd".to_string(),
            diff: "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE".to_string(),
        };
        assert!(validate_apply_diff_params(&params).is_err());
    }

    #[test]
    fn test_validate_empty_diff() {
        let params = ApplyDiffParams {
            path: "test.txt".to_string(),
            diff: "".to_string(),
        };
        assert!(validate_apply_diff_params(&params).is_err());
    }

    #[test]
    fn test_validate_invalid_diff_no_markers() {
        let params = ApplyDiffParams {
            path: "test.txt".to_string(),
            diff: "just some text without markers".to_string(),
        };
        assert!(validate_apply_diff_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid_diff() {
        let params = ApplyDiffParams {
            path: "test.txt".to_string(),
            diff: "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE".to_string(),
        };
        assert!(validate_apply_diff_params(&params).is_ok());
    }

    #[test]
    fn test_count_diff_blocks_single() {
        let diff = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE";
        assert_eq!(count_diff_blocks(diff), 1);
    }

    #[test]
    fn test_count_diff_blocks_multiple() {
        let diff = "\
<<<<<<< SEARCH
old1
=======
new1
>>>>>>> REPLACE
some text
<<<<<<< SEARCH
old2
=======
new2
>>>>>>> REPLACE";
        assert_eq!(count_diff_blocks(diff), 2);
    }

    #[test]
    fn test_parse_diff_blocks_single() {
        let diff = "<<<<<<< SEARCH\n:9:\nold content\n=======\nnew content\n>>>>>>> REPLACE";
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "old content");
        assert_eq!(blocks[0].1, "new content");
    }

    #[test]
    fn test_parse_diff_blocks_multiple() {
        let diff = "\
<<<<<<< SEARCH
:5:
hello
=======
world
>>>>>>> REPLACE
<<<<<<< SEARCH
:10:
foo
=======
bar
>>>>>>> REPLACE";
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, "hello");
        assert_eq!(blocks[0].1, "world");
        assert_eq!(blocks[1].0, "foo");
        assert_eq!(blocks[1].1, "bar");
    }

    #[test]
    fn test_apply_diff_blocks_success() {
        let original = "line1\nhello\nline3";
        let blocks = vec![("hello".to_string(), "world".to_string())];
        let result = apply_diff_blocks(original, &blocks).unwrap();
        assert_eq!(result.blocks_applied, 1);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_apply_diff_blocks_not_found() {
        let original = "line1\nline2\nline3";
        let blocks = vec![("nonexistent".to_string(), "replacement".to_string())];
        let result = apply_diff_blocks(original, &blocks).unwrap();
        assert_eq!(result.blocks_applied, 0);
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_apply_diff_blocks_partial() {
        let original = "alpha\nbeta\ngamma";
        let blocks = vec![
            ("alpha".to_string(), "ALPHA".to_string()),
            ("missing".to_string(), "MISSING".to_string()),
        ];
        let result = apply_diff_blocks(original, &blocks).unwrap();
        assert_eq!(result.blocks_applied, 1);
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_validate_diff_format_missing_end() {
        let diff = "<<<<<<< SEARCH\nfoo\n=======\nbar";
        assert!(validate_diff_format(diff).is_err());
    }

    #[test]
    fn test_validate_diff_format_missing_separator() {
        let diff = "<<<<<<< SEARCH\nfoo\n>>>>>>> REPLACE";
        assert!(validate_diff_format(diff).is_err());
    }

    // --- New tests for :start_line:N and ------- format ---

    #[test]
    fn test_parse_start_line_format() {
        // Test the :start_line:N format (new Roo Code format)
        let diff = concat!(
            "<<<<<<< SEARCH\n",
            ":start_line:5\n",
            "-------\n",
            "old line\n",
            "=======\n",
            "new line\n",
            ">>>>>>> REPLACE"
        );
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "old line");
        assert_eq!(blocks[0].1, "new line");
    }

    #[test]
    fn test_parse_start_line_format_multiline() {
        // Test :start_line:N with multiline search/replace content
        let diff = concat!(
            "<<<<<<< SEARCH\n",
            ":start_line:10\n",
            "-------\n",
            "line one\n",
            "line two\n",
            "line three\n",
            "=======\n",
            "replaced one\n",
            "replaced two\n",
            ">>>>>>> REPLACE"
        );
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "line one\nline two\nline three");
        assert_eq!(blocks[0].1, "replaced one\nreplaced two");
    }

    #[test]
    fn test_parse_start_line_without_dashes() {
        // Test :start_line:N without the ------- separator
        let diff = concat!(
            "<<<<<<< SEARCH\n",
            ":start_line:42\n",
            "search this\n",
            "=======\n",
            "replace that\n",
            ">>>>>>> REPLACE"
        );
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "search this");
        assert_eq!(blocks[0].1, "replace that");
    }

    #[test]
    fn test_parse_no_start_line_header() {
        // Test diff without any :start_line: or :N: header
        let diff = concat!(
            "<<<<<<< SEARCH\n",
            "plain search\n",
            "=======\n",
            "plain replace\n",
            ">>>>>>> REPLACE"
        );
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "plain search");
        assert_eq!(blocks[0].1, "plain replace");
    }

    #[test]
    fn test_parse_legacy_colon_format() {
        // Test legacy :N: format
        let diff = "<<<<<<< SEARCH\n:15:\nlegacy search\n=======\nlegacy replace\n>>>>>>> REPLACE";
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "legacy search");
        assert_eq!(blocks[0].1, "legacy replace");
    }

    #[test]
    fn test_parse_mixed_formats() {
        // Test a diff with both :start_line:N+------- and :N: formats
        let diff = concat!(
            "<<<<<<< SEARCH\n",
            ":start_line:3\n",
            "-------\n",
            "first old\n",
            "=======\n",
            "first new\n",
            ">>>>>>> REPLACE\n",
            "<<<<<<< SEARCH\n",
            ":7:\n",
            "second old\n",
            "=======\n",
            "second new\n",
            ">>>>>>> REPLACE"
        );
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, "first old");
        assert_eq!(blocks[0].1, "first new");
        assert_eq!(blocks[1].0, "second old");
        assert_eq!(blocks[1].1, "second new");
    }

    #[test]
    fn test_parse_start_line_with_leading_whitespace() {
        // Test that leading whitespace in search content is preserved
        let diff = concat!(
            "<<<<<<< SEARCH\n",
            ":start_line:1\n",
            "-------\n",
            "    indented code\n",
            "=======\n",
            "    replaced code\n",
            ">>>>>>> REPLACE"
        );
        let blocks = parse_diff_blocks(diff).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "    indented code");
        assert_eq!(blocks[0].1, "    replaced code");
    }

    #[test]
    fn test_apply_with_start_line_format() {
        // End-to-end test: parse then apply
        let diff = concat!(
            "<<<<<<< SEARCH\n",
            ":start_line:2\n",
            "-------\n",
            "hello\n",
            "=======\n",
            "world\n",
            ">>>>>>> REPLACE"
        );
        let blocks = parse_diff_blocks(diff).unwrap();
        let original = "line1\nhello\nline3";
        let result = apply_diff_blocks(original, &blocks).unwrap();
        assert_eq!(result.blocks_applied, 1);
        assert!(result.warnings.is_empty());
    }
}
