//! codebase_search tool implementation.

use crate::types::*;
use roo_types::tool::CodebaseSearchParams;

/// Validate codebase_search parameters.
pub fn validate_codebase_search_params(
    params: &CodebaseSearchParams,
) -> Result<(), SearchToolError> {
    if params.query.trim().is_empty() {
        return Err(SearchToolError::Validation(
            "query must not be empty".to_string(),
        ));
    }

    Ok(())
}

/// Build a CodebaseSearchResult from search matches.
pub fn build_codebase_search_result(
    query: &str,
    directory_prefix: Option<&str>,
    matches: Vec<CodebaseMatch>,
) -> CodebaseSearchResult {
    let total_results = matches.len();
    CodebaseSearchResult {
        query: query.to_string(),
        directory_prefix: directory_prefix.map(|s| s.to_string()),
        results: matches,
        total_results,
    }
}

/// Format a codebase search result as JSON-like output.
pub fn format_codebase_search_output(result: &CodebaseSearchResult) -> String {
    let mut output = format!("Query: {}\n", result.query);

    if let Some(ref prefix) = result.directory_prefix {
        output.push_str(&format!("Directory: {prefix}\n"));
    }

    output.push_str(&format!("Results: {}\n", result.total_results));
    output.push('\n');

    for m in &result.results {
        output.push_str(&format!(
            "{}:{}: {}\n",
            m.file_path, m.line_number, m.line_content
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty_query() {
        let params = CodebaseSearchParams {
            query: "".to_string(),
            directory_prefix: None,
        };
        assert!(validate_codebase_search_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid() {
        let params = CodebaseSearchParams {
            query: "fn main".to_string(),
            directory_prefix: Some("src".to_string()),
        };
        assert!(validate_codebase_search_params(&params).is_ok());
    }

    #[test]
    fn test_build_result() {
        let matches = vec![
            CodebaseMatch {
                file_path: "src/main.rs".to_string(),
                line_number: 1,
                line_content: "fn main()".to_string(),
                score: 0.99,
            },
        ];
        let result = build_codebase_search_result("fn main", Some("src"), matches);
        assert_eq!(result.total_results, 1);
        assert_eq!(result.query, "fn main");
        assert_eq!(result.directory_prefix, Some("src".to_string()));
    }

    #[test]
    fn test_format_output() {
        let result = CodebaseSearchResult {
            query: "test".to_string(),
            directory_prefix: None,
            results: vec![CodebaseMatch {
                file_path: "a.rs".to_string(),
                line_number: 5,
                line_content: "test fn".to_string(),
                score: 0.8,
            }],
            total_results: 1,
        };
        let output = format_codebase_search_output(&result);
        assert!(output.contains("Query: test"));
        assert!(output.contains("Results: 1"));
        assert!(output.contains("a.rs:5: test fn"));
    }

    #[test]
    fn test_format_output_with_prefix() {
        let result = CodebaseSearchResult {
            query: "q".to_string(),
            directory_prefix: Some("src".to_string()),
            results: vec![],
            total_results: 0,
        };
        let output = format_codebase_search_output(&result);
        assert!(output.contains("Directory: src"));
    }
}
