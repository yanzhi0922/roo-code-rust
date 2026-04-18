//! MCP name utility functions.
//!
//! Corresponds to TS: `sanitizeMcpName` and `toolNamesMatch`.

/// Separator used between MCP prefix, server name, and tool name.
pub const MCP_TOOL_SEPARATOR: &str = "--";

/// Prefix for all MCP tool function names.
pub const MCP_TOOL_PREFIX: &str = "mcp";

/// Sanitize a name to be safe for use in API function names.
///
/// This removes special characters and ensures the name starts correctly.
/// Hyphens are preserved since they are valid in function names.
///
/// Corresponds to TS: `sanitizeMcpName`
pub fn sanitize_mcp_name(name: &str) -> String {
    if name.is_empty() {
        return "_".to_string();
    }

    // Replace runs of whitespace with a single underscore first
    let mut sanitized = String::new();
    let mut last_was_space = false;
    for ch in name.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                sanitized.push('_');
                last_was_space = true;
            }
        } else {
            sanitized.push(ch);
            last_was_space = false;
        }
    }

    // Only allow alphanumeric, underscores, and hyphens
    sanitized = sanitized
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();

    // Replace any double-hyphen sequences with single hyphen to avoid separator conflicts
    while sanitized.contains("--") {
        sanitized = sanitized.replace("--", "-");
    }

    // Ensure the name starts with a letter or underscore
    if !sanitized.is_empty() {
        let first = sanitized.chars().next().unwrap();
        if !first.is_ascii_alphabetic() && first != '_' {
            sanitized = format!("_{}", sanitized);
        }
    }

    // If empty after sanitization, use a placeholder
    if sanitized.is_empty() {
        sanitized = "_unnamed".to_string();
    }

    sanitized
}

/// Normalize a string for comparison by treating hyphens and underscores as equivalent.
///
/// Corresponds to TS: `normalizeForComparison`
fn normalize_for_comparison(name: &str) -> String {
    name.replace('-', "_")
}

/// Check if two tool names match using fuzzy comparison.
///
/// Treats hyphens and underscores as equivalent.
///
/// Corresponds to TS: `toolNamesMatch`
pub fn tool_names_match(name1: &str, name2: &str) -> bool {
    normalize_for_comparison(name1) == normalize_for_comparison(name2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- sanitize_mcp_name tests ----

    #[test]
    fn test_sanitize_empty() {
        assert_eq!(sanitize_mcp_name(""), "_");
    }

    #[test]
    fn test_sanitize_spaces_to_underscores() {
        assert_eq!(sanitize_mcp_name("my server"), "my_server");
        assert_eq!(sanitize_mcp_name("server name here"), "server_name_here");
    }

    #[test]
    fn test_sanitize_removes_invalid_chars() {
        assert_eq!(sanitize_mcp_name("server@name!"), "servername");
        assert_eq!(sanitize_mcp_name("test#$%^&*()"), "test");
    }

    #[test]
    fn test_sanitize_keeps_valid_chars() {
        assert_eq!(sanitize_mcp_name("server_name"), "server_name");
        assert_eq!(sanitize_mcp_name("server-name"), "server-name");
        assert_eq!(sanitize_mcp_name("Server123"), "Server123");
    }

    #[test]
    fn test_sanitize_removes_dots_and_colons() {
        // Dots and colons are NOT allowed due to AWS Bedrock restrictions
        assert_eq!(sanitize_mcp_name("server.name"), "servername");
        assert_eq!(sanitize_mcp_name("server:name"), "servername");
        // Hyphens are preserved
        assert_eq!(
            sanitize_mcp_name("awslabs.aws-documentation-mcp-server"),
            "awslabsaws-documentation-mcp-server"
        );
    }

    #[test]
    fn test_sanitize_prepends_underscore_for_digit_start() {
        assert_eq!(sanitize_mcp_name("123server"), "_123server");
        // Hyphen at start still needs underscore prefix
        assert_eq!(sanitize_mcp_name("-server"), "_-server");
        // Dots are removed, so ".server" becomes "server" which starts with a letter
        assert_eq!(sanitize_mcp_name(".server"), "server");
    }

    #[test]
    fn test_sanitize_no_modify_valid_start() {
        assert_eq!(sanitize_mcp_name("server"), "server");
        assert_eq!(sanitize_mcp_name("_server"), "_server");
        assert_eq!(sanitize_mcp_name("Server"), "Server");
    }

    #[test]
    fn test_sanitize_double_hyphen() {
        assert_eq!(sanitize_mcp_name("server--name"), "server-name");
        assert_eq!(sanitize_mcp_name("test---server"), "test-server");
        assert_eq!(sanitize_mcp_name("my----tool"), "my-tool");
    }

    #[test]
    fn test_sanitize_complex_names() {
        assert_eq!(sanitize_mcp_name("My Server @ Home!"), "My_Server__Home");
        assert_eq!(sanitize_mcp_name("123-test server"), "_123-test_server");
    }

    #[test]
    fn test_sanitize_all_invalid_becomes_unnamed() {
        assert_eq!(sanitize_mcp_name("@#$%"), "_unnamed");
    }

    #[test]
    fn test_sanitize_preserves_hyphens() {
        assert_eq!(sanitize_mcp_name("atlassian-jira_search"), "atlassian-jira_search");
        assert_eq!(
            sanitize_mcp_name("atlassian-confluence_search"),
            "atlassian-confluence_search"
        );
    }

    #[test]
    fn test_sanitize_whitespace_only() {
        assert_eq!(sanitize_mcp_name("   "), "_");
    }

    // ---- tool_names_match tests ----

    #[test]
    fn test_match_identical_names() {
        assert!(tool_names_match("get_user", "get_user"));
        assert!(tool_names_match("get-user", "get-user"));
    }

    #[test]
    fn test_match_hyphens_vs_underscores() {
        assert!(tool_names_match("get-user", "get_user"));
        assert!(tool_names_match("get_user", "get-user"));
    }

    #[test]
    fn test_match_complex_mcp_names() {
        assert!(tool_names_match(
            "mcp--server--get-user-profile",
            "mcp__server__get_user_profile"
        ));
    }

    #[test]
    fn test_no_match_different_names() {
        assert!(!tool_names_match("get_user", "get_profile"));
    }
}
