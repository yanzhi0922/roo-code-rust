//! use_mcp_tool tool implementation.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::UseMcpToolParams;

/// Validate use_mcp_tool parameters.
pub fn validate_use_mcp_tool_params(params: &UseMcpToolParams) -> Result<McpToolValidation, McpToolError> {
    if params.server_name.trim().is_empty() {
        return Err(McpToolError::MissingParameter("server_name".to_string()));
    }

    if params.tool_name.trim().is_empty() {
        return Err(McpToolError::MissingParameter("tool_name".to_string()));
    }

    validate_mcp_arguments(&params.arguments)?;

    Ok(McpToolValidation {
        server_name: params.server_name.clone(),
        tool_name: params.tool_name.clone(),
        is_valid: true,
        error: None,
    })
}

/// Check if a tool exists in the available tools list.
pub fn find_tool<'a>(
    available_tools: &'a [String],
    tool_name: &str,
) -> Option<&'a String> {
    available_tools
        .iter()
        .find(|t| tool_names_match(t, tool_name))
}

/// Prepare an MCP tool call.
pub fn prepare_mcp_tool_call(
    params: &UseMcpToolParams,
    available_tools: &[String],
) -> Result<McpToolResult, McpToolError> {
    let validation = validate_use_mcp_tool_params(params)?;

    // Check if tool exists
    if !available_tools.is_empty() {
        let found = find_tool(available_tools, &params.tool_name);
        if found.is_none() {
            return Err(McpToolError::ToolNotFound(format!(
                "Tool '{}' not found on server '{}'. Available: [{}]",
                params.tool_name,
                params.server_name,
                available_tools.join(", ")
            )));
        }
    }

    Ok(McpToolResult {
        server_name: validation.server_name,
        tool_name: validation.tool_name,
        result: params.arguments.clone(),
        is_error: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_missing_server_name() {
        let params = UseMcpToolParams {
            server_name: "".to_string(),
            tool_name: "tool".to_string(),
            arguments: json!({}),
        };
        assert!(validate_use_mcp_tool_params(&params).is_err());
    }

    #[test]
    fn test_validate_missing_tool_name() {
        let params = UseMcpToolParams {
            server_name: "server".to_string(),
            tool_name: "".to_string(),
            arguments: json!({}),
        };
        assert!(validate_use_mcp_tool_params(&params).is_err());
    }

    #[test]
    fn test_validate_invalid_arguments() {
        let params = UseMcpToolParams {
            server_name: "server".to_string(),
            tool_name: "tool".to_string(),
            arguments: json!("not an object"),
        };
        assert!(validate_use_mcp_tool_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid_params() {
        let params = UseMcpToolParams {
            server_name: "server".to_string(),
            tool_name: "tool".to_string(),
            arguments: json!({"key": "value"}),
        };
        let result = validate_use_mcp_tool_params(&params).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_null_arguments() {
        let params = UseMcpToolParams {
            server_name: "server".to_string(),
            tool_name: "tool".to_string(),
            arguments: json!(null),
        };
        assert!(validate_use_mcp_tool_params(&params).is_ok());
    }

    #[test]
    fn test_find_tool_exact() {
        let tools = vec!["tool-a".to_string(), "tool-b".to_string()];
        assert!(find_tool(&tools, "tool-a").is_some());
    }

    #[test]
    fn test_find_tool_normalized() {
        let tools = vec!["my_tool".to_string()];
        assert!(find_tool(&tools, "my-tool").is_some());
    }

    #[test]
    fn test_find_tool_not_found() {
        let tools = vec!["tool-a".to_string()];
        assert!(find_tool(&tools, "tool-b").is_none());
    }

    #[test]
    fn test_prepare_mcp_tool_call_success() {
        let params = UseMcpToolParams {
            server_name: "server".to_string(),
            tool_name: "my-tool".to_string(),
            arguments: json!({"x": 1}),
        };
        let tools = vec!["my_tool".to_string()];
        let result = prepare_mcp_tool_call(&params, &tools).unwrap();
        assert!(!result.is_error);
        assert_eq!(result.tool_name, "my-tool");
    }

    #[test]
    fn test_prepare_mcp_tool_call_not_found() {
        let params = UseMcpToolParams {
            server_name: "server".to_string(),
            tool_name: "missing".to_string(),
            arguments: json!({}),
        };
        let tools = vec!["other".to_string()];
        assert!(prepare_mcp_tool_call(&params, &tools).is_err());
    }

    #[test]
    fn test_prepare_mcp_tool_call_empty_tools() {
        let params = UseMcpToolParams {
            server_name: "server".to_string(),
            tool_name: "any-tool".to_string(),
            arguments: json!({}),
        };
        // Empty tools list means skip tool existence check
        let result = prepare_mcp_tool_call(&params, &[]).unwrap();
        assert!(!result.is_error);
    }
}
