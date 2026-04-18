//! access_mcp_resource tool implementation.

use crate::helpers::*;
use crate::types::*;
use roo_types::tool::AccessMcpResourceParams;

/// Validate access_mcp_resource parameters.
pub fn validate_access_mcp_resource_params(
    params: &AccessMcpResourceParams,
) -> Result<(), McpToolError> {
    if params.server_name.trim().is_empty() {
        return Err(McpToolError::MissingParameter("server_name".to_string()));
    }

    if params.uri.trim().is_empty() {
        return Err(McpToolError::MissingParameter("uri".to_string()));
    }

    Ok(())
}

/// Process an MCP resource access request.
///
/// Note: The actual resource fetching is handled by the MCP client.
/// This function validates parameters and prepares the result structure.
pub fn prepare_resource_access(
    params: &AccessMcpResourceParams,
) -> Result<McpResourceResult, McpToolError> {
    validate_access_mcp_resource_params(params)?;

    Ok(McpResourceResult {
        server_name: params.server_name.clone(),
        uri: params.uri.clone(),
        content: String::new(), // Filled by actual MCP client
        content_type: "text/plain".to_string(),
    })
}

/// Process raw resource content into a formatted result.
pub fn process_resource_content(
    params: &AccessMcpResourceParams,
    raw_content: &str,
    content_type: &str,
) -> McpResourceResult {
    let formatted = format_resource_content(raw_content, content_type);

    McpResourceResult {
        server_name: params.server_name.clone(),
        uri: params.uri.clone(),
        content: formatted,
        content_type: content_type.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_missing_server_name() {
        let params = AccessMcpResourceParams {
            server_name: "".to_string(),
            uri: "file:///test".to_string(),
        };
        assert!(validate_access_mcp_resource_params(&params).is_err());
    }

    #[test]
    fn test_validate_missing_uri() {
        let params = AccessMcpResourceParams {
            server_name: "server".to_string(),
            uri: "".to_string(),
        };
        assert!(validate_access_mcp_resource_params(&params).is_err());
    }

    #[test]
    fn test_validate_valid_params() {
        let params = AccessMcpResourceParams {
            server_name: "server".to_string(),
            uri: "file:///test".to_string(),
        };
        assert!(validate_access_mcp_resource_params(&params).is_ok());
    }

    #[test]
    fn test_prepare_resource_access() {
        let params = AccessMcpResourceParams {
            server_name: "my-server".to_string(),
            uri: "resource://test".to_string(),
        };
        let result = prepare_resource_access(&params).unwrap();
        assert_eq!(result.server_name, "my-server");
        assert_eq!(result.uri, "resource://test");
    }

    #[test]
    fn test_process_text_content() {
        let params = AccessMcpResourceParams {
            server_name: "server".to_string(),
            uri: "file:///test.txt".to_string(),
        };
        let result = process_resource_content(&params, "hello world", "text/plain");
        assert_eq!(result.content, "hello world");
    }

    #[test]
    fn test_process_image_content() {
        let params = AccessMcpResourceParams {
            server_name: "server".to_string(),
            uri: "file:///image.png".to_string(),
        };
        let result = process_resource_content(&params, "base64data", "image/png");
        assert!(result.content.contains("Image content"));
    }
}
