//! Types for MCP tool results and errors.

use serde::{Deserialize, Serialize};

/// Validation result for MCP tool parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolValidation {
    pub server_name: String,
    pub tool_name: String,
    pub is_valid: bool,
    pub error: Option<String>,
}

/// Result of an MCP resource access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceResult {
    pub server_name: String,
    pub uri: String,
    pub content: String,
    pub content_type: String,
}

/// Result of an MCP tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    pub server_name: String,
    pub tool_name: String,
    pub result: serde_json::Value,
    pub is_error: bool,
}

/// Error type for MCP tool operations.
#[derive(Debug, thiserror::Error)]
pub enum McpToolError {
    #[error("Missing parameter: {0}")]
    MissingParameter(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Server not found: {0}")]
    ServerNotFound(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool name mismatch: expected '{expected}', got '{actual}'")]
    ToolNameMismatch { expected: String, actual: String },

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("MCP error: {0}")]
    Mcp(String),
}
