/// Native tool call parser for parsing assistant messages that contain tool calls.
/// Mirrors src/core/assistant-message/NativeToolCallParser.ts

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a parsed tool call from an assistant message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    /// The name of the tool being called.
    pub name: String,
    /// The parameters for the tool call as a JSON value.
    pub params: Value,
}

/// Represents a parsed assistant message content block.
#[derive(Clone, Debug)]
pub enum ParsedBlock {
    /// Regular text content.
    Text(String),
    /// A tool call.
    ToolCall(ToolCall),
}

/// Parser for extracting tool calls from assistant messages.
/// Handles both XML-style tool calls and JSON-based tool calls.
pub struct NativeToolCallParser;

impl NativeToolCallParser {
    /// Parse an assistant message string and extract tool calls.
    pub fn parse(content: &str) -> Vec<ParsedBlock> {
        let mut blocks = Vec::new();

        // Try to parse as JSON first (for structured tool calls)
        if let Ok(value) = serde_json::from_str::<Value>(content) {
            if let Some(tool_calls) = value.get("tool_calls").and_then(|tc| tc.as_array()) {
                // Process structured tool calls
                let mut text_before = String::new();
                if let Some(text) = value.get("content").and_then(|c| c.as_str()) {
                    text_before = text.to_string();
                }

                if !text_before.is_empty() {
                    blocks.push(ParsedBlock::Text(text_before));
                }

                for tc in tool_calls {
                    let name = tc["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    let params_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                    let params: Value =
                        serde_json::from_str(params_str).unwrap_or(Value::Object(Default::default()));

                    blocks.push(ParsedBlock::ToolCall(ToolCall { name, params }));
                }

                return blocks;
            }
        }

        // Fall back to text-based parsing
        // Look for XML-style tool invocations
        let mut remaining = content.to_string();
        let tool_call_pattern = regex::Regex::new(
            r#"(?s)<tool_call\s+name\s*=\s*"([^"]+)"[^>]*>(.*?)</tool_call\s*>"#
        );

        if let Ok(pattern) = tool_call_pattern {
            while let Some(captures) = pattern.captures(&remaining) {
                let full_match = captures.get(0).unwrap();
                let before = &remaining[..full_match.start()];
                let name = captures[1].to_string();
                let params_str = captures[2].trim();

                if !before.is_empty() {
                    blocks.push(ParsedBlock::Text(before.to_string()));
                }

                let params: Value = serde_json::from_str(params_str)
                    .unwrap_or(Value::Object(Default::default()));

                blocks.push(ParsedBlock::ToolCall(ToolCall { name, params }));

                remaining = remaining[full_match.end()..].to_string();
            }
        }

        // Any remaining text
        if !remaining.is_empty() {
            blocks.push(ParsedBlock::Text(remaining.clone()));
        }

        // If no tool calls found, return the entire content as text
        if blocks.is_empty() {
            blocks.push(ParsedBlock::Text(content.to_string()));
        }

        blocks
    }

    /// Extract just the text portions from parsed blocks.
    pub fn extract_text(blocks: &[ParsedBlock]) -> String {
        blocks
            .iter()
            .filter_map(|b| match b {
                ParsedBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract just the tool calls from parsed blocks.
    pub fn extract_tool_calls(blocks: &[ParsedBlock]) -> Vec<&ToolCall> {
        blocks
            .iter()
            .filter_map(|b| match b {
                ParsedBlock::ToolCall(tc) => Some(tc),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_text() {
        let blocks = NativeToolCallParser::parse("Hello, world!");
        assert_eq!(1, blocks.len());
        match &blocks[0] {
            ParsedBlock::Text(t) => assert_eq!("Hello, world!", t),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_parse_json_tool_calls() {
        let content = r#"{"content": "I'll help you.", "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\": \"test.rs\"}"}}]}"#;
        let blocks = NativeToolCallParser::parse(content);

        assert!(blocks.len() >= 2);
        let tool_calls = NativeToolCallParser::extract_tool_calls(&blocks);
        assert_eq!(1, tool_calls.len());
        assert_eq!("read_file", tool_calls[0].name);
        assert_eq!("test.rs", tool_calls[0].params["path"].as_str().unwrap());
    }

    #[test]
    fn test_extract_text() {
        let blocks = vec![
            ParsedBlock::Text("Hello ".to_string()),
            ParsedBlock::ToolCall(ToolCall {
                name: "test".to_string(),
                params: Value::Null,
            }),
            ParsedBlock::Text("world".to_string()),
        ];
        let text = NativeToolCallParser::extract_text(&blocks);
        assert_eq!("Hello world", text);
    }

    #[test]
    fn test_extract_tool_calls() {
        let blocks = vec![
            ParsedBlock::Text("text".to_string()),
            ParsedBlock::ToolCall(ToolCall {
                name: "tool1".to_string(),
                params: Value::Null,
            }),
            ParsedBlock::ToolCall(ToolCall {
                name: "tool2".to_string(),
                params: Value::Object(Default::default()),
            }),
        ];
        let calls = NativeToolCallParser::extract_tool_calls(&blocks);
        assert_eq!(2, calls.len());
        assert_eq!("tool1", calls[0].name);
        assert_eq!("tool2", calls[1].name);
    }
}
