//! API message builder for constructing conversation history and tool definitions.
//!
//! Builds the messages array and tool definitions sent to the Provider API.
//!
//! Source: `src/core/task/Task.ts` 鈥?message building logic in
//! `recursivelyMakeClineRequests` and `presentAssistantMessage`.

use serde_json::{json, Value};
use tracing::debug;

use roo_types::api::{ApiMessage, ContentBlock, ImageSource, MessageRole, ToolResultContent};

use crate::stream_parser::ParsedStreamContent;
use crate::tool_dispatcher::ToolExecutionResult;

// ---------------------------------------------------------------------------
// MessageBuilder
// ---------------------------------------------------------------------------

/// Builds API messages and tool definitions for Provider API calls.
///
/// Responsibilities:
/// 1. Convert conversation history into the `Vec<ApiMessage>` format
/// 2. Build tool definitions in the format expected by the Provider
/// 3. Create user, assistant, and tool-result messages
///
/// Source: `src/core/task/Task.ts` 鈥?`recursivelyMakeClineRequests` (message
/// building parts), plus `addToApiConversationHistory` calls scattered
/// throughout the Task class.
pub struct MessageBuilder {
    /// System prompt text (sent separately from messages in most providers).
    system_prompt: String,
    /// Whether the model supports image processing.
    supports_images: bool,
}

impl MessageBuilder {
    /// Create a new message builder with the given system prompt.
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            supports_images: false,
        }
    }

    /// Create a message builder with image support flag.
    pub fn with_images_support(mut self, supports: bool) -> Self {
        self.supports_images = supports;
        self
    }

    /// Get the system prompt.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    // -----------------------------------------------------------------------
    // Build tool definitions
    // -----------------------------------------------------------------------

    /// Build tool definitions in the format expected by the Provider API.
    ///
    /// Returns a vector of JSON values, each representing a tool definition
    /// in OpenAI Function Calling format:
    /// ```json
    /// {
    ///   "type": "function",
    ///   "function": {
    ///     "name": "...",
    ///     "description": "...",
    ///     "parameters": { ... }
    ///   }
    /// }
    /// ```
    ///
    /// Source: `src/core/task/Task.ts` 鈥?tool definition building in
    /// `recursivelyMakeClineRequests`
    pub fn build_tool_definitions(&self) -> Vec<Value> {
        let options = roo_tools::definition::NativeToolsOptions {
            supports_images: self.supports_images,
        };
        let native_tools = roo_tools::definition::get_native_tools(options);

        native_tools
            .into_iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Build messages
    // -----------------------------------------------------------------------

    /// Build the complete messages array for an API call.
    ///
    /// Takes the existing conversation history and optionally appends a new
    /// user message. The system prompt is handled separately by the provider.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`recursivelyMakeClineRequests` builds
    /// messages from `apiConversationHistory` before each API call.
    pub fn build_api_messages(
        &self,
        history: &[ApiMessage],
        user_message: Option<&str>,
        images: &[String],
    ) -> Vec<ApiMessage> {
        let mut messages: Vec<ApiMessage> = history.to_vec();

        // Append a new user message if provided
        if let Some(text) = user_message {
            let msg = Self::create_user_message(text, images);
            messages.push(msg);
        }

        debug!(
            "Built {} API messages (history: {}, new user msg: {})",
            messages.len(),
            history.len(),
            user_message.is_some(),
        );

        messages
    }

    // -----------------------------------------------------------------------
    // Message constructors
    // -----------------------------------------------------------------------

    /// Create a user message with text and optional images.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?initial user message construction
    /// in `startTask` / `resumeTask`
    pub fn create_user_message(text: &str, images: &[String]) -> ApiMessage {
        let mut content: Vec<ContentBlock> = Vec::new();

        // Add text block
        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }

        // Add image blocks
        for image_data in images {
            content.push(ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: guess_media_type(image_data),
                    data: image_data.clone(),
                },
            });
        }

        ApiMessage {
            role: MessageRole::User,
            content,
            reasoning: None,
            ts: Some(chrono::Utc::now().timestamp_millis() as f64 / 1000.0),
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        }
    }

    /// Create an assistant message from parsed stream content.
    ///
    /// Converts the accumulated `ParsedStreamContent` into an `ApiMessage`
    /// with the appropriate content blocks (thinking, text, tool_use).
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`presentAssistantMessage` saves the
    /// assistant response to `apiConversationHistory`
    pub fn create_assistant_message(parsed: &ParsedStreamContent) -> ApiMessage {
        let content = parsed.to_content_blocks();

        ApiMessage {
            role: MessageRole::Assistant,
            content,
            reasoning: None,
            ts: Some(chrono::Utc::now().timestamp_millis() as f64 / 1000.0),
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        }
    }

    /// Create a tool result message.
    ///
    /// Converts a `ToolExecutionResult` into an `ApiMessage` with a
    /// `ToolResult` content block.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?tool result is added to
    /// `apiConversationHistory` after each tool execution
    pub fn create_tool_result_message(
        tool_use_id: &str,
        result: &ToolExecutionResult,
    ) -> ApiMessage {
        let mut content: Vec<ToolResultContent> = vec![ToolResultContent::Text {
            text: result.text.clone(),
        }];

        // Add images if present
        if let Some(ref images) = result.images {
            for image_data in images {
                content.push(ToolResultContent::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: image_data.clone(),
                    },
                });
            }
        }

        let is_error = if result.is_error {
            Some(true)
        } else {
            None
        };

        ApiMessage {
            role: MessageRole::User, // Tool results are sent as "user" role
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content,
                is_error,
            }],
            reasoning: None,
            ts: Some(chrono::Utc::now().timestamp_millis() as f64 / 1000.0),
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Guess the media type from base64 image data or file extension.
///
/// If the data starts with a known image signature, return that type.
/// Otherwise default to "image/png".
fn guess_media_type(data: &str) -> String {
    // Check for common base64-encoded image signatures
    // JPEG: starts with /9j/
    if data.starts_with("/9j/") {
        return "image/jpeg".to_string();
    }
    // PNG: starts with iVBOR
    if data.starts_with("iVBOR") {
        return "image/png".to_string();
    }
    // GIF: starts with R0lGOD
    if data.starts_with("R0lGOD") {
        return "image/gif".to_string();
    }
    // WebP: starts with UklGR
    if data.starts_with("UklGR") {
        return "image/webp".to_string();
    }
    // SVG: might start with PHN2Zw (base64 of <svg)
    if data.starts_with("PHN2Zw") {
        return "image/svg+xml".to_string();
    }

    // Default
    "image/png".to_string()
}

impl Default for MessageBuilder {
    fn default() -> Self {
        Self::new(String::new())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_builder_new() {
        let builder = MessageBuilder::new("You are a helpful assistant.");
        assert_eq!(builder.system_prompt(), "You are a helpful assistant.");
        assert!(!builder.supports_images);
    }

    #[test]
    fn test_message_builder_with_images_support() {
        let builder = MessageBuilder::new("test").with_images_support(true);
        assert!(builder.supports_images);
    }

    #[test]
    fn test_message_builder_default() {
        let builder = MessageBuilder::default();
        assert!(builder.system_prompt().is_empty());
    }

    #[test]
    fn test_build_tool_definitions() {
        let builder = MessageBuilder::new("test");
        let tools = builder.build_tool_definitions();

        // Should have 21 native tools
        assert_eq!(tools.len(), 21);

        // Each tool should have the expected structure
        for tool in &tools {
            assert_eq!(tool["type"], "function");
            assert!(tool["function"]["name"].is_string());
            assert!(tool["function"]["description"].is_string());
            assert!(tool["function"]["parameters"].is_object());
        }

        // Check some known tools
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_to_file"));
        assert!(names.contains(&"apply_diff"));
    }

    #[test]
    fn test_create_user_message_text_only() {
        let msg = MessageBuilder::create_user_message("Hello, world!", &[]);

        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content.len(), 1);

        match &msg.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected Text block"),
        }

        assert!(msg.ts.is_some());
    }

    #[test]
    fn test_create_user_message_with_images() {
        let images = vec!["iVBORw0KGgoAAAANSUhEUg==".to_string()]; // PNG-like
        let msg = MessageBuilder::create_user_message("See this image", &images);

        assert_eq!(msg.content.len(), 2); // text + image

        match &msg.content[1] {
            ContentBlock::Image { source } => match source {
                ImageSource::Base64 { media_type, data } => {
                    assert_eq!(media_type, "image/png");
                    assert_eq!(data, "iVBORw0KGgoAAAANSUhEUg==");
                }
                _ => panic!("Expected Base64 image source"),
            },
            _ => panic!("Expected Image block"),
        }
    }

    #[test]
    fn test_create_user_message_empty_text_with_images() {
        let images = vec!["/9j/4AAQSkZJRg==".to_string()]; // JPEG-like
        let msg = MessageBuilder::create_user_message("", &images);

        // No text block since text is empty, only image
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            ContentBlock::Image { .. } => {}
            _ => panic!("Expected Image block"),
        }
    }

    #[test]
    fn test_create_assistant_message() {
        let mut parser = crate::stream_parser::StreamParser::new();
        parser.feed_chunk(&roo_types::api::ApiStreamChunk::Text {
            text: "I'll help you.".into(),
        });
        parser.feed_chunk(&roo_types::api::ApiStreamChunk::ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"main.rs"}"#.into(),
        });

        let content = parser.finalize();
        let msg = MessageBuilder::create_assistant_message(&content);

        assert_eq!(msg.role, MessageRole::Assistant);
        // to_content_blocks() produces: Text + ToolUse = 2 blocks
        assert_eq!(msg.content.len(), 2);
    }

    #[test]
    fn test_create_tool_result_message_success() {
        let result = ToolExecutionResult::success("File contents here");
        let msg = MessageBuilder::create_tool_result_message("call_1", &result);

        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content.len(), 1);

        match &msg.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "call_1");
                assert!(is_error.is_none());
                assert_eq!(content.len(), 1);
                match &content[0] {
                    ToolResultContent::Text { text } => {
                        assert_eq!(text, "File contents here");
                    }
                    _ => panic!("Expected Text content"),
                }
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn test_create_tool_result_message_error() {
        let result = ToolExecutionResult::error("File not found");
        let msg = MessageBuilder::create_tool_result_message("call_2", &result);

        match &msg.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "call_2");
                assert_eq!(*is_error, Some(true));
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn test_create_tool_result_message_with_images() {
        let result =
            ToolExecutionResult::success_with_images("screenshot", vec!["base64data".into()]);
        let msg = MessageBuilder::create_tool_result_message("call_3", &result);

        match &msg.content[0] {
            ContentBlock::ToolResult { content, .. } => {
                assert_eq!(content.len(), 2); // text + image
            }
            _ => panic!("Expected ToolResult block"),
        }
    }

    #[test]
    fn test_build_api_messages_empty_history() {
        let builder = MessageBuilder::new("test");
        let messages = builder.build_api_messages(&[], None, &[]);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_build_api_messages_with_history() {
        let builder = MessageBuilder::new("test");
        let history = vec![
            MessageBuilder::create_user_message("Hello", &[]),
            MessageBuilder::create_assistant_message(&{
                let mut p = crate::stream_parser::StreamParser::new();
                p.feed_chunk(&roo_types::api::ApiStreamChunk::Text {
                    text: "Hi there".into(),
                });
                p.finalize()
            }),
        ];

        let messages = builder.build_api_messages(&history, None, &[]);
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_build_api_messages_with_new_user_message() {
        let builder = MessageBuilder::new("test");
        let history = vec![MessageBuilder::create_user_message("Hello", &[])];
        let messages = builder.build_api_messages(&history, Some("Follow up"), &[]);

        assert_eq!(messages.len(), 2);
        match &messages[1].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Follow up"),
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_guess_media_type() {
        assert_eq!(guess_media_type("/9j/abc"), "image/jpeg");
        assert_eq!(guess_media_type("iVBORabc"), "image/png");
        assert_eq!(guess_media_type("R0lGODabc"), "image/gif");
        assert_eq!(guess_media_type("UklGRabc"), "image/webp");
        assert_eq!(guess_media_type("PHN2Zwabc"), "image/svg+xml");
        assert_eq!(guess_media_type("unknown"), "image/png");
    }
}