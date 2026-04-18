//! Filters non-Anthropic content blocks from messages.
//!
//! Derived from `src/api/transform/anthropic-filter.ts`.
//! Uses an allowlist approach — only blocks with recognized Anthropic types are kept.

use roo_types::api::{ApiMessage, ContentBlock};

/// Set of content block types that are valid for the Anthropic API.
/// Only these types will be passed through to the API.
/// See: <https://docs.anthropic.com/en/api/messages>
pub const VALID_ANTHROPIC_BLOCK_TYPES: &[&str] = &[
    "text",
    "image",
    "tool_use",
    "tool_result",
    "thinking",
    "redacted_thinking",
    "document",
];

/// Returns true if the given block type is valid for the Anthropic API.
fn is_valid_anthropic_block(block: &ContentBlock) -> bool {
    matches!(
        block,
        ContentBlock::Text { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolUse { .. }
            | ContentBlock::ToolResult { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
    )
}

/// Filters out non-Anthropic content blocks from messages before sending to
/// Anthropic/Vertex API.
///
/// Uses an allowlist approach — only blocks with recognized Anthropic types are kept.
/// This automatically filters out:
/// - Internal "reasoning" blocks (Roo Code's internal representation)
/// - Gemini's "thoughtSignature" blocks (encrypted reasoning continuity tokens)
/// - Any other unknown block types
///
/// Source: `src/api/transform/anthropic-filter.ts` — `filterNonAnthropicBlocks`
pub fn filter_non_anthropic_blocks(messages: Vec<ApiMessage>) -> Vec<ApiMessage> {
    messages
        .into_iter()
        .filter_map(|message| {
            let filtered_content: Vec<ContentBlock> = message
                .content
                .into_iter()
                .filter(is_valid_anthropic_block)
                .collect();

            // If all content was filtered out, skip this message entirely
            if filtered_content.is_empty() {
                return None;
            }

            Some(ApiMessage {
                content: filtered_content,
                ..message
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use roo_types::api::MessageRole;

    #[test]
    fn test_filters_empty_messages() {
        let messages = vec![ApiMessage {
            role: MessageRole::User,
            content: vec![],
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        }];
        let result = filter_non_anthropic_blocks(messages);
        assert!(result.is_empty());
    }

    #[test]
    fn test_keeps_valid_blocks() {
        let messages = vec![ApiMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: "hello".to_string(),
            }],
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        }];
        let result = filter_non_anthropic_blocks(messages);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content.len(), 1);
    }
}
