//! Stream parser for API streaming responses.
//!
//! Parses a stream of [`ApiStreamChunk`] into structured content blocks:
//! accumulated text, complete tool calls, thinking blocks, and usage info.
//!
//! Source: `src/core/assistant-message/NativeToolCallParser.ts` 鈥?handles
//! accumulating partial tool calls from streaming chunks.

use std::collections::HashMap;

use roo_types::api::{ApiStreamChunk, ContentBlock, GroundingSource};

// ---------------------------------------------------------------------------
// ParsedToolCall
// ---------------------------------------------------------------------------

/// A fully accumulated tool call extracted from the stream.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    /// Unique tool call ID assigned by the API.
    pub id: String,
    /// Tool name (e.g. "read_file", "write_to_file").
    pub name: String,
    /// Raw JSON arguments string.
    pub arguments: String,
}

impl ParsedToolCall {
    /// Parse the arguments string into a JSON value.
    ///
    /// Returns `serde_json::Value::Null` if parsing fails.
    pub fn parse_arguments(&self) -> serde_json::Value {
        serde_json::from_str(&self.arguments).unwrap_or(serde_json::Value::Null)
    }
}

// ---------------------------------------------------------------------------
// ThinkingBlock
// ---------------------------------------------------------------------------

/// A thinking / reasoning block extracted from the stream.
#[derive(Debug, Clone)]
pub struct ThinkingBlock {
    /// The thinking text.
    pub text: String,
    /// Optional signature (Anthropic extended thinking).
    pub signature: Option<String>,
}

// ---------------------------------------------------------------------------
// ParsedStreamContent
// ---------------------------------------------------------------------------

/// The aggregated result of parsing a complete stream response.
#[derive(Debug, Clone, Default)]
pub struct ParsedStreamContent {
    /// Accumulated text content.
    pub text: String,
    /// Complete tool calls.
    pub tool_calls: Vec<ParsedToolCall>,
    /// Thinking / reasoning blocks.
    pub thinking_blocks: Vec<ThinkingBlock>,
    /// Token usage (input, output, cache, cost).
    pub usage: Option<StreamUsage>,
    /// Grounding sources (Gemini).
    pub grounding_sources: Vec<GroundingSource>,
    /// Stream error, if any.
    pub error: Option<StreamError>,
}

// ---------------------------------------------------------------------------
// StreamUsage
// ---------------------------------------------------------------------------

/// Token usage information from the stream.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_write_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub total_cost: Option<f64>,
}

// ---------------------------------------------------------------------------
// StreamError
// ---------------------------------------------------------------------------

/// An error from the stream.
#[derive(Debug, Clone)]
pub struct StreamError {
    pub error: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// PendingToolCall 鈥?internal state for partial tool calls
// ---------------------------------------------------------------------------

/// Internal state tracking a partially accumulated tool call.
#[derive(Debug, Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

// ---------------------------------------------------------------------------
// StreamParser
// ---------------------------------------------------------------------------

/// Parses a stream of [`ApiStreamChunk`] into [`ParsedStreamContent`].
///
/// Handles both complete tool calls (`ToolCall`) and partial tool calls
/// (`ToolCallPartial` with incremental deltas). The parser accumulates
/// text, tool calls, thinking blocks, and usage information.
///
/// # Example
///
/// ```
/// use roo_task::stream_parser::StreamParser;
/// use roo_types::api::ApiStreamChunk;
///
/// let mut parser = StreamParser::new();
///
/// // Feed chunks one at a time
/// parser.feed_chunk(&ApiStreamChunk::Text { text: "Hello".into() });
/// parser.feed_chunk(&ApiStreamChunk::Text { text: " world".into() });
///
/// // Get the final result
/// let content = parser.finalize();
/// assert_eq!(content.text, "Hello world");
/// ```
pub struct StreamParser {
    /// Accumulated text content.
    text: String,
    /// Complete tool calls (accumulated from ToolCall or ToolCallPartial sequences).
    tool_calls: Vec<ParsedToolCall>,
    /// Pending partial tool calls being accumulated (keyed by index).
    /// Used for the `ToolCallPartial` chunk type (index-based tracking).
    pending_tool_calls: HashMap<u64, PendingToolCall>,
    /// Active tool calls being accumulated via start/delta/end pattern.
    /// Keyed by tool call ID to support multiple concurrent tool calls.
    /// Used for `ToolCallStart` / `ToolCallDelta` / `ToolCallEnd` chunk types.
    ///
    /// Source: TS `NativeToolCallParser` — tracks streaming tool calls by ID
    active_tool_calls: HashMap<String, PendingToolCall>,
    /// Thinking / reasoning blocks.
    thinking_blocks: Vec<ThinkingBlock>,
    /// Current thinking text being accumulated.
    current_thinking: Option<String>,
    /// Current thinking signature (set when thinking_complete is received).
    current_thinking_signature: Option<String>,
    /// Token usage.
    usage: Option<StreamUsage>,
    /// Grounding sources.
    grounding_sources: Vec<GroundingSource>,
    /// Stream error.
    error: Option<StreamError>,
}

impl StreamParser {
    /// Create a new stream parser.
    pub fn new() -> Self {
        Self {
            text: String::new(),
            tool_calls: Vec::new(),
            pending_tool_calls: HashMap::new(),
            active_tool_calls: HashMap::new(),
            thinking_blocks: Vec::new(),
            current_thinking: None,
            current_thinking_signature: None,
            usage: None,
            grounding_sources: Vec::new(),
            error: None,
        }
    }

    /// Feed a single chunk into the parser.
    ///
    /// This method accumulates state incrementally. Call [`finalize()`]
    /// when the stream ends to get the complete parsed content.
    pub fn feed_chunk(&mut self, chunk: &ApiStreamChunk) {
        match chunk {
            ApiStreamChunk::Text { text } => {
                self.text.push_str(text);
            }

            ApiStreamChunk::Usage {
                input_tokens,
                output_tokens,
                cache_write_tokens,
                cache_read_tokens,
                reasoning_tokens,
                total_cost,
            } => {
                self.usage = Some(StreamUsage {
                    input_tokens: *input_tokens,
                    output_tokens: *output_tokens,
                    cache_write_tokens: *cache_write_tokens,
                    cache_read_tokens: *cache_read_tokens,
                    reasoning_tokens: *reasoning_tokens,
                    total_cost: *total_cost,
                });
            }

            ApiStreamChunk::Reasoning { text, signature } => {
                // Accumulate reasoning text
                if let Some(ref mut current) = self.current_thinking {
                    current.push_str(text);
                } else {
                    self.current_thinking = Some(text.clone());
                }
                // If a signature is provided inline, store it
                if let Some(sig) = signature {
                    self.current_thinking_signature = Some(sig.clone());
                }
            }

            ApiStreamChunk::ThinkingComplete { signature } => {
                // Finalize the current thinking block
                if let Some(text) = self.current_thinking.take() {
                    self.thinking_blocks.push(ThinkingBlock {
                        text,
                        signature: Some(signature.clone()),
                    });
                }
                self.current_thinking_signature = None;
            }

            ApiStreamChunk::Grounding { sources } => {
                self.grounding_sources.extend(sources.iter().cloned());
            }

            ApiStreamChunk::ToolCall {
                id,
                name,
                arguments,
            } => {
                self.tool_calls.push(ParsedToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
            }

            ApiStreamChunk::ToolCallStart { id, name } => {
                // Start tracking a new partial tool call using ID-based tracking.
                // This supports multiple concurrent tool calls via start/delta/end pattern.
                //
                // Source: TS `NativeToolCallParser.startStreamingToolCall()` —
                // guards against duplicate start events for the same ID.
                if !self.active_tool_calls.contains_key(id) {
                    self.active_tool_calls.insert(
                        id.clone(),
                        PendingToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            arguments: String::new(),
                        },
                    );
                }
            }

            ApiStreamChunk::ToolCallDelta { id, delta } => {
                // Append the delta to the correct active tool call by ID.
                // This correctly handles multiple interleaved tool calls.
                if let Some(pending) = self.active_tool_calls.get_mut(id) {
                    pending.arguments.push_str(delta);
                }
            }

            ApiStreamChunk::ToolCallEnd { id } => {
                // Finalize the active tool call by ID and move it to completed.
                if let Some(pending) = self.active_tool_calls.remove(id) {
                    self.tool_calls.push(ParsedToolCall {
                        id: pending.id,
                        name: pending.name,
                        arguments: pending.arguments,
                    });
                }
            }

            ApiStreamChunk::ToolCallPartial {
                index,
                id,
                name,
                arguments,
            } => {
                // Handle partial tool calls from providers that use index-based tracking
                let entry = self
                    .pending_tool_calls
                    .entry(*index)
                    .or_insert_with(|| PendingToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: String::new(),
                    });

                if let Some(id) = id {
                    entry.id = id.clone();
                }
                if let Some(name) = name {
                    entry.name = name.clone();
                }
                if let Some(args) = arguments {
                    entry.arguments.push_str(args);
                }
            }

            ApiStreamChunk::Error { error, message } => {
                self.error = Some(StreamError {
                    error: error.clone(),
                    message: message.clone(),
                });
            }
        }
    }

    /// Finalize the stream and return the parsed content.
    ///
    /// This flushes any pending partial tool calls and the current thinking
    /// block, then returns the complete aggregated content.
    pub fn finalize(mut self) -> ParsedStreamContent {
        // Flush pending partial tool calls (index-based tracking)
        for (_, pending) in self.pending_tool_calls.drain() {
            if !pending.id.is_empty() && !pending.name.is_empty() {
                self.tool_calls.push(ParsedToolCall {
                    id: pending.id,
                    name: pending.name,
                    arguments: pending.arguments,
                });
            }
        }

        // Flush active tool calls (ID-based start/delta/end tracking).
        // Source: TS `NativeToolCallParser.finalizeRawChunks()` — finalizes
        // any streaming tool calls that weren't explicitly ended.
        for (_, pending) in self.active_tool_calls.drain() {
            if !pending.id.is_empty() && !pending.name.is_empty() {
                self.tool_calls.push(ParsedToolCall {
                    id: pending.id,
                    name: pending.name,
                    arguments: pending.arguments,
                });
            }
        }

        // Flush current thinking block (if not finalized by ThinkingComplete)
        if let Some(text) = self.current_thinking.take() {
            self.thinking_blocks.push(ThinkingBlock {
                text,
                signature: self.current_thinking_signature.take(),
            });
        }

        ParsedStreamContent {
            text: self.text,
            tool_calls: self.tool_calls,
            thinking_blocks: self.thinking_blocks,
            usage: self.usage,
            grounding_sources: self.grounding_sources,
            error: self.error,
        }
    }

    /// Get the accumulated text so far (without finalizing).
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Check whether any tool calls have been accumulated.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
            || !self.pending_tool_calls.is_empty()
            || !self.active_tool_calls.is_empty()
    }

    /// Get the current usage info (if received).
    pub fn usage(&self) -> Option<&StreamUsage> {
        self.usage.as_ref()
    }

    /// Check whether an error was received.
    pub fn has_error(&self) -> bool {
        self.error.is_some()
    }
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: convert parsed content into API ContentBlocks
// ---------------------------------------------------------------------------

impl ParsedStreamContent {
    /// Convert the parsed content into a vector of [`ContentBlock`] suitable
    /// for adding to the API conversation history as an assistant message.
    pub fn to_content_blocks(&self) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();

        // Thinking blocks
        for thinking in &self.thinking_blocks {
            blocks.push(ContentBlock::Thinking {
                thinking: thinking.text.clone(),
                signature: thinking.signature.clone().unwrap_or_default(),
            });
        }

        // Text block (if non-empty)
        if !self.text.is_empty() {
            blocks.push(ContentBlock::Text {
                text: self.text.clone(),
            });
        }

        // Tool use blocks
        for tc in &self.tool_calls {
            blocks.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input: tc.parse_arguments(),
            });
        }

        blocks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_text_accumulation() {
        let mut parser = StreamParser::new();
        parser.feed_chunk(&ApiStreamChunk::Text {
            text: "Hello".into(),
        });
        parser.feed_chunk(&ApiStreamChunk::Text {
            text: " world".into(),
        });

        assert_eq!(parser.text(), "Hello world");

        let content = parser.finalize();
        assert_eq!(content.text, "Hello world");
        assert!(content.tool_calls.is_empty());
    }

    #[test]
    fn test_parser_complete_tool_call() {
        let mut parser = StreamParser::new();
        parser.feed_chunk(&ApiStreamChunk::Text {
            text: "I'll read that file.".into(),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCall {
            id: "call_123".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"src/main.rs"}"#.into(),
        });

        let content = parser.finalize();
        assert_eq!(content.text, "I'll read that file.");
        assert_eq!(content.tool_calls.len(), 1);
        assert_eq!(content.tool_calls[0].id, "call_123");
        assert_eq!(content.tool_calls[0].name, "read_file");
        assert_eq!(
            content.tool_calls[0].parse_arguments(),
            serde_json::json!({"path": "src/main.rs"})
        );
    }

    #[test]
    fn test_parser_partial_tool_call() {
        let mut parser = StreamParser::new();

        // First partial: id + name
        parser.feed_chunk(&ApiStreamChunk::ToolCallPartial {
            index: 0,
            id: Some("call_1".into()),
            name: Some("write_to_file".into()),
            arguments: None,
        });

        // Second partial: arguments delta
        parser.feed_chunk(&ApiStreamChunk::ToolCallPartial {
            index: 0,
            id: None,
            name: None,
            arguments: Some(r#"{"path":"test.rs","#.into()),
        });

        // Third partial: more arguments
        parser.feed_chunk(&ApiStreamChunk::ToolCallPartial {
            index: 0,
            id: None,
            name: None,
            arguments: Some(r#""content":"hello"}"#.into()),
        });

        assert!(parser.has_tool_calls());

        let content = parser.finalize();
        assert_eq!(content.tool_calls.len(), 1);
        assert_eq!(content.tool_calls[0].name, "write_to_file");
        assert_eq!(
            content.tool_calls[0].arguments,
            r#"{"path":"test.rs","content":"hello"}"#
        );
    }

    #[test]
    fn test_parser_start_delta_end_tool_call() {
        let mut parser = StreamParser::new();

        parser.feed_chunk(&ApiStreamChunk::ToolCallStart {
            id: "tc_1".into(),
            name: "apply_diff".into(),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallDelta {
            id: "tc_1".into(),
            delta: r#"{"path":"a.rs"}"#.into(),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallEnd {
            id: "tc_1".into(),
        });

        let content = parser.finalize();
        assert_eq!(content.tool_calls.len(), 1);
        assert_eq!(content.tool_calls[0].name, "apply_diff");
        assert_eq!(content.tool_calls[0].arguments, r#"{"path":"a.rs"}"#);
    }

    #[test]
    fn test_parser_multiple_tool_calls() {
        let mut parser = StreamParser::new();

        // Two interleaved partial tool calls
        parser.feed_chunk(&ApiStreamChunk::ToolCallPartial {
            index: 0,
            id: Some("call_a".into()),
            name: Some("read_file".into()),
            arguments: None,
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallPartial {
            index: 1,
            id: Some("call_b".into()),
            name: Some("list_files".into()),
            arguments: None,
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallPartial {
            index: 0,
            id: None,
            name: None,
            arguments: Some(r#"{"path":"a.rs"}"#.into()),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallPartial {
            index: 1,
            id: None,
            name: None,
            arguments: Some(r#"{"path":".","recursive":true}"#.into()),
        });

        let content = parser.finalize();
        assert_eq!(content.tool_calls.len(), 2);

        // Note: order from HashMap may vary, so check by name
        let names: Vec<&str> = content.tool_calls.iter().map(|tc| tc.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"list_files"));
    }

    #[test]
    fn test_parser_usage() {
        let mut parser = StreamParser::new();
        parser.feed_chunk(&ApiStreamChunk::Usage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_write_tokens: Some(100),
            cache_read_tokens: Some(50),
            reasoning_tokens: None,
            total_cost: Some(0.05),
        });

        let content = parser.finalize();
        let usage = content.usage.unwrap();
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.output_tokens, 500);
        assert_eq!(usage.cache_write_tokens, Some(100));
        assert_eq!(usage.cache_read_tokens, Some(50));
        assert_eq!(usage.total_cost, Some(0.05));
    }

    #[test]
    fn test_parser_thinking_blocks() {
        let mut parser = StreamParser::new();

        parser.feed_chunk(&ApiStreamChunk::Reasoning {
            text: "Let me think".into(),
            signature: None,
        });
        parser.feed_chunk(&ApiStreamChunk::Reasoning {
            text: " about this...".into(),
            signature: None,
        });
        parser.feed_chunk(&ApiStreamChunk::ThinkingComplete {
            signature: "sig_123".into(),
        });

        let content = parser.finalize();
        assert_eq!(content.thinking_blocks.len(), 1);
        assert_eq!(content.thinking_blocks[0].text, "Let me think about this...");
        assert_eq!(
            content.thinking_blocks[0].signature,
            Some("sig_123".to_string())
        );
    }

    #[test]
    fn test_parser_thinking_without_complete() {
        let mut parser = StreamParser::new();
        parser.feed_chunk(&ApiStreamChunk::Reasoning {
            text: "Partial thought".into(),
            signature: None,
        });

        let content = parser.finalize();
        assert_eq!(content.thinking_blocks.len(), 1);
        assert_eq!(content.thinking_blocks[0].text, "Partial thought");
        assert!(content.thinking_blocks[0].signature.is_none());
    }

    #[test]
    fn test_parser_error() {
        let mut parser = StreamParser::new();
        parser.feed_chunk(&ApiStreamChunk::Error {
            error: "rate_limit".into(),
            message: "Rate limit exceeded".into(),
        });

        assert!(parser.has_error());
        let content = parser.finalize();
        assert!(content.error.is_some());
        assert_eq!(content.error.unwrap().error, "rate_limit");
    }

    #[test]
    fn test_parser_grounding() {
        let mut parser = StreamParser::new();
        parser.feed_chunk(&ApiStreamChunk::Grounding {
            sources: vec![
                GroundingSource {
                    title: Some("Doc".into()),
                    url: Some("https://example.com".into()),
                    snippet: None,
                },
            ],
        });

        let content = parser.finalize();
        assert_eq!(content.grounding_sources.len(), 1);
        assert_eq!(
            content.grounding_sources[0].url,
            Some("https://example.com".into())
        );
    }

    #[test]
    fn test_to_content_blocks() {
        let mut parser = StreamParser::new();
        parser.feed_chunk(&ApiStreamChunk::Reasoning {
            text: "hmm".into(),
            signature: None,
        });
        parser.feed_chunk(&ApiStreamChunk::ThinkingComplete {
            signature: "sig".into(),
        });
        parser.feed_chunk(&ApiStreamChunk::Text {
            text: "Hello".into(),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCall {
            id: "c1".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"x.rs"}"#.into(),
        });

        let content = parser.finalize();
        let blocks = content.to_content_blocks();

        assert_eq!(blocks.len(), 3);

        // Thinking block
        match &blocks[0] {
            ContentBlock::Thinking { thinking, signature } => {
                assert_eq!(thinking, "hmm");
                assert_eq!(signature, "sig");
            }
            _ => panic!("Expected Thinking block"),
        }

        // Text block
        match &blocks[1] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text block"),
        }

        // ToolUse block
        match &blocks[2] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "c1");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "x.rs");
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_parser_empty_stream() {
        let parser = StreamParser::new();
        let content = parser.finalize();
        assert!(content.text.is_empty());
        assert!(content.tool_calls.is_empty());
        assert!(content.thinking_blocks.is_empty());
        assert!(content.usage.is_none());
        assert!(content.error.is_none());
    }

    #[test]
    fn test_parse_arguments_invalid_json() {
        let tc = ParsedToolCall {
            id: "c1".into(),
            name: "test".into(),
            arguments: "not json".into(),
        };
        assert_eq!(tc.parse_arguments(), serde_json::Value::Null);
    }

    // --- Multi-tool start/delta/end concurrency tests ---

    #[test]
    fn test_parser_multiple_concurrent_start_delta_end() {
        // Two interleaved tool calls using start/delta/end pattern.
        // This tests the fix for the bug where all calls used key 0.
        let mut parser = StreamParser::new();

        // Start first tool call
        parser.feed_chunk(&ApiStreamChunk::ToolCallStart {
            id: "tc_a".into(),
            name: "read_file".into(),
        });
        // Start second tool call (interleaved)
        parser.feed_chunk(&ApiStreamChunk::ToolCallStart {
            id: "tc_b".into(),
            name: "list_files".into(),
        });
        // Delta for first tool
        parser.feed_chunk(&ApiStreamChunk::ToolCallDelta {
            id: "tc_a".into(),
            delta: r#"{"path":"a.rs"}"#.into(),
        });
        // Delta for second tool
        parser.feed_chunk(&ApiStreamChunk::ToolCallDelta {
            id: "tc_b".into(),
            delta: r#"{"path":".","recursive":true}"#.into(),
        });
        // End first tool
        parser.feed_chunk(&ApiStreamChunk::ToolCallEnd {
            id: "tc_a".into(),
        });
        // End second tool
        parser.feed_chunk(&ApiStreamChunk::ToolCallEnd {
            id: "tc_b".into(),
        });

        let content = parser.finalize();
        assert_eq!(content.tool_calls.len(), 2);

        let names: Vec<&str> = content.tool_calls.iter().map(|tc| tc.name.as_str()).collect();
        assert!(names.contains(&"read_file"), "Expected read_file in {:?}", names);
        assert!(names.contains(&"list_files"), "Expected list_files in {:?}", names);

        // Verify arguments are correct for each tool
        for tc in &content.tool_calls {
            if tc.name == "read_file" {
                assert_eq!(tc.id, "tc_a");
                assert_eq!(tc.arguments, r#"{"path":"a.rs"}"#);
            } else if tc.name == "list_files" {
                assert_eq!(tc.id, "tc_b");
                assert_eq!(tc.arguments, r#"{"path":".","recursive":true}"#);
            }
        }
    }

    #[test]
    fn test_parser_duplicate_tool_call_start_ignored() {
        // Duplicate start events for the same ID should be ignored,
        // matching TS behavior that guards against duplicate tool_call_start.
        let mut parser = StreamParser::new();

        parser.feed_chunk(&ApiStreamChunk::ToolCallStart {
            id: "tc_1".into(),
            name: "read_file".into(),
        });
        // Duplicate start (should be ignored)
        parser.feed_chunk(&ApiStreamChunk::ToolCallStart {
            id: "tc_1".into(),
            name: "read_file".into(),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallDelta {
            id: "tc_1".into(),
            delta: r#"{"path":"x.rs"}"#.into(),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallEnd {
            id: "tc_1".into(),
        });

        let content = parser.finalize();
        assert_eq!(content.tool_calls.len(), 1);
        assert_eq!(content.tool_calls[0].arguments, r#"{"path":"x.rs"}"#);
    }

    #[test]
    fn test_parser_unfinalized_active_tool_call_flushed() {
        // Active tool calls that weren't explicitly ended should be flushed
        // in finalize(), matching TS `finalizeRawChunks()` behavior.
        let mut parser = StreamParser::new();

        parser.feed_chunk(&ApiStreamChunk::ToolCallStart {
            id: "tc_unfinished".into(),
            name: "write_to_file".into(),
        });
        parser.feed_chunk(&ApiStreamChunk::ToolCallDelta {
            id: "tc_unfinished".into(),
            delta: r#"{"path":"a.rs"}"#.into(),
        });
        // No ToolCallEnd — stream ended prematurely

        let content = parser.finalize();
        assert_eq!(content.tool_calls.len(), 1);
        assert_eq!(content.tool_calls[0].name, "write_to_file");
        assert_eq!(content.tool_calls[0].id, "tc_unfinished");
    }
}