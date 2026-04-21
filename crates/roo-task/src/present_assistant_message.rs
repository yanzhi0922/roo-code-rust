//! Core message presentation state machine.
//!
//! Processes and presents assistant message content blocks sequentially.
//! Handles text display, tool call execution with approval, MCP tool routing,
//! and manages the flow of conversation.
//!
//! Source: `src/core/assistant-message/presentAssistantMessage.ts`


use serde_json::Value;

use crate::types::{AssistantMessageContent, McpToolUse, ToolUse};

// ---------------------------------------------------------------------------
// PresentAssistantMessageState
// ---------------------------------------------------------------------------

/// Mutable state for the present-assistant-message state machine.
///
/// This mirrors the TS `cline` properties used within `presentAssistantMessage`:
/// - `presentAssistantMessageLocked`
/// - `presentAssistantMessageHasPendingUpdates`
/// - `currentStreamingContentIndex`
/// - `didCompleteReadingStream`
/// - `didRejectTool`
/// - `didAlreadyUseTool`
/// - `userMessageContentReady`
/// - `currentStreamingDidCheckpoint`
/// - `assistantMessageContent`
/// - `userMessageContent`
/// - `consecutiveMistakeCount`
/// - `consecutiveMistakeLimit`
#[derive(Debug)]
pub struct PresentAssistantMessageState {
    /// Whether the state machine is currently locked (processing a block).
    pub locked: bool,
    /// Whether there are pending updates to process.
    pub has_pending_updates: bool,
    /// Index of the current content block being streamed.
    pub current_streaming_content_index: usize,
    /// Whether the stream has finished reading.
    pub did_complete_reading_stream: bool,
    /// Whether the user rejected a tool in this cycle.
    pub did_reject_tool: bool,
    /// Whether a tool has already been used (prevents double execution).
    pub did_already_use_tool: bool,
    /// Whether the user message content is ready for the next API call.
    pub user_message_content_ready: bool,
    /// Whether a checkpoint has been saved for the current streaming block.
    pub current_streaming_did_checkpoint: bool,
    /// The accumulated assistant message content blocks.
    pub assistant_message_content: Vec<AssistantMessageContent>,
    /// User message content accumulated during processing (tool results, feedback).
    pub user_message_content: Vec<Value>,
    /// Consecutive mistake counter.
    pub consecutive_mistake_count: usize,
    /// Limit for consecutive mistakes before stopping.
    pub consecutive_mistake_limit: usize,
}

impl Default for PresentAssistantMessageState {
    fn default() -> Self {
        Self {
            locked: false,
            has_pending_updates: false,
            current_streaming_content_index: 0,
            did_complete_reading_stream: false,
            did_reject_tool: false,
            did_already_use_tool: false,
            user_message_content_ready: false,
            current_streaming_did_checkpoint: false,
            assistant_message_content: Vec::new(),
            user_message_content: Vec::new(),
            consecutive_mistake_count: 0,
            consecutive_mistake_limit: 5,
        }
    }
}

impl PresentAssistantMessageState {
    /// Create a new state with the given mistake limit.
    pub fn new(consecutive_mistake_limit: usize) -> Self {
        Self {
            consecutive_mistake_limit,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// ToolResult
// ---------------------------------------------------------------------------

/// The result of a tool execution, to be pushed back as a tool_result.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The text content of the result.
    pub content: String,
    /// Optional image blocks (base64 encoded).
    pub images: Vec<ImageBlock>,
    /// Whether this result represents an error.
    pub is_error: bool,
}

/// An image block in a tool result.
#[derive(Debug, Clone)]
pub struct ImageBlock {
    /// Base64-encoded image data.
    pub data: String,
    /// MIME type of the image.
    pub media_type: String,
}

impl ToolResult {
    /// Create a successful tool result.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            is_error: false,
        }
    }

    /// Create a successful tool result with images.
    pub fn success_with_images(content: impl Into<String>, images: Vec<ImageBlock>) -> Self {
        Self {
            content: content.into(),
            images,
            is_error: false,
        }
    }

    /// Create an error tool result.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            is_error: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ApprovalFeedback
// ---------------------------------------------------------------------------

/// Feedback provided by the user when approving/denying a tool.
#[derive(Debug, Clone)]
pub struct ApprovalFeedback {
    /// The user's text feedback.
    pub text: String,
    /// Optional images provided with the feedback.
    pub images: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// ToolCallbacks
// ---------------------------------------------------------------------------

/// Callbacks provided to tool handlers during execution.
///
/// Mirrors the TS pattern where each tool receives `askApproval`, `handleError`,
/// and `pushToolResult` closures.
#[derive(Debug)]
pub struct ToolCallbacks {
    /// The tool call ID this set of callbacks is associated with.
    pub tool_call_id: String,
    /// Whether a tool result has already been pushed (prevents duplicates).
    pub has_tool_result: bool,
    /// Stored approval feedback to merge into tool result.
    pub approval_feedback: Option<ApprovalFeedback>,
}

impl ToolCallbacks {
    /// Create new callbacks for the given tool call ID.
    pub fn new(tool_call_id: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            has_tool_result: false,
            approval_feedback: None,
        }
    }

    /// Push a tool result, preventing duplicates.
    /// Returns `true` if the result was pushed, `false` if a duplicate was skipped.
    pub fn push_tool_result(&mut self, _result: &ToolResult) -> bool {
        if self.has_tool_result {
            tracing::warn!(
                "[presentAssistantMessage] Skipping duplicate tool_result for tool_use_id: {}",
                self.tool_call_id
            );
            return false;
        }
        self.has_tool_result = true;
        true
    }

    /// Record approval feedback.
    pub fn set_approval_feedback(&mut self, feedback: ApprovalFeedback) {
        self.approval_feedback = Some(feedback);
    }
}

// ---------------------------------------------------------------------------
// BlockProcessingResult
// ---------------------------------------------------------------------------

/// Result of processing a single content block.
#[derive(Debug, Clone)]
pub enum BlockProcessingResult {
    /// The block was processed successfully (non-partial).
    Completed,
    /// The block is still being streamed (partial).
    Partial,
    /// The block was skipped because the user rejected a previous tool.
    Rejected,
    /// The block was skipped because a tool was already used.
    AlreadyUsedTool,
    /// An error occurred during processing.
    Error(String),
}

// ---------------------------------------------------------------------------
// PresentAssistantMessageError
// ---------------------------------------------------------------------------

/// Errors that can occur during message presentation.
#[derive(Debug, thiserror::Error)]
pub enum PresentAssistantMessageError {
    /// The task was aborted.
    #[error("Task aborted")]
    Aborted,

    /// The task is locked (concurrent execution attempt).
    #[error("Task locked, pending updates queued")]
    Locked,

    /// Content index out of bounds.
    #[error("Content index out of bounds")]
    OutOfBounds,

    /// Invalid tool call (missing ID).
    #[error("Invalid tool call: {0}")]
    InvalidToolCall(String),

    /// Tool validation failed.
    #[error("Tool validation failed: {0}")]
    ValidationFailed(String),

    /// Tool repetition limit reached.
    #[error("Tool repetition limit reached for {0}")]
    RepetitionLimit(String),

    /// Unknown tool name.
    #[error("Unknown tool: {0}")]
    UnknownTool(String),

    /// MCP tool execution error.
    #[error("MCP tool error: {0}")]
    McpToolError(String),

    /// Custom tool execution error.
    #[error("Custom tool error: {0}")]
    CustomToolError(String),

    /// General processing error.
    #[error("{0}")]
    General(String),
}

// ---------------------------------------------------------------------------
// PresentAssistantMessage (the state machine)
// ---------------------------------------------------------------------------

/// The core message presentation state machine.
///
/// This struct manages the sequential processing of assistant message content
/// blocks, handling text display, tool call validation, approval flows, and
/// state transitions.
///
/// Source: `src/core/assistant-message/presentAssistantMessage.ts`
pub struct PresentAssistantMessage {
    /// The mutable state.
    state: PresentAssistantMessageState,
}

impl PresentAssistantMessage {
    /// Create a new state machine with default settings.
    pub fn new() -> Self {
        Self {
            state: PresentAssistantMessageState::default(),
        }
    }

    /// Create a new state machine with a specific mistake limit.
    pub fn with_mistake_limit(limit: usize) -> Self {
        Self {
            state: PresentAssistantMessageState::new(limit),
        }
    }

    /// Get a reference to the current state.
    pub fn state(&self) -> &PresentAssistantMessageState {
        &self.state
    }

    /// Get a mutable reference to the current state.
    pub fn state_mut(&mut self) -> &mut PresentAssistantMessageState {
        &mut self.state
    }

    // -----------------------------------------------------------------------
    // Core entry point — mirrors the TS `presentAssistantMessage(cline)` function
    // -----------------------------------------------------------------------

    /// Process the current content block.
    ///
    /// This is the Rust equivalent of the TS `presentAssistantMessage` function.
    /// It:
    /// 1. Checks if the task is aborted
    /// 2. Acquires the lock (or queues updates if already locked)
    /// 3. Processes the current content block based on its type
    /// 4. Updates state and advances to the next block
    ///
    /// Returns the processing result for the current block.
    pub fn present(
        &mut self,
        is_aborted: bool,
    ) -> Result<Option<BlockProcessingResult>, PresentAssistantMessageError> {
        // Step 1: Check abort
        if is_aborted {
            return Err(PresentAssistantMessageError::Aborted);
        }

        // Step 2: Lock check
        if self.state.locked {
            self.state.has_pending_updates = true;
            return Ok(None);
        }

        self.state.locked = true;
        self.state.has_pending_updates = false;

        // Step 3: Bounds check
        if self.state.current_streaming_content_index >= self.state.assistant_message_content.len() {
            if self.state.did_complete_reading_stream {
                self.state.user_message_content_ready = true;
            }
            self.state.locked = false;
            return Ok(None);
        }

        // Step 4: Get the current block (shallow copy for safety)
        let block = self.state.assistant_message_content
            [self.state.current_streaming_content_index]
            .clone();

        // Step 5: Process based on block type
        let result = self.process_block(&block);

        // Step 6: Unlock
        self.state.locked = false;

        // Step 7: Handle post-processing state transitions
        self.handle_post_processing(&block, &result);

        Ok(Some(result))
    }

    // -----------------------------------------------------------------------
    // Block processing
    // -----------------------------------------------------------------------

    /// Process a single content block based on its type.
    fn process_block(
        &mut self,
        block: &AssistantMessageContent,
    ) -> BlockProcessingResult {
        match block {
            AssistantMessageContent::McpToolUse { .. } => {
                // Extract the McpToolUse from the enum
                if let AssistantMessageContent::McpToolUse(mcp_block) = block {
                    self.process_mcp_tool_use(mcp_block)
                } else {
                    unreachable!()
                }
            }
            AssistantMessageContent::Text { .. } => {
                if let AssistantMessageContent::Text { content, partial } = block {
                    self.process_text(content, *partial)
                } else {
                    unreachable!()
                }
            }
            AssistantMessageContent::ToolUse { .. } => {
                if let AssistantMessageContent::ToolUse(tool_block) = block {
                    self.process_tool_use(tool_block)
                } else {
                    unreachable!()
                }
            }
        }
    }

    /// Process an MCP tool use block.
    ///
    /// Mirrors the TS `case "mcp_tool_use"` branch.
    fn process_mcp_tool_use(
        &mut self,
        block: &McpToolUse,
    ) -> BlockProcessingResult {
        // If user rejected a previous tool, skip this one too
        if self.state.did_reject_tool {
            let error_message = if !block.partial {
                format!("Skipping MCP tool {}/{} due to user rejecting a previous tool.", block.server_name, block.tool_name)
            } else {
                format!("MCP tool {}/{} was interrupted and not executed due to user rejecting a previous tool.", block.server_name, block.tool_name)
            };

            if !block.id.is_empty() {
                self.push_tool_result_to_user_content(
                    sanitize_tool_use_id(&block.id),
                    error_message,
                    true,
                );
            }
            return BlockProcessingResult::Rejected;
        }

        // Only record usage for complete blocks
        if !block.partial {
            tracing::debug!(
                tool = "use_mcp_tool",
                server = %block.server_name,
                tool_name = %block.tool_name,
                "Recording MCP tool usage"
            );
        }

        if block.partial {
            BlockProcessingResult::Partial
        } else {
            BlockProcessingResult::Completed
        }
    }

    /// Process a text content block.
    ///
    /// Mirrors the TS `case "text"` branch.
    fn process_text(
        &mut self,
        content: &str,
        partial: bool,
    ) -> BlockProcessingResult {
        if self.state.did_reject_tool || self.state.did_already_use_tool {
            return BlockProcessingResult::AlreadyUsedTool;
        }

        if !content.is_empty() {
            // Strip <thinking> tags from text output
            let _cleaned = strip_thinking_tags(content);
            // In the TS version, this calls `cline.say("text", cleaned, undefined, partial)`
            // Here we just return the result; the actual UI communication is handled externally
        }

        if partial {
            BlockProcessingResult::Partial
        } else {
            BlockProcessingResult::Completed
        }
    }

    /// Process a tool use block.
    ///
    /// Mirrors the TS `case "tool_use"` branch.
    fn process_tool_use(
        &mut self,
        block: &ToolUse,
    ) -> BlockProcessingResult {
        // A tool_use block without an ID is invalid
        if block.id.is_empty() {
            let error_message = "Invalid tool call: missing tool_use.id. XML tool calls are no longer supported. Remove any XML tool markup and use native tool calling instead.".to_string();
            self.state.consecutive_mistake_count += 1;
            self.state.user_message_content.push(serde_json::json!({
                "type": "text",
                "text": error_message,
            }));
            self.state.did_already_use_tool = true;
            return BlockProcessingResult::Error(error_message);
        }

        let tool_call_id = &block.id;

        // If user rejected a previous tool, skip this one
        if self.state.did_reject_tool {
            let tool_desc = self.get_tool_description(block);
            let error_message = if !block.partial {
                format!("Skipping tool {} due to user rejecting a previous tool.", tool_desc)
            } else {
                format!("Tool {} was interrupted and not executed due to user rejecting a previous tool.", tool_desc)
            };

            self.push_tool_result_to_user_content(
                sanitize_tool_use_id(tool_call_id),
                error_message,
                true,
            );
            return BlockProcessingResult::Rejected;
        }

        // Check for missing nativeArgs on complete blocks
        if !block.partial {
            if block.native_args.is_none() {
                let error_message = format!(
                    "Invalid tool call for '{}': missing nativeArgs. This usually means the model streamed invalid or incomplete arguments.",
                    block.name
                );
                self.state.consecutive_mistake_count += 1;

                self.push_tool_result_to_user_content(
                    sanitize_tool_use_id(tool_call_id),
                    format_tool_error(&error_message),
                    true,
                );
                return BlockProcessingResult::Error(error_message);
            }
        }

        // Record tool usage for complete blocks
        if !block.partial {
            tracing::debug!(
                tool = %block.name,
                "Recording tool usage"
            );
        }

        if block.partial {
            BlockProcessingResult::Partial
        } else {
            // Reset mistake count on successful tool processing
            self.state.consecutive_mistake_count = 0;
            BlockProcessingResult::Completed
        }
    }

    // -----------------------------------------------------------------------
    // Tool description
    // -----------------------------------------------------------------------

    /// Get a human-readable description for a tool use block.
    ///
    /// Mirrors the TS `toolDescription()` function.
    fn get_tool_description(&self, block: &ToolUse) -> String {
        let get_param = |key: &str| -> Option<&str> {
            block.params.get(key).map(|s| s.as_str())
        };

        match block.name.as_str() {
            "execute_command" => {
                match get_param("command") {
                    Some(cmd) => format!("[execute_command for '{}']", cmd),
                    None => "[execute_command]".to_string(),
                }
            }
            "read_file" => {
                let path = block.native_args
                    .as_ref()
                    .and_then(|a| a.get("path"))
                    .and_then(|v| v.as_str())
                    .or_else(|| get_param("path"))
                    .unwrap_or("?");
                format!("[read_file for '{}']", path)
            }
            "write_to_file" => {
                match get_param("path") {
                    Some(p) => format!("[write_to_file for '{}']", p),
                    None => "[write_to_file]".to_string(),
                }
            }
            "apply_diff" => {
                match get_param("path") {
                    Some(p) => format!("[apply_diff for '{}']", p),
                    None => "[apply_diff]".to_string(),
                }
            }
            "search_files" => {
                let regex = get_param("regex").unwrap_or("?");
                match get_param("file_pattern") {
                    Some(fp) => format!("[search_files for '{}' in '{}']", regex, fp),
                    None => format!("[search_files for '{}']", regex),
                }
            }
            "edit" | "search_and_replace" => {
                match get_param("file_path") {
                    Some(fp) => format!("[{} for '{}']", block.name, fp),
                    None => format!("[{}]", block.name),
                }
            }
            "search_replace" => {
                match get_param("file_path") {
                    Some(fp) => format!("[search_replace for '{}']", fp),
                    None => "[search_replace]".to_string(),
                }
            }
            "edit_file" => {
                match get_param("file_path") {
                    Some(fp) => format!("[edit_file for '{}']", fp),
                    None => "[edit_file]".to_string(),
                }
            }
            "apply_patch" => "[apply_patch]".to_string(),
            "list_files" => {
                match get_param("path") {
                    Some(p) => format!("[list_files for '{}']", p),
                    None => "[list_files]".to_string(),
                }
            }
            "use_mcp_tool" => {
                match get_param("server_name") {
                    Some(sn) => format!("[use_mcp_tool for '{}']", sn),
                    None => "[use_mcp_tool]".to_string(),
                }
            }
            "access_mcp_resource" => {
                match get_param("server_name") {
                    Some(sn) => format!("[access_mcp_resource for '{}']", sn),
                    None => "[access_mcp_resource]".to_string(),
                }
            }
            "ask_followup_question" => {
                match get_param("question") {
                    Some(q) => format!("[ask_followup_question for '{}']", q),
                    None => "[ask_followup_question]".to_string(),
                }
            }
            "attempt_completion" => "[attempt_completion]".to_string(),
            "switch_mode" => {
                let mode = get_param("mode_slug").unwrap_or("?");
                match get_param("reason") {
                    Some(r) => format!("[switch_mode to '{}' because: {}]", mode, r),
                    None => format!("[switch_mode to '{}']", mode),
                }
            }
            "codebase_search" => {
                match get_param("query") {
                    Some(q) => format!("[codebase_search for '{}']", q),
                    None => "[codebase_search]".to_string(),
                }
            }
            "read_command_output" => {
                match get_param("artifact_id") {
                    Some(a) => format!("[read_command_output for '{}']", a),
                    None => "[read_command_output]".to_string(),
                }
            }
            "update_todo_list" => "[update_todo_list]".to_string(),
            "new_task" => {
                let mode = get_param("mode").unwrap_or("architect");
                let message = get_param("message").unwrap_or("(no message)");
                format!("[new_task in {} mode: '{}']", mode, message)
            }
            "run_slash_command" => {
                let cmd = get_param("command").unwrap_or("?");
                match get_param("args") {
                    Some(a) => format!("[run_slash_command for '{}' with args: {}]", cmd, a),
                    None => format!("[run_slash_command for '{}']", cmd),
                }
            }
            "skill" => {
                let skill = get_param("skill").unwrap_or("?");
                match get_param("args") {
                    Some(a) => format!("[skill for '{}' with args: {}]", skill, a),
                    None => format!("[skill for '{}']", skill),
                }
            }
            "generate_image" => {
                match get_param("path") {
                    Some(p) => format!("[generate_image for '{}']", p),
                    None => "[generate_image]".to_string(),
                }
            }
            _ => format!("[{}]", block.name),
        }
    }

    // -----------------------------------------------------------------------
    // Post-processing state transitions
    // -----------------------------------------------------------------------

    /// Handle state transitions after processing a block.
    ///
    /// Mirrors the TS post-processing logic at the end of `presentAssistantMessage`.
    fn handle_post_processing(
        &mut self,
        block: &AssistantMessageContent,
        _result: &BlockProcessingResult,
    ) {
        let is_partial = block.is_partial();

        // If block is finished (non-partial) or was rejected/already-used
        if !is_partial || self.state.did_reject_tool || self.state.did_already_use_tool {
            // If this is the last block, mark content as ready
            if self.state.current_streaming_content_index
                == self.state.assistant_message_content.len() - 1
            {
                self.state.user_message_content_ready = true;
            }

            // Advance to next block
            self.state.current_streaming_content_index += 1;
        }

        // If block is partial but stream has pending updates, caller should call present() again
    }

    // -----------------------------------------------------------------------
    // Utility methods
    // -----------------------------------------------------------------------

    /// Push a tool result to user message content.
    ///
    /// Mirrors the TS `cline.pushToolResultToUserContent()`.
    pub fn push_tool_result_to_user_content(
        &mut self,
        tool_use_id: String,
        content: String,
        is_error: bool,
    ) {
        self.state.user_message_content.push(serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }));
    }

    /// Reset the state for a new message cycle.
    pub fn reset_for_new_message(&mut self) {
        self.state.current_streaming_content_index = 0;
        self.state.did_complete_reading_stream = false;
        self.state.did_reject_tool = false;
        self.state.did_already_use_tool = false;
        self.state.user_message_content_ready = false;
        self.state.current_streaming_did_checkpoint = false;
        self.state.user_message_content.clear();
        self.state.locked = false;
        self.state.has_pending_updates = false;
    }

    /// Set the assistant message content and reset streaming state.
    pub fn set_assistant_message_content(&mut self, content: Vec<AssistantMessageContent>) {
        self.state.assistant_message_content = content;
        self.state.current_streaming_content_index = 0;
        self.state.did_complete_reading_stream = false;
        self.state.did_reject_tool = false;
        self.state.did_already_use_tool = false;
        self.state.user_message_content_ready = false;
        self.state.current_streaming_did_checkpoint = false;
    }

    /// Mark the stream as complete.
    pub fn mark_stream_complete(&mut self) {
        self.state.did_complete_reading_stream = true;
    }

    /// Check if the state machine needs checkpointing before file-modifying tools.
    pub fn needs_checkpoint(&self) -> bool {
        !self.state.current_streaming_did_checkpoint
    }

    /// Mark that a checkpoint has been saved.
    pub fn mark_checkpoint_saved(&mut self) {
        self.state.current_streaming_did_checkpoint = true;
    }
}

impl Default for PresentAssistantMessage {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Strip `<thinking>` tags from text content.
///
/// Mirrors the TS regex replacements:
/// ```ts
/// content = content.replace(/<thinking>\s?/g, "")
/// content = content.replace(/\s?<\/thinking>/g, "")
/// ```
pub fn strip_thinking_tags(content: &str) -> String {
    let re_open = regex::Regex::new(r"<thinking>\s?").unwrap();
    let re_close = regex::Regex::new(r"\s?</thinking>").unwrap();
    let result = re_open.replace_all(content, "");
    re_close.replace_all(&result, "").into_owned()
}

/// Sanitize a tool use ID to ensure it's valid.
///
/// Mirrors the TS `sanitizeToolUseId()` function.
pub fn sanitize_tool_use_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Format a tool error message.
///
/// Mirrors the TS `formatResponse.toolError()`.
pub fn format_tool_error(message: &str) -> String {
    format!("Error: {}", message)
}

/// Format a tool denied message.
///
/// Mirrors the TS `formatResponse.toolDenied()`.
pub fn format_tool_denied() -> String {
    "Tool execution was denied by user.".to_string()
}

/// Format a tool denied with feedback message.
///
/// Mirrors the TS `formatResponse.toolDeniedWithFeedback()`.
pub fn format_tool_denied_with_feedback(feedback: &str) -> String {
    format!("Tool execution was denied. User feedback: {}", feedback)
}

/// Format a tool approved with feedback message.
///
/// Mirrors the TS `formatResponse.toolApprovedWithFeedback()`.
pub fn format_tool_approved_with_feedback(feedback: &str) -> String {
    format!("Tool approved with feedback: {}", feedback)
}

/// Format a tool result.
///
/// Mirrors the TS `formatResponse.toolResult()`.
pub fn format_tool_result(text: &str, images: Option<&[String]>) -> String {
    let mut result = text.to_string();
    if let Some(imgs) = images {
        if !imgs.is_empty() {
            result.push_str(&format!("\n\n[{} images attached]", imgs.len()));
        }
    }
    result
}

/// Check if a tool name is a file-modifying tool that needs checkpointing.
///
/// Mirrors the TS `checkpointSaveAndMark()` usage pattern.
pub fn is_file_modifying_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "write_to_file"
            | "apply_diff"
            | "edit"
            | "search_and_replace"
            | "edit_file"
            | "apply_patch"
            | "new_task"
            | "generate_image"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_text_block(content: &str, partial: bool) -> AssistantMessageContent {
        AssistantMessageContent::Text {
            content: content.to_string(),
            partial,
        }
    }

    fn make_tool_block(
        id: &str,
        name: &str,
        params: HashMap<String, String>,
        partial: bool,
    ) -> AssistantMessageContent {
        let mut native_args = serde_json::Map::new();
        for (k, v) in &params {
            native_args.insert(k.clone(), Value::String(v.clone()));
        }
        AssistantMessageContent::ToolUse(ToolUse {
            content_type: "tool_use".to_string(),
            name: name.to_string(),
            params,
            partial,
            id: id.to_string(),
            native_args: Some(Value::Object(native_args)),
            original_name: None,
            used_legacy_format: false,
        })
    }

    // ---- Test 1: Basic text block processing ----
    #[test]
    fn test_text_block_processing() {
        let mut pam = PresentAssistantMessage::new();
        pam.set_assistant_message_content(vec![make_text_block("Hello world", false)]);

        let result = pam.present(false).unwrap();
        assert!(matches!(result, Some(BlockProcessingResult::Completed)));
    }

    // ---- Test 2: Partial text block ----
    #[test]
    fn test_partial_text_block() {
        let mut pam = PresentAssistantMessage::new();
        pam.set_assistant_message_content(vec![make_text_block("Hello...", true)]);

        let result = pam.present(false).unwrap();
        assert!(matches!(result, Some(BlockProcessingResult::Partial)));
    }

    // ---- Test 3: Lock mechanism ----
    #[test]
    fn test_lock_mechanism() {
        let mut pam = PresentAssistantMessage::new();
        pam.set_assistant_message_content(vec![make_text_block("Hello", false)]);
        pam.state.locked = true;

        let result = pam.present(false).unwrap();
        assert!(result.is_none());
        assert!(pam.state.has_pending_updates);
    }

    // ---- Test 4: Abort check ----
    #[test]
    fn test_abort_check() {
        let mut pam = PresentAssistantMessage::new();
        pam.set_assistant_message_content(vec![make_text_block("Hello", false)]);

        let result = pam.present(true);
        assert!(matches!(result, Err(PresentAssistantMessageError::Aborted)));
    }

    // ---- Test 5: Tool use block with missing ID ----
    #[test]
    fn test_tool_use_missing_id() {
        let mut pam = PresentAssistantMessage::new();
        let tool_block = ToolUse {
            content_type: "tool_use".to_string(),
            name: "read_file".to_string(),
            params: HashMap::new(),
            native_args: None,
            partial: false,
            id: String::new(),
            original_name: None,
            used_legacy_format: false,
        };
        pam.set_assistant_message_content(vec![AssistantMessageContent::ToolUse(tool_block)]);

        let result = pam.present(false).unwrap();
        assert!(matches!(result, Some(BlockProcessingResult::Error(_))));
        assert!(pam.state.did_already_use_tool);
    }

    // ---- Test 6: Tool rejection cascading ----
    #[test]
    fn test_tool_rejection_cascade() {
        let mut pam = PresentAssistantMessage::new();
        let mut params = HashMap::new();
        params.insert("path".to_string(), "/test.txt".to_string());
        pam.set_assistant_message_content(vec![make_tool_block(
            "tool-1",
            "read_file",
            params,
            false,
        )]);
        // Set after set_assistant_message_content because it resets did_reject_tool
        pam.state.did_reject_tool = true;

        let result = pam.present(false).unwrap();
        assert!(matches!(result, Some(BlockProcessingResult::Rejected)));
    }

    // ---- Test 7: Strip thinking tags ----
    #[test]
    fn test_strip_thinking_tags() {
        let input = "<thinking>Some thought</thinking>Actual content";
        let result = strip_thinking_tags(input);
        // strip_thinking_tags only removes the tags, not the content between them
        assert_eq!(result, "Some thoughtActual content");
    }

    // ---- Test 8: Sanitize tool use ID ----
    #[test]
    fn test_sanitize_tool_use_id() {
        assert_eq!(sanitize_tool_use_id("abc-123_def"), "abc-123_def");
        assert_eq!(sanitize_tool_use_id("abc.123@def"), "abc_123_def");
        assert_eq!(sanitize_tool_use_id("tool/123"), "tool_123");
    }

    // ---- Test 9: Out of bounds handling ----
    #[test]
    fn test_out_of_bounds_handling() {
        let mut pam = PresentAssistantMessage::new();
        pam.set_assistant_message_content(vec![]);
        pam.state.current_streaming_content_index = 5;

        let result = pam.present(false).unwrap();
        assert!(result.is_none());
    }

    // ---- Test 10: Content index advancement ----
    #[test]
    fn test_content_index_advancement() {
        let mut pam = PresentAssistantMessage::new();
        pam.set_assistant_message_content(vec![
            make_text_block("Block 1", false),
            make_text_block("Block 2", false),
        ]);

        let _ = pam.present(false);
        assert_eq!(pam.state.current_streaming_content_index, 1);
    }

    // ---- Test 11: File modifying tool detection ----
    #[test]
    fn test_is_file_modifying_tool() {
        assert!(is_file_modifying_tool("write_to_file"));
        assert!(is_file_modifying_tool("apply_diff"));
        assert!(is_file_modifying_tool("edit"));
        assert!(is_file_modifying_tool("search_and_replace"));
        assert!(is_file_modifying_tool("edit_file"));
        assert!(is_file_modifying_tool("apply_patch"));
        assert!(is_file_modifying_tool("new_task"));
        assert!(is_file_modifying_tool("generate_image"));
        assert!(!is_file_modifying_tool("read_file"));
        assert!(!is_file_modifying_tool("execute_command"));
        assert!(!is_file_modifying_tool("search_files"));
    }

    // ---- Test 12: Tool description generation ----
    #[test]
    fn test_tool_description() {
        let pam = PresentAssistantMessage::new();

        let mut params = HashMap::new();
        params.insert("command".to_string(), "ls -la".to_string());
        let tool = ToolUse {
            content_type: "tool_use".to_string(),
            name: "execute_command".to_string(),
            params: params.clone(),
            partial: false,
            id: "test-id".to_string(),
            native_args: None,
            original_name: None,
            used_legacy_format: false,
        };
        assert_eq!(pam.get_tool_description(&tool), "[execute_command for 'ls -la']");

        let mut params2 = HashMap::new();
        params2.insert("path".to_string(), "/test.rs".to_string());
        let tool2 = ToolUse {
            content_type: "tool_use".to_string(),
            name: "read_file".to_string(),
            params: params2,
            partial: false,
            id: "test-id".to_string(),
            native_args: Some(serde_json::json!({"path": "/test.rs"})),
            original_name: None,
            used_legacy_format: false,
        };
        assert_eq!(pam.get_tool_description(&tool2), "[read_file for '/test.rs']");
    }

    // ---- Test 13: Reset for new message ----
    #[test]
    fn test_reset_for_new_message() {
        let mut pam = PresentAssistantMessage::new();
        pam.state.current_streaming_content_index = 5;
        pam.state.did_reject_tool = true;
        pam.state.did_already_use_tool = true;
        pam.state.user_message_content_ready = true;
        pam.state.locked = true;

        pam.reset_for_new_message();
        assert_eq!(pam.state.current_streaming_content_index, 0);
        assert!(!pam.state.did_reject_tool);
        assert!(!pam.state.did_already_use_tool);
        assert!(!pam.state.user_message_content_ready);
        assert!(!pam.state.locked);
        assert!(pam.state.user_message_content.is_empty());
    }

    // ---- Test 14: ToolCallbacks duplicate prevention ----
    #[test]
    fn test_tool_callbacks_duplicate_prevention() {
        let mut callbacks = ToolCallbacks::new("test-id");
        let result = ToolResult::success("first result");
        assert!(callbacks.push_tool_result(&result));
        assert!(!callbacks.push_tool_result(&result)); // Duplicate should be rejected
    }

    // ---- Test 15: Format utility functions ----
    #[test]
    fn test_format_utilities() {
        assert_eq!(format_tool_error("something failed"), "Error: something failed");
        assert_eq!(format_tool_denied(), "Tool execution was denied by user.");
        assert!(format_tool_denied_with_feedback("try again").contains("try again"));
        assert!(format_tool_approved_with_feedback("looks good").contains("looks good"));
    }
}
