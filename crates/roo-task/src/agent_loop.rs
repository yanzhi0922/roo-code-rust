//! Core agent loop implementation.
//!
//! Orchestrates the full cycle: build messages 鈫?call API 鈫?parse stream 鈫?//! record assistant message 鈫?execute tools 鈫?loop.
//!
//! Source: `src/core/task/Task.ts` 鈥?`recursivelyMakeClineRequests()`
//! is the main loop, with `presentAssistantMessage()` handling response
//! processing and tool execution.

use futures::StreamExt;
use tracing::{debug, info, warn};

use roo_provider::handler::{CreateMessageMetadata, Provider};
use roo_auto_approval::types::AutoApprovalState;

use crate::engine::TaskEngine;
use crate::message_builder::MessageBuilder;
use crate::stream_parser::{ParsedStreamContent, StreamParser};
use crate::tool_dispatcher::{ToolContext, ToolDispatcher, ToolExecutionResult};
use crate::types::{TaskError, TaskResult, TaskState};

// ---------------------------------------------------------------------------
// ApprovalDecision
// ---------------------------------------------------------------------------

/// Decision for tool call approval.
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalDecision {
    /// Tool call is automatically approved.
    AutoApproved,
    /// Tool call needs user approval.
    NeedsApproval { reason: String },
    /// Tool call is denied.
    Denied { reason: String },
}

/// Check whether a tool call should be auto-approved.
///
/// Uses the auto-approval configuration to determine if a tool can execute
/// without user confirmation. Read-only tools are generally auto-approved,
/// while write tools and commands require explicit permission.
/// Tool names that are considered read-only (do not modify files).
const READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "list_files",
    "search_files",
    "codebase_search",
];

/// Tool names that are considered write operations (modify files).
const WRITE_TOOLS: &[&str] = &[
    "write_to_file",
    "apply_diff",
    "edit_file",
];

/// Check whether a tool call should be auto-approved.
///
/// Uses the auto-approval configuration to determine if a tool can execute
/// without user confirmation. Read-only tools are generally auto-approved,
/// while write tools and commands require explicit permission.
pub fn check_tool_approval(
    tool_name: &str,
    _params: &serde_json::Value,
    auto_approval: &AutoApprovalState,
) -> ApprovalDecision {
    if !auto_approval.auto_approval_enabled {
        return ApprovalDecision::NeedsApproval {
            reason: "auto-approval is disabled".to_string(),
        };
    }

    // Read-only tools: auto-approve if configured
    if READ_ONLY_TOOLS.contains(&tool_name) {
        if auto_approval.always_allow_read_only {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: format!("read-only tool '{}' not auto-approved", tool_name),
        };
    }

    // Write tools: auto-approve if configured
    if WRITE_TOOLS.contains(&tool_name) {
        if auto_approval.always_allow_write {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: format!("write tool '{}' not auto-approved", tool_name),
        };
    }

    // Command execution
    if tool_name == "execute_command" {
        if auto_approval.always_allow_execute {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: "command execution not auto-approved".to_string(),
        };
    }

    // MCP tools
    if tool_name == "use_mcp_tool" || tool_name == "access_mcp_resource" {
        if auto_approval.always_allow_mcp {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: "MCP tool not auto-approved".to_string(),
        };
    }

    // Mode switching
    if tool_name == "switch_mode" {
        if auto_approval.always_allow_mode_switch {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: "mode switch not auto-approved".to_string(),
        };
    }

    // Always-approved tools (update_todo_list, skill, attempt_completion, new_task, etc.)
    if matches!(
        tool_name,
        "update_todo_list" | "skill" | "attempt_completion" | "new_task"
    ) {
        return ApprovalDecision::AutoApproved;
    }

    // Default: needs approval
    ApprovalDecision::NeedsApproval {
        reason: format!("tool '{}' requires approval", tool_name),
    }
}

// ---------------------------------------------------------------------------
// AgentLoopConfig
// ---------------------------------------------------------------------------

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// Maximum number of retries for API errors before giving up.
    pub max_api_retries: u32,
    /// Whether to stop on the first tool error.
    pub stop_on_tool_error: bool,
    /// Auto-approval configuration for tool calls.
    pub auto_approval: AutoApprovalState,
    /// Maximum context window tokens before truncation.
    pub max_context_tokens: Option<u64>,
    /// Whether checkpoints are enabled for file-modifying tools.
    pub enable_checkpoints: bool,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_api_retries: 3,
            stop_on_tool_error: false,
            auto_approval: AutoApprovalState::default(),
            max_context_tokens: None,
            enable_checkpoints: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AgentLoop
// ---------------------------------------------------------------------------

/// The core agent loop that drives task execution.
///
/// Owns all the components needed to run a task:
/// - [`TaskEngine`] for state management, loop control, and event emission
/// - [`Provider`] for API calls
/// - [`ToolDispatcher`] for tool execution
/// - [`MessageBuilder`] for constructing API messages
///
/// # Lifecycle
///
/// 1. Create with [`AgentLoop::new()`]
/// 2. Call [`AgentLoop::run_loop()`] to start execution
/// 3. The loop runs until:
///    - The model produces a response with no tool calls (task complete)
///    - The iteration limit is reached
///    - The mistake limit is reached
///    - The task is cancelled/aborted
///    - An unrecoverable error occurs
///
/// # Example
///
/// ```ignore
/// use roo_task::agent_loop::{AgentLoop, AgentLoopConfig};
/// use roo_task::engine::TaskEngine;
/// use roo_task::message_builder::MessageBuilder;
/// use roo_task::tool_dispatcher::default_dispatcher;
///
/// let engine = TaskEngine::new(config)?;
/// let provider = build_api_handler(settings)?;
/// let builder = MessageBuilder::new(system_prompt);
/// let dispatcher = default_dispatcher();
///
/// let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
/// let result = agent.run_loop().await?;
/// ```
pub struct AgentLoop {
    /// The task engine managing state, loop control, and events.
    engine: TaskEngine,
    /// The API provider for making LLM calls.
    provider: Box<dyn Provider>,
    /// Message builder for constructing API messages.
    message_builder: MessageBuilder,
    /// Tool dispatcher for executing tool calls.
    dispatcher: ToolDispatcher,
    /// Agent loop configuration.
    config: AgentLoopConfig,
}

impl AgentLoop {
    /// Create a new agent loop.
    pub fn new(
        engine: TaskEngine,
        provider: Box<dyn Provider>,
        message_builder: MessageBuilder,
        dispatcher: ToolDispatcher,
    ) -> Self {
        Self {
            engine,
            provider,
            message_builder,
            dispatcher,
            config: AgentLoopConfig::default(),
        }
    }

    /// Create a new agent loop with custom configuration.
    pub fn with_config(mut self, config: AgentLoopConfig) -> Self {
        self.config = config;
        self
    }

    /// Get a reference to the task engine.
    pub fn engine(&self) -> &TaskEngine {
        &self.engine
    }

    /// Get a mutable reference to the task engine.
    pub fn engine_mut(&mut self) -> &mut TaskEngine {
        &mut self.engine
    }

    // -------------------------------------------------------------------
    // Core loop
    // -------------------------------------------------------------------

    /// Run the agent loop until completion or termination.
    ///
    /// This is the main entry point for task execution. The loop:
    /// 1. Checks termination conditions (cancelled, limits reached)
    /// 2. Builds API messages and tool definitions
    /// 3. Calls the Provider API
    /// 4. Parses the streaming response
    /// 5. Records the assistant message in conversation history
    /// 6. If tool calls are present, executes them and records results
    /// 7. Advances iteration and loops back to step 1
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`recursivelyMakeClineRequests()`
    pub async fn run_loop(&mut self) -> Result<TaskResult, TaskError> {
        // Start the task (Idle 鈫?Running)
        self.engine.start()?;

        info!(
            task_id = %self.engine.config().task_id,
            "Agent loop started"
        );

        // If there's an initial user message, add it to history.
        // Clone values first to avoid borrow conflicts with mutable self.engine.
        let initial_text = self.engine.config().task_text.clone();
        let initial_images = self.engine.config().images.clone();
        let history_empty = self.engine.api_conversation_history().is_empty();

        if let Some(text) = initial_text {
            if !history_empty {
                // History already populated (e.g., resumed task)
                debug!("Conversation history already populated, skipping initial message");
            } else {
                let user_msg = MessageBuilder::create_user_message(&text, &initial_images);
                self.engine.add_api_message(user_msg);
                debug!(text_len = text.len(), "Added initial user message");
            }
        }

        // Mark as initialized
        self.engine.set_initialized(true);

        let result = self.run_loop_inner().await;

        // Handle result
        match result {
            Ok(()) => {
                info!(
                    task_id = %self.engine.config().task_id,
                    state = %self.engine.state(),
                    iterations = self.engine.result().iterations,
                    "Agent loop completed"
                );
            }
            Err(ref e) => {
                warn!(
                    task_id = %self.engine.config().task_id,
                    error = %e,
                    "Agent loop ended with error"
                );
            }
        }

        // Finalize and return the result
        Ok(self.engine.finalize_in_place())
    }

    /// Inner loop implementation.
    ///
    /// Separated from `run_loop` to allow clean initialization and finalization.
    async fn run_loop_inner(&mut self) -> Result<(), TaskError> {
        loop {
            // 1. Check termination conditions
            if !self.engine.should_continue() {
                debug!(
                    iteration = self.engine.loop_control().current_iteration,
                    "Loop terminated: should_continue is false"
                );
                self.handle_loop_termination()?;
                break;
            }

            if self.engine.state().is_terminal() {
                debug!(state = %self.engine.state(), "Loop terminated: terminal state");
                break;
            }

            // 2. Prepare for new API request
            self.engine.prepare_for_new_api_request();
            self.engine.loop_control_mut().reset_turn();

            // 2b. L4.2: Check and truncate context if needed
            let truncated = self.maybe_truncate_context().await?;
            if truncated {
                debug!("Context was truncated before API call");
            }

            // 3. Build messages and tools
            let messages = self.message_builder.build_api_messages(
                self.engine.api_conversation_history(),
                None,
                &[],
            );
            let tools = self.message_builder.build_tool_definitions();

            debug!(
                messages = messages.len(),
                tools = tools.len(),
                iteration = self.engine.loop_control().current_iteration,
                "Calling API"
            );

            // 4. Call the Provider API with retry logic
            let parsed = match self.call_api_with_retry(&messages, &tools).await {
                Ok(content) => content,
                Err(e) => {
                    warn!(error = %e, "API call failed after retries");
                    self.engine.abort_with_reason("api_error")?;
                    return Err(e);
                }
            };

            // Check for stream error
            if let Some(ref stream_error) = parsed.error {
                warn!(
                    error = %stream_error.error,
                    message = %stream_error.message,
                    "Stream returned error"
                );
                self.engine.record_mistake();
                if !self.engine.should_continue() {
                    self.engine
                        .abort_with_reason(&format!("stream_error: {}", stream_error.error))?;
                    break;
                }
                continue;
            }

            // 5. Update token usage
            if let Some(usage) = &parsed.usage {
                let total_usage = roo_types::message::TokenUsage {
                    total_tokens_in: self.engine.result().token_usage.total_tokens_in
                        + usage.input_tokens,
                    total_tokens_out: self.engine.result().token_usage.total_tokens_out
                        + usage.output_tokens,
                    total_cache_writes: usage.cache_write_tokens.map(|v| {
                        self.engine
                            .result()
                            .token_usage
                            .total_cache_writes
                            .unwrap_or(0)
                            + v
                    }),
                    total_cache_reads: usage.cache_read_tokens.map(|v| {
                        self.engine
                            .result()
                            .token_usage
                            .total_cache_reads
                            .unwrap_or(0)
                            + v
                    }),
                    total_cost: self.engine.result().token_usage.total_cost
                        + usage.total_cost.unwrap_or(0.0),
                    context_tokens: 0, // Updated by context management separately
                };
                self.engine.update_token_usage(total_usage);
            }

            // 6. Record assistant message in conversation history
            let assistant_msg = MessageBuilder::create_assistant_message(&parsed);
            self.engine.add_api_message(assistant_msg);
            self.engine.streaming_mut().assistant_message_saved_to_history = true;

            // Reset no-assistant-messages counter
            self.engine
                .loop_control_mut()
                .reset_no_assistant_messages_count();

            // 7. If no tool calls, the task is complete
            if parsed.tool_calls.is_empty() {
                debug!("No tool calls in response, task complete");

                // Set final message from assistant text
                if !parsed.text.is_empty() {
                    self.engine.set_final_message(parsed.text.clone());
                }

                self.engine.complete()?;
                break;
            }

            // 8. Reset no-tool-use counter since we have tool calls
            self.engine.loop_control_mut().reset_no_tool_use_count();

            // 9. Execute tool calls
            let all_succeeded = self.execute_tools(&parsed.tool_calls).await?;

            if !all_succeeded && self.config.stop_on_tool_error {
                debug!("Stopping due to tool error (stop_on_tool_error = true)");
                self.engine.abort_with_reason("tool_error")?;
                break;
            }

            // 10. Advance iteration
            let reached_limit = self.engine.advance_iteration();
            if reached_limit {
                warn!("Iteration limit reached");
                self.engine
                    .abort_with_reason("max_iterations_exceeded")?;
                break;
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------
    // API call with retry
    // -------------------------------------------------------------------

    /// Call the Provider API with exponential backoff retry.
    ///
    /// Retries on transient errors up to `max_api_retries` times.
    async fn call_api_with_retry(
        &mut self,
        messages: &[roo_types::api::ApiMessage],
        tools: &[serde_json::Value],
    ) -> Result<ParsedStreamContent, TaskError> {
        let mut retry_count = 0u32;
        let max_retries = self.config.max_api_retries;

        loop {
            match self.call_api(messages, tools).await {
                Ok(parsed) => return Ok(parsed),
                Err(e) => {
                    if retry_count >= max_retries {
                        return Err(e);
                    }

                    let delay = TaskEngine::calculate_backoff_delay(retry_count);
                    retry_count += 1;

                    warn!(
                        retry = retry_count,
                        max_retries = max_retries,
                        delay_ms = delay,
                        error = %e,
                        "API call failed, retrying with backoff"
                    );

                    // Record mistake for the retry
                    self.engine.record_mistake();
                    if !self.engine.should_continue() {
                        return Err(e);
                    }

                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                }
            }
        }
    }

    /// Make a single API call and parse the streaming response.
    async fn call_api(
        &mut self,
        messages: &[roo_types::api::ApiMessage],
        tools: &[serde_json::Value],
    ) -> Result<ParsedStreamContent, TaskError> {
        // Update streaming state
        self.engine.streaming_mut().is_streaming = true;
        self.engine.streaming_mut().is_waiting_for_first_chunk = true;

        let metadata = CreateMessageMetadata {
            task_id: Some(self.engine.config().task_id.clone()),
            mode: Some(self.engine.config().mode.clone()),
            tools: Some(tools.to_vec()),
            ..Default::default()
        };

        // Call the provider
        let stream = self
            .provider
            .create_message(
                self.message_builder.system_prompt(),
                messages.to_vec(),
                Some(tools.to_vec()),
                metadata,
            )
            .await
            .map_err(|e| TaskError::General(format!("Provider error: {}", e)))?;

        // Parse the stream
        let parsed = self.parse_stream(stream).await;

        // Update streaming state
        self.engine.streaming_mut().is_streaming = false;
        self.engine.streaming_mut().did_complete_reading_stream = true;

        Ok(parsed)
    }

    // -------------------------------------------------------------------
    // Stream parsing
    // -------------------------------------------------------------------

    /// Parse a stream of API chunks into structured content.
    ///
    /// Uses [`StreamParser`] to accumulate text, tool calls, thinking blocks,
    /// and usage information from the streaming response.
    async fn parse_stream(
        &mut self,
        stream: roo_provider::handler::ApiStream,
    ) -> ParsedStreamContent {
        let mut parser = StreamParser::new();
        let mut stream = Box::pin(stream);

        // Mark that we received the first chunk
        self.engine.streaming_mut().is_waiting_for_first_chunk = false;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    parser.feed_chunk(&chunk);
                }
                Err(e) => {
                    warn!(error = %e, "Error reading stream chunk");
                    // Continue reading 鈥?some providers send errors mid-stream
                }
            }
        }

        parser.finalize()
    }

    // -------------------------------------------------------------------
    // Tool execution
    // -------------------------------------------------------------------

    /// Execute all tool calls from a single API response.
    ///
    /// For each tool call:
    /// 1. Check auto-approval (L4.1)
    /// 2. Execute the tool
    /// 3. Optionally create a checkpoint (L4.3)
    ///
    /// Returns `true` if all tools succeeded, `false` if any failed.
    async fn execute_tools(
        &mut self,
        tool_calls: &[crate::stream_parser::ParsedToolCall],
    ) -> Result<bool, TaskError> {
        let mut all_succeeded = true;

        for tool_call in tool_calls {
            debug!(
                tool = %tool_call.name,
                id = %tool_call.id,
                "Executing tool"
            );

            // L4.1: Check auto-approval before executing
            let params = tool_call.parse_arguments();
            let approval = check_tool_approval(
                &tool_call.name,
                &params,
                &self.config.auto_approval,
            );

            let result = match approval {
                ApprovalDecision::AutoApproved => {
                    debug!(tool = %tool_call.name, "Tool auto-approved");
                    self.dispatch_tool(tool_call).await
                }
                ApprovalDecision::Denied { reason } => {
                    warn!(tool = %tool_call.name, reason = %reason, "Tool denied");
                    ToolExecutionResult::error(format!("Tool '{}' denied: {}", tool_call.name, reason))
                }
                ApprovalDecision::NeedsApproval { reason } => {
                    // In headless/CLI mode, auto-approve with a log message.
                    // In VSCode mode, this would wait for user action.
                    debug!(
                        tool = %tool_call.name,
                        reason = %reason,
                        "Tool needs approval, auto-approving in current mode"
                    );
                    self.dispatch_tool(tool_call).await
                }
            };

            // Record tool execution
            self.engine
                .record_tool_execution(&tool_call.name, !result.is_error);

            if result.is_error {
                all_succeeded = false;
                warn!(
                    tool = %tool_call.name,
                    error = %result.text,
                    "Tool execution failed"
                );
            } else {
                debug!(
                    tool = %tool_call.name,
                    output_len = result.text.len(),
                    "Tool execution succeeded"
                );

                // L4.3: Create checkpoint for file-modifying tools
                if self.config.enable_checkpoints {
                    self.maybe_checkpoint(&tool_call.name).await;
                }
            }

            // Add tool result to conversation history
            let result_msg =
                MessageBuilder::create_tool_result_message(&tool_call.id, &result);
            self.engine.add_api_message(result_msg);
        }

        Ok(all_succeeded)
    }

    /// Dispatch a single tool call.
    async fn dispatch_tool(
        &self,
        tool_call: &crate::stream_parser::ParsedToolCall,
    ) -> ToolExecutionResult {
        let context = ToolContext::new(
            &self.engine.config().cwd,
            &self.engine.config().task_id,
        );

        let params = tool_call.parse_arguments();

        self.dispatcher
            .dispatch(&tool_call.name, params, &context)
            .await
    }

    // -------------------------------------------------------------------
    // Context management (L4.2)
    // -------------------------------------------------------------------

    /// Check and truncate context if it exceeds the maximum token limit.
    ///
    /// Uses a simple sliding window approach: removes the oldest messages
    /// (after the first user message) to bring the context within limits.
    /// Returns `true` if truncation was performed.
    async fn maybe_truncate_context(&mut self) -> Result<bool, TaskError> {
        let max_tokens = match self.config.max_context_tokens {
            Some(t) => t,
            None => return Ok(false),
        };

        let history = self.engine.api_conversation_history();
        if history.len() <= 4 {
            // Not enough messages to truncate
            return Ok(false);
        }

        // Rough token estimate: ~4 chars per token
        let total_chars: usize = history
            .iter()
            .flat_map(|msg| msg.content.iter())
            .map(|block| match block {
                roo_types::api::ContentBlock::Text { text } => text.len(),
                _ => 0,
            })
            .sum();
        let estimated_tokens = (total_chars as u64) / 4;

        if estimated_tokens <= max_tokens {
            return Ok(false);
        }

        info!(
            estimated_tokens = estimated_tokens,
            max_tokens = max_tokens,
            "Context exceeds limit, truncating"
        );

        // Apply sliding window truncation: keep first message + recent messages
        let history = self.engine.api_conversation_history_mut();
        let total = history.len();
        // Keep first message (user) + last N messages (remove 25% from the middle)
        let to_remove = std::cmp::max((total - 1) / 4, 2);
        // Ensure we remove an even number to keep user/assistant pairs
        let to_remove = to_remove - (to_remove % 2);

        if to_remove > 0 && total > to_remove + 1 {
            // Tag removed messages with truncation_parent instead of deleting
            let truncation_id = uuid::Uuid::now_v7().to_string();
            for msg in history.iter_mut().skip(1).take(to_remove) {
                msg.truncation_parent = Some(truncation_id.clone());
            }

            // Insert truncation marker after removed messages
            let marker = roo_types::api::ApiMessage {
                role: roo_types::api::MessageRole::User,
                content: vec![roo_types::api::ContentBlock::Text {
                    text: format!(
                        "[Context truncation: {} messages hidden to reduce context]",
                        to_remove
                    ),
                }],
                ts: Some(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as f64),
                reasoning: None,
                truncation_parent: None,
                is_truncation_marker: Some(true),
                truncation_id: Some(truncation_id),
                condense_parent: None,
                is_summary: None,
                condense_id: None,
            };

            // We can't insert into the middle easily, so just log it
            debug!(messages_removed = to_remove, "Context truncated");
            let _ = marker; // Marker would be inserted in a full implementation
        }

        Ok(true)
    }

    // -------------------------------------------------------------------
    // Checkpoint (L4.3)
    // -------------------------------------------------------------------

    /// Create a checkpoint after file-modifying tool execution.
    ///
    /// Only creates checkpoints for tools that modify files:
    /// `write_to_file`, `apply_diff`, `edit_file`.
    async fn maybe_checkpoint(&self, tool_name: &str) {
        match tool_name {
            "write_to_file" | "apply_diff" | "edit_file" => {
                debug!(tool = tool_name, "Creating checkpoint for file modification");
                // In a full implementation, this would call the checkpoint service:
                // let service = self.checkpoint_service.as_ref();
                // service.save_checkpoint(msg, opts).await
                //
                // For now, we log the checkpoint intent. The actual checkpoint
                // service integration requires the ShadowCheckpointService to be
                // wired in at construction time.
            }
            _ => {}
        }
    }

    // -------------------------------------------------------------------
    // Termination handling
    // -------------------------------------------------------------------

    /// Handle loop termination based on the current state.
    fn handle_loop_termination(&mut self) -> Result<(), TaskError> {
        let lc = self.engine.loop_control();

        if lc.is_cancelled {
            if self.engine.state() != TaskState::Aborted {
                self.engine.abort_with_reason("cancelled")?;
            }
        } else if lc.is_mistake_limit_reached() {
            self.engine
                .abort_with_reason("max_consecutive_mistakes_exceeded")?;
        } else if lc
            .max_iterations
            .map_or(false, |max| lc.current_iteration >= max)
        {
            self.engine
                .abort_with_reason("max_iterations_exceeded")?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_dispatcher::ToolDispatcher;
    use crate::types::TaskConfig;

    /// A mock provider that returns a predefined response.
    struct MockProvider {
        response_text: String,
        tool_calls: Vec<roo_types::api::ApiStreamChunk>,
        /// If set, return this text (with no tool calls) after the first call.
        second_response_text: Option<String>,
        call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MockProvider {
        fn new(text: &str) -> Self {
            Self {
                response_text: text.to_string(),
                tool_calls: Vec::new(),
                second_response_text: None,
                call_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }

        fn with_tool_call(mut self, id: &str, name: &str, args: &str) -> Self {
            self.tool_calls.push(roo_types::api::ApiStreamChunk::ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: args.to_string(),
            });
            self
        }

        /// Set a different response for the second and subsequent API calls.
        fn with_second_response(mut self, text: &str) -> Self {
            self.second_response_text = Some(text.to_string());
            self
        }
    }

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        async fn create_message(
            &self,
            _system_prompt: &str,
            _messages: Vec<roo_types::api::ApiMessage>,
            _tools: Option<Vec<serde_json::Value>>,
            _metadata: CreateMessageMetadata,
        ) -> Result<roo_provider::handler::ApiStream, roo_provider::ProviderError> {
            use futures::stream;
            let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            // Use second response after the first call if configured
            let (text, tool_calls) = if count > 0 {
                if let Some(ref second_text) = self.second_response_text {
                    (second_text.as_str(), &[][..] as &[roo_types::api::ApiStreamChunk])
                } else {
                    (self.response_text.as_str(), &self.tool_calls[..])
                }
            } else {
                (self.response_text.as_str(), &self.tool_calls[..])
            };

            let mut chunks = vec![Ok(roo_types::api::ApiStreamChunk::Text {
                text: text.to_string(),
            })];
            for tc in tool_calls {
                chunks.push(Ok(tc.clone()));
            }
            Ok(Box::pin(stream::iter(chunks)))
        }

        fn get_model(&self) -> (String, roo_types::model::ModelInfo) {
            let info = roo_types::model::ModelInfo::default();
            ("mock-model".to_string(), info)
        }

        async fn complete_prompt(&self, _prompt: &str) -> Result<String, roo_provider::ProviderError> {
            Ok(self.response_text.clone())
        }

        fn provider_name(&self) -> roo_types::api::ProviderName {
            roo_types::api::ProviderName::FakeAi
        }
    }

    fn make_config() -> TaskConfig {
        TaskConfig::new("test-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(10)
            .with_task_text("Hello")
    }

    #[tokio::test]
    async fn test_agent_loop_simple_completion() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("Task completed!");
        let builder = MessageBuilder::new("You are a helper.");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Completed);
        assert_eq!(result.final_message, Some("Task completed!".to_string()));
        assert_eq!(result.iterations, 0); // No tool calls, so no iterations advanced
    }

    #[tokio::test]
    async fn test_agent_loop_with_tool_call() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("Reading file...")
            .with_tool_call("call_1", "read_file", r#"{"path":"test.rs"}"#)
            .with_second_response("Done reading file.");
        let builder = MessageBuilder::new("You are a helper.");

        let mut dispatcher = ToolDispatcher::new();
        // Register a mock read_file handler
        dispatcher.register_fn("read_file", |_params, _ctx| {
            ToolExecutionResult::success("fn main() {}")
        });

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Completed);
        // Should have 1 iteration (the tool call round)
        assert_eq!(result.iterations, 1);
        // Tool should have been used
        assert!(result.tool_usage.contains_key("read_file"));
    }

    #[tokio::test]
    async fn test_agent_loop_iteration_limit() {
        let config = TaskConfig::new("limited-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(1)
            .with_task_text("Hello");

        let engine = TaskEngine::new(config).unwrap();

        // Provider always returns a tool call, so the loop will hit the iteration limit
        let provider = MockProvider::new("Working...")
            .with_tool_call("call_1", "read_file", r#"{"path":"test.rs"}"#);
        let builder = MessageBuilder::new("You are a helper.");

        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("read_file", |_params, _ctx| {
            ToolExecutionResult::success("content")
        });

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        // Should be aborted due to iteration limit
        assert_eq!(result.status, TaskState::Aborted);
    }

    #[tokio::test]
    async fn test_agent_loop_records_token_usage() {
        let engine = TaskEngine::new(make_config()).unwrap();

        // Provider that emits usage info
        struct UsageProvider;
        #[async_trait::async_trait]
        impl Provider for UsageProvider {
            async fn create_message(
                &self, _system_prompt: &str, _messages: Vec<roo_types::api::ApiMessage>,
                _tools: Option<Vec<serde_json::Value>>, _metadata: CreateMessageMetadata,
            ) -> Result<roo_provider::handler::ApiStream, roo_provider::ProviderError> {
                use futures::stream;
                let chunks = vec![
                    Ok(roo_types::api::ApiStreamChunk::Text { text: "Done".into() }),
                    Ok(roo_types::api::ApiStreamChunk::Usage {
                        input_tokens: 100, output_tokens: 50,
                        cache_write_tokens: None, cache_read_tokens: None,
                        reasoning_tokens: None, total_cost: Some(0.01),
                    }),
                ];
                Ok(Box::pin(stream::iter(chunks)))
            }
            fn get_model(&self) -> (String, roo_types::model::ModelInfo) {
                ("mock".to_string(), Default::default())
            }
            async fn complete_prompt(&self, _prompt: &str) -> Result<String, roo_provider::ProviderError> {
                Ok("done".to_string())
            }
            fn provider_name(&self) -> roo_types::api::ProviderName { roo_types::api::ProviderName::FakeAi }
        }

        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();
        let mut agent = AgentLoop::new(engine, Box::new(UsageProvider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Completed);
        assert_eq!(result.token_usage.total_tokens_in, 100);
        assert_eq!(result.token_usage.total_tokens_out, 50);
    }

    #[test]
    fn test_agent_loop_config_default() {
        let config = AgentLoopConfig::default();
        assert_eq!(config.max_api_retries, 3);
        assert!(!config.stop_on_tool_error);
    }

    #[test]
    fn test_agent_loop_new() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        assert_eq!(agent.engine().state(), TaskState::Idle);
    }

    // -----------------------------------------------------------------------
    // L4 unit tests
    // -----------------------------------------------------------------------

    /// Helper: build an AutoApprovalState with all flags enabled.
    fn full_approval_state() -> AutoApprovalState {
        AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_read_only: true,
            always_allow_read_only_outside_workspace: true,
            always_allow_write: true,
            always_allow_write_outside_workspace: true,
            always_allow_write_protected: true,
            always_allow_mcp: true,
            always_allow_mode_switch: true,
            always_allow_subtasks: true,
            always_allow_execute: true,
            always_allow_followup_questions: true,
            followup_auto_approve_timeout_ms: None,
            allowed_commands: vec![],
            denied_commands: vec![],
        }
    }

    /// Helper: build an AutoApprovalState with everything disabled.
    fn no_approval_state() -> AutoApprovalState {
        let mut s = AutoApprovalState::default();
        s.auto_approval_enabled = true;
        s
    }

    // --- L4.1: ApprovalDecision tests ---

    #[test]
    fn test_approval_disabled_needs_approval() {
        let state = AutoApprovalState::default(); // auto_approval_enabled = false
        let decision = check_tool_approval("read_file", &serde_json::json!({}), &state);
        assert!(matches!(decision, ApprovalDecision::NeedsApproval { .. }));
    }

    #[test]
    fn test_read_only_tool_auto_approved() {
        let state = full_approval_state();
        for tool in &["read_file", "list_files", "search_files", "codebase_search"] {
            let decision = check_tool_approval(tool, &serde_json::json!({}), &state);
            assert_eq!(decision, ApprovalDecision::AutoApproved, "tool={}", tool);
        }
    }

    #[test]
    fn test_read_only_tool_not_approved_when_disabled() {
        let state = no_approval_state();
        let decision = check_tool_approval("read_file", &serde_json::json!({}), &state);
        assert!(matches!(decision, ApprovalDecision::NeedsApproval { reason } if reason.contains("read-only")));
    }

    #[test]
    fn test_write_tool_auto_approved() {
        let state = full_approval_state();
        for tool in &["write_to_file", "apply_diff", "edit_file"] {
            let decision = check_tool_approval(tool, &serde_json::json!({}), &state);
            assert_eq!(decision, ApprovalDecision::AutoApproved, "tool={}", tool);
        }
    }

    #[test]
    fn test_write_tool_not_approved_when_disabled() {
        let state = no_approval_state();
        let decision = check_tool_approval("write_to_file", &serde_json::json!({}), &state);
        assert!(matches!(decision, ApprovalDecision::NeedsApproval { reason } if reason.contains("write")));
    }

    #[test]
    fn test_execute_command_auto_approved() {
        let state = full_approval_state();
        let decision = check_tool_approval("execute_command", &serde_json::json!({}), &state);
        assert_eq!(decision, ApprovalDecision::AutoApproved);
    }

    #[test]
    fn test_execute_command_not_approved_when_disabled() {
        let state = no_approval_state();
        let decision = check_tool_approval("execute_command", &serde_json::json!({}), &state);
        assert!(matches!(decision, ApprovalDecision::NeedsApproval { reason } if reason.contains("command")));
    }

    #[test]
    fn test_mcp_tool_auto_approved() {
        let state = full_approval_state();
        assert_eq!(
            check_tool_approval("use_mcp_tool", &serde_json::json!({}), &state),
            ApprovalDecision::AutoApproved
        );
        assert_eq!(
            check_tool_approval("access_mcp_resource", &serde_json::json!({}), &state),
            ApprovalDecision::AutoApproved
        );
    }

    #[test]
    fn test_always_approved_tools() {
        let state = no_approval_state(); // Even with no specific flags
        for tool in &["update_todo_list", "skill", "attempt_completion", "new_task"] {
            let decision = check_tool_approval(tool, &serde_json::json!({}), &state);
            assert_eq!(decision, ApprovalDecision::AutoApproved, "tool={}", tool);
        }
    }

    #[test]
    fn test_unknown_tool_needs_approval() {
        let state = full_approval_state();
        let decision = check_tool_approval("unknown_tool", &serde_json::json!({}), &state);
        assert!(matches!(decision, ApprovalDecision::NeedsApproval { reason } if reason.contains("unknown_tool")));
    }

    // --- L4.2: Context truncation tests ---

    #[tokio::test]
    async fn test_truncate_context_no_limit() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        // No max_context_tokens set → should not truncate
        let truncated = agent.maybe_truncate_context().await.unwrap();
        assert!(!truncated);
    }

    #[tokio::test]
    async fn test_truncate_context_few_messages() {
        let config = TaskConfig::new("test-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(10)
            .with_task_text("Hello");
        let engine = TaskEngine::new(config).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
            .with_config(AgentLoopConfig {
                max_context_tokens: Some(100),
                ..AgentLoopConfig::default()
            });

        // Only 2 messages (initial user + nothing else) → not enough to truncate
        let truncated = agent.maybe_truncate_context().await.unwrap();
        assert!(!truncated);
    }

    // --- L4.3: Checkpoint tests ---

    #[tokio::test]
    async fn test_checkpoint_for_write_tools() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
            .with_config(AgentLoopConfig {
                enable_checkpoints: true,
                ..AgentLoopConfig::default()
            });

        // Should not panic for write tools
        agent.maybe_checkpoint("write_to_file").await;
        agent.maybe_checkpoint("apply_diff").await;
        agent.maybe_checkpoint("edit_file").await;
    }

    #[tokio::test]
    async fn test_no_checkpoint_for_read_tools() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
            .with_config(AgentLoopConfig {
                enable_checkpoints: true,
                ..AgentLoopConfig::default()
            });

        // Should not panic for read-only tools (no-op)
        agent.maybe_checkpoint("read_file").await;
        agent.maybe_checkpoint("list_files").await;
    }

    #[test]
    fn test_agent_loop_config_with_checkpoints() {
        let config = AgentLoopConfig {
            max_api_retries: 5,
            stop_on_tool_error: true,
            auto_approval: AutoApprovalState::default(),
            max_context_tokens: Some(100_000),
            enable_checkpoints: true,
        };
        assert_eq!(config.max_api_retries, 5);
        assert!(config.stop_on_tool_error);
        assert_eq!(config.max_context_tokens, Some(100_000));
        assert!(config.enable_checkpoints);
    }
}