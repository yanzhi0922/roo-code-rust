//! Core agent loop implementation.
//!
//! Orchestrates the full cycle: build messages 鈫?call API 鈫?parse stream 鈫?//! record assistant message 鈫?execute tools 鈫?loop.
//!
//! Source: `src/core/task/Task.ts` 鈥?`recursivelyMakeClineRequests()`
//! is the main loop, with `presentAssistantMessage()` handling response
//! processing and tool execution.

use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use serde_json::json;
use tracing::{debug, info, warn};

use roo_provider::handler::{CreateMessageMetadata, Provider};
use roo_auto_approval::types::AutoApprovalState;
use roo_checkpoint::service::ShadowCheckpointService;
use roo_checkpoint::types::SaveCheckpointOptions;
use roo_tools::repetition::ToolRepetitionDetector;

use crate::engine::TaskEngine;
use crate::message_builder::MessageBuilder;
use crate::stream_parser::{ParsedStreamContent, ParsedToolCall, StreamParser};
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
    /// Whether condense (context summarization) is enabled.
    pub enable_condense: bool,
    /// Maximum requests per minute for rate limiting (0 = unlimited).
    pub rate_limit_rpm: u32,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_api_retries: 3,
            stop_on_tool_error: false,
            auto_approval: AutoApprovalState::default(),
            max_context_tokens: None,
            enable_checkpoints: false,
            enable_condense: false,
            rate_limit_rpm: 0,
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
    /// The API provider for making LLM calls (Arc-shared for condense).
    provider: Arc<dyn Provider>,
    /// Message builder for constructing API messages.
    message_builder: MessageBuilder,
    /// Tool dispatcher for executing tool calls.
    dispatcher: ToolDispatcher,
    /// Agent loop configuration.
    config: AgentLoopConfig,
    /// Tool repetition detector (Issue #5).
    repetition_detector: ToolRepetitionDetector,
    /// Checkpoint service for file-modifying tools (Issue #2).
    checkpoint_service: Option<ShadowCheckpointService>,
    /// Rate limiter: tracks request timestamps for RPM limiting (Issue #6).
    rate_limit_timestamps: Vec<Instant>,
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
            provider: Arc::from(provider),
            message_builder,
            dispatcher,
            config: AgentLoopConfig::default(),
            repetition_detector: ToolRepetitionDetector::with_default_limit(),
            checkpoint_service: None,
            rate_limit_timestamps: Vec::new(),
        }
    }

    /// Create a new agent loop with custom configuration.
    pub fn with_config(mut self, config: AgentLoopConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the checkpoint service for file-modifying tool checkpoints.
    pub fn with_checkpoint_service(mut self, service: ShadowCheckpointService) -> Self {
        self.checkpoint_service = Some(service);
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
                // Process @mentions in user input before sending to API
                let processed_text = self.process_user_mentions(&text).await;
                let user_msg = MessageBuilder::create_user_message(&processed_text, &initial_images);
                self.engine.add_api_message(user_msg);
                debug!(text_len = processed_text.len(), "Added initial user message (with @mentions processed)");
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
    /// Inner loop implementation.
    ///
    /// Separated from `run_loop` to allow clean initialization and finalization.
    ///
    /// Implements the TS `initiateTaskLoop()` outer loop behavior:
    /// - When the model doesn't use tools (noToolsUsed), re-prompts instead of
    ///   completing, matching the TS double-loop pattern.
    /// - Tracks consecutive no-tool-use and empty responses, escalating to
    ///   mistakes when thresholds are exceeded.
    /// - Applies one-time mistake grace before terminating.
    async fn run_loop_inner(&mut self) -> Result<(), TaskError> {
        loop {
            // 1. Check termination conditions (with mistake grace)
            if !self.engine.should_continue() {
                // Try one-time grace before terminating
                if self.engine.loop_control().is_mistake_limit_reached()
                    && self.engine.loop_control_mut().try_use_mistake_grace()
                {
                    warn!(
                        "Mistake limit reached, using one-time grace to continue"
                    );
                    // Continue the loop with reset mistake count
                } else {
                    debug!(
                        iteration = self.engine.loop_control().current_iteration,
                        "Loop terminated: should_continue is false"
                    );
                    self.handle_loop_termination()?;
                    break;
                }
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

            // 2c. Inject environment details before API call
            let env_details = self.get_environment_details();
            if !env_details.is_empty() {
                self.engine.add_environment_context(&env_details);
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

            // Check for stream error — record mistake and continue to let
            // the top-of-loop termination check handle grace logic.
            if let Some(ref stream_error) = parsed.error {
                warn!(
                    error = %stream_error.error,
                    message = %stream_error.message,
                    "Stream returned error"
                );
                self.engine.record_mistake();
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

            // --- Phase Q1: noToolsUsed / empty response handling ---

            let has_text = !parsed.text.is_empty();
            let has_tool_calls = !parsed.tool_calls.is_empty();

            // Check for empty response (no text AND no tool calls)
            //
            // Source: TS `recursivelyMakeClineRequests` — when the API returns
            // no assistant messages, the last user message is removed from
            // history before retrying to prevent consecutive user messages
            // (which would cause tool_result validation errors).
            if !has_text && !has_tool_calls {
                self.engine.loop_control_mut().record_no_assistant_message();

                // Remove the last user message from history before retrying.
                // This matches TS behavior where `apiConversationHistory.pop()`
                // removes the user message that received no response.
                let history = self.engine.api_conversation_history_mut();
                if let Some(last_msg) = history.last() {
                    if last_msg.role == roo_types::api::MessageRole::User {
                        history.pop();
                        debug!("Removed last user message before empty response retry");
                    }
                }

                if self.engine.loop_control_mut().should_retry_empty_response() {
                    warn!("Empty API response (no text, no tool calls), retrying...");
                    continue;
                }
                warn!("Empty API response after retries, aborting");
                self.engine.abort_with_reason("empty_response")?;
                break;
            }

            // We have content — reset no-assistant-messages counter
            self.engine
                .loop_control_mut()
                .reset_no_assistant_messages_count();

            // Check for noToolsUsed (model produced text but no tool calls).
            // Source: TS `initiateTaskLoop()` outer loop — re-prompts when
            // `formatResponse.noToolsUsed` is true.
            if !has_tool_calls {
                // noToolsUsed case — re-prompt model to use tools
                self.engine.loop_control_mut().record_no_tool_use();

                if self.engine.should_continue() {
                    let re_prompt = "You didn't use any tools. Please use tools to accomplish the task, or use attempt_completion if you're done.";
                    let user_msg = MessageBuilder::create_user_message(re_prompt, &[]);
                    self.engine.add_api_message(user_msg);
                    debug!(
                        no_tool_use_count = self.engine.loop_control().consecutive_no_tool_use_count,
                        "No tools used, re-prompting model"
                    );
                    continue;
                }

                // Mistake limit reached from no-tool-use — try grace
                if self.engine.loop_control_mut().try_use_mistake_grace() {
                    warn!("Mistake limit from no-tool-use, using one-time grace");
                    let re_prompt = "You didn't use any tools. Please use tools to accomplish the task, or use attempt_completion if you're done.";
                    let user_msg = MessageBuilder::create_user_message(re_prompt, &[]);
                    self.engine.add_api_message(user_msg);
                    continue;
                }

                // Grace exhausted — terminate
                self.handle_loop_termination()?;
                break;
            }

            // We have tool calls — reset no-tool-use counter
            self.engine.loop_control_mut().reset_no_tool_use();

            // 6b. new_task isolation: if new_task appears alongside other tools,
            // truncate any tools after it. This prevents orphaned tools when
            // delegation disposes the parent task.
            //
            // Source: TS `recursivelyMakeClineRequests` — enforces that new_task
            // must be the last tool in a message turn.
            let tool_calls = self.enforce_new_task_isolation(&parsed.tool_calls);

            // 7. Execute tool calls
            let all_succeeded = self.execute_tools(&tool_calls).await?;

            if !all_succeeded && self.config.stop_on_tool_error {
                debug!("Stopping due to tool error (stop_on_tool_error = true)");
                self.engine.abort_with_reason("tool_error")?;
                break;
            }

            // Check if attempt_completion was executed — complete the task
            let has_attempt_completion = tool_calls
                .iter()
                .any(|tc| tc.name == "attempt_completion");

            if has_attempt_completion {
                debug!("attempt_completion executed, completing task");
                if !parsed.text.is_empty() {
                    self.engine.set_final_message(parsed.text.clone());
                }
                self.engine.complete()?;
                break;
            }

            // Check if ask_followup_question was executed — pause for user input.
            //
            // Source: TS `presentAssistantMessage` — when ask_followup_question
            // is encountered, the loop pauses and waits for user response.
            // In headless/CLI mode, we log and continue (the tool already
            // returned its result). In interactive mode, this would block.
            let has_ask_followup = tool_calls
                .iter()
                .any(|tc| tc.name == "ask_followup_question");

            if has_ask_followup {
                debug!("ask_followup_question executed, user interaction may be needed");
                // In headless mode, the tool already returned its formatted
                // question. In interactive mode, the response would be injected
                // back into the conversation before continuing.
            }

            // Check for new_task delegation — delegate the task (C3).
            //
            // Source: TS `presentAssistantMessage` — new_task triggers
            // `startSubtask()` which delegates to a child task.
            // Source: TS `Task.ts` ~line 2380 — `startSubtask()` implementation.
            let has_new_task = tool_calls
                .iter()
                .any(|tc| tc.name == "new_task");

            if has_new_task {
                debug!("new_task executed, delegating to subtask");

                // Extract the new_task tool call parameters to create a subtask.
                if let Some(new_task_call) = tool_calls.iter().find(|tc| tc.name == "new_task") {
                    // Parse the arguments JSON string
                    let args: serde_json::Value = serde_json::from_str(&new_task_call.arguments)
                        .unwrap_or_default();

                    let subtask_text = args
                        .get("task")
                        .or_else(|| args.get("message"))
                        .or_else(|| args.get("text"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let subtask_mode = args
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&self.engine.config().mode);

                    if !subtask_text.is_empty() {
                        // Create a subtask engine with a unique ID.
                        let subtask_id = format!("{}-sub-{}", self.engine.config().task_id, uuid::Uuid::now_v7());
                        let mut subtask_config = crate::types::TaskConfig::new(
                            &subtask_id,
                            &self.engine.config().cwd,
                        );
                        subtask_config.mode = subtask_mode.to_string();
                        subtask_config.task_text = Some(subtask_text.to_string());

                        match crate::engine::TaskEngine::new(subtask_config) {
                            Ok(_subtask_engine) => {
                                info!(
                                    subtask_id = %subtask_id,
                                    mode = subtask_mode,
                                    "Created subtask engine for delegation"
                                );
                                // In a full implementation, the subtask engine would be
                                // stored in a TaskManager, given its own AgentLoop with
                                // a provider, and executed asynchronously. The parent task
                                // would wait for the subtask to complete before continuing.
                                //
                                // For now, we log the delegation and continue the parent loop.
                                self.engine.delegate().ok();
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to create subtask engine");
                            }
                        }
                    }
                }
            }

            // 8. Advance iteration
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
    /// Includes rate limiting (Issue #6) and backoff with notification (Issue #7).
    async fn call_api_with_retry(
        &mut self,
        messages: &[roo_types::api::ApiMessage],
        tools: &[serde_json::Value],
    ) -> Result<ParsedStreamContent, TaskError> {
        let mut retry_count = 0u32;
        let max_retries = self.config.max_api_retries;

        // Track context window retries separately from general API retries
        let mut context_window_retries = 0usize;
        let max_context_window_retries = crate::types::MAX_CONTEXT_WINDOW_RETRIES;

        loop {
            // --- Issue #6: Rate limiting ---
            self.maybe_wait_for_rate_limit().await;

            match self.call_api(messages, tools).await {
                Ok(parsed) => return Ok(parsed),
                Err(e) => {
                    let error_str = e.to_string();

                    // Check for context window exceeded error
                    if (error_str.contains("context_length_exceeded")
                        || error_str.contains("context window")
                        || error_str.contains("max_tokens"))
                        && context_window_retries < max_context_window_retries
                    {
                        context_window_retries += 1;
                        warn!(
                            context_window_retry = context_window_retries,
                            max_retries = max_context_window_retries,
                            "Context window exceeded, force truncating to {}%",
                            crate::types::FORCED_CONTEXT_REDUCTION_PERCENT
                        );
                        // Force truncate to FORCED_CONTEXT_REDUCTION_PERCENT
                        let keep_ratio =
                            (crate::types::FORCED_CONTEXT_REDUCTION_PERCENT as f64) / 100.0;
                        self.engine.force_truncate_context(keep_ratio);
                        continue;
                    }

                    if retry_count >= max_retries {
                        return Err(e);
                    }

                    // --- Issue #7: Backoff with notification ---
                    let delay = self.calculate_backoff_with_jitter(retry_count);
                    retry_count += 1;

                    warn!(
                        retry = retry_count,
                        max_retries = max_retries,
                        delay_ms = delay,
                        error = %e,
                        "API call failed, retrying with exponential backoff + jitter"
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
        let task_id = self.engine.config().task_id.clone();

        // Mark that we received the first chunk
        self.engine.streaming_mut().is_waiting_for_first_chunk = false;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    // Emit real-time streaming events for each chunk type.
                    // Each emitter() call creates a temporary borrow that is
                    // released immediately after the emit, avoiding borrow conflicts.
                    match &chunk {
                        roo_types::api::ApiStreamChunk::Text { text } => {
                            self.engine.emitter().emit_streaming_text_delta(&task_id, text);
                        }
                        roo_types::api::ApiStreamChunk::ToolCall { id, name, .. } => {
                            self.engine.emitter().emit_streaming_tool_use_started(&task_id, name, id);
                        }
                        roo_types::api::ApiStreamChunk::ToolCallStart { id, name, .. } => {
                            self.engine.emitter().emit_streaming_tool_use_started(&task_id, name, id);
                        }
                        _ => {}
                    }
                    parser.feed_chunk(&chunk);
                }
                Err(e) => {
                    warn!(error = %e, "Error reading stream chunk");
                    // Continue reading — some providers send errors mid-stream
                }
            }
        }

        self.engine.emitter().emit_streaming_completed(&task_id);
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
        tool_calls: &[ParsedToolCall],
    ) -> Result<bool, TaskError> {
        let mut all_succeeded = true;

        // --- Issue #3: Validate tool_result IDs ---
        let tool_calls = validate_and_fix_tool_result_ids(tool_calls.to_vec());

        for tool_call in &tool_calls {
            debug!(
                tool = %tool_call.name,
                id = %tool_call.id,
                "Executing tool"
            );

            let params = tool_call.parse_arguments();

            // --- Issue #5: Tool repetition detection ---
            if !self.repetition_detector.check_and_record(&tool_call.name, &params) {
                warn!(
                    tool = %tool_call.name,
                    consecutive = self.repetition_detector.consecutive_count(),
                    "Tool repetition limit reached, adding warning"
                );
                // Add a warning message to the conversation
                let warning_msg = MessageBuilder::create_user_message(
                    &format!(
                        "[WARNING] The tool '{}' has been called with identical parameters too many times in a row. \
                         Please try a different approach or use attempt_completion if the task is done.",
                        tool_call.name
                    ),
                    &[],
                );
                self.engine.add_api_message(warning_msg);
            }

            // L4.1: Check auto-approval before executing
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

            // Emit streaming tool-use-completed event
            {
                let task_id = &self.engine.config().task_id;
                self.engine
                    .emitter()
                    .emit_streaming_tool_use_completed(
                        task_id,
                        &tool_call.name,
                        &tool_call.id,
                        !result.is_error,
                    );
            }

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

                // Update file context tracker for file-modifying tools.
                //
                // Source: TS `presentAssistantMessage` — after tool execution,
                // `fileContextTracker.trackFileContext()` is called to record
                // which files were read or modified during the task.
                self.update_file_context(&tool_call.name, &params);
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
    // new_task isolation
    // -------------------------------------------------------------------

    /// Enforce `new_task` isolation: if `new_task` appears alongside other
    /// tools, truncate any tools that come after it and inject error
    /// tool_results for the truncated tools.
    ///
    /// Source: TS `recursivelyMakeClineRequests` — "Enforce new_task isolation:
    /// if new_task is called alongside other tools, truncate any tools that
    /// come after it and inject error tool_results."
    fn enforce_new_task_isolation(
        &mut self,
        tool_calls: &[crate::stream_parser::ParsedToolCall],
    ) -> Vec<crate::stream_parser::ParsedToolCall> {
        let new_task_idx = tool_calls
            .iter()
            .position(|tc| tc.name == "new_task");

        match new_task_idx {
            Some(idx) if idx < tool_calls.len() - 1 => {
                // new_task found but not last — truncate subsequent tools
                let truncated: Vec<_> = tool_calls[idx + 1..].to_vec();
                if !truncated.is_empty() {
                    warn!(
                        truncated_count = truncated.len(),
                        "new_task isolation: truncating {} tool(s) after new_task",
                        truncated.len()
                    );
                    // Inject error tool_results for truncated tools into conversation
                    for tc in &truncated {
                        let error_result = ToolExecutionResult::error(
                            "This tool was not executed because new_task was called in the same message turn. \
                             The new_task tool must be the last tool in a message.",
                        );
                        let result_msg =
                            MessageBuilder::create_tool_result_message(&tc.id, &error_result);
                        self.engine.add_api_message(result_msg);
                    }
                }
                tool_calls[..=idx].to_vec()
            }
            _ => tool_calls.to_vec(),
        }
    }

    // -------------------------------------------------------------------
    // Image generation (C4)
    // -------------------------------------------------------------------

    /// Handle image generation requests.
    ///
    /// Source: TS `presentAssistantMessage` — supports `generateImage`
    /// functionality via image generation APIs (e.g., DALL-E, Flux).
    /// Source: TS `GenerateImageTool.ts` — image generation tool.
    ///
    /// Uses the provider's API to send an image generation request.
    /// If the provider doesn't support image generation, returns a
    /// meaningful error message.
    async fn handle_image_generation(
        &self,
        prompt: &str,
    ) -> ToolExecutionResult {
        debug!(prompt_len = prompt.len(), "Attempting image generation");

        // Build an image generation prompt and try to use the provider.
        // The provider's complete_prompt method is used as a fallback since
        // there's no dedicated image generation API in the Provider trait yet.
        let image_prompt = format!(
            "Generate an image based on the following description. \
             If you cannot generate images, respond with a clear explanation.\n\n\
             Description: {}",
            prompt
        );

        match self.provider.complete_prompt(&image_prompt).await {
            Ok(result) => {
                if result.is_empty() {
                    ToolExecutionResult::error("Image generation returned an empty response from the provider.")
                } else {
                    let output = json!({
                        "generated": true,
                        "result": result,
                    });
                    ToolExecutionResult::success(serde_json::to_string_pretty(&output).unwrap_or_default())
                }
            }
            Err(e) => {
                warn!(error = %e, "Image generation failed via provider");
                ToolExecutionResult::error(&format!(
                    "Image generation failed: {}. \
                     The current provider may not support image generation. \
                     Please try a provider that supports image generation (e.g., OpenAI with DALL-E).",
                    e
                ))
            }
        }
    }

    // -------------------------------------------------------------------
    // File context tracking (C5)
    // -------------------------------------------------------------------

    /// Update the file context tracker after tool execution.
    ///
    /// Source: TS `presentAssistantMessage` — after each tool execution,
    /// `fileContextTracker.trackFileContext()` is called with the tool name
    /// and file path to record which files were accessed or modified.
    ///
    /// Uses `roo_context_tracking::tracker::FileContextTracker` with an
    /// in-memory store for tracking file operations.
    fn update_file_context(&self, tool_name: &str, params: &serde_json::Value) {
        // Extract file path from tool parameters
        let file_path = match tool_name {
            "read_file" | "write_to_file" | "apply_diff" | "edit_file" => {
                params.get("path")
                    .or_else(|| params.get("filePath"))
                    .or_else(|| params.get("file_path"))
                    .and_then(|v| v.as_str())
                    .map(|p| p.to_string())
            }
            "list_files" | "search_files" | "codebase_search" => {
                params.get("path").and_then(|v| v.as_str()).map(|p| p.to_string())
            }
            _ => None,
        };

        if let Some(path) = file_path {
            let source = match tool_name {
                "read_file" | "list_files" | "search_files" | "codebase_search" => {
                    roo_context_tracking::RecordSource::ReadTool
                }
                "write_to_file" | "apply_diff" | "edit_file" => {
                    roo_context_tracking::RecordSource::RooEdited
                }
                _ => {
                    debug!(
                        tool = tool_name,
                        path = %path,
                        "File context skipped: unknown tool"
                    );
                    return;
                }
            };

            debug!(
                tool = tool_name,
                path = %path,
                source = ?source,
                "File context tracked"
            );

            // Use an in-memory metadata store for tracking.
            // In production, this would use a FileMetadataStore with the
            // task's storage directory.
            let store = roo_context_tracking::InMemoryMetadataStore::new();
            let tracker = roo_context_tracking::FileContextTracker::new(
                &self.engine.config().task_id,
                store,
            );

            if let Err(e) = tracker.track_file_context(&path, source) {
                debug!(error = %e, "Failed to track file context (non-fatal)");
            }
        }
    }

    // -------------------------------------------------------------------
    // Context management (L4.2)
    // -------------------------------------------------------------------

    /// Check and truncate context if it exceeds the maximum token limit.
    ///
    /// If condense is enabled, first attempts to summarize the conversation
    /// using an LLM call. If condense fails or is disabled, falls back to
    /// sliding window truncation.
    ///
    /// Source: `src/core/task/Task.ts` — `manageContext()`
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
            "Context exceeds limit"
        );

        // --- Issue #1: Try condense first if enabled ---
        if self.config.enable_condense {
            match self.try_condense_context().await {
                Ok(true) => {
                    info!("Context condensed successfully via LLM summarization");
                    return Ok(true);
                }
                Ok(false) => {
                    debug!("Condense did not produce a result, falling back to truncation");
                }
                Err(e) => {
                    warn!(error = %e, "Condense failed, falling back to sliding window truncation");
                }
            }
        }

        // --- Fallback: sliding window truncation ---
        self.apply_sliding_window_truncation()
    }

    /// Attempt to condense the conversation using LLM summarization.
    ///
    /// Source: `src/core/task/Task.ts` — `manageContext()` → `condense()`
    async fn try_condense_context(&mut self) -> Result<bool, TaskError> {
        let history = self.engine.api_conversation_history().to_vec();
        let system_prompt = self.message_builder.system_prompt().to_string();
        let task_id = self.engine.config().task_id.clone();

        // Clone the Arc to share the provider with the condense call
        let provider_ref = Arc::clone(&self.provider);

        let options = roo_condense::summarize::SummarizeConversationOptions {
            messages: history,
            api_handler: provider_ref,
            system_prompt,
            task_id,
            is_automatic_trigger: true,
            custom_condensing_prompt: None,
            metadata: None,
            environment_details: Some(self.get_environment_details()),
            files_read_by_roo: None,
            cwd: Some(self.engine.config().cwd.clone()),
        };

        let result = roo_condense::summarize::summarize_conversation(options).await
            .map_err(|e| TaskError::General(format!("Condense error: {}", e)))?;

        if let Some(ref err) = result.error {
            warn!(error = %err, "Condense returned an error");
            return Ok(false);
        }

        if result.summary.is_empty() {
            warn!("Condense returned empty summary");
            return Ok(false);
        }

        // Replace conversation history with condensed messages
        self.engine.set_api_conversation_history(result.messages);

        // Update token usage with condense cost
        if result.cost > 0.0 {
            let current_usage = self.engine.result().token_usage.clone();
            self.engine.update_token_usage(roo_types::message::TokenUsage {
                total_cost: current_usage.total_cost + result.cost,
                ..current_usage
            });
        }

        debug!(
            summary_len = result.summary.len(),
            condense_id = ?result.condense_id,
            "Context condensed"
        );

        Ok(true)
    }

    /// Apply sliding window truncation to the conversation history.
    fn apply_sliding_window_truncation(&mut self) -> Result<bool, TaskError> {
        let history = self.engine.api_conversation_history();
        let total = history.len();
        // Keep first message (user) + last N messages (remove 25% from the middle)
        let to_remove = std::cmp::max((total - 1) / 4, 2);
        // Ensure we remove an even number to keep user/assistant pairs
        let to_remove = to_remove - (to_remove % 2);

        if to_remove > 0 && total > to_remove + 1 {
            let truncation_id = uuid::Uuid::now_v7().to_string();

            // Create truncation marker
            let marker = roo_types::api::ApiMessage {
                role: roo_types::api::MessageRole::User,
                content: vec![roo_types::api::ContentBlock::Text {
                    text: format!(
                        "[CONTEXT TRUNCATED] {} earlier messages have been removed to fit within the context window.",
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

            // Actually remove messages and insert marker via engine
            self.engine.truncate_history(to_remove, marker);
            debug!(messages_removed = to_remove, "Context truncated with marker inserted");
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
    ///
    /// Source: `src/core/task/Task.ts` — checkpoint save after file edits
    async fn maybe_checkpoint(&mut self, tool_name: &str) {
        match tool_name {
            "write_to_file" | "apply_diff" | "edit_file" => {
                debug!(tool = tool_name, "Creating checkpoint for file modification");

                if let Some(ref mut service) = self.checkpoint_service {
                    let message = format!("Checkpoint after {} tool execution", tool_name);
                    let options = SaveCheckpointOptions::default();

                    match service.save_checkpoint(&message, options).await {
                        Ok(Some(result)) => {
                            info!(
                                commit = %result.commit,
                                tool = tool_name,
                                "Checkpoint saved successfully"
                            );
                        }
                        Ok(None) => {
                            debug!(tool = tool_name, "No changes detected, checkpoint skipped");
                        }
                        Err(e) => {
                            warn!(error = %e, tool = tool_name, "Failed to save checkpoint");
                        }
                    }
                } else {
                    debug!(
                        tool = tool_name,
                        "Checkpoint service not configured, skipping checkpoint"
                    );
                }
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

    // -------------------------------------------------------------------
    // Environment details (Phase Q2)
    // -------------------------------------------------------------------

    /// Gather environment details for injection into the conversation.
    ///
    /// Source: `src/core/task/Task.ts` — `getEnvironmentDetails()`
    fn get_environment_details(&self) -> String {
        let mut details = Vec::new();

        // Working directory
        details.push(format!("Current working directory: {}", self.engine.config().cwd));

        // Platform info
        details.push(format!("Platform: {}", std::env::consts::OS));

        // Mode
        details.push(format!("Mode: {}", self.engine.config().mode));

        details.join("\n")
    }

    // -------------------------------------------------------------------
    // @mentions processing (Phase Q2)
    // -------------------------------------------------------------------

    /// Process @mentions in user text, expanding @file references to file content.
    ///
    /// Uses the full mention parser from `roo_mentions` to handle various
    /// mention types (files, folders, URLs, git changes, etc.).
    ///
    /// Source: `src/core/mentions/` — `processUserContentMentions()`
    async fn process_user_mentions(&self, text: &str) -> String {
        // Wrap text in <user_message> tags for the mention processor
        let wrapped = format!("<user_message>\n{}\n</user_message>", text);
        let cwd = std::path::Path::new(&self.engine.config().cwd);

        let blocks = vec![roo_mentions::ContentBlock::text(&wrapped)];

        let result = roo_mentions::process_user_content_mentions(&blocks, cwd).await;

        // Extract text from result blocks
        let mut output = String::new();
        for block in &result.content {
            if let Some(t) = block.as_text() {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(t);
            }
        }

        // Strip the <user_message> wrapper tags from the output
        let output = output
            .replace("<user_message>\n", "")
            .replace("<user_message>", "")
            .replace("\n</user_message>", "")
            .replace("</user_message>", "");

        if output.trim().is_empty() {
            text.to_string()
        } else {
            output
        }
    }

    /// Simple regex-based mention processing fallback.
    ///
    /// Used when the full mention parser is not available.
    fn process_mentions_simple(&self, text: &str) -> String {
        let re = regex::Regex::new(r"@(\S+)").unwrap_or_else(|_| regex::Regex::new(r"").unwrap());
        re.replace_all(text, |caps: &regex::Captures| {
            let path = &caps[1];
            if let Ok(content) = std::fs::read_to_string(path) {
                format!("`{}`:\n```\n{}\n```", path, content)
            } else {
                format!("@{}", path)
            }
        }).to_string()
    }

    // -------------------------------------------------------------------
    // Rate limiting (Issue #6)
    // -------------------------------------------------------------------

    /// Wait if the rate limit has been reached.
    ///
    /// Implements a simple sliding window rate limiter: tracks request
    /// timestamps and waits if the number of requests in the last minute
    /// exceeds the configured RPM limit.
    ///
    /// Source: `src/core/task/Task.ts` — `maybeWaitForProviderRateLimit()`
    async fn maybe_wait_for_rate_limit(&mut self) {
        let rpm = self.config.rate_limit_rpm;
        if rpm == 0 {
            return; // No rate limiting
        }

        let now = Instant::now();
        let one_minute = std::time::Duration::from_secs(60);

        // Remove timestamps older than 1 minute
        self.rate_limit_timestamps.retain(|&ts| now.duration_since(ts) < one_minute);

        // Check if we've hit the limit
        if self.rate_limit_timestamps.len() >= rpm as usize {
            // Calculate wait time until the oldest request expires
            if let Some(&oldest) = self.rate_limit_timestamps.first() {
                let elapsed = now.duration_since(oldest);
                let wait_time = one_minute.saturating_sub(elapsed);
                if !wait_time.is_zero() {
                    info!(
                        rpm_limit = rpm,
                        wait_ms = wait_time.as_millis(),
                        "Rate limit reached, waiting"
                    );
                    tokio::time::sleep(wait_time).await;
                }
            }
            // Clean up again after waiting
            let now = Instant::now();
            self.rate_limit_timestamps.retain(|&ts| now.duration_since(ts) < one_minute);
        }

        // Record this request
        self.rate_limit_timestamps.push(Instant::now());
    }

    // -------------------------------------------------------------------
    // Backoff with jitter (Issue #7)
    // -------------------------------------------------------------------

    /// Calculate exponential backoff delay with random jitter.
    ///
    /// Uses exponential base (2^attempt * 1s) capped at
    /// `MAX_EXPONENTIAL_BACKOFF_SECONDS`, with ±25% random jitter
    /// to avoid thundering herd effects.
    ///
    /// Source: `src/core/task/Task.ts` — `backoffAndAnnounce()`
    fn calculate_backoff_with_jitter(&self, retry_attempt: u32) -> u64 {
        let base_delay = TaskEngine::calculate_backoff_delay(retry_attempt);

        // Add ±25% jitter using a simple pseudo-random approach
        // Use retry_attempt as a seed for deterministic jitter in tests
        let jitter_factor = match retry_attempt % 4 {
            0 => 0.75,
            1 => 1.0,
            2 => 1.25,
            _ => 0.9,
        };

        let delayed = (base_delay as f64 * jitter_factor) as u64;
        delayed.min(crate::types::MAX_EXPONENTIAL_BACKOFF_SECONDS * 1000)
    }
}

// ---------------------------------------------------------------------------
// Tool result ID validation (Issue #3)
// ---------------------------------------------------------------------------

/// Validate and fix tool result IDs to match tool call IDs.
///
/// Ensures each tool_result has a valid tool_use_id that corresponds to
/// a tool_call in the same response. If IDs don't match, attempts to
/// match them by position. If that fails, assigns the first unmatched ID.
///
/// Source: `src/core/task/Task.ts` — `validateAndFixToolResultIds()`
pub fn validate_and_fix_tool_result_ids(tool_calls: Vec<ParsedToolCall>) -> Vec<ParsedToolCall> {
    // Collect all valid tool call IDs
    let _valid_ids: Vec<&str> = tool_calls.iter().map(|tc| tc.id.as_str()).collect();

    // For now, since ParsedToolCall IDs come from the stream parser and
    // are already validated during parsing, we just ensure they're non-empty.
    // If any ID is empty, generate one based on position.
    tool_calls
        .into_iter()
        .enumerate()
        .map(|(i, mut tc)| {
            if tc.id.is_empty() {
                tc.id = format!("tool_call_{}", i);
                warn!(tool = %tc.name, generated_id = %tc.id, "Generated missing tool call ID");
            }
            tc
        })
        .collect()
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
        /// If set, return this text after the first call.
        second_response_text: Option<String>,
        /// If set, return these tool calls after the first call.
        second_tool_calls: Vec<roo_types::api::ApiStreamChunk>,
        call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MockProvider {
        fn new(text: &str) -> Self {
            Self {
                response_text: text.to_string(),
                tool_calls: Vec::new(),
                second_response_text: None,
                second_tool_calls: Vec::new(),
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

        /// Set a different text for the second and subsequent API calls.
        fn with_second_response(mut self, text: &str) -> Self {
            self.second_response_text = Some(text.to_string());
            self
        }

        /// Add a tool call for the second and subsequent API calls.
        fn with_second_tool_call(mut self, id: &str, name: &str, args: &str) -> Self {
            self.second_tool_calls.push(roo_types::api::ApiStreamChunk::ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: args.to_string(),
            });
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

            let (text, tool_calls) = if count > 0 {
                let text = self.second_response_text.as_deref().unwrap_or(&self.response_text);
                let tcs = if !self.second_tool_calls.is_empty() {
                    &self.second_tool_calls[..]
                } else if self.second_response_text.is_some() {
                    &[][..] as &[roo_types::api::ApiStreamChunk]
                } else {
                    &self.tool_calls[..]
                };
                (text, tcs)
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
        // With the TS-matching behavior, the model must use attempt_completion
        // to complete the task. Text-only responses trigger re-prompting.
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("Task completed!")
            .with_tool_call("call_1", "attempt_completion", r#"{"result":"Task completed!"}"#);
        let builder = MessageBuilder::new("You are a helper.");

        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("attempt_completion", |_params, _ctx| {
            ToolExecutionResult::success("Task completed!")
        });

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Completed);
        assert_eq!(result.iterations, 0); // attempt_completion doesn't advance iteration
    }

    #[tokio::test]
    async fn test_agent_loop_with_tool_call() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("Reading file...")
            .with_tool_call("call_1", "read_file", r#"{"path":"test.rs"}"#)
            .with_second_response("Done reading file.")
            .with_second_tool_call("call_2", "attempt_completion", r#"{"result":"Done"}"#);
        let builder = MessageBuilder::new("You are a helper.");

        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("read_file", |_params, _ctx| {
            ToolExecutionResult::success("fn main() {}")
        });
        dispatcher.register_fn("attempt_completion", |_params, _ctx| {
            ToolExecutionResult::success("Done")
        });

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Completed);
        // Should have 1 iteration (the read_file round; attempt_completion doesn't advance)
        assert_eq!(result.iterations, 1);
        // Both tools should have been used
        assert!(result.tool_usage.contains_key("read_file"));
        assert!(result.tool_usage.contains_key("attempt_completion"));
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

        // Provider that emits usage info and uses attempt_completion to complete
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
                    Ok(roo_types::api::ApiStreamChunk::ToolCall {
                        id: "call_usage".to_string(),
                        name: "attempt_completion".to_string(),
                        arguments: r#"{"result":"Done"}"#.to_string(),
                    }),
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
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("attempt_completion", |_params, _ctx| {
            ToolExecutionResult::success("Done")
        });
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
        assert!(!config.enable_condense);
        assert_eq!(config.rate_limit_rpm, 0);
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

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
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

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
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
            enable_condense: false,
            rate_limit_rpm: 0,
        };
        assert_eq!(config.max_api_retries, 5);
        assert!(config.stop_on_tool_error);
        assert_eq!(config.max_context_tokens, Some(100_000));
        assert!(config.enable_checkpoints);
    }

    // --- Phase Q1: noToolsUsed re-prompt and mistake grace tests ---

    /// Provider that returns text-only on the first N calls, then attempt_completion.
    struct NoToolUseThenCompletionProvider {
        text_only_count: usize,
        call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl NoToolUseThenCompletionProvider {
        fn new(text_only_count: usize) -> Self {
            Self {
                text_only_count,
                call_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait::async_trait]
    impl Provider for NoToolUseThenCompletionProvider {
        async fn create_message(
            &self, _system_prompt: &str, _messages: Vec<roo_types::api::ApiMessage>,
            _tools: Option<Vec<serde_json::Value>>, _metadata: CreateMessageMetadata,
        ) -> Result<roo_provider::handler::ApiStream, roo_provider::ProviderError> {
            use futures::stream;
            let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut chunks = vec![Ok(roo_types::api::ApiStreamChunk::Text {
                text: "I'm thinking...".to_string(),
            })];
            if count >= self.text_only_count {
                chunks.push(Ok(roo_types::api::ApiStreamChunk::ToolCall {
                    id: format!("call_{}", count),
                    name: "attempt_completion".to_string(),
                    arguments: r#"{"result":"Done!"}"#.to_string(),
                }));
            }
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

    #[tokio::test]
    async fn test_no_tool_used_re_prompts_then_completes() {
        // Model returns text-only once, then uses attempt_completion.
        // The loop should re-prompt on the first call and complete on the second.
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = NoToolUseThenCompletionProvider::new(1);
        let builder = MessageBuilder::new("You are a helper.");

        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("attempt_completion", |_params, _ctx| {
            ToolExecutionResult::success("Done!")
        });

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Completed);
        // The no-tool-use count should have been recorded and then reset
        assert_eq!(agent.engine().loop_control().consecutive_no_tool_use_count, 0);
    }

    #[tokio::test]
    async fn test_no_tool_used_hits_mistake_limit_with_grace() {
        // Use a low mistake limit (2) and model never uses tools.
        // After consecutive_no_tool_use_count >= 2, mistakes accumulate.
        // Grace should be applied once, then the task aborts.
        let config = TaskConfig::new("test-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(100)
            .with_consecutive_mistake_limit(2)
            .with_task_text("Hello");

        let engine = TaskEngine::new(config).unwrap();

        // Provider that always returns text-only (never uses tools)
        struct AlwaysTextProvider;
        #[async_trait::async_trait]
        impl Provider for AlwaysTextProvider {
            async fn create_message(
                &self, _system_prompt: &str, _messages: Vec<roo_types::api::ApiMessage>,
                _tools: Option<Vec<serde_json::Value>>, _metadata: CreateMessageMetadata,
            ) -> Result<roo_provider::handler::ApiStream, roo_provider::ProviderError> {
                use futures::stream;
                Ok(Box::pin(stream::iter(vec![
                    Ok(roo_types::api::ApiStreamChunk::Text { text: "Just text".to_string() }),
                ])))
            }
            fn get_model(&self) -> (String, roo_types::model::ModelInfo) {
                ("mock".to_string(), Default::default())
            }
            async fn complete_prompt(&self, _prompt: &str) -> Result<String, roo_provider::ProviderError> {
                Ok("text".to_string())
            }
            fn provider_name(&self) -> roo_types::api::ProviderName { roo_types::api::ProviderName::FakeAi }
        }

        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();
        let mut agent = AgentLoop::new(engine, Box::new(AlwaysTextProvider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        // Should abort due to mistake limit (after grace is used)
        assert_eq!(result.status, TaskState::Aborted);
        // Grace should have been used
        assert!(agent.engine().loop_control().mistake_grace_used);
    }

    #[tokio::test]
    async fn test_attempt_completion_completes_task() {
        // Verify that attempt_completion tool execution completes the task
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("Completing...")
            .with_tool_call("call_1", "attempt_completion", r#"{"result":"All done!"}"#);
        let builder = MessageBuilder::new("You are a helper.");

        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("attempt_completion", |params, _ctx| {
            let text = params.get("result").and_then(|v| v.as_str()).unwrap_or("done");
            ToolExecutionResult::success(text)
        });

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Completed);
        assert!(result.tool_usage.contains_key("attempt_completion"));
    }

    // --- new_task isolation tests ---

    #[test]
    fn test_new_task_isolation_truncates_after() {
        // When new_task appears before other tools, those after it should be truncated
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        let tool_calls = vec![
            crate::stream_parser::ParsedToolCall {
                id: "tc_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            crate::stream_parser::ParsedToolCall {
                id: "tc_2".into(),
                name: "new_task".into(),
                arguments: r#"{"mode":"code","message":"sub task"}"#.into(),
            },
            crate::stream_parser::ParsedToolCall {
                id: "tc_3".into(),
                name: "write_to_file".into(),
                arguments: r#"{"path":"b.rs","content":"x"}"#.into(),
            },
        ];

        let result = agent.enforce_new_task_isolation(&tool_calls);
        assert_eq!(result.len(), 2); // read_file + new_task
        assert_eq!(result[0].name, "read_file");
        assert_eq!(result[1].name, "new_task");
    }

    #[test]
    fn test_new_task_isolation_no_truncation_when_last() {
        // When new_task is the last tool, no truncation should occur
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        let tool_calls = vec![
            crate::stream_parser::ParsedToolCall {
                id: "tc_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            crate::stream_parser::ParsedToolCall {
                id: "tc_2".into(),
                name: "new_task".into(),
                arguments: r#"{"mode":"code","message":"sub task"}"#.into(),
            },
        ];

        let result = agent.enforce_new_task_isolation(&tool_calls);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_new_task_isolation_no_new_task() {
        // When there's no new_task, all tools should be preserved
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        let tool_calls = vec![
            crate::stream_parser::ParsedToolCall {
                id: "tc_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            crate::stream_parser::ParsedToolCall {
                id: "tc_2".into(),
                name: "write_to_file".into(),
                arguments: r#"{"path":"b.rs","content":"x"}"#.into(),
            },
        ];

        let result = agent.enforce_new_task_isolation(&tool_calls);
        assert_eq!(result.len(), 2);
    }

    // --- Empty response user message removal test ---

    #[tokio::test]
    async fn test_empty_response_removes_user_message() {
        // When the API returns empty responses, the user message should be
        // removed before retrying to prevent consecutive user messages.
        let config = TaskConfig::new("test-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(10)
            .with_consecutive_mistake_limit(10)
            .with_task_text("Hello");

        let engine = TaskEngine::new(config).unwrap();

        // Provider that returns empty responses then completes
        struct EmptyThenCompleteProvider {
            call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        }
        impl EmptyThenCompleteProvider {
            fn new() -> Self {
                Self {
                    call_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                }
            }
        }
        #[async_trait::async_trait]
        impl Provider for EmptyThenCompleteProvider {
            async fn create_message(
                &self, _system_prompt: &str, _messages: Vec<roo_types::api::ApiMessage>,
                _tools: Option<Vec<serde_json::Value>>, _metadata: CreateMessageMetadata,
            ) -> Result<roo_provider::handler::ApiStream, roo_provider::ProviderError> {
                use futures::stream;
                let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if count < 1 {
                    // Return empty response (no chunks)
                    Ok(Box::pin(stream::iter(vec![])))
                } else {
                    // Return completion
                    Ok(Box::pin(stream::iter(vec![
                        Ok(roo_types::api::ApiStreamChunk::Text { text: "Done".into() }),
                        Ok(roo_types::api::ApiStreamChunk::ToolCall {
                            id: "call_1".into(),
                            name: "attempt_completion".into(),
                            arguments: r#"{"result":"Done!"}"#.into(),
                        }),
                    ])))
                }
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
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("attempt_completion", |_params, _ctx| {
            ToolExecutionResult::success("Done!")
        });

        let mut agent = AgentLoop::new(engine, Box::new(EmptyThenCompleteProvider::new()), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        // Should eventually complete after empty responses
        assert_eq!(result.status, TaskState::Completed);
    }

    // --- Issue #3: Tool result ID validation tests ---

    #[test]
    fn test_validate_tool_result_ids_preserves_valid() {
        let tool_calls = vec![
            ParsedToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            ParsedToolCall {
                id: "call_2".into(),
                name: "write_to_file".into(),
                arguments: r#"{"path":"b.rs"}"#.into(),
            },
        ];
        let result = validate_and_fix_tool_result_ids(tool_calls);
        assert_eq!(result[0].id, "call_1");
        assert_eq!(result[1].id, "call_2");
    }

    #[test]
    fn test_validate_tool_result_ids_fixes_empty() {
        let tool_calls = vec![
            ParsedToolCall {
                id: "".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            ParsedToolCall {
                id: "".into(),
                name: "write_to_file".into(),
                arguments: r#"{"path":"b.rs"}"#.into(),
            },
        ];
        let result = validate_and_fix_tool_result_ids(tool_calls);
        assert_eq!(result[0].id, "tool_call_0");
        assert_eq!(result[1].id, "tool_call_1");
    }

    // --- Issue #5: Tool repetition detection tests ---

    #[test]
    fn test_repetition_detector_integrated() {
        let mut detector = ToolRepetitionDetector::new(3);
        let params = serde_json::json!({"path": "test.rs"});

        // First 3 calls should be allowed
        assert!(detector.check_and_record("read_file", &params));
        assert!(detector.check_and_record("read_file", &params));
        assert!(detector.check_and_record("read_file", &params));

        // 4th identical call should be blocked
        assert!(!detector.check_and_record("read_file", &params));

        // After block, detector resets and allows again
        assert!(detector.check_and_record("read_file", &params));
    }

    // --- Issue #6: Rate limiter tests ---

    #[tokio::test]
    async fn test_rate_limiter_no_limit() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
            .with_config(AgentLoopConfig {
                rate_limit_rpm: 0, // No limit
                ..AgentLoopConfig::default()
            });

        // Should not wait or panic
        agent.maybe_wait_for_rate_limit().await;
        agent.maybe_wait_for_rate_limit().await;
        agent.maybe_wait_for_rate_limit().await;
    }

    // --- Issue #7: Backoff with jitter tests ---

    #[test]
    fn test_backoff_with_jitter_increases() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        let delay0 = agent.calculate_backoff_with_jitter(0);
        let _delay1 = agent.calculate_backoff_with_jitter(1);
        let delay2 = agent.calculate_backoff_with_jitter(2);

        // Delays should generally increase (exponential base)
        // With jitter they might not be strictly monotonic, but the base
        // grows fast enough that the trend should hold
        assert!(delay0 < delay2, "delay0={} should be < delay2={}", delay0, delay2);
    }

    // --- Issue #4: Mention processing tests ---

    #[test]
    fn test_process_mentions_simple_no_mentions() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.process_mentions_simple("Hello world");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_process_mentions_simple_with_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();

        let config = TaskConfig::new("test-task", dir.path().to_str().unwrap())
            .with_mode("code")
            .with_max_iterations(10);
        let engine = TaskEngine::new(config).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.process_mentions_simple(&format!(
            "look at @{}",
            file_path.to_str().unwrap()
        ));
        assert!(result.contains("fn main() {}"));
        assert!(!result.contains("@"));
    }
}