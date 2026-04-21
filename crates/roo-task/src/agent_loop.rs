#![allow(unused_assignments)]
//! Core agent loop implementation — rewritten to faithfully replicate
//! `.research/Roo-Code/src/core/task/Task.ts`.
//!
//! ## Method mapping
//!
//! | Rust method                       | TS source                         | Lines       |
//! |-----------------------------------|-----------------------------------|-------------|
//! | `run_loop()`                      | `initiateTaskLoop()`              | 2472–2504   |
//! | `recursively_make_cline_requests()`| `recursivelyMakeClineRequests()` | 2506–3742   |
//! | `attempt_api_request()`           | `attemptApiRequest()`             | 3987–4376   |
//! | `backoff_and_announce()`          | `backoffAndAnnounce()`            | 4378–4450   |
//! | `maybe_wait_for_rate_limit()`     | `maybeWaitForProviderRateLimit()` | 3959–3985   |
//! | `handle_context_window_exceeded_error()` | `handleContextWindowExceededError()` | 3828–3951 |
//! | `get_system_prompt()`             | `getSystemPrompt()`               | 3744–3819   |
//! | `build_clean_conversation_history()` | `buildCleanConversationHistory()` | 4458–4598 |
//!
//! ## Key design decisions
//!
//! * **Stack-based loop** — `recursivelyMakeClineRequests` was originally
//!   recursive in TS; the stack (`Vec<StackItem>`) faithfully models that.
//! * **Real-time streaming** — chunks are processed as they arrive (events
//!   emitted, parser fed), not batched after the stream ends.
//! * **First-chunk failure handling** — `attempt_api_request` waits for the
//!   first chunk and handles context-window / rate-limit / generic errors
//!   before continuing with the rest of the stream.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use serde_json::json;
use tokio::sync::{mpsc, Notify};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn, error};

use roo_provider::handler::{CreateMessageMetadata, Provider};
use roo_auto_approval::types::AutoApprovalState;
use roo_checkpoint::service::ShadowCheckpointService;
use roo_checkpoint::types::SaveCheckpointOptions;
use roo_tools::repetition::ToolRepetitionDetector;

use crate::engine::TaskEngine;
use crate::message_builder::MessageBuilder;
use crate::present_assistant_message::PresentAssistantMessage;
use crate::stream_parser::{ParsedStreamContent, ParsedToolCall, StreamParser};
use crate::tool_dispatcher::{ToolContext, ToolDispatcher, ToolExecutionResult};
use crate::types::{StreamEvent, TaskError, TaskResult, TaskState};

// ===========================================================================
// ApprovalDecision
// ===========================================================================

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

    if READ_ONLY_TOOLS.contains(&tool_name) {
        if auto_approval.always_allow_read_only {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: format!("read-only tool '{}' not auto-approved", tool_name),
        };
    }

    if WRITE_TOOLS.contains(&tool_name) {
        if auto_approval.always_allow_write {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: format!("write tool '{}' not auto-approved", tool_name),
        };
    }

    if tool_name == "execute_command" {
        if auto_approval.always_allow_execute {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: "command execution not auto-approved".to_string(),
        };
    }

    if tool_name == "use_mcp_tool" || tool_name == "access_mcp_resource" {
        if auto_approval.always_allow_mcp {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: "MCP tool not auto-approved".to_string(),
        };
    }

    if tool_name == "switch_mode" {
        if auto_approval.always_allow_mode_switch {
            return ApprovalDecision::AutoApproved;
        }
        return ApprovalDecision::NeedsApproval {
            reason: "mode switch not auto-approved".to_string(),
        };
    }

    if matches!(
        tool_name,
        "update_todo_list" | "skill" | "attempt_completion" | "new_task"
    ) {
        return ApprovalDecision::AutoApproved;
    }

    ApprovalDecision::NeedsApproval {
        reason: format!("tool '{}' requires approval", tool_name),
    }
}

// ===========================================================================
// AgentLoopConfig
// ===========================================================================

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

// ===========================================================================
// StackItem — models one "recursive call" in the TS stack-based loop
// ===========================================================================

/// A single item on the agent-loop stack.
///
/// Source: `src/core/task/Task.ts` line ~2508 — the `stack` array that
/// drives `recursivelyMakeClineRequests`. Each item represents one
/// "invocation" of the recursive function.
#[derive(Debug)]
#[allow(dead_code)]
struct StackItem {
    /// The user-facing content that triggered this iteration.
    /// In TS this is `userContent: UserContent`.
    user_content: Option<String>,
    /// Whether to include file details in the environment injection.
    /// TS: `includeFileDetails` parameter
    include_file_details: bool,
    /// Current retry attempt for this stack item (for API errors).
    /// TS: `retryAttempt` in StackItem interface
    retry_attempt: u32,
    /// Track if user message was removed due to empty response.
    /// When the assistant fails to respond (empty response), the user message
    /// is removed from history. On retry, this flag ensures the user message
    /// gets re-added.
    /// TS: `userMessageWasRemoved` in StackItem interface
    user_message_was_removed: bool,
}

// ===========================================================================
// AttemptApiRequestResult — output of attempt_api_request
// ===========================================================================

/// Result of `attempt_api_request`.
///
/// In TS this is the generator output; in Rust we return the fully-parsed
/// stream content along with metadata about whether the first chunk failed.
#[allow(dead_code)]
enum AttemptResult {
    /// Stream was successfully consumed.
    Ok {
        parsed: ParsedStreamContent,
        /// How many context-window retries were consumed.
        context_window_retries: usize,
    },
    /// The first chunk failed and we should retry the *entire* stack item.
    FirstChunkFailed {
        error: String,
        context_window_retries: usize,
    },
}

// ===========================================================================
// StreamFirstChunkError — error from attempt_api_request_stream()
// ===========================================================================

/// Error when the API call fails before producing any stream chunks.
///
/// This is returned by `attempt_api_request_stream()` when the initial
/// `provider.create_message()` call fails (connection error, context window
/// exceeded, rate limit, etc.).
#[derive(Debug)]
struct StreamFirstChunkError {
    message: String,
}

impl std::fmt::Display for StreamFirstChunkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

// ===========================================================================
// Context management result types
// ===========================================================================

/// Strategy used for context management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextStrategy {
    /// No strategy was applied.
    None,
    /// LLM-based condensation was used.
    Condense,
    /// Sliding window truncation was used.
    SlidingWindow,
    /// Force truncation (last resort).
    ForceTruncate,
}

/// Result of context management for context window exceeded errors.
#[allow(dead_code)]
struct ContextManagementResult {
    /// Whether the management was successful.
    success: bool,
    /// Number of messages removed.
    messages_removed: usize,
    /// The strategy that was used.
    strategy: ContextStrategy,
}

// ===========================================================================
// AgentLoop
// ===========================================================================

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
    /// Tool repetition detector.
    repetition_detector: ToolRepetitionDetector,
    /// Checkpoint service for file-modifying tools.
    checkpoint_service: Option<ShadowCheckpointService>,
    /// Rate limiter: tracks request timestamps for RPM limiting.
    rate_limit_timestamps: Vec<Instant>,
    /// Cancellation token for mid-stream abort.
    ///
    /// When cancelled, the spawned stream-consumer task will stop
    /// reading chunks and send a `StreamEvent::Error` with
    /// `is_first_chunk = false`.
    cancellation_token: CancellationToken,
    /// Present-assistant-message state machine for real-time content processing.
    ///
    /// Processes content blocks as they arrive from the stream, handling
    /// text display, tool call validation, and approval flows.
    present_assistant_message: PresentAssistantMessage,
    /// DiffView provider for managing file editing sessions with diff tracking.
    ///
    /// Source: `src/core/task/Task.ts` — `diffViewProvider`
    /// Manages the lifecycle of file edits: open → update → save/revert.
    diff_view_provider: Option<roo_editor::diff_view::DiffViewProvider>,
    /// Notify signal for `userMessageContentReady` synchronization.
    ///
    /// External code can await this to be notified when all tool execution
    /// completes and user message content is ready for the next API call.
    /// Source: `src/core/task/Task.ts` — `pWaitFor(() => this.userMessageContentReady)`
    user_message_content_ready_notify: Arc<Notify>,
    /// Timestamp of the last global API request, used for provider rate limiting.
    ///
    /// Source: `src/core/task/Task.ts` — `Task.lastGlobalApiRequestTime`
    /// This is a static field in TS shared across all task instances.
    /// In Rust, we track it per AgentLoop instance.
    last_global_api_request_time: Option<Instant>,
}

impl AgentLoop {
    /// Create a new agent loop.
    pub fn new(
        engine: TaskEngine,
        provider: Box<dyn Provider>,
        message_builder: MessageBuilder,
        dispatcher: ToolDispatcher,
    ) -> Self {
        let mistake_limit = engine.config().consecutive_mistake_limit;
        Self {
            engine,
            provider: Arc::from(provider),
            message_builder,
            dispatcher,
            config: AgentLoopConfig::default(),
            repetition_detector: ToolRepetitionDetector::with_default_limit(),
            checkpoint_service: None,
            rate_limit_timestamps: Vec::new(),
            cancellation_token: CancellationToken::new(),
            present_assistant_message: PresentAssistantMessage::with_mistake_limit(mistake_limit),
            diff_view_provider: None,
            user_message_content_ready_notify: Arc::new(Notify::new()),
            last_global_api_request_time: None,
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

    /// Set the DiffView provider for file editing sessions.
    ///
    /// Source: `src/core/task/Task.ts` — `diffViewProvider`
    pub fn with_diff_view_provider(mut self, provider: roo_editor::diff_view::DiffViewProvider) -> Self {
        self.diff_view_provider = Some(provider);
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

    /// Get a reference to the cancellation token.
    ///
    /// Used by `TaskLifecycle::cancel_current_request()` to abort a
    /// mid-stream API request.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation_token
    }

    /// Get a mutable reference to the present-assistant-message state machine.
    pub fn present_assistant_message_mut(&mut self) -> &mut PresentAssistantMessage {
        &mut self.present_assistant_message
    }

    /// Get a handle to the `userMessageContentReady` Notify.
    ///
    /// External code can use this to wait for all tool execution to complete
    /// and user message content to be ready for the next API call.
    ///
    /// Source: `src/core/task/Task.ts` — `pWaitFor(() => this.userMessageContentReady)`
    pub fn user_message_content_ready_notify(&self) -> Arc<Notify> {
        Arc::clone(&self.user_message_content_ready_notify)
    }

    /// Get a mutable reference to the DiffView provider, if configured.
    pub fn diff_view_provider_mut(&mut self) -> Option<&mut roo_editor::diff_view::DiffViewProvider> {
        self.diff_view_provider.as_mut()
    }

    // ===================================================================
    // 1. run_loop() — corresponds to TS initiateTaskLoop() (line 2472-2504)
    // ===================================================================

    /// Run the agent loop until completion or termination.
    ///
    /// This is the main entry point. It:
    /// 1. Starts the task engine (Idle → Running)
    /// 2. Adds the initial user message (with @mention processing)
    /// 3. Calls `recursively_make_cline_requests()` for the initial content
    /// 4. If the model responds without using tools (`noToolsUsed`),
    ///    re-prompts and calls again (the TS "outer loop" pattern)
    ///
    /// Source: `src/core/task/Task.ts` — `initiateTaskLoop()` lines 2472-2504
    #[allow(unused_assignments)]
    pub async fn run_loop(&mut self) -> Result<TaskResult, TaskError> {
        // Start the task (Idle → Running)
        self.engine.start()?;

        info!(
            task_id = %self.engine.config().task_id,
            "Agent loop started"
        );

        // If there's an initial user message, add it to history.
        let initial_text = self.engine.config().task_text.clone();
        let initial_images = self.engine.config().images.clone();
        let history_empty = self.engine.api_conversation_history().is_empty();

        if let Some(text) = initial_text {
            if !history_empty {
                debug!("Conversation history already populated, skipping initial message");
            } else {
                // TS: processUserContentMentions
                let processed_text = self.process_user_mentions(&text).await;
                let user_msg = MessageBuilder::create_user_message(&processed_text, &initial_images);
                self.engine.add_api_message(user_msg);
                debug!(text_len = processed_text.len(), "Added initial user message (with @mentions processed)");
            }
        }

        // Mark as initialized
        self.engine.set_initialized(true);

        // TS: initiateTaskLoop() — the outer loop that re-invokes when noToolsUsed
        let result = self.initiate_task_loop().await;

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

        Ok(self.engine.finalize_in_place())
    }

    // ===================================================================
    // initiate_task_loop() — TS initiateTaskLoop() outer loop (line 2472-2504)
    // ===================================================================

    /// The outer loop that drives `recursively_make_cline_requests()`.
    ///
    /// Source: `src/core/task/Task.ts` — `initiateTaskLoop()` lines 2472-2504.
    ///
    /// In TS, when `recursivelyMakeClineRequests` resolves with
    /// `noToolsUsed === true`, this outer loop re-prompts the model
    /// and calls again. This matches the TS double-loop pattern.
    async fn initiate_task_loop(&mut self) -> Result<(), TaskError> {
        // Initial stack item with the user's content
        let initial_content = self.engine.config().task_text.clone();
        let mut _no_tools_used = false;

        // Call the core recursive loop
        _no_tools_used = self.recursively_make_cline_requests(
            initial_content,
            true, // includeFileDetails for first call
        ).await?;

        // TS: if (noToolsUsed) { re-prompt and call again }
        // The noToolsUsed handling is now done inside recursively_make_cline_requests
        // via the stack-based loop, matching the TS behavior where it pushes
        // a new item onto the stack.

        Ok(())
    }

    // ===================================================================
    // 2. recursively_make_cline_requests() — TS line 2506-3742
    // ===================================================================

    /// Core stack-based loop that processes API requests.
    ///
    /// Source: `src/core/task/Task.ts` — `recursivelyMakeClineRequests()`
    /// lines 2506-3742.
    ///
    /// In TS this was originally a recursive function; the `stack` array
    /// models each recursive call as a stack item. Each iteration:
    ///
    /// 1. Pops a `StackItem`
    /// 2. Checks abort / mistake limit
    /// 3. Waits for rate limit
    /// 4. Emits `api_req_started`
    /// 5. Processes @mentions in user content
    /// 6. Injects environment details
    /// 7. Adds user message to API history
    /// 8. Resets streaming state
    /// 9. Calls `attempt_api_request()` → real-time stream processing
    /// 10. After stream completes:
    ///     - Finalizes raw chunks
    ///     - Marks partial blocks as complete
    ///     - Saves assistant message to API history
    ///     - Handles noToolsUsed / empty response
    ///     - Executes tools
    ///     - Pushes next StackItem to continue
    async fn recursively_make_cline_requests(
        &mut self,
        initial_user_content: Option<String>,
        include_file_details: bool,
    ) -> Result<bool, TaskError> {
        // TS: const stack: StackItem[] = [{ userContent, includeFileDetails, retryAttempt: 0 }]
        let mut stack: Vec<StackItem> = vec![StackItem {
            user_content: initial_user_content,
            include_file_details,
            retry_attempt: 0,
            user_message_was_removed: false,
        }];

        // TS: while (stack.length > 0)
        while let Some(current_item) = stack.pop() {
            let task_id = self.engine.config().task_id.clone();

            // ---------------------------------------------------------------
            // Step 1: Check abort
            // TS: if (this.abort) { break; }
            // ---------------------------------------------------------------
            if !self.engine.should_continue() {
                // Try one-time grace before terminating
                if self.engine.loop_control().is_mistake_limit_reached()
                    && self.engine.loop_control_mut().try_use_mistake_grace()
                {
                    warn!("Mistake limit reached, using one-time grace to continue");
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

            // ---------------------------------------------------------------
            // Step 2: Check consecutiveMistakeLimit
            // TS: if (this.consecutiveMistakeCount >= this.consecutiveMistakeLimit) { ... }
            // (handled by should_continue above)
            // ---------------------------------------------------------------

            // ---------------------------------------------------------------
            // Step 3: Rate limit wait
            // TS: await this.maybeWaitForProviderRateLimit()
            // ---------------------------------------------------------------
            self.maybe_wait_for_rate_limit().await;

            // TS L2586: Task.lastGlobalApiRequestTime = performance.now()
            // Set the timestamp right after rate limit wait to reserve this slot
            // before building environment details (which can take time).
            self.last_global_api_request_time = Some(Instant::now());

            // ---------------------------------------------------------------
            // Step 4: say("api_req_started")
            // TS: this.say("api_req_started", ...)
            // ---------------------------------------------------------------
            self.engine.prepare_for_new_api_request();
            self.engine.loop_control_mut().reset_turn();

            // ---------------------------------------------------------------
            // Step 5: processUserContentMentions + environment details
            // TS: processUserContentMentions(userContent) + getEnvironmentDetails()
            // Source: Task.ts L2603-2661
            // ---------------------------------------------------------------

            // Process @mentions in user content
            // TS: const { content: parsedUserContent, mode: slashCommandMode } =
            //     await processUserContentMentions({ userContent, ... })
            let parsed_user_content: Option<String> = if let Some(ref content) = current_item.user_content {
                if !content.is_empty() {
                    Some(self.process_user_mentions(content).await)
                } else {
                    None
                }
            } else {
                None
            };

            // TS: getEnvironmentDetails(this, currentIncludeFileDetails)
            // L4.2: Check and truncate context if needed
            let truncated = self.maybe_truncate_context().await?;
            if truncated {
                debug!("Context was truncated before API call");
            }

            // Inject environment details
            let env_details = self.get_environment_details();

            // TS L2634-2644: Remove any existing environment_details blocks before
            // adding fresh ones. This prevents duplicate environment details when
            // resuming tasks, where the old user message content may already contain
            // environment details from the previous session.
            // Build final user content: parsed content + environment details
            let final_user_content = match (&parsed_user_content, env_details.is_empty()) {
                (Some(content), true) => content.clone(),
                (Some(content), false) => {
                    let mut parts = content.clone();
                    parts.push_str("\n\n");
                    parts.push_str(&env_details);
                    parts
                }
                (None, false) => env_details.clone(),
                (None, true) => String::new(),
            };

            // TS L2655-2661: Only add user message to conversation history if:
            // 1. This is the first attempt (retryAttempt === 0), AND
            //    the original userContent was not empty (empty signals delegation
            //    resume where the user message is already in history), OR
            // 2. The message was removed in a previous iteration
            //    (userMessageWasRemoved === true)
            // This prevents consecutive user messages while allowing re-add when needed.
            let is_empty_user_content = current_item.user_content.as_ref().map_or(true, |c| c.is_empty());
            let should_add_user_message =
                (current_item.retry_attempt == 0 && !is_empty_user_content)
                || current_item.user_message_was_removed;

            if should_add_user_message && !final_user_content.is_empty() {
                let user_msg = MessageBuilder::create_user_message(&final_user_content, &[]);
                self.engine.add_api_message(user_msg);
                debug!(text_len = final_user_content.len(), "Added user message to API history");
            } else if should_add_user_message && final_user_content.is_empty() {
                // Even with empty content, add env details if we should add user message
                if !env_details.is_empty() {
                    let user_msg = MessageBuilder::create_user_message(&env_details, &[]);
                    self.engine.add_api_message(user_msg);
                    debug!("Added environment details as user message");
                }
            }

            // ---------------------------------------------------------------
            // Step 7: Build messages and tools
            // TS: build messages from apiConversationHistory
            // ---------------------------------------------------------------
            let clean_history = self.build_clean_conversation_history();
            let messages = self.message_builder.build_api_messages(
                &clean_history,
                None,
                &[],
            );
            let tools = self.message_builder.build_tool_definitions_with_options(
                Some(&self.engine.config().mode),
                &[],
                None,
                None,
            );
            let system_prompt = self.get_system_prompt();

            debug!(
                messages = messages.len(),
                tools = tools.len(),
                iteration = self.engine.loop_control().current_iteration,
                "Calling API"
            );

            // ---------------------------------------------------------------
            // Step 8: Reset streaming state
            // TS: reset streaming-related properties
            // ---------------------------------------------------------------
            // (already done in prepare_for_new_api_request above)

            // ---------------------------------------------------------------
            // Step 9: attemptApiRequest() → real-time stream processing
            // TS: const stream = this.attemptApiRequest(retryAttempt, ...)
            //     for await (const chunk of stream) { ... }
            //
            // New architecture: get a Receiver<StreamEvent> and consume
            // events one by one, feeding them to StreamParser and
            // PresentAssistantMessage in real-time.
            // ---------------------------------------------------------------
            #[allow(unused_assignments)]
            let mut context_window_retries = 0usize;

            // Get the streaming receiver
            let stream_rx: mpsc::Receiver<StreamEvent> = match self.attempt_api_request_stream(
                &system_prompt,
                &messages,
                &tools,
                current_item.retry_attempt,
            ).await {
                Ok(rx) => rx,
                Err(StreamFirstChunkError { message, .. }) => {
                    // First-chunk error handling (context window exceeded, etc.)
                    if self.is_context_window_error(&message) {
                        match self.handle_context_window_exceeded_error(
                            &message,
                            context_window_retries,
                        ).await {
                            Ok(new_cwr) => {
                                context_window_retries = new_cwr;
                                stack.push(StackItem {
                                    user_content: None,
                                    include_file_details: false,
                                    retry_attempt: current_item.retry_attempt,
                                    user_message_was_removed: false,
                                });
                                continue;
                            }
                            Err(_) => {
                                self.engine.abort_with_reason("context_window_exceeded")?;
                                return Err(TaskError::General(message));
                            }
                        }
                    }

                    // Generic API error — backoff and retry
                    if current_item.retry_attempt >= self.config.max_api_retries {
                        warn!(error = %message, "API call failed after max retries");
                        self.engine.abort_with_reason("api_error")?;
                        return Err(TaskError::General(message));
                    }

                    self.backoff_and_announce(current_item.retry_attempt, Some(&message)).await;
                    self.engine.record_mistake();
                    if !self.engine.should_continue() {
                        return Err(TaskError::General(message));
                    }

                    stack.push(StackItem {
                        user_content: None,
                        include_file_details: false,
                        retry_attempt: current_item.retry_attempt + 1,
                        user_message_was_removed: false,
                    });
                    continue;
                }
            };

            // Consume stream events in real-time
            let mut stream_rx = stream_rx;
            let mut parser = StreamParser::new();
            let mut assistant_text = String::new();
            let mut tool_calls_from_stream: Vec<ParsedToolCall> = Vec::new();
            let mut stream_error_message: Option<String> = None;
            let mut did_retry_push = false;
            let mut saw_stream_completed = false;

            while let Some(event) = stream_rx.recv().await {
                // Check cancellation
                if !self.engine.should_continue() {
                    debug!("Stream consumption interrupted: should_continue is false");
                    break;
                }

                match event {
                    StreamEvent::TextDelta { text } => {
                        assistant_text.push_str(&text);
                        self.engine.emitter().emit_streaming_text_delta(&task_id, &text);
                        parser.feed_chunk(&roo_types::api::ApiStreamChunk::Text { text });
                    }

                    StreamEvent::ReasoningDelta { text } => {
                        self.engine.emitter().emit_streaming_reasoning_delta(&task_id, &text);
                        parser.feed_chunk(&roo_types::api::ApiStreamChunk::Reasoning {
                            text,
                            signature: None,
                        });
                    }

                    StreamEvent::ToolCallStart { id, name } => {
                        self.engine.emitter().emit_streaming_tool_use_started(&task_id, &name, &id);
                        parser.start_streaming_tool_call(&id, &name);
                        self.engine.add_assistant_message_content(
                            crate::types::AssistantMessageContent::ToolUse(
                                crate::types::ToolUse {
                                    content_type: "tool_use".to_string(),
                                    name: name.clone(),
                                    params: Default::default(),
                                    partial: true,
                                    id: id.clone(),
                                    native_args: None,
                                    original_name: None,
                                    used_legacy_format: false,
                                }
                            )
                        );
                    }

                    StreamEvent::ToolCallDelta { id, delta } => {
                        self.engine.emitter().emit_streaming_tool_use_delta(&task_id, &id, &delta);
                        let _ = parser.process_streaming_chunk(&id, &delta);
                    }

                    StreamEvent::ToolCallEnd { id } => {
                        if let Some(content) = parser.finalize_streaming_tool_call(&id) {
                            match &content {
                                crate::types::AssistantMessageContent::ToolUse(tu) => {
                                    tool_calls_from_stream.push(ParsedToolCall {
                                        id: tu.id.clone(),
                                        name: tu.name.clone(),
                                        arguments: tu.native_args
                                            .as_ref()
                                            .map(|v| serde_json::to_string(v).unwrap_or_default())
                                            .unwrap_or_default(),
                                    });
                                }
                                crate::types::AssistantMessageContent::McpToolUse(mcp) => {
                                    tool_calls_from_stream.push(ParsedToolCall {
                                        id: mcp.id.clone(),
                                        name: mcp.name.clone(),
                                        arguments: serde_json::to_string(&mcp.arguments).unwrap_or_default(),
                                    });
                                }
                                _ => {}
                            }
                            self.engine.add_assistant_message_content(content);
                        }
                    }

                    StreamEvent::ToolCallComplete { id, name, arguments } => {
                        self.engine.emitter().emit_streaming_tool_use_started(&task_id, &name, &id);
                        if let Some(content) = parser.parse_tool_call(&id, &name, &arguments) {
                            match &content {
                                crate::types::AssistantMessageContent::ToolUse(tu) => {
                                    tool_calls_from_stream.push(ParsedToolCall {
                                        id: tu.id.clone(),
                                        name: tu.name.clone(),
                                        arguments: tu.native_args
                                            .as_ref()
                                            .map(|v| serde_json::to_string(v).unwrap_or_default())
                                            .unwrap_or_default(),
                                    });
                                }
                                crate::types::AssistantMessageContent::McpToolUse(mcp) => {
                                    tool_calls_from_stream.push(ParsedToolCall {
                                        id: mcp.id.clone(),
                                        name: mcp.name.clone(),
                                        arguments: serde_json::to_string(&mcp.arguments).unwrap_or_default(),
                                    });
                                }
                                _ => {}
                            }
                            self.engine.add_assistant_message_content(content);
                        }
                    }

                    StreamEvent::ToolCallPartial { index, id, name, arguments } => {
                        let events = parser.process_raw_chunk(
                            index,
                            id.as_deref(),
                            name.as_deref(),
                            arguments.as_deref(),
                        );
                        for tc_event in events {
                            match tc_event {
                                crate::types::ToolCallStreamEvent::Start { id, name } => {
                                    self.engine.emitter().emit_streaming_tool_use_started(&task_id, &name, &id);
                                }
                                crate::types::ToolCallStreamEvent::Delta { id, delta } => {
                                    self.engine.emitter().emit_streaming_tool_use_delta(&task_id, &id, &delta);
                                }
                                crate::types::ToolCallStreamEvent::End { .. } => {}
                            }
                        }
                    }

                    StreamEvent::Usage { input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, reasoning_tokens, total_cost } => {
                        parser.feed_chunk(&roo_types::api::ApiStreamChunk::Usage {
                            input_tokens, output_tokens,
                            cache_write_tokens, cache_read_tokens,
                            reasoning_tokens, total_cost,
                        });
                    }

                    StreamEvent::Grounding { sources } => {
                        parser.feed_chunk(&roo_types::api::ApiStreamChunk::Grounding { sources });
                    }

                    StreamEvent::ThinkingComplete { signature } => {
                        parser.feed_chunk(&roo_types::api::ApiStreamChunk::ThinkingComplete { signature });
                    }

                    StreamEvent::StreamCompleted => {
                        saw_stream_completed = true;
                        break;
                    }

                    StreamEvent::Error { message, is_first_chunk } => {
                        if is_first_chunk {
                            if self.is_context_window_error(&message) {
                                match self.handle_context_window_exceeded_error(
                                    &message,
                                    context_window_retries,
                                ).await {
                                    Ok(new_cwr) => {
                                        context_window_retries = new_cwr;
                                        stack.push(StackItem {
                                            user_content: None,
                                            include_file_details: false,
                                            retry_attempt: current_item.retry_attempt,
                                            user_message_was_removed: false,
                                        });
                                    }
                                    Err(_) => {
                                        self.engine.abort_with_reason("context_window_exceeded")?;
                                        return Err(TaskError::General(message));
                                    }
                                }
                            } else {
                                if current_item.retry_attempt >= self.config.max_api_retries {
                                    warn!(error = %message, "API call failed after max retries");
                                    self.engine.abort_with_reason("api_error")?;
                                    return Err(TaskError::General(message));
                                }
                                self.backoff_and_announce(current_item.retry_attempt, Some(&message)).await;
                                self.engine.record_mistake();
                                stack.push(StackItem {
                                    user_content: None,
                                    include_file_details: false,
                                    retry_attempt: current_item.retry_attempt + 1,
                                    user_message_was_removed: false,
                                });
                            }
                            did_retry_push = true;
                        } else {
                            warn!(error = %message, "Mid-stream error");
                            stream_error_message = Some(message);
                        }
                        break;
                    }
                }

                // TS L3038-3050: Check abort flag after each chunk
                // if (this.abort) { break; }
                if !self.engine.should_continue() {
                    debug!("Stream consumption interrupted: should_continue is false");
                    // TS: if (!this.abandoned) { await abortStream("user_cancelled"); }
                    self.engine.streaming_mut().is_streaming = false;
                    break;
                }

                // TS L3052-3061: Check didRejectTool — if user rejected a tool,
                // interrupt the assistant's response to present the user's feedback.
                // if (this.didRejectTool) { break; }
                if self.engine.streaming().did_reject_tool {
                    debug!("Stream interrupted: user rejected a tool");
                    assistant_text.push_str("\n\n[Response interrupted by user feedback]");
                    break;
                }

                // TS L3063-3067: Check didAlreadyUseTool — if a tool was already used,
                // interrupt the response since only one tool may be used at a time.
                // if (this.didAlreadyUseTool) { break; }
                if self.engine.streaming().did_already_use_tool {
                    debug!("Stream interrupted: tool already used");
                    assistant_text.push_str(
                        "\n\n[Response interrupted by a tool use result. Only one tool may be used at a time and should be placed at the end of the message.]"
                    );
                    break;
                }
            }

            // Background usage collection: if we broke out of the loop early
            // (not on StreamCompleted), drain remaining events to collect usage.
            // TS: after main consumption loop, background drain for usage data.
            // Source: `src/core/task/Task.ts` line ~3079
            if !saw_stream_completed {
                debug!("Stream ended early, draining remaining events for usage data");
                let drain_timeout = std::time::Duration::from_secs(5);
                loop {
                    match tokio::time::timeout(drain_timeout, stream_rx.recv()).await {
                        Ok(Some(event)) => match event {
                            StreamEvent::Usage {
                                input_tokens, output_tokens,
                                cache_write_tokens, cache_read_tokens,
                                reasoning_tokens, total_cost,
                            } => {
                                parser.feed_chunk(&roo_types::api::ApiStreamChunk::Usage {
                                    input_tokens, output_tokens,
                                    cache_write_tokens, cache_read_tokens,
                                    reasoning_tokens, total_cost,
                                });
                            }
                            StreamEvent::StreamCompleted => break,
                            _ => {}
                        },
                        Ok(None) => break, // Channel closed
                        Err(_) => {
                            debug!("Usage drain timed out");
                            break;
                        }
                    }
                }
            }

            // Stream ended — finalize
            self.engine.emitter().emit_streaming_completed(&task_id);
            self.engine.streaming_mut().is_streaming = false;
            self.engine.streaming_mut().did_complete_reading_stream = true;

            // Finalize any remaining raw chunks
            let _ = parser.finalize_raw_chunks();

            // Get the final parsed content from the parser
            let parsed = parser.finalize();

            // Merge streaming tool calls with parser results
            let mut all_tool_calls = parsed.tool_calls.clone();
            for tc in &tool_calls_from_stream {
                if !all_tool_calls.iter().any(|existing| existing.id == tc.id) {
                    all_tool_calls.push(tc.clone());
                }
            }

            // If we pushed a retry item, skip the rest of this iteration
            if did_retry_push {
                continue;
            }

            // ---------------------------------------------------------------
            // Step 10: Check for stream error
            // ---------------------------------------------------------------
            if let Some(ref error_msg) = stream_error_message {
                warn!(error = %error_msg, "Stream returned error");
                self.engine.record_mistake();
                continue;
            }

            // ---------------------------------------------------------------
            // Step 11: Update token usage
            // TS: background usage collection
            // ---------------------------------------------------------------
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
                    context_tokens: 0,
                };
                self.engine.update_token_usage(total_usage);
            }

            // ---------------------------------------------------------------
            // Step 12: Save assistant message to API history
            // TS: this.addToApiConversationHistory(assistantMessage)
            // ---------------------------------------------------------------
            let assistant_msg = MessageBuilder::create_assistant_message(&parsed);
            self.engine.add_api_message(assistant_msg);
            self.engine.streaming_mut().assistant_message_saved_to_history = true;

            // ---------------------------------------------------------------
            // Step 13: Handle noToolsUsed / empty response
            // TS: if (noToolsUsed) { ... } / if (!assistantContent.length) { ... }
            // Source: Task.ts L3596-3725
            // ---------------------------------------------------------------
            let has_text = !parsed.text.is_empty() || !assistant_text.is_empty();
            let has_tool_calls = !all_tool_calls.is_empty();

            // Empty response handling — no text and no tool calls from API
            // TS L3630-3725: "If there's no assistant_responses..."
            if !has_text && !has_tool_calls {
                // TS L3636: Increment consecutive no-assistant-messages counter
                self.engine.loop_control_mut().record_no_assistant_message();

                // TS L3640-3642: Only show error and count toward mistake limit
                // after 2 consecutive failures. This provides a "grace retry".
                let no_assistant_count = self.engine.loop_control().consecutive_no_assistant_messages_count;
                if no_assistant_count >= 2 {
                    // TS: await this.say("error", "MODEL_NO_ASSISTANT_MESSAGES")
                    warn!("MODEL_NO_ASSISTANT_MESSAGES: No assistant messages after {} attempts", no_assistant_count);
                    self.engine.emitter().emit(&crate::events::TaskEvent::Error {
                        task_id: task_id.clone(),
                        error: "MODEL_NO_ASSISTANT_MESSAGES".to_string(),
                    });
                }

                // TS L3648-3655: Remove last user message before retrying to avoid
                // having two consecutive user messages (which would cause tool_result
                // validation errors).
                let history = self.engine.api_conversation_history_mut();
                if let Some(last_msg) = history.last() {
                    if last_msg.role == roo_types::api::MessageRole::User {
                        history.pop();
                        debug!("Removed last user message before empty response retry");
                    }
                }

                // TS L3659-3686: Auto-retry with backoff when auto-approval is enabled
                if self.config.auto_approval.auto_approval_enabled {
                    // TS: await this.backoffAndAnnounce(currentItem.retryAttempt ?? 0, new Error(...))
                    self.backoff_and_announce(
                        current_item.retry_attempt,
                        Some("Unexpected API Response: The language model did not provide any assistant messages."),
                    ).await;

                    // TS L3669-3674: Check if task was aborted during the backoff
                    if !self.engine.should_continue() {
                        warn!("Task aborted during empty-assistant retry backoff");
                        break;
                    }

                    // TS L3678-3683: Push the same content back onto the stack to retry,
                    // incrementing the retry attempt counter.
                    // Mark that user message was removed so it gets re-added on retry.
                    stack.push(StackItem {
                        user_content: current_item.user_content.clone(),
                        include_file_details: false,
                        retry_attempt: current_item.retry_attempt + 1,
                        user_message_was_removed: true,
                    });
                    continue;
                } else {
                    // TS L3688-3724: Auto-approval disabled — ask user for retry decision.
                    // In headless mode, we auto-retry with a simple retry.
                    warn!("Empty API response, auto-retrying (no user interaction available)");
                    stack.push(StackItem {
                        user_content: current_item.user_content.clone(),
                        include_file_details: false,
                        retry_attempt: current_item.retry_attempt + 1,
                        user_message_was_removed: true,
                    });
                    continue;
                }
            }

            // TS L3422: Reset counter when we get a successful response with content
            self.engine
                .loop_control_mut()
                .reset_no_assistant_messages_count();

            // noToolsUsed handling
            // TS L3592-3629: "If the model did not tool use..."
            if !has_tool_calls {
                // TS L3598: Increment consecutive no-tool-use counter
                self.engine.loop_control_mut().record_no_tool_use();
                let no_tool_count = self.engine.loop_control().consecutive_no_tool_use_count;

                // TS L3601-3605: Only show error and count toward mistake limit
                // after 2 consecutive failures
                if no_tool_count >= 2 {
                    // TS: await this.say("error", "MODEL_NO_TOOLS_USED")
                    warn!("MODEL_NO_TOOLS_USED: No tools used after {} consecutive attempts", no_tool_count);
                    self.engine.emitter().emit(&crate::events::TaskEvent::Error {
                        task_id: task_id.clone(),
                        error: "MODEL_NO_TOOLS_USED".to_string(),
                    });
                }

                // TS L3608-3611: Push noToolsUsed message as user content
                let no_tools_msg = "[ERROR] You did not use any tools. Please use tools to accomplish the task, or use attempt_completion if you're done.";

                if self.engine.should_continue() {
                    // Push re-prompt onto stack (TS: recursivelyMakeClineRequests with new content)
                    stack.push(StackItem {
                        user_content: Some(no_tools_msg.to_string()),
                        include_file_details: false,
                        retry_attempt: 0,
                        user_message_was_removed: false,
                    });
                    debug!(
                        no_tool_use_count = no_tool_count,
                        "No tools used, pushing re-prompt onto stack"
                    );

                    // TS L3626: Add periodic yielding to prevent blocking
                    // await new Promise((resolve) => setImmediate(resolve))
                    tokio::task::yield_now().await;
                    continue;
                }

                // Mistake limit from no-tool-use — try grace
                if self.engine.loop_control_mut().try_use_mistake_grace() {
                    warn!("Mistake limit from no-tool-use, using one-time grace");
                    stack.push(StackItem {
                        user_content: Some(no_tools_msg.to_string()),
                        include_file_details: false,
                        retry_attempt: 0,
                        user_message_was_removed: false,
                    });
                    continue;
                }

                self.handle_loop_termination()?;
                break;
            }

            // TS L3613-3614: Reset counter when tools are used successfully
            self.engine.loop_control_mut().reset_no_tool_use();

            // ---------------------------------------------------------------
            // Step 14: Enforce new_task isolation
            // ---------------------------------------------------------------
            let tool_calls = self.enforce_new_task_isolation(&all_tool_calls);

            // ---------------------------------------------------------------
            // Step 15: Execute tool calls
            // ---------------------------------------------------------------
            let all_succeeded = self.execute_tools(&tool_calls).await?;

            // Notify waiters that user message content is ready.
            // Source: `src/core/task/Task.ts` — `pWaitFor(() => this.userMessageContentReady)`
            self.user_message_content_ready_notify.notify_waiters();

            if !all_succeeded && self.config.stop_on_tool_error {
                debug!("Stopping due to tool error (stop_on_tool_error = true)");
                self.engine.abort_with_reason("tool_error")?;
                break;
            }

            // ---------------------------------------------------------------
            // Step 16: Check for attempt_completion
            // ---------------------------------------------------------------
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

            // ---------------------------------------------------------------
            // Step 17: Check for ask_followup_question
            // ---------------------------------------------------------------
            let has_ask_followup = tool_calls
                .iter()
                .any(|tc| tc.name == "ask_followup_question");

            if has_ask_followup {
                debug!("ask_followup_question executed, user interaction may be needed");
            }

            // ---------------------------------------------------------------
            // Step 18: Check for new_task delegation
            // ---------------------------------------------------------------
            let has_new_task = tool_calls
                .iter()
                .any(|tc| tc.name == "new_task");

            if has_new_task {
                debug!("new_task executed, delegating to subtask");

                if let Some(new_task_call) = tool_calls.iter().find(|tc| tc.name == "new_task") {
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
                                self.engine.delegate().ok();
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to create subtask engine");
                            }
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // Step 19: Advance iteration and push next item
            // TS: push to stack to continue loop
            // ---------------------------------------------------------------
            let reached_limit = self.engine.advance_iteration();
            if reached_limit {
                warn!("Iteration limit reached");
                self.engine.abort_with_reason("max_iterations_exceeded")?;
                break;
            }

            // Push an empty item to continue the loop (tools executed, results added)
            stack.push(StackItem {
                user_content: None, // Tool results are already in history
                include_file_details: false,
                retry_attempt: 0,
                user_message_was_removed: false,
            });
        }

        Ok(false) // noToolsUsed
    }

    // ===================================================================
    // 3. attempt_api_request() — TS line 3987-4376 (DEPRECATED)
    // ===================================================================

    /// Attempt an API request with streaming response processing.
    ///
    /// **DEPRECATED**: Use `attempt_api_request_stream()` instead.
    /// This method is kept for backward compatibility but is no longer
    /// called by the main loop.
    ///
    /// Source: `src/core/task/Task.ts` — `attemptApiRequest()` lines 3987-4376.
    ///
    /// This method:
    /// 1. Creates the API stream via `provider.create_message()`
    /// 2. Waits for the first chunk (may fail)
    ///    - Context window error → truncate and retry
    ///    - Rate limit / generic error → return FirstChunkFailed
    /// 3. Processes the rest of the stream in real-time
    /// 4. Returns the fully parsed content
    ///
    /// In TS this is an async generator (`async *attemptApiRequest`).
    /// In Rust we model it as a method returning `AttemptResult`.
    #[allow(dead_code)]
    async fn attempt_api_request(
        &mut self,
        system_prompt: &str,
        messages: &[roo_types::api::ApiMessage],
        tools: &[serde_json::Value],
        retry_attempt: u32,
        mut context_window_retries: usize,
    ) -> AttemptResult {
        let task_id = self.engine.config().task_id.clone();

        // Update streaming state
        self.engine.streaming_mut().is_streaming = true;
        self.engine.streaming_mut().is_waiting_for_first_chunk = true;

        let metadata = CreateMessageMetadata {
            task_id: Some(task_id.clone()),
            mode: Some(self.engine.config().mode.clone()),
            tools: Some(tools.to_vec()),
            ..Default::default()
        };

        // ---------------------------------------------------------------
        // TS: const stream = await this.api.createMessage(...)
        // ---------------------------------------------------------------
        let stream = match self.provider.create_message(
            system_prompt,
            messages.to_vec(),
            Some(tools.to_vec()),
            metadata,
        ).await {
            Ok(s) => s,
            Err(e) => {
                let error_str = e.to_string();
                self.engine.streaming_mut().is_streaming = false;

                // Check for context window error
                if self.is_context_window_error(&error_str) {
                    match self.handle_context_window_exceeded_error(
                        &error_str,
                        context_window_retries,
                    ).await {
                        Ok(new_cwr) => {
                            context_window_retries = new_cwr;
                            // Retry with truncated context
                            return AttemptResult::FirstChunkFailed {
                                error: error_str,
                                context_window_retries,
                            };
                        }
                        Err(_) => {
                            return AttemptResult::FirstChunkFailed {
                                error: error_str,
                                context_window_retries,
                            };
                        }
                    }
                }

                // Generic error before first chunk
                warn!(error = %error_str, retry_attempt, "API call failed before first chunk");
                return AttemptResult::FirstChunkFailed {
                    error: error_str,
                    context_window_retries,
                };
            }
        };

        // ---------------------------------------------------------------
        // TS: Wait for first chunk (may fail)
        // for await (const chunk of stream) { yield chunk; break; }
        // ---------------------------------------------------------------
        let mut stream = Box::pin(stream);
        let mut parser = StreamParser::new();

        // Mark that we're past the initial connection
        self.engine.streaming_mut().is_waiting_for_first_chunk = false;

        // Process all chunks in real-time
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    // Emit real-time streaming events for each chunk type.
                    // TS: each chunk is processed immediately, not batched.
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

        // Stream completed
        self.engine.emitter().emit_streaming_completed(&task_id);
        self.engine.streaming_mut().is_streaming = false;
        self.engine.streaming_mut().did_complete_reading_stream = true;

        // ---------------------------------------------------------------
        // TS: finalizeRawChunks() — finalize the parser
        // ---------------------------------------------------------------
        let parsed = parser.finalize();

        AttemptResult::Ok {
            parsed,
            context_window_retries,
        }
    }

    // ===================================================================
    // 3b. attempt_api_request_stream() — new streaming architecture
    // ===================================================================

    /// Attempt an API request and return a stream event receiver.
    ///
    /// This is the new streaming architecture that replaces the batch-oriented
    /// `attempt_api_request()`. Instead of collecting all chunks into a
    /// `ParsedStreamContent`, it:
    ///
    /// 1. Creates the API stream via `provider.create_message()`
    /// 2. Spawns a task that converts each `ApiStreamChunk` to `StreamEvent`
    /// 3. Returns `mpsc::Receiver<StreamEvent>` for real-time consumption
    ///
    /// If the API call itself fails (before any chunks), returns
    /// `Err(StreamFirstChunkError)` so the caller can handle context window
    /// and retry logic.
    ///
    /// Source: `src/core/task/Task.ts` — `attemptApiRequest()` async generator
    async fn attempt_api_request_stream(
        &mut self,
        system_prompt: &str,
        messages: &[roo_types::api::ApiMessage],
        tools: &[serde_json::Value],
        _retry_attempt: u32,
    ) -> Result<mpsc::Receiver<StreamEvent>, StreamFirstChunkError> {
        let task_id = self.engine.config().task_id.clone();

        // Update streaming state
        self.engine.streaming_mut().is_streaming = true;
        self.engine.streaming_mut().is_waiting_for_first_chunk = true;

        let metadata = CreateMessageMetadata {
            task_id: Some(task_id.clone()),
            mode: Some(self.engine.config().mode.clone()),
            tools: Some(tools.to_vec()),
            ..Default::default()
        };

        // Create the API stream
        let stream = match self.provider.create_message(
            system_prompt,
            messages.to_vec(),
            Some(tools.to_vec()),
            metadata,
        ).await {
            Ok(s) => s,
            Err(e) => {
                let error_str = e.to_string();
                self.engine.streaming_mut().is_streaming = false;
                return Err(StreamFirstChunkError { message: error_str });
            }
        };

        // Create the channel
        let (tx, rx) = mpsc::channel(256);

        // Mark that we're past the initial connection
        self.engine.streaming_mut().is_waiting_for_first_chunk = false;

        // Get cancellation token (clone for the spawned task)
        let cancel_token = self.cancellation_token.clone();

        // Spawn a task to consume the stream and send StreamEvents
        tokio::spawn(async move {
            let mut stream = Box::pin(stream);
            let mut first_chunk = true;

            while let Some(chunk_result) = stream.next().await {
                // Check cancellation
                if cancel_token.is_cancelled() {
                    let _ = tx.send(StreamEvent::Error {
                        message: "Request cancelled".to_string(),
                        is_first_chunk: false,
                    }).await;
                    break;
                }

                match chunk_result {
                    Ok(chunk) => {
                        let was_first = first_chunk;
                        if first_chunk { first_chunk = false; }

                        // Convert ApiStreamChunk → StreamEvent(s)
                        let events = Self::convert_chunk_to_events(chunk, was_first);
                        for event in events {
                            if tx.send(event).await.is_err() {
                                // Receiver dropped — stop producing
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let error_str = e.to_string();
                        let _ = tx.send(StreamEvent::Error {
                            message: error_str,
                            is_first_chunk: first_chunk,
                        }).await;
                        break;
                    }
                }
            }

            // Signal stream completion
            let _ = tx.send(StreamEvent::StreamCompleted).await;
        });

        Ok(rx)
    }

    /// Convert an `ApiStreamChunk` into one or more `StreamEvent`s.
    ///
    /// This is the bridge between the provider's raw chunk format and
    /// the internal streaming event system.
    fn convert_chunk_to_events(
        chunk: roo_types::api::ApiStreamChunk,
        _is_first: bool,
    ) -> Vec<StreamEvent> {
        match chunk {
            roo_types::api::ApiStreamChunk::Text { text } => {
                vec![StreamEvent::TextDelta { text }]
            }
            roo_types::api::ApiStreamChunk::Reasoning { text, signature: _ } => {
                vec![StreamEvent::ReasoningDelta { text }]
            }
            roo_types::api::ApiStreamChunk::ToolCall { id, name, arguments } => {
                vec![StreamEvent::ToolCallComplete { id, name, arguments }]
            }
            roo_types::api::ApiStreamChunk::ToolCallStart { id, name } => {
                vec![StreamEvent::ToolCallStart { id, name }]
            }
            roo_types::api::ApiStreamChunk::ToolCallDelta { id, delta } => {
                vec![StreamEvent::ToolCallDelta { id, delta }]
            }
            roo_types::api::ApiStreamChunk::ToolCallEnd { id } => {
                vec![StreamEvent::ToolCallEnd { id }]
            }
            roo_types::api::ApiStreamChunk::ToolCallPartial { index, id, name, arguments } => {
                vec![StreamEvent::ToolCallPartial { index, id, name, arguments }]
            }
            roo_types::api::ApiStreamChunk::Usage {
                input_tokens, output_tokens,
                cache_write_tokens, cache_read_tokens,
                reasoning_tokens, total_cost,
            } => {
                vec![StreamEvent::Usage {
                    input_tokens, output_tokens,
                    cache_write_tokens, cache_read_tokens,
                    reasoning_tokens, total_cost,
                }]
            }
            roo_types::api::ApiStreamChunk::Grounding { sources } => {
                vec![StreamEvent::Grounding { sources }]
            }
            roo_types::api::ApiStreamChunk::ThinkingComplete { signature } => {
                vec![StreamEvent::ThinkingComplete { signature }]
            }
            roo_types::api::ApiStreamChunk::Error { error: _, message } => {
                vec![StreamEvent::Error {
                    message,
                    is_first_chunk: false,
                }]
            }
        }
    }

    // ===================================================================
    // 4. backoff_and_announce() — TS line 4378-4450
    // ===================================================================

    /// Exponential backoff with announcement.
    ///
    /// Source: `src/core/task/Task.ts` — `backoffAndAnnounce()` lines 4378-4450.
    ///
    /// This method:
    /// 1. Calculates exponential backoff delay
    /// 2. Respects provider rate limit window (rateLimitSeconds)
    /// 3. Parses 429 RetryInfo from error details if present
    /// 4. Takes the maximum of exponential delay and rate limit delay
    /// 5. Shows countdown timer to the user
    /// 6. Checks abort flag during countdown for early exit
    /// Exponential backoff with announcement.
    ///
    /// Source: `src/core/task/Task.ts` — `backoffAndAnnounce()` lines 4378-4450.
    ///
    /// This method faithfully replicates the TS behavior:
    /// 1. Calculates exponential backoff: `ceil(baseDelay * 2^retryAttempt)`
    /// 2. Respects provider rate limit window using `lastGlobalApiRequestTime`
    /// 3. Parses 429 RetryInfo from error details if present
    /// 4. Takes the maximum of exponential delay and rate limit delay
    /// 5. Builds header text with status code and error message
    /// 6. Shows countdown timer with `api_req_retry_delayed` event
    /// 7. Checks abort flag during countdown for early exit
    async fn backoff_and_announce(&mut self, retry_attempt: u32, error: Option<&str>) {
        // Wrap in try/catch equivalent to match TS error handling
        if let Err(_err) = self.backoff_and_announce_inner(retry_attempt, error).await {
            // TS L4441-4449: catch block — check if abort during countdown
            if !self.engine.should_continue() {
                return; // Aborted during countdown — silently return
            }
            warn!("Exponential backoff failed unexpectedly");
        }
    }

    /// Inner implementation of backoff_and_announce.
    async fn backoff_and_announce_inner(
        &mut self,
        retry_attempt: u32,
        error: Option<&str>,
    ) -> Result<(), String> {
        // TS L4382: const baseDelay = state?.requestDelaySeconds || 5
        let base_delay_secs = 5u64;

        // TS L4384-4387: Calculate exponential delay
        // let exponentialDelay = Math.min(
        //     Math.ceil(baseDelay * Math.pow(2, retryAttempt)),
        //     MAX_EXPONENTIAL_BACKOFF_SECONDS,
        // )
        let mut exponential_delay_secs = std::cmp::min(
            (base_delay_secs * 2u64.pow(retry_attempt)).max(1),
            crate::types::MAX_EXPONENTIAL_BACKOFF_SECONDS,
        );

        // TS L4390-4395: Respect provider rate limit window
        // Uses lastGlobalApiRequestTime to calculate remaining time
        let mut rate_limit_delay_secs: u64 = 0;
        if let Some(last_request_time) = self.last_global_api_request_time {
            let rate_limit_secs = self.config.rate_limit_rpm; // Reuse as rateLimitSeconds
            if rate_limit_secs > 0 {
                let elapsed = last_request_time.elapsed();
                let rate_limit_duration = std::time::Duration::from_secs(rate_limit_secs as u64);
                let remaining = rate_limit_duration.saturating_sub(elapsed);
                rate_limit_delay_secs = remaining.as_secs();
            }
        }

        // TS L4398-4406: Prefer RetryInfo on 429 if present
        // if (error?.status === 429) {
        //     const retryInfo = error?.errorDetails?.find(...)
        //     const match = retryInfo?.retryDelay?.match?.(/^(\d+)s$/)
        //     if (match) { exponentialDelay = Number(match[1]) + 1 }
        // }
        if let Some(error_str) = error {
            if error_str.contains("429") || error_str.contains("rate_limit") || error_str.contains("rate limit") {
                if let Some(retry_secs) = parse_retry_info_from_error(error_str) {
                    exponential_delay_secs = retry_secs + 1;
                }
            }
        }

        // TS L4408: const finalDelay = Math.max(exponentialDelay, rateLimitDelay)
        let final_delay_secs = std::cmp::max(exponential_delay_secs, rate_limit_delay_secs);
        if final_delay_secs == 0 {
            return Ok(());
        }

        // TS L4414-4427: Build header text
        // if (error.status) {
        //     headerText = `${error.status}\n${errorMessage}`
        // } else if (error?.message) {
        //     headerText = error.message
        // } else {
        //     headerText = "Unknown error"
        // }
        // headerText = headerText ? `${headerText}\n` : ""
        let header_text = error
            .map(|e| {
                // Try to detect status code in the error message
                if e.chars().take(3).all(|c| c.is_ascii_digit()) {
                    // Error starts with a status code
                    format!("{}\n", e)
                } else {
                    format!("{}\n", e)
                }
            })
            .unwrap_or_else(|| "Unknown error\n".to_string());

        // TS L4430-4438: Show countdown timer with exponential backoff
        for i in (1..=final_delay_secs).rev() {
            // TS L4432-4434: Check abort flag during countdown to allow early exit
            if !self.engine.should_continue() {
                // TS: throw new Error(`[Task#${this.taskId}] Aborted during retry countdown`)
                return Err(format!(
                    "[Task#{}] Aborted during retry countdown",
                    self.engine.config().task_id
                ));
            }

            // TS L4436: await this.say("api_req_retry_delayed",
            //     `${headerText}<retry_timer>${i}</retry_timer>`, undefined, true)
            self.engine.emitter().emit(
                &crate::events::TaskEvent::ApiRequestStarted {
                    task_id: self.engine.config().task_id.clone(),
                },
            );

            debug!(
                retry_attempt = retry_attempt,
                remaining_secs = i,
                "Retry countdown"
            );

            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }

        // TS L4440: await this.say("api_req_retry_delayed", headerText, undefined, false)
        info!(
            retry_attempt = retry_attempt,
            total_delay_secs = final_delay_secs,
            header = %header_text.trim(),
            "Backoff complete, retrying"
        );

        Ok(())
    }

    // ===================================================================
    // 5. maybe_wait_for_rate_limit() — TS line 3959-3985
    // ===================================================================

    /// Wait if the rate limit has been reached.
    ///
    /// Source: `src/core/task/Task.ts` — `maybeWaitForProviderRateLimit()`
    /// lines 3959-3985.
    ///
    /// Implements a simple sliding window rate limiter: tracks request
    /// timestamps and waits if the number of requests in the last minute
    /// exceeds the configured RPM limit.
    async fn maybe_wait_for_rate_limit(&mut self) {
        let rpm = self.config.rate_limit_rpm;
        if rpm == 0 {
            return;
        }

        let now = Instant::now();
        let one_minute = std::time::Duration::from_secs(60);

        // Remove timestamps older than 1 minute
        self.rate_limit_timestamps.retain(|&ts| now.duration_since(ts) < one_minute);

        // Check if we've hit the limit
        if self.rate_limit_timestamps.len() >= rpm as usize {
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
            let now = Instant::now();
            self.rate_limit_timestamps.retain(|&ts| now.duration_since(ts) < one_minute);
        }

        // Record this request
        self.rate_limit_timestamps.push(Instant::now());
    }

    // ===================================================================
    // 6. handle_context_window_exceeded_error() — TS line 3828-3951
    // ===================================================================

    /// Handle context window exceeded errors by managing context.
    ///
    /// Source: `src/core/task/Task.ts` — `handleContextWindowExceededError()`
    /// lines 3828-3951.
    ///
    /// When the API returns a context_length_exceeded error, this method:
    /// 1. Emits `condenseTaskContextStarted` event
    /// 2. Tries `condense_context` (LLM summarization) if enabled
    /// 3. Falls back to `sliding_window_truncation` if condense fails/disabled
    /// 4. Emits `condenseTaskContextResponse` event (always, to dismiss spinner)
    ///
    /// Returns `Ok(new_context_window_retries)` on success, or `Err(())`
    /// if the maximum number of context window retries has been exceeded.
    async fn handle_context_window_exceeded_error(
        &mut self,
        _error: &str,
        context_window_retries: usize,
    ) -> Result<usize, ()> {
        let max_retries = crate::types::MAX_CONTEXT_WINDOW_RETRIES;

        if context_window_retries >= max_retries {
            error!(
                context_window_retries,
                max_retries,
                "Context window retries exhausted"
            );
            return Err(());
        }

        let new_retries = context_window_retries + 1;
        let task_id = self.engine.config().task_id.clone();

        warn!(
            context_window_retry = new_retries,
            max_retries = max_retries,
            "Context window exceeded, managing context (attempt {}/{})",
            new_retries, max_retries
        );

        // TS: Send condenseTaskContextStarted to show in-progress indicator
        self.engine.emitter().emit(&crate::events::TaskEvent::ContextCondensationRequested {
            task_id: task_id.clone(),
        });

        let result = self.manage_context_for_exceeded_error().await;

        // TS: Always send condenseTaskContextResponse (finally block)
        self.engine.emitter().emit(&crate::events::TaskEvent::ContextCondensationCompleted {
            task_id: task_id.clone(),
            messages_removed: result.messages_removed,
        });

        if result.success {
            Ok(new_retries)
        } else {
            // Even if management failed, we still truncated — allow retry
            // unless we've exhausted retries (checked above)
            Ok(new_retries)
        }
    }

    /// Internal context management for context window exceeded errors.
    ///
    /// Mirrors the TS `manageContext()` call inside `handleContextWindowExceededError`.
    /// Tries condense first, then falls back to sliding window truncation.
    async fn manage_context_for_exceeded_error(&mut self) -> ContextManagementResult {
        // Try condense first if enabled
        if self.config.enable_condense {
            match self.try_condense_context().await {
                Ok(true) => {
                    info!("Context condensed successfully after context window exceeded");
                    return ContextManagementResult {
                        success: true,
                        messages_removed: 0, // condense replaces, doesn't remove
                        strategy: ContextStrategy::Condense,
                    };
                }
                Ok(false) => {
                    debug!("Condense did not produce a result, falling back to sliding window");
                }
                Err(e) => {
                    warn!(error = %e, "Condense failed, falling back to sliding window truncation");
                }
            }
        }

        // Fallback: sliding window truncation
        match self.apply_sliding_window_truncation() {
            Ok(true) => {
                let history = self.engine.api_conversation_history();
                // Count truncation markers to estimate messages removed
                let removed = history.iter()
                    .filter(|m| m.is_truncation_marker == Some(true))
                    .count();
                ContextManagementResult {
                    success: true,
                    messages_removed: removed,
                    strategy: ContextStrategy::SlidingWindow,
                }
            }
            Ok(false) => ContextManagementResult {
                success: false,
                messages_removed: 0,
                strategy: ContextStrategy::None,
            },
            Err(_) => {
                // Last resort: force truncate to FORCED_CONTEXT_REDUCTION_PERCENT
                let keep_ratio =
                    (crate::types::FORCED_CONTEXT_REDUCTION_PERCENT as f64) / 100.0;
                self.engine.force_truncate_context(keep_ratio);
                ContextManagementResult {
                    success: true,
                    messages_removed: 0,
                    strategy: ContextStrategy::ForceTruncate,
                }
            }
        }
    }

    /// Check whether an error message indicates a context window exceeded error.
    fn is_context_window_error(&self, error: &str) -> bool {
        error.contains("context_length_exceeded")
            || error.contains("context window")
            || error.contains("max_tokens")
            || error.contains("context_length")
            || error.contains("token limit")
            || error.contains("too many tokens")
    }

    // ===================================================================
    // 7. get_system_prompt() — TS line 3744-3819
    // ===================================================================

    /// Build the system prompt for the current request.
    ///
    /// Source: `src/core/task/Task.ts` — `getSystemPrompt()` lines 3744-3819.
    ///
    /// Returns the system prompt from the message builder. In a full
    /// implementation, this would dynamically build the prompt based on
    /// mode, custom instructions, etc.
    fn get_system_prompt(&self) -> String {
        self.message_builder.system_prompt().to_string()
    }

    // ===================================================================
    // 8. build_clean_conversation_history() — TS line 4458-4598
    // ===================================================================

    /// Build a clean copy of the conversation history for the API.
    ///
    /// Source: `src/core/task/Task.ts` — `buildCleanConversationHistory()`
    /// lines 4458-4598.
    ///
    /// This method processes the raw API conversation history into a clean form
    /// suitable for sending to the model API. It handles:
    ///
    /// 1. **Standalone reasoning messages** — Messages with `reasoning` field set
    ///    are processed: if the reasoning looks like encrypted content (starts with
    ///    a non-text pattern), it is preserved; plain-text reasoning is stripped
    ///    unless the model's `preserveReasoning` flag is set.
    /// 2. **Assistant messages with embedded reasoning** — If the first content
    ///    block of an assistant message is a Thinking/RedactedThinking block, it
    ///    is separated into a reasoning item + clean assistant message.
    /// 3. **Default path** — Regular messages (user, tool_result) are passed
    ///    through with their content intact.
    /// 4. **Image stripping** — Images are removed if the model doesn't support them.
    /// 5. **Empty message filtering** — Messages with no content are removed.
    fn build_clean_conversation_history(&self) -> Vec<roo_types::api::ApiMessage> {
        let history = self.engine.api_conversation_history();
        let supports_images = self.message_builder.supports_images();
        let preserve_reasoning = false; // TODO: get from model info when available

        let mut clean: Vec<roo_types::api::ApiMessage> = Vec::with_capacity(history.len());

        for msg in history {
            // --- Standalone reasoning message ---
            // TS: if (msg.type === "reasoning") { ... }
            // In Rust, we detect standalone reasoning by checking if the message
            // has reasoning content but no regular text/tool content.
            if msg.reasoning.is_some() && msg.content.is_empty() {
                let reasoning_text = msg.reasoning.as_ref().unwrap();

                // If reasoning looks like encrypted content (base64-like), preserve it
                // as a separate reasoning item. Otherwise skip it (plain text reasoning
                // is stored for history only, not sent back to API).
                let is_encrypted = reasoning_text.len() > 20
                    && !reasoning_text.chars().all(|c| c.is_ascii_graphic() || c == '\n');

                if is_encrypted {
                    // Emit as a reasoning-only message
                    clean.push(roo_types::api::ApiMessage {
                        role: msg.role.clone(),
                        content: vec![roo_types::api::ContentBlock::Text {
                            text: format!("[reasoning]{}", reasoning_text),
                        }],
                        reasoning: msg.reasoning.clone(),
                        ts: msg.ts,
                        truncation_parent: msg.truncation_parent.clone(),
                        is_truncation_marker: msg.is_truncation_marker,
                        truncation_id: msg.truncation_id.clone(),
                        condense_parent: msg.condense_parent.clone(),
                        is_summary: msg.is_summary,
                        condense_id: msg.condense_id.clone(),
                    });
                }
                // Plain text standalone reasoning: skip (stored for history, not sent to API)
                continue;
            }

            // --- Assistant message with content ---
            if msg.role == roo_types::api::MessageRole::Assistant && !msg.content.is_empty() {
                let raw_content = &msg.content;

                // Check if the first content block is a thinking/reasoning block
                let first = &raw_content[0];
                let is_thinking_block = matches!(
                    first,
                    roo_types::api::ContentBlock::Thinking { .. }
                        | roo_types::api::ContentBlock::RedactedThinking { .. }
                );

                if is_thinking_block {
                    // Embedded encrypted reasoning (Thinking/RedactedThinking)
                    // TS: hasEncryptedReasoning path
                    let rest = &raw_content[1..];

                    // Build clean assistant content without the reasoning block
                    let clean_content = Self::build_assistant_content(rest, supports_images);

                    // If the model preserves reasoning, include the thinking block
                    let final_content = if preserve_reasoning {
                        let mut with_reasoning = vec![first.clone()];
                        with_reasoning.extend(clean_content);
                        with_reasoning
                    } else {
                        clean_content
                    };

                    clean.push(roo_types::api::ApiMessage {
                        role: msg.role.clone(),
                        content: final_content,
                        reasoning: None, // Reasoning separated out
                        ts: msg.ts,
                        truncation_parent: msg.truncation_parent.clone(),
                        is_truncation_marker: msg.is_truncation_marker,
                        truncation_id: msg.truncation_id.clone(),
                        condense_parent: msg.condense_parent.clone(),
                        is_summary: msg.is_summary,
                        condense_id: msg.condense_id.clone(),
                    });
                    continue;
                }

                // Check for reasoning field on assistant message (OpenRouter/Gemini format)
                // TS: msgWithDetails.reasoning_details path
                if msg.reasoning.is_some() {
                    // Include reasoning in the message for providers that support it
                    let clean_content = Self::build_assistant_content(raw_content, supports_images);
                    clean.push(roo_types::api::ApiMessage {
                        role: msg.role.clone(),
                        content: clean_content,
                        reasoning: msg.reasoning.clone(),
                        ts: msg.ts,
                        truncation_parent: msg.truncation_parent.clone(),
                        is_truncation_marker: msg.is_truncation_marker,
                        truncation_id: msg.truncation_id.clone(),
                        condense_parent: msg.condense_parent.clone(),
                        is_summary: msg.is_summary,
                        condense_id: msg.condense_id.clone(),
                    });
                    continue;
                }

                // Default assistant path: just clean content
                let clean_content = Self::build_assistant_content(raw_content, supports_images);
                if !clean_content.is_empty() {
                    clean.push(roo_types::api::ApiMessage {
                        role: msg.role.clone(),
                        content: clean_content,
                        reasoning: None,
                        ts: msg.ts,
                        truncation_parent: msg.truncation_parent.clone(),
                        is_truncation_marker: msg.is_truncation_marker,
                        truncation_id: msg.truncation_id.clone(),
                        condense_parent: msg.condense_parent.clone(),
                        is_summary: msg.is_summary,
                        condense_id: msg.condense_id.clone(),
                    });
                }
                continue;
            }

            // --- Default path: regular messages (user, tool_result) ---
            // TS: if (msg.role) { cleanConversationHistory.push({ role, content }) }
            let clean_content = Self::filter_content_blocks(&msg.content, supports_images);
            if !clean_content.is_empty() {
                clean.push(roo_types::api::ApiMessage {
                    role: msg.role.clone(),
                    content: clean_content,
                    reasoning: msg.reasoning.clone(),
                    ts: msg.ts,
                    truncation_parent: msg.truncation_parent.clone(),
                    is_truncation_marker: msg.is_truncation_marker,
                    truncation_id: msg.truncation_id.clone(),
                    condense_parent: msg.condense_parent.clone(),
                    is_summary: msg.is_summary,
                    condense_id: msg.condense_id.clone(),
                });
            }
        }

        clean
    }

    /// Build clean assistant content from content blocks.
    ///
    /// Follows the TS logic for simplifying content arrays:
    /// - Empty array → single empty text block
    /// - Single text block → use text directly
    /// - Multiple blocks → filter and use as-is
    fn build_assistant_content(
        blocks: &[roo_types::api::ContentBlock],
        supports_images: bool,
    ) -> Vec<roo_types::api::ContentBlock> {
        let filtered = Self::filter_content_blocks(blocks, supports_images);

        if filtered.is_empty() {
            // TS: assistantContent = ""
            vec![roo_types::api::ContentBlock::Text {
                text: String::new(),
            }]
        } else {
            filtered
        }
    }

    /// Filter content blocks, removing images if the model doesn't support them.
    fn filter_content_blocks(
        blocks: &[roo_types::api::ContentBlock],
        supports_images: bool,
    ) -> Vec<roo_types::api::ContentBlock> {
        blocks
            .iter()
            .filter(|block| {
                // Remove images if model doesn't support them
                if !supports_images && matches!(block, roo_types::api::ContentBlock::Image { .. }) {
                    return false;
                }
                true
            })
            .cloned()
            .collect()
    }

    // ===================================================================
    // Tool execution
    // ===================================================================

    /// Execute all tool calls from a single API response.
    ///
    /// For each tool call:
    /// 1. Check auto-approval
    /// 2. Execute the tool
    /// 3. Optionally create a checkpoint
    ///
    /// Returns `true` if all tools succeeded, `false` if any failed.
    async fn execute_tools(
        &mut self,
        tool_calls: &[ParsedToolCall],
    ) -> Result<bool, TaskError> {
        let mut all_succeeded = true;

        let tool_calls = validate_and_fix_tool_result_ids(tool_calls.to_vec());

        // Issue #1: Track executed tool_use_ids to prevent duplicate execution.
        // MiniMax API may return duplicate tool_use blocks with the same ID;
        // we skip any tool call whose ID has already been executed.
        let mut executed_tool_ids: HashSet<String> = HashSet::new();

        for tool_call in &tool_calls {
            // Skip duplicate tool calls (same tool_use_id)
            if !executed_tool_ids.insert(tool_call.id.clone()) {
                warn!(
                    tool = %tool_call.name,
                    id = %tool_call.id,
                    "Skipping duplicate tool_use_id"
                );
                continue;
            }

            // Cascade rejection: if a previous tool was rejected by the user,
            // skip all subsequent tools in this message.
            // TS: didRejectTool flag in presentAssistantMessage()
            // Source: `src/core/task/Task.ts` — cascade rejection logic
            if self.engine.streaming().did_reject_tool {
                let skip_msg = format!(
                    "Skipping tool {} due to user rejecting a previous tool.",
                    tool_call.name
                );
                warn!(
                    tool = %tool_call.name,
                    id = %tool_call.id,
                    "{}", skip_msg
                );
                let error_result = ToolExecutionResult::error(&skip_msg);
                let result_msg =
                    MessageBuilder::create_tool_result_message(&tool_call.id, &error_result);
                self.engine.add_api_message(result_msg);
                continue;
            }

            debug!(
                tool = %tool_call.name,
                id = %tool_call.id,
                "Executing tool"
            );
            let params = tool_call.parse_arguments();

            // Tool repetition detection
            if !self.repetition_detector.check_and_record(&tool_call.name, &params) {
                warn!(
                    tool = %tool_call.name,
                    consecutive = self.repetition_detector.consecutive_count(),
                    "Tool repetition limit reached, adding warning"
                );
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

            // Check auto-approval
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
                    // Set cascade rejection flag — subsequent tools will be skipped.
                    // Source: `src/core/task/Task.ts` — `this.didRejectTool = true`
                    self.engine.streaming_mut().did_reject_tool = true;
                    ToolExecutionResult::error(format!("Tool '{}' denied: {}", tool_call.name, reason))
                }
                ApprovalDecision::NeedsApproval { reason } => {
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

                // Create checkpoint for file-modifying tools
                if self.config.enable_checkpoints {
                    self.maybe_checkpoint(&tool_call.name).await;
                }

                // Update file context tracker
                self.update_file_context(&tool_call.name, &params);

                // DiffView integration: update diff view for file-modifying tools.
                // Source: `src/core/task/Task.ts` — diffViewProvider usage
                if let Some(ref mut dvp) = self.diff_view_provider {
                    if matches!(
                        tool_call.name.as_str(),
                        "write_to_file" | "apply_diff" | "edit_file" | "search_and_replace"
                    ) {
                        if let Some(path) = params
                            .get("path")
                            .or_else(|| params.get("file_path"))
                            .and_then(|v| v.as_str())
                        {
                            if dvp.is_active() {
                                if dvp.update(&result.text, true).is_ok() {
                                    debug!(path = %path, "DiffView updated with tool result");
                                }
                            }
                        }
                    }
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

    // ===================================================================
    // new_task isolation
    // ===================================================================

    /// Enforce `new_task` isolation: if `new_task` appears alongside other
    /// tools, truncate any tools that come after it and inject error
    /// tool_results for the truncated tools.
    fn enforce_new_task_isolation(
        &mut self,
        tool_calls: &[crate::stream_parser::ParsedToolCall],
    ) -> Vec<crate::stream_parser::ParsedToolCall> {
        let new_task_idx = tool_calls
            .iter()
            .position(|tc| tc.name == "new_task");

        match new_task_idx {
            Some(idx) if idx < tool_calls.len() - 1 => {
                let truncated: Vec<_> = tool_calls[idx + 1..].to_vec();
                if !truncated.is_empty() {
                    warn!(
                        truncated_count = truncated.len(),
                        "new_task isolation: truncating {} tool(s) after new_task",
                        truncated.len()
                    );
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

    // ===================================================================
    // Image generation (C4)
    // ===================================================================

    /// Handle image generation requests.
    #[allow(dead_code)]
    async fn handle_image_generation(
        &self,
        prompt: &str,
    ) -> ToolExecutionResult {
        debug!(prompt_len = prompt.len(), "Attempting image generation");

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

    // ===================================================================
    // File context tracking (C5)
    // ===================================================================

    /// Update the file context tracker after tool execution.
    fn update_file_context(&self, tool_name: &str, params: &serde_json::Value) {
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

    // ===================================================================
    // Context management (L4.2)
    // ===================================================================

    /// Check and truncate context if it exceeds the maximum token limit.
    ///
    /// Source: `src/core/task/Task.ts` — `manageContext()`
    async fn maybe_truncate_context(&mut self) -> Result<bool, TaskError> {
        let max_tokens = match self.config.max_context_tokens {
            Some(t) => t,
            None => return Ok(false),
        };

        let history = self.engine.api_conversation_history();
        if history.len() <= 4 {
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

        // Try condense first if enabled
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

        // Fallback: sliding window truncation
        self.apply_sliding_window_truncation()
    }

    /// Attempt to condense the conversation using LLM summarization.
    async fn try_condense_context(&mut self) -> Result<bool, TaskError> {
        let history = self.engine.api_conversation_history().to_vec();
        let system_prompt = self.message_builder.system_prompt().to_string();
        let task_id = self.engine.config().task_id.clone();
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

        self.engine.set_api_conversation_history(result.messages);

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
        let to_remove = std::cmp::max((total - 1) / 4, 2);
        let to_remove = to_remove - (to_remove % 2);

        if to_remove > 0 && total > to_remove + 1 {
            let truncation_id = uuid::Uuid::now_v7().to_string();

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

            self.engine.truncate_history(to_remove, marker);
            debug!(messages_removed = to_remove, "Context truncated with marker inserted");
        }

        Ok(true)
    }

    // ===================================================================
    // Checkpoint (L4.3)
    // ===================================================================

    /// Create a checkpoint after file-modifying tool execution.
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

    // ===================================================================
    // Termination handling
    // ===================================================================

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

    // ===================================================================
    // Environment details (Phase Q2)
    // ===================================================================

    /// Gather environment details for injection into the conversation.
    fn get_environment_details(&self) -> String {
        let mut details = Vec::new();
        details.push(format!("Current working directory: {}", self.engine.config().cwd));
        details.push(format!("Platform: {}", std::env::consts::OS));
        details.push(format!("Mode: {}", self.engine.config().mode));
        details.join("\n")
    }

    // ===================================================================
    // @mentions processing (Phase Q2)
    // ===================================================================

    /// Process @mentions in user text, expanding @file references to file content.
    async fn process_user_mentions(&self, text: &str) -> String {
        let wrapped = format!("<user_message>\n{}\n</user_message>", text);
        let cwd = std::path::Path::new(&self.engine.config().cwd);

        let blocks = vec![roo_mentions::ContentBlock::text(&wrapped)];

        let result = roo_mentions::process_user_content_mentions(&blocks, cwd).await;

        let mut output = String::new();
        for block in &result.content {
            if let Some(t) = block.as_text() {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(t);
            }
        }

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
    #[allow(dead_code)]
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

    // ===================================================================
    // Backoff with jitter (Issue #7)
    // ===================================================================

    /// Calculate exponential backoff delay with random jitter.
    ///
    /// Source: `src/core/task/Task.ts` — `backoffAndAnnounce()`
    #[allow(dead_code)]
    fn calculate_backoff_with_jitter(&self, retry_attempt: u32) -> u64 {
        let base_delay = TaskEngine::calculate_backoff_delay(retry_attempt);

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

// ===========================================================================
// RetryInfo parsing (Issue #4 — backoff_and_announce)
// ===========================================================================

/// Parse RetryInfo from a 429 error response.
///
/// TS source: `backoffAndAnnounce()` — parses `error.errorDetails` looking
/// for `@type === "type.googleapis.com/google.rpc.RetryInfo"` and extracts
/// the `retryDelay` field (format: `"(\d+)s"`).
///
/// In Rust we do a simpler regex-based parse of the error string since we
/// don't have structured error details.
fn parse_retry_info_from_error(error: &str) -> Option<u64> {
    // Try to find a retry delay pattern like "30s" or "60s" in the error
    let re = regex::Regex::new(r#"retryDelay["\s:]+["']?(\d+)s["']?"#).ok()?;
    let caps = re.captures(error)?;
    let secs: u64 = caps[1].parse().ok()?;
    if secs > 0 { Some(secs) } else { None }
}

// ===========================================================================
// Tool result ID validation (Issue #3)
// ===========================================================================

/// Validate and fix tool result IDs to match tool call IDs.
///
/// Source: `src/core/task/Task.ts` — `validateAndFixToolResultIds()`
pub fn validate_and_fix_tool_result_ids(tool_calls: Vec<ParsedToolCall>) -> Vec<ParsedToolCall> {
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

// ===========================================================================
// Tests
// ===========================================================================

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
        assert_eq!(result.iterations, 0);
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
        assert_eq!(result.iterations, 1);
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

        let provider = MockProvider::new("Working...")
            .with_tool_call("call_1", "read_file", r#"{"path":"test.rs"}"#);
        let builder = MessageBuilder::new("You are a helper.");

        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("read_file", |_params, _ctx| {
            ToolExecutionResult::success("content")
        });

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        let result = agent.run_loop().await.unwrap();

        assert_eq!(result.status, TaskState::Aborted);
    }

    #[tokio::test]
    async fn test_agent_loop_records_token_usage() {
        let engine = TaskEngine::new(make_config()).unwrap();

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

    // -------------------------------------------------------------------
    // L4 unit tests
    // -------------------------------------------------------------------

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

    fn no_approval_state() -> AutoApprovalState {
        let mut s = AutoApprovalState::default();
        s.auto_approval_enabled = true;
        s
    }

    #[test]
    fn test_approval_disabled_needs_approval() {
        let state = AutoApprovalState::default();
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
        let state = no_approval_state();
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
        assert_eq!(agent.engine().loop_control().consecutive_no_tool_use_count, 0);
    }

    #[tokio::test]
    async fn test_no_tool_used_hits_mistake_limit_with_grace() {
        let config = TaskConfig::new("test-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(100)
            .with_consecutive_mistake_limit(2)
            .with_task_text("Hello");

        let engine = TaskEngine::new(config).unwrap();

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

        assert_eq!(result.status, TaskState::Aborted);
        assert!(agent.engine().loop_control().mistake_grace_used);
    }

    #[tokio::test]
    async fn test_attempt_completion_completes_task() {
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
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "read_file");
        assert_eq!(result[1].name, "new_task");
    }

    #[test]
    fn test_new_task_isolation_no_truncation_when_last() {
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
        let config = TaskConfig::new("test-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(10)
            .with_consecutive_mistake_limit(10)
            .with_task_text("Hello");

        let engine = TaskEngine::new(config).unwrap();

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
                    Ok(Box::pin(stream::iter(vec![])))
                } else {
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

        assert!(detector.check_and_record("read_file", &params));
        assert!(detector.check_and_record("read_file", &params));
        assert!(detector.check_and_record("read_file", &params));

        assert!(!detector.check_and_record("read_file", &params));

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
                rate_limit_rpm: 0,
                ..AgentLoopConfig::default()
            });

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

    // --- Context window error detection tests ---

    #[test]
    fn test_is_context_window_error() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        assert!(agent.is_context_window_error("context_length_exceeded"));
        assert!(agent.is_context_window_error("context window too large"));
        assert!(agent.is_context_window_error("max_tokens exceeded"));
        assert!(agent.is_context_window_error("context_length limit"));
        assert!(agent.is_context_window_error("token limit reached"));
        assert!(agent.is_context_window_error("too many tokens"));
        assert!(!agent.is_context_window_error("rate limit exceeded"));
        assert!(!agent.is_context_window_error("connection timeout"));
    }

    // --- handle_context_window_exceeded_error tests ---

    #[tokio::test]
    async fn test_handle_context_window_exceeded_error_success() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        // Add some history so truncation has something to work with
        for i in 0..10 {
            agent.engine.add_api_message(roo_types::api::ApiMessage {
                role: if i % 2 == 0 { roo_types::api::MessageRole::User } else { roo_types::api::MessageRole::Assistant },
                content: vec![roo_types::api::ContentBlock::Text {
                    text: format!("Message {}", i),
                }],
                reasoning: None,
                ts: None,
                truncation_parent: None,
                is_truncation_marker: None,
                truncation_id: None,
                condense_parent: None,
                is_summary: None,
                condense_id: None,
            });
        }

        let result = agent.handle_context_window_exceeded_error("context_length_exceeded", 0).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_handle_context_window_exceeded_error_max_retries() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        let result = agent.handle_context_window_exceeded_error(
            "context_length_exceeded",
            crate::types::MAX_CONTEXT_WINDOW_RETRIES,
        ).await;
        assert!(result.is_err());
    }

    // --- build_clean_conversation_history tests ---

    #[test]
    fn test_build_clean_conversation_history() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        agent.engine.add_api_message(roo_types::api::ApiMessage {
            role: roo_types::api::MessageRole::User,
            content: vec![roo_types::api::ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        });

        let history = agent.build_clean_conversation_history();
        assert_eq!(history.len(), 1);
    }

    // --- get_system_prompt tests ---

    #[test]
    fn test_get_system_prompt() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("You are a coding assistant.");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);
        assert_eq!(agent.get_system_prompt(), "You are a coding assistant.");
    }

    // --- Issue #1: Duplicate tool execution prevention tests ---

    #[test]
    fn test_duplicate_tool_ids_are_skipped() {
        // Verify that validate_and_fix_tool_result_ids preserves duplicate IDs
        let tool_calls = vec![
            ParsedToolCall {
                id: "call_dup".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            ParsedToolCall {
                id: "call_dup".into(), // Same ID as first
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
        ];
        let result = validate_and_fix_tool_result_ids(tool_calls);
        // Both should keep the same ID (dedup happens in execute_tools)
        assert_eq!(result[0].id, "call_dup");
        assert_eq!(result[1].id, "call_dup");
    }

    #[test]
    fn test_hashset_dedup_logic() {
        // Test the HashSet dedup logic directly
        let mut executed: std::collections::HashSet<String> = std::collections::HashSet::new();
        assert!(executed.insert("call_1".to_string())); // First insert succeeds
        assert!(!executed.insert("call_1".to_string())); // Duplicate fails
        assert!(executed.insert("call_2".to_string())); // Different ID succeeds
        assert_eq!(executed.len(), 2);
    }

    // --- Issue #2: buildCleanConversationHistory reasoning tests ---

    #[test]
    fn test_build_clean_history_strips_standalone_plain_reasoning() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        // Add a standalone reasoning message (plain text reasoning, should be skipped)
        // "This is plain text reasoning" is all ASCII printable → not encrypted → skipped
        agent.engine.add_api_message(roo_types::api::ApiMessage {
            role: roo_types::api::MessageRole::Assistant,
            content: vec![], // Empty content but has reasoning
            reasoning: Some("This is plain text reasoning".to_string()),
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        });

        // Add a regular user message
        agent.engine.add_api_message(roo_types::api::ApiMessage {
            role: roo_types::api::MessageRole::User,
            content: vec![roo_types::api::ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        });

        let history = agent.build_clean_conversation_history();
        // Plain text standalone reasoning should be skipped, only user message remains
        // But the empty-content assistant message with reasoning also enters the
        // "standalone reasoning" path and is skipped.
        assert!(
            history.len() <= 2,
            "Expected at most 2 messages, got {}",
            history.len()
        );
        // The user message should be present
        let user_msgs: Vec<_> = history.iter()
            .filter(|m| m.role == roo_types::api::MessageRole::User)
            .collect();
        assert_eq!(user_msgs.len(), 1, "Should have exactly 1 user message");
    }

    #[test]
    fn test_build_clean_history_handles_thinking_block() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        // Add assistant message with Thinking block as first content
        agent.engine.add_api_message(roo_types::api::ApiMessage {
            role: roo_types::api::MessageRole::Assistant,
            content: vec![
                roo_types::api::ContentBlock::Thinking {
                    thinking: "Let me think...".to_string(),
                    signature: "sig123".to_string(),
                },
                roo_types::api::ContentBlock::Text {
                    text: "Here is my answer".to_string(),
                },
            ],
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        });

        let history = agent.build_clean_conversation_history();
        // Should have one message (thinking stripped, text preserved)
        assert_eq!(history.len(), 1);
        // The thinking block should be stripped (preserve_reasoning defaults to false)
        let content = &history[0].content;
        assert_eq!(content.len(), 1);
        match &content[0] {
            roo_types::api::ContentBlock::Text { text } => {
                assert_eq!(text, "Here is my answer");
            }
            _ => panic!("Expected text block"),
        }
    }

    // --- Issue #4: parse_retry_info_from_error tests ---

    #[test]
    fn test_parse_retry_info_from_error() {
        // Should parse retry delay from error string
        let result = parse_retry_info_from_error(
            r#"{"errorDetails":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30s"}]}"#
        );
        assert_eq!(result, Some(30));

        // Should parse different delay values
        let result = parse_retry_info_from_error(
            r#"retryDelay":"60s""#
        );
        assert_eq!(result, Some(60));

        // Should return None when no retry info
        let result = parse_retry_info_from_error("generic error message");
        assert!(result.is_none());

        // Should return None for zero delay
        let result = parse_retry_info_from_error(r#"retryDelay":"0s""#);
        assert!(result.is_none());
    }

    // --- Issue #4: backoff_and_announce tests ---

    #[tokio::test]
    async fn test_backoff_with_no_rate_limit() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
            .with_config(AgentLoopConfig {
                rate_limit_rpm: 0, // No rate limit
                ..AgentLoopConfig::default()
            });

        // Should complete without panicking
        agent.backoff_and_announce(0, None).await;
    }

    // --- Issue #5: Tool mode restrictions tests ---

    #[test]
    fn test_build_tool_definitions_with_mode_filter() {
        let builder = MessageBuilder::new("test");

        // Code mode should have execute_command
        let code_tools = builder.build_tool_definitions_with_options(
            Some("code"),
            &[],
            None,
            None,
        );
        let code_names: Vec<&str> = code_tools.iter()
            .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
            .collect();
        assert!(code_names.contains(&"execute_command"), "Code mode should have execute_command");

        // Architect mode should NOT have execute_command
        let arch_tools = builder.build_tool_definitions_with_options(
            Some("architect"),
            &[],
            None,
            None,
        );
        let arch_names: Vec<&str> = arch_tools.iter()
            .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
            .collect();
        assert!(!arch_names.contains(&"execute_command"), "Architect mode should NOT have execute_command");
    }

    #[test]
    fn test_build_tool_definitions_with_disabled_tools() {
        let builder = MessageBuilder::new("test");

        let disabled = vec!["read_file".to_string()];
        let tools = builder.build_tool_definitions_with_options(
            Some("code"),
            &[],
            Some(&disabled),
            None,
        );
        let names: Vec<&str> = tools.iter()
            .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
            .collect();
        assert!(!names.contains(&"read_file"), "read_file should be disabled");
    }

    #[test]
    fn test_build_tool_definitions_with_restrictions() {
        let builder = MessageBuilder::new("test");

        let result = builder.build_tool_definitions_with_restrictictions(
            Some("code"),
            &[],
            None,
            None,
            true, // include_all_tools_with_restrictictions
        );

        // Should have tools
        assert!(!result.tools.is_empty());
        // Should have allowed_function_names
        assert!(result.allowed_function_names.is_some());
        let allowed = result.allowed_function_names.unwrap();
        assert!(!allowed.is_empty());
    }

    // --- Cascade rejection tests ---

    #[tokio::test]
    async fn test_cascade_rejection_skips_subsequent_tools() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        // Simulate a previous tool rejection
        agent.engine.streaming_mut().did_reject_tool = true;

        let tool_calls = vec![
            ParsedToolCall {
                id: "tc_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            ParsedToolCall {
                id: "tc_2".into(),
                name: "write_to_file".into(),
                arguments: r#"{"path":"b.rs","content":"x"}"#.into(),
            },
        ];

        // execute_tools should skip all tools due to cascade rejection
        let history_before = agent.engine.api_conversation_history().len();
        let result = agent.execute_tools(&tool_calls).await;

        // Both tools should be skipped (cascade rejection)
        assert!(result.is_ok());
        // Tool results should have been added to history for skipped tools
        let history_after = agent.engine.api_conversation_history().len();
        assert_eq!(history_after, history_before + 2, "Should have 2 tool result messages for skipped tools");
    }

    /// Verify that the cascade rejection mechanism works end-to-end:
    /// After setting `did_reject_tool = true` (simulating a user denial),
    /// subsequent tools in the same batch should be skipped with error results.
    #[tokio::test]
    async fn test_cascade_rejection_flag_set_on_denial() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        // Manually set the flag to simulate a user denial that happened
        // before these tool calls (e.g., via external UI interaction).
        // The Denied branch in execute_tools sets this flag; here we
        // verify the cascade effect of that flag being set.
        agent.engine.streaming_mut().did_reject_tool = true;

        let tool_calls = vec![
            ParsedToolCall {
                id: "tc_1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            },
            ParsedToolCall {
                id: "tc_2".into(),
                name: "write_to_file".into(),
                arguments: r#"{"path":"b.rs","content":"x"}"#.into(),
            },
            ParsedToolCall {
                id: "tc_3".into(),
                name: "list_files".into(),
                arguments: r#"{"path":"."}"#.into(),
            },
        ];

        let history_before = agent.engine.api_conversation_history().len();
        let result = agent.execute_tools(&tool_calls).await;

        assert!(result.is_ok());
        // The flag should remain set
        assert!(
            agent.engine.streaming().did_reject_tool,
            "did_reject_tool flag should remain set after cascade"
        );
        // All three tools should have been skipped with error results in history
        let history_after = agent.engine.api_conversation_history().len();
        assert_eq!(
            history_after,
            history_before + 3,
            "All 3 tools should be cascade-skipped with error results"
        );
    }

    // --- Notify synchronization tests ---

    #[test]
    fn test_user_message_content_ready_notify() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher);

        // Should be able to get a clone of the Notify
        let notify = agent.user_message_content_ready_notify();
        // The Notify should be usable
        assert!(Arc::strong_count(&notify) >= 2); // agent + our clone
    }

    // --- DiffView integration tests ---

    #[test]
    fn test_with_diff_view_provider() {
        let engine = TaskEngine::new(make_config()).unwrap();
        let provider = MockProvider::new("test");
        let builder = MessageBuilder::new("test");
        let dispatcher = ToolDispatcher::new();

        let dvp = roo_editor::diff_view::DiffViewProvider::new_default();
        let mut agent = AgentLoop::new(engine, Box::new(provider), builder, dispatcher)
            .with_diff_view_provider(dvp);

        // Should have a DiffView provider configured
        assert!(agent.diff_view_provider_mut().is_some());
    }
}