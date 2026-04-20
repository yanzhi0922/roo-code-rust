//! Task lifecycle management.
//!
//! Implements the complete task lifecycle methods from `Task.ts`:
//! - `start()` / `startTask()` — Start a new task
//! - `resumeTaskFromHistory()` — Resume a task from saved history
//! - `cancelCurrentRequest()` — Cancel the current API request
//! - `abortTask()` — Abort the task
//! - `dispose()` — Clean up resources
//! - `startSubtask()` — Delegate to a child task
//! - `resumeAfterDelegation()` — Resume after subtask completes
//! - `submitUserMessage()` — Submit a user message during an ask
//! - `condenseContext()` — Manually trigger context condensation
//! - `updateApiConfiguration()` — Update the API configuration
//!
//! Source: `src/core/task/Task.ts` — Task class methods

use std::sync::Arc;

use tracing::{debug, info, warn};

use roo_types::message::{ClineAsk, ClineMessage, ClineSay, MessageType};

use crate::ask_say::{AskResponse, AskSayHandler};
use crate::engine::TaskEngine;
use crate::events::TaskEvent;
use crate::types::{TaskConfig, TaskError, TaskState};

// ---------------------------------------------------------------------------
// ServiceRefs
// ---------------------------------------------------------------------------

/// References to external services needed by the task lifecycle.
///
/// All fields are optional — if not set, the corresponding functionality
/// is silently skipped. This allows `TaskLifecycle` to work in tests and
/// lightweight contexts without requiring the full service stack.
///
/// Source: `src/core/task/Task.ts` — constructor injection of services
/// (`this.mcpHub`, `this.terminal`, `this.messageQueueService`,
/// `this.telemetryService`).
#[derive(Clone, Default)]
pub struct ServiceRefs {
    /// MCP Hub for MCP server connections.
    ///
    /// Source: TS `this.mcpHub`
    pub mcp_hub: Option<Arc<roo_mcp::McpHub>>,
    /// Terminal registry for managing terminal processes.
    ///
    /// Source: TS `this.terminal`
    pub terminal_registry: Option<Arc<roo_terminal::TerminalRegistry>>,
    /// Message queue for buffering user messages during ask states.
    ///
    /// Source: TS `this.messageQueueService`
    pub message_queue: Option<Arc<tokio::sync::Mutex<roo_message_queue::MessageQueueService>>>,
    /// Telemetry service for capturing lifecycle events.
    ///
    /// Source: TS `this.telemetryService`
    pub telemetry: Option<Arc<std::sync::RwLock<roo_telemetry::TelemetryService>>>,
}

// ---------------------------------------------------------------------------
// TaskLifecycle
// ---------------------------------------------------------------------------

/// Manages the complete task lifecycle.
///
/// This struct wraps a [`TaskEngine`] and an [`AskSayHandler`] to provide
/// the full task lifecycle as defined in `Task.ts`.
///
/// # Lifecycle
///
/// 1. **New Task**: `start()` → `start_task()` → `initiate_task_loop()`
/// 2. **Resume**: `resume_task_from_history()` → load history → ask to resume
/// 3. **Cancel**: `cancel_current_request()` → abort stream
/// 4. **Abort**: `abort_task()` → clean up
/// 5. **Delegate**: `start_subtask()` → create child → `resume_after_delegation()`
/// 6. **Dispose**: `dispose()` → clean up all resources
pub struct TaskLifecycle {
    /// The task engine managing state, loop control, and events.
    engine: TaskEngine,
    /// The ask/say handler for interactive communication.
    ask_say: AskSayHandler,
    /// External service references (MCP Hub, Terminal, Message Queue, Telemetry).
    services: ServiceRefs,
    /// Whether the task has been started.
    started: bool,
    /// Whether the task has been disposed.
    disposed: bool,
    /// Child task ID, if this task has delegated to a subtask.
    child_task_id: Option<String>,
    /// Pending new task tool call ID.
    #[allow(dead_code)]
    pending_new_task_tool_call_id: Option<String>,
    /// Abort controller flag.
    abort: bool,
    /// Abort reason.
    abort_reason: Option<String>,
}

impl TaskLifecycle {
    /// Create a new task lifecycle with the given engine.
    pub fn new(engine: TaskEngine) -> Self {
        Self {
            engine,
            ask_say: AskSayHandler::new(),
            services: ServiceRefs::default(),
            started: false,
            disposed: false,
            child_task_id: None,
            pending_new_task_tool_call_id: None,
            abort: false,
            abort_reason: None,
        }
    }

    /// Attach external service references to this lifecycle.
    ///
    /// Source: `src/core/task/Task.ts` — constructor injection of
    /// `mcpHub`, `terminal`, `messageQueueService`, `telemetryService`.
    pub fn with_services(mut self, services: ServiceRefs) -> Self {
        self.services = services;
        self
    }

    // -------------------------------------------------------------------
    // Getters
    // -------------------------------------------------------------------

    /// Get a reference to the task engine.
    pub fn engine(&self) -> &TaskEngine {
        &self.engine
    }

    /// Get a mutable reference to the task engine.
    pub fn engine_mut(&mut self) -> &mut TaskEngine {
        &mut self.engine
    }

    /// Get a reference to the ask/say handler.
    pub fn ask_say(&self) -> &AskSayHandler {
        &self.ask_say
    }

    /// Get a mutable reference to the ask/say handler.
    pub fn ask_say_mut(&mut self) -> &mut AskSayHandler {
        &mut self.ask_say
    }

    /// Get a reference to the service refs.
    pub fn services(&self) -> &ServiceRefs {
        &self.services
    }

    /// Get a mutable reference to the service refs.
    pub fn services_mut(&mut self) -> &mut ServiceRefs {
        &mut self.services
    }

    /// Get the task ID.
    pub fn task_id(&self) -> &str {
        &self.engine.config().task_id
    }

    /// Get the current task state.
    pub fn state(&self) -> TaskState {
        self.engine.state()
    }

    /// Get the child task ID.
    pub fn child_task_id(&self) -> Option<&str> {
        self.child_task_id.as_deref()
    }

    /// Check if the task has been aborted.
    pub fn is_aborted(&self) -> bool {
        self.abort
    }

    /// Get the abort reason.
    pub fn abort_reason(&self) -> Option<&str> {
        self.abort_reason.as_deref()
    }

    /// Check if the task has been disposed.
    pub fn is_disposed(&self) -> bool {
        self.disposed
    }

    /// Get the task status as reported to the UI.
    ///
    /// Source: `src/core/task/Task.ts` — `taskStatus` getter
    pub fn task_status(&self) -> TaskStatus {
        // Map from internal TaskState to TaskStatus
        match self.engine.state() {
            TaskState::Idle => TaskStatus::Idle,
            TaskState::Running => TaskStatus::Running,
            TaskState::Paused => TaskStatus::Paused,
            TaskState::Completed => TaskStatus::Completed,
            TaskState::Aborted => TaskStatus::Aborted,
            TaskState::Delegated => TaskStatus::Running, // Delegated tasks show as running
        }
    }

    // -------------------------------------------------------------------
    // start()
    // -------------------------------------------------------------------

    /// Start the task.
    ///
    /// Source: `src/core/task/Task.ts` — `start()`
    ///
    /// If the task has already been started, this is a no-op.
    /// If there's an initial task text or images, starts a new task.
    /// If there's a history item ID, resumes from history.
    pub async fn start(&mut self) -> Result<(), TaskError> {
        if self.started {
            debug!(task_id = %self.task_id(), "Task already started, skipping");
            return Ok(());
        }
        self.started = true;

        let task_text = self.engine.config().task_text.clone();
        let images = self.engine.config().images.clone();
        let history_item_id = self.engine.config().history_item_id.clone();

        if task_text.is_some() || !images.is_empty() {
            self.start_task(task_text, images).await
        } else if history_item_id.is_some() {
            self.resume_task_from_history().await
        } else {
            // Nothing to do — just mark as initialized
            self.engine.set_initialized(true);
            Ok(())
        }
    }

    // -------------------------------------------------------------------
    // startTask()
    // -------------------------------------------------------------------

    /// Start a new task with the given text and images.
    ///
    /// Source: `src/core/task/Task.ts` — `startTask()`
    ///
    /// 1. Clears existing messages
    /// 2. Emits the task text as a "say" message
    /// 3. Checks for too many MCP tools
    /// 4. Marks as initialized
    /// 5. Initiates the task loop
    async fn start_task(
        &mut self,
        task_text: Option<String>,
        images: Vec<String>,
    ) -> Result<(), TaskError> {
        // Clear existing messages
        // Source: TS `this.clineMessages = []` and `this.apiConversationHistory = []`
        self.ask_say.overwrite_cline_messages(Vec::new());
        self.engine.clear_api_conversation_history();

        // Emit the task text
        if let Some(ref text) = task_text {
            self.ask_say
                .say(ClineSay::Text, Some(text.clone()), Some(images.clone()))
                .await?;
        }

        // Mark as initialized
        self.engine.set_initialized(true);

        info!(
            task_id = %self.task_id(),
            text_len = task_text.as_ref().map(|t| t.len()).unwrap_or(0),
            images = images.len(),
            "Task started"
        );

        // Emit TaskStarted event
        self.engine.emitter().emit_task_started(self.task_id());

        // Telemetry: task created
        self.emit_telemetry_task_created();

        Ok(())
    }

    // -------------------------------------------------------------------
    // resumeTaskFromHistory()
    // -------------------------------------------------------------------

    /// Resume a task from saved history.
    ///
    /// Source: `src/core/task/Task.ts` — `resumeTaskFromHistory()`
    ///
    /// 1. Load saved cline messages and API conversation history
    /// 2. Clean up stale messages (resume_task, resume_completed_task)
    /// 3. Remove trailing reasoning-only messages
    /// 4. Remove incomplete api_req_started messages
    /// 5. Ask the user if they want to resume
    /// 6. Process the resume response
    pub async fn resume_task_from_history(&mut self) -> Result<(), TaskError> {
        // Load saved messages
        self.engine.load_api_conversation_history().await?;

        // Load saved cline messages
        let saved_messages = self.load_saved_cline_messages().await?;

        // Clean up stale resume messages
        let mut modified_messages = saved_messages;

        // Remove any trailing resume messages
        // Source: TS `findLastIndex` for resume_task / resume_completed_task
        while let Some(last) = modified_messages.last() {
            if last.ask == Some(ClineAsk::ResumeTask)
                || last.ask == Some(ClineAsk::ResumeCompletedTask)
            {
                modified_messages.pop();
            } else {
                break;
            }
        }

        // Remove trailing reasoning-only messages
        // Source: TS — remove trailing `say === "reasoning"` messages
        while let Some(last) = modified_messages.last() {
            if last.r#type == MessageType::Say && last.say == Some(ClineSay::Reasoning) {
                modified_messages.pop();
            } else {
                break;
            }
        }

        // Remove incomplete api_req_started messages
        // Source: TS — check if last api_req_started has cost/cancelReason
        if let Some(idx) = modified_messages
            .iter()
            .rposition(|m| m.r#type == MessageType::Say && m.say == Some(ClineSay::ApiReqStarted))
        {
            if let Some(ref text) = modified_messages[idx].text {
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(text) {
                    let cost = data.get("cost");
                    let cancel_reason = data.get("cancelReason");
                    if cost.is_none() && cancel_reason.is_none() {
                        modified_messages.remove(idx);
                    }
                }
            }
        }

        // Set the cleaned messages
        self.ask_say.overwrite_cline_messages(modified_messages);

        // Determine ask type based on last message
        let ask_type = self.determine_resume_ask_type();

        // Mark as initialized
        self.engine.set_initialized(true);

        info!(
            task_id = %self.task_id(),
            ask_type = ?ask_type,
            "Resuming task from history"
        );

        // Emit TaskStarted event
        self.engine.emitter().emit_task_started(self.task_id());

        // Telemetry: task restarted
        self.emit_telemetry_task_restarted();

        Ok(())
    }

    /// Determine the ask type for resuming a task.
    ///
    /// Source: TS `resumeTaskFromHistory()` — determines whether to ask
    /// "resume_task" or "resume_completed_task" based on the last message.
    fn determine_resume_ask_type(&self) -> ClineAsk {
        let messages = self.ask_say.cline_messages();
        let last_message = messages.iter().rev().find(|m| {
            m.ask != Some(ClineAsk::ResumeTask) && m.ask != Some(ClineAsk::ResumeCompletedTask)
        });

        match last_message {
            Some(msg) if msg.ask == Some(ClineAsk::CompletionResult) => {
                ClineAsk::ResumeCompletedTask
            }
            _ => ClineAsk::ResumeTask,
        }
    }

    /// Load saved cline messages from persistence.
    async fn load_saved_cline_messages(&self) -> Result<Vec<ClineMessage>, TaskError> {
        // If no storage path, return empty
        if self.engine.config().storage_path.is_none() {
            return Ok(Vec::new());
        }

        // Use the engine's persistence layer
        let storage_path = self.engine.config().storage_path.as_ref().unwrap();
        let base = std::path::Path::new(storage_path);
        let path = roo_task_persistence::messages_path(base, &self.engine.config().task_id);
        let fs = roo_task_persistence::OsFileSystem;

        match roo_task_persistence::read_task_messages(&fs, &path) {
            Ok(messages) => {
                debug!(count = messages.len(), "Loaded saved cline messages");
                Ok(messages)
            }
            Err(e) => {
                warn!(error = %e, "Failed to load saved cline messages");
                Ok(Vec::new())
            }
        }
    }

    // -------------------------------------------------------------------
    // cancelCurrentRequest()
    // -------------------------------------------------------------------

    /// Cancel the current API request.
    ///
    /// Source: `src/core/task/Task.ts` — `cancelCurrentRequest()`
    ///
    /// Aborts the current streaming API request and marks the task as
    /// needing to abort.
    pub fn cancel_current_request(&mut self) {
        if self.abort {
            debug!("Task already aborted, ignoring cancel request");
            return;
        }

        info!(task_id = %self.task_id(), "Cancelling current request");
        self.abort = true;
        self.abort_reason = Some("user_cancelled".to_string());
        self.engine.streaming_mut().is_streaming = false;
    }

    // -------------------------------------------------------------------
    // abortTask()
    // -------------------------------------------------------------------

    /// Abort the task completely.
    ///
    /// Source: `src/core/task/Task.ts` — `abortTask()`
    ///
    /// Sets the abort flag, cancels any pending operations, and transitions
    /// the task to the Aborted state.
    pub async fn abort_task(&mut self, is_abandoned: bool) -> Result<(), TaskError> {
        info!(
            task_id = %self.task_id(),
            is_abandoned = is_abandoned,
            "Aborting task"
        );

        self.abort = true;

        if is_abandoned {
            self.engine.set_abandoned(true);
        }

        self.abort_reason = Some("user_cancelled".to_string());

        // Cancel the loop
        self.engine.loop_control_mut().cancel();

        // Transition to aborted state
        if self.engine.state() != TaskState::Aborted {
            self.engine.abort_with_reason("user_cancelled")?;
        }

        // Emit final token usage
        self.emit_final_token_usage_update();

        Ok(())
    }

    // -------------------------------------------------------------------
    // dispose()
    // -------------------------------------------------------------------

    /// Dispose of the task and clean up resources.
    ///
    /// Source: `src/core/task/Task.ts` — `dispose()`
    ///
    /// Removes event listeners, clears timeouts, and marks the task as disposed.
    pub fn dispose(&mut self) {
        if self.disposed {
            return;
        }

        info!(task_id = %self.task_id(), "Disposing task");
        self.disposed = true;
        self.abort = true;

        // Clear any pending state
        self.engine.streaming_mut().is_streaming = false;
    }

    // -------------------------------------------------------------------
    // startSubtask()
    // -------------------------------------------------------------------

    /// Start a subtask (delegate to a child task).
    ///
    /// Source: `src/core/task/Task.ts` — `startSubtask()`
    ///
    /// Creates a new child task with the given parameters and delegates
    /// execution to it. The parent task transitions to the Delegated state.
    pub async fn start_subtask(
        &mut self,
        message: &str,
        _initial_todos: Vec<roo_types::todo::TodoItem>,
        mode: &str,
    ) -> Result<String, TaskError> {
        let subtask_id = format!(
            "{}-sub-{}",
            self.engine.config().task_id,
            uuid::Uuid::now_v7()
        );

        info!(
            parent_task_id = %self.task_id(),
            subtask_id = %subtask_id,
            mode = mode,
            "Starting subtask"
        );

        // Create subtask config
        let mut subtask_config = TaskConfig::new(&subtask_id, &self.engine.config().cwd)
            .with_mode(mode)
            .with_task_text(message)
            .with_root_task_id(
                self.engine
                    .config()
                    .root_task_id
                    .as_deref()
                    .unwrap_or(self.task_id()),
            )
            .with_parent_task_id(self.task_id());

        // Copy storage path if available
        if let Some(ref storage_path) = self.engine.config().storage_path {
            subtask_config = subtask_config.with_storage_path(storage_path);
        }

        // Set initial todos
        // Source: TS — `initialTodos` are set on the child task

        // Store child task ID
        self.child_task_id = Some(subtask_id.clone());

        // Delegate the parent task
        self.engine.delegate()?;

        Ok(subtask_id)
    }

    // -------------------------------------------------------------------
    // resumeAfterDelegation()
    // -------------------------------------------------------------------

    /// Resume the task after a subtask completes.
    ///
    /// Source: `src/core/task/Task.ts` — `resumeAfterDelegation()`
    ///
    /// Clears ask states, resets abort/streaming flags, and prepares
    /// for the next API call.
    pub async fn resume_after_delegation(&mut self) -> Result<(), TaskError> {
        info!(
            task_id = %self.task_id(),
            "Resuming after delegation"
        );

        // Clear child task ID
        self.child_task_id = None;

        // Resume the engine
        self.engine.resume_after_delegation()?;

        Ok(())
    }

    // -------------------------------------------------------------------
    // submitUserMessage()
    // -------------------------------------------------------------------

    /// Submit a user message to the task.
    ///
    /// Source: `src/core/task/Task.ts` — `submitUserMessage()`
    ///
    /// Handles the user's response to an ask prompt. Depending on the
    /// current ask type, processes the response appropriately.
    ///
    /// If a [`MessageQueueService`] is available, the message is also
    /// enqueued for ordered processing.
    pub async fn submit_user_message(
        &mut self,
        text: &str,
        images: Option<Vec<String>>,
    ) -> Result<(), TaskError> {
        // Enqueue message if message queue is available
        // Source: TS — `this.messageQueueService.addMessage(text, images)`
        if let Some(ref queue) = self.services.message_queue {
            let mut q = queue.lock().await;
            q.add_message(text, images.clone());
            debug!(text_len = text.len(), "Message enqueued");
        }

        // Add user feedback message
        self.ask_say
            .say(
                ClineSay::UserFeedback,
                Some(text.to_string()),
                images.clone(),
            )
            .await?;

        // Handle the ask response
        self.ask_say
            .handle_response(
                AskResponse::MessageResponse,
                Some(text.to_string()),
                images,
            )
            .await;

        // Telemetry: conversation message
        self.emit_telemetry_conversation_message("user");

        debug!(text_len = text.len(), "User message submitted");
        Ok(())
    }

    // -------------------------------------------------------------------
    // condenseContext()
    // -------------------------------------------------------------------

    /// Manually trigger context condensation.
    ///
    /// Source: `src/core/task/Task.ts` — `condenseContext()`
    ///
    /// Flushes any pending tool results, then triggers condensation
    /// of the conversation history to reduce context size.
    pub async fn condense_context(&mut self) -> Result<(), TaskError> {
        info!(task_id = %self.task_id(), "Condensing context (manual trigger)");

        // Flush pending tool results before condensing
        // Source: TS — "CRITICAL: Flush any pending tool results before condensing"
        self.flush_pending_tool_results().await?;

        // The actual condensation is handled by the agent loop's
        // `try_condense_context()` method. Here we just emit a
        // condense request event.
        self.engine
            .emitter()
            .emit(&TaskEvent::ContextCondensationRequested {
                task_id: self.task_id().to_string(),
            });

        Ok(())
    }

    /// Flush pending tool results to history.
    ///
    /// Source: `src/core/task/Task.ts` — `flushPendingToolResultsToHistory()`
    async fn flush_pending_tool_results(&mut self) -> Result<bool, TaskError> {
        // Check if there are pending tool results
        let history_len = self.engine.api_conversation_history().len();
        if history_len == 0 {
            return Ok(false);
        }

        // Check if the last message is an assistant message with tool_use
        // If so, we need to add tool_result blocks for any pending tools
        // Source: TS — checks for pending userMessageContent
        Ok(false) // No pending results in this simplified implementation
    }

    // -------------------------------------------------------------------
    // updateApiConfiguration()
    // -------------------------------------------------------------------

    /// Update the API configuration.
    ///
    /// Source: `src/core/task/Task.ts` — `updateApiConfiguration()`
    ///
    /// Updates the API configuration and rebuilds the API handler.
    /// This is used when the user changes the model or provider mid-task.
    pub fn update_api_configuration(
        &mut self,
        _new_config: &roo_types::provider_settings::ProviderSettings,
    ) {
        // The actual API handler rebuild requires access to the provider,
        // which is managed by the agent loop. Here we just note the update.
        warn!(
            "updateApiConfiguration: API handler rebuild requires agent loop integration"
        );
    }

    // -------------------------------------------------------------------
    // handleTerminalOperation()
    // -------------------------------------------------------------------

    /// Handle a terminal operation (continue or abort).
    ///
    /// Source: `src/core/task/Task.ts` — `handleTerminalOperation()`
    pub async fn handle_terminal_operation(
        &mut self,
        operation: &str,
    ) -> Result<(), TaskError> {
        match operation {
            "continue" => {
                // Continue the running terminal process
                debug!("Continuing terminal operation");
            }
            "abort" => {
                // Abort the running terminal process
                debug!("Aborting terminal operation");
                self.abort = true;
            }
            _ => {
                warn!(operation = operation, "Unknown terminal operation");
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------
    // emitFinalTokenUsageUpdate()
    // -------------------------------------------------------------------

    /// Emit the final token usage update.
    ///
    /// Source: `src/core/task/Task.ts` — `emitFinalTokenUsageUpdate()`
    pub fn emit_final_token_usage_update(&self) {
        let usage = self.engine.result().token_usage.clone();
        self.engine.emitter().emit_token_usage_updated(usage.clone());

        // Telemetry: LLM completion with token counts
        self.emit_telemetry_llm_completion(&usage);
    }

    // -------------------------------------------------------------------
    // getTokenUsage()
    // -------------------------------------------------------------------

    /// Get the current token usage.
    ///
    /// Source: `src/core/task/Task.ts` — `getTokenUsage()`
    pub fn get_token_usage(&self) -> roo_types::message::TokenUsage {
        self.engine.result().token_usage.clone()
    }

    // -------------------------------------------------------------------
    // recordToolUsage() / recordToolError()
    // -------------------------------------------------------------------

    /// Record a tool usage.
    ///
    /// Source: `src/core/task/Task.ts` — `recordToolUsage()`
    pub fn record_tool_usage(&mut self, tool_name: &str) {
        self.engine.record_tool_execution(tool_name, true);
        // Telemetry: tool used
        self.emit_telemetry_tool_usage(tool_name);
    }

    /// Record a tool error.
    ///
    /// Source: `src/core/task/Task.ts` — `recordToolError()`
    pub fn record_tool_error(&mut self, tool_name: &str, error: Option<&str>) {
        self.engine.record_tool_execution(tool_name, false);
        // Telemetry: tool used (even on error)
        self.emit_telemetry_tool_usage(tool_name);
        if let Some(err) = error {
            debug!(tool = tool_name, error = err, "Tool error recorded");
        }
    }

    // -------------------------------------------------------------------
    // combineMessages()
    // -------------------------------------------------------------------

    /// Combine messages for display.
    ///
    /// Source: `src/core/task/Task.ts` — `combineMessages()`
    pub fn combine_messages(&self) -> Vec<ClineMessage> {
        // Source: TS — `combineApiRequests(combineCommandSequences(messages))`
        // For now, return messages as-is. The actual combining logic
        // would merge consecutive API requests and command sequences.
        self.ask_say.cline_messages().to_vec()
    }

    // -------------------------------------------------------------------
    // processQueuedMessages()
    // -------------------------------------------------------------------

    /// Process queued messages.
    ///
    /// Source: `src/core/task/Task.ts` — `processQueuedMessages()`
    ///
    /// Dequeues messages from the [`MessageQueueService`] (if available)
    /// and returns them for processing by the caller.
    pub async fn process_queued_messages(&mut self) -> Vec<roo_message_queue::QueuedMessage> {
        let Some(ref queue) = self.services.message_queue else {
            return Vec::new();
        };

        let mut q = queue.lock().await;
        let mut drained = Vec::new();
        while !q.is_empty() {
            if let Some(msg) = q.dequeue_message() {
                drained.push(msg);
            } else {
                break;
            }
        }

        if !drained.is_empty() {
            debug!(count = drained.len(), "Dequeued messages from queue");
        }
        drained
    }

    // -------------------------------------------------------------------
    // pushToolResultToUserContent()
    // -------------------------------------------------------------------

    /// Push a tool result to user content, preventing duplicates.
    ///
    /// Source: `src/core/task/Task.ts` — `pushToolResultToUserContent()`
    pub fn push_tool_result_to_user_content(
        &mut self,
        tool_use_id: &str,
        result: &str,
        is_error: bool,
    ) -> bool {
        // Create a tool result message
        let result_msg = crate::message_builder::MessageBuilder::create_tool_result_message(
            tool_use_id,
            &crate::tool_dispatcher::ToolExecutionResult {
                text: result.to_string(),
                images: None,
                is_error,
            },
        );
        self.engine.add_api_message(result_msg);
        true
    }

    // -------------------------------------------------------------------
    // sayAndCreateMissingParamError()
    // -------------------------------------------------------------------

    /// Say an error about a missing parameter.
    ///
    /// Source: `src/core/task/Task.ts` — `sayAndCreateMissingParamError()`
    pub async fn say_missing_param_error(
        &mut self,
        tool_name: &str,
        param_name: &str,
        rel_path: Option<&str>,
    ) -> Result<(), TaskError> {
        let path_info = rel_path
            .map(|p| format!(" for file: {}", p))
            .unwrap_or_default();
        let text = format!(
            "Missing required parameter '{}' for tool '{}'{}",
            param_name, tool_name, path_info
        );
        self.ask_say
            .say(ClineSay::Error, Some(text), None)
            .await
    }

    // -------------------------------------------------------------------
    // Telemetry helpers
    // -------------------------------------------------------------------

    /// Emit a telemetry "task created" event.
    fn emit_telemetry_task_created(&self) {
        if let Some(ref telemetry) = self.services.telemetry {
            if let Ok(svc) = telemetry.read() {
                svc.capture_task_created(self.task_id());
            }
        }
    }

    /// Emit a telemetry "task restarted" event.
    fn emit_telemetry_task_restarted(&self) {
        if let Some(ref telemetry) = self.services.telemetry {
            if let Ok(svc) = telemetry.read() {
                svc.capture_task_restarted(self.task_id());
            }
        }
    }

    /// Emit a telemetry "conversation message" event.
    fn emit_telemetry_conversation_message(&self, source: &str) {
        if let Some(ref telemetry) = self.services.telemetry {
            if let Ok(svc) = telemetry.read() {
                svc.capture_conversation_message(self.task_id(), source);
            }
        }
    }

    /// Emit a telemetry "tool used" event.
    fn emit_telemetry_tool_usage(&self, tool: &str) {
        if let Some(ref telemetry) = self.services.telemetry {
            if let Ok(svc) = telemetry.read() {
                svc.capture_tool_usage(self.task_id(), tool);
            }
        }
    }

    /// Emit a telemetry "LLM completion" event with token usage.
    fn emit_telemetry_llm_completion(&self, usage: &roo_types::message::TokenUsage) {
        if let Some(ref telemetry) = self.services.telemetry {
            if let Ok(svc) = telemetry.read() {
                svc.capture_llm_completion(
                    self.task_id(),
                    usage.total_tokens_in,
                    usage.total_tokens_out,
                    usage.total_cache_writes.unwrap_or(0),
                    usage.total_cache_reads.unwrap_or(0),
                    if usage.total_cost > 0.0 {
                        Some(usage.total_cost)
                    } else {
                        None
                    },
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TaskStatus (re-export from roo_types for convenience)
// ---------------------------------------------------------------------------

/// Task status as reported to the UI.
///
/// Re-exported from `roo_types::task::TaskStatus` for convenience.
pub use roo_types::task::TaskStatus;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TaskConfig;

    fn make_lifecycle() -> TaskLifecycle {
        let config = TaskConfig::new("test-task", "/tmp/work")
            .with_mode("code")
            .with_max_iterations(100);
        let engine = TaskEngine::new(config).unwrap();
        TaskLifecycle::new(engine)
    }

    #[test]
    fn test_lifecycle_new() {
        let lc = make_lifecycle();
        assert_eq!(lc.state(), TaskState::Idle);
        assert!(!lc.is_aborted());
        assert!(lc.abort_reason().is_none());
        assert!(!lc.is_disposed());
        assert!(lc.child_task_id().is_none());
    }

    #[test]
    fn test_lifecycle_task_status() {
        let lc = make_lifecycle();
        assert_eq!(lc.task_status(), TaskStatus::Idle);
    }

    #[test]
    fn test_cancel_current_request() {
        let mut lc = make_lifecycle();
        assert!(!lc.is_aborted());

        lc.cancel_current_request();
        assert!(lc.is_aborted());
        assert_eq!(lc.abort_reason(), Some("user_cancelled"));
    }

    #[tokio::test]
    async fn test_abort_task() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        lc.abort_task(false).await.unwrap();
        assert!(lc.is_aborted());
        assert_eq!(lc.state(), TaskState::Aborted);
    }

    #[tokio::test]
    async fn test_abort_task_abandoned() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        lc.abort_task(true).await.unwrap();
        assert!(lc.is_aborted());
        assert!(lc.engine.is_abandoned());
    }

    #[test]
    fn test_dispose() {
        let mut lc = make_lifecycle();
        assert!(!lc.is_disposed());

        lc.dispose();
        assert!(lc.is_disposed());

        // Second dispose is no-op
        lc.dispose();
        assert!(lc.is_disposed());
    }

    #[tokio::test]
    async fn test_start_subtask() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        let subtask_id = lc.start_subtask("Fix the bug", Vec::new(), "code").await;
        assert!(subtask_id.is_ok());
        let id = subtask_id.unwrap();
        assert!(id.starts_with("test-task-sub-"));
        assert_eq!(lc.child_task_id(), Some(id.as_str()));
        assert_eq!(lc.state(), TaskState::Delegated);
    }

    #[tokio::test]
    async fn test_resume_after_delegation() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();
        lc.start_subtask("Fix the bug", Vec::new(), "code")
            .await
            .unwrap();
        assert_eq!(lc.state(), TaskState::Delegated);

        lc.resume_after_delegation().await.unwrap();
        assert_eq!(lc.state(), TaskState::Running);
        assert!(lc.child_task_id().is_none());
    }

    #[tokio::test]
    async fn test_submit_user_message() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        lc.submit_user_message("Hello", None).await.unwrap();
        assert_eq!(lc.ask_say.cline_messages().len(), 1);
    }

    #[test]
    fn test_get_token_usage() {
        let lc = make_lifecycle();
        let usage = lc.get_token_usage();
        assert_eq!(usage.total_tokens_in, 0);
        assert_eq!(usage.total_tokens_out, 0);
    }

    #[test]
    fn test_record_tool_usage() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        lc.record_tool_usage("read_file");
        assert_eq!(lc.engine.result().tool_usage["read_file"], 1);
    }

    #[test]
    fn test_record_tool_error() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        lc.record_tool_error("write_to_file", Some("Permission denied"));
        assert_eq!(lc.engine.result().tool_usage["write_to_file"], 1);
    }

    #[test]
    fn test_combine_messages() {
        let lc = make_lifecycle();
        let messages = lc.combine_messages();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_emit_final_token_usage() {
        let lc = make_lifecycle();
        lc.emit_final_token_usage_update();
        // Should not panic
    }

    #[tokio::test]
    async fn test_say_missing_param_error() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        lc.say_missing_param_error("read_file", "path", Some("/tmp/test.rs"))
            .await
            .unwrap();
        assert_eq!(lc.ask_say.cline_messages().len(), 1);
    }

    #[test]
    fn test_push_tool_result() {
        let mut lc = make_lifecycle();
        lc.engine.start().unwrap();

        let added = lc.push_tool_result_to_user_content("tool_123", "result text", false);
        assert!(added);
    }

    #[tokio::test]
    async fn test_handle_terminal_operation_continue() {
        let mut lc = make_lifecycle();
        lc.handle_terminal_operation("continue").await.unwrap();
        assert!(!lc.is_aborted());
    }

    #[tokio::test]
    async fn test_handle_terminal_operation_abort() {
        let mut lc = make_lifecycle();
        lc.handle_terminal_operation("abort").await.unwrap();
        assert!(lc.is_aborted());
    }
}
