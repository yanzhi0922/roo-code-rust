//! Task engine core logic.
//!
//! Provides [`TaskEngine`] which orchestrates the task lifecycle including
//! state management, loop control, event emission, streaming state, and
//! result computation.
//!
//! Source: `src/core/task/Task.ts` 鈥?Task class

use crate::config::validate_config;
use crate::events::TaskEventEmitter;
use crate::loop_control::LoopControl;
use crate::state::StateMachine;
use crate::types::{
    StreamingState, TaskConfig, TaskError, TaskResult, TaskState,
    MAX_EXPONENTIAL_BACKOFF_SECONDS,
};

// ---------------------------------------------------------------------------
// TaskEngine
// ---------------------------------------------------------------------------

/// The core task engine.
///
/// Manages the task lifecycle: state transitions, loop control, event emission,
/// streaming state, and result aggregation.
///
/// Unlike the TypeScript `Task` class (a ~4700-line god object), this Rust
/// implementation decomposes responsibilities into separate components:
/// - [`StateMachine`] for state transitions
/// - [`LoopControl`] for iteration and mistake limits
/// - [`TaskEventEmitter`] for event-driven communication
/// - [`StreamingState`] for API streaming state
/// - [`TaskResult`] for result aggregation
///
/// The actual agent loop (API call 鈫?parse response 鈫?tool execution 鈫?loop)
/// is coordinated at the application layer using these components.
pub struct TaskEngine {
    config: TaskConfig,
    state_machine: StateMachine,
    loop_control: LoopControl,
    streaming: StreamingState,
    result: TaskResult,
    /// API conversation history (messages sent to/from the API).
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`apiConversationHistory`
    api_conversation_history: Vec<roo_types::api::ApiMessage>,
    /// UI-facing conversation messages (ClineMessages).
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`clineMessages`
    cline_messages: Vec<roo_types::message::ClineMessage>,
    /// Whether the task has been initialized.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`isInitialized`
    is_initialized: bool,
    /// Whether the task has been abandoned (for delegation).
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`abandoned`
    abandoned: bool,
    /// Whether the task is paused.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`isPaused`
    is_paused: bool,
    /// Abort reason, if any.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`abortReason`
    abort_reason: Option<String>,
}

impl TaskEngine {
    /// Create a new task engine with the given configuration.
    pub fn new(config: TaskConfig) -> Result<Self, TaskError> {
        validate_config(&config)?;

        let max_iterations = config.max_iterations;
        let task_id = config.task_id.clone();
        let consecutive_mistake_limit = config.consecutive_mistake_limit;

        Ok(Self {
            config,
            state_machine: StateMachine::new(),
            loop_control: LoopControl::with_max_iterations(
                consecutive_mistake_limit,
                max_iterations.unwrap_or(usize::MAX),
            ),
            streaming: StreamingState::new(),
            result: TaskResult::new(task_id, TaskState::Idle),
            api_conversation_history: Vec::new(),
            cline_messages: Vec::new(),
            is_initialized: false,
            abandoned: false,
            is_paused: false,
            abort_reason: None,
        })
    }

    /// Get the current task state.
    pub fn state(&self) -> TaskState {
        self.state_machine.current()
    }

    /// Get a reference to the task configuration.
    pub fn config(&self) -> &TaskConfig {
        &self.config
    }

    /// Get a reference to the task result.
    pub fn result(&self) -> &TaskResult {
        &self.result
    }

    /// Get a reference to the event emitter.
    pub fn emitter(&self) -> &TaskEventEmitter {
        self.state_machine.emitter()
    }

    /// Get a reference to the loop control.
    pub fn loop_control(&self) -> &LoopControl {
        &self.loop_control
    }

    /// Get a mutable reference to the loop control.
    pub fn loop_control_mut(&mut self) -> &mut LoopControl {
        &mut self.loop_control
    }

    /// Get a reference to the streaming state.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?streaming-related properties
    pub fn streaming(&self) -> &StreamingState {
        &self.streaming
    }

    /// Get a mutable reference to the streaming state.
    pub fn streaming_mut(&mut self) -> &mut StreamingState {
        &mut self.streaming
    }

    /// Get a reference to the API conversation history.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`apiConversationHistory`
    pub fn api_conversation_history(&self) -> &[roo_types::api::ApiMessage] {
        &self.api_conversation_history
    }

    /// Get a mutable reference to the API conversation history.
    pub fn api_conversation_history_mut(&mut self) -> &mut Vec<roo_types::api::ApiMessage> {
        &mut self.api_conversation_history
    }

    /// Add a message to the API conversation history.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`addToApiConversationHistory`
    pub fn add_api_message(&mut self, message: roo_types::api::ApiMessage) {
        self.api_conversation_history.push(message);
    }

    /// Set the API conversation history (e.g., when loading from persistence).
    pub fn set_api_conversation_history(&mut self, history: Vec<roo_types::api::ApiMessage>) {
        self.api_conversation_history = history;
    }

    /// Clear the API conversation history.
    pub fn clear_api_conversation_history(&mut self) {
        self.api_conversation_history.clear();
    }

    /// Get a reference to the cline messages (UI-facing messages).
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`clineMessages`
    pub fn cline_messages(&self) -> &[roo_types::message::ClineMessage] {
        &self.cline_messages
    }

    /// Get a mutable reference to the cline messages.
    pub fn cline_messages_mut(&mut self) -> &mut Vec<roo_types::message::ClineMessage> {
        &mut self.cline_messages
    }

    /// Add a cline message.
    pub fn add_cline_message(&mut self, message: roo_types::message::ClineMessage) {
        self.cline_messages.push(message);
    }

    /// Check whether the task is initialized.
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    /// Mark the task as initialized.
    pub fn set_initialized(&mut self, initialized: bool) {
        self.is_initialized = initialized;
    }

    /// Check whether the task has been abandoned.
    pub fn is_abandoned(&self) -> bool {
        self.abandoned
    }

    /// Mark the task as abandoned (for delegation).
    pub fn set_abandoned(&mut self, abandoned: bool) {
        self.abandoned = abandoned;
    }

    /// Get the abort reason.
    pub fn abort_reason(&self) -> Option<&str> {
        self.abort_reason.as_deref()
    }

    /// Start the task.
    pub fn start(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.start()?;
        self.result.status = state;
        Ok(state)
    }

    /// Pause the task.
    pub fn pause(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.pause()?;
        self.is_paused = true;
        self.result.status = state;
        Ok(state)
    }

    /// Resume the task.
    pub fn resume(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.resume()?;
        self.is_paused = false;
        self.result.status = state;
        Ok(state)
    }

    /// Complete the task.
    pub fn complete(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.complete()?;
        self.result.status = state;
        Ok(state)
    }

    /// Abort the task.
    pub fn abort(&mut self) -> Result<TaskState, TaskError> {
        self.loop_control.cancel();
        self.abort_reason = Some("user_cancelled".to_string());
        let state = self.state_machine.abort()?;
        self.result.status = state;
        Ok(state)
    }

    /// Abort the task with a specific reason.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`abortReason` can be various values
    /// like "user_cancelled", "rate_limit_hit", "max_tokens_exceeded", etc.
    pub fn abort_with_reason(&mut self, reason: &str) -> Result<TaskState, TaskError> {
        self.loop_control.cancel();
        self.abort_reason = Some(reason.to_string());
        let state = self.state_machine.abort()?;
        self.result.status = state;
        Ok(state)
    }

    /// Delegate the task.
    pub fn delegate(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.delegate()?;
        self.result.status = state;
        Ok(state)
    }

    /// Check whether the task loop should continue.
    pub fn should_continue(&self) -> bool {
        self.loop_control.should_continue()
    }

    /// Advance one iteration.
    ///
    /// Returns `true` if the iteration limit was reached.
    pub fn advance_iteration(&mut self) -> bool {
        let reached = self.loop_control.increment_iteration();
        self.result.iterations = self.loop_control.current_iteration;
        reached
    }

    /// Record a tool execution.
    pub fn record_tool_execution(&mut self, tool_name: &str, success: bool) {
        self.result.record_tool_usage(tool_name);
        self.state_machine.emitter().emit_tool_executed(tool_name, success);

        if success {
            self.loop_control.reset_mistake_count();
        } else {
            let exceeded = self.loop_control.record_mistake();
            if exceeded {
                self.state_machine.emitter().emit_state_changed(
                    self.state_machine.current(),
                    TaskState::Aborted,
                );
            }
        }
    }

    /// Record a mistake without tool execution.
    pub fn record_mistake(&mut self) -> bool {
        self.loop_control.record_mistake()
    }

    /// Reset the mistake count.
    pub fn reset_mistakes(&mut self) {
        self.loop_control.reset_mistake_count();
    }

    /// Set the final message.
    pub fn set_final_message(&mut self, message: String) {
        self.result.final_message = Some(message);
    }

    /// Update token usage in the result.
    pub fn update_token_usage(&mut self, usage: roo_types::message::TokenUsage) {
        self.state_machine.emitter().emit_token_usage_updated(usage.clone());
        self.result.token_usage = usage;
    }

    /// Prepare for a new API request by resetting streaming state.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?streaming state reset block in
    /// `recursivelyMakeClineRequests`
    pub fn prepare_for_new_api_request(&mut self) {
        self.streaming.reset_for_new_request();
    }

    /// Calculate exponential backoff delay for retry attempts.
    ///
    /// Returns the delay in milliseconds.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`MAX_EXPONENTIAL_BACKOFF_SECONDS`
    pub fn calculate_backoff_delay(retry_attempt: u32) -> u64 {
        let base_delay = 1000u64; // 1 second base
        let delay = base_delay * 2u64.pow(retry_attempt);
        delay.min(MAX_EXPONENTIAL_BACKOFF_SECONDS * 1000)
    }

    /// Resume the task after delegation.
    ///
    /// Clears ask states, resets abort/streaming flags, and prepares for
    /// the next API call.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`resumeAfterDelegation`
    pub fn resume_after_delegation(&mut self) -> Result<TaskState, TaskError> {
        // Reset abort and streaming state
        self.loop_control.reset_turn();
        self.abandoned = false;
        self.abort_reason = None;
        self.is_paused = false;
        self.streaming.reset_for_new_request();
        self.streaming.did_finish_aborting_stream = false;
        self.streaming.is_streaming = false;
        self.streaming.is_waiting_for_first_chunk = false;

        // Transition back to running
        let state = self.state_machine.resume()?;
        self.result.status = state;
        self.is_initialized = true;
        Ok(state)
    }

    /// Check whether the task is paused.
    ///
    /// Source: `src/core/task/Task.ts` 鈥?`isPaused`
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    /// Finalize the task and return the result.
    pub fn finalize(mut self) -> TaskResult {
        if self.state_machine.current() == TaskState::Running {
            // Force to completed if still running
            let _ = self.state_machine.complete();
            self.result.status = TaskState::Completed;
        }
        self.result
    }

    /// Build a result snapshot without consuming the engine.
    ///
    /// This is useful for the agent loop which needs to return a result
    /// while keeping the engine alive during execution.
    pub fn build_result_snapshot(&self) -> TaskResult {
        self.result.clone()
    }

    /// Complete the task if still running and return a result snapshot.
    ///
    /// Combines `complete()` (if needed) and `build_result_snapshot()`.
    pub fn finalize_in_place(&mut self) -> TaskResult {
        if self.state_machine.current() == TaskState::Running {
            let _ = self.state_machine.complete();
            self.result.status = TaskState::Completed;
        }
        self.result.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> TaskEngine {
        TaskEngine::new(
            TaskConfig::new("test-task", "/tmp/work")
                .with_mode("code")
                .with_max_iterations(100),
        )
        .unwrap()
    }

    #[test]
    fn test_engine_new() {
        let engine = make_engine();
        assert_eq!(engine.state(), TaskState::Idle);
        assert_eq!(engine.config().task_id, "test-task");
        assert!(!engine.is_initialized());
        assert!(!engine.is_abandoned());
        assert!(engine.abort_reason().is_none());
        assert!(engine.api_conversation_history().is_empty());
    }

    #[test]
    fn test_engine_start() {
        let mut engine = make_engine();
        let state = engine.start().unwrap();
        assert_eq!(state, TaskState::Running);
    }

    #[test]
    fn test_engine_full_lifecycle() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.complete().unwrap();
        assert_eq!(engine.state(), TaskState::Completed);
    }

    #[test]
    fn test_engine_pause_resume() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.pause().unwrap();
        assert_eq!(engine.state(), TaskState::Paused);
        engine.resume().unwrap();
        assert_eq!(engine.state(), TaskState::Running);
    }

    #[test]
    fn test_engine_abort() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.abort().unwrap();
        assert_eq!(engine.state(), TaskState::Aborted);
        assert!(!engine.should_continue());
        assert_eq!(engine.abort_reason(), Some("user_cancelled"));
    }

    #[test]
    fn test_engine_delegate() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.delegate().unwrap();
        assert_eq!(engine.state(), TaskState::Delegated);
    }

    #[test]
    fn test_engine_advance_iteration() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.advance_iteration();
        engine.advance_iteration();
        assert_eq!(engine.result().iterations, 2);
    }

    #[test]
    fn test_engine_record_tool_execution_success() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.record_tool_execution("read_file", true);
        assert_eq!(engine.result().tool_usage["read_file"], 1);
    }

    #[test]
    fn test_engine_record_tool_execution_failure() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.record_tool_execution("write_to_file", false);
        assert_eq!(engine.result().tool_usage["write_to_file"], 1);
        assert_eq!(engine.loop_control().consecutive_mistake_count, 1);
    }

    #[test]
    fn test_engine_record_multiple_tool_usages() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.record_tool_execution("read_file", true);
        engine.record_tool_execution("read_file", true);
        engine.record_tool_execution("write_to_file", true);
        assert_eq!(engine.result().tool_usage["read_file"], 2);
        assert_eq!(engine.result().tool_usage["write_to_file"], 1);
    }

    #[test]
    fn test_engine_set_final_message() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.set_final_message("Task completed successfully".to_string());
        assert_eq!(
            engine.result().final_message,
            Some("Task completed successfully".to_string())
        );
    }

    #[test]
    fn test_engine_finalize() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.advance_iteration();
        engine.set_final_message("Done".to_string());
        let result = engine.finalize();
        assert_eq!(result.status, TaskState::Completed);
        assert_eq!(result.final_message, Some("Done".to_string()));
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_engine_finalize_already_completed() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.complete().unwrap();
        let result = engine.finalize();
        assert_eq!(result.status, TaskState::Completed);
    }

    #[test]
    fn test_engine_invalid_config() {
        let config = TaskConfig::new("", "/tmp/work").with_mode("code");
        let result = TaskEngine::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_iteration_limit() {
        let mut engine = TaskEngine::new(
            TaskConfig::new("limited-task", "/tmp/work")
                .with_mode("code")
                .with_max_iterations(2),
        )
        .unwrap();
        engine.start().unwrap();
        engine.advance_iteration(); // 1
        assert!(engine.should_continue());
        engine.advance_iteration(); // 2 = limit
        assert!(!engine.should_continue());
    }

    #[test]
    fn test_engine_update_token_usage() {
        let mut engine = make_engine();
        engine.start().unwrap();
        let usage = roo_types::message::TokenUsage {
            total_tokens_in: 1000,
            total_tokens_out: 500,
            total_cache_writes: Some(100),
            total_cache_reads: Some(50),
            total_cost: 1.5,
            context_tokens: 2000,
        };
        engine.update_token_usage(usage);
        assert_eq!(engine.result().token_usage.total_tokens_in, 1000);
        assert_eq!(engine.result().token_usage.total_cost, 1.5);
    }

    #[test]
    fn test_engine_event_emission() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mut engine = make_engine();
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        engine.emitter().on(move |event| {
            if let crate::events::TaskEvent::StateChanged { from, to } = event {
                if *from == TaskState::Idle && *to == TaskState::Running {
                    count_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });

        engine.start().unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    // --- Streaming state tests ---

    #[test]
    fn test_engine_streaming_state() {
        let mut engine = make_engine();
        assert!(!engine.streaming().is_streaming);

        engine.streaming_mut().is_streaming = true;
        assert!(engine.streaming().is_streaming);
    }

    #[test]
    fn test_engine_prepare_for_new_api_request() {
        let mut engine = make_engine();
        engine.streaming_mut().assistant_message_saved_to_history = true;
        engine.streaming_mut().current_streaming_content_index = 5;
        engine.streaming_mut().did_tool_fail_in_current_turn = true;

        engine.prepare_for_new_api_request();

        assert!(!engine.streaming().assistant_message_saved_to_history);
        assert_eq!(engine.streaming().current_streaming_content_index, 0);
        assert!(!engine.streaming().did_tool_fail_in_current_turn);
    }

    // --- API conversation history tests ---

    #[test]
    fn test_engine_api_conversation_history() {
        let mut engine = make_engine();
        assert!(engine.api_conversation_history().is_empty());

        let msg = roo_types::api::ApiMessage {
            role: roo_types::api::MessageRole::User,
            content: vec![roo_types::api::ContentBlock::Text {
                text: "Hello".to_string(),
            }],
            reasoning: None,
            ts: Some(1000.0),
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        };

        engine.add_api_message(msg.clone());
        assert_eq!(engine.api_conversation_history().len(), 1);

        engine.clear_api_conversation_history();
        assert!(engine.api_conversation_history().is_empty());
    }

    #[test]
    fn test_engine_set_api_conversation_history() {
        let mut engine = make_engine();
        let history = vec![
            roo_types::api::ApiMessage {
                role: roo_types::api::MessageRole::User,
                content: vec![roo_types::api::ContentBlock::Text {
                    text: "msg1".to_string(),
                }],
                reasoning: None,
                ts: None,
                truncation_parent: None,
                is_truncation_marker: None,
                truncation_id: None,
                condense_parent: None,
                is_summary: None,
                condense_id: None,
            },
            roo_types::api::ApiMessage {
                role: roo_types::api::MessageRole::Assistant,
                content: vec![roo_types::api::ContentBlock::Text {
                    text: "msg2".to_string(),
                }],
                reasoning: None,
                ts: None,
                truncation_parent: None,
                is_truncation_marker: None,
                truncation_id: None,
                condense_parent: None,
                is_summary: None,
                condense_id: None,
            },
        ];

        engine.set_api_conversation_history(history);
        assert_eq!(engine.api_conversation_history().len(), 2);
    }

    // --- Initialization and abandonment tests ---

    #[test]
    fn test_engine_initialization() {
        let mut engine = make_engine();
        assert!(!engine.is_initialized());
        engine.set_initialized(true);
        assert!(engine.is_initialized());
    }

    #[test]
    fn test_engine_abandonment() {
        let mut engine = make_engine();
        assert!(!engine.is_abandoned());
        engine.set_abandoned(true);
        assert!(engine.is_abandoned());
    }

    // --- Backoff calculation tests ---

    #[test]
    fn test_calculate_backoff_delay() {
        assert_eq!(TaskEngine::calculate_backoff_delay(0), 1000); // 1s
        assert_eq!(TaskEngine::calculate_backoff_delay(1), 2000); // 2s
        assert_eq!(TaskEngine::calculate_backoff_delay(2), 4000); // 4s
        assert_eq!(TaskEngine::calculate_backoff_delay(3), 8000); // 8s
        assert_eq!(TaskEngine::calculate_backoff_delay(10), MAX_EXPONENTIAL_BACKOFF_SECONDS * 1000); // capped
    }

    // --- Resume after delegation tests ---

    #[test]
    fn test_engine_resume_after_delegation() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.pause().unwrap();

        let state = engine.resume_after_delegation().unwrap();
        assert_eq!(state, TaskState::Running);
        assert!(engine.is_initialized());
        assert!(!engine.is_abandoned());
    }

    // --- Mistake limit tests ---

    #[test]
    fn test_engine_mistake_limit_with_new_config() {
        let mut engine = TaskEngine::new(
            TaskConfig::new("test", "/tmp")
                .with_mode("code")
                .with_consecutive_mistake_limit(2),
        )
        .unwrap();

        engine.start().unwrap();

        // First mistake
        engine.record_tool_execution("tool1", false);
        assert!(engine.should_continue()); // 1 < 2

        // Second mistake hits the limit (>= comparison)
        engine.record_tool_execution("tool1", false);
        assert!(!engine.should_continue()); // 2 >= 2 鈫?STOP
    }

    // --- Cline messages tests ---

    #[test]
    fn test_engine_cline_messages() {
        let mut engine = make_engine();
        assert!(engine.cline_messages().is_empty());

        let msg = roo_types::message::ClineMessage {
            ts: 1700000000.0,
            r#type: roo_types::message::MessageType::Say,
            ask: None,
            say: Some(roo_types::message::ClineSay::Text),
            text: Some("Hello".to_string()),
            images: None,
            partial: None,
            reasoning: None,
            conversation_history_index: None,
            checkpoint: None,
            progress_status: None,
            context_condense: None,
            context_truncation: None,
            is_protected: None,
            api_protocol: None,
            is_answered: None,
        };

        engine.add_cline_message(msg);
        assert_eq!(engine.cline_messages().len(), 1);
    }

    // --- is_paused tests ---

    #[test]
    fn test_engine_is_paused() {
        let mut engine = make_engine();
        assert!(!engine.is_paused());

        engine.start().unwrap();
        assert!(!engine.is_paused());

        engine.pause().unwrap();
        assert!(engine.is_paused());

        engine.resume().unwrap();
        assert!(!engine.is_paused());
    }

    // --- abort_with_reason tests ---

    #[test]
    fn test_engine_abort_with_reason() {
        let mut engine = make_engine();
        engine.start().unwrap();

        engine.abort_with_reason("rate_limit_hit").unwrap();
        assert_eq!(engine.state(), TaskState::Aborted);
        assert_eq!(engine.abort_reason(), Some("rate_limit_hit"));
        assert!(!engine.should_continue());
    }

    // --- resume_after_delegation resets is_paused ---

    #[test]
    fn test_engine_resume_after_delegation_resets_paused() {
        let mut engine = make_engine();
        engine.start().unwrap();
        engine.pause().unwrap();
        assert!(engine.is_paused());

        engine.resume_after_delegation().unwrap();
        assert!(!engine.is_paused());
        assert_eq!(engine.state(), TaskState::Running);
    }
}