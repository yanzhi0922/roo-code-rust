//! Task event system.
//!
//! Provides [`TaskEvent`] enum and [`TaskEventEmitter`] for emitting
//! task lifecycle events to registered listeners.

use std::sync::{Arc, Mutex};

use roo_types::message::{ClineMessage, TokenUsage};

use crate::types::TaskState;

// ---------------------------------------------------------------------------
// TaskEvent
// ---------------------------------------------------------------------------

/// Events emitted during task execution.
///
/// Source: `src/core/task/Task.ts` — all `emit()` calls and `RooCodeEventName`
#[derive(Debug, Clone)]
pub enum TaskEvent {
    // --- Message events ---
    /// A new message was created.
    /// Source: TS `addToClineMessages()` → emit
    MessageCreated { message: ClineMessage },
    /// An existing message was updated.
    /// Source: TS `updateClineMessage()` → emit
    MessageUpdated { message: ClineMessage },
    /// The task state changed.
    /// Source: TS state transitions → emit
    StateChanged { from: TaskState, to: TaskState },
    /// A tool was executed.
    /// Source: TS `executeTool()` → emit
    ToolExecuted { tool_name: String, success: bool },
    /// Token usage was updated.
    /// Source: TS `recursivelyMakeClineRequests()` → debounced emit
    TokenUsageUpdated { usage: TokenUsage },

    // --- Task lifecycle events ---
    /// Task started.
    /// Source: TS `initiateTaskLoop()` → `emit(RooCodeEventName.TaskStarted)`
    TaskStarted { task_id: String },
    /// Task completed (attempt_completion or no more tools).
    /// Source: TS `attemptCompletion` flow
    TaskCompleted { task_id: String },
    /// Task aborted by user or error.
    /// Source: TS `abortTask()` flow
    TaskAborted { task_id: String, reason: Option<String> },
    /// Task paused.
    /// Source: TS `pause()` flow
    TaskPaused { task_id: String },
    /// Task resumed from pause.
    /// Source: TS `resume()` flow
    TaskResumed { task_id: String },
    /// Task delegated to subtask.
    /// Source: TS `startSubtask()` flow
    TaskDelegated {
        parent_task_id: String,
        child_task_id: String,
    },

    // --- Interactive events ---
    /// Task is waiting for user interaction.
    /// Source: TS `ask()` → `emit(RooCodeEventName.TaskInteractive)`
    TaskInteractive { task_id: String },
    /// Task is idle and waiting for user input.
    /// Source: TS `ask()` → `emit(RooCodeEventName.TaskIdle)`
    TaskIdle { task_id: String },
    /// Task is resumable (can be resumed from history).
    /// Source: TS `ask()` → `emit(RooCodeEventName.TaskResumable)`
    TaskResumable { task_id: String },

    // --- API events ---
    /// API request started.
    /// Source: TS `recursivelyMakeClineRequests()` → `say("api_req_started")`
    ApiRequestStarted { task_id: String },
    /// API request finished.
    /// Source: TS `recursivelyMakeClineRequests()` → update api_req_started
    ApiRequestFinished {
        task_id: String,
        cost: Option<f64>,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
    },

    // --- Context events ---
    /// Context condensation requested.
    /// Source: TS `condenseContext()` → emit
    ContextCondensationRequested { task_id: String },
    /// Context condensation completed.
    /// Source: TS `manageContext()` → after condense
    ContextCondensationCompleted {
        task_id: String,
        messages_removed: usize,
    },
    /// Context truncation performed.
    /// Source: TS `maybeTruncateMessages()` → emit
    ContextTruncationPerformed {
        task_id: String,
        messages_removed: usize,
    },

    // --- Error events ---
    /// An error occurred during task execution.
    /// Source: TS `say("error", ...)` — e.g., "MODEL_NO_TOOLS_USED", "MODEL_NO_ASSISTANT_MESSAGES"
    Error {
        task_id: String,
        error: String,
    },

    // --- Checkpoint events ---
    /// Checkpoint saved.
    /// Source: TS `checkpointSave()` → emit
    CheckpointSaved {
        task_id: String,
        commit: Option<String>,
    },
    /// Checkpoint restored.
    /// Source: TS `checkpointRestore()` → emit
    CheckpointRestored { task_id: String },

    // --- Subtask events ---
    /// Subtask created.
    /// Source: TS `startSubtask()` → emit
    SubtaskCreated {
        parent_task_id: String,
        child_task_id: String,
    },
    /// Subtask completed.
    /// Source: TS subtask completion flow
    SubtaskCompleted {
        parent_task_id: String,
        child_task_id: String,
    },

    // --- Mode events ---
    /// Mode switched.
    /// Source: TS `switch_mode` tool → emit
    ModeSwitched { task_id: String, mode: String },

    // --- Streaming events (real-time) ---
    /// A text delta was received from the streaming API response.
    /// Source: TS `presentAssistantMessage()` — text streaming state machine
    StreamingTextDelta { task_id: String, text: String },
    /// A reasoning/thinking delta was received from the streaming API response.
    /// Source: TS `presentAssistantMessage()` — reasoning streaming
    StreamingReasoningDelta { task_id: String, text: String },
    /// A tool use started (tool call header received from stream).
    /// Source: TS `presentAssistantMessage()` — tool_use detection
    StreamingToolUseStarted {
        task_id: String,
        tool_name: String,
        tool_id: String,
    },
    /// A tool use received an argument delta during streaming.
    /// Source: TS `presentAssistantMessage()` — tool_use streaming
    StreamingToolUseDelta {
        task_id: String,
        tool_id: String,
        delta: String,
    },
    /// A tool use completed (tool result available).
    /// Source: TS `presentAssistantMessage()` — after tool execution
    StreamingToolUseCompleted {
        task_id: String,
        tool_name: String,
        tool_id: String,
        success: bool,
    },
    /// Streaming response completed.
    /// Source: TS `recursivelyMakeClineRequests()` — after stream ends
    StreamingCompleted { task_id: String },

    // --- Rate limit events ---
    /// Rate limit countdown tick.
    /// Source: TS `maybeWaitForProviderRateLimit()` → `say("api_req_rate_limit_wait")`
    ApiRateLimitWait {
        task_id: String,
        seconds: u64,
    },
}

// ---------------------------------------------------------------------------
// EventListener
// ---------------------------------------------------------------------------

/// A function that handles task events.
pub type EventListenerFn = dyn Fn(&TaskEvent) + Send + Sync;

// ---------------------------------------------------------------------------
// TaskEventEmitter
// ---------------------------------------------------------------------------

/// An event emitter for task lifecycle events.
///
/// Listeners are stored in a thread-safe manner and called in order of registration.
pub struct TaskEventEmitter {
    listeners: Arc<Mutex<Vec<Arc<EventListenerFn>>>>,
}

impl TaskEventEmitter {
    /// Create a new event emitter with no listeners.
    pub fn new() -> Self {
        Self {
            listeners: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a new event listener.
    pub fn on(&self, listener: impl Fn(&TaskEvent) + Send + Sync + 'static) {
        self.listeners.lock().unwrap().push(Arc::new(listener));
    }

    /// Emit a state changed event.
    pub fn emit_state_changed(&self, from: TaskState, to: TaskState) {
        self.emit(&TaskEvent::StateChanged { from, to });
    }

    /// Emit a message created event.
    pub fn emit_message_created(&self, message: ClineMessage) {
        self.emit(&TaskEvent::MessageCreated { message });
    }

    /// Emit a message updated event.
    pub fn emit_message_updated(&self, message: ClineMessage) {
        self.emit(&TaskEvent::MessageUpdated { message });
    }

    /// Emit a tool executed event.
    pub fn emit_tool_executed(&self, tool_name: &str, success: bool) {
        self.emit(&TaskEvent::ToolExecuted {
            tool_name: tool_name.to_string(),
            success,
        });
    }

    /// Emit a token usage updated event.
    pub fn emit_token_usage_updated(&self, usage: TokenUsage) {
        self.emit(&TaskEvent::TokenUsageUpdated { usage });
    }

    // --- Task lifecycle emit methods ---

    /// Emit a task started event.
    pub fn emit_task_started(&self, task_id: &str) {
        self.emit(&TaskEvent::TaskStarted {
            task_id: task_id.to_string(),
        });
    }

    /// Emit a task completed event.
    pub fn emit_task_completed(&self, task_id: &str) {
        self.emit(&TaskEvent::TaskCompleted {
            task_id: task_id.to_string(),
        });
    }

    /// Emit a task aborted event.
    pub fn emit_task_aborted(&self, task_id: &str, reason: Option<String>) {
        self.emit(&TaskEvent::TaskAborted {
            task_id: task_id.to_string(),
            reason,
        });
    }

    /// Emit a task paused event.
    pub fn emit_task_paused(&self, task_id: &str) {
        self.emit(&TaskEvent::TaskPaused {
            task_id: task_id.to_string(),
        });
    }

    /// Emit a task resumed event.
    pub fn emit_task_resumed(&self, task_id: &str) {
        self.emit(&TaskEvent::TaskResumed {
            task_id: task_id.to_string(),
        });
    }

    /// Emit a task delegated event.
    pub fn emit_task_delegated(&self, parent_task_id: &str, child_task_id: &str) {
        self.emit(&TaskEvent::TaskDelegated {
            parent_task_id: parent_task_id.to_string(),
            child_task_id: child_task_id.to_string(),
        });
    }

    // --- Interactive emit methods ---

    /// Emit a task interactive event.
    pub fn emit_task_interactive(&self, task_id: &str) {
        self.emit(&TaskEvent::TaskInteractive {
            task_id: task_id.to_string(),
        });
    }

    /// Emit a task idle event.
    pub fn emit_task_idle(&self, task_id: &str) {
        self.emit(&TaskEvent::TaskIdle {
            task_id: task_id.to_string(),
        });
    }

    /// Emit a task resumable event.
    pub fn emit_task_resumable(&self, task_id: &str) {
        self.emit(&TaskEvent::TaskResumable {
            task_id: task_id.to_string(),
        });
    }

    // --- API emit methods ---

    /// Emit an API request started event.
    pub fn emit_api_request_started(&self, task_id: &str) {
        self.emit(&TaskEvent::ApiRequestStarted {
            task_id: task_id.to_string(),
        });
    }

    /// Emit an API request finished event.
    pub fn emit_api_request_finished(
        &self,
        task_id: &str,
        cost: Option<f64>,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
    ) {
        self.emit(&TaskEvent::ApiRequestFinished {
            task_id: task_id.to_string(),
            cost,
            tokens_in,
            tokens_out,
        });
    }

    // --- Context emit methods ---

    /// Emit a context condensation completed event.
    pub fn emit_context_condensation_completed(&self, task_id: &str, messages_removed: usize) {
        self.emit(&TaskEvent::ContextCondensationCompleted {
            task_id: task_id.to_string(),
            messages_removed,
        });
    }

    /// Emit a context truncation performed event.
    pub fn emit_context_truncation_performed(&self, task_id: &str, messages_removed: usize) {
        self.emit(&TaskEvent::ContextTruncationPerformed {
            task_id: task_id.to_string(),
            messages_removed,
        });
    }

    // --- Checkpoint emit methods ---

    /// Emit a checkpoint saved event.
    pub fn emit_checkpoint_saved(&self, task_id: &str, commit: Option<String>) {
        self.emit(&TaskEvent::CheckpointSaved {
            task_id: task_id.to_string(),
            commit,
        });
    }

    /// Emit a checkpoint restored event.
    pub fn emit_checkpoint_restored(&self, task_id: &str) {
        self.emit(&TaskEvent::CheckpointRestored {
            task_id: task_id.to_string(),
        });
    }

    // --- Subtask emit methods ---

    /// Emit a subtask created event.
    pub fn emit_subtask_created(&self, parent_task_id: &str, child_task_id: &str) {
        self.emit(&TaskEvent::SubtaskCreated {
            parent_task_id: parent_task_id.to_string(),
            child_task_id: child_task_id.to_string(),
        });
    }

    /// Emit a subtask completed event.
    pub fn emit_subtask_completed(&self, parent_task_id: &str, child_task_id: &str) {
        self.emit(&TaskEvent::SubtaskCompleted {
            parent_task_id: parent_task_id.to_string(),
            child_task_id: child_task_id.to_string(),
        });
    }

    // --- Mode emit methods ---

    /// Emit a mode switched event.
    pub fn emit_mode_switched(&self, task_id: &str, mode: &str) {
        self.emit(&TaskEvent::ModeSwitched {
            task_id: task_id.to_string(),
            mode: mode.to_string(),
        });
    }

    // --- Streaming emit methods ---

    /// Emit a streaming text delta event.
    pub fn emit_streaming_text_delta(&self, task_id: &str, text: &str) {
        self.emit(&TaskEvent::StreamingTextDelta {
            task_id: task_id.to_string(),
            text: text.to_string(),
        });
    }

    /// Emit a streaming tool use started event.
    pub fn emit_streaming_tool_use_started(&self, task_id: &str, tool_name: &str, tool_id: &str) {
        self.emit(&TaskEvent::StreamingToolUseStarted {
            task_id: task_id.to_string(),
            tool_name: tool_name.to_string(),
            tool_id: tool_id.to_string(),
        });
    }

    /// Emit a streaming tool use completed event.
    pub fn emit_streaming_tool_use_completed(
        &self,
        task_id: &str,
        tool_name: &str,
        tool_id: &str,
        success: bool,
    ) {
        self.emit(&TaskEvent::StreamingToolUseCompleted {
            task_id: task_id.to_string(),
            tool_name: tool_name.to_string(),
            tool_id: tool_id.to_string(),
            success,
        });
    }

    /// Emit a streaming reasoning delta event.
    pub fn emit_streaming_reasoning_delta(&self, task_id: &str, text: &str) {
        self.emit(&TaskEvent::StreamingReasoningDelta {
            task_id: task_id.to_string(),
            text: text.to_string(),
        });
    }

    /// Emit a streaming tool use delta event.
    pub fn emit_streaming_tool_use_delta(&self, task_id: &str, tool_id: &str, delta: &str) {
        self.emit(&TaskEvent::StreamingToolUseDelta {
            task_id: task_id.to_string(),
            tool_id: tool_id.to_string(),
            delta: delta.to_string(),
        });
    }

    /// Emit a streaming completed event.
    pub fn emit_streaming_completed(&self, task_id: &str) {
        self.emit(&TaskEvent::StreamingCompleted {
            task_id: task_id.to_string(),
        });
    }

    /// Emit an event to all registered listeners.
    pub fn emit(&self, event: &TaskEvent) {
        let listeners = self.listeners.lock().unwrap();
        for listener in listeners.iter() {
            listener(event);
        }
    }

    /// Get the number of registered listeners.
    pub fn listener_count(&self) -> usize {
        self.listeners.lock().unwrap().len()
    }
}

impl Default for TaskEventEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TaskEventEmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskEventEmitter")
            .field("listener_count", &self.listener_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use roo_types::message::MessageType;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_test_message(text: &str) -> ClineMessage {
        ClineMessage {
            ts: 1700000000.0,
            r#type: MessageType::Say,
            ask: None,
            say: Some(roo_types::message::ClineSay::Text),
            text: Some(text.to_string()),
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
        }
    }

    #[test]
    fn test_event_emitter_no_listeners() {
        let emitter = TaskEventEmitter::new();
        emitter.emit_state_changed(TaskState::Idle, TaskState::Running);
        // Should not panic
    }

    #[test]
    fn test_event_emitter_single_listener() {
        let count = Arc::new(AtomicUsize::new(0));
        let emitter = TaskEventEmitter::new();

        let count_clone = count.clone();
        emitter.on(move |_event| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        emitter.emit_state_changed(TaskState::Idle, TaskState::Running);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_event_emitter_multiple_listeners() {
        let count = Arc::new(AtomicUsize::new(0));
        let emitter = TaskEventEmitter::new();

        for _ in 0..3 {
            let count_clone = count.clone();
            emitter.on(move |_event| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        emitter.emit_state_changed(TaskState::Idle, TaskState::Running);
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_event_emitter_message_created() {
        let received = Arc::new(std::sync::Mutex::new(None));
        let emitter = TaskEventEmitter::new();

        let received_clone = received.clone();
        emitter.on(move |event| {
            if let TaskEvent::MessageCreated { message } = event {
                *received_clone.lock().unwrap() = Some(message.text.clone());
            }
        });

        let msg = make_test_message("Hello");
        emitter.emit_message_created(msg);

        let text = received.lock().unwrap().clone();
        assert_eq!(text, Some(Some("Hello".to_string())));
    }

    #[test]
    fn test_event_emitter_tool_executed() {
        let received = Arc::new(std::sync::Mutex::new(None));
        let emitter = TaskEventEmitter::new();

        let received_clone = received.clone();
        emitter.on(move |event| {
            if let TaskEvent::ToolExecuted { tool_name, success } = event {
                *received_clone.lock().unwrap() = Some((tool_name.clone(), *success));
            }
        });

        emitter.emit_tool_executed("read_file", true);

        let data = received.lock().unwrap().clone();
        assert_eq!(data, Some(("read_file".to_string(), true)));
    }

    #[test]
    fn test_event_emitter_token_usage_updated() {
        let received = Arc::new(std::sync::Mutex::new(None));
        let emitter = TaskEventEmitter::new();

        let received_clone = received.clone();
        emitter.on(move |event| {
            if let TaskEvent::TokenUsageUpdated { usage } = event {
                *received_clone.lock().unwrap() = Some(usage.total_tokens_in);
            }
        });

        let usage = TokenUsage {
            total_tokens_in: 1000,
            total_tokens_out: 500,
            total_cache_writes: Some(100),
            total_cache_reads: Some(50),
            total_cost: 1.5,
            context_tokens: 2000,
        };
        emitter.emit_token_usage_updated(usage);

        let tokens = received.lock().unwrap().clone();
        assert_eq!(tokens, Some(1000u64));
    }

    #[test]
    fn test_event_emitter_listener_count() {
        let emitter = TaskEventEmitter::new();
        assert_eq!(emitter.listener_count(), 0);

        emitter.on(|_| {});
        assert_eq!(emitter.listener_count(), 1);

        emitter.on(|_| {});
        assert_eq!(emitter.listener_count(), 2);
    }

    #[test]
    fn test_task_event_clone() {
        let event = TaskEvent::StateChanged {
            from: TaskState::Idle,
            to: TaskState::Running,
        };
        let cloned = event.clone();
        if let TaskEvent::StateChanged { from, to } = cloned {
            assert_eq!(from, TaskState::Idle);
            assert_eq!(to, TaskState::Running);
        } else {
            panic!("Expected StateChanged event");
        }
    }

    #[test]
    fn test_task_event_debug() {
        let event = TaskEvent::ToolExecuted {
            tool_name: "read_file".to_string(),
            success: true,
        };
        let debug_str = format!("{event:?}");
        assert!(debug_str.contains("read_file"));
    }
}
