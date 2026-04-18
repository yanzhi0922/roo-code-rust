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
#[derive(Debug, Clone)]
pub enum TaskEvent {
    /// A new message was created.
    MessageCreated { message: ClineMessage },
    /// An existing message was updated.
    MessageUpdated { message: ClineMessage },
    /// The task state changed.
    StateChanged { from: TaskState, to: TaskState },
    /// A tool was executed.
    ToolExecuted { tool_name: String, success: bool },
    /// Token usage was updated.
    TokenUsageUpdated { usage: TokenUsage },
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

    /// Emit an event to all registered listeners.
    fn emit(&self, event: &TaskEvent) {
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
