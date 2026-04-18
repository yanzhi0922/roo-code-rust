//! Task engine core logic.
//!
//! Provides [`TaskEngine`] which orchestrates the task lifecycle including
//! state management, loop control, event emission, and result computation.

use crate::config::{self, validate_config};
use crate::events::TaskEventEmitter;
use crate::loop_control::LoopControl;
use crate::state::StateMachine;
use crate::types::{TaskConfig, TaskError, TaskResult, TaskState};

// ---------------------------------------------------------------------------
// TaskEngine
// ---------------------------------------------------------------------------

/// The core task engine.
///
/// Manages the task lifecycle: state transitions, loop control, event emission,
/// and result aggregation.
pub struct TaskEngine {
    config: TaskConfig,
    state_machine: StateMachine,
    loop_control: LoopControl,
    result: TaskResult,
}

impl TaskEngine {
    /// Create a new task engine with the given configuration.
    pub fn new(config: TaskConfig) -> Result<Self, TaskError> {
        validate_config(&config)?;

        let max_iterations = config.max_iterations;
        let task_id = config.task_id.clone();

        Ok(Self {
            config,
            state_machine: StateMachine::new(),
            loop_control: LoopControl::with_max_iterations(
                config::DEFAULT_MAX_MISTAKES,
                max_iterations.unwrap_or(usize::MAX),
            ),
            result: TaskResult::new(task_id, TaskState::Idle),
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

    /// Start the task.
    pub fn start(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.start()?;
        self.result.status = state;
        Ok(state)
    }

    /// Pause the task.
    pub fn pause(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.pause()?;
        self.result.status = state;
        Ok(state)
    }

    /// Resume the task.
    pub fn resume(&mut self) -> Result<TaskState, TaskError> {
        let state = self.state_machine.resume()?;
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

    /// Finalize the task and return the result.
    pub fn finalize(mut self) -> TaskResult {
        if self.state_machine.current() == TaskState::Running {
            // Force to completed if still running
            let _ = self.state_machine.complete();
            self.result.status = TaskState::Completed;
        }
        self.result
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
}
