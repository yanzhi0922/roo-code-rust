//! Loop control for the task engine.
//!
//! Provides [`LoopControl`] which manages iteration counting, consecutive
//! mistake limits, cancellation, and the decision of whether to continue
//! the task loop.

// ---------------------------------------------------------------------------
// LoopControl
// ---------------------------------------------------------------------------

/// Controls the task loop execution.
///
/// Tracks:
/// - Consecutive mistake count and limit
/// - Maximum iterations
/// - Current iteration
/// - Cancellation state
/// - Whether a tool failed in the current turn
#[derive(Debug, Clone)]
pub struct LoopControl {
    /// Number of consecutive mistakes (e.g., tool failures, API errors).
    pub consecutive_mistake_count: usize,
    /// Maximum allowed consecutive mistakes before stopping.
    pub max_consecutive_mistakes: usize,
    /// Maximum number of iterations (None = unlimited).
    pub max_iterations: Option<usize>,
    /// Current iteration number (0-based).
    pub current_iteration: usize,
    /// Whether the task has been cancelled.
    pub is_cancelled: bool,
    /// Whether a tool failed in the current turn.
    pub did_tool_fail_in_current_turn: bool,
}

impl LoopControl {
    /// Create a new loop control with the given mistake limit.
    pub fn new(max_consecutive_mistakes: usize) -> Self {
        Self {
            consecutive_mistake_count: 0,
            max_consecutive_mistakes,
            max_iterations: None,
            current_iteration: 0,
            is_cancelled: false,
            did_tool_fail_in_current_turn: false,
        }
    }

    /// Create loop control with a maximum iteration limit.
    pub fn with_max_iterations(max_consecutive_mistakes: usize, max_iterations: usize) -> Self {
        Self {
            consecutive_mistake_count: 0,
            max_consecutive_mistakes,
            max_iterations: Some(max_iterations),
            current_iteration: 0,
            is_cancelled: false,
            did_tool_fail_in_current_turn: false,
        }
    }

    /// Check whether the task loop should continue.
    ///
    /// Returns `false` if:
    /// - The task has been cancelled
    /// - The maximum iteration limit has been reached
    /// - The consecutive mistake limit has been exceeded
    pub fn should_continue(&self) -> bool {
        if self.is_cancelled {
            return false;
        }

        if self.consecutive_mistake_count > self.max_consecutive_mistakes {
            return false;
        }

        if let Some(max) = self.max_iterations {
            if self.current_iteration >= max {
                return false;
            }
        }

        true
    }

    /// Record a mistake (e.g., tool failure).
    ///
    /// Returns `true` if the mistake limit has been exceeded.
    pub fn record_mistake(&mut self) -> bool {
        self.consecutive_mistake_count += 1;
        self.did_tool_fail_in_current_turn = true;
        self.consecutive_mistake_count > self.max_consecutive_mistakes
    }

    /// Reset the consecutive mistake count.
    ///
    /// Called when a successful action occurs.
    pub fn reset_mistake_count(&mut self) {
        self.consecutive_mistake_count = 0;
        self.did_tool_fail_in_current_turn = false;
    }

    /// Increment the iteration counter.
    ///
    /// Returns `true` if the maximum iteration limit has been reached.
    pub fn increment_iteration(&mut self) -> bool {
        self.current_iteration += 1;
        if let Some(max) = self.max_iterations {
            self.current_iteration >= max
        } else {
            false
        }
    }

    /// Cancel the task loop.
    pub fn cancel(&mut self) {
        self.is_cancelled = true;
    }

    /// Reset the turn-level state (called at the beginning of each iteration).
    pub fn reset_turn(&mut self) {
        self.did_tool_fail_in_current_turn = false;
    }

    /// Get the remaining iterations, if a limit is set.
    pub fn remaining_iterations(&self) -> Option<usize> {
        self.max_iterations.map(|max| max.saturating_sub(self.current_iteration))
    }
}

impl Default for LoopControl {
    fn default() -> Self {
        // Default: allow up to 3 consecutive mistakes, no iteration limit
        Self::new(3)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_control_default() {
        let lc = LoopControl::default();
        assert_eq!(lc.consecutive_mistake_count, 0);
        assert_eq!(lc.max_consecutive_mistakes, 3);
        assert!(lc.max_iterations.is_none());
        assert_eq!(lc.current_iteration, 0);
        assert!(!lc.is_cancelled);
        assert!(!lc.did_tool_fail_in_current_turn);
    }

    #[test]
    fn test_loop_control_should_continue_initially() {
        let lc = LoopControl::new(3);
        assert!(lc.should_continue());
    }

    #[test]
    fn test_loop_control_should_stop_on_cancel() {
        let mut lc = LoopControl::new(3);
        lc.cancel();
        assert!(!lc.should_continue());
    }

    #[test]
    fn test_loop_control_record_mistake_under_limit() {
        let mut lc = LoopControl::new(3);
        let exceeded = lc.record_mistake();
        assert!(!exceeded);
        assert_eq!(lc.consecutive_mistake_count, 1);
        assert!(lc.did_tool_fail_in_current_turn);
    }

    #[test]
    fn test_loop_control_record_mistake_at_limit() {
        let mut lc = LoopControl::new(3);
        lc.record_mistake(); // 1
        lc.record_mistake(); // 2
        lc.record_mistake(); // 3 — count == limit, should_continue still true
        assert!(lc.should_continue()); // 3 > 3 is false
        let exceeded = lc.record_mistake(); // 4 — exceeds limit
        assert!(exceeded); // 4 > 3
        assert!(!lc.should_continue()); // 4 > 3
    }

    #[test]
    fn test_loop_control_reset_mistake_count() {
        let mut lc = LoopControl::new(3);
        lc.record_mistake();
        lc.record_mistake();
        assert_eq!(lc.consecutive_mistake_count, 2);

        lc.reset_mistake_count();
        assert_eq!(lc.consecutive_mistake_count, 0);
        assert!(!lc.did_tool_fail_in_current_turn);
    }

    #[test]
    fn test_loop_control_increment_iteration_no_limit() {
        let mut lc = LoopControl::new(3);
        for _ in 0..100 {
            let reached = lc.increment_iteration();
            assert!(!reached);
        }
        assert_eq!(lc.current_iteration, 100);
    }

    #[test]
    fn test_loop_control_increment_iteration_with_limit() {
        let mut lc = LoopControl::with_max_iterations(3, 5);

        assert!(!lc.increment_iteration()); // 1
        assert!(!lc.increment_iteration()); // 2
        assert!(!lc.increment_iteration()); // 3
        assert!(!lc.increment_iteration()); // 4
        assert!(lc.increment_iteration());  // 5 = limit
    }

    #[test]
    fn test_loop_control_should_stop_at_max_iterations() {
        let mut lc = LoopControl::with_max_iterations(3, 2);
        lc.increment_iteration(); // 1
        lc.increment_iteration(); // 2
        assert!(!lc.should_continue()); // 2 >= 2
    }

    #[test]
    fn test_loop_control_remaining_iterations() {
        let mut lc = LoopControl::with_max_iterations(3, 10);
        assert_eq!(lc.remaining_iterations(), Some(10));
        lc.increment_iteration();
        assert_eq!(lc.remaining_iterations(), Some(9));
    }

    #[test]
    fn test_loop_control_remaining_iterations_no_limit() {
        let lc = LoopControl::new(3);
        assert_eq!(lc.remaining_iterations(), None);
    }

    #[test]
    fn test_loop_control_reset_turn() {
        let mut lc = LoopControl::new(3);
        lc.record_mistake();
        assert!(lc.did_tool_fail_in_current_turn);
        lc.reset_turn();
        assert!(!lc.did_tool_fail_in_current_turn);
        // Mistake count should NOT be reset
        assert_eq!(lc.consecutive_mistake_count, 1);
    }

    #[test]
    fn test_loop_control_full_scenario() {
        let mut lc = LoopControl::with_max_iterations(2, 10);

        // Iteration 1: success
        assert!(lc.should_continue());
        lc.increment_iteration();
        lc.reset_mistake_count();

        // Iteration 2: mistake
        assert!(lc.should_continue());
        lc.increment_iteration();
        let exceeded = lc.record_mistake();
        assert!(!exceeded);

        // Iteration 3: another mistake
        assert!(lc.should_continue());
        lc.increment_iteration();
        let exceeded = lc.record_mistake();
        assert!(!exceeded); // 2 mistakes, limit is 2, 2 > 2 = false

        // Iteration 4: another mistake (exceeds limit)
        lc.reset_turn();
        let exceeded = lc.record_mistake();
        assert!(exceeded); // 3 > 2
        assert!(!lc.should_continue());
    }

    #[test]
    fn test_loop_control_cancel_idempotent() {
        let mut lc = LoopControl::new(3);
        lc.cancel();
        lc.cancel();
        assert!(lc.is_cancelled);
        assert!(!lc.should_continue());
    }
}
