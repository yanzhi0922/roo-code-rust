//! Interactive ask/say flow for task communication.
//!
//! Implements the `ask()` and `say()` methods from `Task.ts` which handle
//! bidirectional communication between the task engine and the user/UI.
//!
//! Source: `src/core/task/Task.ts` — `ask()`, `say()`, `handleWebviewAskResponse()`,
//! `approveAsk()`, `denyAsk()`, `supersedePendingAsk()`, `cancelAutoApprovalTimeout()`

use std::sync::Arc;

use tokio::sync::{watch, Mutex};
use tracing::{debug, warn};

use roo_types::message::{ClineAsk, ClineMessage, ClineSay, MessageType};

use crate::types::TaskError;

// ---------------------------------------------------------------------------
// AskResponse
// ---------------------------------------------------------------------------

/// Response from a user to an ask prompt.
///
/// Source: `src/shared/WebviewMessage.ts` — `ClineAskResponse`
#[derive(Debug, Clone, PartialEq)]
pub enum AskResponse {
    /// User clicked "Yes" / approved.
    YesButtonClicked,
    /// User clicked "No" / denied.
    NoButtonClicked,
    /// User provided a message response.
    MessageResponse,
}

// ---------------------------------------------------------------------------
// AskIgnoredError
// ---------------------------------------------------------------------------

/// Error indicating the ask was ignored because it was a partial update.
///
/// Source: `src/core/task/AskIgnoredError.ts`
#[derive(Debug, Clone)]
pub struct AskIgnoredError {
    pub reason: String,
}

impl std::fmt::Display for AskIgnoredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AskIgnoredError: {}", self.reason)
    }
}

impl std::error::Error for AskIgnoredError {}

// ---------------------------------------------------------------------------
// AskResult
// ---------------------------------------------------------------------------

/// Result of an ask operation.
#[derive(Debug, Clone)]
pub struct AskResult {
    pub response: AskResponse,
    pub text: Option<String>,
    pub images: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// AskSayHandler
// ---------------------------------------------------------------------------

/// Handler for the ask/say interactive communication flow.
///
/// Source: `src/core/task/Task.ts` — `ask()`, `say()`, and related methods
///
/// The ask/say flow works as follows:
/// - `say()` emits a message to the UI without waiting for a response
/// - `ask()` emits a message and waits for a user response
/// - `handle_response()` fulfills the pending ask with a user response
///
/// In the TS implementation, `ask()` uses `pWaitFor` to poll for a response.
/// In Rust, we use `tokio::sync::watch` for efficient async waiting.
pub struct AskSayHandler {
    /// Messages emitted during the task (clineMessages).
    cline_messages: Vec<ClineMessage>,
    /// Current pending ask response.
    ask_response: Arc<Mutex<Option<AskResult>>>,
    /// Timestamp of the last message.
    last_message_ts: Option<f64>,
    /// Watch channel for signaling ask response availability.
    ask_signal: watch::Sender<bool>,
    ask_signal_rx: watch::Receiver<bool>,
}

impl AskSayHandler {
    /// Create a new ask/say handler.
    pub fn new() -> Self {
        let (tx, rx) = watch::channel(false);
        Self {
            cline_messages: Vec::new(),
            ask_response: Arc::new(Mutex::new(None)),
            last_message_ts: None,
            ask_signal: tx,
            ask_signal_rx: rx,
        }
    }

    // -------------------------------------------------------------------
    // say()
    // -------------------------------------------------------------------

    /// Emit a message to the UI without waiting for a response.
    ///
    /// Source: `src/core/task/Task.ts` — `say()`
    ///
    /// Creates a ClineMessage with type "say" and adds it to the message list.
    /// Also saves messages and posts state to webview.
    pub async fn say(
        &mut self,
        say_type: ClineSay,
        text: Option<String>,
        images: Option<Vec<String>>,
    ) -> Result<(), TaskError> {
        let ts = now_ts();
        self.last_message_ts = Some(ts);

        let message = ClineMessage {
            ts,
            r#type: MessageType::Say,
            ask: None,
            say: Some(say_type),
            text,
            images,
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

        let say_type = message.say;
        self.cline_messages.push(message);
        debug!(say_type = ?say_type, "say() emitted");
        Ok(())
    }

    /// Emit a partial (streaming) say message.
    ///
    /// Source: `src/core/task/Task.ts` — `say()` with `partial` parameter
    pub async fn say_partial(
        &mut self,
        say_type: ClineSay,
        text: Option<String>,
        partial: bool,
    ) -> Result<(), TaskError> {
        if partial {
            // Check if we can update an existing partial message
            if let Some(last) = self.cline_messages.last_mut() {
                if last.partial == Some(true) && last.say == Some(say_type) {
                    last.text = text;
                    return Ok(());
                }
            }
            // New partial message
            let ts = now_ts();
            self.last_message_ts = Some(ts);
            let message = ClineMessage {
                ts,
                r#type: MessageType::Say,
                ask: None,
                say: Some(say_type),
                text,
                images: None,
                partial: Some(true),
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
            self.cline_messages.push(message);
        } else {
            // Finalize partial message
            if let Some(last) = self.cline_messages.last_mut() {
                if last.partial == Some(true) && last.say == Some(say_type) {
                    last.text = text;
                    last.partial = Some(false);
                    return Ok(());
                }
            }
            // No matching partial — just do a regular say
            return self.say(say_type, text, None).await;
        }
        Ok(())
    }

    // -------------------------------------------------------------------
    // ask()
    // -------------------------------------------------------------------

    /// Ask the user a question and wait for a response.
    ///
    /// Source: `src/core/task/Task.ts` — `ask()`
    ///
    /// Creates a ClineMessage with type "ask" and waits for the user to respond.
    /// Supports partial updates (streaming) where the question text is updated
    /// before the final version is presented.
    ///
    /// # Arguments
    /// * `ask_type` - The type of question being asked
    /// * `text` - Optional text for the question
    /// * `partial` - Whether this is a partial (streaming) update
    ///
    /// # Returns
    /// * `Ok(AskResult)` - The user's response
    /// * `Err(AskIgnoredError)` - If this was a partial update (not a real ask)
    pub async fn ask(
        &mut self,
        ask_type: ClineAsk,
        text: Option<String>,
        partial: Option<bool>,
    ) -> Result<AskResult, AskIgnoredError> {
        let ask_ts = match partial {
            Some(true) => {
                // Partial update — check if updating existing partial
                if let Some(last) = self.cline_messages.last() {
                    if last.partial == Some(true)
                        && last.r#type == MessageType::Ask
                        && last.ask == Some(ask_type)
                    {
                        // Update existing partial
                        if let Some(last) = self.cline_messages.last_mut() {
                            last.text = text;
                        }
                        return Err(AskIgnoredError {
                            reason: "updating existing partial".to_string(),
                        });
                    }
                }
                // New partial message
                let ts = now_ts();
                self.last_message_ts = Some(ts);
                let message = ClineMessage {
                    ts,
                    r#type: MessageType::Ask,
                    ask: Some(ask_type),
                    say: None,
                    text,
                    images: None,
                    partial: Some(true),
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
                self.cline_messages.push(message);
                return Err(AskIgnoredError {
                    reason: "new partial".to_string(),
                });
            }
            Some(false) => {
                // Finalize partial
                if let Some(last) = self.cline_messages.last() {
                    if last.partial == Some(true)
                        && last.r#type == MessageType::Ask
                        && last.ask == Some(ask_type)
                    {
                        let ts = last.ts;
                        self.last_message_ts = Some(ts);
                        if let Some(last) = self.cline_messages.last_mut() {
                            last.text = text;
                            last.partial = Some(false);
                        }
                        ts
                    } else {
                        // New complete message
                        self.clear_ask_response();
                        let ts = now_ts();
                        self.last_message_ts = Some(ts);
                        let message = ClineMessage {
                            ts,
                            r#type: MessageType::Ask,
                            ask: Some(ask_type),
                            say: None,
                            text,
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
                        self.cline_messages.push(message);
                        ts
                    }
                } else {
                    // No messages yet — new complete message
                    self.clear_ask_response();
                    let ts = now_ts();
                    self.last_message_ts = Some(ts);
                    let message = ClineMessage {
                        ts,
                        r#type: MessageType::Ask,
                        ask: Some(ask_type),
                        say: None,
                        text,
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
                    self.cline_messages.push(message);
                    ts
                }
            }
            None => {
                // New non-partial message
                self.clear_ask_response();
                let ts = now_ts();
                self.last_message_ts = Some(ts);
                let message = ClineMessage {
                    ts,
                    r#type: MessageType::Ask,
                    ask: Some(ask_type),
                    say: None,
                    text,
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
                self.cline_messages.push(message);
                ts
            }
        };

        // Wait for response
        self.wait_for_ask_response(ask_ts).await
    }

    /// Wait for the ask response to be set.
    ///
    /// Source: `src/core/task/Task.ts` — `pWaitFor` in `ask()`
    async fn wait_for_ask_response(&self, ask_ts: f64) -> Result<AskResult, AskIgnoredError> {
        let mut rx = self.ask_signal_rx.clone();
        let response = self.ask_response.clone();

        loop {
            // Check if response is already available
            {
                let guard = response.lock().await;
                if let Some(result) = guard.as_ref() {
                    return Ok(result.clone());
                }
            }

            // Check if the message was superseded
            if let Some(ts) = self.last_message_ts {
                if ts != ask_ts {
                    // Message was superseded — check if there's a response
                    let guard = response.lock().await;
                    if let Some(result) = guard.as_ref() {
                        return Ok(result.clone());
                    }
                    // No response but superseded — this shouldn't normally happen
                    warn!("Ask was superseded without a response");
                }
            }

            // Wait for signal
            if rx.changed().await.is_err() {
                // Channel closed — task was likely aborted
                return Err(AskIgnoredError {
                    reason: "ask channel closed (task likely aborted)".to_string(),
                });
            }
        }
    }

    // -------------------------------------------------------------------
    // handle_response()
    // -------------------------------------------------------------------

    /// Handle a response from the webview/user to a pending ask.
    ///
    /// Source: `src/core/task/Task.ts` — `handleWebviewAskResponse()`
    pub async fn handle_response(
        &self,
        ask_response: AskResponse,
        text: Option<String>,
        images: Option<Vec<String>>,
    ) {
        let result = AskResult {
            response: ask_response,
            text,
            images,
        };

        {
            let mut guard = self.ask_response.lock().await;
            *guard = Some(result);
        }

        // Signal that a response is available
        let _ = self.ask_signal.send(true);
    }

    /// Approve the current ask (auto-approve).
    ///
    /// Source: `src/core/task/Task.ts` — `approveAsk()`
    pub async fn approve_ask(&self, text: Option<String>, images: Option<Vec<String>>) {
        self.handle_response(AskResponse::YesButtonClicked, text, images)
            .await;
    }

    /// Deny the current ask (auto-deny).
    ///
    /// Source: `src/core/task/Task.ts` — `denyAsk()`
    pub async fn deny_ask(&self, text: Option<String>, images: Option<Vec<String>>) {
        self.handle_response(AskResponse::NoButtonClicked, text, images)
            .await;
    }

    /// Supersede the pending ask by updating the timestamp.
    ///
    /// Source: `src/core/task/Task.ts` — `supersedePendingAsk()`
    pub fn supersede_pending_ask(&mut self) {
        self.last_message_ts = Some(now_ts());
    }

    // -------------------------------------------------------------------
    // Getters
    // -------------------------------------------------------------------

    /// Get all cline messages.
    pub fn cline_messages(&self) -> &[ClineMessage] {
        &self.cline_messages
    }

    /// Get a mutable reference to cline messages.
    pub fn cline_messages_mut(&mut self) -> &mut Vec<ClineMessage> {
        &mut self.cline_messages
    }

    /// Get the last message timestamp.
    pub fn last_message_ts(&self) -> Option<f64> {
        self.last_message_ts
    }

    /// Find a message by timestamp.
    ///
    /// Source: `src/core/task/Task.ts` — `findMessageByTimestamp()`
    pub fn find_message_by_timestamp(&self, ts: f64) -> Option<&ClineMessage> {
        for i in (0..self.cline_messages.len()).rev() {
            if (self.cline_messages[i].ts - ts).abs() < 0.001 {
                return Some(&self.cline_messages[i]);
            }
        }
        None
    }

    /// Find a mutable message by timestamp.
    pub fn find_message_by_timestamp_mut(&mut self, ts: f64) -> Option<&mut ClineMessage> {
        for i in (0..self.cline_messages.len()).rev() {
            if (self.cline_messages[i].ts - ts).abs() < 0.001 {
                return Some(&mut self.cline_messages[i]);
            }
        }
        None
    }

    /// Overwrite all cline messages.
    ///
    /// Source: `src/core/task/Task.ts` — `overwriteClineMessages()`
    pub fn overwrite_cline_messages(&mut self, messages: Vec<ClineMessage>) {
        self.cline_messages = messages;
    }

    /// Update a specific cline message.
    ///
    /// Source: `src/core/task/Task.ts` — `updateClineMessage()`
    pub fn update_cline_message(&mut self, message: &ClineMessage) {
        if let Some(existing) = self
            .cline_messages
            .iter_mut()
            .rev()
            .find(|m| m.ts == message.ts)
        {
            *existing = message.clone();
        }
    }

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    /// Clear the current ask response.
    fn clear_ask_response(&self) {
        // We can't clear the Mutex directly from a non-async context,
        // so we use a blocking lock here (acceptable since it's very brief).
        if let Ok(mut guard) = self.ask_response.try_lock() {
            *guard = None;
        }
    }
}

impl Default for AskSayHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the current timestamp in milliseconds.
fn now_ts() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ask_say_handler_new() {
        let handler = AskSayHandler::new();
        assert!(handler.cline_messages().is_empty());
        assert!(handler.last_message_ts().is_none());
    }

    #[tokio::test]
    async fn test_say_creates_message() {
        let mut handler = AskSayHandler::new();
        handler
            .say(ClineSay::Text, Some("Hello".to_string()), None)
            .await
            .unwrap();

        assert_eq!(handler.cline_messages().len(), 1);
        let msg = &handler.cline_messages()[0];
        assert_eq!(msg.r#type, MessageType::Say);
        assert_eq!(msg.say, Some(ClineSay::Text));
        assert_eq!(msg.text, Some("Hello".to_string()));
    }

    #[tokio::test]
    async fn test_say_partial_update() {
        let mut handler = AskSayHandler::new();

        // First partial
        handler
            .say_partial(ClineSay::Text, Some("Hello".to_string()), true)
            .await
            .unwrap();
        assert_eq!(handler.cline_messages().len(), 1);

        // Update partial
        handler
            .say_partial(ClineSay::Text, Some("Hello world".to_string()), true)
            .await
            .unwrap();
        assert_eq!(handler.cline_messages().len(), 1); // Updated, not added
        assert_eq!(
            handler.cline_messages()[0].text,
            Some("Hello world".to_string())
        );

        // Finalize
        handler
            .say_partial(ClineSay::Text, Some("Hello world!".to_string()), false)
            .await
            .unwrap();
        assert_eq!(handler.cline_messages().len(), 1);
        assert_eq!(handler.cline_messages()[0].partial, Some(false));
    }

    #[tokio::test]
    async fn test_ask_partial_returns_error() {
        let mut handler = AskSayHandler::new();
        let result = handler
            .ask(ClineAsk::Followup, Some("Question?".to_string()), Some(true))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.reason.contains("partial"));
    }

    #[tokio::test]
    async fn test_ask_and_respond() {
        let mut handler = AskSayHandler::new();

        // Spawn a task that will respond after a short delay
        let response = handler.ask_response.clone();
        let signal = handler.ask_signal.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let mut guard = response.lock().await;
            *guard = Some(AskResult {
                response: AskResponse::YesButtonClicked,
                text: Some("Yes".to_string()),
                images: None,
            });
            let _ = signal.send(true);
        });

        let result = handler
            .ask(ClineAsk::Tool, Some("Approve?".to_string()), None)
            .await;
        assert!(result.is_ok());
        let ask_result = result.unwrap();
        assert_eq!(ask_result.response, AskResponse::YesButtonClicked);
        assert_eq!(ask_result.text, Some("Yes".to_string()));
    }

    #[tokio::test]
    async fn test_handle_response() {
        let handler = AskSayHandler::new();
        handler
            .handle_response(
                AskResponse::MessageResponse,
                Some("My response".to_string()),
                None,
            )
            .await;

        let guard = handler.ask_response.lock().await;
        assert!(guard.is_some());
        let result = guard.as_ref().unwrap();
        assert_eq!(result.response, AskResponse::MessageResponse);
        assert_eq!(result.text, Some("My response".to_string()));
    }

    #[tokio::test]
    async fn test_approve_deny_ask() {
        let handler = AskSayHandler::new();

        handler
            .approve_ask(Some("ok".to_string()), None)
            .await;
        {
            let guard = handler.ask_response.lock().await;
            assert_eq!(
                guard.as_ref().unwrap().response,
                AskResponse::YesButtonClicked
            );
        }

        // Reset and deny
        {
            let mut guard = handler.ask_response.lock().await;
            *guard = None;
        }

        handler.deny_ask(None, None).await;
        {
            let guard = handler.ask_response.lock().await;
            assert_eq!(
                guard.as_ref().unwrap().response,
                AskResponse::NoButtonClicked
            );
        }
    }

    #[test]
    fn test_supersede_pending_ask() {
        let mut handler = AskSayHandler::new();
        assert!(handler.last_message_ts().is_none());
        handler.supersede_pending_ask();
        assert!(handler.last_message_ts().is_some());
    }

    #[test]
    fn test_find_message_by_timestamp() {
        let mut handler = AskSayHandler::new();
        let ts = 1700000000.0;
        let msg = ClineMessage {
            ts,
            r#type: MessageType::Say,
            ask: None,
            say: Some(ClineSay::Text),
            text: Some("test".to_string()),
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
        handler.cline_messages.push(msg);

        let found = handler.find_message_by_timestamp(ts);
        assert!(found.is_some());
        assert_eq!(found.unwrap().text, Some("test".to_string()));

        let not_found = handler.find_message_by_timestamp(99999.0);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_overwrite_cline_messages() {
        let mut handler = AskSayHandler::new();
        handler.cline_messages.push(ClineMessage {
            ts: 1.0,
            r#type: MessageType::Say,
            ask: None,
            say: None,
            text: None,
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
        });
        assert_eq!(handler.cline_messages().len(), 1);

        handler.overwrite_cline_messages(Vec::new());
        assert!(handler.cline_messages().is_empty());
    }

    #[test]
    fn test_ask_response_equality() {
        assert_eq!(AskResponse::YesButtonClicked, AskResponse::YesButtonClicked);
        assert_ne!(AskResponse::YesButtonClicked, AskResponse::NoButtonClicked);
        assert_ne!(AskResponse::YesButtonClicked, AskResponse::MessageResponse);
    }

    #[test]
    fn test_ask_ignored_error_display() {
        let err = AskIgnoredError {
            reason: "test".to_string(),
        };
        assert!(err.to_string().contains("test"));
    }
}
