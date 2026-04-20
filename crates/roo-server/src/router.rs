//! JSON-RPC method router.
//!
//! Provides a simple routing mechanism that maps JSON-RPC method names
//! to handler functions. This is the dispatch layer between the transport
//! and the handler.

use roo_jsonrpc::types::Message;

use crate::handler::Handler;

/// A JSON-RPC method router.
///
/// The router receives incoming JSON-RPC messages and dispatches them
/// to the appropriate handler method.
pub struct Router {
    handler: Handler,
}

impl Router {
    /// Create a new router with the given handler.
    pub fn new(handler: Handler) -> Self {
        Self { handler }
    }

    /// Route a single JSON-RPC message to the handler.
    ///
    /// For requests, returns the handler's response.
    /// For notifications, returns a notification acknowledgment.
    /// For responses (shouldn't happen server-side), returns an error.
    pub async fn route(&self, message: &Message) -> Message {
        if message.is_request() {
            self.handler.handle(message).await
        } else if message.is_notification() {
            // Notifications don't get responses, but we still process them
            tracing::debug!(
                method = ?message.method,
                "Processing notification"
            );
            self.handler.handle(message).await
        } else if message.is_response() || message.is_error() {
            // Server shouldn't receive responses
            tracing::warn!("Received unexpected response message, ignoring");
            Message::error_response(
                message.id.clone().unwrap_or(serde_json::Value::Null),
                roo_jsonrpc::types::error_codes::INVALID_REQUEST,
                "Server does not accept response messages",
            )
        } else {
            Message::error_response(
                message.id.clone().unwrap_or(serde_json::Value::Null),
                roo_jsonrpc::types::error_codes::INVALID_REQUEST,
                "Unknown message type",
            )
        }
    }

    /// Route a batch of JSON-RPC messages.
    ///
    /// Processes each message independently and returns a batch of responses.
    /// Notifications in the batch are processed but don't generate responses.
    pub async fn route_batch(&self, messages: &[Message]) -> Vec<Message> {
        let mut responses = Vec::with_capacity(messages.len());
        for msg in messages {
            let response = self.route(msg).await;
            // Only include responses for requests (not notifications)
            if msg.is_request() {
                responses.push(response);
            }
        }
        responses
    }
}

/// List of all supported JSON-RPC method names.
///
/// Used for validation and capability reporting.
pub fn supported_methods() -> &'static [&'static str] {
    &[
        crate::handler::methods::INITIALIZE,
        crate::handler::methods::SHUTDOWN,
        crate::handler::methods::PING,
        crate::handler::methods::TASK_START,
        crate::handler::methods::TASK_CANCEL,
        crate::handler::methods::TASK_CLOSE,
        crate::handler::methods::TASK_RESUME,
        crate::handler::methods::TASK_SEND_MESSAGE,
        crate::handler::methods::TASK_GET_COMMANDS,
        crate::handler::methods::TASK_GET_MODES,
        crate::handler::methods::TASK_GET_MODELS,
        crate::handler::methods::TASK_DELETE_QUEUED_MESSAGE,
        crate::handler::methods::TASK_CONDENSE,
        crate::handler::methods::STATE_GET,
        crate::handler::methods::STATE_SET_MODE,
        crate::handler::methods::SYSTEM_PROMPT_BUILD,
        crate::handler::methods::HISTORY_GET,
        crate::handler::methods::HISTORY_DELETE,
        crate::handler::methods::HISTORY_EXPORT,
        crate::handler::methods::TODO_UPDATE,
        crate::handler::methods::ASK_RESPONSE,
        crate::handler::methods::TERMINAL_OPERATION,
        crate::handler::methods::CHECKPOINT_DIFF,
        crate::handler::methods::CHECKPOINT_RESTORE,
        crate::handler::methods::PROMPT_ENHANCE,
        crate::handler::methods::SEARCH_FILES,
        crate::handler::methods::FILE_READ,
        crate::handler::methods::MCP_LIST_SERVERS,
        crate::handler::methods::MCP_RESTART_SERVER,
        crate::handler::methods::MCP_TOGGLE_SERVER,
        crate::handler::methods::MCP_USE_TOOL,
        crate::handler::methods::MCP_ACCESS_RESOURCE,
        crate::handler::methods::MCP_DELETE_SERVER,
        crate::handler::methods::MCP_UPDATE_TIMEOUT,
        crate::handler::methods::SETTINGS_UPDATE,
        crate::handler::methods::SETTINGS_SAVE_API_CONFIG,
        crate::handler::methods::SETTINGS_LOAD_API_CONFIG,
        crate::handler::methods::SETTINGS_LOAD_API_CONFIG_BY_ID,
        crate::handler::methods::SETTINGS_DELETE_API_CONFIG,
        crate::handler::methods::SETTINGS_LIST_API_CONFIGS,
        crate::handler::methods::SETTINGS_UPSERT_API_CONFIG,
        crate::handler::methods::SKILLS_LIST,
        crate::handler::methods::SKILLS_CREATE,
        crate::handler::methods::SKILLS_DELETE,
        crate::handler::methods::SKILLS_MOVE,
        crate::handler::methods::SKILLS_UPDATE_MODES,
        crate::handler::methods::MODE_UPDATE_CUSTOM,
        crate::handler::methods::MODE_DELETE_CUSTOM,
        crate::handler::methods::MESSAGE_DELETE,
        crate::handler::methods::MESSAGE_EDIT,
        crate::handler::methods::MESSAGE_QUEUE,
        crate::handler::methods::TELEMETRY_SET_SETTING,
    ]
}

/// Check if a method name is supported.
pub fn is_supported_method(method: &str) -> bool {
    supported_methods().contains(&method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handler::Handler;
    use roo_app::{App, AppConfig};
    use roo_jsonrpc::types::Message;
    use serde_json::json;

    fn test_router() -> Router {
        let config = AppConfig {
            cwd: "/tmp/test".to_string(),
            mode: "code".to_string(),
            ..Default::default()
        };
        let handler = Handler::new(App::new(config));
        Router::new(handler)
    }

    #[tokio::test]
    async fn test_route_request() {
        let router = test_router();
        let request = Message::request(1, "ping", json!(null));
        let response = router.route(&request).await;
        assert_eq!(response.result.unwrap(), json!("pong"));
    }

    #[tokio::test]
    async fn test_route_batch() {
        let router = test_router();
        let messages = vec![
            Message::request(1, "initialize", json!(null)),
            Message::request(2, "ping", json!(null)),
            Message::request(3, "state/get", json!(null)),
        ];
        let responses = router.route_batch(&messages).await;
        assert_eq!(responses.len(), 3);
    }

    #[test]
    fn test_supported_methods() {
        let methods = supported_methods();
        assert!(methods.contains(&"initialize"));
        assert!(methods.contains(&"ping"));
        assert!(methods.contains(&"task/start"));
        assert!(methods.contains(&"state/get"));
    }

    #[test]
    fn test_is_supported_method() {
        assert!(is_supported_method("initialize"));
        assert!(is_supported_method("ping"));
        assert!(!is_supported_method("nonexistent"));
    }
}
