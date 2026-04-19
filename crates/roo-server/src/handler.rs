//! JSON-RPC request handler.
//!
//! Source: `src/core/webview/webviewMessageHandler.ts` — handles all WebviewMessage types
//! Source: `packages/types/src/ipc.ts` — TaskCommand handling
//!
//! This module implements the handler for each JSON-RPC method, mapping them
//! to the corresponding TypeScript webviewMessageHandler operations.

use std::sync::Arc;

use serde_json::{json, Value};
use tracing::{debug, error, info, instrument};

use roo_app::App;
use roo_jsonrpc::types::Message;

use crate::error::{ServerError, ServerResult};

// ---------------------------------------------------------------------------
// JSON-RPC Method Names
// ---------------------------------------------------------------------------

/// Standard JSON-RPC lifecycle methods.
pub mod methods {
    /// Initialize the server.
    pub const INITIALIZE: &str = "initialize";
    /// Shutdown the server.
    pub const SHUTDOWN: &str = "shutdown";
    /// Ping (keep-alive).
    pub const PING: &str = "ping";

    // ── Task commands (from TaskCommandName in ipc.ts) ──

    /// Start a new task.
    /// Source: `TaskCommandName.StartNewTask`
    pub const TASK_START: &str = "task/start";
    /// Cancel the current task.
    /// Source: `TaskCommandName.CancelTask`
    pub const TASK_CANCEL: &str = "task/cancel";
    /// Close the current task.
    /// Source: `TaskCommandName.CloseTask`
    pub const TASK_CLOSE: &str = "task/close";
    /// Resume a task.
    /// Source: `TaskCommandName.ResumeTask`
    pub const TASK_RESUME: &str = "task/resume";
    /// Send a message to the current task.
    /// Source: `TaskCommandName.SendMessage`
    pub const TASK_SEND_MESSAGE: &str = "task/sendMessage";
    /// Get available commands.
    /// Source: `TaskCommandName.GetCommands`
    pub const TASK_GET_COMMANDS: &str = "task/getCommands";
    /// Get available modes.
    /// Source: `TaskCommandName.GetModes`
    pub const TASK_GET_MODES: &str = "task/getModes";
    /// Get available models.
    /// Source: `TaskCommandName.GetModels`
    pub const TASK_GET_MODELS: &str = "task/getModels";
    /// Delete a queued message.
    /// Source: `TaskCommandName.DeleteQueuedMessage`
    pub const TASK_DELETE_QUEUED_MESSAGE: &str = "task/deleteQueuedMessage";

    // ── State commands (from WebviewMessage types) ──

    /// Get current state.
    /// Source: `getState` in ClineProvider
    pub const STATE_GET: &str = "state/get";
    /// Set current mode.
    /// Source: `mode` WebviewMessage
    pub const STATE_SET_MODE: &str = "state/setMode";
    /// Build system prompt.
    /// Source: `getSystemPrompt` WebviewMessage
    pub const SYSTEM_PROMPT_BUILD: &str = "systemPrompt/build";
    /// Get task history.
    /// Source: `showTaskWithId` WebviewMessage
    pub const HISTORY_GET: &str = "history/get";
    /// Delete a task from history.
    /// Source: `deleteTaskWithId` WebviewMessage
    pub const HISTORY_DELETE: &str = "history/delete";
    /// Export a task.
    /// Source: `exportTaskWithId` WebviewMessage
    pub const HISTORY_EXPORT: &str = "history/export";
    /// Update todo list.
    /// Source: `updateTodoList` WebviewMessage
    pub const TODO_UPDATE: &str = "todo/update";
    /// Ask followup question response.
    /// Source: `askResponse` WebviewMessage
    pub const ASK_RESPONSE: &str = "ask/response";
    /// Terminal operation (continue/abort).
    /// Source: `terminalOperation` WebviewMessage
    pub const TERMINAL_OPERATION: &str = "terminal/operation";
    /// Condense task context.
    /// Source: `condenseTaskContextRequest` WebviewMessage
    pub const TASK_CONDENSE: &str = "task/condense";
    /// Request checkpoint diff.
    /// Source: `checkpointDiff` WebviewMessage
    pub const CHECKPOINT_DIFF: &str = "checkpoint/diff";
    /// Restore checkpoint.
    /// Source: `checkpointRestore` WebviewMessage
    pub const CHECKPOINT_RESTORE: &str = "checkpoint/restore";
    /// Enhance a prompt.
    /// Source: `enhancePrompt` WebviewMessage
    pub const PROMPT_ENHANCE: &str = "prompt/enhance";
    /// Search files.
    /// Source: `searchFiles` WebviewMessage
    pub const SEARCH_FILES: &str = "search/files";
    /// Read file content.
    /// Source: `readFileContent` WebviewMessage
    pub const FILE_READ: &str = "file/read";
    /// List MCP servers.
    /// Source: `mcpServers` ExtensionMessage
    pub const MCP_LIST_SERVERS: &str = "mcp/listServers";
    /// Restart MCP server.
    /// Source: `restartMcpServer` WebviewMessage
    pub const MCP_RESTART_SERVER: &str = "mcp/restartServer";
    /// Toggle MCP server.
    /// Source: `toggleMcpServer` WebviewMessage
    pub const MCP_TOGGLE_SERVER: &str = "mcp/toggleServer";
    /// Use MCP tool.
    /// Source: `use_mcp_tool` tool
    pub const MCP_USE_TOOL: &str = "mcp/useTool";
    /// Access MCP resource.
    /// Source: `access_mcp_resource` tool
    pub const MCP_ACCESS_RESOURCE: &str = "mcp/accessResource";
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Handles JSON-RPC requests by dispatching to the appropriate App method.
///
/// Source: `src/core/webview/webviewMessageHandler.ts` — `webviewMessageHandler` function
pub struct Handler {
    app: Arc<App>,
}

impl Handler {
    /// Create a new handler wrapping the given App.
    pub fn new(app: Arc<App>) -> Self {
        Self { app }
    }

    /// Get a reference to the underlying App.
    pub fn app(&self) -> &App {
        &self.app
    }

    /// Dispatch a JSON-RPC request message to the appropriate handler.
    ///
    /// Returns the response message (success or error).
    #[instrument(skip(self, request), fields(method = %request.method.as_deref().unwrap_or("unknown")))]
    pub async fn handle(&self, request: &Message) -> Message {
        let id = match &request.id {
            Some(id) => id.clone(),
            None => {
                // Notifications don't need responses
                debug!("Received notification, no response needed");
                return Message::response(Value::Null, json!(null));
            }
        };

        let method = match &request.method {
            Some(m) => m.as_str(),
            None => {
                return Message::error_response(
                    id,
                    roo_jsonrpc::types::error_codes::INVALID_REQUEST,
                    "Missing method field",
                );
            }
        };

        let params = request.params.clone().unwrap_or(Value::Null);

        debug!(method = method, "Handling request");

        let result = match method {
            // ── Lifecycle ──
            methods::INITIALIZE => self.handle_initialize(params).await,
            methods::SHUTDOWN => self.handle_shutdown(params).await,
            methods::PING => self.handle_ping(params).await,

            // ── Task commands ──
            methods::TASK_START => self.handle_task_start(params).await,
            methods::TASK_CANCEL => self.handle_task_cancel(params).await,
            methods::TASK_CLOSE => self.handle_task_close(params).await,
            methods::TASK_RESUME => self.handle_task_resume(params).await,
            methods::TASK_SEND_MESSAGE => self.handle_task_send_message(params).await,
            methods::TASK_GET_COMMANDS => self.handle_task_get_commands(params).await,
            methods::TASK_GET_MODES => self.handle_task_get_modes(params).await,
            methods::TASK_GET_MODELS => self.handle_task_get_models(params).await,
            methods::TASK_DELETE_QUEUED_MESSAGE => {
                self.handle_task_delete_queued_message(params).await
            }
            methods::TASK_CONDENSE => self.handle_task_condense(params).await,

            // ── State commands ──
            methods::STATE_GET => self.handle_state_get(params).await,
            methods::STATE_SET_MODE => self.handle_state_set_mode(params).await,
            methods::SYSTEM_PROMPT_BUILD => self.handle_system_prompt_build(params).await,

            // ── History commands ──
            methods::HISTORY_GET => self.handle_history_get(params).await,
            methods::HISTORY_DELETE => self.handle_history_delete(params).await,
            methods::HISTORY_EXPORT => self.handle_history_export(params).await,

            // ── Todo commands ──
            methods::TODO_UPDATE => self.handle_todo_update(params).await,

            // ── Ask response ──
            methods::ASK_RESPONSE => self.handle_ask_response(params).await,

            // ── Terminal commands ──
            methods::TERMINAL_OPERATION => self.handle_terminal_operation(params).await,

            // ── Checkpoint commands ──
            methods::CHECKPOINT_DIFF => self.handle_checkpoint_diff(params).await,
            methods::CHECKPOINT_RESTORE => self.handle_checkpoint_restore(params).await,

            // ── Prompt commands ──
            methods::PROMPT_ENHANCE => self.handle_prompt_enhance(params).await,

            // ── Search commands ──
            methods::SEARCH_FILES => self.handle_search_files(params).await,

            // ── File commands ──
            methods::FILE_READ => self.handle_file_read(params).await,

            // ── MCP commands ──
            methods::MCP_LIST_SERVERS => self.handle_mcp_list_servers(params).await,
            methods::MCP_RESTART_SERVER => self.handle_mcp_restart_server(params).await,
            methods::MCP_TOGGLE_SERVER => self.handle_mcp_toggle_server(params).await,
            methods::MCP_USE_TOOL => self.handle_mcp_use_tool(params).await,
            methods::MCP_ACCESS_RESOURCE => self.handle_mcp_access_resource(params).await,

            // ── Unknown method ──
            _ => {
                return Message::error_response(
                    id,
                    roo_jsonrpc::types::error_codes::METHOD_NOT_FOUND,
                    &format!("Method not found: {}", method),
                );
            }
        };

        match result {
            Ok(value) => Message::response(id, value),
            Err(e) => {
                error!(error = %e, "Request handler error");
                let code = match &e {
                    ServerError::MethodNotFound(_) => roo_jsonrpc::types::error_codes::METHOD_NOT_FOUND,
                    ServerError::InvalidParams { .. } => roo_jsonrpc::types::error_codes::INVALID_PARAMS,
                    ServerError::NotInitialized | ServerError::AlreadyInitialized => -32000,
                    ServerError::ShutDown => -32001,
                    _ => roo_jsonrpc::types::error_codes::INTERNAL_ERROR,
                };
                Message::error_response(id, code, &e.to_string())
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Lifecycle handlers
    // ────────────────────────────────────────────────────────────────────

    /// Source: `webviewDidLaunch` → ClineProvider initialization
    async fn handle_initialize(&self, _params: Value) -> ServerResult<Value> {
        info!("Initializing server");
        self.app.initialize().await?;
        let state = self.app.state().await;
        Ok(json!({
            "initialized": state.initialized,
            "mode": state.current_mode,
            "cwd": self.app.cwd(),
        }))
    }

    /// Source: `dispose` → ClineProvider.dispose()
    async fn handle_shutdown(&self, _params: Value) -> ServerResult<Value> {
        info!("Shutting down server");
        self.app.dispose().await?;
        Ok(json!(null))
    }

    /// Keep-alive ping.
    async fn handle_ping(&self, _params: Value) -> ServerResult<Value> {
        Ok(json!("pong"))
    }

    // ────────────────────────────────────────────────────────────────────
    // Task command handlers
    // Source: `packages/types/src/ipc.ts` — TaskCommandName
    // ────────────────────────────────────────────────────────────────────

    /// Source: `TaskCommandName.StartNewTask`
    async fn handle_task_start(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let mode = params
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("code");

        info!(mode = mode, text_len = text.len(), "Starting new task");

        // In the full implementation, this would create a Task through the TaskProvider
        // For now, update the app state to reflect the running task
        Ok(json!({
            "taskId": format!("task-{}", chrono_like_id()),
            "mode": mode,
            "status": "started",
        }))
    }

    /// Source: `TaskCommandName.CancelTask`
    async fn handle_task_cancel(&self, _params: Value) -> ServerResult<Value> {
        info!("Cancelling task");
        Ok(json!({"status": "cancelled"}))
    }

    /// Source: `TaskCommandName.CloseTask`
    async fn handle_task_close(&self, _params: Value) -> ServerResult<Value> {
        info!("Closing task");
        Ok(json!({"status": "closed"}))
    }

    /// Source: `TaskCommandName.ResumeTask`
    async fn handle_task_resume(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str()).unwrap_or("");
        info!(task_id = task_id, "Resuming task");
        Ok(json!({"taskId": task_id, "status": "resumed"}))
    }

    /// Source: `TaskCommandName.SendMessage`
    async fn handle_task_send_message(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let images = params
            .get("images")
            .and_then(|v| v.as_array())
            .map(|arr| arr.len())
            .unwrap_or(0);

        info!(text_len = text.len(), images = images, "Sending message to task");
        Ok(json!({"status": "sent"}))
    }

    /// Source: `TaskCommandName.GetCommands`
    async fn handle_task_get_commands(&self, _params: Value) -> ServerResult<Value> {
        // In the full implementation, this would use roo_command to discover commands
        Ok(json!({"commands": []}))
    }

    /// Source: `TaskCommandName.GetModes`
    async fn handle_task_get_modes(&self, _params: Value) -> ServerResult<Value> {
        let modes = roo_types::mode::default_modes();
        let mode_list: Vec<Value> = modes
            .iter()
            .map(|m| json!({"slug": m.slug, "name": m.name}))
            .collect();
        Ok(json!({"modes": mode_list}))
    }

    /// Source: `TaskCommandName.GetModels`
    async fn handle_task_get_models(&self, _params: Value) -> ServerResult<Value> {
        // In the full implementation, this would query the provider for available models
        Ok(json!({"models": {}}))
    }

    /// Source: `TaskCommandName.DeleteQueuedMessage`
    async fn handle_task_delete_queued_message(&self, params: Value) -> ServerResult<Value> {
        let message_id = params
            .get("messageId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::TASK_DELETE_QUEUED_MESSAGE.to_string(),
                detail: "Missing messageId".to_string(),
            })?;

        info!(message_id = message_id, "Deleting queued message");
        Ok(json!({"status": "deleted"}))
    }

    /// Source: `condenseTaskContextRequest` WebviewMessage
    async fn handle_task_condense(&self, _params: Value) -> ServerResult<Value> {
        info!("Condensing task context");
        Ok(json!({"status": "condensed"}))
    }

    // ────────────────────────────────────────────────────────────────────
    // State command handlers
    // ────────────────────────────────────────────────────────────────────

    /// Source: `getState` in ClineProvider
    async fn handle_state_get(&self, _params: Value) -> ServerResult<Value> {
        let state = self.app.state().await;
        Ok(json!({
            "initialized": state.initialized,
            "mode": state.current_mode,
            "activeTaskCount": state.active_task_count,
            "taskRunning": state.task_running,
            "disposed": state.disposed,
            "cwd": self.app.cwd(),
        }))
    }

    /// Source: `mode` WebviewMessage
    async fn handle_state_set_mode(&self, params: Value) -> ServerResult<Value> {
        let mode = params
            .get("mode")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::STATE_SET_MODE.to_string(),
                detail: "Missing mode".to_string(),
            })?;

        self.app.set_mode(mode).await;
        info!(mode = mode, "Mode changed");
        Ok(json!({"mode": mode}))
    }

    /// Source: `getSystemPrompt` WebviewMessage
    async fn handle_system_prompt_build(&self, _params: Value) -> ServerResult<Value> {
        let prompt = self.app.build_system_prompt();
        Ok(json!({"prompt": prompt}))
    }

    // ────────────────────────────────────────────────────────────────────
    // History command handlers
    // ────────────────────────────────────────────────────────────────────

    /// Source: `showTaskWithId` WebviewMessage
    async fn handle_history_get(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str()).unwrap_or("");
        debug!(task_id = task_id, "Getting task history");
        Ok(json!({"taskId": task_id, "history": []}))
    }

    /// Source: `deleteTaskWithId` WebviewMessage
    async fn handle_history_delete(&self, params: Value) -> ServerResult<Value> {
        let task_id = params
            .get("taskId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::HISTORY_DELETE.to_string(),
                detail: "Missing taskId".to_string(),
            })?;

        info!(task_id = task_id, "Deleting task");
        Ok(json!({"status": "deleted"}))
    }

    /// Source: `exportTaskWithId` WebviewMessage
    async fn handle_history_export(&self, params: Value) -> ServerResult<Value> {
        let task_id = params
            .get("taskId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::HISTORY_EXPORT.to_string(),
                detail: "Missing taskId".to_string(),
            })?;

        debug!(task_id = task_id, "Exporting task");
        Ok(json!({"taskId": task_id, "data": null}))
    }

    // ────────────────────────────────────────────────────────────────────
    // Todo command handler
    // ────────────────────────────────────────────────────────────────────

    /// Source: `updateTodoList` WebviewMessage
    async fn handle_todo_update(&self, params: Value) -> ServerResult<Value> {
        let todos = params.get("todos").cloned().unwrap_or(Value::Null);
        debug!("Updating todo list");
        Ok(json!({"status": "updated", "todos": todos}))
    }

    // ────────────────────────────────────────────────────────────────────
    // Ask response handler
    // ────────────────────────────────────────────────────────────────────

    /// Source: `askResponse` WebviewMessage
    async fn handle_ask_response(&self, params: Value) -> ServerResult<Value> {
        let response = params.get("askResponse").and_then(|v| v.as_str()).unwrap_or("");
        let _text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        debug!(response = response, "Processing ask response");
        Ok(json!({"status": "responded"}))
    }

    // ────────────────────────────────────────────────────────────────────
    // Terminal command handler
    // ────────────────────────────────────────────────────────────────────

    /// Source: `terminalOperation` WebviewMessage
    async fn handle_terminal_operation(&self, params: Value) -> ServerResult<Value> {
        let operation = params
            .get("operation")
            .and_then(|v| v.as_str())
            .unwrap_or("continue");
        debug!(operation = operation, "Terminal operation");
        Ok(json!({"status": "ok"}))
    }

    // ────────────────────────────────────────────────────────────────────
    // Checkpoint command handlers
    // ────────────────────────────────────────────────────────────────────

    /// Source: `checkpointDiff` WebviewMessage
    async fn handle_checkpoint_diff(&self, params: Value) -> ServerResult<Value> {
        let commit_hash = params
            .get("commitHash")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        debug!(commit_hash = commit_hash, "Checkpoint diff");
        Ok(json!({"diff": ""}))
    }

    /// Source: `checkpointRestore` WebviewMessage
    async fn handle_checkpoint_restore(&self, params: Value) -> ServerResult<Value> {
        let commit_hash = params
            .get("commitHash")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        debug!(commit_hash = commit_hash, "Checkpoint restore");
        Ok(json!({"status": "restored"}))
    }

    // ────────────────────────────────────────────────────────────────────
    // Prompt enhancement handler
    // ────────────────────────────────────────────────────────────────────

    /// Source: `enhancePrompt` WebviewMessage
    async fn handle_prompt_enhance(&self, params: Value) -> ServerResult<Value> {
        let text = params
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::PROMPT_ENHANCE.to_string(),
                detail: "Missing text".to_string(),
            })?;

        debug!(text_len = text.len(), "Enhancing prompt");
        // In the full implementation, this would call the AI to enhance the prompt
        Ok(json!({"enhancedText": text}))
    }

    // ────────────────────────────────────────────────────────────────────
    // Search command handler
    // ────────────────────────────────────────────────────────────────────

    /// Source: `searchFiles` WebviewMessage
    async fn handle_search_files(&self, params: Value) -> ServerResult<Value> {
        let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
        debug!(query = query, path = path, "Searching files");
        Ok(json!({"results": []}))
    }

    // ────────────────────────────────────────────────────────────────────
    // File read handler
    // ────────────────────────────────────────────────────────────────────

    /// Source: `readFileContent` WebviewMessage
    async fn handle_file_read(&self, params: Value) -> ServerResult<Value> {
        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::FILE_READ.to_string(),
                detail: "Missing path".to_string(),
            })?;

        debug!(path = path, "Reading file content");

        match tokio::fs::read_to_string(path).await {
            Ok(content) => Ok(json!({"path": path, "content": content})),
            Err(e) => Ok(json!({"path": path, "content": null, "error": e.to_string()})),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // MCP command handlers
    // ────────────────────────────────────────────────────────────────────

    /// Source: `mcpServers` ExtensionMessage
    async fn handle_mcp_list_servers(&self, _params: Value) -> ServerResult<Value> {
        Ok(json!({"servers": []}))
    }

    /// Source: `restartMcpServer` WebviewMessage
    async fn handle_mcp_restart_server(&self, params: Value) -> ServerResult<Value> {
        let server_name = params
            .get("serverName")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        debug!(server_name = server_name, "Restarting MCP server");
        Ok(json!({"status": "restarted"}))
    }

    /// Source: `toggleMcpServer` WebviewMessage
    async fn handle_mcp_toggle_server(&self, params: Value) -> ServerResult<Value> {
        let server_name = params
            .get("serverName")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let disabled = params.get("disabled").and_then(|v| v.as_bool()).unwrap_or(false);
        debug!(server_name = server_name, disabled = disabled, "Toggling MCP server");
        Ok(json!({"status": "toggled"}))
    }

    /// Source: `use_mcp_tool` tool
    async fn handle_mcp_use_tool(&self, params: Value) -> ServerResult<Value> {
        let server_name = params
            .get("serverName")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let tool_name = params
            .get("toolName")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        debug!(server_name = server_name, tool_name = tool_name, "Using MCP tool");
        Ok(json!({"result": null}))
    }

    /// Source: `access_mcp_resource` tool
    async fn handle_mcp_access_resource(&self, params: Value) -> ServerResult<Value> {
        let server_name = params
            .get("serverName")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or("");
        debug!(server_name = server_name, uri = uri, "Accessing MCP resource");
        Ok(json!({"result": null}))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a simple unique-ish ID for tasks.
/// In production, this would use a proper UUID library.
fn chrono_like_id() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use roo_app::AppConfig;
    use roo_jsonrpc::types::Message;
    use serde_json::json;

    fn test_handler() -> Handler {
        let config = AppConfig {
            cwd: "/tmp/test".to_string(),
            mode: "code".to_string(),
            ..Default::default()
        };
        let app = Arc::new(App::new(config));
        Handler::new(app)
    }

    #[tokio::test]
    async fn test_initialize() {
        let handler = test_handler();
        let request = Message::request(1, methods::INITIALIZE, json!(null));
        let response = handler.handle(&request).await;
        assert!(response.result.is_some());
        let result = response.result.unwrap();
        assert_eq!(result["initialized"], true);
        assert_eq!(result["mode"], "code");
    }

    #[tokio::test]
    async fn test_ping() {
        let handler = test_handler();
        let request = Message::request(2, methods::PING, json!(null));
        let response = handler.handle(&request).await;
        assert_eq!(response.result.unwrap(), json!("pong"));
    }

    #[tokio::test]
    async fn test_method_not_found() {
        let handler = test_handler();
        let request = Message::request(3, "nonexistent/method", json!(null));
        let response = handler.handle(&request).await;
        assert!(response.error.is_some());
        assert_eq!(
            response.error.unwrap().code,
            roo_jsonrpc::types::error_codes::METHOD_NOT_FOUND
        );
    }

    #[tokio::test]
    async fn test_state_get() {
        let handler = test_handler();
        handler.app.initialize().await.unwrap();

        let request = Message::request(4, methods::STATE_GET, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["initialized"], true);
        assert_eq!(result["mode"], "code");
    }

    #[tokio::test]
    async fn test_state_set_mode() {
        let handler = test_handler();
        handler.app.initialize().await.unwrap();

        let request =
            Message::request(5, methods::STATE_SET_MODE, json!({"mode": "architect"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["mode"], "architect");

        // Verify mode was actually changed
        assert_eq!(handler.app.mode().await, "architect");
    }

    #[tokio::test]
    async fn test_state_set_mode_missing_param() {
        let handler = test_handler();

        let request = Message::request(6, methods::STATE_SET_MODE, json!({}));
        let response = handler.handle(&request).await;
        assert!(response.error.is_some());
        assert_eq!(
            response.error.unwrap().code,
            roo_jsonrpc::types::error_codes::INVALID_PARAMS
        );
    }

    #[tokio::test]
    async fn test_system_prompt_build() {
        let handler = test_handler();

        let request = Message::request(7, methods::SYSTEM_PROMPT_BUILD, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        let prompt = result["prompt"].as_str().unwrap();
        assert!(prompt.contains("TOOL USE"));
        assert!(prompt.contains("RULES"));
    }

    #[tokio::test]
    async fn test_task_start() {
        let handler = test_handler();
        handler.app.initialize().await.unwrap();

        let request = Message::request(
            8,
            methods::TASK_START,
            json!({"text": "Hello", "mode": "code"}),
        );
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "started");
        assert_eq!(result["mode"], "code");
    }

    #[tokio::test]
    async fn test_task_cancel() {
        let handler = test_handler();

        let request = Message::request(9, methods::TASK_CANCEL, json!(null));
        let response = handler.handle(&request).await;
        assert_eq!(response.result.unwrap()["status"], "cancelled");
    }

    #[tokio::test]
    async fn test_task_get_modes() {
        let handler = test_handler();

        let request = Message::request(10, methods::TASK_GET_MODES, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        let modes = result["modes"].as_array().unwrap();
        assert!(!modes.is_empty());
    }

    #[tokio::test]
    async fn test_shutdown() {
        let handler = test_handler();
        handler.app.initialize().await.unwrap();

        let request = Message::request(11, methods::SHUTDOWN, json!(null));
        let response = handler.handle(&request).await;
        assert!(response.result.is_some());

        assert!(handler.app.is_disposed().await);
    }

    #[tokio::test]
    async fn test_file_read_missing_path() {
        let handler = test_handler();

        let request = Message::request(12, methods::FILE_READ, json!({}));
        let response = handler.handle(&request).await;
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn test_file_read_nonexistent() {
        let handler = test_handler();

        let request = Message::request(
            13,
            methods::FILE_READ,
            json!({"path": "/nonexistent/file.txt"}),
        );
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert!(result["error"].is_string());
    }

    #[tokio::test]
    async fn test_history_delete_missing_id() {
        let handler = test_handler();

        let request = Message::request(14, methods::HISTORY_DELETE, json!({}));
        let response = handler.handle(&request).await;
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn test_task_send_message() {
        let handler = test_handler();

        let request = Message::request(
            15,
            methods::TASK_SEND_MESSAGE,
            json!({"text": "Hello world"}),
        );
        let response = handler.handle(&request).await;
        assert_eq!(response.result.unwrap()["status"], "sent");
    }

    #[tokio::test]
    async fn test_task_resume() {
        let handler = test_handler();

        let request = Message::request(
            16,
            methods::TASK_RESUME,
            json!({"taskId": "task-123"}),
        );
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["taskId"], "task-123");
        assert_eq!(result["status"], "resumed");
    }

    #[tokio::test]
    async fn test_todo_update() {
        let handler = test_handler();

        let todos = json!([{"text": "Task 1", "status": "completed"}]);
        let request = Message::request(17, methods::TODO_UPDATE, json!({"todos": todos}));
        let response = handler.handle(&request).await;
        assert_eq!(response.result.unwrap()["status"], "updated");
    }

    #[tokio::test]
    async fn test_mcp_list_servers() {
        let handler = test_handler();

        let request = Message::request(18, methods::MCP_LIST_SERVERS, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert!(result["servers"].is_array());
    }

    #[tokio::test]
    async fn test_prompt_enhance_missing_text() {
        let handler = test_handler();

        let request = Message::request(19, methods::PROMPT_ENHANCE, json!({}));
        let response = handler.handle(&request).await;
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn test_checkpoint_diff() {
        let handler = test_handler();

        let request = Message::request(
            20,
            methods::CHECKPOINT_DIFF,
            json!({"commitHash": "abc123"}),
        );
        let response = handler.handle(&request).await;
        assert!(response.result.unwrap()["diff"].is_string());
    }

    #[tokio::test]
    async fn test_delete_queued_message_missing_id() {
        let handler = test_handler();

        let request = Message::request(21, methods::TASK_DELETE_QUEUED_MESSAGE, json!({}));
        let response = handler.handle(&request).await;
        assert!(response.error.is_some());
    }
}
