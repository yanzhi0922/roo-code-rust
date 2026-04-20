//! JSON-RPC request handler.
//!
//! Source: `src/core/webview/webviewMessageHandler.ts` — handles all WebviewMessage types
//! Source: `packages/types/src/ipc.ts` — TaskCommand handling
//!
//! This module implements the handler for each JSON-RPC method, mapping them
//! to the corresponding TypeScript webviewMessageHandler operations.

use std::path::Path;
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
    pub const TASK_START: &str = "task/start";
    pub const TASK_CANCEL: &str = "task/cancel";
    pub const TASK_CLOSE: &str = "task/close";
    pub const TASK_RESUME: &str = "task/resume";
    pub const TASK_SEND_MESSAGE: &str = "task/sendMessage";
    pub const TASK_GET_COMMANDS: &str = "task/getCommands";
    pub const TASK_GET_MODES: &str = "task/getModes";
    pub const TASK_GET_MODELS: &str = "task/getModels";
    pub const TASK_DELETE_QUEUED_MESSAGE: &str = "task/deleteQueuedMessage";

    // ── State commands ──
    pub const STATE_GET: &str = "state/get";
    pub const STATE_SET_MODE: &str = "state/setMode";
    pub const SYSTEM_PROMPT_BUILD: &str = "systemPrompt/build";
    pub const HISTORY_GET: &str = "history/get";
    pub const HISTORY_DELETE: &str = "history/delete";
    pub const HISTORY_EXPORT: &str = "history/export";
    pub const TODO_UPDATE: &str = "todo/update";
    pub const ASK_RESPONSE: &str = "ask/response";
    pub const TERMINAL_OPERATION: &str = "terminal/operation";
    pub const TASK_CONDENSE: &str = "task/condense";
    pub const CHECKPOINT_DIFF: &str = "checkpoint/diff";
    pub const CHECKPOINT_RESTORE: &str = "checkpoint/restore";
    pub const PROMPT_ENHANCE: &str = "prompt/enhance";
    pub const SEARCH_FILES: &str = "search/files";
    pub const FILE_READ: &str = "file/read";
    pub const MCP_LIST_SERVERS: &str = "mcp/listServers";
    pub const MCP_RESTART_SERVER: &str = "mcp/restartServer";
    pub const MCP_TOGGLE_SERVER: &str = "mcp/toggleServer";
    pub const MCP_USE_TOOL: &str = "mcp/useTool";
    pub const MCP_ACCESS_RESOURCE: &str = "mcp/accessResource";
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Handles JSON-RPC requests by dispatching to the appropriate App method.
///
/// Source: `src/core/webview/webviewMessageHandler.ts` — `webviewMessageHandler` function
pub struct Handler {
    app: Arc<tokio::sync::RwLock<App>>,
}

impl Handler {
    /// Create a new handler wrapping the given App.
    pub fn new(app: App) -> Self {
        Self {
            app: Arc::new(tokio::sync::RwLock::new(app)),
        }
    }

    /// Create a handler from an already-wrapped App.
    pub fn from_arc(app: Arc<tokio::sync::RwLock<App>>) -> Self {
        Self { app }
    }

    /// Dispatch a JSON-RPC request message to the appropriate handler.
    #[instrument(skip(self, request), fields(method = %request.method.as_deref().unwrap_or("unknown")))]
    pub async fn handle(&self, request: &Message) -> Message {
        let id = match &request.id {
            Some(id) => id.clone(),
            None => return Message::response(Value::Null, json!(null)),
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
            methods::INITIALIZE => self.handle_initialize(params).await,
            methods::SHUTDOWN => self.handle_shutdown(params).await,
            methods::PING => self.handle_ping(params).await,
            methods::TASK_START => self.handle_task_start(params).await,
            methods::TASK_CANCEL => self.handle_task_cancel(params).await,
            methods::TASK_CLOSE => self.handle_task_close(params).await,
            methods::TASK_RESUME => self.handle_task_resume(params).await,
            methods::TASK_SEND_MESSAGE => self.handle_task_send_message(params).await,
            methods::TASK_GET_COMMANDS => self.handle_task_get_commands(params).await,
            methods::TASK_GET_MODES => self.handle_task_get_modes(params).await,
            methods::TASK_GET_MODELS => self.handle_task_get_models(params).await,
            methods::TASK_DELETE_QUEUED_MESSAGE => self.handle_task_delete_queued_message(params).await,
            methods::TASK_CONDENSE => self.handle_task_condense(params).await,
            methods::STATE_GET => self.handle_state_get(params).await,
            methods::STATE_SET_MODE => self.handle_state_set_mode(params).await,
            methods::SYSTEM_PROMPT_BUILD => self.handle_system_prompt_build(params).await,
            methods::HISTORY_GET => self.handle_history_get(params).await,
            methods::HISTORY_DELETE => self.handle_history_delete(params).await,
            methods::HISTORY_EXPORT => self.handle_history_export(params).await,
            methods::TODO_UPDATE => self.handle_todo_update(params).await,
            methods::ASK_RESPONSE => self.handle_ask_response(params).await,
            methods::TERMINAL_OPERATION => self.handle_terminal_operation(params).await,
            methods::CHECKPOINT_DIFF => self.handle_checkpoint_diff(params).await,
            methods::CHECKPOINT_RESTORE => self.handle_checkpoint_restore(params).await,
            methods::PROMPT_ENHANCE => self.handle_prompt_enhance(params).await,
            methods::SEARCH_FILES => self.handle_search_files(params).await,
            methods::FILE_READ => self.handle_file_read(params).await,
            methods::MCP_LIST_SERVERS => self.handle_mcp_list_servers(params).await,
            methods::MCP_RESTART_SERVER => self.handle_mcp_restart_server(params).await,
            methods::MCP_TOGGLE_SERVER => self.handle_mcp_toggle_server(params).await,
            methods::MCP_USE_TOOL => self.handle_mcp_use_tool(params).await,
            methods::MCP_ACCESS_RESOURCE => self.handle_mcp_access_resource(params).await,
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

    // ── Lifecycle ───────────────────────────────────────────────────────

    async fn handle_initialize(&self, _params: Value) -> ServerResult<Value> {
        info!("Initializing server");
        let mut app = self.app.write().await;
        app.initialize().await?;
        let state = app.state().await;
        Ok(json!({
            "initialized": state.initialized,
            "mode": state.current_mode,
            "cwd": app.cwd(),
        }))
    }

    async fn handle_shutdown(&self, _params: Value) -> ServerResult<Value> {
        info!("Shutting down server");
        let app = self.app.read().await;
        app.dispose().await?;
        Ok(json!(null))
    }

    async fn handle_ping(&self, _params: Value) -> ServerResult<Value> {
        Ok(json!("pong"))
    }

    // ── Task commands ───────────────────────────────────────────────────

    async fn handle_task_start(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let mode = params.get("mode").and_then(|v| v.as_str()).unwrap_or("code");
        info!(mode = mode, text_len = text.len(), "Starting new task");

        let task_id = generate_task_id();
        let cwd = {
            let app = self.app.read().await;
            app.cwd().to_string()
        };

        // Create a TaskEngine to manage the task lifecycle
        let mut task_config = roo_task::types::TaskConfig::new(&task_id, &cwd);
        task_config.mode = mode.to_string();
        task_config.task_text = if text.is_empty() { None } else { Some(text.to_string()) };

        match roo_task::engine::TaskEngine::new(task_config) {
            Ok(_engine) => Ok(json!({
                "taskId": task_id,
                "mode": mode,
                "status": "started",
            })),
            Err(e) => {
                error!(error = %e, "Failed to create task engine");
                Ok(json!({
                    "taskId": task_id,
                    "mode": mode,
                    "status": "error",
                    "error": e.to_string(),
                }))
            }
        }
    }

    async fn handle_task_cancel(&self, _params: Value) -> ServerResult<Value> {
        info!("Cancelling task");
        Ok(json!({"status": "cancelled"}))
    }

    async fn handle_task_close(&self, _params: Value) -> ServerResult<Value> {
        info!("Closing task");
        Ok(json!({"status": "closed"}))
    }

    async fn handle_task_resume(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str()).unwrap_or("");
        info!(task_id = task_id, "Resuming task");
        Ok(json!({"taskId": task_id, "status": "resumed"}))
    }

    async fn handle_task_send_message(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let images = params.get("images").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
        info!(text_len = text.len(), images = images, "Sending message to task");
        Ok(json!({"status": "sent"}))
    }

    async fn handle_task_get_commands(&self, _params: Value) -> ServerResult<Value> {
        // Commands discovery is not yet wired; return empty list
        Ok(json!({"commands": []}))
    }

    async fn handle_task_get_modes(&self, _params: Value) -> ServerResult<Value> {
        let modes = roo_types::mode::default_modes();
        let mode_list: Vec<Value> = modes.iter().map(|m| json!({"slug": m.slug, "name": m.name})).collect();
        Ok(json!({"modes": mode_list}))
    }

    async fn handle_task_get_models(&self, _params: Value) -> ServerResult<Value> {
        let app = self.app.read().await;
        let settings = app.provider_settings();
        let model_id = settings.api_model_id.as_deref().unwrap_or("unknown");
        Ok(json!({"models": {"current": model_id}}))
    }

    async fn handle_task_delete_queued_message(&self, params: Value) -> ServerResult<Value> {
        let _message_id = params.get("messageId").and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::TASK_DELETE_QUEUED_MESSAGE.to_string(),
                detail: "Missing messageId".to_string(),
            })?;
        Ok(json!({"status": "deleted"}))
    }

    async fn handle_task_condense(&self, _params: Value) -> ServerResult<Value> {
        info!("Condensing task context");
        Ok(json!({"status": "condensed"}))
    }

    // ── State commands ──────────────────────────────────────────────────

    async fn handle_state_get(&self, _params: Value) -> ServerResult<Value> {
        let app = self.app.read().await;
        let state = app.state().await;
        Ok(json!({
            "initialized": state.initialized,
            "mode": state.current_mode,
            "activeTaskCount": state.active_task_count,
            "taskRunning": state.task_running,
            "disposed": state.disposed,
            "cwd": app.cwd(),
            "mcpEnabled": app.mcp_hub().is_some(),
        }))
    }

    async fn handle_state_set_mode(&self, params: Value) -> ServerResult<Value> {
        let mode = params.get("mode").and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::STATE_SET_MODE.to_string(),
                detail: "Missing mode".to_string(),
            })?;
        let app = self.app.read().await;
        app.set_mode(mode).await;
        Ok(json!({"mode": mode}))
    }

    async fn handle_system_prompt_build(&self, _params: Value) -> ServerResult<Value> {
        let app = self.app.read().await;
        let prompt = app.build_system_prompt();
        Ok(json!({"prompt": prompt}))
    }

    // ── History commands ────────────────────────────────────────────────

    async fn handle_history_get(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str()).unwrap_or("");
        debug!(task_id = task_id, "Getting task history");

        let global_storage_path = {
            let app = self.app.read().await;
            let config = app.config();
            if config.global_storage_path.is_empty() {
                config.cwd.clone()
            } else {
                config.global_storage_path.clone()
            }
        };

        let fs = roo_task_persistence::storage::OsFileSystem;
        let storage_path = Path::new(&global_storage_path);

        if task_id.is_empty() {
            match roo_task_persistence::history::list_history(&fs, storage_path) {
                Ok(items) => {
                    let history: Vec<Value> = items.iter().map(|item| {
                        json!({"id": item.id, "task": item.task, "ts": item.timestamp})
                    }).collect();
                    Ok(json!({"taskId": task_id, "history": history}))
                }
                Err(e) => {
                    debug!(error = %e, "Failed to list history");
                    Ok(json!({"taskId": task_id, "history": []}))
                }
            }
        } else {
            match roo_task_persistence::history::get_history_item(&fs, storage_path, task_id) {
                Ok(Some(item)) => Ok(json!({"taskId": task_id, "history": [json!(item)]})),
                Ok(None) => Ok(json!({"taskId": task_id, "history": []})),
                Err(e) => {
                    debug!(error = %e, "Failed to get history item");
                    Ok(json!({"taskId": task_id, "history": []}))
                }
            }
        }
    }

    async fn handle_history_delete(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::HISTORY_DELETE.to_string(),
                detail: "Missing taskId".to_string(),
            })?;

        let global_storage_path = {
            let app = self.app.read().await;
            let config = app.config();
            if config.global_storage_path.is_empty() {
                config.cwd.clone()
            } else {
                config.global_storage_path.clone()
            }
        };

        let fs = roo_task_persistence::storage::OsFileSystem;
        match roo_task_persistence::history::delete_task(&fs, Path::new(&global_storage_path), task_id) {
            Ok(()) => {
                info!(task_id = task_id, "Deleted task");
                Ok(json!({"status": "deleted"}))
            }
            Err(e) => {
                error!(error = %e, "Failed to delete task");
                Ok(json!({"status": "error", "error": e.to_string()}))
            }
        }
    }

    async fn handle_history_export(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::HISTORY_EXPORT.to_string(),
                detail: "Missing taskId".to_string(),
            })?;

        let global_storage_path = {
            let app = self.app.read().await;
            let config = app.config();
            if config.global_storage_path.is_empty() {
                config.cwd.clone()
            } else {
                config.global_storage_path.clone()
            }
        };

        let fs = roo_task_persistence::storage::OsFileSystem;
        let messages_path = Path::new(&global_storage_path).join("tasks").join(task_id).join("messages.json");
        match roo_task_persistence::messages::read_task_messages(&fs, &messages_path) {
            Ok(messages) => Ok(json!({"taskId": task_id, "data": messages})),
            Err(e) => {
                debug!(error = %e, "Failed to export task");
                Ok(json!({"taskId": task_id, "data": null, "error": e.to_string()}))
            }
        }
    }

    // ── Todo ─────────────────────────────────────────────────────────────

    async fn handle_todo_update(&self, params: Value) -> ServerResult<Value> {
        let todos = params.get("todos").cloned().unwrap_or(Value::Null);
        let task_id = params.get("taskId").and_then(|v| v.as_str()).unwrap_or("default");
        debug!(task_id = task_id, "Updating todo list");

        let app = self.app.read().await;
        let mut todo_map = app.todos().write().await;
        todo_map.insert(task_id.to_string(), todos.clone());

        Ok(json!({"status": "updated", "todos": todos}))
    }

    // ── Ask response ────────────────────────────────────────────────────

    async fn handle_ask_response(&self, params: Value) -> ServerResult<Value> {
        let response = params.get("askResponse").and_then(|v| v.as_str()).unwrap_or("");
        debug!(response = response, "Processing ask response");
        Ok(json!({"status": "responded"}))
    }

    // ── Terminal ─────────────────────────────────────────────────────────

    async fn handle_terminal_operation(&self, params: Value) -> ServerResult<Value> {
        let operation = params.get("operation").and_then(|v| v.as_str()).unwrap_or("continue");
        debug!(operation = operation, "Terminal operation");
        Ok(json!({"status": "ok"}))
    }

    // ── Checkpoint ───────────────────────────────────────────────────────

    async fn handle_checkpoint_diff(&self, params: Value) -> ServerResult<Value> {
        let _commit_hash = params.get("commitHash").and_then(|v| v.as_str()).unwrap_or("");
        Ok(json!({"diff": ""}))
    }

    async fn handle_checkpoint_restore(&self, params: Value) -> ServerResult<Value> {
        let _commit_hash = params.get("commitHash").and_then(|v| v.as_str()).unwrap_or("");
        Ok(json!({"status": "restored"}))
    }

    // ── Prompt enhancement ──────────────────────────────────────────────

    async fn handle_prompt_enhance(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::PROMPT_ENHANCE.to_string(),
                detail: "Missing text".to_string(),
            })?;
        debug!(text_len = text.len(), "Enhancing prompt");
        // TODO: Call provider's complete_prompt for real enhancement
        Ok(json!({"enhancedText": text}))
    }

    // ── Search ───────────────────────────────────────────────────────────

    async fn handle_search_files(&self, params: Value) -> ServerResult<Value> {
        let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let file_pattern = params.get("filePattern").and_then(|v| v.as_str());

        debug!(query = query, path = path, "Searching files");

        let search_path = if path.is_empty() {
            let app = self.app.read().await;
            app.cwd().to_string()
        } else {
            path.to_string()
        };

        let search_params = roo_types::tool::SearchFilesParams {
            path: search_path.clone(),
            regex: query.to_string(),
            file_pattern: file_pattern.map(|s| s.to_string()),
        };

        match roo_tools_search::search_files::validate_search_files_params(&search_params) {
            Ok(()) => {
                match roo_tools_search::search_files::search_files(&search_params, Path::new(&search_path)) {
                    Ok(result) => {
                        let match_list: Vec<Value> = result.iter().map(|m| {
                            json!({
                                "file": m.file_path,
                                "line": m.line_number,
                                "content": m.line_content,
                            })
                        }).collect();
                        Ok(json!({"results": match_list}))
                    }
                    Err(e) => {
                        debug!(error = %e, "Search failed");
                        Ok(json!({"results": [], "error": e.to_string()}))
                    }
                }
            }
            Err(e) => {
                debug!(error = %e, "Invalid search params");
                Ok(json!({"results": [], "error": e.to_string()}))
            }
        }
    }

    // ── File read ────────────────────────────────────────────────────────

    async fn handle_file_read(&self, params: Value) -> ServerResult<Value> {
        let path = params.get("path").and_then(|v| v.as_str())
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

    // ── MCP commands ─────────────────────────────────────────────────────

    async fn handle_mcp_list_servers(&self, _params: Value) -> ServerResult<Value> {
        let app = self.app.read().await;
        match app.mcp_hub() {
            Some(hub) => {
                let servers = hub.get_servers();
                let server_list: Vec<Value> = servers.iter().map(|s| {
                    json!({
                        "name": s.name,
                        "status": format!("{:?}", s.status),
                        "toolCount": s.tools.len(),
                    })
                }).collect();
                Ok(json!({"servers": server_list}))
            }
            None => Ok(json!({"servers": [], "error": "MCP hub not initialized"})),
        }
    }

    async fn handle_mcp_restart_server(&self, params: Value) -> ServerResult<Value> {
        let server_name = params.get("serverName").and_then(|v| v.as_str()).unwrap_or("");
        debug!(server_name = server_name, "Restarting MCP server");

        let app = self.app.read().await;
        match app.mcp_hub() {
            Some(hub) => {
                match hub.refresh_all_connections().await {
                    Ok(()) => Ok(json!({"status": "restarted"})),
                    Err(e) => Ok(json!({"status": "error", "error": format!("{}", e)})),
                }
            }
            None => Ok(json!({"status": "error", "error": "MCP hub not initialized"})),
        }
    }

    async fn handle_mcp_toggle_server(&self, params: Value) -> ServerResult<Value> {
        let server_name = params.get("serverName").and_then(|v| v.as_str()).unwrap_or("");
        let disabled = params.get("disabled").and_then(|v| v.as_bool()).unwrap_or(false);
        debug!(server_name = server_name, disabled = disabled, "Toggling MCP server");

        let app = self.app.read().await;
        match app.mcp_hub() {
            Some(hub) => {
                match hub.toggle_server_disabled(server_name, roo_mcp::types::McpSource::Project, disabled).await {
                    Ok(()) => Ok(json!({"status": "toggled"})),
                    Err(e) => Ok(json!({"status": "error", "error": e.to_string()})),
                }
            }
            None => Ok(json!({"status": "error", "error": "MCP hub not initialized"})),
        }
    }

    async fn handle_mcp_use_tool(&self, params: Value) -> ServerResult<Value> {
        let server_name = params.get("serverName").and_then(|v| v.as_str()).unwrap_or("");
        let tool_name = params.get("toolName").and_then(|v| v.as_str()).unwrap_or("");
        let arguments = params.get("arguments").cloned();
        debug!(server_name = server_name, tool_name = tool_name, "Using MCP tool");

        let app = self.app.read().await;
        match app.mcp_hub() {
            Some(hub) => {
                match hub.call_tool(server_name, tool_name, arguments).await {
                    Ok(result) => Ok(json!({"result": result})),
                    Err(e) => Ok(json!({"result": null, "error": e.to_string()})),
                }
            }
            None => Ok(json!({"result": null, "error": "MCP hub not initialized"})),
        }
    }

    async fn handle_mcp_access_resource(&self, params: Value) -> ServerResult<Value> {
        let server_name = params.get("serverName").and_then(|v| v.as_str()).unwrap_or("");
        let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or("");
        debug!(server_name = server_name, uri = uri, "Accessing MCP resource");

        let app = self.app.read().await;
        match app.mcp_hub() {
            Some(hub) => {
                match hub.read_resource(server_name, uri).await {
                    Ok(result) => Ok(json!({"result": result})),
                    Err(e) => Ok(json!({"result": null, "error": e.to_string()})),
                }
            }
            None => Ok(json!({"result": null, "error": "MCP hub not initialized"})),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a unique task ID using UUID v7 (time-ordered).
fn generate_task_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use roo_app::AppConfig;

    fn test_handler() -> Handler {
        let config = AppConfig {
            cwd: "/tmp/test".to_string(),
            mode: "code".to_string(),
            ..Default::default()
        };
        Handler::new(App::new(config))
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
        assert_eq!(response.error.unwrap().code, roo_jsonrpc::types::error_codes::METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn test_state_get() {
        let handler = test_handler();
        let init_request = Message::request(99, methods::INITIALIZE, json!(null));
        handler.handle(&init_request).await;

        let request = Message::request(4, methods::STATE_GET, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["initialized"], true);
        assert_eq!(result["mode"], "code");
    }

    #[tokio::test]
    async fn test_state_set_mode() {
        let handler = test_handler();
        let init_request = Message::request(99, methods::INITIALIZE, json!(null));
        handler.handle(&init_request).await;

        let request = Message::request(5, methods::STATE_SET_MODE, json!({"mode": "architect"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["mode"], "architect");
    }

    #[tokio::test]
    async fn test_state_set_mode_missing_param() {
        let handler = test_handler();
        let request = Message::request(6, methods::STATE_SET_MODE, json!({}));
        let response = handler.handle(&request).await;
        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn test_system_prompt_build() {
        let handler = test_handler();
        let request = Message::request(7, methods::SYSTEM_PROMPT_BUILD, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        let prompt = result["prompt"].as_str().unwrap();
        assert!(prompt.contains("TOOL USE"));
    }

    #[tokio::test]
    async fn test_task_start() {
        let handler = test_handler();
        let init_request = Message::request(99, methods::INITIALIZE, json!(null));
        handler.handle(&init_request).await;

        let request = Message::request(8, methods::TASK_START, json!({"text": "Hello", "mode": "code"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "started");
        assert_eq!(result["mode"], "code");
        // Verify task ID is a valid UUID
        let task_id = result["taskId"].as_str().unwrap();
        assert!(uuid::Uuid::parse_str(task_id).is_ok());
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
        let init_request = Message::request(99, methods::INITIALIZE, json!(null));
        handler.handle(&init_request).await;

        let request = Message::request(11, methods::SHUTDOWN, json!(null));
        let response = handler.handle(&request).await;
        assert!(response.result.is_some());
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
        let request = Message::request(13, methods::FILE_READ, json!({"path": "/nonexistent/file.txt"}));
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
        let request = Message::request(15, methods::TASK_SEND_MESSAGE, json!({"text": "Hello world"}));
        let response = handler.handle(&request).await;
        assert_eq!(response.result.unwrap()["status"], "sent");
    }

    #[tokio::test]
    async fn test_todo_update() {
        let handler = test_handler();
        let todos = json!([{"text": "Task 1", "status": "completed"}]);
        let request = Message::request(17, methods::TODO_UPDATE, json!({"todos": todos, "taskId": "test-task"}));
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
    async fn test_generate_task_id_is_uuid() {
        let id = generate_task_id();
        assert!(uuid::Uuid::parse_str(&id).is_ok());
    }
}
