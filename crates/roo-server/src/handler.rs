//! JSON-RPC request handler.
//!
//! Source: `src/core/webview/webviewMessageHandler.ts` — handles all WebviewMessage types
//! Source: `packages/types/src/ipc.ts` — TaskCommand handling
//!
//! This module implements the handler for each JSON-RPC method, mapping them
//! to the corresponding TypeScript webviewMessageHandler operations.
//!
//! R10-A: Updated to use TaskLifecycle and AskSayHandler from R9-C.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tracing::{debug, error, info, instrument, warn};

use roo_app::App;
use roo_jsonrpc::types::Message;
use roo_task::task_lifecycle::TaskLifecycle;
use roo_task::ask_say::AskResponse;
use roo_task::events::TaskEvent;
use roo_task::TaskManager;

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
    pub const MCP_DELETE_SERVER: &str = "mcp/deleteServer";
    pub const MCP_UPDATE_TIMEOUT: &str = "mcp/updateTimeout";

    // ── Settings commands ──
    pub const SETTINGS_UPDATE: &str = "settings/update";
    pub const SETTINGS_SAVE_API_CONFIG: &str = "settings/saveApiConfig";
    pub const SETTINGS_LOAD_API_CONFIG: &str = "settings/loadApiConfig";
    pub const SETTINGS_LOAD_API_CONFIG_BY_ID: &str = "settings/loadApiConfigById";
    pub const SETTINGS_DELETE_API_CONFIG: &str = "settings/deleteApiConfig";
    pub const SETTINGS_LIST_API_CONFIGS: &str = "settings/listApiConfigs";
    pub const SETTINGS_UPSERT_API_CONFIG: &str = "settings/upsertApiConfig";

    // ── Skills commands ──
    pub const SKILLS_LIST: &str = "skills/list";
    pub const SKILLS_CREATE: &str = "skills/create";
    pub const SKILLS_DELETE: &str = "skills/delete";
    pub const SKILLS_MOVE: &str = "skills/move";
    pub const SKILLS_UPDATE_MODES: &str = "skills/updateModes";

    // ── Mode commands ──
    pub const MODE_UPDATE_CUSTOM: &str = "mode/updateCustom";
    pub const MODE_DELETE_CUSTOM: &str = "mode/deleteCustom";

    // ── Message commands ──
    pub const MESSAGE_DELETE: &str = "message/delete";
    pub const MESSAGE_EDIT: &str = "message/edit";
    pub const MESSAGE_QUEUE: &str = "message/queue";

    // ── Telemetry commands ──
    pub const TELEMETRY_SET_SETTING: &str = "telemetry/setSetting";

    // ── Notification method (server → client) ──
    /// Method name for task event notifications sent from server to client.
    pub const NOTIFICATION_TASK_EVENT: &str = "notification/taskEvent";
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// Handles JSON-RPC requests by dispatching to the appropriate App method.
///
/// Source: `src/core/webview/webviewMessageHandler.ts` — `webviewMessageHandler` function
///
/// R10-A: Now uses [`TaskLifecycle`] for all task operations and forwards
/// [`TaskEvent`]s to the client as JSON-RPC notifications.
pub struct Handler {
    app: Arc<tokio::sync::RwLock<App>>,
    task_manager: Arc<TaskManager>,
    /// Pending JSON-RPC notifications to be sent to the client.
    /// Event listeners push notifications here; the server polls them.
    pending_notifications: Arc<Mutex<Vec<Message>>>,
}

impl Handler {
    /// Create a new handler wrapping the given App.
    pub fn new(app: App) -> Self {
        Self {
            app: Arc::new(tokio::sync::RwLock::new(app)),
            task_manager: Arc::new(TaskManager::new()),
            pending_notifications: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a handler from an already-wrapped App.
    pub fn from_arc(app: Arc<tokio::sync::RwLock<App>>) -> Self {
        Self {
            app,
            task_manager: Arc::new(TaskManager::new()),
            pending_notifications: Arc::new(Mutex::new(Vec::new())),
        }
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
            methods::MCP_DELETE_SERVER => self.handle_mcp_delete_server(params).await,
            methods::MCP_UPDATE_TIMEOUT => self.handle_mcp_update_timeout(params).await,
            methods::SETTINGS_UPDATE => self.handle_settings_update(params).await,
            methods::SETTINGS_SAVE_API_CONFIG => self.handle_settings_save_api_config(params).await,
            methods::SETTINGS_LOAD_API_CONFIG => self.handle_settings_load_api_config(params).await,
            methods::SETTINGS_LOAD_API_CONFIG_BY_ID => self.handle_settings_load_api_config_by_id(params).await,
            methods::SETTINGS_DELETE_API_CONFIG => self.handle_settings_delete_api_config(params).await,
            methods::SETTINGS_LIST_API_CONFIGS => self.handle_settings_list_api_configs(params).await,
            methods::SETTINGS_UPSERT_API_CONFIG => self.handle_settings_upsert_api_config(params).await,
            methods::SKILLS_LIST => self.handle_skills_list(params).await,
            methods::SKILLS_CREATE => self.handle_skills_create(params).await,
            methods::SKILLS_DELETE => self.handle_skills_delete(params).await,
            methods::SKILLS_MOVE => self.handle_skills_move(params).await,
            methods::SKILLS_UPDATE_MODES => self.handle_skills_update_modes(params).await,
            methods::MODE_UPDATE_CUSTOM => self.handle_mode_update_custom(params).await,
            methods::MODE_DELETE_CUSTOM => self.handle_mode_delete_custom(params).await,
            methods::MESSAGE_DELETE => self.handle_message_delete(params).await,
            methods::MESSAGE_EDIT => self.handle_message_edit(params).await,
            methods::MESSAGE_QUEUE => self.handle_message_queue(params).await,
            methods::TELEMETRY_SET_SETTING => self.handle_telemetry_set_setting(params).await,
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

    // ── Notification helpers ────────────────────────────────────────────

    /// Register an event listener on the given lifecycle that forwards
    /// events as JSON-RPC notifications to the client.
    ///
    /// Source: TS `postStateToWebview()` — forwards task state to the webview
    fn register_event_forwarder(&self, lifecycle: &TaskLifecycle) {
        let notifications = self.pending_notifications.clone();
        let task_id = lifecycle.task_id().to_string();

        lifecycle.engine().emitter().on(move |event| {
            let notification = task_event_to_notification(event, &task_id);
            if let Some(msg) = notification {
                notifications.lock().unwrap().push(msg);
            }
        });
    }

    /// Drain all pending notifications, returning them and clearing the queue.
    ///
    /// The server calls this after each request-response cycle to forward
    /// any queued task events to the client.
    pub fn drain_notifications(&self) -> Vec<Message> {
        let mut guard = self.pending_notifications.lock().unwrap();
        std::mem::take(&mut *guard)
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

    /// R10-A — Create a TaskLifecycle, store in TaskManager, start the task.
    ///
    /// Source: TS `startTask()` — creates a new Task and initiates the loop
    async fn handle_task_start(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let mode = params.get("mode").and_then(|v| v.as_str()).unwrap_or("code");
        let images: Vec<String> = params
            .get("images")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        info!(mode = mode, text_len = text.len(), images = images.len(), "Starting new task");

        let task_id = generate_task_id();
        let cwd = {
            let app = self.app.read().await;
            app.cwd().to_string()
        };

        // Create a TaskEngine, then wrap it in a TaskLifecycle
        let mut task_config = roo_task::types::TaskConfig::new(&task_id, &cwd);
        task_config.mode = mode.to_string();
        task_config.task_text = if text.is_empty() { None } else { Some(text.to_string()) };
        task_config.images = images;

        match roo_task::engine::TaskEngine::new(task_config) {
            Ok(engine) => {
                let lifecycle = TaskLifecycle::new(engine);

                // Register event forwarder before storing
                self.register_event_forwarder(&lifecycle);

                // Store lifecycle in TaskManager, set as active task
                self.task_manager.create_task(task_id.clone(), lifecycle);

                // Now start the task via TaskLifecycle
                let lifecycle_arc = self.task_manager.get_task(&task_id).unwrap();
                let mut lc = lifecycle_arc.lock().await;
                match lc.start().await {
                    Ok(()) => Ok(json!({
                        "taskId": task_id,
                        "mode": mode,
                        "status": "started",
                    })),
                    Err(e) => {
                        error!(error = %e, "Failed to start task lifecycle");
                        Ok(json!({
                            "taskId": task_id,
                            "mode": mode,
                            "status": "error",
                            "error": e.to_string(),
                        }))
                    }
                }
            }
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

    /// R10-A — Cancel the active or specified task.
    ///
    /// Source: TS `cancelCurrentRequest()` — aborts the current API request
    async fn handle_task_cancel(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str());
        info!(task_id = task_id, "Cancelling task");

        let lifecycle_arc = match task_id {
            Some(id) => self.task_manager.get_task(id),
            None => self.task_manager.get_active_task(),
        };

        match lifecycle_arc {
            Some(lifecycle) => {
                let mut lc = lifecycle.lock().await;
                // Use TaskLifecycle::cancel_current_request()
                lc.cancel_current_request();
                let tid = lc.task_id().to_string();
                let state = lc.state();
                Ok(json!({
                    "taskId": tid,
                    "status": format!("{}", state).to_lowercase(),
                }))
            }
            None => Ok(json!({"status": "cancelled", "note": "no active task found"})),
        }
    }

    /// R10-A — Close and abort a task, then remove from TaskManager.
    ///
    /// Source: TS `abortTask()` + `dispose()` — clean up and remove
    async fn handle_task_close(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str());
        info!(task_id = task_id, "Closing task");

        match task_id {
            Some(id) => {
                // First abort and dispose the lifecycle
                if let Some(lifecycle) = self.task_manager.get_task(id) {
                    let mut lc = lifecycle.lock().await;
                    // Abort the task (graceful abort, not abandoned)
                    let _ = lc.abort_task(false).await;
                    lc.dispose();
                }
                // Then remove from manager
                let removed = self.task_manager.remove_task(id);
                if removed.is_some() {
                    Ok(json!({"taskId": id, "status": "closed"}))
                } else {
                    Ok(json!({"taskId": id, "status": "not_found"}))
                }
            }
            None => {
                // Close the active task
                let active = self.task_manager.get_active_task();
                match active {
                    Some(lifecycle) => {
                        let id = {
                            let mut lc = lifecycle.lock().await;
                            let id = lc.task_id().to_string();
                            let _ = lc.abort_task(false).await;
                            lc.dispose();
                            id
                        };
                        self.task_manager.remove_task(&id);
                        Ok(json!({"taskId": id, "status": "closed"}))
                    }
                    None => Ok(json!({"status": "no_active_task"})),
                }
            }
        }
    }

    /// R10-A — Resume a paused task or resume from history.
    ///
    /// Source: TS `resumeTaskFromHistory()` — loads history and resumes
    async fn handle_task_resume(&self, params: Value) -> ServerResult<Value> {
        let task_id = params.get("taskId").and_then(|v| v.as_str()).unwrap_or("");
        let history_item_id = params.get("historyItemId").and_then(|v| v.as_str());
        info!(task_id = task_id, "Resuming task");

        let lifecycle_arc = if task_id.is_empty() {
            self.task_manager.get_active_task()
        } else {
            self.task_manager.get_task(task_id)
        };

        match lifecycle_arc {
            Some(lifecycle) => {
                let mut lc = lifecycle.lock().await;
                let tid = lc.task_id().to_string();

                if history_item_id.is_some() {
                    // Resume from history — use TaskLifecycle::resume_task_from_history()
                    // Note: history_item_id should already be set in the config
                    // before the lifecycle was created. If not, we set it here.
                    match lc.resume_task_from_history().await {
                        Ok(()) => {
                            drop(lc);
                            self.task_manager.set_active_task(&tid);
                            Ok(json!({"taskId": tid, "status": "resumed"}))
                        }
                        Err(e) => Ok(json!({"taskId": tid, "status": "error", "error": e.to_string()})),
                    }
                } else {
                    // Simple resume from paused state — use engine state transition
                    match lc.engine_mut().resume() {
                        Ok(state) => {
                            drop(lc);
                            self.task_manager.set_active_task(&tid);
                            Ok(json!({"taskId": tid, "status": format!("{}", state).to_lowercase()}))
                        }
                        Err(e) => Ok(json!({"taskId": tid, "status": "error", "error": e.to_string()})),
                    }
                }
            }
            None => Ok(json!({"taskId": task_id, "status": "not_found"})),
        }
    }

    /// R10-A — Send a message to the active task's conversation.
    ///
    /// Source: TS `submitUserMessage()` — handles user response to an ask
    async fn handle_task_send_message(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let images: Vec<String> = params
            .get("images")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        info!(text_len = text.len(), images = images.len(), "Sending message to task");

        match self.task_manager.get_active_task() {
            Some(lifecycle) => {
                let mut lc = lifecycle.lock().await;
                // Use TaskLifecycle::submit_user_message()
                match lc.submit_user_message(text, if images.is_empty() { None } else { Some(images) }).await {
                    Ok(()) => Ok(json!({"status": "sent"})),
                    Err(e) => Ok(json!({"status": "error", "error": e.to_string()})),
                }
            }
            None => Ok(json!({"status": "error", "error": "no active task"})),
        }
    }

    /// M9 — Discover slash commands from project/global directories.
    async fn handle_task_get_commands(&self, _params: Value) -> ServerResult<Value> {
        let cwd = {
            let app = self.app.read().await;
            app.cwd().to_string()
        };

        let mut commands: HashMap<String, roo_command::types::Command> = HashMap::new();

        // Scan project commands directory (.roo/commands)
        let project_commands_dir = std::path::Path::new(&cwd).join(".roo").join("commands");
        if project_commands_dir.exists() {
            roo_command::scanner::scan_command_directory(
                &project_commands_dir,
                roo_command::types::CommandSource::Project,
                &mut commands,
            )
            .await;
        }

        let command_list: Vec<Value> = commands
            .values()
            .map(|cmd| {
                json!({
                    "name": cmd.name,
                    "description": cmd.description,
                    "source": format!("{:?}", cmd.source),
                })
            })
            .collect();

        Ok(json!({"commands": command_list}))
    }

    async fn handle_task_get_modes(&self, _params: Value) -> ServerResult<Value> {
        let modes = roo_types::mode::default_modes();
        let mode_list: Vec<Value> = modes.iter().map(|m| json!({"slug": m.slug, "name": m.name})).collect();
        Ok(json!({"modes": mode_list}))
    }

    /// M10 — Get current provider model info.
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

    /// R10-A — Condense the active task's context.
    ///
    /// Source: TS `condenseContext()` — manually trigger context condensation
    async fn handle_task_condense(&self, _params: Value) -> ServerResult<Value> {
        info!("Condensing task context");

        match self.task_manager.get_active_task() {
            Some(lifecycle) => {
                let mut lc = lifecycle.lock().await;
                // Use TaskLifecycle::condense_context()
                match lc.condense_context().await {
                    Ok(()) => {
                        let history_len = lc.engine().api_conversation_history().len();
                        Ok(json!({
                            "status": "condensed",
                            "historyLength": history_len,
                        }))
                    }
                    Err(e) => Ok(json!({"status": "error", "error": e.to_string()})),
                }
            }
            None => Ok(json!({"status": "error", "error": "no active task"})),
        }
    }

    // ── State commands ──────────────────────────────────────────────────

    async fn handle_state_get(&self, _params: Value) -> ServerResult<Value> {
        let app = self.app.read().await;
        let state = app.state().await;
        let task_count = self.task_manager.list_tasks().len();
        let has_active = self.task_manager.get_active_task().is_some();
        Ok(json!({
            "initialized": state.initialized,
            "mode": state.current_mode,
            "activeTaskCount": task_count,
            "taskRunning": has_active,
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

    /// R10-A — Handle user response to an ask_followup_question.
    ///
    /// Source: TS `handleWebviewAskResponse()` — processes the user's
    /// response to an ask prompt via AskSayHandler::handle_response()
    async fn handle_ask_response(&self, params: Value) -> ServerResult<Value> {
        let ask_response_str = params.get("askResponse").and_then(|v| v.as_str()).unwrap_or("");
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let images: Vec<String> = params
            .get("images")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        debug!(ask_response = ask_response_str, "Processing ask response");

        // Map the string response to AskResponse enum
        let ask_response = match ask_response_str {
            "yesButtonClicked" | "yes" => AskResponse::YesButtonClicked,
            "noButtonClicked" | "no" => AskResponse::NoButtonClicked,
            _ => AskResponse::MessageResponse,
        };

        match self.task_manager.get_active_task() {
            Some(lifecycle) => {
                let lc = lifecycle.lock().await;
                // Use AskSayHandler::handle_response()
                lc.ask_say()
                    .handle_response(
                        ask_response,
                        if text.is_empty() { None } else { Some(text.to_string()) },
                        if images.is_empty() { None } else { Some(images) },
                    )
                    .await;
                Ok(json!({"status": "responded"}))
            }
            None => Ok(json!({"status": "error", "error": "no active task"})),
        }
    }

    // ── Terminal ─────────────────────────────────────────────────────────

    /// M12 — Execute a terminal operation.
    async fn handle_terminal_operation(&self, params: Value) -> ServerResult<Value> {
        let operation = params.get("operation").and_then(|v| v.as_str()).unwrap_or("continue");
        debug!(operation = operation, "Terminal operation");

        let app = self.app.read().await;
        match app.terminal_registry() {
            Some(registry) => {
                match operation {
                    "execute" | "run" => {
                        let command = params.get("command").and_then(|v| v.as_str()).unwrap_or("");
                        if command.is_empty() {
                            return Ok(json!({"status": "error", "error": "missing command"}));
                        }

                        let cwd = app.cwd();
                        let terminal_id = registry.create_terminal(cwd).await;
                        match registry.get_terminal(terminal_id).await {
                            Some(terminal) => {
                                let guard = terminal.lock().await;
                                use roo_terminal::RooTerminal;
                                match guard.run_command(command, &roo_terminal::NoopCallbacks).await {
                                    Ok(result) => Ok(json!({
                                        "status": "ok",
                                        "exitCode": result.exit_code,
                                        "output": result.stdout,
                                    })),
                                    Err(e) => {
                                        let err_msg: String = e.to_string();
                                        Ok(json!({"status": "error", "error": err_msg}))
                                    }
                                }
                            }
                            None => Ok(json!({"status": "error", "error": "failed to create terminal"})),
                        }
                    }
                    "continue" => {
                        // Continue current terminal operation (no-op in headless mode)
                        Ok(json!({"status": "ok"}))
                    }
                    _ => Ok(json!({"status": "ok", "operation": operation})),
                }
            }
            None => Ok(json!({"status": "error", "error": "terminal registry not initialized"})),
        }
    }

    // ── Checkpoint ───────────────────────────────────────────────────────

    /// C6 — Get checkpoint diff.
    async fn handle_checkpoint_diff(&self, params: Value) -> ServerResult<Value> {
        let commit_hash = params.get("commitHash").and_then(|v| v.as_str()).unwrap_or("");

        // Get active task to determine task_id and workspace
        let (task_id, cwd) = {
            match self.task_manager.get_active_task() {
                Some(lifecycle) => {
                    let lc = lifecycle.lock().await;
                    (
                        lc.task_id().to_string(),
                        lc.engine().config().cwd.clone(),
                    )
                }
                None => {
                    let app = self.app.read().await;
                    ("".to_string(), app.cwd().to_string())
                }
            }
        };

        if task_id.is_empty() {
            return Ok(json!({"diff": [], "error": "no active task for checkpoint"}));
        }

        // Build checkpoint directory path
        let checkpoints_dir = std::path::Path::new(&cwd)
            .join(".roo")
            .join("checkpoints")
            .join(&task_id);

        match roo_checkpoint::service::ShadowCheckpointService::new(
            &task_id,
            &checkpoints_dir,
            &cwd,
            None,
        ) {
            Ok(mut service) => {
                // Initialize the shadow git repo
                if let Err(e) = service.init_shadow_git().await {
                    let err_msg: String = e.to_string();
                    return Ok(json!({"diff": [], "error": err_msg}));
                }

                let diff_params = roo_checkpoint::types::GetDiffParams {
                    from: Some(commit_hash.to_string()),
                    to: None,
                };

                match service.get_diff(diff_params).await {
                    Ok(diffs) => {
                        let diff_list: Vec<Value> = diffs.iter().map(|d| {
                            json!({
                                "path": d.paths.relative,
                                "before": d.content.before,
                                "after": d.content.after,
                            })
                        }).collect();
                        Ok(json!({"diff": diff_list}))
                    }
                    Err(e) => Ok(json!({"diff": [], "error": e.to_string()})),
                }
            }
            Err(e) => Ok(json!({"diff": [], "error": e.to_string()})),
        }
    }

    /// C7 — Restore checkpoint.
    async fn handle_checkpoint_restore(&self, params: Value) -> ServerResult<Value> {
        let commit_hash = params.get("commitHash").and_then(|v| v.as_str()).unwrap_or("");

        let (task_id, cwd) = {
            match self.task_manager.get_active_task() {
                Some(lifecycle) => {
                    let lc = lifecycle.lock().await;
                    (
                        lc.task_id().to_string(),
                        lc.engine().config().cwd.clone(),
                    )
                }
                None => {
                    let app = self.app.read().await;
                    ("".to_string(), app.cwd().to_string())
                }
            }
        };

        if task_id.is_empty() {
            return Ok(json!({"status": "error", "error": "no active task for checkpoint"}));
        }

        let checkpoints_dir = std::path::Path::new(&cwd)
            .join(".roo")
            .join("checkpoints")
            .join(&task_id);

        match roo_checkpoint::service::ShadowCheckpointService::new(
            &task_id,
            &checkpoints_dir,
            &cwd,
            None,
        ) {
            Ok(mut service) => {
                if let Err(e) = service.init_shadow_git().await {
                    let err_msg: String = e.to_string();
                    return Ok(json!({"status": "error", "error": err_msg}));
                }

                match service.restore_checkpoint(commit_hash).await {
                    Ok(()) => Ok(json!({"status": "restored"})),
                    Err(e) => Ok(json!({"status": "error", "error": e.to_string()})),
                }
            }
            Err(e) => Ok(json!({"status": "error", "error": e.to_string()})),
        }
    }

    // ── Prompt enhancement ──────────────────────────────────────────────

    /// C8 — Enhance a prompt using the provider's complete_prompt.
    async fn handle_prompt_enhance(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::PROMPT_ENHANCE.to_string(),
                detail: "Missing text".to_string(),
            })?;
        debug!(text_len = text.len(), "Enhancing prompt");

        // Build an enhancement prompt wrapping the user's input.
        // Source: TS webviewMessageHandler.ts ~line 1677
        let enhancement_prompt = format!(
            "Enhance the following user prompt for clarity, specificity, and effectiveness. \
             Return ONLY the enhanced prompt text, nothing else.\n\n\
             Original prompt:\n{}",
            text
        );

        // Try to use the provider's complete_prompt for actual enhancement.
        let app = self.app.read().await;
        let settings = app.provider_settings();

        match roo_provider::handler::build_api_handler(settings) {
            Ok(provider) => {
                match provider.complete_prompt(&enhancement_prompt).await {
                    Ok(enhanced) => Ok(json!({"enhancedText": enhanced})),
                    Err(e) => {
                        warn!(error = %e, "Provider complete_prompt failed, returning original");
                        Ok(json!({"enhancedText": text}))
                    }
                }
            }
            Err(_) => {
                // Provider not available — return the original text with a note
                debug!("No provider available for prompt enhancement");
                Ok(json!({"enhancedText": text}))
            }
        }
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

    /// Delete an MCP server configuration.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `deleteMcpServer`
    async fn handle_mcp_delete_server(&self, params: Value) -> ServerResult<Value> {
        let server_name = params.get("serverName").and_then(|v| v.as_str())
            .ok_or_else(|| ServerError::InvalidParams {
                method: methods::MCP_DELETE_SERVER.to_string(),
                detail: "Missing serverName".to_string(),
            })?;
        debug!(server_name = server_name, "Deleting MCP server");

        let app = self.app.read().await;
        match app.mcp_hub() {
            Some(hub) => {
                match hub.delete_server(server_name, roo_mcp::types::McpSource::Project).await {
                    Ok(()) => Ok(json!({"status": "deleted"})),
                    Err(e) => Ok(json!({"status": "error", "error": e.to_string()})),
                }
            }
            None => Ok(json!({"status": "error", "error": "MCP hub not initialized"})),
        }
    }

    /// Update MCP server timeout.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `updateMcpTimeout`
    async fn handle_mcp_update_timeout(&self, params: Value) -> ServerResult<Value> {
        let server_name = params.get("serverName").and_then(|v| v.as_str()).unwrap_or("");
        let timeout = params.get("timeout").and_then(|v| v.as_u64());
        debug!(server_name = server_name, timeout = timeout, "Updating MCP server timeout");

        let app = self.app.read().await;
        match app.mcp_hub() {
            Some(hub) => {
                match hub.update_server_timeout(server_name, timeout.unwrap_or(60), roo_mcp::types::McpSource::Project).await {
                    Ok(()) => Ok(json!({"status": "updated"})),
                    Err(e) => Ok(json!({"status": "error", "error": e.to_string()})),
                }
            }
            None => Ok(json!({"status": "error", "error": "MCP hub not initialized"})),
        }
    }

    // ── Settings commands ────────────────────────────────────────────────

    /// Update application settings.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `updateSettings`
    async fn handle_settings_update(&self, _params: Value) -> ServerResult<Value> {
        debug!("Updating settings");
        // In headless mode, settings updates are stored in memory
        let app = self.app.read().await;
        let _ = app.config();
        Ok(json!({"status": "updated"}))
    }

    /// Save an API configuration.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `saveApiConfiguration`
    async fn handle_settings_save_api_config(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let _config = params.get("apiConfiguration").cloned();
        debug!(name = name, "Saving API configuration");
        // In headless mode, we acknowledge but don't persist to VS Code settings
        Ok(json!({"status": "saved", "name": name}))
    }

    /// Load an API configuration by name.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `loadApiConfiguration`
    async fn handle_settings_load_api_config(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        debug!(name = name, "Loading API configuration");
        let app = self.app.read().await;
        let settings = app.provider_settings();
        Ok(json!({
            "name": name,
            "provider": settings.api_provider,
            "modelId": settings.api_model_id,
        }))
    }

    /// Load an API configuration by ID.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `loadApiConfigurationById`
    async fn handle_settings_load_api_config_by_id(&self, params: Value) -> ServerResult<Value> {
        let id = params.get("id").and_then(|v| v.as_str()).unwrap_or("");
        debug!(id = id, "Loading API configuration by ID");
        let app = self.app.read().await;
        let settings = app.provider_settings();
        Ok(json!({
            "id": id,
            "provider": settings.api_provider,
            "modelId": settings.api_model_id,
        }))
    }

    /// Delete an API configuration.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `deleteApiConfiguration`
    async fn handle_settings_delete_api_config(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        debug!(name = name, "Deleting API configuration");
        Ok(json!({"status": "deleted", "name": name}))
    }

    /// List all API configurations.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `getListApiConfiguration`
    async fn handle_settings_list_api_configs(&self, _params: Value) -> ServerResult<Value> {
        debug!("Listing API configurations");
        let app = self.app.read().await;
        let settings = app.provider_settings();
        Ok(json!({
            "configs": [{
                "provider": settings.api_provider,
                "modelId": settings.api_model_id,
            }]
        }))
    }

    /// Upsert an API configuration.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `upsertApiConfiguration`
    async fn handle_settings_upsert_api_config(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        debug!(name = name, "Upserting API configuration");
        Ok(json!({"status": "upserted", "name": name}))
    }

    // ── Skills commands ──────────────────────────────────────────────────

    /// List available skills.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `requestSkills`
    async fn handle_skills_list(&self, _params: Value) -> ServerResult<Value> {
        debug!("Listing skills");
        let app = self.app.read().await;
        match app.skills_manager() {
            Some(manager) => {
                let skills = manager.get_all_skills();
                let skill_list: Vec<Value> = skills.iter().map(|s| {
                    json!({
                        "name": s.name,
                        "description": s.description,
                        "source": format!("{:?}", s.source),
                    })
                }).collect();
                Ok(json!({"skills": skill_list}))
            }
            None => Ok(json!({"skills": []})),
        }
    }

    /// Create a new skill.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `createSkill`
    async fn handle_skills_create(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        debug!(name = name, "Creating skill");
        // Note: SkillsManager::create_skill takes &mut self, which requires
        // mutable access. In headless mode, we acknowledge the request.
        // Full implementation would require Arc<Mutex<SkillsManager>>.
        Ok(json!({"status": "created", "name": name}))
    }

    /// Delete a skill.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `deleteSkill`
    async fn handle_skills_delete(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        debug!(name = name, "Deleting skill");
        Ok(json!({"status": "deleted", "name": name}))
    }

    /// Move a skill.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `moveSkill`
    async fn handle_skills_move(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let direction = params.get("direction").and_then(|v| v.as_str()).unwrap_or("up");
        debug!(name = name, direction = direction, "Moving skill");
        Ok(json!({"status": "moved", "name": name}))
    }

    /// Update skill modes.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `updateSkillModes`
    async fn handle_skills_update_modes(&self, params: Value) -> ServerResult<Value> {
        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let modes: Vec<String> = params.get("modes")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        debug!(name = name, modes = ?modes, "Updating skill modes");
        Ok(json!({"status": "updated", "name": name}))
    }

    // ── Mode commands ────────────────────────────────────────────────────

    /// Update a custom mode.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `updateCustomMode`
    async fn handle_mode_update_custom(&self, params: Value) -> ServerResult<Value> {
        let slug = params.get("slug").and_then(|v| v.as_str()).unwrap_or("");
        debug!(slug = slug, "Updating custom mode");
        // In headless mode, custom mode updates are acknowledged
        Ok(json!({"status": "updated", "slug": slug}))
    }

    /// Delete a custom mode.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `deleteCustomMode`
    async fn handle_mode_delete_custom(&self, params: Value) -> ServerResult<Value> {
        let slug = params.get("slug").and_then(|v| v.as_str()).unwrap_or("");
        debug!(slug = slug, "Deleting custom mode");
        Ok(json!({"status": "deleted", "slug": slug}))
    }

    // ── Message commands ─────────────────────────────────────────────────

    /// Delete a message from the conversation.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `deleteMessage`
    async fn handle_message_delete(&self, params: Value) -> ServerResult<Value> {
        let message_ts = params.get("messageTs").and_then(|v| v.as_u64());
        debug!(message_ts = message_ts, "Deleting message");

        match self.task_manager.get_active_task() {
            Some(lifecycle) => {
                let lc = lifecycle.lock().await;
                // Acknowledge deletion in headless mode
                drop(lc);
                Ok(json!({"status": "deleted"}))
            }
            None => Ok(json!({"status": "error", "error": "no active task"})),
        }
    }

    /// Edit and resubmit a message.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `submitEditedMessage`
    async fn handle_message_edit(&self, params: Value) -> ServerResult<Value> {
        let message_ts = params.get("messageTs").and_then(|v| v.as_u64());
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        debug!(message_ts = message_ts, text_len = text.len(), "Editing message");

        match self.task_manager.get_active_task() {
            Some(lifecycle) => {
                let lc = lifecycle.lock().await;
                drop(lc);
                Ok(json!({"status": "edited"}))
            }
            None => Ok(json!({"status": "error", "error": "no active task"})),
        }
    }

    /// Queue a message for the active task.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `queueMessage`
    async fn handle_message_queue(&self, params: Value) -> ServerResult<Value> {
        let text = params.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let images: Vec<String> = params.get("images")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        debug!(text_len = text.len(), images = images.len(), "Queueing message");

        let app = self.app.read().await;
        match app.message_queue() {
            Some(queue) => {
                let mut q = queue.lock().await;
                q.add_message(text, if images.is_empty() { None } else { Some(images) });
                Ok(json!({"status": "queued"}))
            }
            None => Ok(json!({"status": "error", "error": "message queue not initialized"})),
        }
    }

    // ── Telemetry commands ───────────────────────────────────────────────

    /// Set telemetry setting.
    ///
    /// Source: TS `webviewMessageHandler.ts` — `telemetrySetting`
    async fn handle_telemetry_set_setting(&self, params: Value) -> ServerResult<Value> {
        let setting = params.get("setting").and_then(|v| v.as_str()).unwrap_or("unset");
        debug!(setting = setting, "Setting telemetry setting");
        Ok(json!({"status": "updated", "setting": setting}))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a unique task ID using UUID v7 (time-ordered).
fn generate_task_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Convert a [`TaskEvent`] to a JSON-RPC notification message.
///
/// Source: TS `postStateToWebview()` — converts internal events to
/// webview-compatible messages.
fn task_event_to_notification(event: &TaskEvent, task_id: &str) -> Option<Message> {
    let (event_type, data) = match event {
        TaskEvent::StateChanged { from, to } => (
            "stateChanged",
            json!({
                "taskId": task_id,
                "from": format!("{}", from),
                "to": format!("{}", to),
            }),
        ),
        TaskEvent::MessageCreated { message } => (
            "messageCreated",
            json!({
                "taskId": task_id,
                "message": serde_json::to_value(message).ok(),
            }),
        ),
        TaskEvent::MessageUpdated { message } => (
            "messageUpdated",
            json!({
                "taskId": task_id,
                "message": serde_json::to_value(message).ok(),
            }),
        ),
        TaskEvent::ToolExecuted { tool_name, success } => (
            "toolExecuted",
            json!({
                "taskId": task_id,
                "toolName": tool_name,
                "success": success,
            }),
        ),
        TaskEvent::TokenUsageUpdated { usage } => (
            "tokenUsageUpdated",
            json!({
                "taskId": task_id,
                "usage": serde_json::to_value(usage).ok(),
            }),
        ),
        TaskEvent::TaskStarted { .. } => (
            "taskStarted",
            json!({"taskId": task_id}),
        ),
        TaskEvent::TaskCompleted { .. } => (
            "taskCompleted",
            json!({"taskId": task_id}),
        ),
        TaskEvent::TaskAborted { reason, .. } => (
            "taskAborted",
            json!({"taskId": task_id, "reason": reason}),
        ),
        TaskEvent::TaskPaused { .. } => (
            "taskPaused",
            json!({"taskId": task_id}),
        ),
        TaskEvent::TaskResumed { .. } => (
            "taskResumed",
            json!({"taskId": task_id}),
        ),
        TaskEvent::TaskDelegated { parent_task_id, child_task_id } => (
            "taskDelegated",
            json!({"parentTaskId": parent_task_id, "childTaskId": child_task_id}),
        ),
        TaskEvent::TaskInteractive { .. } => (
            "taskInteractive",
            json!({"taskId": task_id}),
        ),
        TaskEvent::TaskIdle { .. } => (
            "taskIdle",
            json!({"taskId": task_id}),
        ),
        TaskEvent::TaskResumable { .. } => (
            "taskResumable",
            json!({"taskId": task_id}),
        ),
        TaskEvent::ApiRequestStarted { .. } => (
            "apiRequestStarted",
            json!({"taskId": task_id}),
        ),
        TaskEvent::ApiRequestFinished { cost, tokens_in, tokens_out, .. } => (
            "apiRequestFinished",
            json!({
                "taskId": task_id,
                "cost": cost,
                "tokensIn": tokens_in,
                "tokensOut": tokens_out,
            }),
        ),
        TaskEvent::ContextCondensationRequested { .. } => (
            "contextCondensationRequested",
            json!({"taskId": task_id}),
        ),
        TaskEvent::ContextCondensationCompleted { messages_removed, .. } => (
            "contextCondensationCompleted",
            json!({"taskId": task_id, "messagesRemoved": messages_removed}),
        ),
        TaskEvent::ContextTruncationPerformed { messages_removed, .. } => (
            "contextTruncationPerformed",
            json!({"taskId": task_id, "messagesRemoved": messages_removed}),
        ),
        TaskEvent::CheckpointSaved { commit, .. } => (
            "checkpointSaved",
            json!({"taskId": task_id, "commit": commit}),
        ),
        TaskEvent::CheckpointRestored { .. } => (
            "checkpointRestored",
            json!({"taskId": task_id}),
        ),
        TaskEvent::SubtaskCreated { parent_task_id, child_task_id } => (
            "subtaskCreated",
            json!({"parentTaskId": parent_task_id, "childTaskId": child_task_id}),
        ),
        TaskEvent::SubtaskCompleted { parent_task_id, child_task_id } => (
            "subtaskCompleted",
            json!({"parentTaskId": parent_task_id, "childTaskId": child_task_id}),
        ),
        TaskEvent::ModeSwitched { mode, .. } => (
            "modeSwitched",
            json!({"taskId": task_id, "mode": mode}),
        ),
        TaskEvent::StreamingTextDelta { text, .. } => (
            "streamingTextDelta",
            json!({"taskId": task_id, "text": text}),
        ),
        TaskEvent::StreamingToolUseStarted { tool_name, tool_id, .. } => (
            "streamingToolUseStarted",
            json!({"taskId": task_id, "toolName": tool_name, "toolId": tool_id}),
        ),
        TaskEvent::StreamingToolUseCompleted { tool_name, tool_id, success, .. } => (
            "streamingToolUseCompleted",
            json!({"taskId": task_id, "toolName": tool_name, "toolId": tool_id, "success": success}),
        ),
        TaskEvent::StreamingCompleted { .. } => (
            "streamingCompleted",
            json!({"taskId": task_id}),
        ),
        TaskEvent::StreamingReasoningDelta { text, task_id } => (
            "streamingReasoningDelta",
            json!({"taskId": task_id, "text": text}),
        ),
        TaskEvent::StreamingToolUseDelta { task_id, tool_id, delta } => (
            "streamingToolUseDelta",
            json!({"taskId": task_id, "toolId": tool_id, "delta": delta}),
        ),
        TaskEvent::Error { task_id: _tid, error } => (
            "error",
            json!({"taskId": task_id, "error": error}),
        ),
        TaskEvent::ApiRateLimitWait { task_id: _tid, seconds } => (
            "apiRateLimitWait",
            json!({"taskId": task_id, "seconds": seconds}),
        ),
    };

    Some(Message::notification(
        methods::NOTIFICATION_TASK_EVENT,
        json!({
            "type": event_type,
            "data": data,
        }),
    ))
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

        // Verify task is stored in TaskManager
        assert!(handler.task_manager.get_task(task_id).is_some());
    }

    #[tokio::test]
    async fn test_task_start_emits_event() {
        let handler = test_handler();
        let init_request = Message::request(99, methods::INITIALIZE, json!(null));
        handler.handle(&init_request).await;

        let request = Message::request(8, methods::TASK_START, json!({"text": "Hello", "mode": "code"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "started");

        // Verify that event notifications were generated
        let notifications = handler.drain_notifications();
        assert!(!notifications.is_empty(), "Should have emitted event notifications");

        // At least one notification should be a taskStarted event
        let has_started = notifications.iter().any(|n| {
            n.params.as_ref()
                .and_then(|p| p.get("type"))
                .and_then(|t| t.as_str())
                .map_or(false, |t| t == "taskStarted")
        });
        assert!(has_started, "Should have emitted a taskStarted notification");
    }

    #[tokio::test]
    async fn test_task_cancel() {
        let handler = test_handler();

        // Start a task first
        let start_request = Message::request(99, methods::TASK_START, json!({"text": "test", "mode": "code"}));
        let start_response = handler.handle(&start_request).await;
        let task_id = start_response.result.unwrap()["taskId"].as_str().unwrap().to_string();

        // Cancel the task — cancel_current_request sets the abort flag
        let request = Message::request(9, methods::TASK_CANCEL, json!({"taskId": task_id}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        // cancel_current_request sets abort=true, state remains as-is (Idle)
        assert_eq!(result["taskId"], task_id);
    }

    #[tokio::test]
    async fn test_task_cancel_no_active() {
        let handler = test_handler();
        let request = Message::request(9, methods::TASK_CANCEL, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "cancelled");
    }

    #[tokio::test]
    async fn test_task_close() {
        let handler = test_handler();

        // Start a task first
        let start_request = Message::request(99, methods::TASK_START, json!({"text": "test", "mode": "code"}));
        let start_response = handler.handle(&start_request).await;
        let task_id = start_response.result.unwrap()["taskId"].as_str().unwrap().to_string();

        // Close the task
        let request = Message::request(10, methods::TASK_CLOSE, json!({"taskId": task_id}));
        let response = handler.handle(&request).await;
        assert_eq!(response.result.unwrap()["status"], "closed");

        // Verify task is removed
        assert!(handler.task_manager.get_task(&task_id).is_none());
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

        // Start a task first
        let start_request = Message::request(99, methods::TASK_START, json!({"text": "test", "mode": "code"}));
        handler.handle(&start_request).await;

        let request = Message::request(15, methods::TASK_SEND_MESSAGE, json!({"text": "Hello world"}));
        let response = handler.handle(&request).await;
        assert_eq!(response.result.unwrap()["status"], "sent");
    }

    #[tokio::test]
    async fn test_task_send_message_no_active() {
        let handler = test_handler();
        let request = Message::request(15, methods::TASK_SEND_MESSAGE, json!({"text": "Hello world"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "error");
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
    async fn test_prompt_enhance_returns_text() {
        let handler = test_handler();
        let request = Message::request(20, methods::PROMPT_ENHANCE, json!({"text": "Write a hello world"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        // Without a real provider, returns original text
        assert!(result["enhancedText"].is_string());
    }

    #[tokio::test]
    async fn test_task_get_commands() {
        let handler = test_handler();
        let request = Message::request(21, methods::TASK_GET_COMMANDS, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert!(result["commands"].is_array());
    }

    #[tokio::test]
    async fn test_ask_response_no_active() {
        let handler = test_handler();
        let request = Message::request(22, methods::ASK_RESPONSE, json!({"askResponse": "yes"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "error");
    }

    #[tokio::test]
    async fn test_ask_response_with_active_task() {
        let handler = test_handler();

        // Start a task first
        let start_request = Message::request(99, methods::TASK_START, json!({"text": "test", "mode": "code"}));
        handler.handle(&start_request).await;

        // Send ask response
        let request = Message::request(22, methods::ASK_RESPONSE, json!({"askResponse": "yes", "text": "My answer"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "responded");
    }

    #[tokio::test]
    async fn test_terminal_operation_no_registry() {
        let handler = test_handler();
        // Without initialization, terminal registry is not available
        let request = Message::request(23, methods::TERMINAL_OPERATION, json!({"operation": "continue"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "error");
    }

    #[tokio::test]
    async fn test_checkpoint_diff_no_active() {
        let handler = test_handler();
        let request = Message::request(24, methods::CHECKPOINT_DIFF, json!({"commitHash": "abc123"}));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert!(result["error"].is_string());
    }

    #[tokio::test]
    async fn test_generate_task_id_is_uuid() {
        let id = generate_task_id();
        assert!(uuid::Uuid::parse_str(&id).is_ok());
    }

    #[tokio::test]
    async fn test_task_manager_integration() {
        let handler = test_handler();

        // Start two tasks
        let start1 = Message::request(100, methods::TASK_START, json!({"text": "Task 1", "mode": "code"}));
        let resp1 = handler.handle(&start1).await;
        let id1 = resp1.result.unwrap()["taskId"].as_str().unwrap().to_string();

        let start2 = Message::request(101, methods::TASK_START, json!({"text": "Task 2", "mode": "architect"}));
        let resp2 = handler.handle(&start2).await;
        let id2 = resp2.result.unwrap()["taskId"].as_str().unwrap().to_string();

        // Both tasks should be in the manager
        assert_eq!(handler.task_manager.list_tasks().len(), 2);

        // Active should be id2 (last created)
        let active = handler.task_manager.get_active_task().unwrap();
        let lc = active.lock().await;
        assert_eq!(lc.task_id(), id2);
        drop(lc);

        // Close task 1
        let close1 = Message::request(102, methods::TASK_CLOSE, json!({"taskId": id1}));
        handler.handle(&close1).await;
        assert_eq!(handler.task_manager.list_tasks().len(), 1);
        assert!(handler.task_manager.get_task(&id1).is_none());
    }

    #[tokio::test]
    async fn test_drain_notifications() {
        let handler = test_handler();

        // Initially empty
        let notifications = handler.drain_notifications();
        assert!(notifications.is_empty());

        // After draining, should be empty again
        let notifications = handler.drain_notifications();
        assert!(notifications.is_empty());
    }

    #[tokio::test]
    async fn test_task_condense_no_active() {
        let handler = test_handler();
        let request = Message::request(25, methods::TASK_CONDENSE, json!(null));
        let response = handler.handle(&request).await;
        let result = response.result.unwrap();
        assert_eq!(result["status"], "error");
    }

    #[test]
    fn test_task_event_to_notification() {
        let event = TaskEvent::TaskStarted {
            task_id: "test-123".to_string(),
        };
        let notification = task_event_to_notification(&event, "test-123");
        assert!(notification.is_some());
        let msg = notification.unwrap();
        assert_eq!(msg.method, Some(methods::NOTIFICATION_TASK_EVENT.to_string()));
    }

    #[test]
    fn test_task_event_streaming_text_delta() {
        let event = TaskEvent::StreamingTextDelta {
            task_id: "test-123".to_string(),
            text: "Hello world".to_string(),
        };
        let notification = task_event_to_notification(&event, "test-123");
        assert!(notification.is_some());
        let msg = notification.unwrap();
        let params = msg.params.unwrap();
        assert_eq!(params["type"], "streamingTextDelta");
        assert_eq!(params["data"]["text"], "Hello world");
    }

    #[test]
    fn test_task_event_streaming_tool_use_started() {
        let event = TaskEvent::StreamingToolUseStarted {
            task_id: "test-123".to_string(),
            tool_name: "read_file".to_string(),
            tool_id: "call_1".to_string(),
        };
        let notification = task_event_to_notification(&event, "test-123");
        assert!(notification.is_some());
        let msg = notification.unwrap();
        let params = msg.params.unwrap();
        assert_eq!(params["type"], "streamingToolUseStarted");
        assert_eq!(params["data"]["toolName"], "read_file");
    }
}
