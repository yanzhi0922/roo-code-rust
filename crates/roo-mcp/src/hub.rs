//! McpHub — core MCP connection manager.
//!
//! Corresponds to TS: `McpHub` class.
//! Manages MCP server connections, tool calls, resource reads, and configuration updates.

use std::collections::HashMap;
use std::sync::Arc;

use roo_types::mcp::{
    McpConnectionStatus, McpServerConnection, McpTool, McpToolCallResponse,
};
use tokio::sync::RwLock;

use crate::client::McpClient;
use crate::config::{validate_server_config, ValidatedServerConfig};
use crate::error::{McpError, McpResult};
use crate::name_utils::{sanitize_mcp_name, tool_names_match};
use crate::transport::{
    McpTransport, SseTransport, StdioTransport, StreamableHttpTransport,
};
use crate::types::{
    ConnectedMcpConnection, DisableReason, DisconnectedMcpConnection, McpConnection,
    McpServerState, McpSource,
};

/// Configuration change callback type.
pub type ConfigChangeCallback = Box<dyn Fn(&str, McpSource) + Send + Sync>;

/// State change callback type.
pub type StateChangeCallback = Box<dyn Fn() + Send + Sync>;

/// McpHub manages all MCP server connections.
///
/// It handles connecting to servers, discovering tools and resources,
/// and routing tool calls and resource reads to the appropriate server.
pub struct McpHub {
    /// All server connections (connected and disconnected).
    connections: Arc<RwLock<Vec<McpConnection>>>,
    /// Reference count for active clients.
    ref_count: Arc<RwLock<usize>>,
    /// Whether MCP is globally enabled.
    mcp_enabled: Arc<RwLock<bool>>,
    /// Whether the hub has been disposed.
    disposed: Arc<RwLock<bool>>,
    /// Registry mapping sanitized names to original names.
    sanitized_name_registry: Arc<RwLock<HashMap<String, String>>>,
    /// Callback for state changes (to notify UI).
    on_state_change: Option<StateChangeCallback>,
    /// Initialization promise handle.
    initialized: Arc<RwLock<bool>>,
}

impl McpHub {
    /// Create a new McpHub.
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(Vec::new())),
            ref_count: Arc::new(RwLock::new(0)),
            mcp_enabled: Arc::new(RwLock::new(true)),
            disposed: Arc::new(RwLock::new(false)),
            sanitized_name_registry: Arc::new(RwLock::new(HashMap::new())),
            on_state_change: None,
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Create a new McpHub with a state change callback.
    pub fn with_state_change_callback(callback: StateChangeCallback) -> Self {
        let mut hub = Self::new();
        hub.on_state_change = Some(callback);
        hub
    }

    /// Register a client (increment reference count).
    pub async fn register_client(&self) {
        let mut count = self.ref_count.write().await;
        *count += 1;
        tracing::debug!("McpHub: Client registered. Ref count: {}", *count);
    }

    /// Unregister a client (decrement reference count).
    /// If count reaches zero, disposes the hub.
    pub async fn unregister_client(&self) -> McpResult<()> {
        let mut count = self.ref_count.write().await;
        if *count > 0 {
            *count -= 1;
        }
        tracing::debug!("McpHub: Client unregistered. Ref count: {}", *count);

        if *count == 0 {
            tracing::info!("McpHub: Last client unregistered. Disposing hub.");
            drop(count);
            self.dispose().await?;
        }
        Ok(())
    }

    /// Wait until the hub is marked as initialized.
    pub async fn wait_until_ready(&self) {
        // Simple spin-wait; in practice, the hub is initialized synchronously
        // or via an initialization method.
        let ready = self.initialized.read().await;
        if *ready {
            return;
        }
        drop(ready);
        // Wait briefly
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    /// Mark the hub as initialized.
    pub async fn set_initialized(&self) {
        let mut initialized = self.initialized.write().await;
        *initialized = true;
    }

    /// Connect to an MCP server.
    ///
    /// Validates the configuration, creates the appropriate transport,
    /// and performs the MCP handshake.
    pub async fn connect_to_server(
        &self,
        name: &str,
        config: &serde_json::Value,
        source: McpSource,
    ) -> McpResult<()> {
        self.check_disposed()?;

        // Validate configuration
        let validated = validate_server_config(config, Some(name))?;

        // Remove existing connection if it exists with the same source
        self.delete_connection(name, source).await?;

        // Register the sanitized name
        let sanitized = sanitize_mcp_name(name);
        let mut registry = self.sanitized_name_registry.write().await;
        registry.insert(sanitized, name.to_string());
        drop(registry);

        // Check if MCP is globally enabled
        let mcp_enabled = *self.mcp_enabled.read().await;
        if !mcp_enabled {
            let connection = self.create_placeholder_connection(
                name,
                &validated,
                source,
                DisableReason::McpDisabled,
            );
            self.connections.write().await.push(connection);
            self.notify_state_change();
            return Ok(());
        }

        // Skip connecting to disabled servers
        if validated.is_disabled() {
            let connection = self.create_placeholder_connection(
                name,
                &validated,
                source,
                DisableReason::ServerDisabled,
            );
            self.connections.write().await.push(connection);
            self.notify_state_change();
            return Ok(());
        }

        // Create server state
        let mut server_state = McpServerState::new(
            name.to_string(),
            serde_json::to_string(&validated)?,
            source,
        );
        server_state.status = McpConnectionStatus::Connecting;

        // Create a placeholder disconnected connection while connecting
        let placeholder = McpConnection::Disconnected(DisconnectedMcpConnection {
            server: server_state.clone(),
        });
        self.connections.write().await.push(placeholder);

        // Attempt to connect
        match self.establish_connection(name, &validated, source).await {
            Ok((client, transport, tools, resources, resource_templates, instructions)) => {
                // Replace the placeholder with a connected connection
                let mut connections = self.connections.write().await;
                if let Some(conn) = connections.find_mut(name, source) {
                    conn.server_mut().status = McpConnectionStatus::Connected;
                    conn.server_mut().error.clear();
                    conn.server_mut().tools = tools;
                    conn.server_mut().resources = resources;
                    conn.server_mut().resource_templates = resource_templates;
                    conn.server_mut().instructions = instructions;

                    // Replace with connected connection
                    let connected = ConnectedMcpConnection {
                        server: conn.server_mut().clone(),
                        client,
                        transport,
                    };
                    *conn = McpConnection::Connected(connected);
                }

                tracing::info!("Connected to MCP server '{}'", name);
                self.notify_state_change();
                Ok(())
            }
            Err(e) => {
                // Update the placeholder with error info
                let mut connections = self.connections.write().await;
                if let Some(conn) = connections.find_mut(name, source) {
                    conn.server_mut().status = McpConnectionStatus::Error;
                    conn.server_mut().append_error(&e.to_string());
                }

                tracing::error!("Failed to connect to MCP server '{}': {}", name, e);
                self.notify_state_change();
                Err(e)
            }
        }
    }

    /// Delete a connection by name and source.
    ///
    /// Properly closes the transport for connected servers before removing.
    /// Corresponds to TS: `McpHub.deleteConnection`.
    pub async fn delete_connection(&self, name: &str, source: McpSource) -> McpResult<()> {
        let mut connections = self.connections.write().await;
        let before_len = connections.len();

        // Partition into matching (to delete) and remaining connections
        let (to_delete, remaining): (Vec<_>, Vec<_>) =
            connections.drain(..).partition(|conn| {
                conn.name() == name && conn.source() == source
            });

        // Close transports for connected servers being deleted (best-effort)
        for conn in to_delete {
            if let McpConnection::Connected(mut c) = conn {
                tracing::debug!("Closing transport for '{}' during deletion", name);
                let _ = c.transport.close().await;
            }
        }

        // Restore remaining connections
        *connections = remaining;

        // Clean up sanitized name registry if no more connections with this name
        let has_remaining = connections.iter().any(|c| c.name() == name);
        if !has_remaining {
            let sanitized = sanitize_mcp_name(name);
            let mut registry = self.sanitized_name_registry.write().await;
            registry.remove(&sanitized);
        }

        if connections.len() < before_len {
            tracing::info!("Deleted MCP connection '{}' ({:?})", name, source);
            drop(connections);
            self.notify_state_change();
        }

        Ok(())
    }

    /// Update server connections for a given source.
    ///
    /// Removes connections that no longer exist in the config, adds new ones,
    /// and restarts connections whose config has changed.
    pub async fn update_server_connections(
        &self,
        servers: &HashMap<String, serde_json::Value>,
        source: McpSource,
    ) -> McpResult<()> {
        self.check_disposed()?;

        let mut connections = self.connections.write().await;

        // Remove connections for this source that are no longer in the config
        connections.retain(|conn| {
            if conn.source() != source {
                return true;
            }
            servers.contains_key(conn.name())
        });

        drop(connections);

        // Connect or update each server
        for (name, config) in servers {
            match self.connect_to_server(name, config, source).await {
                Ok(()) => {}
                Err(e) => {
                    tracing::error!("Failed to connect to server '{}': {}", name, e);
                }
            }
        }

        self.notify_state_change();
        Ok(())
    }

    /// Restart a connection by name and source.
    pub async fn restart_connection(&self, name: &str, source: McpSource) -> McpResult<()> {
        self.check_disposed()?;

        let config_str = {
            let connections = self.connections.read().await;
            connections
                .find(name, source)
                .and_then(|c| Some(c.server().config.clone()))
        };

        if let Some(config_str) = config_str {
            let config: serde_json::Value = serde_json::from_str(&config_str)?;
            self.connect_to_server(name, &config, source).await?;
        }

        Ok(())
    }

    /// Refresh all connections (restart all).
    pub async fn refresh_all_connections(&self) -> McpResult<()> {
        self.check_disposed()?;

        let connection_infos: Vec<(String, McpSource, String)> = {
            let connections = self.connections.read().await;
            connections
                .iter()
                .map(|c| {
                    (
                        c.name().to_string(),
                        c.source(),
                        c.server().config.clone(),
                    )
                })
                .collect()
        };

        for (name, source, config_str) in connection_infos {
            let config: serde_json::Value = serde_json::from_str(&config_str).unwrap_or_default();
            if let Err(e) = self.connect_to_server(&name, &config, source).await {
                tracing::error!("Failed to refresh connection '{}': {}", name, e);
            }
        }

        Ok(())
    }

    /// Call a tool on a specific server with timeout support.
    ///
    /// The timeout is read from the server configuration (default: 60 seconds).
    /// Corresponds to TS: `McpHub.callTool`.
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Option<serde_json::Value>,
    ) -> McpResult<McpToolCallResponse> {
        self.check_disposed()?;

        // Extract timeout from server config before acquiring write lock
        let timeout_duration = {
            let connections = self.connections.read().await;
            let conn = connections
                .iter()
                .find(|c| c.name() == server_name);
            match conn {
                Some(c) => {
                    let config: serde_json::Value =
                        serde_json::from_str(&c.server().config).unwrap_or_default();
                    let timeout_secs = config
                        .get("timeout")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(60);
                    std::time::Duration::from_secs(timeout_secs)
                }
                None => std::time::Duration::from_secs(60),
            }
        };

        let mut connections = self.connections.write().await;

        let connection = connections
            .find_connected_mut(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        match connection {
            McpConnection::Connected(conn) => {
                let result = tokio::time::timeout(
                    timeout_duration,
                    conn.client
                        .call_tool(&mut *conn.transport, tool_name, arguments),
                )
                .await;

                match result {
                    Ok(Ok(response)) => {
                        tracing::info!(
                            "Tool call '{}' on '{}' succeeded",
                            tool_name,
                            server_name
                        );
                        Ok(response)
                    }
                    Ok(Err(e)) => {
                        conn.server.status = McpConnectionStatus::Error;
                        conn.server.append_error(&e.to_string());
                        tracing::error!(
                            "Tool call '{}' on '{}' failed: {}",
                            tool_name,
                            server_name,
                            e
                        );
                        Err(e)
                    }
                    Err(_) => {
                        let err_msg = format!(
                            "Tool call '{}' on '{}' timed out after {}s",
                            tool_name,
                            server_name,
                            timeout_duration.as_secs()
                        );
                        conn.server.status = McpConnectionStatus::Error;
                        conn.server.append_error(&err_msg);
                        tracing::error!("{}", err_msg);
                        Err(McpError::ToolCallFailed {
                            server_name: server_name.to_string(),
                            tool_name: tool_name.to_string(),
                            error: err_msg,
                        })
                    }
                }
            }
            _ => Err(McpError::NotConnected(server_name.to_string())),
        }
    }

    /// Read a resource from a specific server.
    pub async fn read_resource(
        &self,
        server_name: &str,
        uri: &str,
    ) -> McpResult<roo_types::mcp::McpResourceResponse> {
        self.check_disposed()?;

        let mut connections = self.connections.write().await;

        let connection = connections
            .find_connected_mut(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        match connection {
            McpConnection::Connected(conn) => {
                let result = conn
                    .client
                    .read_resource(&mut *conn.transport, uri)
                    .await;

                match result {
                    Ok(response) => {
                        tracing::info!(
                            "Resource read '{}' on '{}' succeeded",
                            uri,
                            server_name
                        );
                        Ok(response)
                    }
                    Err(e) => {
                        conn.server.status = McpConnectionStatus::Error;
                        conn.server.append_error(&e.to_string());
                        Err(e)
                    }
                }
            }
            _ => Err(McpError::NotConnected(server_name.to_string())),
        }
    }

    /// Fetch the tools list from a specific server and update the server state.
    ///
    /// Tools are annotated with `always_allow` and `enabled_for_prompt` based on
    /// the server configuration's `alwaysAllow` and `disabledTools` lists.
    ///
    /// Corresponds to TS: `McpHub.fetchToolsList`.
    pub async fn fetch_tools_list(&self, server_name: &str) -> McpResult<Vec<McpTool>> {
        self.check_disposed()?;

        let mut connections = self.connections.write().await;

        let connection = connections
            .find_connected_mut(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        match connection {
            McpConnection::Connected(conn) => {
                let raw_tools = conn
                    .client
                    .list_tools(&mut *conn.transport)
                    .await?;

                // Read alwaysAllow and disabledTools from config
                let always_allow_config: Vec<String> = serde_json::from_str(&conn.server.config)
                    .ok()
                    .and_then(|v: serde_json::Value| v.get("alwaysAllow").cloned())
                    .and_then(|v| serde_json::from_value(v).ok())
                    .unwrap_or_default();

                let disabled_tools_list: Vec<String> = serde_json::from_str(&conn.server.config)
                    .ok()
                    .and_then(|v: serde_json::Value| v.get("disabledTools").cloned())
                    .and_then(|v| serde_json::from_value(v).ok())
                    .unwrap_or_default();

                // Check if wildcard "*" is in the alwaysAllow config
                let has_wildcard = always_allow_config.contains(&"*".to_string());

                // Annotate tools with always_allow and enabled_for_prompt
                let tools: Vec<McpTool> = raw_tools
                    .into_iter()
                    .map(|mut tool| {
                        tool.always_allow =
                            has_wildcard || always_allow_config.contains(&tool.name);
                        tool.enabled_for_prompt = !disabled_tools_list.contains(&tool.name);
                        tool
                    })
                    .collect();

                conn.server.tools = tools.clone();
                tracing::info!(
                    "Fetched {} tools from '{}'",
                    tools.len(),
                    server_name
                );
                Ok(tools)
            }
            _ => Err(McpError::NotConnected(server_name.to_string())),
        }
    }

    /// Toggle whether a tool is always allowed (auto-approved).
    ///
    /// Updates the tool's `always_allow` flag in the server state and
    /// persists the change to the server configuration's `alwaysAllow` list.
    ///
    /// Corresponds to TS: `McpHub.toggleToolAlwaysAllow`.
    pub async fn toggle_tool_always_allow(
        &self,
        server_name: &str,
        source: McpSource,
        tool_name: &str,
        should_allow: bool,
    ) -> McpResult<()> {
        self.check_disposed()?;

        let mut connections = self.connections.write().await;

        let connection = connections
            .find_mut(server_name, source)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        let server = connection.server_mut();

        // Update the tool's always_allow flag in the server state
        if let Some(tool) = server.tools.iter_mut().find(|t| t.name == tool_name) {
            tool.always_allow = should_allow;
        }

        // Update the config's alwaysAllow list
        let mut config: serde_json::Value = serde_json::from_str(&server.config)?;
        if config.is_null() {
            config = serde_json::json!({});
        }
        if config.get("alwaysAllow").is_none() {
            config["alwaysAllow"] = serde_json::json!([]);
        }

        let always_allow = config["alwaysAllow"].as_array_mut()
            .ok_or_else(|| McpError::ConfigError("alwaysAllow is not an array".to_string()))?;

        if should_allow {
            if !always_allow.iter().any(|v| v.as_str() == Some(tool_name)) {
                always_allow.push(serde_json::Value::String(tool_name.to_string()));
            }
        } else {
            always_allow.retain(|v| v.as_str() != Some(tool_name));
        }

        server.config = serde_json::to_string(&config)?;
        tracing::info!(
            "Toggle always-allow for tool '{}' on '{}': {}",
            tool_name,
            server_name,
            should_allow
        );

        drop(connections);
        self.notify_state_change();
        Ok(())
    }

    /// Toggle whether a tool is enabled for prompts.
    ///
    /// Updates the tool's `enabled_for_prompt` flag in the server state and
    /// persists the change to the server configuration's `disabledTools` list.
    ///
    /// When `is_enabled` is true, removes the tool from `disabledTools`.
    /// When `is_enabled` is false, adds the tool to `disabledTools`.
    ///
    /// Corresponds to TS: `McpHub.toggleToolEnabledForPrompt`.
    pub async fn toggle_tool_enabled_for_prompt(
        &self,
        server_name: &str,
        source: McpSource,
        tool_name: &str,
        is_enabled: bool,
    ) -> McpResult<()> {
        self.check_disposed()?;

        let mut connections = self.connections.write().await;

        let connection = connections
            .find_mut(server_name, source)
            .ok_or_else(|| McpError::ServerNotFound(server_name.to_string()))?;

        let server = connection.server_mut();

        // Update the tool's enabled_for_prompt flag in the server state
        if let Some(tool) = server.tools.iter_mut().find(|t| t.name == tool_name) {
            tool.enabled_for_prompt = is_enabled;
        }

        // Update the config's disabledTools list
        // When is_enabled is true, remove from disabledTools.
        // When is_enabled is false, add to disabledTools.
        let mut config: serde_json::Value = serde_json::from_str(&server.config)?;
        if config.is_null() {
            config = serde_json::json!({});
        }
        if config.get("disabledTools").is_none() {
            config["disabledTools"] = serde_json::json!([]);
        }

        let disabled_tools = config["disabledTools"].as_array_mut()
            .ok_or_else(|| McpError::ConfigError("disabledTools is not an array".to_string()))?;

        if is_enabled {
            // Remove from disabled list
            disabled_tools.retain(|v| v.as_str() != Some(tool_name));
        } else {
            // Add to disabled list
            if !disabled_tools.iter().any(|v| v.as_str() == Some(tool_name)) {
                disabled_tools.push(serde_json::Value::String(tool_name.to_string()));
            }
        }

        server.config = serde_json::to_string(&config)?;
        tracing::info!(
            "Toggle enabled-for-prompt for tool '{}' on '{}': {}",
            tool_name,
            server_name,
            is_enabled
        );

        drop(connections);
        self.notify_state_change();
        Ok(())
    }

    /// Toggle whether a server is disabled.
    pub async fn toggle_server_disabled(
        &self,
        server_name: &str,
        source: McpSource,
        disabled: bool,
    ) -> McpResult<()> {
        self.check_disposed()?;

        let config_str = {
            let connections = self.connections.read().await;
            connections
                .find(server_name, source)
                .map(|c| c.server().config.clone())
        };

        if let Some(config_str) = config_str {
            let mut config: serde_json::Value = serde_json::from_str(&config_str)?;
            config["disabled"] = serde_json::Value::Bool(disabled);
            self.connect_to_server(server_name, &config, source).await?;
        }

        Ok(())
    }

    /// Handle MCP globally enabled/disabled change.
    ///
    /// Corresponds to TS: `McpHub.handleMcpEnabledChange`.
    /// When disabled, properly closes all transports and converts to disconnected placeholders.
    /// When enabled, refreshes all connections.
    pub async fn handle_mcp_enabled_change(&self, enabled: bool) -> McpResult<()> {
        *self.mcp_enabled.write().await = enabled;

        if enabled {
            // Re-connect all servers
            self.refresh_all_connections().await?;
        } else {
            // Disconnect all servers: close transports and convert to disconnected placeholders
            let mut connections = self.connections.write().await;
            let drained: Vec<McpConnection> = connections.drain(..).collect();
            let mut new_connections = Vec::with_capacity(drained.len());
            for conn in drained {
                match conn {
                    McpConnection::Connected(mut c) => {
                        // Properly close the transport
                        let _ = c.transport.close().await;
                        c.server.status = McpConnectionStatus::Disconnected;
                        new_connections.push(McpConnection::Disconnected(
                            DisconnectedMcpConnection { server: c.server },
                        ));
                    }
                    other => {
                        new_connections.push(other);
                    }
                }
            }
            *connections = new_connections;
        }

        self.notify_state_change();
        Ok(())
    }

    /// Get enabled servers, deduplicating by name with project servers taking priority.
    ///
    /// Corresponds to TS: `McpHub.getServers`.
    /// Only returns enabled (non-disabled) servers. When the same server name exists
    /// in both global and project sources, the project version takes priority.
    pub fn get_servers(&self) -> Vec<McpServerConnection> {
        // Synchronous version — use try_read to avoid blocking
        match self.connections.try_read() {
            Ok(connections) => {
                let enabled: Vec<_> = connections
                    .iter()
                    .filter(|c| !c.server().disabled)
                    .collect();

                // Deduplicate: project servers override global servers with same name
                let mut seen = std::collections::HashMap::new();
                for conn in &enabled {
                    let name = conn.name().to_string();
                    match seen.get(&name) {
                        Some(existing_source) => {
                            // Project takes priority over global
                            if conn.source() == McpSource::Project
                                && *existing_source != McpSource::Project
                            {
                                seen.insert(name, conn.source());
                            }
                        }
                        None => {
                            seen.insert(name, conn.source());
                        }
                    }
                }

                // Build result from deduplicated connections
                enabled
                    .into_iter()
                    .filter(|conn| {
                        seen.get(conn.name()).map_or(false, |s| *s == conn.source())
                    })
                    .map(|conn| McpServerConnection {
                        name: conn.name().to_string(),
                        status: conn.server().status,
                        tools: conn.server().tools.clone(),
                        resources: conn.server().resources.clone(),
                        resource_templates: conn.server().resource_templates.clone(),
                        disabled_tools: conn
                            .server()
                            .tools
                            .iter()
                            .filter(|t| !t.enabled_for_prompt)
                            .map(|t| t.name.clone())
                            .collect(),
                        errors: conn.server().error_history.clone(),
                    })
                    .collect()
            }
            Err(_) => Vec::new(),
        }
    }

    /// Get all servers asynchronously (including disabled ones).
    ///
    /// Corresponds to TS: `McpHub.getAllServers`.
    pub async fn get_all_servers(&self) -> Vec<McpServerConnection> {
        let connections = self.connections.read().await;
        connections
            .iter()
            .map(|conn| McpServerConnection {
                name: conn.name().to_string(),
                status: conn.server().status,
                tools: conn.server().tools.clone(),
                resources: conn.server().resources.clone(),
                resource_templates: conn.server().resource_templates.clone(),
                disabled_tools: conn
                    .server()
                    .tools
                    .iter()
                    .filter(|t| !t.enabled_for_prompt)
                    .map(|t| t.name.clone())
                    .collect(),
                errors: conn.server().error_history.clone(),
            })
            .collect()
    }

    /// Find a server name by its sanitized name.
    ///
    /// Corresponds to TS: `McpHub.findServerNameBySanitizedName`.
    /// Checks in order: exact match → registry → fuzzy match.
    pub async fn find_server_name_by_sanitized_name(&self, sanitized_name: &str) -> Option<String> {
        // 1. First try exact match against connection names
        let connections = self.connections.read().await;
        if let Some(exact) = connections.iter().find(|c| c.name() == sanitized_name) {
            return Some(exact.name().to_string());
        }
        drop(connections);

        // 2. Check the sanitized name registry
        let registry = self.sanitized_name_registry.read().await;
        if let Some(original) = registry.get(sanitized_name) {
            return Some(original.clone());
        }
        drop(registry);

        // 3. Fallback: fuzzy match (treat hyphens and underscores as equivalent)
        let connections = self.connections.read().await;
        for conn in connections.iter() {
            if tool_names_match(conn.name(), sanitized_name) {
                return Some(conn.name().to_string());
            }
        }

        None
    }

    /// Dispose the hub and all connections.
    pub async fn dispose(&self) -> McpResult<()> {
        let mut disposed = self.disposed.write().await;
        if *disposed {
            return Ok(());
        }
        *disposed = true;
        drop(disposed);

        let mut connections = self.connections.write().await;
        for conn in connections.drain(..) {
            if let McpConnection::Connected(mut c) = conn {
                let _ = c.transport.close().await;
            }
        }

        tracing::info!("McpHub disposed");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn check_disposed(&self) -> McpResult<()> {
        let disposed = self.disposed.try_read().map_err(|_| McpError::Disposed)?;
        if *disposed {
            return Err(McpError::Disposed);
        }
        Ok(())
    }

    fn notify_state_change(&self) {
        if let Some(callback) = &self.on_state_change {
            callback();
        }
    }

    fn create_placeholder_connection(
        &self,
        name: &str,
        config: &ValidatedServerConfig,
        source: McpSource,
        reason: DisableReason,
    ) -> McpConnection {
        let mut server_state = McpServerState::new(
            name.to_string(),
            serde_json::to_string(config).unwrap_or_default(),
            source,
        );
        server_state.disabled = reason == DisableReason::ServerDisabled || config.is_disabled();
        server_state.status = McpConnectionStatus::Disconnected;

        McpConnection::Disconnected(DisconnectedMcpConnection {
            server: server_state,
        })
    }

    /// Establish a connection to an MCP server.
    ///
    /// Creates the transport, performs the MCP handshake, and discovers
    /// tools, resources, and resource templates.
    ///
    /// Returns the client, transport, discovered tools/resources/templates,
    /// and server instructions.
    async fn establish_connection(
        &self,
        name: &str,
        config: &ValidatedServerConfig,
        _source: McpSource,
    ) -> McpResult<(
        McpClient,
        Box<dyn crate::transport::McpTransport>,
        Vec<McpTool>,
        Vec<roo_types::mcp::McpResource>,
        Vec<roo_types::mcp::McpResourceTemplate>,
        Option<String>,
    )> {
        let mut transport: Box<dyn McpTransport> = match config {
            ValidatedServerConfig::Stdio {
                command,
                args,
                env,
                cwd,
                ..
            } => {
                let mut t = StdioTransport::new(
                    command.clone(),
                    args.clone(),
                    env.clone(),
                    cwd.clone(),
                );
                t.connect().await?;
                Box::new(t)
            }
            ValidatedServerConfig::Sse { url, headers, .. } => {
                let mut t = SseTransport::new(url.clone(), headers.clone());
                t.connect().await?;
                Box::new(t)
            }
            ValidatedServerConfig::StreamableHttp { url, headers, .. } => {
                let mut t =
                    StreamableHttpTransport::new(url.clone(), headers.clone());
                t.connect().await?;
                Box::new(t)
            }
        };

        // Create client and initialize
        let mut client = McpClient::new("Roo Code", "1.0.0");
        let init_result = client.initialize(&mut *transport).await?;

        // Discover tools, resources, and templates
        let raw_tools = client.list_tools(&mut *transport).await.unwrap_or_default();
        let resources = client
            .list_resources(&mut *transport)
            .await
            .unwrap_or_default();
        let resource_templates = client
            .list_resource_templates(&mut *transport)
            .await
            .unwrap_or_default();

        // Read alwaysAllow and disabledTools from config to annotate tools
        let always_allow_config = config.always_allow();
        let disabled_tools_list = config.disabled_tools();
        let has_wildcard = always_allow_config.contains(&"*".to_string());

        // Annotate tools with always_allow and enabled_for_prompt
        let tools: Vec<McpTool> = raw_tools
            .into_iter()
            .map(|mut tool| {
                tool.always_allow =
                    has_wildcard || always_allow_config.contains(&tool.name);
                tool.enabled_for_prompt = !disabled_tools_list.contains(&tool.name);
                tool
            })
            .collect();

        let instructions = init_result.instructions;

        tracing::info!(
            "Established connection to '{}': {} tools, {} resources, {} templates",
            name,
            tools.len(),
            resources.len(),
            resource_templates.len()
        );

        Ok((client, transport, tools, resources, resource_templates, instructions))
    }
}

impl Default for McpHub {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper trait for finding connections in a Vec
// ---------------------------------------------------------------------------

/// Extension trait for searching connections in a Vec<McpConnection>.
pub trait McpConnectionExt {
    /// Find a connection by name and source.
    fn find(&self, name: &str, source: McpSource) -> Option<&McpConnection>;
    /// Find a mutable connection by name and source.
    fn find_mut(&mut self, name: &str, source: McpSource) -> Option<&mut McpConnection>;
    /// Find a connected connection by name (any source).
    fn find_connected_mut(&mut self, name: &str) -> Option<&mut McpConnection>;
}

impl McpConnectionExt for Vec<McpConnection> {
    fn find(&self, name: &str, source: McpSource) -> Option<&McpConnection> {
        self.iter()
            .find(|conn| conn.name() == name && conn.source() == source)
    }

    fn find_mut(&mut self, name: &str, source: McpSource) -> Option<&mut McpConnection> {
        self.iter_mut()
            .find(|conn| conn.name() == name && conn.source() == source)
    }

    fn find_connected_mut(&mut self, name: &str) -> Option<&mut McpConnection> {
        let sanitized = sanitize_mcp_name(name);

        // Try exact match first, then fuzzy match
        self.iter_mut().find(|conn| {
            (conn.name() == name || tool_names_match(conn.name(), &sanitized))
                && conn.is_connected()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hub_creation() {
        let hub = McpHub::new();
        let servers = hub.get_servers();
        assert!(servers.is_empty());
    }

    #[tokio::test]
    async fn test_register_unregister_client() {
        let hub = McpHub::new();
        hub.register_client().await;
        hub.register_client().await;

        let count = *hub.ref_count.read().await;
        assert_eq!(count, 2);

        hub.unregister_client().await.unwrap();
        let count = *hub.ref_count.read().await;
        assert_eq!(count, 1);

        // Last unregister should dispose
        hub.unregister_client().await.unwrap();
        let disposed = *hub.disposed.read().await;
        assert!(disposed);
    }

    #[tokio::test]
    async fn test_find_server_name_by_sanitized() {
        let hub = McpHub::new();
        // No connections yet
        let result = hub.find_server_name_by_sanitized_name("test").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_mcp_enabled_change() {
        let hub = McpHub::new();
        hub.handle_mcp_enabled_change(false).await.unwrap();
        assert!(!*hub.mcp_enabled.read().await);

        hub.handle_mcp_enabled_change(true).await.unwrap();
        assert!(*hub.mcp_enabled.read().await);
    }

    #[tokio::test]
    async fn test_dispose() {
        let hub = McpHub::new();
        hub.dispose().await.unwrap();
        assert!(*hub.disposed.read().await);

        // Operations should fail after dispose
        let result = hub.connect_to_server(
            "test",
            &serde_json::json!({"command": "echo"}),
            McpSource::Global,
        ).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_servers_empty() {
        let hub = McpHub::new();
        let servers = hub.get_all_servers().await;
        assert!(servers.is_empty());
    }

    #[test]
    fn test_connection_vec_find() {
        let mut connections: Vec<McpConnection> = Vec::new();

        let server = McpServerState::new(
            "test-server".to_string(),
            "{}".to_string(),
            McpSource::Global,
        );
        connections.push(McpConnection::Disconnected(DisconnectedMcpConnection {
            server,
        }));

        let found = connections.find("test-server", McpSource::Global);
        assert!(found.is_some());

        let not_found = connections.find("test-server", McpSource::Project);
        assert!(not_found.is_none());

        let not_found2 = connections.find("other-server", McpSource::Global);
        assert!(not_found2.is_none());
    }

    #[tokio::test]
    async fn test_toggle_tool_always_allow() {
        let hub = McpHub::new();

        // Create a disconnected connection with a tool
        let mut server = McpServerState::new(
            "test-server".to_string(),
            r#"{"type":"stdio","command":"echo","alwaysAllow":[],"disabledTools":[]}"#.to_string(),
            McpSource::Global,
        );
        server.tools = vec![McpTool {
            name: "my-tool".to_string(),
            description: None,
            input_schema: None,
            always_allow: false,
            enabled_for_prompt: true,
        }];
        hub.connections.write().await.push(
            McpConnection::Disconnected(DisconnectedMcpConnection { server }),
        );

        // Toggle always-allow on
        hub.toggle_tool_always_allow("test-server", McpSource::Global, "my-tool", true)
            .await
            .unwrap();

        // Verify the tool was updated
        let connections = hub.connections.read().await;
        let conn = connections.find("test-server", McpSource::Global).unwrap();
        let tool = conn.server().tools.iter().find(|t| t.name == "my-tool").unwrap();
        assert!(tool.always_allow);

        // Verify config was updated
        let config: serde_json::Value = serde_json::from_str(&conn.server().config).unwrap();
        let always_allow = config["alwaysAllow"].as_array().unwrap();
        assert!(always_allow.iter().any(|v| v.as_str() == Some("my-tool")));
    }

    #[tokio::test]
    async fn test_toggle_tool_always_allow_off() {
        let hub = McpHub::new();

        // Create a disconnected connection with a tool that is already allowed
        let mut server = McpServerState::new(
            "test-server".to_string(),
            r#"{"type":"stdio","command":"echo","alwaysAllow":["existing-tool"],"disabledTools":[]}"#.to_string(),
            McpSource::Global,
        );
        server.tools = vec![McpTool {
            name: "existing-tool".to_string(),
            description: None,
            input_schema: None,
            always_allow: true,
            enabled_for_prompt: true,
        }];
        hub.connections.write().await.push(
            McpConnection::Disconnected(DisconnectedMcpConnection { server }),
        );

        // Toggle always-allow off
        hub.toggle_tool_always_allow("test-server", McpSource::Global, "existing-tool", false)
            .await
            .unwrap();

        // Verify the tool was updated
        let connections = hub.connections.read().await;
        let conn = connections.find("test-server", McpSource::Global).unwrap();
        let tool = conn.server().tools.iter().find(|t| t.name == "existing-tool").unwrap();
        assert!(!tool.always_allow);

        // Verify config was updated
        let config: serde_json::Value = serde_json::from_str(&conn.server().config).unwrap();
        let always_allow = config["alwaysAllow"].as_array().unwrap();
        assert!(!always_allow.iter().any(|v| v.as_str() == Some("existing-tool")));
    }

    #[tokio::test]
    async fn test_toggle_tool_enabled_for_prompt_disable() {
        let hub = McpHub::new();

        // Create a disconnected connection with a tool
        let mut server = McpServerState::new(
            "test-server".to_string(),
            r#"{"type":"stdio","command":"echo","alwaysAllow":[],"disabledTools":[]}"#.to_string(),
            McpSource::Global,
        );
        server.tools = vec![McpTool {
            name: "my-tool".to_string(),
            description: None,
            input_schema: None,
            always_allow: false,
            enabled_for_prompt: true,
        }];
        hub.connections.write().await.push(
            McpConnection::Disconnected(DisconnectedMcpConnection { server }),
        );

        // Disable the tool for prompts
        hub.toggle_tool_enabled_for_prompt("test-server", McpSource::Global, "my-tool", false)
            .await
            .unwrap();

        // Verify the tool was updated
        let connections = hub.connections.read().await;
        let conn = connections.find("test-server", McpSource::Global).unwrap();
        let tool = conn.server().tools.iter().find(|t| t.name == "my-tool").unwrap();
        assert!(!tool.enabled_for_prompt);

        // Verify config was updated (tool added to disabledTools)
        let config: serde_json::Value = serde_json::from_str(&conn.server().config).unwrap();
        let disabled_tools = config["disabledTools"].as_array().unwrap();
        assert!(disabled_tools.iter().any(|v| v.as_str() == Some("my-tool")));
    }

    #[tokio::test]
    async fn test_toggle_tool_enabled_for_prompt_enable() {
        let hub = McpHub::new();

        // Create a disconnected connection with a disabled tool
        let mut server = McpServerState::new(
            "test-server".to_string(),
            r#"{"type":"stdio","command":"echo","alwaysAllow":[],"disabledTools":["existing-tool"]}"#.to_string(),
            McpSource::Global,
        );
        server.tools = vec![McpTool {
            name: "existing-tool".to_string(),
            description: None,
            input_schema: None,
            always_allow: false,
            enabled_for_prompt: false,
        }];
        hub.connections.write().await.push(
            McpConnection::Disconnected(DisconnectedMcpConnection { server }),
        );

        // Re-enable the tool for prompts
        hub.toggle_tool_enabled_for_prompt("test-server", McpSource::Global, "existing-tool", true)
            .await
            .unwrap();

        // Verify the tool was updated
        let connections = hub.connections.read().await;
        let conn = connections.find("test-server", McpSource::Global).unwrap();
        let tool = conn.server().tools.iter().find(|t| t.name == "existing-tool").unwrap();
        assert!(tool.enabled_for_prompt);

        // Verify config was updated (tool removed from disabledTools)
        let config: serde_json::Value = serde_json::from_str(&conn.server().config).unwrap();
        let disabled_tools = config["disabledTools"].as_array().unwrap();
        assert!(!disabled_tools.iter().any(|v| v.as_str() == Some("existing-tool")));
    }

    #[tokio::test]
    async fn test_toggle_tool_server_not_found() {
        let hub = McpHub::new();

        let result = hub
            .toggle_tool_always_allow("nonexistent", McpSource::Global, "tool", true)
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_get_servers_extracts_disabled_tools() {
        let mut connections: Vec<McpConnection> = Vec::new();

        let mut server = McpServerState::new(
            "test-server".to_string(),
            "{}".to_string(),
            McpSource::Global,
        );
        server.tools = vec![
            McpTool {
                name: "enabled-tool".to_string(),
                description: None,
                input_schema: None,
                always_allow: false,
                enabled_for_prompt: true,
            },
            McpTool {
                name: "disabled-tool".to_string(),
                description: None,
                input_schema: None,
                always_allow: false,
                enabled_for_prompt: false,
            },
        ];
        connections.push(McpConnection::Disconnected(DisconnectedMcpConnection {
            server,
        }));

        // Verify disabled_tools is extracted from enabled_for_prompt flags
        let server_conn = McpServerConnection {
            name: connections[0].name().to_string(),
            status: connections[0].server().status,
            tools: connections[0].server().tools.clone(),
            resources: connections[0].server().resources.clone(),
            resource_templates: connections[0].server().resource_templates.clone(),
            disabled_tools: connections[0]
                .server()
                .tools
                .iter()
                .filter(|t| !t.enabled_for_prompt)
                .map(|t| t.name.clone())
                .collect(),
            errors: connections[0].server().error_history.clone(),
        };

        assert_eq!(server_conn.disabled_tools, vec!["disabled-tool"]);
    }
}
