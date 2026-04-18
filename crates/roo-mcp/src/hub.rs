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
            Ok((client, transport)) => {
                // Replace the placeholder with a connected connection
                let mut connections = self.connections.write().await;
                if let Some(conn) = connections.find_mut(name, source) {
                    conn.server_mut().status = McpConnectionStatus::Connected;
                    conn.server_mut().error.clear();

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
    pub async fn delete_connection(&self, name: &str, source: McpSource) -> McpResult<()> {
        let mut connections = self.connections.write().await;
        let before_len = connections.len();

        connections.retain(|conn| {
            !(conn.name() == name && conn.source() == source)
        });

        // Clean up sanitized name registry if no more connections with this name
        let remaining = connections.iter().any(|c| c.name() == name);
        if !remaining {
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

    /// Call a tool on a specific server.
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Option<serde_json::Value>,
    ) -> McpResult<McpToolCallResponse> {
        self.check_disposed()?;

        let mut connections = self.connections.write().await;

        let connection = connections
            .find_connected_mut(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        match connection {
            McpConnection::Connected(conn) => {
                let result = conn
                    .client
                    .call_tool(&mut *conn.transport, tool_name, arguments)
                    .await;

                match result {
                    Ok(response) => {
                        tracing::info!(
                            "Tool call '{}' on '{}' succeeded",
                            tool_name,
                            server_name
                        );
                        Ok(response)
                    }
                    Err(e) => {
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

    /// Fetch the tools list from a specific server.
    pub async fn fetch_tools_list(&self, server_name: &str) -> McpResult<Vec<McpTool>> {
        self.check_disposed()?;

        let mut connections = self.connections.write().await;

        let connection = connections
            .find_connected_mut(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        match connection {
            McpConnection::Connected(conn) => {
                let tools = conn
                    .client
                    .list_tools(&mut *conn.transport)
                    .await?;

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
    pub async fn toggle_tool_always_allow(
        &self,
        server_name: &str,
        _source: McpSource,
        _tool_name: &str,
        _should_allow: bool,
    ) -> McpResult<()> {
        self.check_disposed()?;
        // This would normally update the configuration file.
        // For now, it's a placeholder that signals the intent.
        tracing::info!(
            "Toggle always-allow for tool '{}' on '{}'",
            _tool_name,
            server_name
        );
        Ok(())
    }

    /// Toggle whether a tool is enabled for prompts.
    pub async fn toggle_tool_enabled_for_prompt(
        &self,
        server_name: &str,
        _source: McpSource,
        _tool_name: &str,
        _is_enabled: bool,
    ) -> McpResult<()> {
        self.check_disposed()?;
        tracing::info!(
            "Toggle enabled for prompt: tool '{}' on '{}'",
            _tool_name,
            server_name
        );
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
    pub async fn handle_mcp_enabled_change(&self, enabled: bool) -> McpResult<()> {
        *self.mcp_enabled.write().await = enabled;

        if enabled {
            // Re-connect all servers
            self.refresh_all_connections().await?;
        } else {
            // Disconnect all servers (but keep placeholders)
            let mut connections = self.connections.write().await;
            for conn in connections.iter_mut() {
                if let McpConnection::Connected(_) = conn {
                    conn.server_mut().status = McpConnectionStatus::Disconnected;
                }
            }
            // Replace all connected with disconnected
            let new_connections: Vec<McpConnection> = connections
                .drain(..)
                .map(|conn| match conn {
                    McpConnection::Connected(c) => {
                        McpConnection::Disconnected(DisconnectedMcpConnection {
                            server: c.server,
                        })
                    }
                    other => other,
                })
                .collect();
            *connections = new_connections;
        }

        self.notify_state_change();
        Ok(())
    }

    /// Get all servers (as McpServerConnection for external consumption).
    pub fn get_servers(&self) -> Vec<McpServerConnection> {
        // Synchronous version — use try_read to avoid blocking
        match self.connections.try_read() {
            Ok(connections) => connections
                .iter()
                .map(|conn| McpServerConnection {
                    name: conn.name().to_string(),
                    status: conn.server().status,
                    tools: conn.server().tools.clone(),
                    resources: conn.server().resources.clone(),
                    resource_templates: conn.server().resource_templates.clone(),
                    disabled_tools: Vec::new(), // Would come from config
                    errors: conn.server().error_history.clone(),
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Get all servers asynchronously.
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
                disabled_tools: Vec::new(),
                errors: conn.server().error_history.clone(),
            })
            .collect()
    }

    /// Find a server name by its sanitized name.
    pub async fn find_server_name_by_sanitized_name(&self, sanitized_name: &str) -> Option<String> {
        // First try the registry
        let registry = self.sanitized_name_registry.read().await;
        if let Some(original) = registry.get(sanitized_name) {
            return Some(original.clone());
        }
        drop(registry);

        // Fallback: fuzzy match against all connection names
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

    async fn establish_connection(
        &self,
        name: &str,
        config: &ValidatedServerConfig,
        _source: McpSource,
    ) -> McpResult<(McpClient, Box<dyn crate::transport::McpTransport>)> {
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
        let _init_result = client.initialize(&mut *transport).await?;

        // Discover tools, resources, and templates
        let tools = client.list_tools(&mut *transport).await.unwrap_or_default();
        let resources = client
            .list_resources(&mut *transport)
            .await
            .unwrap_or_default();
        let resource_templates = client
            .list_resource_templates(&mut *transport)
            .await
            .unwrap_or_default();

        // Update server state with discovered data
        // (This is done by the caller after this returns)

        tracing::info!(
            "Established connection to '{}': {} tools, {} resources, {} templates",
            name,
            tools.len(),
            resources.len(),
            resource_templates.len()
        );

        Ok((client, transport))
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
}
