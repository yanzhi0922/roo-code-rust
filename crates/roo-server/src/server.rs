//! Main server struct.
//!
//! Source: `src/core/webview/ClineProvider.ts` — ClineProvider class
//!
//! The `Server` struct is the top-level entry point for the JSON-RPC server.
//! It ties together the transport, router, and handler layers.

use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{error, info, instrument, warn};

use roo_app::App;

use crate::error::ServerResult;
use crate::handler::Handler;
use crate::router::Router;
use crate::transport::Transport;

// ---------------------------------------------------------------------------
// Server error codes (application-specific, -32000 to -32099)
// ---------------------------------------------------------------------------

/// Server error code range start (per JSON-RPC spec, -32000 to -32099 are reserved for implementation-defined errors)
pub const SERVER_ERROR_START: i64 = -32000;

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The Roo Code JSON-RPC server.
///
/// This is the main entry point for running the Roo Code server. It manages
/// the lifecycle of the server, including initialization, request processing,
/// and shutdown.
///
/// # Example
///
/// ```rust,ignore
/// use roo_server::Server;
/// use roo_app::{App, AppConfig};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let app = App::new(AppConfig::default());
///     let server = Server::new(app);
///     server.serve_stdio().await?;
///     Ok(())
/// }
/// ```
pub struct Server {
    app: Arc<App>,
    initialized: Arc<RwLock<bool>>,
    shut_down: Arc<RwLock<bool>>,
}

impl Server {
    /// Create a new server wrapping the given App.
    pub fn new(app: App) -> Self {
        Self {
            app: Arc::new(app),
            initialized: Arc::new(RwLock::new(false)),
            shut_down: Arc::new(RwLock::new(false)),
        }
    }

    /// Get a reference to the underlying App.
    pub fn app(&self) -> &App {
        &self.app
    }

    /// Check if the server has been initialized.
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Check if the server has been shut down.
    pub async fn is_shut_down(&self) -> bool {
        *self.shut_down.read().await
    }

    /// Run the server using stdin/stdout transport.
    ///
    /// This is the primary way to run the server for IPC communication.
    /// It reads Content-Length framed JSON-RPC messages from stdin and
    /// writes responses to stdout.
    #[instrument(skip(self))]
    pub async fn serve_stdio(&self) -> ServerResult<()> {
        info!("Starting Roo Code server (stdio transport)");

        let handler = Handler::new(self.app.clone());
        let router = Router::new(handler);
        let mut transport = crate::transport::StdioTransport::new();

        self.run_message_loop(&router, &mut transport).await
    }

    /// Run the server using a custom transport.
    ///
    /// This allows using different transport implementations (e.g., TCP,
    /// WebSocket, or in-memory channels for testing).
    #[instrument(skip(self, router, transport))]
    pub async fn serve_with_transport<T: Transport>(
        &self,
        router: &Router,
        transport: &mut T,
    ) -> ServerResult<()> {
        info!("Starting Roo Code server (custom transport)");
        self.run_message_loop(router, transport).await
    }

    /// The main message processing loop.
    ///
    /// Reads messages from the transport, routes them through the router,
    /// and sends responses back through the transport.
    async fn run_message_loop<T: Transport>(
        &self,
        router: &Router,
        transport: &mut T,
    ) -> ServerResult<()> {
        loop {
            match transport.receive().await {
                Ok(Some(message)) => {
                    // Check for shutdown
                    if *self.shut_down.read().await {
                        warn!("Received message after shutdown, ignoring");
                        continue;
                    }

                    // Route the message
                    let response = router.route(&message).await;

                    // Check if this was an initialize request
                    if let Some(method) = &message.method {
                        if method == crate::handler::methods::INITIALIZE {
                            *self.initialized.write().await = true;
                        } else if method == crate::handler::methods::SHUTDOWN {
                            *self.shut_down.write().await = true;
                        }
                    }

                    // Send the response
                    if let Err(e) = transport.send(&response).await {
                        error!(error = %e, "Failed to send response");
                        break;
                    }

                    // If we just shut down, close the transport
                    if *self.shut_down.read().await {
                        info!("Server shutting down");
                        transport.close().await?;
                        break;
                    }
                }
                Ok(None) => {
                    info!("Transport closed (EOF)");
                    break;
                }
                Err(e) => {
                    error!(error = %e, "Transport receive error");
                    break;
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handler::Handler;
    use crate::router::Router;
    use crate::transport::MemoryTransport;
    use roo_app::AppConfig;
    use roo_jsonrpc::types::Message;
    use serde_json::json;

    fn test_server() -> Server {
        let config = AppConfig {
            cwd: "/tmp/test".to_string(),
            mode: "code".to_string(),
            ..Default::default()
        };
        let app = App::new(config);
        Server::new(app)
    }

    #[tokio::test]
    async fn test_server_creation() {
        let server = test_server();
        assert!(!server.is_initialized().await);
        assert!(!server.is_shut_down().await);
    }

    #[tokio::test]
    async fn test_server_initialize_and_shutdown() {
        let server = test_server();
        let handler = Handler::new(server.app.clone());
        let router = Router::new(handler);
        let (client, mut transport) = MemoryTransport::new(10);

        // Send initialize
        let init_request = Message::request(1, "initialize", json!(null));
        client.send(init_request).await.unwrap();

        // Process in background
        let _server_clone = server.app.clone();
        let initialized = server.initialized.clone();
        let shut_down = server.shut_down.clone();

        let handle = tokio::spawn(async move {
            // Receive and process
            let msg = transport.receive().await.unwrap().unwrap();
            let response = router.route(&msg).await;
            transport.send(&response).await.unwrap();

            // Update state
            if msg.method.as_deref() == Some("initialize") {
                *initialized.write().await = true;
            }
            if msg.method.as_deref() == Some("shutdown") {
                *shut_down.write().await = true;
            }
        });

        handle.await.unwrap();

        // Read response
        let mut client = client;
        let response = client.receive().await.unwrap().unwrap();
        assert_eq!(response.id_as_u64(), Some(1));
        assert!(response.result.is_some());
    }

    #[tokio::test]
    async fn test_server_app_accessible() {
        let server = test_server();
        assert_eq!(server.app().cwd(), "/tmp/test");
    }
}
