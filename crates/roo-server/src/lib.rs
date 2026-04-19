//! # Roo Server — JSON-RPC Server for Roo Code
//!
//! A JSON-RPC 2.0 server that provides IPC communication for Roo Code.
//!
//! ## Architecture
//!
//! The server wraps the [`roo_app::App`] application layer and exposes its
//! functionality through JSON-RPC 2.0 methods. This enables communication
//! between a client (CLI, TUI, or WebView) and the Roo Code backend.
//!
//! ## Source Mapping
//!
//! - TypeScript: `src/core/webview/webviewMessageHandler.ts` — WebviewMessage handler
//! - TypeScript: `packages/types/src/ipc.ts` — IPC message types
//! - TypeScript: `packages/types/src/vscode-extension-host.ts` — ExtensionMessage/WebviewMessage
//!
//! ## JSON-RPC Method Mapping
//!
//! | JSON-RPC Method           | TypeScript Equivalent                    |
//! |---------------------------|------------------------------------------|
//! | `initialize`              | `webviewDidLaunch`                       |
//! | `shutdown`                | `dispose`                                |
//! | `task/start`              | `newTask` / `StartNewTask`              |
//! | `task/cancel`             | `cancelTask` / `CancelTask`             |
//! | `task/close`              | `clearTask` / `CloseTask`               |
//! | `task/resume`             | `ResumeTask`                             |
//! | `task/sendMessage`        | `SendMessage`                            |
//! | `task/getCommands`        | `GetCommands`                            |
//! | `task/getModes`           | `GetModes`                               |
//! | `task/getModels`          | `GetModels`                              |
//! | `state/get`               | `getState`                               |
//! | `state/setMode`           | `mode`                                   |
//! | `systemPrompt/build`      | `getSystemPrompt`                        |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use roo_server::Server;
//! use roo_app::{App, AppConfig};
//!
//! let app = App::new(AppConfig::default());
//! let server = Server::new(app);
//! server.serve_stdio().await?;
//! ```

pub mod error;
pub mod handler;
pub mod router;
pub mod server;
pub mod transport;

pub use error::ServerResult;
pub use router::Router;
pub use server::Server;
pub use transport::Transport;
