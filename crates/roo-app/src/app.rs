//! Main application controller.
//!
//! Source: `src/core/ClineProvider.ts` — ClineProvider class
//!
//! The `App` struct is the top-level coordinator that manages all Roo Code subsystems.
//! It provides the high-level API for creating tasks, managing providers, and handling
//! the overall lifecycle of the application.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::AppConfig;
use crate::error::AppResult;
use crate::state::{AppState, SharedState};

/// The main Roo Code application controller.
///
/// Source: `src/core/ClineProvider.ts` — `ClineProvider` class
///
/// This struct coordinates all subsystems:
/// - Provider management (AI model connections)
/// - Task lifecycle (create, run, cancel)
/// - MCP server connections
/// - Configuration management
/// - Skills management
/// - Checkpoint system
/// - Terminal management
/// - Telemetry
pub struct App {
    /// Application configuration.
    config: AppConfig,

    /// Shared application state.
    state: SharedState,
}

impl App {
    /// Create a new App instance with the given configuration.
    ///
    /// Source: `src/core/ClineProvider.ts` — constructor
    pub fn new(config: AppConfig) -> Self {
        let state = Arc::new(RwLock::new(AppState::new()));
        Self { config, state }
    }

    /// Initialize the application.
    ///
    /// This sets up all subsystems, loads configuration, and prepares
    /// the application for use.
    ///
    /// Source: `src/core/ClineProvider.ts` — initialization logic
    pub async fn initialize(&self) -> AppResult<()> {
        let mut state = self.state.write().await;

        // Initialize the roo-ignore controller for the cwd
        // In a real VS Code extension, this would set up file watchers
        tracing::info!(
            "Initializing Roo Code App in workspace: {}",
            self.config.cwd
        );

        state.current_mode = self.config.mode.clone();
        state.initialized = true;

        tracing::info!("App initialized with mode: {}", state.current_mode);
        Ok(())
    }

    /// Get the current working directory.
    pub fn cwd(&self) -> &str {
        &self.config.cwd
    }

    /// Get the current mode.
    pub async fn mode(&self) -> String {
        self.state.read().await.current_mode.clone()
    }

    /// Set the current mode.
    pub async fn set_mode(&self, mode: &str) {
        let mut state = self.state.write().await;
        tracing::info!("Switching mode from {} to {}", state.current_mode, mode);
        state.current_mode = mode.to_string();
    }

    /// Get a reference to the provider settings.
    pub fn provider_settings(&self) -> &roo_types::provider_settings::ProviderSettings {
        &self.config.provider_settings
    }

    /// Get a reference to the app config.
    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    /// Get a snapshot of the current application state.
    pub async fn state(&self) -> AppState {
        self.state.read().await.clone()
    }

    /// Check if the app has been disposed.
    pub async fn is_disposed(&self) -> bool {
        self.state.read().await.disposed
    }

    /// Dispose of the application and clean up resources.
    ///
    /// Source: `src/core/ClineProvider.ts` — dispose logic
    pub async fn dispose(&self) -> AppResult<()> {
        let mut state = self.state.write().await;
        state.disposed = true;
        state.task_running = false;
        tracing::info!("App disposed");
        Ok(())
    }

    /// Build the system prompt for the current mode and configuration.
    ///
    /// This uses the roo-prompt crate to generate the complete system prompt
    /// based on the current mode, provider settings, and other configuration.
    pub fn build_system_prompt(&self) -> String {
        let settings = roo_prompt::types::SystemPromptSettings {
            is_stealth_model: false,
            ..Default::default()
        };

        roo_prompt::build_system_prompt(
            &self.config.cwd,
            &self.config.mode,
            None,   // custom_modes
            None,   // custom_mode_prompts
            false,  // has_mcp
            None,   // global_custom_instructions
            self.config.language.as_deref(),
            None,   // roo_ignore_instructions
            Some(&settings),
            &[],    // skills
            &format!("{} {}", std::env::consts::OS, env!("CARGO_PKG_VERSION")),
            "bash", // shell
            &std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| "~".to_string()),
        )
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Best-effort cleanup
        if let Ok(state) = self.state.try_write() {
            drop(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_creation() {
        let config = AppConfig::default();
        let app = App::new(config);
        assert_eq!(app.cwd(), std::env::current_dir().unwrap().to_str().unwrap());
    }

    #[tokio::test]
    async fn test_app_initialize() {
        let config = AppConfig {
            cwd: "/tmp/test".to_string(),
            mode: "code".to_string(),
            ..Default::default()
        };
        let app = App::new(config);
        app.initialize().await.unwrap();

        let state = app.state().await;
        assert!(state.initialized);
        assert_eq!(state.current_mode, "code");
    }

    #[tokio::test]
    async fn test_set_mode() {
        let config = AppConfig {
            cwd: "/tmp/test".to_string(),
            mode: "code".to_string(),
            ..Default::default()
        };
        let app = App::new(config);
        app.initialize().await.unwrap();

        app.set_mode("architect").await;
        assert_eq!(app.mode().await, "architect");
    }

    #[tokio::test]
    async fn test_dispose() {
        let config = AppConfig::default();
        let app = App::new(config);
        app.initialize().await.unwrap();

        app.dispose().await.unwrap();
        assert!(app.is_disposed().await);
    }

    #[test]
    fn test_build_system_prompt() {
        let config = AppConfig {
            cwd: "/tmp/test-project".to_string(),
            mode: "code".to_string(),
            ..Default::default()
        };
        let app = App::new(config);
        let prompt = app.build_system_prompt();

        assert!(prompt.contains("TOOL USE"));
        assert!(prompt.contains("RULES"));
        assert!(prompt.contains("OBJECTIVE"));
        assert!(prompt.contains("CAPABILITIES"));
        assert!(prompt.contains("SYSTEM INFORMATION"));
    }

    #[test]
    fn test_config_default() {
        let config = AppConfig::default();
        assert_eq!(config.mode, "code");
        assert!(config.telemetry_enabled == false);
        assert!(config.checkpoints_enabled);
    }
}
