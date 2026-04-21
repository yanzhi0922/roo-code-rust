//! # roo-config — Configuration Management for Roo Code
//!
//! This crate provides configuration directory resolution, file system utilities,
//! configuration loading with merge support, settings import/export,
//! provider settings management, and settings migration.

pub mod error;
pub mod filesystem;
pub mod import_export;
pub mod loader;
pub mod migrate_settings;
pub mod network_proxy;
pub mod paths;
pub mod provider_settings_manager;
pub mod safe_write_json;

// Re-export key types and functions
pub use error::ConfigError;
pub use filesystem::{directory_exists, file_exists, read_file_if_exists};
pub use import_export::{
    sanitize_provider_config, import_settings_from_path, export_settings,
    ImportExportError, ImportResult, ProviderProfiles, ExportData,
};
pub use loader::{
    load_configuration, load_roo_configuration, build_merged_content, LoadedConfiguration,
};
pub use migrate_settings::{
    migrate_settings, migrate_custom_modes_to_yaml, migrate_default_commands,
    default_file_migrations, MigrationError, FileMigration,
};
pub use network_proxy::{
    NetworkProxy, ProxyConfig, ProxyProtocol, redact_proxy_url,
};
pub use paths::{
    discover_subfolder_roo_directories, get_agents_directories_for_cwd,
    get_all_roo_directories_for_cwd, get_global_agents_directory, get_global_roo_directory,
    get_project_agents_directory_for_cwd, get_project_roo_directory_for_cwd,
    get_roo_directories_for_cwd,
};
pub use provider_settings_manager::{
    ProviderSettingsManager, ProviderSettingsError, ProviderSettingsWithId,
    ProviderProfiles as ProviderSettingsProfiles, MigrationState,
};
pub use safe_write_json::{safe_write_json, SafeWriteJsonError, SafeWriteJsonOptions};
