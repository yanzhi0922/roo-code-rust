//! # roo-config — Configuration Management for Roo Code
//!
//! This crate provides configuration directory resolution, file system utilities,
//! and configuration loading with merge support for the Roo Code Rust rewrite.
//!
//! ## Module Structure
//!
//! - [`paths`] — Path resolution for `.roo` and `.agents` directories
//! - [`filesystem`] — Async file system utilities (exists checks, safe reads)
//! - [`loader`] — Configuration loading with global/project merge support
//! - [`error`] — Error types for configuration operations
//!
//! ## Maps to TypeScript Source
//!
//! `src/services/roo-config/index.ts`

pub mod error;
pub mod filesystem;
pub mod loader;
pub mod paths;

// Re-export key types and functions
pub use error::ConfigError;
pub use filesystem::{directory_exists, file_exists, read_file_if_exists};
pub use loader::{
    load_configuration, load_roo_configuration, build_merged_content, LoadedConfiguration,
};
pub use paths::{
    discover_subfolder_roo_directories, get_agents_directories_for_cwd,
    get_all_roo_directories_for_cwd, get_global_agents_directory, get_global_roo_directory,
    get_project_agents_directory_for_cwd, get_project_roo_directory_for_cwd,
    get_roo_directories_for_cwd,
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[tokio::test]
    async fn test_full_config_workflow() {
        // Create temp project structure
        let tmp = tempfile::tempdir().unwrap();

        // Create project .roo/rules/rules.md
        let project_rules_dir = tmp.path().join(".roo").join("rules");
        fs::create_dir_all(&project_rules_dir).unwrap();
        fs::write(
            project_rules_dir.join("rules.md"),
            "Always use TypeScript\nUse strict mode",
        )
        .unwrap();

        // Create subfolder .roo
        let sub_roo = tmp.path().join("packages").join("core").join(".roo");
        fs::create_dir_all(&sub_roo).unwrap();

        // Test path resolution
        let dirs = get_roo_directories_for_cwd(tmp.path());
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[1], tmp.path().join(".roo"));

        // Test file existence
        assert!(file_exists(&project_rules_dir.join("rules.md")).await);
        assert!(!file_exists(&project_rules_dir.join("nonexistent.md")).await);

        // Test directory existence
        assert!(directory_exists(&tmp.path().join(".roo")).await);
        assert!(!directory_exists(&tmp.path().join(".nonexistent")).await);

        // Test configuration loading
        let config = load_configuration(Path::new("rules/rules.md"), tmp.path())
            .await
            .unwrap();
        assert_eq!(
            config.project,
            Some("Always use TypeScript\nUse strict mode".to_string())
        );
        assert!(config.merged.contains("Use strict mode"));

        // Test subfolder discovery
        let subfolders = discover_subfolder_roo_directories(tmp.path())
            .await
            .unwrap();
        assert_eq!(subfolders.len(), 1);
        assert!(subfolders[0].ends_with("core/.roo"));
    }
}
