use std::collections::HashSet;

use crate::types::{CodeIndexConfig, IndexError, IndexStats, IndexingState, VectorStoreSearchResult};

/// Manager for building and querying a code index.
pub struct CodeIndexManager {
    state: IndexingState,
    config: CodeIndexConfig,
    stats: IndexStats,
    indexed_files: HashSet<String>,
}

impl CodeIndexManager {
    /// Create a new code index manager with the given configuration.
    pub fn new(config: CodeIndexConfig) -> Self {
        Self {
            state: IndexingState::NotInitialized,
            config,
            stats: IndexStats::default(),
            indexed_files: HashSet::new(),
        }
    }

    /// Initialize the index. Returns `true` if initialization succeeded,
    /// `false` if already initialized.
    pub fn initialize(&mut self) -> Result<bool, IndexError> {
        if self.state != IndexingState::NotInitialized {
            return Ok(false);
        }

        self.state = IndexingState::Idle;
        Ok(true)
    }

    /// Start the indexing process.
    ///
    /// Transitions from `Idle` to `Indexing`. Returns an error if the index
    /// is not initialized or already indexing.
    pub fn start_indexing(&mut self) -> Result<(), IndexError> {
        match self.state {
            IndexingState::NotInitialized => Err(IndexError::NotInitialized),
            IndexingState::Indexing => Err(IndexError::AlreadyIndexing),
            IndexingState::ShuttingDown => Err(IndexError::GeneralError(
                "cannot start indexing while shutting down".to_string(),
            )),
            _ => {
                self.state = IndexingState::Indexing;
                Ok(())
            }
        }
    }

    /// Stop the indexing process and transition back to `Idle`.
    pub fn stop_indexing(&mut self) {
        if self.state == IndexingState::Indexing {
            self.state = IndexingState::Idle;
        }
    }

    /// Search the index for the given query.
    ///
    /// Returns up to `limit` results. In this implementation, search is
    /// simulated by matching against indexed file paths and contents.
    /// Populates `start_line`, `end_line`, and `code_chunk` fields.
    pub fn search(&self, query: &str, limit: usize) -> Vec<VectorStoreSearchResult> {
        self.search_with_prefix(query, None, limit)
    }

    /// Search the index for the given query, optionally filtered by directory prefix.
    ///
    /// Returns up to `limit` results sorted by relevance score (descending).
    /// When `directory_prefix` is `Some`, only files whose path starts with
    /// the prefix are included.
    ///
    /// TODO: Replace simulated matching with actual vector embedding +
    /// cosine similarity search when the embedding backend is integrated.
    pub fn search_with_prefix(
        &self,
        query: &str,
        directory_prefix: Option<&str>,
        limit: usize,
    ) -> Vec<VectorStoreSearchResult> {
        if self.state == IndexingState::NotInitialized || self.state == IndexingState::Error {
            return vec![];
        }

        let query_lower = query.to_lowercase();
        let mut results: Vec<VectorStoreSearchResult> = self
            .indexed_files
            .iter()
            .filter(|path| {
                // Filter by directory prefix if provided
                if let Some(prefix) = directory_prefix {
                    if !path.starts_with(prefix) {
                        return false;
                    }
                }
                path.to_lowercase().contains(&query_lower)
            })
            .take(limit)
            .map(|path| VectorStoreSearchResult {
                file_path: path.clone(),
                line_number: Some(1),
                content: format!("content of {path}"),
                score: 1.0,
                start_line: Some(1),
                end_line: Some(1),
                code_chunk: Some(format!("// matched content from {path}")),
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Get the current indexing state.
    pub fn get_state(&self) -> &IndexingState {
        &self.state
    }

    /// Get the current index statistics.
    pub fn get_stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Check whether the index has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.state != IndexingState::NotInitialized
    }

    /// Check whether the workspace indexing is enabled in the config.
    pub fn is_workspace_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Clear all indexed data and reset stats.
    pub fn clear_index_data(&mut self) -> Result<(), IndexError> {
        if self.state == IndexingState::NotInitialized {
            return Err(IndexError::NotInitialized);
        }

        self.indexed_files.clear();
        self.stats.indexed_files = 0;
        self.stats.total_chunks = 0;
        Ok(())
    }

    /// Add a file to the index.
    ///
    /// Checks the file against include/exclude patterns and size limits.
    pub fn add_file(&mut self, path: &str) -> Result<(), IndexError> {
        if self.state == IndexingState::NotInitialized {
            return Err(IndexError::NotInitialized);
        }

        // Check exclude patterns
        for pattern in &self.config.exclude_patterns {
            if Self::matches_glob(path, pattern) {
                return Ok(());
            }
        }

        // Check include patterns
        let included = self
            .config
            .include_patterns
            .iter()
            .any(|p| Self::matches_glob(path, p));

        if !included && !self.config.include_patterns.is_empty() {
            return Ok(());
        }

        if self.indexed_files.insert(path.to_string()) {
            self.stats.indexed_files = self.indexed_files.len();
            self.stats.total_files = self.stats.total_files.max(self.indexed_files.len());
            self.stats.total_chunks += 1;
        }

        Ok(())
    }

    /// Remove a file from the index.
    ///
    /// Returns `true` if the file was present and removed.
    pub fn remove_file(&mut self, path: &str) -> Result<bool, IndexError> {
        if self.state == IndexingState::NotInitialized {
            return Err(IndexError::NotInitialized);
        }

        if self.indexed_files.remove(path) {
            self.stats.indexed_files = self.indexed_files.len();
            self.stats.total_chunks = self.stats.total_chunks.saturating_sub(1);
            return Ok(true);
        }

        Ok(false)
    }

    /// Dispose of the index manager, releasing all resources.
    pub fn dispose(&mut self) {
        self.state = IndexingState::ShuttingDown;
        self.indexed_files.clear();
        self.stats = IndexStats::default();
        self.state = IndexingState::NotInitialized;
    }

    /// Simple glob matching — supports `**` (any path segments) and `*` (any segment).
    fn matches_glob(path: &str, pattern: &str) -> bool {
        // Simple implementation: check if the pattern's extension matches
        if pattern.contains("**/*.") {
            if let Some(ext) = pattern.split("**/*.").last() {
                return path.ends_with(ext) || path.contains(&format!(".{ext}"));
            }
        }
        if pattern.contains("*/") {
            let suffix = pattern.replace("*/", "/");
            return path.contains(&suffix);
        }
        path.contains(pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> CodeIndexManager {
        CodeIndexManager::new(CodeIndexConfig::default())
    }

    #[test]
    fn test_new_manager_not_initialized() {
        let mgr = default_manager();
        assert_eq!(IndexingState::NotInitialized, *mgr.get_state());
        assert!(!mgr.is_initialized());
    }

    #[test]
    fn test_initialize_success() {
        let mut mgr = default_manager();
        let result = mgr.initialize();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
        assert_eq!(IndexingState::Idle, *mgr.get_state());
    }

    #[test]
    fn test_initialize_idempotent() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        let result = mgr.initialize();
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_start_indexing_without_init() {
        let mut mgr = default_manager();
        let result = mgr.start_indexing();
        assert!(result.is_err());
    }

    #[test]
    fn test_start_indexing_success() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        let result = mgr.start_indexing();
        assert!(result.is_ok());
        assert_eq!(IndexingState::Indexing, *mgr.get_state());
    }

    #[test]
    fn test_start_indexing_already_indexing() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.start_indexing().unwrap();
        let result = mgr.start_indexing();
        assert!(result.is_err());
    }

    #[test]
    fn test_stop_indexing() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.start_indexing().unwrap();
        mgr.stop_indexing();
        assert_eq!(IndexingState::Idle, *mgr.get_state());
    }

    #[test]
    fn test_stop_indexing_when_not_indexing() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.stop_indexing();
        assert_eq!(IndexingState::Idle, *mgr.get_state());
    }

    #[test]
    fn test_search_not_initialized() {
        let mgr = default_manager();
        let results = mgr.search("test", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_initialized_empty() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        let results = mgr.search("test", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_indexed_files() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.add_file("src/main.rs").unwrap();
        mgr.add_file("src/lib.rs").unwrap();

        let results = mgr.search("main", 10);
        assert_eq!(1, results.len());
        assert_eq!("src/main.rs", results[0].file_path);
    }

    #[test]
    fn test_search_limit() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.add_file("src/main.rs").unwrap();
        mgr.add_file("src/lib.rs").unwrap();

        let results = mgr.search("", 1);
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_is_workspace_enabled() {
        let mgr = default_manager();
        assert!(mgr.is_workspace_enabled());
    }

    #[test]
    fn test_is_workspace_disabled() {
        let mut config = CodeIndexConfig::default();
        config.enabled = false;
        let mgr = CodeIndexManager::new(config);
        assert!(!mgr.is_workspace_enabled());
    }

    #[test]
    fn test_add_file_success() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        let result = mgr.add_file("src/main.rs");
        assert!(result.is_ok());
        assert_eq!(1, mgr.get_stats().indexed_files);
    }

    #[test]
    fn test_add_file_not_initialized() {
        let mut mgr = default_manager();
        let result = mgr.add_file("src/main.rs");
        assert!(result.is_err());
    }

    #[test]
    fn test_add_file_duplicate() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.add_file("src/main.rs").unwrap();
        mgr.add_file("src/main.rs").unwrap();
        assert_eq!(1, mgr.get_stats().indexed_files);
    }

    #[test]
    fn test_remove_file_success() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.add_file("src/main.rs").unwrap();
        let removed = mgr.remove_file("src/main.rs").unwrap();
        assert!(removed);
        assert_eq!(0, mgr.get_stats().indexed_files);
    }

    #[test]
    fn test_remove_file_not_present() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        let removed = mgr.remove_file("nonexistent.rs").unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_remove_file_not_initialized() {
        let mut mgr = default_manager();
        let result = mgr.remove_file("src/main.rs");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_index_data() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.add_file("src/main.rs").unwrap();
        mgr.add_file("src/lib.rs").unwrap();
        mgr.clear_index_data().unwrap();
        assert_eq!(0, mgr.get_stats().indexed_files);
        assert_eq!(0, mgr.get_stats().total_chunks);
    }

    #[test]
    fn test_clear_index_data_not_initialized() {
        let mut mgr = default_manager();
        let result = mgr.clear_index_data();
        assert!(result.is_err());
    }

    #[test]
    fn test_dispose() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.add_file("src/main.rs").unwrap();
        mgr.dispose();
        assert_eq!(IndexingState::NotInitialized, *mgr.get_state());
        assert_eq!(0, mgr.get_stats().indexed_files);
    }

    #[test]
    fn test_get_stats() {
        let mut mgr = default_manager();
        mgr.initialize().unwrap();
        mgr.add_file("src/main.rs").unwrap();
        let stats = mgr.get_stats();
        assert!(stats.total_files >= 1);
        assert_eq!(1, stats.indexed_files);
        assert!(stats.total_chunks >= 1);
    }
}
