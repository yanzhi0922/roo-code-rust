//! Tool dispatcher for routing tool calls to appropriate handlers.
//!
//! Provides a trait-based dispatch mechanism where tool names are mapped
//! to [`ToolHandler`] implementations. The dispatcher supports both
//! synchronous and asynchronous tool execution.
//!
//! Source: `src/core/task/Task.ts` 鈥?tool execution logic scattered across
//! `executeTool`, `presentAssistantMessage`, and various tool handlers.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::Value;

// ---------------------------------------------------------------------------
// ToolExecutionResult
// ---------------------------------------------------------------------------

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    /// The text output of the tool.
    pub text: String,
    /// Optional images (base64 encoded).
    pub images: Option<Vec<String>>,
    /// Whether the tool execution resulted in an error.
    pub is_error: bool,
}

impl ToolExecutionResult {
    /// Create a successful result with text.
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: None,
            is_error: false,
        }
    }

    /// Create a successful result with text and images.
    pub fn success_with_images(text: impl Into<String>, images: Vec<String>) -> Self {
        Self {
            text: text.into(),
            images: Some(images),
            is_error: false,
        }
    }

    /// Create an error result.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: None,
            is_error: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ToolContext
// ---------------------------------------------------------------------------

/// Context provided to tool handlers during execution.
///
/// Contains the working directory and other environment information
/// needed by tools to perform their operations.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Current working directory for file operations.
    pub cwd: PathBuf,
    /// Task ID for logging and correlation.
    pub task_id: String,
}

impl ToolContext {
    /// Create a new tool context.
    pub fn new(cwd: impl Into<PathBuf>, task_id: impl Into<String>) -> Self {
        Self {
            cwd: cwd.into(),
            task_id: task_id.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolHandler trait
// ---------------------------------------------------------------------------

/// Trait for tool handlers.
///
/// Each tool handler is responsible for:
/// 1. Parsing the JSON parameters into the appropriate type
/// 2. Validating the parameters
/// 3. Executing the tool operation
/// 4. Returning the result
///
/// Handlers are async to support tools that require async operations
/// (e.g., command execution, MCP calls).
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Execute the tool with the given parameters and context.
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult;

    /// Returns the tool name this handler is responsible for.
    fn tool_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// ToolDispatcher
// ---------------------------------------------------------------------------

/// Dispatches tool calls to the appropriate handler based on tool name.
///
/// # Example
///
/// ```ignore
/// use roo_task::tool_dispatcher::{ToolDispatcher, ToolContext, ToolExecutionResult, ToolHandler};
///
/// let mut dispatcher = ToolDispatcher::new();
/// dispatcher.register("read_file", MyReadFileHandler);
///
/// let ctx = ToolContext::new("/tmp/work", "task-1");
/// let result = dispatcher.dispatch("read_file", json!({"path": "main.rs"}), &ctx).await;
/// ```
pub struct ToolDispatcher {
    handlers: HashMap<String, Box<dyn ToolHandler>>,
}

impl ToolDispatcher {
    /// Create a new empty dispatcher.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a tool handler.
    pub fn register(&mut self, name: impl Into<String>, handler: Box<dyn ToolHandler>) {
        self.handlers.insert(name.into(), handler);
    }

    /// Register a handler using a closure.
    ///
    /// Convenience method for simple tools that don't need a full handler struct.
    pub fn register_fn<F>(&mut self, name: impl Into<String>, handler_fn: F)
    where
        F: Fn(Value, &ToolContext) -> ToolExecutionResult + Send + Sync + 'static,
    {
        struct FnHandler<F> {
            name: String,
            f: F,
        }

        #[async_trait]
        impl<F> ToolHandler for FnHandler<F>
        where
            F: Fn(Value, &ToolContext) -> ToolExecutionResult + Send + Sync,
        {
            async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
                (self.f)(params, context)
            }

            fn tool_name(&self) -> &str {
                &self.name
            }
        }

        let name_str = name.into();
        self.handlers.insert(
            name_str.clone(),
            Box::new(FnHandler {
                name: name_str,
                f: handler_fn,
            }),
        );
    }

    /// Register an async handler using a closure.
    pub fn register_async_fn<F, Fut>(&mut self, name: impl Into<String>, handler_fn: F)
    where
        F: Fn(Value, &ToolContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ToolExecutionResult> + Send + 'static,
    {
        struct AsyncFnHandler<F> {
            name: String,
            f: F,
        }

        #[async_trait]
        impl<F, Fut> ToolHandler for AsyncFnHandler<F>
        where
            F: Fn(Value, &ToolContext) -> Fut + Send + Sync,
            Fut: std::future::Future<Output = ToolExecutionResult> + Send,
        {
            async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
                (self.f)(params, context).await
            }

            fn tool_name(&self) -> &str {
                &self.name
            }
        }

        let name_str = name.into();
        self.handlers.insert(
            name_str.clone(),
            Box::new(AsyncFnHandler {
                name: name_str,
                f: handler_fn,
            }),
        );
    }

    /// Dispatch a tool call to the appropriate handler.
    ///
    /// Returns an error result if no handler is registered for the tool name.
    pub async fn dispatch(
        &self,
        tool_name: &str,
        params: Value,
        context: &ToolContext,
    ) -> ToolExecutionResult {
        match self.handlers.get(tool_name) {
            Some(handler) => handler.execute(params, context).await,
            None => ToolExecutionResult::error(format!(
                "Unknown tool: '{}'. No handler registered.",
                tool_name
            )),
        }
    }

    /// Check whether a handler is registered for the given tool name.
    pub fn has_handler(&self, tool_name: &str) -> bool {
        self.handlers.contains_key(tool_name)
    }

    /// Get the list of registered tool names.
    pub fn registered_tools(&self) -> Vec<&str> {
        self.handlers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ToolDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in handler implementations
// ---------------------------------------------------------------------------

/// Handler for the `read_file` tool.
pub struct ReadFileHandler;

#[async_trait]
impl ToolHandler for ReadFileHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: path"),
        };

        let read_params = roo_types::tool::ReadFileParams {
            path,
            offset: params.get("offset").and_then(|v| v.as_u64()),
            limit: params.get("limit").and_then(|v| v.as_u64()),
        };

        match roo_tools_fs::process_read_file(&read_params, &context.cwd) {
            Ok(result) => ToolExecutionResult::success(result.content),
            Err(e) => ToolExecutionResult::error(format!("read_file error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "read_file"
    }
}

/// Handler for the `write_to_file` tool.
pub struct WriteToFileHandler;

#[async_trait]
impl ToolHandler for WriteToFileHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: path"),
        };
        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: content"),
        };

        let write_params = roo_types::tool::WriteToFileParams { path, content };

        match roo_tools_fs::process_write_to_file(&write_params, &context.cwd) {
            Ok(result) => {
                let msg = if result.is_new_file {
                    format!(
                        "Created new file: {} ({} lines)",
                        result.path, result.lines_written
                    )
                } else {
                    format!(
                        "Modified file: {} ({} lines written)",
                        result.path, result.lines_written
                    )
                };
                ToolExecutionResult::success(msg)
            }
            Err(e) => ToolExecutionResult::error(format!("write_to_file error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "write_to_file"
    }
}

/// Handler for the `apply_diff` tool.
///
/// Validates params, reads the original file, parses diff blocks,
/// applies them, and writes the result.
pub struct ApplyDiffHandler;

#[async_trait]
impl ToolHandler for ApplyDiffHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: path"),
        };
        let diff = match params.get("diff").and_then(|v| v.as_str()) {
            Some(d) => d.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: diff"),
        };

        let diff_params = roo_types::tool::ApplyDiffParams {
            path: path.clone(),
            diff,
        };

        // Validate
        if let Err(e) = roo_tools_fs::validate_apply_diff_params(&diff_params) {
            return ToolExecutionResult::error(format!("apply_diff validation error: {}", e));
        }

        // Resolve path
        let file_path = context.cwd.join(&path);
        if !file_path.exists() {
            return ToolExecutionResult::error(format!("File not found: {}", path));
        }

        // Read original content
        let original = match std::fs::read_to_string(&file_path) {
            Ok(s) => s,
            Err(e) => return ToolExecutionResult::error(format!("Failed to read file: {}", e)),
        };

        // Parse diff blocks
        let blocks = match roo_tools_fs::parse_diff_blocks(&diff_params.diff) {
            Ok(b) => b,
            Err(e) => return ToolExecutionResult::error(format!("Failed to parse diff: {}", e)),
        };

        // Apply diff blocks manually (apply_diff_blocks doesn't return new content)
        let mut new_content = original;
        let mut blocks_applied = 0usize;
        let mut warnings = Vec::new();

        for (i, (search, replace)) in blocks.iter().enumerate() {
            if let Some(pos) = new_content.find(search.as_str()) {
                new_content.replace_range(pos..pos + search.len(), replace);
                blocks_applied += 1;
            } else {
                warnings.push(format!("Block {}: search content not found in file", i + 1));
            }
        }

        // Write result
        if let Err(e) = std::fs::write(&file_path, &new_content) {
            return ToolExecutionResult::error(format!("Failed to write file: {}", e));
        }

        let mut msg = format!("Applied {} diff block(s) to {}", blocks_applied, path);
        if !warnings.is_empty() {
            msg.push_str(&format!("\nWarnings: {}", warnings.join(", ")));
        }
        ToolExecutionResult::success(msg)
    }

    fn tool_name(&self) -> &str {
        "apply_diff"
    }
}

/// Handler for the `edit_file` tool.
pub struct EditFileHandler;

#[async_trait]
impl ToolHandler for EditFileHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let file_path = match params
            .get("filePath")
            .or_else(|| params.get("file_path"))
            .and_then(|v| v.as_str())
        {
            Some(p) => p.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: filePath"),
        };
        let old_string = match params
            .get("oldString")
            .or_else(|| params.get("old_string"))
            .and_then(|v| v.as_str())
        {
            Some(s) => s.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: oldString"),
        };
        let new_string = match params
            .get("newString")
            .or_else(|| params.get("new_string"))
            .and_then(|v| v.as_str())
        {
            Some(s) => s.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: newString"),
        };

        let edit_params = roo_types::tool::EditFileParams {
            file_path,
            old_string,
            new_string,
            expected_replacements: params
                .get("expectedReplacements")
                .or_else(|| params.get("expected_replacements"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        };

        match roo_tools_fs::process_edit_file(&edit_params, &context.cwd) {
            Ok(result) => {
                let msg = result
                    .message
                    .unwrap_or_else(|| format!("Edit applied to {}", result.path));
                if result.success {
                    ToolExecutionResult::success(msg)
                } else {
                    ToolExecutionResult::error(msg)
                }
            }
            Err(e) => ToolExecutionResult::error(format!("edit_file error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "edit_file"
    }
}

/// Handler for the `list_files` tool.
///
/// Lists files and directories in the given path.
pub struct ListFilesHandler;

#[async_trait]
impl ToolHandler for ListFilesHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: path"),
        };
        let recursive = params
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let list_params = roo_types::tool::ListFilesParams {
            path: path.clone(),
            recursive,
        };

        // Validate
        if let Err(e) = roo_tools_search::validate_list_files_params(&list_params) {
            return ToolExecutionResult::error(format!("list_files error: {}", e));
        }

        // Resolve path
        let dir_path = context.cwd.join(&path);
        if !dir_path.exists() {
            return ToolExecutionResult::error(format!("Directory not found: {}", path));
        }
        if !dir_path.is_dir() {
            return ToolExecutionResult::error(format!("{} is not a directory", path));
        }

        // Collect entries
        let mut entries = Vec::new();
        collect_entries(&dir_path, &dir_path, recursive, &mut entries, 500);

        let result = roo_tools_search::build_file_list_result(
            &path,
            recursive,
            entries,
            roo_tools_search::MAX_FILE_LIST_ENTRIES,
        );

        // Format output
        let mut output = String::new();
        if !result.directories.is_empty() {
            output.push_str("Directories:\n");
            for d in &result.directories {
                output.push_str(&format!("  {}/\n", d));
            }
        }
        if !result.files.is_empty() {
            output.push_str("Files:\n");
            for f in &result.files {
                output.push_str(&format!("  {}\n", f));
            }
        }
        if result.truncated {
            output.push_str(&format!(
                "\n(truncated, showing {} of {} entries)",
                result.files.len() + result.directories.len(),
                result.total_count
            ));
        }

        ToolExecutionResult::success(output)
    }

    fn tool_name(&self) -> &str {
        "list_files"
    }
}

/// Helper to recursively collect directory entries.
fn collect_entries(
    base: &std::path::Path,
    current: &std::path::Path,
    recursive: bool,
    entries: &mut Vec<String>,
    max_entries: usize,
) {
    if entries.len() >= max_entries {
        return;
    }
    if let Ok(read_dir) = std::fs::read_dir(current) {
        for entry in read_dir.flatten() {
            if entries.len() >= max_entries {
                break;
            }
            let entry_path = entry.path();
            let relative = entry_path.strip_prefix(base).unwrap_or(&entry_path);
            let entry_str = relative.to_string_lossy().to_string();
            if entry.path().is_dir() {
                entries.push(format!("{}/", entry_str));
                if recursive {
                    collect_entries(base, &entry.path(), recursive, entries, max_entries);
                }
            } else {
                entries.push(entry_str);
            }
        }
    }
}

/// Handler for the `search_files` tool.
pub struct SearchFilesHandler;

#[async_trait]
impl ToolHandler for SearchFilesHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: path"),
        };
        let regex = match params.get("regex").and_then(|v| v.as_str()) {
            Some(r) => r.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: regex"),
        };
        let file_pattern = params
            .get("filePattern")
            .or_else(|| params.get("file_pattern"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let search_params = roo_types::tool::SearchFilesParams {
            path,
            regex,
            file_pattern,
        };

        // Validate
        if let Err(e) = roo_tools_search::validate_search_files_params(&search_params) {
            return ToolExecutionResult::error(format!("search_files error: {}", e));
        }

        // Resolve path
        let dir_path = context.cwd.join(&search_params.path);
        if !dir_path.exists() {
            return ToolExecutionResult::error(format!(
                "Directory not found: {}",
                search_params.path
            ));
        }

        // Compile regex
        let re = match regex::Regex::new(&search_params.regex) {
            Ok(re) => re,
            Err(e) => return ToolExecutionResult::error(format!("Invalid regex: {}", e)),
        };

        // Search files
        let mut matches = Vec::new();
        search_in_dir(
            &dir_path,
            &dir_path,
            &re,
            &search_params.file_pattern,
            &mut matches,
            200,
        );

        if matches.is_empty() {
            return ToolExecutionResult::success("No matches found.");
        }

        let output = roo_tools_search::format_search_results(&matches);
        ToolExecutionResult::success(output)
    }

    fn tool_name(&self) -> &str {
        "search_files"
    }
}

/// Helper to recursively search files for regex matches.
fn search_in_dir(
    base: &std::path::Path,
    current: &std::path::Path,
    re: &regex::Regex,
    file_pattern: &Option<String>,
    matches: &mut Vec<roo_tools_search::FileMatch>,
    max_matches: usize,
) {
    if matches.len() >= max_matches {
        return;
    }
    if let Ok(read_dir) = std::fs::read_dir(current) {
        for entry in read_dir.flatten() {
            if matches.len() >= max_matches {
                break;
            }
            let path = entry.path();
            if path.is_dir() {
                search_in_dir(base, &path, re, file_pattern, matches, max_matches);
            } else if path.is_file() {
                // Check file pattern
                let relative = path.strip_prefix(base).unwrap_or(path.as_path());
                let relative_str = relative.to_string_lossy();
                if let Some(pattern) = file_pattern {
                    if !roo_tools_search::matches_file_pattern(&relative_str, pattern) {
                        continue;
                    }
                }

                // Read and search
                if let Ok(content) = std::fs::read_to_string(&path) {
                    for (i, line) in content.lines().enumerate() {
                        if matches.len() >= max_matches {
                            break;
                        }
                        if re.is_match(line) {
                            matches.push(roo_tools_search::FileMatch {
                                file_path: relative_str.to_string(),
                                line_number: i + 1,
                                line_content: line.to_string(),
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Handler for the `codebase_search` tool.
pub struct CodebaseSearchHandler;

#[async_trait]
impl ToolHandler for CodebaseSearchHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) => q.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: query"),
        };

        let search_params = roo_types::tool::CodebaseSearchParams {
            query,
            directory_prefix: params
                .get("directoryPrefix")
                .or_else(|| params.get("directory_prefix"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        };

        // Validate
        if let Err(e) = roo_tools_search::validate_codebase_search_params(&search_params) {
            return ToolExecutionResult::error(format!("codebase_search error: {}", e));
        }

        // Codebase search requires an index; return placeholder
        ToolExecutionResult::success(format!(
            "Codebase search for '{}' (index not yet available)",
            search_params.query
        ))
    }

    fn tool_name(&self) -> &str {
        "codebase_search"
    }
}

// ---------------------------------------------------------------------------
// Default dispatcher builder
// ---------------------------------------------------------------------------

/// Build a [`ToolDispatcher`] pre-loaded with all built-in tool handlers.
///
/// This creates a dispatcher with handlers for:
/// - `read_file`, `write_to_file`, `apply_diff`, `edit_file` (file system)
/// - `list_files`, `search_files`, `codebase_search` (search)
///
/// Tools that require runtime infrastructure (command execution, MCP,
/// mode switching, etc.) are **not** included and must be registered
/// separately by the application layer.
pub fn default_dispatcher() -> ToolDispatcher {
    let mut dispatcher = ToolDispatcher::new();

    // File system tools
    dispatcher.register("read_file", Box::new(ReadFileHandler));
    dispatcher.register("write_to_file", Box::new(WriteToFileHandler));
    dispatcher.register("apply_diff", Box::new(ApplyDiffHandler));
    dispatcher.register("edit_file", Box::new(EditFileHandler));

    // Search tools
    dispatcher.register("list_files", Box::new(ListFilesHandler));
    dispatcher.register("search_files", Box::new(SearchFilesHandler));
    dispatcher.register("codebase_search", Box::new(CodebaseSearchHandler));

    dispatcher
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_context() -> ToolContext {
        ToolContext::new("/tmp/test-workspace", "test-task")
    }

    #[tokio::test]
    async fn test_dispatch_unknown_tool() {
        let dispatcher = ToolDispatcher::new();
        let ctx = make_context();
        let result = dispatcher
            .dispatch("nonexistent_tool", serde_json::json!({}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_register_fn_handler() {
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("test_tool", |_params, _ctx| {
            ToolExecutionResult::success("test output")
        });

        let ctx = make_context();
        let result = dispatcher
            .dispatch("test_tool", serde_json::json!({}), &ctx)
            .await;
        assert!(!result.is_error);
        assert_eq!(result.text, "test output");
    }

    #[tokio::test]
    async fn test_register_async_fn_handler() {
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_async_fn("async_tool", |_params, _ctx| async move {
            ToolExecutionResult::success("async output")
        });

        let ctx = make_context();
        let result = dispatcher
            .dispatch("async_tool", serde_json::json!({}), &ctx)
            .await;
        assert!(!result.is_error);
        assert_eq!(result.text, "async output");
    }

    #[test]
    fn test_has_handler() {
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("tool_a", |_, _| ToolExecutionResult::success("a"));
        assert!(dispatcher.has_handler("tool_a"));
        assert!(!dispatcher.has_handler("tool_b"));
    }

    #[test]
    fn test_registered_tools() {
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("tool_a", |_, _| ToolExecutionResult::success("a"));
        dispatcher.register_fn("tool_b", |_, _| ToolExecutionResult::success("b"));

        let mut tools = dispatcher.registered_tools();
        tools.sort();
        assert_eq!(tools, vec!["tool_a", "tool_b"]);
    }

    #[test]
    fn test_tool_execution_result_success() {
        let result = ToolExecutionResult::success("done");
        assert_eq!(result.text, "done");
        assert!(result.images.is_none());
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_execution_result_error() {
        let result = ToolExecutionResult::error("failed");
        assert_eq!(result.text, "failed");
        assert!(result.images.is_none());
        assert!(result.is_error);
    }

    #[test]
    fn test_tool_execution_result_with_images() {
        let result =
            ToolExecutionResult::success_with_images("screenshot", vec!["base64data".into()]);
        assert_eq!(result.text, "screenshot");
        let expected: &[String] = &["base64data".to_string()];
        assert_eq!(result.images.as_deref(), Some(expected));
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_context_new() {
        let ctx = ToolContext::new("/tmp/work", "task-1");
        assert_eq!(ctx.cwd, PathBuf::from("/tmp/work"));
        assert_eq!(ctx.task_id, "task-1");
    }

    #[test]
    fn test_default_dispatcher_has_core_tools() {
        let dispatcher = default_dispatcher();
        assert!(dispatcher.has_handler("read_file"));
        assert!(dispatcher.has_handler("write_to_file"));
        assert!(dispatcher.has_handler("apply_diff"));
        assert!(dispatcher.has_handler("edit_file"));
        assert!(dispatcher.has_handler("list_files"));
        assert!(dispatcher.has_handler("search_files"));
        assert!(dispatcher.has_handler("codebase_search"));
    }

    #[tokio::test]
    async fn test_read_file_handler_missing_path() {
        let handler = ReadFileHandler;
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("path"));
    }

    #[tokio::test]
    async fn test_write_to_file_handler_missing_content() {
        let handler = WriteToFileHandler;
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"path": "test.rs"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("content"));
    }

    #[tokio::test]
    async fn test_apply_diff_handler_missing_diff() {
        let handler = ApplyDiffHandler;
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"path": "test.rs"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("diff"));
    }

    #[tokio::test]
    async fn test_list_files_handler_missing_path() {
        let handler = ListFilesHandler;
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("path"));
    }

    #[tokio::test]
    async fn test_search_files_handler_missing_regex() {
        let handler = SearchFilesHandler;
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"path": "."}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("regex"));
    }

    #[tokio::test]
    async fn test_codebase_search_handler_missing_query() {
        let handler = CodebaseSearchHandler;
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("query"));
    }
}