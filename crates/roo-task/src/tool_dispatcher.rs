//! Tool dispatcher for routing tool calls to appropriate handlers.
//!
//! Provides a trait-based dispatch mechanism where tool names are mapped
//! to [`ToolHandler`] implementations. The dispatcher supports both
//! synchronous and asynchronous tool execution.
//!
//! Source: `src/core/task/Task.ts` — tool execution logic scattered across
//! `executeTool`, `presentAssistantMessage`, and various tool handlers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use roo_tools::ToolRepetitionDetector;

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
    /// Optional repetition detector to prevent infinite tool-call loops.
    repetition_detector: Option<std::sync::Mutex<ToolRepetitionDetector>>,
}

impl ToolDispatcher {
    /// Create a new empty dispatcher.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            repetition_detector: None,
        }
    }

    /// Attach a repetition detector to this dispatcher.
    ///
    /// When set, every [`dispatch`](Self::dispatch) call will be checked
    /// against the detector before the handler is invoked. If the same
    /// tool + params combination is called too many times consecutively,
    /// a warning result is returned instead of executing the handler.
    pub fn set_repetition_detector(&mut self, detector: ToolRepetitionDetector) {
        self.repetition_detector = Some(std::sync::Mutex::new(detector));
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
    /// If a [`ToolRepetitionDetector`] is installed and the call is deemed
    /// repetitive, a warning result is returned **without** invoking the
    /// handler.
    ///
    /// Returns an error result if no handler is registered for the tool name.
    pub async fn dispatch(
        &self,
        tool_name: &str,
        params: Value,
        context: &ToolContext,
    ) -> ToolExecutionResult {
        // --- Repetition check ---
        if let Some(detector_mtx) = &self.repetition_detector {
            if let Ok(mut detector) = detector_mtx.lock() {
                if !detector.check_and_record(tool_name, &params) {
                    return ToolExecutionResult::success(format!(
                        "Warning: The tool '{}' has been called with similar \
                         parameters multiple times. Consider using a different \
                         approach.",
                        tool_name
                    ));
                }
            }
        }

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

        // Parse mode
        let mode = params.get("mode").and_then(|v| v.as_str()).and_then(|m| {
            match m {
                "slice" => Some(roo_types::tool_params::ReadFileMode::Slice),
                "indentation" => Some(roo_types::tool_params::ReadFileMode::Indentation),
                _ => None,
            }
        });

        // Parse indentation params if present
        let indentation = params.get("indentation").and_then(|v| {
            Some(roo_types::tool_params::IndentationParams {
                anchor_line: v.get("anchorLine").or_else(|| v.get("anchor_line")).and_then(|v2| v2.as_u64()),
                max_levels: v.get("maxLevels").or_else(|| v.get("max_levels")).and_then(|v2| v2.as_u64()),
                include_siblings: v.get("includeSiblings").or_else(|| v.get("include_siblings")).and_then(|v2| v2.as_bool()),
                include_header: v.get("includeHeader").or_else(|| v.get("include_header")).and_then(|v2| v2.as_bool()),
                max_lines: v.get("maxLines").or_else(|| v.get("max_lines")).and_then(|v2| v2.as_u64()),
            })
        });

        let read_params = roo_types::tool_params::ReadFileParams {
            path,
            mode,
            offset: params.get("offset").and_then(|v| v.as_u64()),
            limit: params.get("limit").and_then(|v| v.as_u64()),
            indentation,
        };

        match roo_tools_fs::process_read_file(&read_params, &context.cwd, None) {
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

        match roo_tools_fs::process_write_to_file(&write_params, &context.cwd, None) {
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

        match roo_tools_fs::process_edit_file(&edit_params, &context.cwd, None) {
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
                                context_before: vec![],
                                context_after: vec![],
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
// Command execution handlers
// ---------------------------------------------------------------------------

/// Handler for the `execute_command` tool.
///
/// Executes shell commands via the [`TerminalRegistry`] with optional
/// timeout and output persistence.
pub struct ExecuteCommandHandler {
    registry: Arc<roo_terminal::TerminalRegistry>,
    output_dir: PathBuf,
    max_preview_lines: usize,
}

impl ExecuteCommandHandler {
    /// Create a new execute_command handler.
    pub fn new(
        registry: Arc<roo_terminal::TerminalRegistry>,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            registry,
            output_dir,
            max_preview_lines: 50,
        }
    }
}

#[async_trait]
impl ToolHandler for ExecuteCommandHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let command = match params.get("command").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: command"),
        };

        let cwd = params
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(|s| PathBuf::from(s));

        let timeout_ms = params
            .get("timeout")
            .and_then(|v| v.as_u64())
            .or_else(|| params.get("timeout_ms").and_then(|v| v.as_u64()));

        let cwd_ref = cwd.as_deref();
        let registry = self.registry.clone();
        let output_dir = self.output_dir.clone();
        let max_preview = self.max_preview_lines;
        let working_dir = context.cwd.clone();

        match roo_tools_command::execute_command::execute_command(
            &command,
            cwd_ref,
            timeout_ms,
            registry,
            &working_dir,
            Some(&output_dir),
            max_preview,
        )
        .await
        {
            Ok(result) => ToolExecutionResult::success(result.output),
            Err(e) => ToolExecutionResult::error(format!("Error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "execute_command"
    }
}

/// Handler for the `read_command_output` tool.
///
/// Reads persisted command output artifacts from disk with optional
/// search filtering and pagination.
pub struct ReadCommandOutputHandler {
    storage_dir: PathBuf,
}

impl ReadCommandOutputHandler {
    /// Create a new read_command_output handler.
    pub fn new(storage_dir: PathBuf) -> Self {
        Self { storage_dir }
    }
}

#[async_trait]
impl ToolHandler for ReadCommandOutputHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let artifact_id = match params.get("artifact_id").and_then(|v| v.as_str()) {
            Some(id) => id.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: artifact_id"),
        };

        let read_params = roo_types::tool::ReadCommandOutputParams {
            artifact_id,
            offset: params.get("offset").and_then(|v| v.as_u64()),
            limit: params.get("limit").and_then(|v| v.as_u64()),
            search: params.get("search").and_then(|v| v.as_str()).map(String::from),
        };

        match roo_tools_command::read_command_output::read_command_output_from_disk(
            &read_params,
            &self.storage_dir,
        )
        .await
        {
            Ok(result) => ToolExecutionResult::success(result.content),
            Err(e) => ToolExecutionResult::error(format!("read_command_output error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "read_command_output"
    }
}

// ---------------------------------------------------------------------------
// Miscellaneous tool handlers
// ---------------------------------------------------------------------------

/// Handler for the `ask_followup_question` tool.
pub struct AskFollowupQuestionHandler;

#[async_trait]
impl ToolHandler for AskFollowupQuestionHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let question = match params.get("question").and_then(|v| v.as_str()) {
            Some(q) => q.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: question"),
        };

        let follow_up: Vec<roo_types::tool::FollowUpOption> = match params.get("follow_up") {
            Some(opts) => match serde_json::from_value::<Vec<roo_types::tool::FollowUpOption>>(
                opts.clone(),
            ) {
                Ok(o) => o,
                Err(e) => return ToolExecutionResult::error(format!("Invalid follow_up: {}", e)),
            },
            None => return ToolExecutionResult::error("Missing required parameter: follow_up"),
        };

        let ask_params = roo_types::tool::AskFollowupQuestionParams {
            question,
            follow_up,
        };

        match roo_tools_misc::process_followup(&ask_params) {
            Ok(result) => {
                let output = format!(
                    "Question: {}\nSuggestions:\n{}",
                    result.question,
                    result
                        .suggestions
                        .iter()
                        .enumerate()
                        .map(|(i, s)| format!("  {}. {}", i + 1, s))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                ToolExecutionResult::success(output)
            }
            Err(e) => ToolExecutionResult::error(format!("ask_followup_question error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "ask_followup_question"
    }
}

/// Handler for the `attempt_completion` tool.
pub struct AttemptCompletionHandler;

#[async_trait]
impl ToolHandler for AttemptCompletionHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let result_text = match params.get("result").and_then(|v| v.as_str()) {
            Some(r) => r.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: result"),
        };

        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .map(String::from);

        let completion_params = roo_types::tool::AttemptCompletionParams {
            result: result_text,
            command,
        };

        match roo_tools_misc::process_attempt_completion(&completion_params, &[]) {
            Ok(result) => {
                let mut output = completion_params.result.clone();
                if let Some(warning) = &result.todo_warning {
                    output = format!("{}\n\n{}", warning, output);
                }
                ToolExecutionResult::success(output)
            }
            Err(e) => ToolExecutionResult::error(format!("attempt_completion error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "attempt_completion"
    }
}

/// Handler for the `update_todo_list` tool.
pub struct UpdateTodoListHandler;

#[async_trait]
impl ToolHandler for UpdateTodoListHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let todos = match params.get("todos").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: todos"),
        };

        let todo_params = roo_types::tool::UpdateTodoListParams { todos };

        match roo_tools_misc::process_update_todo(&todo_params) {
            Ok(items) => {
                let output = items
                    .iter()
                    .map(|item| format!("{} {}", item.status.to_checkbox(), item.text))
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolExecutionResult::success(output)
            }
            Err(e) => ToolExecutionResult::error(format!("update_todo_list error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "update_todo_list"
    }
}

// ---------------------------------------------------------------------------
// Mode tool handlers
// ---------------------------------------------------------------------------

/// Handler for the `switch_mode` tool.
pub struct SwitchModeHandler {
    /// Current mode slug, used to detect same-mode switches.
    current_mode: String,
}

impl SwitchModeHandler {
    /// Create a new switch_mode handler with the given current mode.
    pub fn new(current_mode: impl Into<String>) -> Self {
        Self {
            current_mode: current_mode.into(),
        }
    }
}

#[async_trait]
impl ToolHandler for SwitchModeHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let mode_slug = match params.get("mode_slug").and_then(|v| v.as_str()) {
            Some(m) => m.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: mode_slug"),
        };

        let reason = params
            .get("reason")
            .and_then(|v| v.as_str())
            .map(String::from);

        let switch_params = roo_types::tool::SwitchModeParams { mode_slug, reason };

        match roo_tools_mode::process_switch_mode(&switch_params, &self.current_mode) {
            Ok(result) => {
                let msg = if let Some(reason) = &result.reason {
                    format!("Switching to '{}' mode: {}", result.mode_slug, reason)
                } else {
                    format!("Switching to '{}' mode", result.mode_slug)
                };
                ToolExecutionResult::success(msg)
            }
            Err(e) => ToolExecutionResult::error(format!("switch_mode error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "switch_mode"
    }
}

/// Handler for the `new_task` tool.
pub struct NewTaskHandler;

#[async_trait]
impl ToolHandler for NewTaskHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let mode = match params.get("mode").and_then(|v| v.as_str()) {
            Some(m) => m.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: mode"),
        };

        let message = match params.get("message").and_then(|v| v.as_str()) {
            Some(m) => m.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: message"),
        };

        let todos = params
            .get("todos")
            .and_then(|v| v.as_str())
            .map(String::from);

        let task_params = roo_types::tool::NewTaskParams {
            mode,
            message,
            todos,
        };

        match roo_tools_mode::process_new_task(&task_params) {
            Ok(result) => {
                let mut output = format!("New task in '{}' mode:\n{}", result.mode, result.message);
                if let Some(todos) = &result.todos {
                    output.push_str(&format!("\n\nTodos:\n{}", todos));
                }
                ToolExecutionResult::success(output)
            }
            Err(e) => ToolExecutionResult::error(format!("new_task error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "new_task"
    }
}

// ---------------------------------------------------------------------------
// MCP tool handlers
// ---------------------------------------------------------------------------

/// Handler for the `use_mcp_tool` tool.
///
/// Calls an MCP tool on a connected server via [`roo_mcp::McpHub`].
pub struct UseMcpToolHandler {
    hub: Arc<roo_mcp::McpHub>,
}

impl UseMcpToolHandler {
    /// Create a new use_mcp_tool handler.
    pub fn new(hub: Arc<roo_mcp::McpHub>) -> Self {
        Self { hub }
    }
}

#[async_trait]
impl ToolHandler for UseMcpToolHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let server_name = match params.get("server_name").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: server_name"),
        };
        let tool_name = match params.get("tool_name").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: tool_name"),
        };
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));

        let mcp_params = roo_types::tool::UseMcpToolParams {
            server_name,
            tool_name,
            arguments,
        };

        let result = roo_tools_mcp::execute_mcp_tool(&self.hub, &mcp_params).await;

        let mut exec_result = if result.is_error {
            ToolExecutionResult::error(&result.text)
        } else {
            ToolExecutionResult::success(&result.text)
        };

        // Attach images if present
        if !result.images.is_empty() {
            exec_result.images = Some(result.images);
        }

        exec_result
    }

    fn tool_name(&self) -> &str {
        "use_mcp_tool"
    }
}

/// Handler for the `access_mcp_resource` tool.
///
/// Reads a resource from a connected MCP server via [`roo_mcp::McpHub`].
pub struct AccessMcpResourceHandler {
    hub: Arc<roo_mcp::McpHub>,
}

impl AccessMcpResourceHandler {
    /// Create a new access_mcp_resource handler.
    pub fn new(hub: Arc<roo_mcp::McpHub>) -> Self {
        Self { hub }
    }
}

#[async_trait]
impl ToolHandler for AccessMcpResourceHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let server_name = match params.get("server_name").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: server_name"),
        };
        let uri = match params.get("uri").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: uri"),
        };

        let mcp_params = roo_types::tool::AccessMcpResourceParams {
            server_name,
            uri,
        };

        let result = roo_tools_mcp::access_mcp_resource(&self.hub, &mcp_params).await;

        if result.is_error {
            ToolExecutionResult::error(&result.text)
        } else {
            ToolExecutionResult::success(&result.text)
        }
    }

    fn tool_name(&self) -> &str {
        "access_mcp_resource"
    }
}

// ---------------------------------------------------------------------------
// Skill & Slash Command handlers
// ---------------------------------------------------------------------------

/// Handler for the `skill` tool.
///
/// Loads and executes an Agent Skill (SKILL.md file).
/// Parameters: `skill` (skill name), `args` (optional context).
///
/// When an [`roo_skills::SkillsManager`] is provided, the handler looks up
/// the skill by name and returns its full instructions. Otherwise, a
/// placeholder message is returned.
pub struct SkillHandler {
    skills_manager: Option<Arc<roo_skills::SkillsManager>>,
}

impl SkillHandler {
    /// Create a new skill handler without a manager (fallback mode).
    pub fn new() -> Self {
        Self {
            skills_manager: None,
        }
    }

    /// Create a skill handler backed by a [`roo_skills::SkillsManager`].
    pub fn with_manager(manager: Arc<roo_skills::SkillsManager>) -> Self {
        Self {
            skills_manager: Some(manager),
        }
    }
}

#[async_trait]
impl ToolHandler for SkillHandler {
    async fn execute(&self, params: Value, _context: &ToolContext) -> ToolExecutionResult {
        let skill_name = match params.get("skill").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: skill"),
        };

        let args = params
            .get("args")
            .and_then(|v| v.as_str())
            .map(String::from);

        let skill_params = roo_types::tool::SkillParams {
            skill: skill_name,
            args,
        };

        match roo_tools_misc::process_skill(&skill_params, self.skills_manager.as_deref()) {
            Ok(result) => {
                if let Some(content) = &result.content {
                    ToolExecutionResult::success(content.clone())
                } else {
                    let mut msg = format!(
                        "Skill '{}' loaded. Follow the skill instructions.",
                        result.skill_name
                    );
                    if let Some(args) = &result.args {
                        msg.push_str(&format!("\nContext: {}", args));
                    }
                    ToolExecutionResult::success(msg)
                }
            }
            Err(e) => ToolExecutionResult::error(format!("skill error: {}", e)),
        }
    }

    fn tool_name(&self) -> &str {
        "skill"
    }
}

/// Handler for the `run_slash_command` tool.
///
/// Executes a slash command (e.g. /init, /test, /deploy).
/// Parameters: `command` (command name), `args` (optional arguments).
///
/// Looks up the command via [`roo_command::get_command`] using the current
/// working directory from the tool context. If the command is found, its
/// content is returned; otherwise a fallback message is produced.
pub struct SlashCommandHandler;

#[async_trait]
impl ToolHandler for SlashCommandHandler {
    async fn execute(&self, params: Value, context: &ToolContext) -> ToolExecutionResult {
        let command = match params.get("command").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return ToolExecutionResult::error("Missing required parameter: command"),
        };

        let args = params
            .get("args")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Try to load command from project/global/built-in sources
        if let Some(cmd) = roo_command::get_command(&context.cwd, &command).await {
            let mut result = format!("Command: {}\n", cmd.name);
            if let Some(desc) = &cmd.description {
                result.push_str(&format!("Description: {}\n", desc));
            }
            if !cmd.content.is_empty() {
                result.push_str(&format!("\n{}", cmd.content));
            }
            if let Some(args) = &args {
                result.push_str(&format!("\n\nArguments: {}", args));
            }
            return ToolExecutionResult::success(result);
        }

        // Fallback: command not found in any source
        let mut msg = format!("Slash command '/{}' executed.", command);
        if let Some(args) = &args {
            msg.push_str(&format!(" Arguments: {}", args));
        }

        ToolExecutionResult::success(msg)
    }

    fn tool_name(&self) -> &str {
        "run_slash_command"
    }
}

// ---------------------------------------------------------------------------
// Default dispatcher builders
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

/// Build a [`ToolDispatcher`] pre-loaded with **all** built-in tool handlers,
/// including those that require runtime infrastructure.
///
/// In addition to the tools registered by [`default_dispatcher`], this also
/// registers:
/// - `execute_command`, `read_command_output` (command execution)
/// - `ask_followup_question`, `attempt_completion`, `update_todo_list` (misc)
/// - `switch_mode`, `new_task` (mode switching)
/// - `skill`, `run_slash_command` (skill & slash command)
///
/// # Arguments
/// * `registry` — Shared [`TerminalRegistry`] for command execution.
/// * `output_dir` — Directory where command output artifacts are persisted.
/// * `current_mode` — The current mode slug (used by `switch_mode` validation).
pub fn default_dispatcher_with_terminal(
    registry: Arc<roo_terminal::TerminalRegistry>,
    output_dir: PathBuf,
    current_mode: &str,
) -> ToolDispatcher {
    // Start with the base set of tools
    let mut dispatcher = default_dispatcher();

    // Command execution tools
    dispatcher.register(
        "execute_command",
        Box::new(ExecuteCommandHandler::new(registry.clone(), output_dir.clone())),
    );
    dispatcher.register(
        "read_command_output",
        Box::new(ReadCommandOutputHandler::new(output_dir)),
    );

    // Miscellaneous tools
    dispatcher.register(
        "ask_followup_question",
        Box::new(AskFollowupQuestionHandler),
    );
    dispatcher.register(
        "attempt_completion",
        Box::new(AttemptCompletionHandler),
    );
    dispatcher.register(
        "update_todo_list",
        Box::new(UpdateTodoListHandler),
    );

    // Mode tools
    dispatcher.register(
        "switch_mode",
        Box::new(SwitchModeHandler::new(current_mode)),
    );
    dispatcher.register("new_task", Box::new(NewTaskHandler));

    // Skill & Slash Command tools
    dispatcher.register("skill", Box::new(SkillHandler::new()));
    dispatcher.register("run_slash_command", Box::new(SlashCommandHandler));

    dispatcher
}

/// Build a [`ToolDispatcher`] pre-loaded with **all** built-in tool handlers,
/// including MCP tools that require an [`roo_mcp::McpHub`] reference.
///
/// In addition to the tools registered by [`default_dispatcher_with_terminal`],
/// this also registers:
/// - `use_mcp_tool` (MCP tool calls)
/// - `access_mcp_resource` (MCP resource access)
/// - `skill`, `run_slash_command` (registered via `default_dispatcher_with_terminal`)
///
/// # Arguments
/// * `registry` — Shared [`TerminalRegistry`] for command execution.
/// * `output_dir` — Directory where command output artifacts are persisted.
/// * `current_mode` — The current mode slug (used by `switch_mode` validation).
/// * `mcp_hub` — Shared [`roo_mcp::McpHub`] for MCP tool and resource operations.
pub fn default_dispatcher_full(
    registry: Arc<roo_terminal::TerminalRegistry>,
    output_dir: PathBuf,
    current_mode: &str,
    mcp_hub: Arc<roo_mcp::McpHub>,
) -> ToolDispatcher {
    let mut dispatcher = default_dispatcher_with_terminal(registry, output_dir, current_mode);

    // MCP tools
    dispatcher.register(
        "use_mcp_tool",
        Box::new(UseMcpToolHandler::new(mcp_hub.clone())),
    );
    dispatcher.register(
        "access_mcp_resource",
        Box::new(AccessMcpResourceHandler::new(mcp_hub)),
    );

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

    // ---- Command handler tests ----

    #[tokio::test]
    async fn test_execute_command_handler_missing_command() {
        let registry = Arc::new(roo_terminal::TerminalRegistry::new());
        let dir = tempfile::tempdir().unwrap();
        let handler = ExecuteCommandHandler::new(registry, dir.path().to_path_buf());
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("command"));
    }

    #[tokio::test]
    async fn test_execute_command_handler_simple() {
        let registry = Arc::new(roo_terminal::TerminalRegistry::new());
        let dir = tempfile::tempdir().unwrap();
        let handler = ExecuteCommandHandler::new(registry, dir.path().to_path_buf());
        let ctx = ToolContext::new(dir.path().to_path_buf(), "test-task");
        let result = handler
            .execute(
                serde_json::json!({"command": "echo hello", "timeout": 5000}),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(
            result.text.contains("hello"),
            "expected 'hello' in: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn test_read_command_output_handler_missing_artifact_id() {
        let dir = tempfile::tempdir().unwrap();
        let handler = ReadCommandOutputHandler::new(dir.path().to_path_buf());
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("artifact_id"));
    }

    #[tokio::test]
    async fn test_read_command_output_handler_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let handler = ReadCommandOutputHandler::new(dir.path().to_path_buf());
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({"artifact_id": "cmd-nonexistent.txt"}),
                &ctx,
            )
            .await;
        assert!(result.is_error);
    }

    // ---- Misc handler tests ----

    #[tokio::test]
    async fn test_ask_followup_question_handler_missing_question() {
        let handler = AskFollowupQuestionHandler;
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({"follow_up": [{"text": "yes"}]}),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("question"));
    }

    #[tokio::test]
    async fn test_ask_followup_question_handler_valid() {
        let handler = AskFollowupQuestionHandler;
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({
                    "question": "Continue?",
                    "follow_up": [{"text": "Yes"}, {"text": "No"}]
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(result.text.contains("Continue?"));
    }

    #[tokio::test]
    async fn test_attempt_completion_handler_missing_result() {
        let handler = AttemptCompletionHandler;
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("result"));
    }

    #[tokio::test]
    async fn test_attempt_completion_handler_valid() {
        let handler = AttemptCompletionHandler;
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"result": "Task completed!"}), &ctx)
            .await;
        assert!(!result.is_error);
        assert!(result.text.contains("Task completed!"));
    }

    #[tokio::test]
    async fn test_update_todo_list_handler_missing_todos() {
        let handler = UpdateTodoListHandler;
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("todos"));
    }

    #[tokio::test]
    async fn test_update_todo_list_handler_valid() {
        let handler = UpdateTodoListHandler;
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({"todos": "[x] done\n[ ] pending"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(result.text.contains("done"));
        assert!(result.text.contains("pending"));
    }

    // ---- Mode handler tests ----

    #[tokio::test]
    async fn test_switch_mode_handler_missing_mode() {
        let handler = SwitchModeHandler::new("code");
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("mode_slug"));
    }

    #[tokio::test]
    async fn test_switch_mode_handler_valid() {
        let handler = SwitchModeHandler::new("code");
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({"mode_slug": "architect", "reason": "need to plan"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(result.text.contains("architect"));
    }

    #[tokio::test]
    async fn test_new_task_handler_missing_mode() {
        let handler = NewTaskHandler;
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"message": "do something"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("mode"));
    }

    #[tokio::test]
    async fn test_new_task_handler_valid() {
        let handler = NewTaskHandler;
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({
                    "mode": "code",
                    "message": "implement feature",
                    "todos": "[ ] task1\n[x] task2"
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(result.text.contains("implement feature"));
    }

    // ---- MCP handler tests ----

    #[tokio::test]
    async fn test_use_mcp_tool_handler_missing_server_name() {
        let hub = Arc::new(roo_mcp::McpHub::new());
        let handler = UseMcpToolHandler::new(hub);
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({"tool_name": "tool", "arguments": {}}),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("server_name"));
    }

    #[tokio::test]
    async fn test_use_mcp_tool_handler_missing_tool_name() {
        let hub = Arc::new(roo_mcp::McpHub::new());
        let handler = UseMcpToolHandler::new(hub);
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({"server_name": "server", "arguments": {}}),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("tool_name"));
    }

    #[tokio::test]
    async fn test_use_mcp_tool_handler_server_not_found() {
        let hub = Arc::new(roo_mcp::McpHub::new());
        let handler = UseMcpToolHandler::new(hub);
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({
                    "server_name": "nonexistent",
                    "tool_name": "tool",
                    "arguments": {}
                }),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("not found"));
    }

    #[tokio::test]
    async fn test_access_mcp_resource_handler_missing_server_name() {
        let hub = Arc::new(roo_mcp::McpHub::new());
        let handler = AccessMcpResourceHandler::new(hub);
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"uri": "file:///test"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("server_name"));
    }

    #[tokio::test]
    async fn test_access_mcp_resource_handler_missing_uri() {
        let hub = Arc::new(roo_mcp::McpHub::new());
        let handler = AccessMcpResourceHandler::new(hub);
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"server_name": "server"}), &ctx)
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("uri"));
    }

    #[tokio::test]
    async fn test_access_mcp_resource_handler_server_not_found() {
        let hub = Arc::new(roo_mcp::McpHub::new());
        let handler = AccessMcpResourceHandler::new(hub);
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({
                    "server_name": "nonexistent",
                    "uri": "file:///test"
                }),
                &ctx,
            )
            .await;
        assert!(result.is_error);
        assert!(result.text.contains("not found"));
    }

    // ---- Dispatcher builder tests ----

    #[test]
    fn test_default_dispatcher_with_terminal_has_all_tools() {
        let registry = Arc::new(roo_terminal::TerminalRegistry::new());
        let dir = tempfile::tempdir().unwrap();
        let dispatcher = default_dispatcher_with_terminal(
            registry,
            dir.path().to_path_buf(),
            "code",
        );

        // Core tools
        assert!(dispatcher.has_handler("read_file"));
        assert!(dispatcher.has_handler("write_to_file"));
        assert!(dispatcher.has_handler("apply_diff"));
        assert!(dispatcher.has_handler("edit_file"));
        assert!(dispatcher.has_handler("list_files"));
        assert!(dispatcher.has_handler("search_files"));
        assert!(dispatcher.has_handler("codebase_search"));

        // Command tools
        assert!(dispatcher.has_handler("execute_command"));
        assert!(dispatcher.has_handler("read_command_output"));

        // Misc tools
        assert!(dispatcher.has_handler("ask_followup_question"));
        assert!(dispatcher.has_handler("attempt_completion"));
        assert!(dispatcher.has_handler("update_todo_list"));

        // Mode tools
        assert!(dispatcher.has_handler("switch_mode"));
        assert!(dispatcher.has_handler("new_task"));

        // Skill & Slash Command tools
        assert!(dispatcher.has_handler("skill"));
        assert!(dispatcher.has_handler("run_slash_command"));

        // MCP tools should NOT be registered
        assert!(!dispatcher.has_handler("use_mcp_tool"));
        assert!(!dispatcher.has_handler("access_mcp_resource"));
    }

    #[test]
    fn test_default_dispatcher_full_has_all_tools_including_mcp() {
        let registry = Arc::new(roo_terminal::TerminalRegistry::new());
        let dir = tempfile::tempdir().unwrap();
        let mcp_hub = Arc::new(roo_mcp::McpHub::new());
        let dispatcher = default_dispatcher_full(
            registry,
            dir.path().to_path_buf(),
            "code",
            mcp_hub,
        );

        // All core tools
        assert!(dispatcher.has_handler("read_file"));
        assert!(dispatcher.has_handler("write_to_file"));
        assert!(dispatcher.has_handler("apply_diff"));
        assert!(dispatcher.has_handler("edit_file"));
        assert!(dispatcher.has_handler("list_files"));
        assert!(dispatcher.has_handler("search_files"));
        assert!(dispatcher.has_handler("codebase_search"));

        // Command tools
        assert!(dispatcher.has_handler("execute_command"));
        assert!(dispatcher.has_handler("read_command_output"));

        // Misc tools
        assert!(dispatcher.has_handler("ask_followup_question"));
        assert!(dispatcher.has_handler("attempt_completion"));
        assert!(dispatcher.has_handler("update_todo_list"));

        // Mode tools
        assert!(dispatcher.has_handler("switch_mode"));
        assert!(dispatcher.has_handler("new_task"));

        // Skill & Slash Command tools
        assert!(dispatcher.has_handler("skill"));
        assert!(dispatcher.has_handler("run_slash_command"));

        // MCP tools
        assert!(dispatcher.has_handler("use_mcp_tool"));
        assert!(dispatcher.has_handler("access_mcp_resource"));
    }

    // ---- Skill & Slash Command handler tests ----

    #[tokio::test]
    async fn test_skill_handler_missing_skill_name() {
        let handler = SkillHandler::new();
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("skill"));
    }

    #[tokio::test]
    async fn test_skill_handler_valid() {
        let handler = SkillHandler::new();
        let ctx = make_context();
        let result = handler
            .execute(
                serde_json::json!({"skill": "react-native-dev", "args": "build a component"}),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(result.text.contains("react-native-dev"));
        assert!(result.text.contains("build a component"));
    }

    #[tokio::test]
    async fn test_skill_handler_valid_no_args() {
        let handler = SkillHandler::new();
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"skill": "flutter-dev"}), &ctx)
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(result.text.contains("flutter-dev"));
    }

    #[tokio::test]
    async fn test_skill_handler_empty_name() {
        let handler = SkillHandler::new();
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"skill": ""}), &ctx)
            .await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn test_slash_command_handler_missing_command() {
        let handler = SlashCommandHandler;
        let ctx = make_context();
        let result = handler.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_error);
        assert!(result.text.contains("command"));
    }

    #[tokio::test]
    async fn test_slash_command_handler_valid() {
        let handler = SlashCommandHandler;
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"command": "init", "args": "setup project"}), &ctx)
            .await;
        assert!(!result.is_error, "unexpected error: {}", result.text);
        assert!(result.text.contains("init"));
        assert!(result.text.contains("setup project"));
    }

    #[tokio::test]
    async fn test_slash_command_handler_no_args() {
        let handler = SlashCommandHandler;
        let ctx = make_context();
        let result = handler
            .execute(serde_json::json!({"command": "test"}), &ctx)
            .await;
        assert!(!result.is_error);
        assert!(result.text.contains("test"));
    }

    // ---- Repetition detector integration tests ----

    #[tokio::test]
    async fn test_dispatch_with_repetition_detector_allows_initial_calls() {
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("test_tool", |_params, _ctx| {
            ToolExecutionResult::success("ok")
        });
        dispatcher.set_repetition_detector(ToolRepetitionDetector::new(3));

        let ctx = make_context();
        let params = serde_json::json!({"key": "value"});

        // First 3 identical calls should succeed
        for _ in 0..3 {
            let result = dispatcher.dispatch("test_tool", params.clone(), &ctx).await;
            assert!(!result.is_error, "unexpected error: {}", result.text);
            assert_eq!(result.text, "ok");
        }
    }

    #[tokio::test]
    async fn test_dispatch_with_repetition_detector_blocks_repetition() {
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("test_tool", |_params, _ctx| {
            ToolExecutionResult::success("ok")
        });
        dispatcher.set_repetition_detector(ToolRepetitionDetector::new(3));

        let ctx = make_context();
        let params = serde_json::json!({"key": "value"});

        // First 3 identical calls succeed (count goes 0→1→2)
        for _ in 0..3 {
            let result = dispatcher.dispatch("test_tool", params.clone(), &ctx).await;
            assert!(!result.is_error);
        }

        // 4th identical call should be blocked by repetition detector
        let result = dispatcher.dispatch("test_tool", params.clone(), &ctx).await;
        assert!(!result.is_error); // it's a warning, not an error
        assert!(
            result.text.contains("Warning"),
            "expected repetition warning, got: {}",
            result.text
        );
    }

    #[tokio::test]
    async fn test_dispatch_without_repetition_detector() {
        let mut dispatcher = ToolDispatcher::new();
        dispatcher.register_fn("test_tool", |_params, _ctx| {
            ToolExecutionResult::success("ok")
        });
        // No repetition detector set

        let ctx = make_context();
        let params = serde_json::json!({"key": "value"});

        // Should allow unlimited identical calls
        for _ in 0..10 {
            let result = dispatcher.dispatch("test_tool", params.clone(), &ctx).await;
            assert!(!result.is_error);
            assert_eq!(result.text, "ok");
        }
    }
}
