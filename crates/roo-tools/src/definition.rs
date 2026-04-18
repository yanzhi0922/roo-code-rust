//! Tool definitions in OpenAI Function Calling format.
//!
//! Corresponds to `src/core/prompts/tools/native-tools/*.ts`.

use serde_json::{json, Value};

use roo_types::tool::ToolName;

// ---------------------------------------------------------------------------
// ToolDefinition
// ---------------------------------------------------------------------------

/// A tool definition in OpenAI Function Calling format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Options for customizing the native tools array.
#[derive(Debug, Clone, Default)]
pub struct NativeToolsOptions {
    /// Whether the model supports image processing (default: false).
    pub supports_images: bool,
}

// ---------------------------------------------------------------------------
// Tool definition builders
// ---------------------------------------------------------------------------

fn access_mcp_resource() -> ToolDefinition {
    ToolDefinition {
        name: "access_mcp_resource".into(),
        description: "Request to access a resource provided by a connected MCP server. Resources represent data sources that can be used as context, such as files, API responses, or system information.\n\nParameters:\n- server_name: (required) The name of the MCP server providing the resource\n- uri: (required) The URI identifying the specific resource to access\n\nExample: Accessing a weather resource\n{ \"server_name\": \"weather-server\", \"uri\": \"weather://san-francisco/current\" }\n\nExample: Accessing a file resource from an MCP server\n{ \"server_name\": \"filesystem-server\", \"uri\": \"file:///path/to/data.json\" }".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "server_name": { "type": "string", "description": "The name of the MCP server providing the resource" },
                "uri": { "type": "string", "description": "The URI identifying the specific resource to access" }
            },
            "required": ["server_name", "uri"],
            "additionalProperties": false
        }),
    }
}

fn apply_diff_tool() -> ToolDefinition {
    ToolDefinition {
        name: "apply_diff".into(),
        description: "Apply precise, targeted modifications to an existing file using one or more search/replace blocks. This tool is for surgical edits only; the 'SEARCH' block must exactly match the existing content, including whitespace and indentation. To make multiple targeted changes, provide multiple SEARCH/REPLACE blocks in the 'diff' parameter. Use the 'read_file' tool first if you are not confident in the exact content to search for.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "The path of the file to modify, relative to the current workspace directory." },
                "diff": { "type": "string", "description": "A string containing one or more search/replace blocks defining the changes. The ':start_line:' is required and indicates the starting line number of the original content. You must not add a start line for the replacement content. Each block must follow this format:\n<<<<<<< SEARCH\n:start_line:[line_number]\n-------\n[exact content to find]\n=======\n[new content to replace with]\n>>>>>>> REPLACE" }
            },
            "required": ["path", "diff"],
            "additionalProperties": false
        }),
    }
}

fn apply_patch_tool() -> ToolDefinition {
    ToolDefinition {
        name: "apply_patch".into(),
        description: "Apply patches to files using a stripped-down, file-oriented diff format. This tool supports creating new files, deleting files, and updating existing files with precise changes.\n\nThe patch format uses a simple, human-readable structure:\n\n*** Begin Patch\n[ one or more file sections ]\n*** End Patch\n\nEach file section starts with one of three headers:\n- *** Add File: <path> - Create a new file. Every following line is a + line (the initial contents).\n- *** Delete File: <path> - Remove an existing file. Nothing follows.\n- *** Update File: <path> - Patch an existing file in place.\n\nFor Update File operations:\n- May be immediately followed by *** Move to: <new path> if you want to rename the file.\n- Then one or more \"hunks\", each introduced by @@ (optionally followed by context like a class or function name).\n- Within a hunk each line starts with:\n  - ' ' (space) for context lines (unchanged)\n  - '-' for lines to remove\n  - '+' for lines to add\n\nContext guidelines:\n- Show 3 lines of code above and below each change.\n- Use @@ with a class/function name if 3 lines of context is insufficient to uniquely identify the location.\n- Multiple @@ statements can be used for deeply nested code.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "patch": { "type": "string", "description": "The complete patch text in the apply_patch format, starting with '*** Begin Patch' and ending with '*** End Patch'." }
            },
            "required": ["patch"],
            "additionalProperties": false
        }),
    }
}

fn ask_followup_question() -> ToolDefinition {
    ToolDefinition {
        name: "ask_followup_question".into(),
        description: "Ask the user a question to gather additional information needed to complete the task. Use when you need clarification or more details to proceed effectively.\n\nParameters:\n- question: (required) A clear, specific question addressing the information needed\n- follow_up: (required) A list of 2-4 suggested answers. Suggestions must be complete, actionable answers without placeholders. Optionally include mode to switch modes (code/architect/etc.)".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "question": { "type": "string", "description": "Clear, specific question that captures the missing information you need" },
                "follow_up": {
                    "type": "array",
                    "description": "Required list of 2-4 suggested responses; each suggestion must be a complete, actionable answer and may include a mode switch",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Suggested answer the user can pick" },
                            "mode": { "type": ["string", "null"], "description": "Optional mode slug to switch to if this suggestion is chosen (e.g., code, architect)" }
                        },
                        "required": ["text", "mode"],
                        "additionalProperties": false
                    },
                    "minItems": 1,
                    "maxItems": 4
                }
            },
            "required": ["question", "follow_up"],
            "additionalProperties": false
        }),
    }
}

fn attempt_completion() -> ToolDefinition {
    ToolDefinition {
        name: "attempt_completion".into(),
        description: "After each tool use, the user will respond with the result of that tool use, i.e. if it succeeded or failed, along with any reasons for failure. Once you've received the results of tool uses and can confirm that the task is complete, use this tool to present the result of your work to the user. The user may respond with feedback if they are not satisfied with the result, which you can use to make improvements and try again.\n\nIMPORTANT NOTE: This tool CANNOT be used until you've confirmed from the user that any previous tool uses were successful. Failure to do so will result in code corruption and system failure. Before using this tool, you must confirm that you've received successful results from the user for any previous tool uses. If not, then DO NOT use this tool.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "result": { "type": "string", "description": "Final result message to deliver to the user once the task is complete" }
            },
            "required": ["result"],
            "additionalProperties": false
        }),
    }
}

fn codebase_search() -> ToolDefinition {
    ToolDefinition {
        name: "codebase_search".into(),
        description: "Find files most relevant to the search query using semantic search. Searches based on meaning rather than exact text matches. By default searches entire workspace. Reuse the user's exact wording unless there's a clear reason not to - their phrasing often helps semantic search. Queries MUST be in English (translate if needed).\n\n**CRITICAL: For ANY exploration of code you haven't examined yet in this conversation, you MUST use this tool FIRST before any other search or file exploration tools.** This applies throughout the entire conversation, not just at the beginning.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Meaning-based search query describing the information you need" },
                "path": { "type": ["string", "null"], "description": "Optional subdirectory (relative to the workspace) to limit the search scope" }
            },
            "required": ["query", "path"],
            "additionalProperties": false
        }),
    }
}

fn execute_command() -> ToolDefinition {
    ToolDefinition {
        name: "execute_command".into(),
        description: "Request to execute a CLI command on the system. Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task. You must tailor your command to the user's system and provide a clear explanation of what the command does. For command chaining, use the appropriate chaining syntax for the user's shell. Prefer to execute complex CLI commands over creating executable scripts, as they are more flexible and easier to run. Prefer relative commands and paths that avoid location sensitivity for terminal consistency.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to execute" },
                "cwd": { "type": ["string", "null"], "description": "Optional working directory for the command, relative or absolute" },
                "timeout": { "type": ["number", "null"], "description": "Timeout in seconds. When exceeded, the command continues running in the background and output collected so far is returned." }
            },
            "required": ["command", "cwd", "timeout"],
            "additionalProperties": false
        }),
    }
}

fn generate_image() -> ToolDefinition {
    ToolDefinition {
        name: "generate_image".into(),
        description: "Request to generate or edit an image using AI models through OpenRouter API. This tool can create new images from text prompts or modify existing images based on your instructions. When an input image is provided, the AI will apply the requested edits, transformations, or enhancements to that image.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "prompt": { "type": "string", "description": "Text description of the image to generate or the edits to apply" },
                "path": { "type": "string", "description": "Filesystem path (relative to the workspace) where the resulting image should be saved" },
                "image": { "type": ["string", "null"], "description": "Optional path (relative to the workspace) to an existing image to edit; supports PNG, JPG, JPEG, GIF, and WEBP" }
            },
            "required": ["prompt", "path", "image"],
            "additionalProperties": false
        }),
    }
}

fn list_files() -> ToolDefinition {
    ToolDefinition {
        name: "list_files".into(),
        description: "Request to list files and directories within the specified directory. If recursive is true, it will list all files and directories recursively. If recursive is false or not provided, it will only list the top-level contents. Do not use this tool to confirm the existence of files you may have created, as the user will let you know if the files were created successfully or not.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path to inspect, relative to the workspace" },
                "recursive": { "type": "boolean", "description": "Set true to list contents recursively; false to show only the top level" }
            },
            "required": ["path", "recursive"],
            "additionalProperties": false
        }),
    }
}

fn new_task() -> ToolDefinition {
    ToolDefinition {
        name: "new_task".into(),
        description: "Create a new task instance in the chosen mode using your provided message and initial todo list (if required).\n\nCRITICAL: This tool MUST be called alone. Do NOT call this tool alongside other tools in the same message turn. If you need to gather information before delegating, use other tools in a separate turn first, then call new_task by itself in the next turn.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "mode": { "type": "string", "description": "Slug of the mode to begin the new task in (e.g., code, debug, architect)" },
                "message": { "type": "string", "description": "Initial user instructions or context for the new task" },
                "todos": { "type": ["string", "null"], "description": "Optional initial todo list written as a markdown checklist; required when the workspace mandates todos" }
            },
            "required": ["mode", "message", "todos"],
            "additionalProperties": false
        }),
    }
}

fn read_command_output() -> ToolDefinition {
    ToolDefinition {
        name: "read_command_output".into(),
        description: "Retrieve the full output from a command that was truncated in execute_command. Use this tool when:\n1. The execute_command result shows \"[OUTPUT TRUNCATED - Full output saved to artifact: cmd-XXXX.txt]\"\n2. You need to see more of the command output beyond the preview\n3. You want to search for specific content in large command output\n\nThe tool supports two modes:\n- **Read mode**: Read output starting from a byte offset with optional limit\n- **Search mode**: Filter lines matching a regex or literal pattern (like grep)".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "artifact_id": { "type": "string", "description": "The artifact filename from the truncated command output (e.g., \"cmd-1706119234567.txt\")" },
                "search": { "type": "string", "description": "Optional regex or literal pattern to filter lines (case-insensitive, like grep). Omit this parameter if not searching." },
                "offset": { "type": "number", "description": "Byte offset to start reading from (default: 0, for pagination)" },
                "limit": { "type": "number", "description": "Maximum bytes to return (default: 40KB)" }
            },
            "required": ["artifact_id"],
            "additionalProperties": false
        }),
    }
}

fn read_file_tool(supports_images: bool) -> ToolDefinition {
    let supports_note = if supports_images {
        "Supports text extraction from PDF and DOCX files. Automatically processes and returns image files (PNG, JPG, JPEG, GIF, BMP, SVG, WEBP, ICO, AVIF) for visual analysis. May not handle other binary files properly."
    } else {
        "Supports text extraction from PDF and DOCX files, but may not handle other binary files properly."
    };

    let description = format!(
        "Read a file and return its contents with line numbers for diffing or discussion. IMPORTANT: This tool reads exactly one file per call. If you need multiple files, issue multiple parallel read_file calls. Supports two modes: 'slice' (default) reads lines sequentially with offset/limit; 'indentation' extracts complete semantic code blocks around an anchor line based on indentation hierarchy. Slice mode is ideal for initial file exploration. PREFER indentation mode when you have a specific line number from search results, error messages, or definition lookups. By default, returns up to 2000 lines per file. Lines longer than 2000 characters are truncated. {supports_note}"
    );

    ToolDefinition {
        name: "read_file".into(),
        description,
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read, relative to the workspace" },
                "mode": { "type": "string", "enum": ["slice", "indentation"], "description": "Reading mode. 'slice' (default): read lines sequentially with offset/limit. 'indentation': extract complete semantic code blocks containing anchor_line." },
                "offset": { "type": "integer", "description": "1-based line offset to start reading from (slice mode, default: 1)" },
                "limit": { "type": "integer", "description": "Maximum number of lines to return (slice mode, default: 2000)" },
                "indentation": {
                    "type": "object",
                    "description": "Indentation mode options. Only used when mode='indentation'.",
                    "properties": {
                        "anchor_line": { "type": "integer", "description": "1-based line number to anchor the extraction. REQUIRED for meaningful indentation mode results." },
                        "max_levels": { "type": "integer", "description": "Maximum indentation levels to include above the anchor (0 = unlimited)." },
                        "include_siblings": { "type": "boolean", "description": "Include sibling blocks at the same indentation level (default: false)." },
                        "include_header": { "type": "boolean", "description": "Include file header content at the top of output (default: true)." },
                        "max_lines": { "type": "integer", "description": "Hard cap on lines returned for indentation mode." }
                    },
                    "required": [],
                    "additionalProperties": false
                }
            },
            "required": ["path"],
            "additionalProperties": false
        }),
    }
}

fn run_slash_command() -> ToolDefinition {
    ToolDefinition {
        name: "run_slash_command".into(),
        description: "Execute a slash command to get specific instructions or content. Slash commands are predefined templates that provide detailed guidance for common tasks.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Name of the slash command to run (e.g., init, test, deploy)" },
                "args": { "type": ["string", "null"], "description": "Optional additional context or arguments for the command" }
            },
            "required": ["command", "args"],
            "additionalProperties": false
        }),
    }
}

fn skill_tool() -> ToolDefinition {
    ToolDefinition {
        name: "skill".into(),
        description: "Load and execute a skill by name. Skills provide specialized instructions for common tasks like creating MCP servers or custom modes.\n\nUse this tool when you need to follow specific procedures documented in a skill. Available skills are listed in the AVAILABLE SKILLS section of the system prompt.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "skill": { "type": "string", "description": "Name of the skill to load (e.g., create-mcp-server, create-mode). Must match a skill name from the available skills list." },
                "args": { "type": ["string", "null"], "description": "Optional context or arguments to pass to the skill" }
            },
            "required": ["skill", "args"],
            "additionalProperties": false
        }),
    }
}

fn search_replace_tool() -> ToolDefinition {
    ToolDefinition {
        name: "search_replace".into(),
        description: "Use this tool to propose a search and replace operation on an existing file.\n\nThe tool will replace ONE occurrence of old_string with new_string in the specified file.\n\nCRITICAL REQUIREMENTS FOR USING THIS TOOL:\n\n1. UNIQUENESS: The old_string MUST uniquely identify the specific instance you want to change.\n2. SINGLE INSTANCE: This tool can only change ONE instance at a time.\n3. VERIFICATION: Before using this tool, gather enough context to uniquely identify each instance.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The path to the file you want to search and replace in." },
                "old_string": { "type": "string", "description": "The text to replace (must be unique within the file, and must match the file contents exactly)" },
                "new_string": { "type": "string", "description": "The edited text to replace the old_string (must be different from the old_string)" }
            },
            "required": ["file_path", "old_string", "new_string"],
            "additionalProperties": false
        }),
    }
}

fn edit_file_tool() -> ToolDefinition {
    ToolDefinition {
        name: "edit_file".into(),
        description: "Use this tool to replace text in an existing file, or create a new file.\n\nThis tool performs literal string replacement with support for multiple occurrences.\n\nUSAGE PATTERNS:\n\n1. MODIFY EXISTING FILE (default): Provide file_path, old_string, and new_string.\n2. CREATE NEW FILE: Set old_string to empty string \"\".\n\nCRITICAL REQUIREMENTS:\n\n1. EXACT MATCHING (BEST): The old_string should match the file contents EXACTLY.\n2. CONTEXT FOR UNIQUENESS: For single replacements, include at least 3 lines of context BEFORE and AFTER.\n3. MULTIPLE REPLACEMENTS: Set expected_replacements to the exact count.\n4. NO ESCAPING: Provide the literal text - do not escape special characters.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The path to the file to modify or create." },
                "old_string": { "type": "string", "description": "The exact literal text to replace. Use empty string to create a new file." },
                "new_string": { "type": "string", "description": "The exact literal text to replace old_string with." },
                "expected_replacements": { "type": "number", "description": "Number of replacements expected. Defaults to 1.", "minimum": 1 }
            },
            "required": ["file_path", "old_string", "new_string"],
            "additionalProperties": false
        }),
    }
}

fn edit_tool() -> ToolDefinition {
    ToolDefinition {
        name: "edit".into(),
        description: "Performs exact string replacements in files.\n\nUsage:\n- You must use your `Read` tool at least once in the conversation before editing.\n- When editing text from Read tool output, ensure you preserve the exact indentation.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context or use `replace_all` to change every instance.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "file_path": { "type": "string", "description": "The path of the file to edit (relative to the working directory)" },
                "old_string": { "type": "string", "description": "The exact text to find in the file. Must match exactly, including all whitespace, indentation, and line endings." },
                "new_string": { "type": "string", "description": "The replacement text that will replace old_string. Must include all necessary whitespace and indentation." },
                "replace_all": { "type": "boolean", "description": "When true, replaces ALL occurrences of old_string in the file. When false (default), only replaces the first occurrence.", "default": false }
            },
            "required": ["file_path", "old_string", "new_string"],
            "additionalProperties": false
        }),
    }
}

fn search_files_tool() -> ToolDefinition {
    ToolDefinition {
        name: "search_files".into(),
        description: "Request to perform a regex search across files in a specified directory, providing context-rich results. This tool searches for patterns or specific content across multiple files, displaying each match with encapsulating context.\n\nCraft your regex patterns carefully to balance specificity and flexibility. Use this tool to find code patterns, TODO comments, function definitions, or any text-based information across the project.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to search recursively, relative to the workspace" },
                "regex": { "type": "string", "description": "Rust-compatible regular expression pattern to match" },
                "file_pattern": { "type": ["string", "null"], "description": "Optional glob to limit which files are searched (e.g., *.ts)" }
            },
            "required": ["path", "regex", "file_pattern"],
            "additionalProperties": false
        }),
    }
}

fn switch_mode() -> ToolDefinition {
    ToolDefinition {
        name: "switch_mode".into(),
        description: "Request to switch to a different mode. This tool allows modes to request switching to another mode when needed, such as switching to Code mode to make code changes. The user must approve the mode switch.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "mode_slug": { "type": "string", "description": "Slug of the mode to switch to (e.g., code, ask, architect)" },
                "reason": { "type": "string", "description": "Explanation for why the mode switch is needed" }
            },
            "required": ["mode_slug", "reason"],
            "additionalProperties": false
        }),
    }
}

fn update_todo_list() -> ToolDefinition {
    ToolDefinition {
        name: "update_todo_list".into(),
        description: "Replace the entire TODO list with an updated checklist reflecting the current state. Always provide the full list; the system will overwrite the previous one. This tool is designed for step-by-step task tracking.\n\nChecklist Format:\n- Use a single-level markdown checklist (no nesting or subtasks)\n- List todos in the intended execution order\n- Status options: [ ] (pending), [x] (completed), [-] (in progress)".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "todos": { "type": "string", "description": "Full markdown checklist in execution order, using [ ] for pending, [x] for completed, and [-] for in progress" }
            },
            "required": ["todos"],
            "additionalProperties": false
        }),
    }
}

fn write_to_file() -> ToolDefinition {
    ToolDefinition {
        name: "write_to_file".into(),
        description: "Request to write content to a file. This tool is primarily used for creating new files or for scenarios where a complete rewrite of an existing file is intentionally required. If the file exists, it will be overwritten. If it doesn't exist, it will be created. This tool will automatically create any directories needed to write the file.\n\n**Important:** You should prefer using other editing tools over write_to_file when making changes to existing files, since write_to_file is slower and cannot handle large files. Use write_to_file primarily for new file creation.\n\nWhen using this tool, ALWAYS provide the COMPLETE file content in your response. This is NON-NEGOTIABLE. Partial updates or placeholders like '// rest of code unchanged' are STRICTLY FORBIDDEN.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "The path of the file to write to (relative to the current workspace directory)" },
                "content": { "type": "string", "description": "The content to write to the file. ALWAYS provide the COMPLETE intended content of the file." }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
    }
}

// ---------------------------------------------------------------------------
// get_native_tools
// ---------------------------------------------------------------------------

/// Returns all 21 native tool definitions.
///
/// Source: `src/core/prompts/tools/native-tools/index.ts` — `getNativeTools`
pub fn get_native_tools(options: NativeToolsOptions) -> Vec<ToolDefinition> {
    vec![
        access_mcp_resource(),
        apply_diff_tool(),
        apply_patch_tool(),
        ask_followup_question(),
        attempt_completion(),
        codebase_search(),
        execute_command(),
        generate_image(),
        list_files(),
        new_task(),
        read_command_output(),
        read_file_tool(options.supports_images),
        run_slash_command(),
        skill_tool(),
        search_replace_tool(),
        edit_file_tool(),
        edit_tool(),
        search_files_tool(),
        switch_mode(),
        update_todo_list(),
        write_to_file(),
    ]
}

/// Returns the canonical tool name for a given tool name string.
/// Returns None if the tool name is not recognized.
pub fn find_tool_by_name(name: &str) -> Option<ToolName> {
    ToolName::all().iter().find(|t| t.as_str() == name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_native_tools_count() {
        let tools = get_native_tools(NativeToolsOptions::default());
        assert_eq!(tools.len(), 21);
    }

    #[test]
    fn test_all_tool_names_unique() {
        let tools = get_native_tools(NativeToolsOptions::default());
        let names: std::collections::HashSet<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), 21);
    }

    #[test]
    fn test_expected_tool_names_present() {
        let tools = get_native_tools(NativeToolsOptions::default());
        let names: std::collections::HashSet<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        let expected = [
            "access_mcp_resource", "apply_diff", "apply_patch", "ask_followup_question",
            "attempt_completion", "codebase_search", "execute_command", "generate_image",
            "list_files", "new_task", "read_command_output", "read_file",
            "run_slash_command", "skill", "search_replace", "edit_file",
            "edit", "search_files", "switch_mode", "update_todo_list", "write_to_file",
        ];
        for name in &expected {
            assert!(names.contains(name), "Missing tool: {name}");
        }
    }

    #[test]
    fn test_tool_definitions_have_required_fields() {
        let tools = get_native_tools(NativeToolsOptions::default());
        for tool in &tools {
            assert!(!tool.name.is_empty());
            assert!(!tool.description.is_empty(), "Tool {} has empty description", tool.name);
            assert!(tool.parameters.get("type").is_some(), "Tool {} missing type", tool.name);
            assert!(tool.parameters.get("properties").is_some(), "Tool {} missing properties", tool.name);
            assert!(tool.parameters.get("required").is_some(), "Tool {} missing required", tool.name);
        }
    }

    #[test]
    fn test_read_file_supports_images_option() {
        let tools_no = get_native_tools(NativeToolsOptions { supports_images: false });
        let tools_yes = get_native_tools(NativeToolsOptions { supports_images: true });
        let rf_no = tools_no.iter().find(|t| t.name == "read_file").unwrap();
        let rf_yes = tools_yes.iter().find(|t| t.name == "read_file").unwrap();
        assert!(rf_yes.description.contains("PNG, JPG, JPEG, GIF, BMP, SVG, WEBP, ICO, AVIF"));
        assert!(!rf_no.description.contains("PNG, JPG, JPEG, GIF, BMP, SVG, WEBP, ICO, AVIF"));
    }

    #[test]
    fn test_find_tool_by_name() {
        assert_eq!(find_tool_by_name("execute_command"), Some(ToolName::ExecuteCommand));
        assert_eq!(find_tool_by_name("read_file"), Some(ToolName::ReadFile));
        assert_eq!(find_tool_by_name("unknown_tool"), None);
    }
}
