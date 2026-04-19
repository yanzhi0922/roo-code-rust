//! Roo CLI — command-line interface for Roo Code Rust.
//!
//! Supports sending messages to AI providers and streaming responses.
//! Implements a full tool-call execution loop: user input → API call →
//! tool execution → feedback → loop until text-only response.

use std::io::{self, Write as IoWrite};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;

use roo_provider::handler::{CreateMessageMetadata, Provider};
use roo_provider_anthropic::{AnthropicConfig, AnthropicHandler};
use roo_task::tool_dispatcher::{
    ToolContext, ToolDispatcher, ToolExecutionResult,
    default_dispatcher_with_terminal,
};
use roo_terminal::TerminalRegistry;
use roo_tools::definition::{NativeToolsOptions, get_native_tools};
use roo_types::api::{ApiMessage, ApiStreamChunk, ContentBlock, MessageRole, ToolResultContent};

// ---------------------------------------------------------------------------
// CLI argument definitions
// ---------------------------------------------------------------------------

/// Roo CLI — interact with AI providers from the command line.
#[derive(Debug, Parser)]
#[command(name = "roo", version, about = "Roo Code CLI")]
struct Cli {
    /// AI provider to use (e.g. "anthropic").
    #[arg(long, default_value = "anthropic")]
    provider: String,

    /// Base URL for the provider API.
    #[arg(long)]
    base_url: Option<String>,

    /// API key for authentication.
    #[arg(long)]
    api_key: Option<String>,

    /// Model ID to use (e.g. "claude-sonnet-4-20250514").
    #[arg(long)]
    model: Option<String>,

    /// Temperature for generation (0.0 – 1.0).
    #[arg(long)]
    temperature: Option<f64>,

    /// Enable extended / thinking mode (Anthropic only).
    #[arg(long)]
    thinking: bool,

    /// Max thinking tokens when extended thinking is enabled.
    #[arg(long)]
    max_thinking_tokens: Option<u64>,

    /// Request timeout in milliseconds.
    #[arg(long)]
    timeout: Option<u64>,

    /// System prompt override. If omitted, a default prompt is generated.
    #[arg(long)]
    system_prompt: Option<String>,

    /// Single message to send (non-interactive mode).
    #[arg(short, long)]
    message: Option<String>,

    /// Launch interactive REPL mode.
    #[arg(short, long)]
    interactive: bool,

    /// Path to a JSON configuration file.
    #[arg(short, long)]
    config: Option<String>,
}

/// Configuration loaded from a JSON file.
#[derive(Debug, serde::Deserialize)]
struct ConfigFile {
    provider: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    temperature: Option<f64>,
    thinking: Option<bool>,
    max_thinking_tokens: Option<u64>,
    timeout: Option<u64>,
    system_prompt: Option<String>,
}

// ---------------------------------------------------------------------------
// Collected tool call (accumulated from stream chunks)
// ---------------------------------------------------------------------------

/// A tool call accumulated from streaming chunks.
#[derive(Debug, Clone)]
struct CollectedToolCall {
    id: String,
    name: String,
    arguments: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Install a default tracing subscriber so provider internals can log.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    // Load optional config file and merge with CLI flags (CLI takes priority).
    let config = load_config(&cli)?;

    // Build the provider handler.
    let provider_name = config.provider.as_deref().unwrap_or("anthropic");
    let handler = build_handler(provider_name, &config)
        .context("Failed to create provider handler")?;

    // Build the system prompt.
    let system_prompt = config.system_prompt.unwrap_or_else(|| {
        roo_prompt::build_system_prompt(
            &std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| ".".into()),
            "code",
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            &[],
            &os_info(),
            &shell_name(),
            &home_dir(),
        )
    });

    // Build tool definitions (JSON values for the API).
    let tool_defs = get_native_tools(NativeToolsOptions::default());
    let tools_json: Vec<serde_json::Value> = tool_defs
        .iter()
        .map(|td| serde_json::to_value(td).unwrap_or_default())
        .collect();

    // Build the terminal registry and tool dispatcher.
    let registry = Arc::new(TerminalRegistry::new());
    let working_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let output_dir = std::env::temp_dir().join("roo-cli-output");
    std::fs::create_dir_all(&output_dir).ok();

    let dispatcher = default_dispatcher_with_terminal(registry, output_dir, "code");

    if cli.interactive {
        run_interactive(&*handler, &system_prompt, &tools_json, &dispatcher, &working_dir).await
    } else if let Some(msg) = &cli.message {
        run_single(
            &*handler,
            &system_prompt,
            &tools_json,
            &dispatcher,
            &working_dir,
            msg,
        )
        .await
    } else {
        anyhow::bail!(
            "Provide --message <text> for single-shot mode or --interactive for REPL mode."
        )
    }
}

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

/// Merge CLI flags with an optional JSON config file. CLI flags take priority.
fn load_config(cli: &Cli) -> Result<ConfigFile> {
    let mut cfg = if let Some(path) = &cli.config {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {path}"))?;
        serde_json::from_str::<ConfigFile>(&raw)
            .with_context(|| format!("Failed to parse config file: {path}"))?
    } else {
        ConfigFile {
            provider: None,
            base_url: None,
            api_key: None,
            model: None,
            temperature: None,
            thinking: None,
            max_thinking_tokens: None,
            timeout: None,
            system_prompt: None,
        }
    };

    // CLI flags override config file values.
    if cli.provider != "anthropic" || cfg.provider.is_none() {
        cfg.provider = Some(cli.provider.clone());
    }
    if cli.base_url.is_some() {
        cfg.base_url = cli.base_url.clone();
    }
    if cli.api_key.is_some() {
        cfg.api_key = cli.api_key.clone();
    }
    if cli.model.is_some() {
        cfg.model = cli.model.clone();
    }
    if cli.temperature.is_some() {
        cfg.temperature = cli.temperature;
    }
    if cli.thinking {
        cfg.thinking = Some(true);
    }
    if cli.max_thinking_tokens.is_some() {
        cfg.max_thinking_tokens = cli.max_thinking_tokens;
    }
    if cli.timeout.is_some() {
        cfg.timeout = cli.timeout;
    }
    if cli.system_prompt.is_some() {
        cfg.system_prompt = cli.system_prompt.clone();
    }

    Ok(cfg)
}

// ---------------------------------------------------------------------------
// Provider construction
// ---------------------------------------------------------------------------

/// Build a boxed Provider based on the provider name.
fn build_handler(
    provider_name: &str,
    config: &ConfigFile,
) -> Result<Box<dyn Provider>> {
    match provider_name {
        "anthropic" => {
            let api_key = config
                .api_key
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--api-key is required for the anthropic provider"))?;
            let base_url = config
                .base_url
                .clone()
                .unwrap_or_else(|| AnthropicConfig::DEFAULT_BASE_URL.to_string());

            let anthropic_config = AnthropicConfig {
                api_key: api_key.to_string(),
                base_url,
                model_id: config.model.clone(),
                temperature: config.temperature,
                use_extended_thinking: config.thinking,
                max_thinking_tokens: config.max_thinking_tokens,
                request_timeout: config.timeout,
            };

            let handler =
                AnthropicHandler::new(anthropic_config).context("Failed to create Anthropic handler")?;
            Ok(Box::new(handler))
        }
        other => anyhow::bail!(
            "Unsupported provider: '{other}'. Currently supported: anthropic"
        ),
    }
}

// ---------------------------------------------------------------------------
// Single-shot mode (with tool execution loop)
// ---------------------------------------------------------------------------

/// Send one message and run the full tool-call loop.
async fn run_single(
    handler: &dyn Provider,
    system_prompt: &str,
    tools_json: &[serde_json::Value],
    dispatcher: &ToolDispatcher,
    working_dir: &PathBuf,
    message: &str,
) -> Result<()> {
    let mut messages = vec![ApiMessage {
        role: MessageRole::User,
        content: vec![ContentBlock::Text {
            text: message.to_string(),
        }],
        reasoning: None,
        ts: None,
        truncation_parent: None,
        is_truncation_marker: None,
        truncation_id: None,
        condense_parent: None,
        is_summary: None,
        condense_id: None,
    }];

    loop {
        let metadata = CreateMessageMetadata::default();

        let stream = handler
            .create_message(system_prompt, messages.clone(), Some(tools_json.to_vec()), metadata)
            .await
            .context("Failed to create message")?;

        let (assistant_text, tool_calls) = collect_stream(stream).await?;

        // Build assistant message content blocks.
        let mut assistant_content: Vec<ContentBlock> = Vec::new();
        if !assistant_text.is_empty() {
            assistant_content.push(ContentBlock::Text {
                text: assistant_text.clone(),
            });
        }
        for tc in &tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
            assistant_content.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                input,
            });
        }

        messages.push(ApiMessage {
            role: MessageRole::Assistant,
            content: assistant_content,
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        });

        // If no tool calls, we're done.
        if tool_calls.is_empty() {
            println!();
            return Ok(());
        }

        // Execute tool calls and collect results.
        let tool_results = execute_tool_calls(tool_calls, dispatcher, working_dir).await;

        messages.push(ApiMessage {
            role: MessageRole::User,
            content: tool_results,
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        });

        // Continue loop — let the model process tool results.
    }
}

// ---------------------------------------------------------------------------
// Interactive REPL mode (with tool execution loop)
// ---------------------------------------------------------------------------

/// Interactive read-eval-print loop with full tool-call support.
async fn run_interactive(
    handler: &dyn Provider,
    system_prompt: &str,
    tools_json: &[serde_json::Value],
    dispatcher: &ToolDispatcher,
    working_dir: &PathBuf,
) -> Result<()> {
    println!("Roo CLI — interactive mode (type :quit or Ctrl-C to exit)\n");

    let mut conversation: Vec<ApiMessage> = Vec::new();

    loop {
        // Read user input.
        let input = read_line("you> ")?;
        let trimmed = input.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == ":quit" || trimmed == ":q" {
            println!("Bye!");
            return Ok(());
        }
        if trimmed == ":clear" {
            conversation.clear();
            println!("[Conversation cleared]\n");
            continue;
        }

        // Append user message.
        conversation.push(ApiMessage {
            role: MessageRole::User,
            content: vec![ContentBlock::Text { text: trimmed }],
            reasoning: None,
            ts: None,
            truncation_parent: None,
            is_truncation_marker: None,
            truncation_id: None,
            condense_parent: None,
            is_summary: None,
            condense_id: None,
        });

        // Tool-call loop: keep calling the API until we get a text-only response.
        loop {
            let metadata = CreateMessageMetadata::default();

            let stream = handler
                .create_message(
                    system_prompt,
                    conversation.clone(),
                    Some(tools_json.to_vec()),
                    metadata,
                )
                .await
                .context("Failed to create message")?;

            print!("ai> ");
            io::stdout().flush().ok();

            let (assistant_text, tool_calls) = collect_stream(stream).await?;

            // Build assistant message content blocks.
            let mut assistant_content: Vec<ContentBlock> = Vec::new();
            if !assistant_text.is_empty() {
                assistant_content.push(ContentBlock::Text {
                    text: assistant_text.clone(),
                });
            }
            for tc in &tool_calls {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);
                assistant_content.push(ContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    input,
                });
            }

            conversation.push(ApiMessage {
                role: MessageRole::Assistant,
                content: assistant_content,
                reasoning: None,
                ts: None,
                truncation_parent: None,
                is_truncation_marker: None,
                truncation_id: None,
                condense_parent: None,
                is_summary: None,
                condense_id: None,
            });

            // If no tool calls, we're done — wait for next user input.
            if tool_calls.is_empty() {
                println!();
                break;
            }

            // Execute tool calls and collect results.
            let tool_results = execute_tool_calls(tool_calls, dispatcher, working_dir).await;

            conversation.push(ApiMessage {
                role: MessageRole::User,
                content: tool_results,
                reasoning: None,
                ts: None,
                truncation_parent: None,
                is_truncation_marker: None,
                truncation_id: None,
                condense_parent: None,
                is_summary: None,
                condense_id: None,
            });

            // Continue loop — let the model process tool results.
        }
    }
}

// ---------------------------------------------------------------------------
// Stream collection — accumulate text + tool calls from API stream
// ---------------------------------------------------------------------------

/// Collect all chunks from an API stream, printing text in real-time,
/// and return the accumulated text and tool calls.
async fn collect_stream(
    mut stream: roo_provider::handler::ApiStream,
) -> Result<(String, Vec<CollectedToolCall>)> {
    let mut collected_text = String::new();
    let mut tool_calls: Vec<CollectedToolCall> = Vec::new();
    // Map from tool call id → index in tool_calls vec (for delta accumulation).
    let mut tool_call_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => process_chunk(
                &chunk,
                &mut collected_text,
                &mut tool_calls,
                &mut tool_call_index,
            ),
            Err(e) => {
                eprintln!("\n[stream error] {e}");
            }
        }
    }

    // Deduplicate tool call IDs — some providers (e.g. MiniMax) may return
    // multiple tool calls with the same ID. Append a suffix to duplicates so
    // that subsequent API calls don't fail with "duplicate tool_call id".
    {
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for tc in &mut tool_calls {
            if seen_ids.contains(&tc.id) {
                let original_id = tc.id.clone();
                let mut suffix = 2u32;
                loop {
                    let new_id = format!("{}_dedup{}", original_id, suffix);
                    if !seen_ids.contains(&new_id) {
                        tc.id = new_id;
                        break;
                    }
                    suffix += 1;
                }
                eprintln!("\n\x1b[33m[warn] deduplicated tool call id: {} -> {}\x1b[0m", original_id, tc.id);
            }
            seen_ids.insert(tc.id.clone());
        }
    }

    Ok((collected_text, tool_calls))
}

/// Process a single stream chunk, printing text and accumulating tool calls.
fn process_chunk(
    chunk: &ApiStreamChunk,
    collected_text: &mut String,
    tool_calls: &mut Vec<CollectedToolCall>,
    tool_call_index: &mut std::collections::HashMap<String, usize>,
) {
    match chunk {
        ApiStreamChunk::Text { text } => {
            print!("{text}");
            io::stdout().flush().ok();
            collected_text.push_str(text);
        }
        ApiStreamChunk::Reasoning { text, .. } => {
            print!("\x1b[90m{text}\x1b[0m");
            io::stdout().flush().ok();
        }
        ApiStreamChunk::ThinkingComplete { signature } => {
            print!(
                "\x1b[90m[thinking complete: sig={}]\x1b[0m",
                &signature[..signature.len().min(16)]
            );
        }
        ApiStreamChunk::Usage {
            input_tokens,
            output_tokens,
            cache_write_tokens,
            cache_read_tokens,
            reasoning_tokens,
            total_cost,
        } => {
            eprintln!(
                "\n[usage] in={input_tokens} out={output_tokens} cache_w={:?} cache_r={:?} reasoning={:?} cost={:?}",
                cache_write_tokens, cache_read_tokens, reasoning_tokens, total_cost,
            );
        }
        ApiStreamChunk::ToolCallStart { id, name } => {
            print!("\n\x1b[36m[tool: {name}]\x1b[0m ");
            io::stdout().flush().ok();
            let idx = tool_calls.len();
            tool_calls.push(CollectedToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: String::new(),
            });
            tool_call_index.insert(id.clone(), idx);
        }
        ApiStreamChunk::ToolCallDelta { id, delta } => {
            if let Some(&idx) = tool_call_index.get(id) {
                tool_calls[idx].arguments.push_str(delta);
            }
        }
        ApiStreamChunk::ToolCallEnd { .. } => {
            // Tool call complete — nothing extra to print.
        }
        ApiStreamChunk::ToolCall {
            id,
            name,
            arguments,
        } => {
            // Complete tool call in one shot.
            print!("\n\x1b[36m[tool: {name}]\x1b[0m ");
            io::stdout().flush().ok();
            let idx = tool_calls.len();
            tool_calls.push(CollectedToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            });
            tool_call_index.insert(id.clone(), idx);
        }
        ApiStreamChunk::ToolCallPartial {
            index,
            id,
            name,
            arguments,
        } => {
            // Partial tool call from OpenAI-compatible providers.
            let idx = *index as usize;
            if idx >= tool_calls.len() {
                tool_calls.resize(idx + 1, CollectedToolCall {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
            }
            if let Some(partial_id) = id {
                tool_call_index.insert(partial_id.clone(), idx);
                tool_calls[idx].id = partial_id.clone();
            }
            if let Some(partial_name) = name {
                if tool_calls[idx].name.is_empty() {
                    print!("\n\x1b[36m[tool: {partial_name}]\x1b[0m ");
                    io::stdout().flush().ok();
                }
                tool_calls[idx].name = partial_name.clone();
            }
            if let Some(partial_args) = arguments {
                tool_calls[idx].arguments.push_str(partial_args);
            }
        }
        ApiStreamChunk::Grounding { sources } => {
            eprintln!("\n[grounding: {} sources]", sources.len());
        }
        ApiStreamChunk::Error { error, message } => {
            eprintln!("\n\x1b[31m[error] {error}: {message}\x1b[0m");
        }
    }
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

/// Execute a list of tool calls and return content blocks for the results.
async fn execute_tool_calls(
    tool_calls: Vec<CollectedToolCall>,
    dispatcher: &ToolDispatcher,
    working_dir: &PathBuf,
) -> Vec<ContentBlock> {
    let mut results: Vec<ContentBlock> = Vec::new();

    for tc in &tool_calls {
        // Parse the arguments as JSON Value.
        let params: serde_json::Value =
            serde_json::from_str(&tc.arguments).unwrap_or(serde_json::Value::Null);

        // Display tool invocation.
        let args_preview = serde_json::to_string_pretty(&params)
            .unwrap_or_else(|_| tc.arguments.clone());
        println!(
            "\n\x1b[33m[executing] {}({})\x1b[0m",
            tc.name,
            truncate_str(&args_preview, 200)
        );

        let context = ToolContext::new(working_dir, "cli-session");

        let result = dispatcher.dispatch(&tc.name, params, &context).await;

        let (output_text, is_error) = match result {
            ToolExecutionResult {
                text,
                is_error: err,
                ..
            } => {
                let preview = truncate_str(&text, 300);
                if err {
                    eprintln!("\x1b[31m[result] Error: {}\x1b[0m", preview);
                } else {
                    println!("\x1b[32m[result] {}\x1b[0m", preview);
                }
                (text, err)
            }
        };

        results.push(ContentBlock::ToolResult {
            tool_use_id: tc.id.clone(),
            content: vec![ToolResultContent::Text {
                text: if is_error {
                    format!("Error: {}", output_text)
                } else {
                    output_text
                },
            }],
            is_error: if is_error { Some(true) } else { None },
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Read a line from stdin with a prompt.
fn read_line(prompt: &str) -> Result<String> {
    print!("{prompt}");
    io::stdout().flush().context("Failed to flush stdout")?;
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .context("Failed to read from stdin")?;
    Ok(buf)
}

/// Truncate a string to `max_len` characters, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut truncated = s[..max_len].to_string();
        truncated.push_str("...");
        truncated
    }
}

/// Best-effort OS info string.
fn os_info() -> String {
    if cfg!(windows) {
        "Windows".to_string()
    } else if cfg!(target_os = "macos") {
        "macOS".to_string()
    } else if cfg!(target_os = "linux") {
        "Linux".to_string()
    } else {
        "Unknown".to_string()
    }
}

/// Best-effort shell name.
fn shell_name() -> String {
    if cfg!(windows) {
        "cmd.exe".to_string()
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".into())
    }
}

/// Best-effort home directory.
fn home_dir() -> String {
    if cfg!(windows) {
        std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into())
    } else {
        std::env::var("HOME").unwrap_or_else(|_| ".".into())
    }
}
