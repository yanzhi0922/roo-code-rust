//! Roo CLI — command-line interface for Roo Code Rust.
//!
//! Supports sending messages to AI providers and streaming responses.
//! Currently supports the `anthropic` provider (Anthropic Messages API).

use std::io::{self, Write as IoWrite};

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;

use roo_provider::handler::{CreateMessageMetadata, Provider};
use roo_provider_anthropic::{AnthropicConfig, AnthropicHandler};
use roo_types::api::{ApiMessage, ContentBlock, MessageRole};

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

    if cli.interactive {
        run_interactive(&*handler, &system_prompt).await
    } else if let Some(msg) = &cli.message {
        run_single(&*handler, &system_prompt, msg).await
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
// Single-shot mode
// ---------------------------------------------------------------------------

/// Send one message and stream the response to stdout.
async fn run_single(handler: &dyn Provider, system_prompt: &str, message: &str) -> Result<()> {
    let messages = vec![ApiMessage {
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

    let metadata = CreateMessageMetadata::default();

    let stream = handler
        .create_message(system_prompt, messages, None, metadata)
        .await
        .context("Failed to create message")?;

    print_stream(stream).await
}

// ---------------------------------------------------------------------------
// Interactive REPL mode
// ---------------------------------------------------------------------------

/// Simple interactive read-eval-print loop.
async fn run_interactive(handler: &dyn Provider, system_prompt: &str) -> Result<()> {
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

        let metadata = CreateMessageMetadata::default();

        let stream = handler
            .create_message(system_prompt, conversation.clone(), None, metadata)
            .await
            .context("Failed to create message")?;

        print!("ai> ");
        io::stdout().flush().ok();

        let assistant_text = print_stream_collect(stream).await?;

        // Append assistant response to conversation history.
        if !assistant_text.is_empty() {
            conversation.push(ApiMessage {
                role: MessageRole::Assistant,
                content: vec![ContentBlock::Text { text: assistant_text }],
                reasoning: None,
                ts: None,
                truncation_parent: None,
                is_truncation_marker: None,
                truncation_id: None,
                condense_parent: None,
                is_summary: None,
                condense_id: None,
            });
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Stream printing helpers
// ---------------------------------------------------------------------------

/// Print all chunks from an API stream to stdout (fire-and-forget).
async fn print_stream(
    mut stream: roo_provider::handler::ApiStream,
) -> Result<()> {
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => print_chunk(&chunk)?,
            Err(e) => eprintln!("\n[stream error] {e}"),
        }
    }
    println!(); // trailing newline
    Ok(())
}

/// Print all chunks and collect the assistant text for conversation history.
async fn print_stream_collect(
    mut stream: roo_provider::handler::ApiStream,
) -> Result<String> {
    let mut collected = String::new();
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                if let Some(text) = print_chunk_collected(&chunk) {
                    collected.push_str(&text);
                }
            }
            Err(e) => eprintln!("\n[stream error] {e}"),
        }
    }
    Ok(collected)
}

/// Print a single stream chunk. Returns `true` if text was printed.
fn print_chunk(chunk: &roo_types::api::ApiStreamChunk) -> Result<()> {
    match chunk {
        roo_types::api::ApiStreamChunk::Text { text } => {
            print!("{text}");
            io::stdout().flush().ok();
        }
        roo_types::api::ApiStreamChunk::Reasoning { text, .. } => {
            print!("\x1b[90m{text}\x1b[0m");
            io::stdout().flush().ok();
        }
        roo_types::api::ApiStreamChunk::ThinkingComplete { signature } => {
            print!("\x1b[90m[thinking complete: sig={}]\x1b[0m", &signature[..signature.len().min(16)]);
        }
        roo_types::api::ApiStreamChunk::Usage {
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
        roo_types::api::ApiStreamChunk::ToolCallStart { id, name } => {
            print!("\n[tool call: {name} (id={id})]");
            io::stdout().flush().ok();
        }
        roo_types::api::ApiStreamChunk::ToolCallDelta { delta, .. } => {
            print!("{delta}");
            io::stdout().flush().ok();
        }
        roo_types::api::ApiStreamChunk::ToolCall { id, name, arguments } => {
            print!("\n[tool call: {name} (id={id}) args={arguments}]");
            io::stdout().flush().ok();
        }
        roo_types::api::ApiStreamChunk::ToolCallEnd { id } => {
            print!("[tool end: {id}]");
            io::stdout().flush().ok();
        }
        roo_types::api::ApiStreamChunk::ToolCallPartial { index, id, name, arguments } => {
            print!(
                "\x1b[36m[tool partial #{index} id={:?} name={:?} args={:?}]\x1b[0m",
                id, name, arguments,
            );
            io::stdout().flush().ok();
        }
        roo_types::api::ApiStreamChunk::Grounding { sources } => {
            eprintln!("\n[grounding: {} sources]", sources.len());
        }
        roo_types::api::ApiStreamChunk::Error { error, message } => {
            eprintln!("\n\x1b[31m[error] {error}: {message}\x1b[0m");
        }
    }
    Ok(())
}

/// Like `print_chunk` but also returns the text content for conversation history.
fn print_chunk_collected(chunk: &roo_types::api::ApiStreamChunk) -> Option<String> {
    match chunk {
        roo_types::api::ApiStreamChunk::Text { text } => {
            print!("{text}");
            io::stdout().flush().ok();
            Some(text.clone())
        }
        roo_types::api::ApiStreamChunk::Reasoning { text, .. } => {
            print!("\x1b[90m{text}\x1b[0m");
            io::stdout().flush().ok();
            None
        }
        roo_types::api::ApiStreamChunk::Usage {
            input_tokens,
            output_tokens,
            ..
        } => {
            eprintln!("\n[usage] in={input_tokens} out={output_tokens}");
            None
        }
        roo_types::api::ApiStreamChunk::Error { error, message } => {
            eprintln!("\n\x1b[31m[error] {error}: {message}\x1b[0m");
            None
        }
        _ => {
            // Delegate other variants to print_chunk for display.
            print_chunk(chunk).ok();
            None
        }
    }
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
