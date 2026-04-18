//! Converts Anthropic content blocks to Gemini Part format.
//!
//! Derived from `src/api/transform/gemini-format.ts`.

use serde::{Deserialize, Serialize};

use roo_types::api::{ApiMessage, ContentBlock, ImageSource, MessageRole, ToolResultContent};

// ---------------------------------------------------------------------------
// Gemini types
// ---------------------------------------------------------------------------

/// A Gemini API Part — polymorphic content unit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiPart {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<GeminiInlineData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<GeminiFunctionCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GeminiFunctionResponse>,
    /// Thought signature for Gemini 3+ thinking models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

impl GeminiPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            inline_data: None,
            function_call: None,
            function_response: None,
            thought_signature: None,
        }
    }

    pub fn inline_data(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            text: None,
            inline_data: Some(GeminiInlineData {
                data: data.into(),
                mime_type: mime_type.into(),
            }),
            function_call: None,
            function_response: None,
            thought_signature: None,
        }
    }

    pub fn function_call(name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            text: None,
            inline_data: None,
            function_call: Some(GeminiFunctionCall {
                name: name.into(),
                args,
            }),
            function_response: None,
            thought_signature: None,
        }
    }

    pub fn function_response(name: impl Into<String>, content: impl Into<String>) -> Self {
        let name = name.into();
        let response_name = name.clone();
        Self {
            text: None,
            inline_data: None,
            function_call: None,
            function_response: Some(GeminiFunctionResponse {
                name,
                response: GeminiFunctionResponseContent {
                    name: response_name,
                    content: content.into(),
                },
            }),
            thought_signature: None,
        }
    }

    fn has_function_call(&self) -> bool {
        self.function_call.is_some()
    }

    fn has_thought_signature(&self) -> bool {
        self.thought_signature.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiInlineData {
    pub data: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: GeminiFunctionResponseContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFunctionResponseContent {
    pub name: String,
    pub content: String,
}

/// A Gemini API Content — a role + list of parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

// ---------------------------------------------------------------------------
// Conversion options
// ---------------------------------------------------------------------------

/// Options for converting Anthropic content to Gemini format.
pub struct GeminiConversionOptions {
    /// Whether to include thought signatures (default: true).
    pub include_thought_signatures: bool,
    /// Map from tool_use_id to tool name (built from conversation history).
    pub tool_id_to_name: std::collections::HashMap<String, String>,
}

impl Default for GeminiConversionOptions {
    fn default() -> Self {
        Self {
            include_thought_signatures: true,
            tool_id_to_name: std::collections::HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/// Converts Anthropic content blocks to Gemini Parts.
///
/// Source: `src/api/transform/gemini-format.ts` — `convertAnthropicContentToGemini`
pub fn convert_anthropic_content_to_gemini(
    content: &[ContentBlock],
    options: &GeminiConversionOptions,
) -> Vec<GeminiPart> {
    let include_thought_signatures = options.include_thought_signatures;

    // First pass: find thoughtSignature if it exists in the content blocks
    // (In our current type system, we don't have a dedicated ThoughtSignature block,
    //  but we handle it through the Thinking block's signature field)
    let active_thought_signature: Option<String> = content.iter().find_map(|block| {
        if let ContentBlock::Thinking { signature, .. } = block {
            Some(signature.clone())
        } else {
            None
        }
    });

    // Determine the signature to attach to function calls.
    // If we're in a mode that expects signatures:
    // 1. Use the actual signature if we found one in the history/content.
    // 2. Fallback to "skip_thought_signature_validator" if missing.
    let function_call_signature = if include_thought_signatures {
        active_thought_signature
            .clone()
            .or_else(|| Some("skip_thought_signature_validator".to_string()))
    } else {
        None
    };

    let mut parts: Vec<GeminiPart> = Vec::new();

    for block in content {
        match block {
            ContentBlock::Text { text } => {
                parts.push(GeminiPart::text(text));
            }
            ContentBlock::Image { source } => {
                if let ImageSource::Base64 { data, media_type } = source {
                    parts.push(GeminiPart::inline_data(data, media_type));
                } else {
                    // Unsupported image source type (URL-based images need fetching first)
                    // Skip for now
                }
            }
            ContentBlock::ToolUse { name, input, .. } => {
                let mut part = GeminiPart::function_call(name, input.clone());
                // Inject the thoughtSignature into the functionCall part if required.
                if let Some(ref sig) = function_call_signature {
                    part.thought_signature = Some(sig.clone());
                }
                parts.push(part);
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content: result_content,
                ..
            } => {
                if result_content.is_empty() {
                    continue;
                }

                // Get tool name from the map
                let tool_name = options.tool_id_to_name.get(tool_use_id).cloned();
                let tool_name = match tool_name {
                    Some(name) => name,
                    None => {
                        // Tool name not found — skip this block
                        continue;
                    }
                };

                let mut text_parts: Vec<String> = Vec::new();
                let mut image_parts: Vec<GeminiPart> = Vec::new();

                for item in result_content {
                    match item {
                        ToolResultContent::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ToolResultContent::Image { source } => {
                            if let ImageSource::Base64 { data, media_type } = source {
                                image_parts.push(GeminiPart::inline_data(data, media_type));
                            }
                        }
                    }
                }

                // Create content text with a note about images if present
                let content_text = if !image_parts.is_empty() {
                    format!(
                        "{}\n\n(See next part for image)",
                        text_parts.join("\n\n")
                    )
                } else {
                    text_parts.join("\n\n")
                };

                // Return function response followed by any images
                parts.push(GeminiPart::function_response(&tool_name, &content_text));
                parts.extend(image_parts);
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {
                // Skip thinking blocks — they're metadata from other providers
                // that don't need to be sent to Gemini
            }
        }
    }

    // Post-processing:
    // 1) Ensure thought signature is attached if required
    // 2) For multiple function calls in a single message, keep the signature
    //    only on the first functionCall part to match Gemini 3 parallel-calling behavior.
    if include_thought_signatures {
        if let Some(ref sig) = active_thought_signature {
            let has_signature = parts.iter().any(|p| p.has_thought_signature());

            if !has_signature {
                if let Some(first) = parts.first_mut() {
                    first.thought_signature = Some(sig.clone());
                } else {
                    // Create a placeholder part if no other content exists
                    parts.push(GeminiPart {
                        text: Some(String::new()),
                        thought_signature: Some(sig.clone()),
                        ..Default::default()
                    });
                }
            }
        }

        // Keep signature only on the first functionCall part
        let mut seen_first_function_call = false;
        for part in &mut parts {
            if part.has_function_call() {
                if !seen_first_function_call {
                    seen_first_function_call = true;
                } else {
                    // Remove signature from subsequent function calls
                    part.thought_signature = None;
                }
            }
        }
    }

    parts
}

/// Converts an Anthropic message to Gemini Content format.
///
/// Source: `src/api/transform/gemini-format.ts` — `convertAnthropicMessageToGemini`
pub fn convert_anthropic_message_to_gemini(
    message: &ApiMessage,
    options: &GeminiConversionOptions,
) -> Vec<GeminiContent> {
    let parts = convert_anthropic_content_to_gemini(&message.content, options);

    if parts.is_empty() {
        return vec![];
    }

    let role = match message.role {
        MessageRole::Assistant => "model",
        MessageRole::User => "user",
    };

    vec![GeminiContent {
        role: role.to_string(),
        parts,
    }]
}

/// Builds a tool ID to name mapping from a conversation history.
/// Scans all messages for `tool_use` blocks and maps their IDs to names.
pub fn build_tool_id_to_name_map(messages: &[ApiMessage]) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for message in messages {
        for block in &message.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                map.insert(id.clone(), name.clone());
            }
        }
    }
    map
}
