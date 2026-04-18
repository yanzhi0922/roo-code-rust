//! Markdown formatting section.
//!
//! Source: `src/core/prompts/sections/markdown-formatting.ts`

/// Returns the markdown formatting rules section.
///
/// Source: `src/core/prompts/sections/markdown-formatting.ts` — `markdownFormattingSection`
pub fn markdown_formatting_section() -> &'static str {
    r#"====

MARKDOWN RULES

ALL responses MUST show ANY `language construct` OR filename reference as clickable, exactly as [`filename OR language.declaration()`](relative/file/path.ext:line); line is required for `syntax` and optional for filename links. This applies to ALL markdown responses and ALSO those in attempt_completion"#
}
