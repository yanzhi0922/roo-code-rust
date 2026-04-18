//! Tool use section.
//!
//! Source: `src/core/prompts/sections/tool-use.ts`

/// Returns the shared tool use section.
///
/// Source: `src/core/prompts/sections/tool-use.ts` — `getSharedToolUseSection`
pub fn get_shared_tool_use_section() -> &'static str {
    r#"====

TOOL USE

You have access to a set of tools that are executed upon the user's approval. Use the provider-native tool-calling mechanism. Do not include XML markup or examples. You must call at least one tool per assistant response. Prefer calling as many tools as are reasonably needed in a single response to reduce back-and-forth and complete tasks faster."#
}
