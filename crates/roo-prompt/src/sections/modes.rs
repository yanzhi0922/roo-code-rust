//! Modes section.
//!
//! Source: `src/core/prompts/sections/modes.ts`

use roo_types::mode::ModeConfig;

/// Returns the modes section listing all available modes.
///
/// Source: `src/core/prompts/sections/modes.ts` — `getModesSection`
pub fn get_modes_section(modes: &[ModeConfig]) -> String {
    let modes_content = modes
        .iter()
        .map(|mode| {
            let description = if let Some(wtu) = &mode.when_to_use {
                if !wtu.trim().is_empty() {
                    // Use whenToUse as the primary description, indenting subsequent lines
                    wtu.replace('\n', "\n    ")
                } else {
                    // Fallback to the first sentence of roleDefinition
                    mode.role_definition.split('.').next().unwrap_or("").to_string()
                }
            } else {
                // Fallback to the first sentence of roleDefinition
                mode.role_definition.split('.').next().unwrap_or("").to_string()
            };
            format!("  * \"{}\" mode ({}) - {}", mode.name, mode.slug, description)
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"====

MODES

- These are the currently available modes:
{modes_content}"#
    )
}
