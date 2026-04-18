//! Terminal details formatting.
//!
//! Ported from `src/core/environment/getEnvironmentDetails.ts` (terminal sections).

use crate::types::{InactiveTerminalInfo, TerminalInfo};

/// Format the "Actively Running Terminals" section.
///
/// Produces output like:
/// ```text
///
/// # Actively Running Terminals
/// ## Terminal {id} (Active)
/// ### Working Directory: `{cwd}`
/// ### Original command: `{command}`
/// ### New Output
/// {output}
/// ```
pub fn format_active_terminals(terminals: &[TerminalInfo]) -> String {
    if terminals.is_empty() {
        return String::new();
    }

    let mut details = String::from("\n\n# Actively Running Terminals");

    for terminal in terminals {
        details.push_str(&format!(
            "\n## Terminal {} (Active)",
            terminal.id
        ));
        details.push_str(&format!(
            "\n### Working Directory: `{}`",
            terminal.working_directory
        ));
        details.push_str(&format!(
            "\n### Original command: `{}`",
            terminal.last_command
        ));

        if let Some(ref output) = terminal.new_output {
            if !output.is_empty() {
                // Stub: in the TS version this goes through
                // Terminal.compressTerminalOutput. We keep the output as-is
                // (compression belongs to the terminal crate).
                let compressed = compress_terminal_output(output);
                details.push_str(&format!("\n### New Output\n{}", compressed));
            }
        }
    }

    details
}

/// Format the "Inactive Terminals with Completed Process Output" section.
///
/// Produces output like:
/// ```text
///
/// # Inactive Terminals with Completed Process Output
/// ## Terminal {id} (Inactive)
/// ### Working Directory: `{cwd}`
/// ### New Output
/// Command: `{command}`
/// {output}
/// ```
pub fn format_inactive_terminals(terminals: &[InactiveTerminalInfo]) -> String {
    // Only include terminals that actually have completed processes with output.
    let terminals_with_output: Vec<&InactiveTerminalInfo> = terminals
        .iter()
        .filter(|t| {
            t.completed_processes
                .iter()
                .any(|p| !p.output.is_empty())
        })
        .collect();

    if terminals_with_output.is_empty() {
        return String::new();
    }

    let mut details = String::from("\n\n# Inactive Terminals with Completed Process Output");

    for terminal in &terminals_with_output {
        let mut terminal_outputs: Vec<String> = Vec::new();

        for process in &terminal.completed_processes {
            if !process.output.is_empty() {
                let compressed = compress_terminal_output(&process.output);
                terminal_outputs.push(format!(
                    "Command: `{}`\n{}",
                    process.command, compressed
                ));
            }
        }

        if !terminal_outputs.is_empty() {
            details.push_str(&format!(
                "\n## Terminal {} (Inactive)",
                terminal.id
            ));
            details.push_str(&format!(
                "\n### Working Directory: `{}`",
                terminal.working_directory
            ));

            for output in &terminal_outputs {
                details.push_str(&format!("\n### New Output\n{}", output));
            }
        }
    }

    details
}

/// Stub for terminal output compression.
///
/// In the TypeScript codebase this calls `Terminal.compressTerminalOutput`.
/// That logic lives in the `roo-terminal` crate; here we simply return the
/// input unchanged.
fn compress_terminal_output(output: &str) -> String {
    output.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompletedProcess;

    #[test]
    fn test_active_terminals_empty() {
        let result = format_active_terminals(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_active_terminal_with_output() {
        let terminals = vec![TerminalInfo {
            id: "terminal-1".to_string(),
            working_directory: "/test/path/src".to_string(),
            last_command: "npm test".to_string(),
            new_output: Some("Test output".to_string()),
        }];
        let result = format_active_terminals(&terminals);
        assert!(result.contains("# Actively Running Terminals"));
        assert!(result.contains("## Terminal terminal-1 (Active)"));
        assert!(result.contains("### Working Directory: `/test/path/src`"));
        assert!(result.contains("### Original command: `npm test`"));
        assert!(result.contains("### New Output\nTest output"));
    }

    #[test]
    fn test_active_terminal_without_output() {
        let terminals = vec![TerminalInfo {
            id: "t1".to_string(),
            working_directory: "/home".to_string(),
            last_command: "ls".to_string(),
            new_output: None,
        }];
        let result = format_active_terminals(&terminals);
        assert!(result.contains("## Terminal t1 (Active)"));
        assert!(!result.contains("### New Output"));
    }

    #[test]
    fn test_active_terminal_with_empty_output() {
        let terminals = vec![TerminalInfo {
            id: "t1".to_string(),
            working_directory: "/home".to_string(),
            last_command: "ls".to_string(),
            new_output: Some(String::new()),
        }];
        let result = format_active_terminals(&terminals);
        assert!(!result.contains("### New Output"));
    }

    #[test]
    fn test_multiple_active_terminals() {
        let terminals = vec![
            TerminalInfo {
                id: "t1".to_string(),
                working_directory: "/a".to_string(),
                last_command: "cmd1".to_string(),
                new_output: Some("out1".to_string()),
            },
            TerminalInfo {
                id: "t2".to_string(),
                working_directory: "/b".to_string(),
                last_command: "cmd2".to_string(),
                new_output: Some("out2".to_string()),
            },
        ];
        let result = format_active_terminals(&terminals);
        assert!(result.contains("## Terminal t1 (Active)"));
        assert!(result.contains("## Terminal t2 (Active)"));
    }

    #[test]
    fn test_inactive_terminals_empty() {
        let result = format_inactive_terminals(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_inactive_terminal_with_completed_process() {
        let terminals = vec![InactiveTerminalInfo {
            id: "terminal-2".to_string(),
            working_directory: "/test/path/build".to_string(),
            completed_processes: vec![CompletedProcess {
                command: "npm build".to_string(),
                output: "Build output".to_string(),
            }],
        }];
        let result = format_inactive_terminals(&terminals);
        assert!(result.contains("# Inactive Terminals with Completed Process Output"));
        assert!(result.contains("## Terminal terminal-2 (Inactive)"));
        assert!(result.contains("### Working Directory: `/test/path/build`"));
        assert!(result.contains("Command: `npm build`"));
        assert!(result.contains("Build output"));
    }

    #[test]
    fn test_inactive_terminal_no_completed_processes() {
        let terminals = vec![InactiveTerminalInfo {
            id: "t1".to_string(),
            working_directory: "/home".to_string(),
            completed_processes: vec![],
        }];
        let result = format_inactive_terminals(&terminals);
        assert!(result.is_empty());
    }

    #[test]
    fn test_inactive_terminal_empty_output() {
        let terminals = vec![InactiveTerminalInfo {
            id: "t1".to_string(),
            working_directory: "/home".to_string(),
            completed_processes: vec![CompletedProcess {
                command: "echo".to_string(),
                output: String::new(),
            }],
        }];
        let result = format_inactive_terminals(&terminals);
        assert!(result.is_empty());
    }

    #[test]
    fn test_inactive_terminal_multiple_processes() {
        let terminals = vec![InactiveTerminalInfo {
            id: "t1".to_string(),
            working_directory: "/project".to_string(),
            completed_processes: vec![
                CompletedProcess {
                    command: "npm build".to_string(),
                    output: "build ok".to_string(),
                },
                CompletedProcess {
                    command: "npm test".to_string(),
                    output: "tests passed".to_string(),
                },
            ],
        }];
        let result = format_inactive_terminals(&terminals);
        assert!(result.contains("Command: `npm build`\nbuild ok"));
        assert!(result.contains("Command: `npm test`\ntests passed"));
        // Two "### New Output" sections
        assert_eq!(result.matches("### New Output").count(), 2);
    }
}
