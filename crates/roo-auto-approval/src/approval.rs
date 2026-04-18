//! Main auto-approval logic.
//!
//! Mirrors `checkAutoApproval` from `auto-approval/index.ts`.

use crate::commands::get_command_decision;
use crate::tools::{is_read_only_tool_action, is_write_tool_action};
use crate::types::{
    AskType, AutoApprovalState, CheckAutoApprovalResult, CommandDecision, McpServer,
    McpServerUse, ToolAction,
};

// ---------------------------------------------------------------------------
// MCP tool approval
// ---------------------------------------------------------------------------

/// Check if an MCP tool is always allowed based on the server configuration.
///
/// Mirrors `isMcpToolAlwaysAllowed` from `mcp.ts`.
pub fn is_mcp_tool_always_allowed(
    mcp_server_use: &McpServerUse,
    mcp_servers: &[McpServer],
) -> bool {
    match mcp_server_use {
        McpServerUse::UseMcpTool {
            server_name,
            tool_name,
        } => {
            let server = mcp_servers.iter().find(|s| s.name == *server_name);
            let Some(server) = server else {
                return false;
            };
            let tool = server.tools.iter().find(|t| t.name == *tool_name);
            tool.map(|t| t.always_allow).unwrap_or(false)
        }
        McpServerUse::AccessMcpResource { .. } => false,
    }
}

// ---------------------------------------------------------------------------
// Follow-up data parsing
// ---------------------------------------------------------------------------

/// Minimal representation of the follow-up data JSON.
#[derive(Debug, Clone, serde::Deserialize)]
struct FollowUpData {
    #[serde(default)]
    suggest: Vec<FollowUpSuggestion>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct FollowUpSuggestion {
    answer: String,
}

// ---------------------------------------------------------------------------
// Tool parsing helper
// ---------------------------------------------------------------------------

/// Attempt to parse the `text` field as a JSON [`ToolAction`].
fn parse_tool_action(text: Option<&str>) -> Option<ToolAction> {
    let text = text?;
    serde_json::from_str(text).ok()
}

// ---------------------------------------------------------------------------
// check_auto_approval
// ---------------------------------------------------------------------------

/// Parameters for [`check_auto_approval`].
pub struct CheckAutoApprovalParams<'a> {
    pub state: &'a AutoApprovalState,
    pub ask: &'a AskType,
    pub text: Option<&'a str>,
    pub is_protected: bool,
    /// MCP server configuration for tool allowlist checks.
    pub mcp_servers: &'a [McpServer],
}

/// Main auto-approval decision function.
///
/// Mirrors `checkAutoApproval` from `auto-approval/index.ts`.
///
/// **Logic branches:**
/// 1. Non-blocking ask → `Approve`
/// 2. Auto-approval disabled → `Ask`
/// 3. `followup` → check `always_allow_followup_questions` + timeout
/// 4. `use_mcp_server` → check `always_allow_mcp` + MCP tool whitelist
/// 5. `command` → check `always_allow_execute` + `get_command_decision`
/// 6. `tool` → check specific tool type
/// 7. Default → `Ask`
pub fn check_auto_approval(params: CheckAutoApprovalParams<'_>) -> CheckAutoApprovalResult {
    // 1. Non-blocking asks are always approved
    if params.ask.is_non_blocking() {
        return CheckAutoApprovalResult::Approve;
    }

    // 2. If auto-approval is not enabled, ask
    if !params.state.auto_approval_enabled {
        return CheckAutoApprovalResult::Ask;
    }

    // 3. Followup handling
    if *params.ask == AskType::Followup {
        if params.state.always_allow_followup_questions {
            if let Some(text) = params.text {
                if let Ok(data) = serde_json::from_str::<FollowUpData>(text) {
                    if let Some(suggestion) = data.suggest.first() {
                        if let Some(timeout) = params.state.followup_auto_approve_timeout_ms {
                            if timeout > 0 {
                                return CheckAutoApprovalResult::Timeout {
                                    timeout_ms: timeout,
                                    auto_response: suggestion.answer.clone(),
                                };
                            }
                        }
                    }
                }
            }
            // If we can't parse or no suggestion, ask
            return CheckAutoApprovalResult::Ask;
        } else {
            return CheckAutoApprovalResult::Ask;
        }
    }

    // 4. MCP server handling
    if *params.ask == AskType::UseMcpServer {
        let Some(text) = params.text else {
            return CheckAutoApprovalResult::Ask;
        };

        if let Ok(mcp_use) = serde_json::from_str::<McpServerUse>(text) {
            match &mcp_use {
                McpServerUse::UseMcpTool { .. } => {
                    if params.state.always_allow_mcp
                        && is_mcp_tool_always_allowed(&mcp_use, params.mcp_servers)
                    {
                        return CheckAutoApprovalResult::Approve;
                    }
                }
                McpServerUse::AccessMcpResource { .. } => {
                    if params.state.always_allow_mcp {
                        return CheckAutoApprovalResult::Approve;
                    }
                }
            }
        }

        return CheckAutoApprovalResult::Ask;
    }

    // 5. Command handling
    if *params.ask == AskType::Command {
        let Some(text) = params.text else {
            return CheckAutoApprovalResult::Ask;
        };

        if params.state.always_allow_execute {
            let decision = get_command_decision(
                text,
                &params.state.allowed_commands,
                &params.state.denied_commands,
            );

            return match decision {
                CommandDecision::AutoApprove => CheckAutoApprovalResult::Approve,
                CommandDecision::AutoDeny => CheckAutoApprovalResult::Deny,
                CommandDecision::AskUser => CheckAutoApprovalResult::Ask,
            };
        }
    }

    // 6. Tool handling
    if *params.ask == AskType::Tool {
        let Some(tool) = parse_tool_action(params.text) else {
            return CheckAutoApprovalResult::Ask;
        };

        // updateTodoList is always approved
        if matches!(tool, ToolAction::UpdateTodoList) {
            return CheckAutoApprovalResult::Approve;
        }

        // skill is always approved
        if matches!(tool, ToolAction::Skill) {
            return CheckAutoApprovalResult::Approve;
        }

        // switchMode requires always_allow_mode_switch
        if matches!(tool, ToolAction::SwitchMode) {
            return if params.state.always_allow_mode_switch {
                CheckAutoApprovalResult::Approve
            } else {
                CheckAutoApprovalResult::Ask
            };
        }

        // newTask / finishTask require always_allow_subtasks
        if matches!(tool, ToolAction::NewTask | ToolAction::FinishTask) {
            return if params.state.always_allow_subtasks {
                CheckAutoApprovalResult::Approve
            } else {
                CheckAutoApprovalResult::Ask
            };
        }

        let is_outside = tool.is_outside_workspace();

        // Read-only tools
        if is_read_only_tool_action(&tool) {
            return if params.state.always_allow_read_only
                && (!is_outside || params.state.always_allow_read_only_outside_workspace)
            {
                CheckAutoApprovalResult::Approve
            } else {
                CheckAutoApprovalResult::Ask
            };
        }

        // Write tools
        if is_write_tool_action(&tool) {
            return if params.state.always_allow_write
                && (!is_outside || params.state.always_allow_write_outside_workspace)
                && (!params.is_protected || params.state.always_allow_write_protected)
            {
                CheckAutoApprovalResult::Approve
            } else {
                CheckAutoApprovalResult::Ask
            };
        }
    }

    // 7. Default
    CheckAutoApprovalResult::Ask
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::McpTool;

    fn default_state() -> AutoApprovalState {
        AutoApprovalState {
            auto_approval_enabled: true,
            ..Default::default()
        }
    }

    fn full_access_state() -> AutoApprovalState {
        AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_read_only: true,
            always_allow_read_only_outside_workspace: true,
            always_allow_write: true,
            always_allow_write_outside_workspace: true,
            always_allow_write_protected: true,
            always_allow_mcp: true,
            always_allow_mode_switch: true,
            always_allow_subtasks: true,
            always_allow_execute: true,
            always_allow_followup_questions: true,
            followup_auto_approve_timeout_ms: Some(5000),
            allowed_commands: vec!["*".to_string()],
            denied_commands: vec![],
            ..Default::default()
        }
    }

    // ---- Basic ----

    #[test]
    fn test_auto_approval_disabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: false,
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: None,
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_non_blocking_ask_approved() {
        let state = default_state();
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::CommandOutput,
            text: None,
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    // ---- Followup ----

    #[test]
    fn test_followup_with_suggestion_timeout() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_followup_questions: true,
            followup_auto_approve_timeout_ms: Some(3000),
            ..Default::default()
        };
        let text = r#"{"suggest":[{"answer":"yes, do it"}]}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Followup,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(
            result,
            CheckAutoApprovalResult::Timeout {
                timeout_ms: 3000,
                auto_response: "yes, do it".to_string(),
            }
        );
    }

    #[test]
    fn test_followup_without_suggestion() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_followup_questions: true,
            followup_auto_approve_timeout_ms: Some(3000),
            ..Default::default()
        };
        let text = r#"{"suggest":[]}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Followup,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_followup_not_enabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_followup_questions: false,
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Followup,
            text: Some(r#"{"suggest":[{"answer":"yes"}]}"#),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_followup_invalid_json() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_followup_questions: true,
            followup_auto_approve_timeout_ms: Some(3000),
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Followup,
            text: Some("not json"),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_followup_no_timeout_configured() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_followup_questions: true,
            followup_auto_approve_timeout_ms: None,
            ..Default::default()
        };
        let text = r#"{"suggest":[{"answer":"yes"}]}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Followup,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    // ---- MCP ----

    #[test]
    fn test_mcp_tool_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_mcp: true,
            ..Default::default()
        };
        let mcp_servers = vec![McpServer {
            name: "test-server".to_string(),
            tools: vec![McpTool {
                name: "read_file".to_string(),
                always_allow: true,
            }],
        }];
        let text = r#"{"type":"use_mcp_tool","server_name":"test-server","tool_name":"read_file"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::UseMcpServer,
            text: Some(text),
            is_protected: false,
            mcp_servers: &mcp_servers,
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_mcp_tool_not_always_allowed() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_mcp: true,
            ..Default::default()
        };
        let mcp_servers = vec![McpServer {
            name: "test-server".to_string(),
            tools: vec![McpTool {
                name: "dangerous_tool".to_string(),
                always_allow: false,
            }],
        }];
        let text = r#"{"type":"use_mcp_tool","server_name":"test-server","tool_name":"dangerous_tool"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::UseMcpServer,
            text: Some(text),
            is_protected: false,
            mcp_servers: &mcp_servers,
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_mcp_resource_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_mcp: true,
            ..Default::default()
        };
        let text = r#"{"type":"access_mcp_resource","server_name":"test-server","uri":"test://resource"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::UseMcpServer,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_mcp_not_enabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_mcp: false,
            ..Default::default()
        };
        let text = r#"{"type":"access_mcp_resource","server_name":"test-server","uri":"test://resource"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::UseMcpServer,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_mcp_no_text() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_mcp: true,
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::UseMcpServer,
            text: None,
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    // ---- Command ----

    #[test]
    fn test_command_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_execute: true,
            allowed_commands: vec!["git".to_string()],
            denied_commands: vec![],
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Command,
            text: Some("git status"),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_command_denied() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_execute: true,
            allowed_commands: vec!["git".to_string()],
            denied_commands: vec!["rm".to_string()],
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Command,
            text: Some("rm -rf /"),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Deny);
    }

    #[test]
    fn test_command_not_enabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_execute: false,
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Command,
            text: Some("git status"),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_command_no_text() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_execute: true,
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Command,
            text: None,
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_command_wildcard_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_execute: true,
            allowed_commands: vec!["*".to_string()],
            denied_commands: vec![],
            ..Default::default()
        };
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Command,
            text: Some("anything here"),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    // ---- Tool: read-only ----

    #[test]
    fn test_read_only_tool_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_read_only: true,
            ..Default::default()
        };
        let text = r#"{"tool":"readFile","isOutsideWorkspace":false}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_read_only_outside_workspace_blocked() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_read_only: true,
            always_allow_read_only_outside_workspace: false,
            ..Default::default()
        };
        let text = r#"{"tool":"readFile","isOutsideWorkspace":true}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_read_only_outside_workspace_allowed() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_read_only: true,
            always_allow_read_only_outside_workspace: true,
            ..Default::default()
        };
        let text = r#"{"tool":"readFile","isOutsideWorkspace":true}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_read_only_not_enabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_read_only: false,
            ..Default::default()
        };
        let text = r#"{"tool":"readFile","isOutsideWorkspace":false}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    // ---- Tool: write ----

    #[test]
    fn test_write_tool_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_write: true,
            ..Default::default()
        };
        let text = r#"{"tool":"editedExistingFile","isOutsideWorkspace":false}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_write_outside_workspace_blocked() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_write: true,
            always_allow_write_outside_workspace: false,
            ..Default::default()
        };
        let text = r#"{"tool":"editedExistingFile","isOutsideWorkspace":true}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_write_outside_workspace_allowed() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_write: true,
            always_allow_write_outside_workspace: true,
            ..Default::default()
        };
        let text = r#"{"tool":"editedExistingFile","isOutsideWorkspace":true}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_write_protected_blocked() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_write: true,
            always_allow_write_protected: false,
            ..Default::default()
        };
        let text = r#"{"tool":"editedExistingFile","isOutsideWorkspace":false}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: true,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_write_protected_allowed() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_write: true,
            always_allow_write_protected: true,
            ..Default::default()
        };
        let text = r#"{"tool":"editedExistingFile","isOutsideWorkspace":false}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: true,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_write_not_enabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_write: false,
            ..Default::default()
        };
        let text = r#"{"tool":"editedExistingFile","isOutsideWorkspace":false}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    // ---- Tool: special tools ----

    #[test]
    fn test_todo_list_always_approved() {
        let state = default_state();
        let text = r#"{"tool":"updateTodoList"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_skill_always_approved() {
        let state = default_state();
        let text = r#"{"tool":"skill"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_switch_mode_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_mode_switch: true,
            ..Default::default()
        };
        let text = r#"{"tool":"switchMode"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_switch_mode_not_enabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_mode_switch: false,
            ..Default::default()
        };
        let text = r#"{"tool":"switchMode"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_subtasks_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_subtasks: true,
            ..Default::default()
        };
        let text = r#"{"tool":"newTask"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_finish_task_approved() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_subtasks: true,
            ..Default::default()
        };
        let text = r#"{"tool":"finishTask"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Approve);
    }

    #[test]
    fn test_subtasks_not_enabled() {
        let state = AutoApprovalState {
            auto_approval_enabled: true,
            always_allow_subtasks: false,
            ..Default::default()
        };
        let text = r#"{"tool":"newTask"}"#;
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some(text),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    // ---- Tool: invalid JSON ----

    #[test]
    fn test_tool_invalid_json() {
        let state = full_access_state();
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: Some("not json"),
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    #[test]
    fn test_tool_no_text() {
        let state = full_access_state();
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::Tool,
            text: None,
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    // ---- Default ----

    #[test]
    fn test_unknown_ask_returns_ask() {
        let state = full_access_state();
        let result = check_auto_approval(CheckAutoApprovalParams {
            state: &state,
            ask: &AskType::CompletionResult,
            text: None,
            is_protected: false,
            mcp_servers: &[],
        });
        assert_eq!(result, CheckAutoApprovalResult::Ask);
    }

    // ---- MCP tool always allowed helper ----

    #[test]
    fn test_mcp_tool_always_allowed_match() {
        let mcp_servers = vec![McpServer {
            name: "server1".to_string(),
            tools: vec![McpTool {
                name: "tool1".to_string(),
                always_allow: true,
            }],
        }];
        let mcp_use = McpServerUse::UseMcpTool {
            server_name: "server1".to_string(),
            tool_name: "tool1".to_string(),
        };
        assert!(is_mcp_tool_always_allowed(&mcp_use, &mcp_servers));
    }

    #[test]
    fn test_mcp_tool_always_allowed_no_match() {
        let mcp_servers = vec![McpServer {
            name: "server1".to_string(),
            tools: vec![McpTool {
                name: "tool1".to_string(),
                always_allow: false,
            }],
        }];
        let mcp_use = McpServerUse::UseMcpTool {
            server_name: "server1".to_string(),
            tool_name: "tool1".to_string(),
        };
        assert!(!is_mcp_tool_always_allowed(&mcp_use, &mcp_servers));
    }

    #[test]
    fn test_mcp_tool_always_allowed_resource() {
        let mcp_use = McpServerUse::AccessMcpResource {
            server_name: "server1".to_string(),
            uri: "test://resource".to_string(),
        };
        assert!(!is_mcp_tool_always_allowed(&mcp_use, &[]));
    }
}
