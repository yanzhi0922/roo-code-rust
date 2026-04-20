//! VS Code Extension Host message type definitions.
//!
//! Derived from `packages/types/src/vscode-extension-host.ts`.
//! These types define the communication protocol between the extension host
//! (backend) and the webview / CLI frontend.
//!
//! Note: This module defines core type stubs. Not every field from the TS
//! source is replicated — only the structural skeleton needed for
//! serialization and cross-crate type safety.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ExtensionMessage  (Extension → Webview | CLI)
// ---------------------------------------------------------------------------

/// Discriminator for [`ExtensionMessage`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ExtensionMessageType {
    Action,
    State,
    TaskHistoryUpdated,
    TaskHistoryItemUpdated,
    SelectedImages,
    Theme,
    WorkspaceUpdated,
    Invoke,
    MessageUpdated,
    McpServers,
    EnhancedPrompt,
    CommitSearchResults,
    ListApiConfig,
    RouterModels,
    OpenAiModels,
    OllamaModels,
    LmStudioModels,
    VsCodeLmModels,
    VsCodeLmApiAvailable,
    UpdatePrompt,
    SystemPrompt,
    AutoApprovalEnabled,
    UpdateCustomMode,
    DeleteCustomMode,
    ExportModeResult,
    ImportModeResult,
    CheckRulesDirectoryResult,
    DeleteCustomModeCheck,
    CurrentCheckpointUpdated,
    CheckpointInitWarning,
    TtsStart,
    TtsStop,
    FileSearchResults,
    ToggleApiConfigPin,
    AcceptInput,
    SetHistoryPreviewCollapsed,
    CommandExecutionStatus,
    McpExecutionStatus,
    VsCodeSetting,
    AuthenticatedUser,
    CondenseTaskContextStarted,
    CondenseTaskContextResponse,
    SingleRouterModelFetchResponse,
    RooCreditBalance,
    IndexingStatusUpdate,
    IndexCleared,
    CodebaseIndexConfig,
    MarketplaceInstallResult,
    MarketplaceRemoveResult,
    MarketplaceData,
    ShareTaskSuccess,
    CodeIndexSettingsSaved,
    CodeIndexSecretStatus,
    ShowDeleteMessageDialog,
    ShowEditMessageDialog,
    Commands,
    InsertTextIntoTextarea,
    DismissedUpsells,
    OrganizationSwitchResult,
    InteractionRequired,
    CustomToolsResult,
    Modes,
    TaskWithAggregatedCosts,
    OpenAiCodexRateLimits,
    WorktreeList,
    WorktreeResult,
    WorktreeCopyProgress,
    BranchList,
    WorktreeDefaults,
    WorktreeIncludeStatus,
    BranchWorktreeIncludeResult,
    FolderSelected,
    Skills,
    FileContent,
}

/// Checkpoint warning payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointWarning {
    #[serde(rename = "type")]
    pub warning_type: CheckpointWarningType,
    pub timeout: u64,
}

/// Type of checkpoint warning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CheckpointWarningType {
    WaitTimeout,
    InitTimeout,
}

/// Action types sent from extension to webview.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ExtensionAction {
    ChatButtonClicked,
    SettingsButtonClicked,
    HistoryButtonClicked,
    MarketplaceButtonClicked,
    CloudButtonClicked,
    DidBecomeVisible,
    FocusInput,
    SwitchTab,
    ToggleAutoApprove,
}

/// Invoke types.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum InvokeType {
    NewChat,
    SendMessage,
    PrimaryButtonClick,
    SecondaryButtonClick,
    SetChatBoxMessage,
}

/// Aggregated cost breakdown.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AggregatedCosts {
    pub total_cost: f64,
    pub own_cost: f64,
    pub children_cost: f64,
}

/// Opened tab info.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenedTab {
    pub label: String,
    pub is_active: bool,
    pub path: Option<String>,
}

/// File content response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileContentPayload {
    pub path: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A simplified representation of the `ExtensionMessage` interface.
///
/// The full TS type carries dozens of optional fields; this struct captures
/// the most commonly used ones. Fields that are not present are simply
/// `None`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExtensionMessage {
    #[serde(rename = "type")]
    pub msg_type: Option<ExtensionMessageType>,
    pub text: Option<String>,
    pub payload: Option<serde_json::Value>,
    pub checkpoint_warning: Option<CheckpointWarning>,
    pub action: Option<ExtensionAction>,
    pub invoke: Option<InvokeType>,
    pub state: Option<serde_json::Value>,
    pub images: Option<Vec<String>>,
    pub file_paths: Option<Vec<String>>,
    pub opened_tabs: Option<Vec<OpenedTab>>,
    pub error: Option<String>,
    pub success: Option<bool>,
    pub mode: Option<String>,
    pub slug: Option<String>,
    pub request_id: Option<String>,
    pub prompt_text: Option<String>,
    pub file_content: Option<FileContentPayload>,
    pub aggregated_costs: Option<AggregatedCosts>,
    pub list: Option<Vec<String>>,
    pub organization_id: Option<Option<String>>,
    pub commands: Option<Vec<Command>>,
}

// ---------------------------------------------------------------------------
// WebviewMessage  (Webview → Extension)
// ---------------------------------------------------------------------------

/// Discriminator for [`WebviewMessage`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WebviewMessageType {
    // Existing types (representative subset)
    ApiConfiguration,
    ApiKey,
    SetHistoryPreviewCollapsed,
    SetTaskHistoryCollapsed,
    Theme,
    OpenFile,
    OpenImage,
    DeleteImage,
    GetTaskHistory,
    GetTaskWithAggregatedCosts,
    ClearTaskHistory,
    DeleteTask,
    SaveTask,
    ExportTask,
    ResetState,
    RequestVsCodeLmModels,
    CancelTask,
    SendMessage,
    AskResponse,
    Question,
    ToggleAutoApproval,
    ToggleCheckpointsEnabled,
    SetCheckpointWarningTimeout,
    SetAlwaysAllowReadOnly,
    SetAlwaysAllowWrite,
    SetAlwaysAllowExecute,
    SetAlwaysAllowMcp,
    SetAlwaysAllowModeSwitch,
    SetAlwaysAllowSubtasks,
    SetAlwaysAllowFollowupQuestions,
    SetFollowupAutoApproveTimeoutMs,
    SetAllowedCommands,
    SetDeniedCommands,
    SetAllowedMaxRequests,
    SetAllowedMaxCost,
    SetTtsEnabled,
    SetTtsSpeed,
    SetSoundEnabled,
    SetSoundVolume,
    SetTerminalOutputPreviewSize,
    SetLanguage,
    SetCustomInstructions,
    SetPromptComponent,
    SetApiConfiguration,
    SetApiConfigPin,
    DeleteApiConfigPin,
    GetLatestState,
    GetOpenAiModels,
    GetOllamaModels,
    GetLmStudioModels,
    GetVsCodeLmModels,
    GetRouterModels,
    SearchCommits,
    SearchFiles,
    ListFiles,
    GetFileContent,
    EnhancePrompt,
    UpdateCustomMode,
    DeleteCustomMode,
    ExportMode,
    ImportMode,
    CheckRulesDirectory,
    InsertContentIntoFile,
    AcceptInput,
    SetMcpServers,
    EnableMcpServer,
    DisableMcpServer,
    RestartMcpServer,
    DeleteMcpServer,
    ToggleToolAutoApprove,
    ToggleToolAlwaysAllow,
    MarketplaceInstall,
    MarketplaceRemove,
    MarketplaceGetInstalled,
    MarketplaceSearch,
    CloudLogin,
    CloudLogout,
    CloudGetUserInfo,
    CloudGetOrganizationAllowList,
    CloudSetOrganization,
    ShareTask,
    SetCodebaseIndexConfig,
    GetCodebaseIndexConfig,
    GetCodebaseIndexSecretStatus,
    GetDismissedUpsells,
    DismissUpsell,
    GetCustomTools,
    CreateCustomTool,
    UpdateCustomTool,
    DeleteCustomTool,
    GetCommands,
    GetModes,
    GetSkills,
    GetSkillFileContent,
    OpenMcpSettings,
    NewTask,
    SwitchTab,
    SwitchMode,
    GetWorktreeList,
    CreateWorktree,
    DeleteWorktree,
    GetBranchList,
    GetWorktreeDefaults,
    GetWorktreeIncludeStatus,
    SetWorktreeIncludeContent,
    CheckBranchWorktreeInclude,
    CancelWorktreeCopy,
    SelectFolder,
    OpenExternal,
    SetInteractionRequired,
    CondenseTaskContext,
    SetDiagnosticEnabled,
    SetIncludeDiagnosticMessages,
    SetMaxDiagnosticMessages,
}

/// A simplified representation of the `WebviewMessage` interface.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WebviewMessage {
    #[serde(rename = "type")]
    pub msg_type: Option<WebviewMessageType>,
    pub text: Option<String>,
    pub payload: Option<serde_json::Value>,
    pub images: Option<Vec<String>>,
    pub mode: Option<String>,
    pub slug: Option<String>,
    pub request_id: Option<String>,
    pub bool_value: Option<bool>,
    pub number_value: Option<f64>,
    pub string_value: Option<String>,
    pub array_value: Option<Vec<serde_json::Value>>,
    pub object_value: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

/// A command exposed by the extension.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Command {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argument_hint: Option<String>,
    pub source: CommandSource,
}

/// Source of a command.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CommandSource {
    Global,
    Project,
    #[serde(rename = "built-in")]
    BuiltIn,
}
