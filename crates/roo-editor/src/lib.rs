//! # Roo Editor
//!
//! Editor integration for Roo Code Rust — diff view, file editing, and undo stack.
//!
//! This crate provides a cross-platform Rust analogue of the VS Code
//! `DiffViewProvider`, along with file system operations and an undo/redo
//! stack for tracking edit history.
//!
//! ## Modules
//!
//! - [`types`] — Core types: `EditOperation`, `FileChange`, `DiffViewOptions`, `EditResult`, `EditorState`
//! - [`diff_view`] — `DiffViewProvider` for managing editing sessions
//! - [`file_editor`] — `FileEditor` for async file CRUD and line-diff computation
//! - [`undo_stack`] — `UndoStack` for undo/redo of edit operations

pub mod types;
pub mod diff_view;
pub mod file_editor;
pub mod undo_stack;

// Re-export the primary public API at the crate root for convenience.
pub use types::{DiffViewOptions, EditOperation, EditResult, EditorState, FileChange};
pub use diff_view::{DiffViewError, DiffViewProvider};
pub use file_editor::{DiffTag, FileEditor, FileEditorError, LineDiff};
pub use undo_stack::UndoStack;
