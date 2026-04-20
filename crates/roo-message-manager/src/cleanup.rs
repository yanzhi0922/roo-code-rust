//! Post-truncation cleanup logic.
//!
//! After API history is truncated, orphaned condense / truncation tags may
//! remain. This module provides [`cleanup_after_truncation`] which:
//!
//! - Removes Summary messages whose corresponding `condense_context` event no
//!   longer exists in the cline message list.
//! - Removes truncation markers whose corresponding `sliding_window_truncation`
//!   event no longer exists.
//! - Clears orphaned `condense_parent` / `truncation_parent` labels on
//!   remaining messages whose target summary/marker was deleted.
//!
//! This matches the combined behavior of the TS `cleanupAfterTruncation` and
//! the orphan-removal steps in `truncateApiHistoryWithCleanup`.

use std::collections::HashSet;

use crate::types::{ApiMessage, ClineMessage};

/// Remove orphaned Summary messages, truncation markers, and parent tags from
/// `api_history`.
///
/// An entry is considered *orphaned* when its linked context-management event
/// (`condense_context` / `sliding_window_truncation`) is **not** present in
/// the (post-rewind) `cline_messages`.
///
/// Additionally, any `condense_parent` or `truncation_parent` references that
/// point to a summary/marker which was deleted are cleared so those messages
/// become active again.
///
/// Returns a cleaned-up `Vec<ApiMessage>`.
pub fn cleanup_after_truncation(
    api_history: &[ApiMessage],
    cline_messages: &[ClineMessage],
) -> Vec<ApiMessage> {
    // Collect all condense IDs that are still present in cline messages.
    let active_condense_ids: HashSet<String> = cline_messages
        .iter()
        .filter(|m| m.say == "condense_context")
        .filter_map(|m| m.context_condense.as_ref())
        .map(|c| c.condense_id.clone())
        .collect();

    // Collect all truncation IDs that are still present in cline messages.
    let active_truncation_ids: HashSet<String> = cline_messages
        .iter()
        .filter(|m| m.say == "sliding_window_truncation")
        .filter_map(|m| m.context_truncation.as_ref())
        .map(|t| t.truncation_id.clone())
        .collect();

    // Also collect IDs of summaries and markers that exist within the API
    // history itself (some summaries/markers may not have a corresponding
    // cline message but should still be considered "existing").
    let api_summary_ids: HashSet<String> = api_history
        .iter()
        .filter(|m| m.is_summary)
        .filter_map(|m| m.condense_id.clone())
        .collect();

    let api_truncation_ids: HashSet<String> = api_history
        .iter()
        .filter(|m| m.is_truncation_marker)
        .filter_map(|m| m.truncation_id.clone())
        .collect();

    // A summary/marker is valid if its ID exists in cline messages OR if it
    // exists as another entry in the API history that is NOT being removed.
    let valid_condense_ids: HashSet<String> = active_condense_ids
        .union(&api_summary_ids)
        .cloned()
        .collect();

    let valid_truncation_ids: HashSet<String> = active_truncation_ids
        .union(&api_truncation_ids)
        .cloned()
        .collect();

    let mut result: Vec<ApiMessage> = Vec::with_capacity(api_history.len());

    for msg in api_history {
        // Remove orphaned Summary messages
        if msg.is_summary {
            if let Some(ref condense_id) = msg.condense_id {
                if !active_condense_ids.contains(condense_id) {
                    // Orphaned — skip it
                    continue;
                }
            }
        }

        // Remove orphaned truncation markers
        if msg.is_truncation_marker {
            if let Some(ref truncation_id) = msg.truncation_id {
                if !active_truncation_ids.contains(truncation_id) {
                    // Orphaned — skip it
                    continue;
                }
            }
        }

        // Clear orphaned parent references
        let mut cleaned = msg.clone();

        if let Some(ref parent) = msg.condense_parent {
            if !valid_condense_ids.contains(parent) {
                cleaned.condense_parent = None;
            }
        }

        if let Some(ref parent) = msg.truncation_parent {
            if !valid_truncation_ids.contains(parent) {
                cleaned.truncation_parent = None;
            }
        }

        result.push(cleaned);
    }

    result
}

/// Detect orphaned condense IDs — Summary messages whose condense_context
/// event is absent from cline messages.
///
/// Returns the set of orphaned condense IDs.
pub fn find_orphaned_condense_ids(
    api_history: &[ApiMessage],
    cline_messages: &[ClineMessage],
) -> Vec<String> {
    let active_condense_ids: Vec<String> = cline_messages
        .iter()
        .filter(|m| m.say == "condense_context")
        .filter_map(|m| m.context_condense.as_ref())
        .map(|c| c.condense_id.clone())
        .collect();

    api_history
        .iter()
        .filter(|m| m.is_summary)
        .filter_map(|m| m.condense_id.clone())
        .filter(|id| !active_condense_ids.contains(id))
        .collect()
}

/// Detect orphaned truncation IDs — truncation markers whose
/// sliding_window_truncation event is absent from cline messages.
///
/// Returns the set of orphaned truncation IDs.
pub fn find_orphaned_truncation_ids(
    api_history: &[ApiMessage],
    cline_messages: &[ClineMessage],
) -> Vec<String> {
    let active_truncation_ids: Vec<String> = cline_messages
        .iter()
        .filter(|m| m.say == "sliding_window_truncation")
        .filter_map(|m| m.context_truncation.as_ref())
        .map(|t| t.truncation_id.clone())
        .collect();

    api_history
        .iter()
        .filter(|m| m.is_truncation_marker)
        .filter_map(|m| m.truncation_id.clone())
        .filter(|id| !active_truncation_ids.contains(id))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_orphaned_summary() {
        let api = vec![
            ApiMessage::user(100),
            ApiMessage::summary(Some(150), "condense-orphan"),
            ApiMessage::assistant(200),
        ];
        let cline = vec![ClineMessage::new(100, "user_feedback")];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].ts, Some(100));
        assert_eq!(result[1].ts, Some(200));
    }

    #[test]
    fn keeps_valid_summary() {
        let api = vec![
            ApiMessage::user(100),
            ApiMessage::summary(Some(150), "condense-valid"),
            ApiMessage::assistant(200),
        ];
        let cline = vec![
            ClineMessage::new(100, "user_feedback"),
            ClineMessage::condense(140, "condense-valid"),
        ];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn removes_orphaned_truncation_marker() {
        let api = vec![
            ApiMessage::user(100),
            ApiMessage::truncation_marker(Some(150), "trunc-orphan"),
            ApiMessage::assistant(200),
        ];
        let cline = vec![ClineMessage::new(100, "user_feedback")];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].ts, Some(100));
        assert_eq!(result[1].ts, Some(200));
    }

    #[test]
    fn keeps_valid_truncation_marker() {
        let api = vec![
            ApiMessage::user(100),
            ApiMessage::truncation_marker(Some(150), "trunc-valid"),
            ApiMessage::assistant(200),
        ];
        let cline = vec![
            ClineMessage::new(100, "user_feedback"),
            ClineMessage::truncation(140, "trunc-valid"),
        ];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn mixed_orphans_and_valid() {
        let api = vec![
            ApiMessage::user(100),
            ApiMessage::summary(Some(150), "condense-orphan"),
            ApiMessage::summary(Some(160), "condense-valid"),
            ApiMessage::truncation_marker(Some(170), "trunc-orphan"),
            ApiMessage::truncation_marker(Some(180), "trunc-valid"),
            ApiMessage::assistant(200),
        ];
        let cline = vec![
            ClineMessage::new(100, "user_feedback"),
            ClineMessage::condense(155, "condense-valid"),
            ClineMessage::truncation(175, "trunc-valid"),
        ];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 4); // user + valid summary + valid marker + assistant
        assert_eq!(result[0].ts, Some(100));
        assert!(result[1].is_summary);
        assert!(result[2].is_truncation_marker);
        assert_eq!(result[3].ts, Some(200));
    }

    #[test]
    fn empty_inputs() {
        let result = cleanup_after_truncation(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn find_orphaned_condense_ids_works() {
        let api = vec![
            ApiMessage::summary(Some(100), "c1"),
            ApiMessage::summary(Some(200), "c2"),
        ];
        let cline = vec![ClineMessage::condense(90, "c1")];

        let orphans = find_orphaned_condense_ids(&api, &cline);
        assert_eq!(orphans, vec!["c2"]);
    }

    #[test]
    fn find_orphaned_truncation_ids_works() {
        let api = vec![
            ApiMessage::truncation_marker(Some(100), "t1"),
            ApiMessage::truncation_marker(Some(200), "t2"),
        ];
        let cline = vec![ClineMessage::truncation(90, "t1")];

        let orphans = find_orphaned_truncation_ids(&api, &cline);
        assert_eq!(orphans, vec!["t2"]);
    }

    #[test]
    fn summary_without_condense_id_is_kept() {
        // A summary with no condense_id should be kept (not orphaned).
        let mut msg = ApiMessage::summary(Some(100), "some-id");
        msg.condense_id = None;

        let api = vec![msg];
        let cline: Vec<ClineMessage> = vec![];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn truncation_marker_without_truncation_id_is_kept() {
        let mut msg = ApiMessage::truncation_marker(Some(100), "some-id");
        msg.truncation_id = None;

        let api = vec![msg];
        let cline: Vec<ClineMessage> = vec![];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 1);
    }

    // -- orphaned parent tag cleanup tests ----------------------------------

    #[test]
    fn clears_orphaned_condense_parent() {
        // Message has condense_parent pointing to a deleted summary
        let mut msg = ApiMessage::user(100);
        msg.condense_parent = Some("deleted-condense".into());

        let api = vec![msg];
        // No condense_context in cline → "deleted-condense" is orphaned
        let cline: Vec<ClineMessage> = vec![];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 1);
        assert!(result[0].condense_parent.is_none());
    }

    #[test]
    fn keeps_valid_condense_parent() {
        let mut msg = ApiMessage::user(100);
        msg.condense_parent = Some("valid-condense".into());

        let api = vec![msg];
        let cline = vec![ClineMessage::condense(90, "valid-condense")];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].condense_parent.as_deref(), Some("valid-condense"));
    }

    #[test]
    fn clears_orphaned_truncation_parent() {
        let mut msg = ApiMessage::user(100);
        msg.truncation_parent = Some("deleted-trunc".into());

        let api = vec![msg];
        let cline: Vec<ClineMessage> = vec![];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 1);
        assert!(result[0].truncation_parent.is_none());
    }

    #[test]
    fn keeps_valid_truncation_parent() {
        let mut msg = ApiMessage::user(100);
        msg.truncation_parent = Some("valid-trunc".into());

        let api = vec![msg];
        let cline = vec![ClineMessage::truncation(90, "valid-trunc")];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].truncation_parent.as_deref(), Some("valid-trunc"));
    }

    #[test]
    fn mixed_parent_cleanup() {
        let mut msg1 = ApiMessage::user(100);
        msg1.condense_parent = Some("condense-deleted".into());
        msg1.truncation_parent = Some("trunc-deleted".into());

        let mut msg2 = ApiMessage::user(200);
        msg2.condense_parent = Some("condense-valid".into());
        msg2.truncation_parent = Some("trunc-valid".into());

        let api = vec![msg1, msg2];
        let cline = vec![
            ClineMessage::condense(90, "condense-valid"),
            ClineMessage::truncation(95, "trunc-valid"),
        ];

        let result = cleanup_after_truncation(&api, &cline);
        assert_eq!(result.len(), 2);

        // msg1: both parents cleared
        assert!(result[0].condense_parent.is_none());
        assert!(result[0].truncation_parent.is_none());

        // msg2: both parents kept
        assert_eq!(result[1].condense_parent.as_deref(), Some("condense-valid"));
        assert_eq!(result[1].truncation_parent.as_deref(), Some("trunc-valid"));
    }
}
