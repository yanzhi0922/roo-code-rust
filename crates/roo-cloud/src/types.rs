use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Authentication state for the cloud service.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthState {
    LoggedOut,
    AttemptingSession,
    InactiveSession,
    ActiveSession,
}

/// Information about the authenticated cloud user.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudUserInfo {
    pub id: String,
    pub email: String,
    pub name: String,
    pub avatar_url: Option<String>,
}

/// Membership of a user in an organization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrganizationMembership {
    pub org_id: String,
    pub org_name: String,
    pub role: String,
}

/// Settings for an organization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrganizationSettings {
    pub allow_list: Vec<String>,
    pub mcps: Vec<Value>,
}

/// User-specific cloud settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserSettings {
    pub task_sync_enabled: bool,
    pub telemetry_setting: String,
}

/// Errors that can occur during cloud operations.
#[derive(Clone, Debug, thiserror::Error)]
pub enum CloudError {
    #[error("not authenticated")]
    NotAuthenticated,

    #[error("authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("session expired")]
    SessionExpired,

    #[error("network error: {0}")]
    NetworkError(String),

    #[error("serialization error: {0}")]
    SerializationError(String),
}

impl From<serde_json::Error> for CloudError {
    fn from(err: serde_json::Error) -> Self {
        CloudError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_state_serde_roundtrip() {
        let states = vec![
            AuthState::LoggedOut,
            AuthState::AttemptingSession,
            AuthState::InactiveSession,
            AuthState::ActiveSession,
        ];
        for state in states {
            let json = serde_json::to_string(&state).unwrap();
            let deserialized: AuthState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, deserialized);
        }
    }

    #[test]
    fn test_cloud_user_info_serialization() {
        let info = CloudUserInfo {
            id: "user-1".to_string(),
            email: "test@example.com".to_string(),
            name: "Test User".to_string(),
            avatar_url: Some("https://example.com/avatar.png".to_string()),
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: CloudUserInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.id, deserialized.id);
        assert_eq!(info.email, deserialized.email);
        assert_eq!(info.avatar_url, deserialized.avatar_url);
    }

    #[test]
    fn test_cloud_user_info_no_avatar() {
        let info = CloudUserInfo {
            id: "user-2".to_string(),
            email: "noavatar@example.com".to_string(),
            name: "No Avatar".to_string(),
            avatar_url: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: CloudUserInfo = serde_json::from_str(&json).unwrap();
        assert!(deserialized.avatar_url.is_none());
    }

    #[test]
    fn test_organization_membership_serialization() {
        let membership = OrganizationMembership {
            org_id: "org-1".to_string(),
            org_name: "Test Org".to_string(),
            role: "admin".to_string(),
        };
        let json = serde_json::to_string(&membership).unwrap();
        let deserialized: OrganizationMembership = serde_json::from_str(&json).unwrap();
        assert_eq!(membership.org_id, deserialized.org_id);
        assert_eq!(membership.role, deserialized.role);
    }

    #[test]
    fn test_organization_settings_serialization() {
        let settings = OrganizationSettings {
            allow_list: vec!["openai".to_string()],
            mcps: vec![Value::String("mcp-server".to_string())],
        };
        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: OrganizationSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(1, deserialized.allow_list.len());
        assert_eq!(1, deserialized.mcps.len());
    }

    #[test]
    fn test_user_settings_serialization() {
        let settings = UserSettings {
            task_sync_enabled: true,
            telemetry_setting: "enabled".to_string(),
        };
        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: UserSettings = serde_json::from_str(&json).unwrap();
        assert!(deserialized.task_sync_enabled);
        assert_eq!("enabled", deserialized.telemetry_setting);
    }

    #[test]
    fn test_cloud_error_display() {
        assert_eq!(
            "not authenticated",
            format!("{}", CloudError::NotAuthenticated)
        );
        assert_eq!(
            "authentication failed: bad token",
            format!("{}", CloudError::AuthenticationFailed("bad token".to_string()))
        );
        assert_eq!("session expired", format!("{}", CloudError::SessionExpired));
        assert_eq!(
            "network error: timeout",
            format!("{}", CloudError::NetworkError("timeout".to_string()))
        );
    }

    #[test]
    fn test_cloud_error_from_serde_json() {
        let result: Result<Value, serde_json::Error> = serde_json::from_str("bad");
        if let Err(json_err) = result {
            let cloud_err: CloudError = CloudError::from(json_err);
            match cloud_err {
                CloudError::SerializationError(_) => {}
                _ => panic!("expected SerializationError"),
            }
        }
    }
}
