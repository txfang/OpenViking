//! Redis provider configuration.

use crate::cache_runtime::{CacheError, CacheResult};

/// Connection and execution settings for the built-in Redis provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedisProviderConfig {
    /// Redis deployment mode.
    pub mode: String,
    /// Redis endpoints.
    pub endpoints: Vec<String>,
    /// Optional ACL username.
    pub username: String,
    /// Environment variable containing the Redis password.
    pub password_env: String,
    /// Maximum concurrent commands.
    pub pool_size: usize,
    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,
    /// Command timeout in milliseconds.
    pub command_timeout_ms: u64,
    /// Reserved compatibility field; unified Runtime keys require this to be empty.
    pub key_prefix: String,
    /// Default TTL in seconds; zero disables expiration.
    pub default_ttl_seconds: u64,
    /// Whether reads may use replicas.
    pub read_from_replica: bool,
}

impl Default for RedisProviderConfig {
    fn default() -> Self {
        Self {
            mode: "standalone".into(),
            endpoints: vec!["redis://127.0.0.1:6379".into()],
            username: String::new(),
            password_env: String::new(),
            pool_size: 32,
            connect_timeout_ms: 1_000,
            command_timeout_ms: 20,
            key_prefix: String::new(),
            default_ttl_seconds: 3_600,
            read_from_replica: false,
        }
    }
}

impl RedisProviderConfig {
    pub(super) fn validate(&self) -> CacheResult<()> {
        if self.mode != "standalone" {
            return Err(CacheError::InvalidArgument(
                "Redis mode must be standalone in this adapter stage".into(),
            ));
        }
        if self.endpoints.is_empty()
            || self
                .endpoints
                .iter()
                .any(|endpoint| endpoint.trim().is_empty())
        {
            return Err(CacheError::InvalidArgument(
                "Redis endpoints must not be empty".into(),
            ));
        }
        if self.pool_size == 0 {
            return Err(CacheError::InvalidArgument(
                "Redis pool_size must be greater than zero".into(),
            ));
        }
        if self.connect_timeout_ms == 0 || self.command_timeout_ms == 0 {
            return Err(CacheError::InvalidArgument(
                "Redis timeouts must be greater than zero".into(),
            ));
        }
        if !self.key_prefix.is_empty() {
            return Err(CacheError::InvalidArgument(
                "Redis provider key_prefix must be empty because Runtime keys are fully qualified"
                    .into(),
            ));
        }
        if self.default_ttl_seconds.checked_mul(1_000).is_none() {
            return Err(CacheError::InvalidArgument(
                "Redis default TTL is too large".into(),
            ));
        }
        if self.read_from_replica {
            return Err(CacheError::InvalidArgument(
                "Redis read_from_replica is not supported in standalone mode".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_default_ttl_that_cannot_be_sent_as_milliseconds() {
        let config = RedisProviderConfig {
            default_ttl_seconds: u64::MAX,
            ..RedisProviderConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(CacheError::InvalidArgument(message)) if message.contains("TTL")
        ));
    }

    #[test]
    fn rejects_provider_key_prefix_because_runtime_keys_are_fully_qualified() {
        let config = RedisProviderConfig {
            key_prefix: "provider-prefix".into(),
            ..RedisProviderConfig::default()
        };

        assert!(matches!(
            config.validate(),
            Err(CacheError::InvalidArgument(message)) if message.contains("key_prefix")
        ));
    }
}
