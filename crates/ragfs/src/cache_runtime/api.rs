//! Public CacheRuntime operation types and interfaces.

use super::{CacheResult, ScriptRequest, ScriptResult};
use async_trait::async_trait;
use bytes::Bytes;
use std::time::Duration;

/// Options applied to one cache write.
#[derive(Debug, Clone, Copy, Default)]
pub struct PutOptions {
    /// Optional provider-side expiration.
    pub ttl: Option<Duration>,
}

/// Asynchronous primitive cache interface.
#[async_trait]
pub trait AsyncCacheRuntime: Send + Sync {
    /// Read one value.
    async fn get(&self, key: &str) -> CacheResult<Option<Bytes>>;
    /// Write one value.
    async fn put(&self, key: &str, value: Bytes, options: PutOptions) -> CacheResult<()>;
    /// Delete one value.
    async fn delete(&self, key: &str) -> CacheResult<()>;
    /// Check whether one value exists.
    async fn exists(&self, key: &str) -> CacheResult<bool>;
    /// Read multiple values while preserving input order.
    async fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>>;
    /// Write multiple values.
    async fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()>;
    /// Delete multiple values.
    async fn batch_delete(&self, keys: &[String]) -> CacheResult<()>;
    /// Execute one provider-specific named atomic program.
    async fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult>;
}

/// Synchronous facade over the same CacheRuntime instance.
pub trait SyncCacheRuntime: Send + Sync {
    /// Read one value.
    fn get(&self, key: &str) -> CacheResult<Option<Bytes>>;
    /// Write one value.
    fn put(&self, key: &str, value: Bytes, options: PutOptions) -> CacheResult<()>;
    /// Delete one value.
    fn delete(&self, key: &str) -> CacheResult<()>;
    /// Check whether one value exists.
    fn exists(&self, key: &str) -> CacheResult<bool>;
    /// Read multiple values while preserving input order.
    fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>>;
    /// Write multiple values.
    fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()>;
    /// Delete multiple values.
    fn batch_delete(&self, keys: &[String]) -> CacheResult<()>;
    /// Execute one provider-specific named atomic program.
    fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult>;
}
