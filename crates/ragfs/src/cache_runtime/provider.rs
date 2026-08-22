//! Internal provider seam used by CacheRuntime.

use super::{CacheError, CacheResult, PutOptions, ScriptRequest, ScriptResult};
use async_trait::async_trait;
use bytes::Bytes;

#[async_trait]
pub(crate) trait CacheProvider: Send + Sync {
    async fn get(&self, key: &str) -> CacheResult<Option<Bytes>>;
    async fn put(&self, key: &str, value: Bytes, options: PutOptions) -> CacheResult<()>;
    async fn delete(&self, key: &str) -> CacheResult<()>;

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        Ok(self.get(key).await?.is_some())
    }

    async fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>> {
        let mut values = Vec::with_capacity(keys.len());
        for key in keys {
            values.push(self.get(key).await?);
        }
        Ok(values)
    }

    async fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()> {
        for (key, value) in entries {
            self.put(&key, value, PutOptions::default()).await?;
        }
        Ok(())
    }

    async fn batch_delete(&self, keys: &[String]) -> CacheResult<()> {
        for key in keys {
            self.delete(key).await?;
        }
        Ok(())
    }

    async fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult> {
        Err(CacheError::UnsupportedScript(request.script_id))
    }

    async fn close(&self) -> CacheResult<()> {
        Ok(())
    }
}
