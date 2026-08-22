use super::{RedisClient, RedisProviderConfig};
use crate::cache_runtime::provider::CacheProvider;
use crate::cache_runtime::{
    CacheError, CacheResult, PutOptions, ScriptRegistry, ScriptRequest, ScriptResult,
};
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Duration;

pub(crate) struct RedisProvider {
    client: Arc<RedisClient>,
    scripts: Arc<ScriptRegistry>,
    default_ttl: Option<Duration>,
}

impl RedisProvider {
    pub(crate) async fn connect(
        config: RedisProviderConfig,
        scripts: Arc<ScriptRegistry>,
    ) -> CacheResult<Self> {
        config.validate()?;
        let default_ttl = if config.default_ttl_seconds == 0 {
            None
        } else {
            Some(Duration::from_secs(config.default_ttl_seconds))
        };
        let client = Arc::new(RedisClient::connect(&config).await?);
        Ok(Self {
            client,
            scripts,
            default_ttl,
        })
    }

    fn ttl_ms(&self, options: PutOptions) -> CacheResult<Option<u64>> {
        options
            .ttl
            .or(self.default_ttl)
            .map(|ttl| {
                u64::try_from(ttl.as_millis())
                    .map_err(|_| CacheError::InvalidArgument("Redis TTL is too large".to_string()))
            })
            .transpose()
    }
}

#[async_trait]
impl CacheProvider for RedisProvider {
    async fn get(&self, key: &str) -> CacheResult<Option<Bytes>> {
        Ok(self.client.get(key.to_string()).await?.map(Bytes::from))
    }

    async fn put(&self, key: &str, value: Bytes, options: PutOptions) -> CacheResult<()> {
        self.client
            .set(key.to_string(), value.to_vec(), self.ttl_ms(options)?)
            .await
    }

    async fn delete(&self, key: &str) -> CacheResult<()> {
        self.client.delete(key.to_string()).await
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        self.client.exists(key.to_string()).await
    }

    async fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>> {
        Ok(self
            .client
            .batch_get(keys.to_vec())
            .await?
            .into_iter()
            .map(|value| value.map(Bytes::from))
            .collect())
    }

    async fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()> {
        let ttl_ms = self.ttl_ms(PutOptions::default())?;
        self.client
            .batch_set(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, value.to_vec()))
                    .collect(),
                ttl_ms,
            )
            .await
    }

    async fn batch_delete(&self, keys: &[String]) -> CacheResult<()> {
        self.client.batch_delete(keys.to_vec()).await
    }

    async fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult> {
        let lua = self.scripts.resolve(&request.script_id)?;
        let value = self
            .client
            .execute_script(
                lua,
                request.keys,
                request.args.into_iter().map(|arg| arg.to_vec()).collect(),
            )
            .await?;
        ScriptResult::encode(&value)
    }

    async fn close(&self) -> CacheResult<()> {
        self.client.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_runtime::{AsyncCacheRuntime, ScriptDefinition, ScriptRequest, ScriptValue};

    #[tokio::test]
    async fn executes_registered_script_and_recovers_after_script_flush() {
        let Ok(endpoint) = std::env::var("REDIS_URL") else {
            return;
        };
        let scripts = Arc::new(crate::cache_runtime::ScriptRegistry::default());
        scripts
            .register(ScriptDefinition {
                id: "runtime.test.echo.v1",
                redis_lua: "return {KEYS[1], ARGV[1]}",
            })
            .unwrap();
        let config = RedisProviderConfig {
            endpoints: vec![endpoint],
            key_prefix: String::new(),
            command_timeout_ms: 1_000,
            ..RedisProviderConfig::default()
        };
        let expected_key = format!("ragfs-script-test:{}:key", std::process::id());
        let provider = Arc::new(
            RedisProvider::connect(config, Arc::clone(&scripts))
                .await
                .unwrap(),
        );
        let runtime = crate::cache_runtime::CacheRuntime::from_provider(provider.clone());
        let request = ScriptRequest {
            script_id: "runtime.test.echo.v1".into(),
            keys: vec![expected_key.clone()],
            args: vec![Bytes::from_static(b"value")],
        };

        let first = runtime.execute_script(request.clone()).await.unwrap();
        assert_eq!(
            first.decode().unwrap(),
            ScriptValue::Array(vec![
                ScriptValue::Bytes(expected_key.into_bytes()),
                ScriptValue::Bytes(b"value".to_vec()),
            ])
        );

        provider.client.script_flush().await.unwrap();
        let second = runtime.execute_script(request).await.unwrap();
        assert_eq!(second.decode().unwrap(), first.decode().unwrap());
    }
}
