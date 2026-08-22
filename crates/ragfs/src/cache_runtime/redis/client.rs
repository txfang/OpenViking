use super::RedisProviderConfig;
use crate::cache_runtime::{CacheError, CacheResult, ScriptValue};
use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, RedisError};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};
use url::Url;

pub(super) struct RedisClient {
    connection: RwLock<Option<MultiplexedConnection>>,
    concurrency: Arc<Semaphore>,
    concurrency_limit: u32,
    command_timeout: Duration,
    closed: AtomicBool,
}

impl RedisClient {
    pub(super) async fn connect(config: &RedisProviderConfig) -> CacheResult<Self> {
        config.validate()?;
        let redis_client = redis::Client::open(endpoint_url(config)?)
            .map_err(|error| map_redis_error("open", error))?;
        let connect = async {
            redis_client
                .get_multiplexed_async_connection()
                .await
                .map_err(|error| map_redis_error("connect", error))
        };
        let connection =
            tokio::time::timeout(Duration::from_millis(config.connect_timeout_ms), connect)
                .await
                .map_err(|_| {
                    CacheError::Timeout(format!(
                        "Redis connect exceeded {} ms",
                        config.connect_timeout_ms
                    ))
                })??;
        let client = Self {
            connection: RwLock::new(Some(connection)),
            concurrency: Arc::new(Semaphore::new(config.pool_size)),
            concurrency_limit: config.pool_size as u32,
            command_timeout: Duration::from_millis(config.command_timeout_ms),
            closed: AtomicBool::new(false),
        };
        client.health_check().await?;
        Ok(client)
    }

    async fn execute<T, F, Fut>(&self, operation: &'static str, call: F) -> CacheResult<T>
    where
        T: Send,
        F: FnOnce(MultiplexedConnection) -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, RedisError>> + Send,
    {
        if self.closed.load(Ordering::Acquire) {
            return Err(CacheError::Unavailable("Redis provider is closed".into()));
        }
        let work = async {
            let _permit = Arc::clone(&self.concurrency)
                .acquire_owned()
                .await
                .map_err(|_| CacheError::Unavailable("Redis client is closing".into()))?;
            let connection = self.connection.read().await.clone().ok_or_else(|| {
                CacheError::Unavailable("Redis connection has been released".into())
            })?;
            call(connection)
                .await
                .map_err(|error| map_redis_error(operation, error))
        };
        tokio::time::timeout(self.command_timeout, work)
            .await
            .map_err(|_| {
                CacheError::Timeout(format!(
                    "Redis {operation} exceeded {} ms",
                    self.command_timeout.as_millis()
                ))
            })?
    }

    pub(super) async fn health_check(&self) -> CacheResult<()> {
        self.execute("PING", |mut connection| async move {
            redis::cmd("PING").query_async(&mut connection).await
        })
        .await
    }

    pub(super) async fn get(&self, key: String) -> CacheResult<Option<Vec<u8>>> {
        self.execute(
            "GET",
            |mut connection| async move { connection.get(key).await },
        )
        .await
    }

    pub(super) async fn set(
        &self,
        key: String,
        value: Vec<u8>,
        ttl_ms: Option<u64>,
    ) -> CacheResult<()> {
        self.execute("SET", |mut connection| async move {
            if let Some(ttl_ms) = ttl_ms {
                redis::cmd("SET")
                    .arg(key)
                    .arg(value)
                    .arg("PX")
                    .arg(ttl_ms)
                    .query_async(&mut connection)
                    .await
            } else {
                connection.set(key, value).await
            }
        })
        .await
    }

    pub(super) async fn delete(&self, key: String) -> CacheResult<()> {
        self.execute("DEL", |mut connection| async move {
            let _: u64 = connection.del(key).await?;
            Ok(())
        })
        .await
    }

    pub(super) async fn exists(&self, key: String) -> CacheResult<bool> {
        self.execute("EXISTS", |mut connection| async move {
            connection.exists(key).await
        })
        .await
    }

    pub(super) async fn batch_get(&self, keys: Vec<String>) -> CacheResult<Vec<Option<Vec<u8>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        self.execute("MGET", |mut connection| async move {
            redis::cmd("MGET")
                .arg(keys)
                .query_async(&mut connection)
                .await
        })
        .await
    }

    pub(super) async fn batch_set(
        &self,
        entries: Vec<(String, Vec<u8>)>,
        ttl_ms: Option<u64>,
    ) -> CacheResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.execute("pipeline SET", |mut connection| async move {
            let mut pipe = redis::pipe();
            for (key, value) in entries {
                if let Some(ttl_ms) = ttl_ms {
                    pipe.cmd("SET")
                        .arg(key)
                        .arg(value)
                        .arg("PX")
                        .arg(ttl_ms)
                        .ignore();
                } else {
                    pipe.cmd("SET").arg(key).arg(value).ignore();
                }
            }
            pipe.query_async(&mut connection).await
        })
        .await
    }

    pub(super) async fn batch_delete(&self, keys: Vec<String>) -> CacheResult<()> {
        if keys.is_empty() {
            return Ok(());
        }
        self.execute("DEL", |mut connection| async move {
            let _: u64 = connection.del(keys).await?;
            Ok(())
        })
        .await
    }

    pub(super) async fn execute_script(
        &self,
        lua: &'static str,
        keys: Vec<String>,
        args: Vec<Vec<u8>>,
    ) -> CacheResult<ScriptValue> {
        self.execute("EVALSHA", |mut connection| async move {
            let script = redis::Script::new(lua);
            let mut invocation = script.prepare_invoke();
            for key in keys {
                invocation.key(key);
            }
            for arg in args {
                invocation.arg(arg);
            }
            invocation
                .invoke_async::<redis::Value>(&mut connection)
                .await
        })
        .await
        .and_then(redis_value_to_script_value)
    }

    #[cfg(test)]
    pub(super) async fn script_flush(&self) -> CacheResult<()> {
        self.execute("SCRIPT FLUSH", |mut connection| async move {
            redis::cmd("SCRIPT")
                .arg("FLUSH")
                .query_async(&mut connection)
                .await
        })
        .await
    }

    pub(super) async fn close(&self) -> CacheResult<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let permits = Arc::clone(&self.concurrency)
            .acquire_many_owned(self.concurrency_limit)
            .await
            .map_err(|_| CacheError::Unavailable("Redis client is closing".into()))?;
        self.connection.write().await.take();
        drop(permits);
        Ok(())
    }
}

fn redis_value_to_script_value(value: redis::Value) -> CacheResult<ScriptValue> {
    match value {
        redis::Value::Nil => Ok(ScriptValue::Null),
        redis::Value::Int(value) => Ok(ScriptValue::Integer(value)),
        redis::Value::BulkString(value) => Ok(ScriptValue::Bytes(value)),
        redis::Value::Array(values) | redis::Value::Set(values) => values
            .into_iter()
            .map(redis_value_to_script_value)
            .collect::<CacheResult<Vec<_>>>()
            .map(ScriptValue::Array),
        redis::Value::SimpleString(value) => Ok(ScriptValue::Bytes(value.into_bytes())),
        redis::Value::Okay => Ok(ScriptValue::Bytes(b"OK".to_vec())),
        redis::Value::Boolean(value) => Ok(ScriptValue::Boolean(value)),
        other => Err(CacheError::InvalidData(format!(
            "unsupported Redis script result: {other:?}"
        ))),
    }
}

fn endpoint_url(config: &RedisProviderConfig) -> CacheResult<String> {
    let endpoint = config.endpoints[0].clone();
    if config.username.is_empty() && config.password_env.is_empty() {
        return Ok(endpoint);
    }
    let mut url = Url::parse(&endpoint).map_err(|error| {
        CacheError::InvalidArgument(format!("Redis endpoint URL is invalid: {error}"))
    })?;
    if !config.username.is_empty() {
        url.set_username(&config.username)
            .map_err(|_| CacheError::InvalidArgument("Redis username is invalid".into()))?;
    }
    if !config.password_env.is_empty() {
        let password = env::var(&config.password_env).map_err(|_| {
            CacheError::InvalidArgument(format!(
                "Redis password_env {} is not set",
                config.password_env
            ))
        })?;
        url.set_password(Some(&password))
            .map_err(|_| CacheError::InvalidArgument("Redis password is invalid".into()))?;
    }
    Ok(url.to_string())
}

fn map_redis_error(operation: &str, error: RedisError) -> CacheError {
    if error.is_timeout() {
        return CacheError::Timeout(format!("Redis {operation} timed out: {error}"));
    }
    if error.is_connection_refusal()
        || error.is_connection_dropped()
        || error.is_cluster_error()
        || error.is_io_error()
    {
        return CacheError::Unavailable(format!("Redis {operation} unavailable: {error}"));
    }
    CacheError::Internal(format!("Redis {operation} failed: {error}"))
}
