//! Unified provider-independent cache runtime.

mod api;
mod dynamic;
mod error;
mod executor;
mod memory;
mod provider;
mod redis;
mod script;

pub use api::{AsyncCacheRuntime, PutOptions, SyncCacheRuntime};
pub use dynamic::DynamicProviderConfig;
pub use error::{CacheError, CacheResult};
pub use memory::MemoryMockProvider;
pub use redis::RedisProviderConfig;

use async_trait::async_trait;
use bytes::Bytes;
use executor::RuntimeExecutor;
use provider::CacheProvider;
pub(crate) use script::{ScriptDefinition, ScriptRegistry, ScriptValue};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Request for one named provider-side atomic program.
#[derive(Debug, Clone)]
pub struct ScriptRequest {
    /// Stable program identifier.
    pub script_id: String,
    /// Fully-qualified keys used by the program.
    pub keys: Vec<String>,
    /// Opaque program arguments.
    pub args: Vec<Bytes>,
}

/// Opaque provider-side program result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptResult {
    /// Encoded result payload interpreted by the business module.
    pub payload: Bytes,
}

/// One cache runtime bound to one provider instance.
pub struct CacheRuntime {
    provider: Arc<dyn CacheProvider>,
    scripts: Arc<ScriptRegistry>,
    executor: Arc<RuntimeExecutor>,
    closed: AtomicBool,
}

impl CacheRuntime {
    pub(crate) fn from_provider(provider: Arc<dyn CacheProvider>) -> Arc<Self> {
        Self::from_provider_with_scripts(provider, Arc::new(ScriptRegistry::default()))
    }

    fn from_provider_with_scripts(
        provider: Arc<dyn CacheProvider>,
        scripts: Arc<ScriptRegistry>,
    ) -> Arc<Self> {
        Arc::new(Self {
            provider,
            scripts,
            executor: Arc::new(
                RuntimeExecutor::new().expect("CacheRuntime executor must initialize"),
            ),
            closed: AtomicBool::new(false),
        })
    }

    /// Build an in-process runtime for tests and smoke validation.
    pub fn memory() -> Arc<Self> {
        Self::memory_with_provider(Arc::new(MemoryMockProvider::new()))
    }

    /// Build a Runtime around one controllable in-memory provider.
    pub fn memory_with_provider(provider: Arc<MemoryMockProvider>) -> Arc<Self> {
        Self::from_provider(provider)
    }

    /// Connect the built-in Redis provider and create one Runtime.
    pub async fn redis(config: RedisProviderConfig) -> CacheResult<Arc<Self>> {
        let scripts = Arc::new(ScriptRegistry::default());
        let provider = redis::RedisProvider::connect(config, Arc::clone(&scripts)).await?;
        Ok(Self::from_provider_with_scripts(
            Arc::new(provider),
            scripts,
        ))
    }

    /// Load one provider through the versioned dynamic C ABI.
    pub async fn dynamic(config: DynamicProviderConfig) -> CacheResult<Arc<Self>> {
        let provider = dynamic::DynamicProvider::connect(config).await?;
        Ok(Self::from_provider(Arc::new(provider)))
    }

    pub(crate) fn register_script(&self, definition: ScriptDefinition) -> CacheResult<()> {
        self.scripts.register(definition)
    }

    /// Wrap the current Runtime with synchronous primitive operations.
    pub fn sync_facade(self: &Arc<Self>) -> SyncCacheRuntimeFacade {
        SyncCacheRuntimeFacade {
            runtime: Arc::clone(self),
            executor: Arc::clone(&self.executor),
        }
    }

    /// Close the provider and reject future operations.
    pub async fn close(&self) -> CacheResult<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.provider.close().await
    }

    fn ensure_open(&self) -> CacheResult<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(CacheError::Closed)
        } else {
            Ok(())
        }
    }
}

#[async_trait]
impl AsyncCacheRuntime for CacheRuntime {
    async fn get(&self, key: &str) -> CacheResult<Option<Bytes>> {
        self.ensure_open()?;
        self.provider.get(key).await
    }

    async fn put(&self, key: &str, value: Bytes, options: PutOptions) -> CacheResult<()> {
        self.ensure_open()?;
        self.provider.put(key, value, options).await
    }

    async fn delete(&self, key: &str) -> CacheResult<()> {
        self.ensure_open()?;
        self.provider.delete(key).await
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        self.ensure_open()?;
        self.provider.exists(key).await
    }

    async fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>> {
        self.ensure_open()?;
        self.provider.batch_get(keys).await
    }

    async fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()> {
        self.ensure_open()?;
        self.provider.batch_put(entries).await
    }

    async fn batch_delete(&self, keys: &[String]) -> CacheResult<()> {
        self.ensure_open()?;
        self.provider.batch_delete(keys).await
    }

    async fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult> {
        self.ensure_open()?;
        self.provider.execute_script(request).await
    }
}

/// Stateless synchronous facade over one CacheRuntime.
pub struct SyncCacheRuntimeFacade {
    runtime: Arc<CacheRuntime>,
    executor: Arc<RuntimeExecutor>,
}

impl SyncCacheRuntime for SyncCacheRuntimeFacade {
    fn get(&self, key: &str) -> CacheResult<Option<Bytes>> {
        let runtime = Arc::clone(&self.runtime);
        let key = key.to_string();
        self.executor
            .run(async move { AsyncCacheRuntime::get(runtime.as_ref(), &key).await })
    }

    fn put(&self, key: &str, value: Bytes, options: PutOptions) -> CacheResult<()> {
        let runtime = Arc::clone(&self.runtime);
        let key = key.to_string();
        self.executor.run(async move {
            AsyncCacheRuntime::put(runtime.as_ref(), &key, value, options).await
        })
    }

    fn delete(&self, key: &str) -> CacheResult<()> {
        let runtime = Arc::clone(&self.runtime);
        let key = key.to_string();
        self.executor
            .run(async move { AsyncCacheRuntime::delete(runtime.as_ref(), &key).await })
    }

    fn exists(&self, key: &str) -> CacheResult<bool> {
        let runtime = Arc::clone(&self.runtime);
        let key = key.to_string();
        self.executor
            .run(async move { AsyncCacheRuntime::exists(runtime.as_ref(), &key).await })
    }

    fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>> {
        let runtime = Arc::clone(&self.runtime);
        let keys = keys.to_vec();
        self.executor
            .run(async move { AsyncCacheRuntime::batch_get(runtime.as_ref(), &keys).await })
    }

    fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()> {
        let runtime = Arc::clone(&self.runtime);
        self.executor
            .run(async move { AsyncCacheRuntime::batch_put(runtime.as_ref(), entries).await })
    }

    fn batch_delete(&self, keys: &[String]) -> CacheResult<()> {
        let runtime = Arc::clone(&self.runtime);
        let keys = keys.to_vec();
        self.executor
            .run(async move { AsyncCacheRuntime::batch_delete(runtime.as_ref(), &keys).await })
    }

    fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult> {
        let runtime = Arc::clone(&self.runtime);
        self.executor
            .run(async move { AsyncCacheRuntime::execute_script(runtime.as_ref(), request).await })
    }
}
