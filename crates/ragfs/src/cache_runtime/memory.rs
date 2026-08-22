//! In-process provider used by Runtime tests and smoke validation.

use super::provider::CacheProvider;
use super::{CacheError, CacheResult, PutOptions};
use async_trait::async_trait;
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tokio::sync::RwLock;

/// Controllable in-memory provider for tests and smoke validation.
pub struct MemoryMockProvider {
    values: RwLock<HashMap<String, Bytes>>,
    closed: AtomicBool,
    unavailable: AtomicBool,
    delete_failure: AtomicBool,
    gets: AtomicU64,
    batch_gets: AtomicU64,
    active_gets: AtomicU64,
    max_active_gets: AtomicU64,
    seen_get_keys: Mutex<Vec<String>>,
    seen_batch_get_keys: Mutex<Vec<Vec<String>>>,
    get_delay: Duration,
}

impl MemoryMockProvider {
    /// Create an empty provider.
    pub fn new() -> Self {
        Self {
            values: RwLock::new(HashMap::new()),
            closed: AtomicBool::new(false),
            unavailable: AtomicBool::new(false),
            delete_failure: AtomicBool::new(false),
            gets: AtomicU64::new(0),
            batch_gets: AtomicU64::new(0),
            active_gets: AtomicU64::new(0),
            max_active_gets: AtomicU64::new(0),
            seen_get_keys: Mutex::new(Vec::new()),
            seen_batch_get_keys: Mutex::new(Vec::new()),
            get_delay: Duration::ZERO,
        }
    }

    /// Delay individual get calls to exercise inflight and concurrency behavior.
    pub fn with_get_delay(mut self, delay: Duration) -> Self {
        self.get_delay = delay;
        self
    }

    /// Make all provider operations fail or recover them again.
    pub fn set_unavailable(&self, unavailable: bool) {
        self.unavailable.store(unavailable, Ordering::Release);
    }

    /// Make delete operations fail or recover them again.
    pub fn set_delete_failure(&self, fail: bool) {
        self.delete_failure.store(fail, Ordering::Release);
    }

    /// Return the current number of stored objects.
    pub async fn len(&self) -> usize {
        self.values.read().await.len()
    }

    /// Return whether the provider currently stores no objects.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Return a snapshot of stored keys.
    pub async fn keys(&self) -> Vec<String> {
        self.values.read().await.keys().cloned().collect()
    }

    /// Reset observed read calls and concurrency counters.
    pub fn reset_observed_reads(&self) {
        self.gets.store(0, Ordering::Relaxed);
        self.batch_gets.store(0, Ordering::Relaxed);
        self.active_gets.store(0, Ordering::Relaxed);
        self.max_active_gets.store(0, Ordering::Relaxed);
        self.seen_get_keys.lock().unwrap().clear();
        self.seen_batch_get_keys.lock().unwrap().clear();
    }

    /// Return the number of batch_get calls since the last reset.
    pub fn batch_get_count(&self) -> u64 {
        self.batch_gets.load(Ordering::Relaxed)
    }

    /// Return all keys observed by get and batch_get calls.
    pub fn observed_read_keys(&self) -> Vec<String> {
        let mut keys = self.seen_get_keys.lock().unwrap().clone();
        keys.extend(
            self.seen_batch_get_keys
                .lock()
                .unwrap()
                .iter()
                .flat_map(|batch| batch.iter().cloned()),
        );
        keys
    }

    /// Return the maximum number of concurrent get calls since the last reset.
    pub fn max_concurrent_gets(&self) -> u64 {
        self.max_active_gets.load(Ordering::Relaxed)
    }

    fn ensure_open(&self) -> CacheResult<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(CacheError::Unavailable(
                "memory provider is closed".to_string(),
            ))
        } else if self.unavailable.load(Ordering::Acquire) {
            Err(CacheError::Unavailable(
                "memory provider is unavailable".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn enter_get(&self) {
        let active = self.active_gets.fetch_add(1, Ordering::Relaxed) + 1;
        let mut current = self.max_active_gets.load(Ordering::Relaxed);
        while active > current {
            match self.max_active_gets.compare_exchange_weak(
                current,
                active,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
}

impl Default for MemoryMockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CacheProvider for MemoryMockProvider {
    async fn get(&self, key: &str) -> CacheResult<Option<Bytes>> {
        self.ensure_open()?;
        self.gets.fetch_add(1, Ordering::Relaxed);
        self.seen_get_keys.lock().unwrap().push(key.to_string());
        self.enter_get();
        if !self.get_delay.is_zero() {
            tokio::time::sleep(self.get_delay).await;
        }
        let value = self.values.read().await.get(key).cloned();
        self.active_gets.fetch_sub(1, Ordering::Relaxed);
        Ok(value)
    }

    async fn put(&self, key: &str, value: Bytes, _options: PutOptions) -> CacheResult<()> {
        self.ensure_open()?;
        self.values.write().await.insert(key.to_string(), value);
        Ok(())
    }

    async fn delete(&self, key: &str) -> CacheResult<()> {
        self.ensure_open()?;
        if self.delete_failure.load(Ordering::Acquire) {
            return Err(CacheError::Unavailable(
                "memory provider delete intentionally failed".to_string(),
            ));
        }
        self.values.write().await.remove(key);
        Ok(())
    }

    async fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>> {
        self.ensure_open()?;
        self.batch_gets.fetch_add(1, Ordering::Relaxed);
        self.seen_batch_get_keys.lock().unwrap().push(keys.to_vec());
        let values = self.values.read().await;
        Ok(keys.iter().map(|key| values.get(key).cloned()).collect())
    }

    async fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()> {
        self.ensure_open()?;
        self.values.write().await.extend(entries);
        Ok(())
    }

    async fn batch_delete(&self, keys: &[String]) -> CacheResult<()> {
        self.ensure_open()?;
        if self.delete_failure.load(Ordering::Acquire) {
            return Err(CacheError::Unavailable(
                "memory provider delete intentionally failed".to_string(),
            ));
        }
        let mut values = self.values.write().await;
        for key in keys {
            values.remove(key);
        }
        Ok(())
    }

    async fn close(&self) -> CacheResult<()> {
        self.closed.store(true, Ordering::Release);
        self.values.write().await.clear();
        Ok(())
    }
}
