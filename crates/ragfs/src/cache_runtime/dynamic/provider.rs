use super::abi::*;
use super::DynamicProviderConfig;
use crate::cache_runtime::provider::CacheProvider;
use crate::cache_runtime::{CacheError, CacheResult, PutOptions, ScriptRequest, ScriptResult};
use async_trait::async_trait;
use bytes::Bytes;
use libloading::Library;
use std::ffi::{c_void, CStr};
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Semaphore;

const MAX_INFLIGHT: u32 = 1024;

pub(crate) struct DynamicProvider {
    inner: Arc<DynamicProviderInner>,
}

struct DynamicProviderInner {
    _library: Library,
    api: OvCacheProviderV1,
    handle: NonNull<c_void>,
    closed: AtomicBool,
    inflight: Arc<Semaphore>,
}

unsafe impl Send for DynamicProviderInner {}
unsafe impl Sync for DynamicProviderInner {}

impl DynamicProvider {
    pub(crate) async fn connect(config: DynamicProviderConfig) -> CacheResult<Self> {
        let inner = tokio::task::spawn_blocking(move || unsafe { load(config) })
            .await
            .map_err(|error| {
                CacheError::Internal(format!("dynamic provider init failed: {error}"))
            })??;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    async fn call<T, F>(&self, operation: &'static str, call: F) -> CacheResult<T>
    where
        T: Send + 'static,
        F: FnOnce(Arc<DynamicProviderInner>) -> CacheResult<T> + Send + 'static,
    {
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(CacheError::Closed);
        }
        let permit = Arc::clone(&self.inner.inflight)
            .acquire_owned()
            .await
            .map_err(|_| CacheError::Closed)?;
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(CacheError::Closed);
        }
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            call(inner)
        })
        .await
        .map_err(|error| CacheError::Internal(format!("dynamic {operation} failed: {error}")))?
    }
}

#[async_trait]
impl CacheProvider for DynamicProvider {
    async fn get(&self, key: &str) -> CacheResult<Option<Bytes>> {
        let key = key.as_bytes().to_vec();
        self.call("get", move |inner| unsafe {
            let mut buffer = OvBuffer::default();
            let status = required(inner.api.get, "get")?(
                inner.handle.as_ptr(),
                OvSlice::new(&key),
                &mut buffer,
            );
            if status == STATUS_NOT_FOUND {
                return Ok(None);
            }
            status_ok(&inner, "get", status)?;
            take_buffer(&inner, buffer).map(Some)
        })
        .await
    }

    async fn put(&self, key: &str, value: Bytes, options: PutOptions) -> CacheResult<()> {
        let key = key.as_bytes().to_vec();
        self.call("put", move |inner| unsafe {
            let ttl_ms = options
                .ttl
                .map(|ttl| {
                    u64::try_from(ttl.as_millis()).map_err(|_| {
                        CacheError::InvalidArgument("dynamic provider TTL is too large".into())
                    })
                })
                .transpose()?;
            let options = OvPutOptions {
                ttl_ms: ttl_ms.unwrap_or_default(),
                has_ttl: u8::from(ttl_ms.is_some()),
            };
            let status = required(inner.api.put, "put")?(
                inner.handle.as_ptr(),
                OvSlice::new(&key),
                OvSlice::new(&value),
                &options,
            );
            status_ok(&inner, "put", status)
        })
        .await
    }

    async fn delete(&self, key: &str) -> CacheResult<()> {
        let key = key.as_bytes().to_vec();
        self.call("delete", move |inner| unsafe {
            let status = required(inner.api.delete_key, "delete")?(
                inner.handle.as_ptr(),
                OvSlice::new(&key),
            );
            status_ok(&inner, "delete", status)
        })
        .await
    }

    async fn exists(&self, key: &str) -> CacheResult<bool> {
        let key = key.as_bytes().to_vec();
        self.call("exists", move |inner| unsafe {
            let mut exists = 0;
            let status = required(inner.api.exists, "exists")?(
                inner.handle.as_ptr(),
                OvSlice::new(&key),
                &mut exists,
            );
            status_ok(&inner, "exists", status)?;
            Ok(exists != 0)
        })
        .await
    }

    async fn batch_get(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>> {
        let keys = keys.to_vec();
        self.call("batch_get", move |inner| unsafe {
            let slices = keys
                .iter()
                .map(|key| OvSlice::new(key.as_bytes()))
                .collect::<Vec<_>>();
            let mut buffer = OvBuffer::default();
            let status = required(inner.api.batch_get, "batch_get")?(
                inner.handle.as_ptr(),
                slices.as_ptr(),
                slices.len(),
                &mut buffer,
            );
            status_ok(&inner, "batch_get", status)?;
            let payload = take_buffer(&inner, buffer)?;
            serde_json::from_slice::<Vec<Option<Vec<u8>>>>(&payload)
                .map(|values| {
                    values
                        .into_iter()
                        .map(|value| value.map(Bytes::from))
                        .collect()
                })
                .map_err(|error| CacheError::InvalidData(error.to_string()))
        })
        .await
    }

    async fn batch_put(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()> {
        self.call("batch_put", move |inner| unsafe {
            let ffi_entries = entries
                .iter()
                .map(|(key, value)| OvEntry {
                    key: OvSlice::new(key.as_bytes()),
                    value: OvSlice::new(value),
                    ttl_ms: 0,
                    has_ttl: 0,
                })
                .collect::<Vec<_>>();
            let status = required(inner.api.batch_put, "batch_put")?(
                inner.handle.as_ptr(),
                ffi_entries.as_ptr(),
                ffi_entries.len(),
            );
            status_ok(&inner, "batch_put", status)
        })
        .await
    }

    async fn batch_delete(&self, keys: &[String]) -> CacheResult<()> {
        let keys = keys.to_vec();
        self.call("batch_delete", move |inner| unsafe {
            let slices = keys
                .iter()
                .map(|key| OvSlice::new(key.as_bytes()))
                .collect::<Vec<_>>();
            let status = required(inner.api.batch_delete, "batch_delete")?(
                inner.handle.as_ptr(),
                slices.as_ptr(),
                slices.len(),
            );
            status_ok(&inner, "batch_delete", status)
        })
        .await
    }

    async fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult> {
        self.call("execute_script", move |inner| unsafe {
            let key_slices = request
                .keys
                .iter()
                .map(|key| OvSlice::new(key.as_bytes()))
                .collect::<Vec<_>>();
            let arg_slices = request
                .args
                .iter()
                .map(|arg| OvSlice::new(arg))
                .collect::<Vec<_>>();
            let ffi_request = OvScriptRequest {
                script_id: OvSlice::new(request.script_id.as_bytes()),
                keys: key_slices.as_ptr(),
                key_count: key_slices.len(),
                args: arg_slices.as_ptr(),
                arg_count: arg_slices.len(),
            };
            let mut buffer = OvBuffer::default();
            let status = required(inner.api.execute_script, "execute_script")?(
                inner.handle.as_ptr(),
                &ffi_request,
                &mut buffer,
            );
            status_ok(&inner, "execute_script", status)?;
            take_buffer(&inner, buffer).map(|payload| ScriptResult { payload })
        })
        .await
    }

    async fn close(&self) -> CacheResult<()> {
        if self.inner.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let permits = Arc::clone(&self.inner.inflight)
            .acquire_many_owned(MAX_INFLIGHT)
            .await
            .map_err(|_| CacheError::Closed)?;
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || unsafe {
            required(inner.api.close, "close").map(|close| close(inner.handle.as_ptr()))
        })
        .await
        .map_err(|error| CacheError::Internal(format!("dynamic close failed: {error}")))??;
        drop(permits);
        Ok(())
    }
}

impl Drop for DynamicProviderInner {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            if let Some(close) = self.api.close {
                unsafe { close(self.handle.as_ptr()) };
            }
        }
    }
}

unsafe fn load(config: DynamicProviderConfig) -> CacheResult<DynamicProviderInner> {
    let library = Library::new(&config.library_path).map_err(|error| {
        CacheError::AbiMismatch(format!(
            "failed to load {}: {error}",
            config.library_path.display()
        ))
    })?;
    let entry = library
        .get::<ProviderEntryV1>(b"openviking_cache_provider_v1\0")
        .map_err(|error| CacheError::AbiMismatch(format!("missing provider entry: {error}")))?;
    let api_ptr = entry();
    let api = api_ptr
        .as_ref()
        .copied()
        .ok_or_else(|| CacheError::AbiMismatch("provider entry returned null".into()))?;
    if api.abi_version != ABI_VERSION_V1 {
        return Err(CacheError::AbiMismatch(format!(
            "expected ABI {}, got {}",
            ABI_VERSION_V1, api.abi_version
        )));
    }
    if api.struct_size as usize != size_of::<OvCacheProviderV1>() {
        return Err(CacheError::AbiMismatch(format!(
            "expected struct size {}, got {}",
            size_of::<OvCacheProviderV1>(),
            api.struct_size
        )));
    }
    validate_api(&api)?;
    let mut handle = std::ptr::null_mut();
    let status =
        required(api.init, "init")?(OvSlice::new(config.params_json.as_bytes()), &mut handle);
    let handle = NonNull::new(handle)
        .ok_or_else(|| CacheError::Unavailable("dynamic provider init returned null".into()))?;
    let inner = DynamicProviderInner {
        _library: library,
        api,
        handle,
        closed: AtomicBool::new(false),
        inflight: Arc::new(Semaphore::new(MAX_INFLIGHT as usize)),
    };
    status_ok(&inner, "init", status)?;
    let health = required(inner.api.health, "health")?(inner.handle.as_ptr());
    status_ok(&inner, "health", health)?;
    Ok(inner)
}

fn validate_api(api: &OvCacheProviderV1) -> CacheResult<()> {
    required(api.init, "init")?;
    required(api.get, "get")?;
    required(api.put, "put")?;
    required(api.delete_key, "delete")?;
    required(api.exists, "exists")?;
    required(api.batch_get, "batch_get")?;
    required(api.batch_put, "batch_put")?;
    required(api.batch_delete, "batch_delete")?;
    required(api.execute_script, "execute_script")?;
    required(api.health, "health")?;
    required(api.free_buffer, "free_buffer")?;
    required(api.close, "close")?;
    required(api.last_error, "last_error")?;
    Ok(())
}

fn required<T: Copy>(function: Option<T>, name: &str) -> CacheResult<T> {
    function.ok_or_else(|| CacheError::AbiMismatch(format!("missing function {name}")))
}

fn status_ok(inner: &DynamicProviderInner, operation: &str, status: i32) -> CacheResult<()> {
    if status == STATUS_OK {
        return Ok(());
    }
    let last_error = required(inner.api.last_error, "last_error")?;
    let error_ptr = unsafe { last_error(inner.handle.as_ptr()) };
    let message = if error_ptr.is_null() {
        format!("status {status}")
    } else {
        unsafe { CStr::from_ptr(error_ptr) }
            .to_string_lossy()
            .into_owned()
    };
    Err(CacheError::Unavailable(format!(
        "dynamic provider {operation} failed: {message}"
    )))
}

unsafe fn take_buffer(inner: &DynamicProviderInner, mut buffer: OvBuffer) -> CacheResult<Bytes> {
    let result = if buffer.len == 0 {
        Ok(Bytes::new())
    } else if buffer.ptr.is_null() {
        Err(CacheError::InvalidData(
            "dynamic provider returned a null buffer".into(),
        ))
    } else {
        Ok(Bytes::copy_from_slice(std::slice::from_raw_parts(
            buffer.ptr, buffer.len,
        )))
    };
    required(inner.api.free_buffer, "free_buffer")?(&mut buffer);
    result
}
