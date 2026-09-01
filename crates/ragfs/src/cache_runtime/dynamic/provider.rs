use super::abi::*;
use super::config::DynamicProviderConfig;
use super::loader;
use crate::cache_runtime::provider::{CacheOperation, CacheProvider};
use crate::cache_runtime::{
    CacheError, CacheResult, Expiration, ListDirection, ListInsertPosition, ListInsertRequest,
    ListMoveRequest, ScriptRegistry, ScriptRequest, ScriptResult, ScriptValue, SetCondition,
    SetOptions, SetResult,
};
use async_trait::async_trait;
use bytes::Bytes;
use libloading::Library;
use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_void;
use std::mem::{align_of, size_of};
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

static HOST_API: HostApiV1 = HostApiV1 {
    abi_version: ABI_VERSION_V1,
    struct_size: size_of::<HostApiV1>(),
    alloc: host_alloc,
    dealloc: host_dealloc,
};

pub(crate) struct DynamicProvider {
    state: Arc<DynamicState>,
    scripts: Arc<ScriptRegistry>,
}

struct DynamicState {
    _library: Library,
    api: ProviderApiV1,
    handle: *mut c_void,
    calls: RwLock<()>,
    closed: AtomicBool,
}

// The ABI requires the provider handle to support concurrent calls. A provider
// backed by a non-thread-safe SDK must serialize or pool internally.
unsafe impl Send for DynamicState {}
unsafe impl Sync for DynamicState {}

impl DynamicProvider {
    pub(crate) async fn connect(
        config: DynamicProviderConfig,
        scripts: Arc<ScriptRegistry>,
    ) -> CacheResult<Self> {
        let state = tokio::task::spawn_blocking(move || connect_state(config))
            .await
            .map_err(|error| {
                CacheError::Internal(format!("dynamic provider create task failed: {error}"))
            })??;
        let provider = Self { state, scripts };
        if let Err(error) = provider.ping().await {
            let _ = provider.close().await;
            return Err(error);
        }
        Ok(provider)
    }

    async fn call<T, F>(&self, operation: &'static str, call: F) -> CacheResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&DynamicState) -> CacheResult<T> + Send + 'static,
    {
        let state = Arc::clone(&self.state);
        tokio::task::spawn_blocking(move || call(&state))
            .await
            .map_err(|error| {
                CacheError::Internal(format!("dynamic provider {operation} task failed: {error}"))
            })?
    }

    async fn list_push(
        &self,
        operation: CacheOperation,
        key: &str,
        values: Vec<Bytes>,
    ) -> CacheResult<u64> {
        if values.is_empty() {
            return Err(CacheError::InvalidArgument(format!(
                "{} requires at least one value",
                operation.name()
            )));
        }
        let key = key.to_string();
        self.call(operation.name(), move |state| {
            state.with_call(operation.name(), |api, handle| {
                let callback = match operation {
                    CacheOperation::Lpush => required(api.lpush, operation)?,
                    CacheOperation::Rpush => required(api.rpush, operation)?,
                    _ => unreachable!("list_push only accepts push operations"),
                };
                let raw_values = byte_slices(&values);
                let mut length = 0;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        raw_values.as_ptr(),
                        raw_values.len(),
                        &mut length,
                        &mut error,
                    )
                };
                status_result(status, error, operation.name())?;
                Ok(length)
            })
        })
        .await
    }

    async fn list_pop(
        &self,
        operation: CacheOperation,
        key: &str,
        count: Option<u64>,
    ) -> CacheResult<Vec<Bytes>> {
        let key = key.to_string();
        self.call(operation.name(), move |state| {
            state.with_call(operation.name(), |api, handle| {
                let callback = match operation {
                    CacheOperation::Lpop => required(api.lpop, operation)?,
                    CacheOperation::Rpop => required(api.rpop, operation)?,
                    _ => unreachable!("list_pop only accepts pop operations"),
                };
                let mut output = OwnedBufferArrayV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        u8::from(count.is_some()),
                        count.unwrap_or_default(),
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, operation.name())?;
                take_buffer_array(output)
            })
        })
        .await
    }
}

fn connect_state(config: DynamicProviderConfig) -> CacheResult<Arc<DynamicState>> {
    let loaded = loader::load(&config.library_path)?;
    let create = loaded.api.create.expect("validated create callback");
    let mut handle = ptr::null_mut();
    let mut error = OwnedBufferV1::default();
    let params = ByteSliceV1::from_slice(config.params_json.as_bytes());
    let status = unsafe { create(&HOST_API, params, &mut handle, &mut error) };
    if let Err(error) = status_result(status, error, "create") {
        if !handle.is_null() {
            close_handle(loaded.api, handle);
        }
        return Err(error);
    }
    if handle.is_null() {
        return Err(CacheError::InvalidData(
            "dynamic provider create returned a null handle".into(),
        ));
    }

    Ok(Arc::new(DynamicState {
        _library: loaded.library,
        api: loaded.api,
        handle,
        calls: RwLock::new(()),
        closed: AtomicBool::new(false),
    }))
}

impl DynamicState {
    fn with_call<T>(
        &self,
        operation: &str,
        call: impl FnOnce(&ProviderApiV1, *mut c_void) -> CacheResult<T>,
    ) -> CacheResult<T> {
        let _guard = self
            .calls
            .read()
            .map_err(|_| CacheError::Internal("dynamic provider call lock poisoned".into()))?;
        if self.closed.load(Ordering::Acquire) {
            return Err(CacheError::Closed);
        }
        call(&self.api, self.handle).map_err(|error| match error {
            CacheError::UnsupportedOperation(_) => {
                CacheError::UnsupportedOperation(operation.to_string())
            }
            other => other,
        })
    }

    fn supports(&self, operation: CacheOperation) -> bool {
        match operation {
            CacheOperation::Get => self.api.get.is_some(),
            CacheOperation::Set => self.api.set.is_some(),
            CacheOperation::Del => self.api.del.is_some(),
            CacheOperation::Mget => self.api.mget.is_some(),
            CacheOperation::Mset => self.api.mset.is_some(),
            CacheOperation::IncrBy => self.api.incrby.is_some(),
            CacheOperation::Sismember => self.api.sismember.is_some(),
            CacheOperation::Smembers => self.api.smembers.is_some(),
            CacheOperation::Scard => self.api.scard.is_some(),
            CacheOperation::Lpush => self.api.lpush.is_some(),
            CacheOperation::Rpush => self.api.rpush.is_some(),
            CacheOperation::Lpop => self.api.lpop.is_some(),
            CacheOperation::Rpop => self.api.rpop.is_some(),
            CacheOperation::Llen => self.api.llen.is_some(),
            CacheOperation::Lrange => self.api.lrange.is_some(),
            CacheOperation::Lindex => self.api.lindex.is_some(),
            CacheOperation::Lset => self.api.lset.is_some(),
            CacheOperation::Ltrim => self.api.ltrim.is_some(),
            CacheOperation::Lrem => self.api.lrem.is_some(),
            CacheOperation::Linsert => self.api.linsert.is_some(),
            CacheOperation::Lmove => self.api.lmove.is_some(),
            CacheOperation::ExecuteScript => self.api.execute_script.is_some(),
        }
    }

    fn ping(&self) -> CacheResult<()> {
        self.with_call("ping", |api, handle| {
            let callback = api.ping.expect("validated ping callback");
            let mut error = OwnedBufferV1::default();
            let status = unsafe { callback(handle, &mut error) };
            status_result(status, error, "ping")
        })
    }

    fn close(&self) -> CacheResult<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let _guard = self
            .calls
            .write()
            .map_err(|_| CacheError::Internal("dynamic provider call lock poisoned".into()))?;
        let callback = self.api.close.expect("validated close callback");
        let mut error = OwnedBufferV1::default();
        let status = unsafe { callback(self.handle, &mut error) };
        status_result(status, error, "close")
    }
}

impl Drop for DynamicState {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            close_handle(self.api, self.handle);
        }
    }
}

#[async_trait]
impl CacheProvider for DynamicProvider {
    fn validate_operations(&self, operations: &[CacheOperation]) -> CacheResult<()> {
        if let Some(operation) = operations
            .iter()
            .copied()
            .find(|operation| !self.state.supports(*operation))
        {
            return Err(CacheError::UnsupportedOperation(
                operation.name().to_string(),
            ));
        }
        Ok(())
    }

    async fn get(&self, key: &str) -> CacheResult<Option<Bytes>> {
        let key = key.to_string();
        self.call("get", move |state| {
            state.with_call("get", |api, handle| {
                let callback = required(api.get, CacheOperation::Get)?;
                let mut output = OptionalOwnedBufferV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, "get")?;
                take_optional_buffer(output)
            })
        })
        .await
    }

    async fn set(&self, key: &str, value: Bytes, options: SetOptions) -> CacheResult<SetResult> {
        if options.keep_ttl && options.expiration.is_some() {
            return Err(CacheError::InvalidArgument(
                "set cannot combine expiration with keep_ttl".into(),
            ));
        }
        let options = set_options(options)?;
        let key = key.to_string();
        self.call("set", move |state| {
            state.with_call("set", |api, handle| {
                let callback = required(api.set, CacheOperation::Set)?;
                let mut result = SET_APPLIED;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        ByteSliceV1::from_slice(&value),
                        options,
                        &mut result,
                        &mut error,
                    )
                };
                status_result(status, error, "set")?;
                match result {
                    SET_APPLIED => Ok(SetResult::Applied),
                    SET_CONDITION_NOT_MET => Ok(SetResult::ConditionNotMet),
                    other => Err(CacheError::InvalidData(format!(
                        "dynamic provider set returned invalid result {other}"
                    ))),
                }
            })
        })
        .await
    }

    async fn del(&self, keys: &[String]) -> CacheResult<u64> {
        if keys.is_empty() {
            return Ok(0);
        }
        let keys = keys.to_vec();
        self.call("del", move |state| {
            state.with_call("del", |api, handle| {
                let callback = required(api.del, CacheOperation::Del)?;
                let raw_keys = string_slices(&keys);
                let mut removed = 0;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        raw_keys.as_ptr(),
                        raw_keys.len(),
                        &mut removed,
                        &mut error,
                    )
                };
                status_result(status, error, "del")?;
                Ok(removed)
            })
        })
        .await
    }

    async fn mget(&self, keys: &[String]) -> CacheResult<Vec<Option<Bytes>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let keys = keys.to_vec();
        self.call("mget", move |state| {
            state.with_call("mget", |api, handle| {
                let callback = required(api.mget, CacheOperation::Mget)?;
                let raw_keys = string_slices(&keys);
                let mut output = OptionalOwnedBufferArrayV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        raw_keys.as_ptr(),
                        raw_keys.len(),
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, "mget")?;
                let values = take_optional_buffer_array(output)?;
                if values.len() != keys.len() {
                    return Err(CacheError::InvalidData(format!(
                        "dynamic provider mget returned {} values for {} keys",
                        values.len(),
                        keys.len()
                    )));
                }
                Ok(values)
            })
        })
        .await
    }

    async fn mset(&self, entries: Vec<(String, Bytes)>) -> CacheResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.call("mset", move |state| {
            state.with_call("mset", |api, handle| {
                let callback = required(api.mset, CacheOperation::Mset)?;
                let raw_entries = entries
                    .iter()
                    .map(|(key, value)| KeyValueV1 {
                        key: ByteSliceV1::from_slice(key.as_bytes()),
                        value: ByteSliceV1::from_slice(value),
                    })
                    .collect::<Vec<_>>();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(handle, raw_entries.as_ptr(), raw_entries.len(), &mut error)
                };
                status_result(status, error, "mset")
            })
        })
        .await
    }

    async fn incr_by(&self, key: &str, delta: i64) -> CacheResult<i64> {
        let key = key.to_string();
        self.call("incrby", move |state| {
            state.with_call("incrby", |api, handle| {
                let callback = required(api.incrby, CacheOperation::IncrBy)?;
                let mut value = 0;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        delta,
                        &mut value,
                        &mut error,
                    )
                };
                status_result(status, error, "incrby")?;
                Ok(value)
            })
        })
        .await
    }

    async fn sismember(&self, key: &str, member: &[u8]) -> CacheResult<bool> {
        let key = key.to_string();
        let member = member.to_vec();
        self.call("sismember", move |state| {
            state.with_call("sismember", |api, handle| {
                let callback = required(api.sismember, CacheOperation::Sismember)?;
                let mut present = 0;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        ByteSliceV1::from_slice(&member),
                        &mut present,
                        &mut error,
                    )
                };
                status_result(status, error, "sismember")?;
                take_bool(present, "sismember")
            })
        })
        .await
    }

    async fn smembers(&self, key: &str) -> CacheResult<Vec<Bytes>> {
        let key = key.to_string();
        self.call("smembers", move |state| {
            state.with_call("smembers", |api, handle| {
                let callback = required(api.smembers, CacheOperation::Smembers)?;
                let mut output = OwnedBufferArrayV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, "smembers")?;
                take_buffer_array(output)
            })
        })
        .await
    }

    async fn scard(&self, key: &str) -> CacheResult<u64> {
        scalar_u64(self, CacheOperation::Scard, key).await
    }

    async fn lpush(&self, key: &str, values: Vec<Bytes>) -> CacheResult<u64> {
        self.list_push(CacheOperation::Lpush, key, values).await
    }

    async fn rpush(&self, key: &str, values: Vec<Bytes>) -> CacheResult<u64> {
        self.list_push(CacheOperation::Rpush, key, values).await
    }

    async fn lpop(&self, key: &str, count: Option<u64>) -> CacheResult<Vec<Bytes>> {
        self.list_pop(CacheOperation::Lpop, key, count).await
    }

    async fn rpop(&self, key: &str, count: Option<u64>) -> CacheResult<Vec<Bytes>> {
        self.list_pop(CacheOperation::Rpop, key, count).await
    }

    async fn llen(&self, key: &str) -> CacheResult<u64> {
        scalar_u64(self, CacheOperation::Llen, key).await
    }

    async fn lrange(&self, key: &str, start: i64, stop: i64) -> CacheResult<Vec<Bytes>> {
        let key = key.to_string();
        self.call("lrange", move |state| {
            state.with_call("lrange", |api, handle| {
                let callback = required(api.lrange, CacheOperation::Lrange)?;
                let mut output = OwnedBufferArrayV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        start,
                        stop,
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, "lrange")?;
                take_buffer_array(output)
            })
        })
        .await
    }

    async fn lindex(&self, key: &str, index: i64) -> CacheResult<Option<Bytes>> {
        let key = key.to_string();
        self.call("lindex", move |state| {
            state.with_call("lindex", |api, handle| {
                let callback = required(api.lindex, CacheOperation::Lindex)?;
                let mut output = OptionalOwnedBufferV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        index,
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, "lindex")?;
                take_optional_buffer(output)
            })
        })
        .await
    }

    async fn lset(&self, key: &str, index: i64, value: Bytes) -> CacheResult<()> {
        let key = key.to_string();
        self.call("lset", move |state| {
            state.with_call("lset", |api, handle| {
                let callback = required(api.lset, CacheOperation::Lset)?;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        index,
                        ByteSliceV1::from_slice(&value),
                        &mut error,
                    )
                };
                status_result(status, error, "lset")
            })
        })
        .await
    }

    async fn ltrim(&self, key: &str, start: i64, stop: i64) -> CacheResult<()> {
        let key = key.to_string();
        self.call("ltrim", move |state| {
            state.with_call("ltrim", |api, handle| {
                let callback = required(api.ltrim, CacheOperation::Ltrim)?;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        start,
                        stop,
                        &mut error,
                    )
                };
                status_result(status, error, "ltrim")
            })
        })
        .await
    }

    async fn lrem(&self, key: &str, count: i64, value: Bytes) -> CacheResult<u64> {
        let key = key.to_string();
        self.call("lrem", move |state| {
            state.with_call("lrem", |api, handle| {
                let callback = required(api.lrem, CacheOperation::Lrem)?;
                let mut removed = 0;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        count,
                        ByteSliceV1::from_slice(&value),
                        &mut removed,
                        &mut error,
                    )
                };
                status_result(status, error, "lrem")?;
                Ok(removed)
            })
        })
        .await
    }

    async fn linsert(&self, request: ListInsertRequest) -> CacheResult<i64> {
        self.call("linsert", move |state| {
            state.with_call("linsert", |api, handle| {
                let callback = required(api.linsert, CacheOperation::Linsert)?;
                let position = match request.position {
                    ListInsertPosition::Before => LIST_BEFORE,
                    ListInsertPosition::After => LIST_AFTER,
                };
                let mut length = 0;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(request.key.as_bytes()),
                        position,
                        ByteSliceV1::from_slice(&request.pivot),
                        ByteSliceV1::from_slice(&request.value),
                        &mut length,
                        &mut error,
                    )
                };
                status_result(status, error, "linsert")?;
                Ok(length)
            })
        })
        .await
    }

    async fn lmove(&self, request: ListMoveRequest) -> CacheResult<Option<Bytes>> {
        self.call("lmove", move |state| {
            state.with_call("lmove", |api, handle| {
                let callback = required(api.lmove, CacheOperation::Lmove)?;
                let mut output = OptionalOwnedBufferV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(request.source.as_bytes()),
                        ByteSliceV1::from_slice(request.destination.as_bytes()),
                        direction(request.source_direction),
                        direction(request.destination_direction),
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, "lmove")?;
                take_optional_buffer(output)
            })
        })
        .await
    }

    async fn execute_script(&self, request: ScriptRequest) -> CacheResult<ScriptResult> {
        let script_source = self.scripts.resolve(&request.script_id)?.to_string();
        self.call("execute_script", move |state| {
            state.with_call("execute_script", |api, handle| {
                let callback = required(api.execute_script, CacheOperation::ExecuteScript)?;
                let raw_keys = string_slices(&request.keys);
                let raw_args = byte_slices(&request.args);
                let mut output = ScriptValueV1::default();
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(request.script_id.as_bytes()),
                        ByteSliceV1::from_slice(script_source.as_bytes()),
                        raw_keys.as_ptr(),
                        raw_keys.len(),
                        raw_args.as_ptr(),
                        raw_args.len(),
                        &mut output,
                        &mut error,
                    )
                };
                status_result(status, error, "execute_script")?;
                ScriptResult::encode(&take_script_value(output)?)
            })
        })
        .await
    }

    async fn ping(&self) -> CacheResult<()> {
        self.call("ping", DynamicState::ping).await
    }

    async fn close(&self) -> CacheResult<()> {
        self.call("close", DynamicState::close).await
    }
}

async fn scalar_u64(
    provider: &DynamicProvider,
    operation: CacheOperation,
    key: &str,
) -> CacheResult<u64> {
    let key = key.to_string();
    provider
        .call(operation.name(), move |state| {
            state.with_call(operation.name(), |api, handle| {
                let callback = match operation {
                    CacheOperation::Scard => required(api.scard, operation)?,
                    CacheOperation::Llen => required(api.llen, operation)?,
                    _ => unreachable!("scalar_u64 only accepts scalar operations"),
                };
                let mut value = 0;
                let mut error = OwnedBufferV1::default();
                let status = unsafe {
                    callback(
                        handle,
                        ByteSliceV1::from_slice(key.as_bytes()),
                        &mut value,
                        &mut error,
                    )
                };
                status_result(status, error, operation.name())?;
                Ok(value)
            })
        })
        .await
}

fn required<T: Copy>(callback: Option<T>, operation: CacheOperation) -> CacheResult<T> {
    callback.ok_or_else(|| CacheError::UnsupportedOperation(operation.name().to_string()))
}

fn set_options(options: SetOptions) -> CacheResult<SetOptionsV1> {
    let expiration_ms = match options.expiration {
        None => -1,
        Some(Expiration::After(duration)) => i64::try_from(duration.as_millis())
            .map_err(|_| CacheError::InvalidArgument("set expiration is too large".to_string()))?,
    };
    let condition = match options.condition {
        SetCondition::None => SET_CONDITION_NONE,
        SetCondition::Nx => SET_CONDITION_NX,
        SetCondition::Xx => SET_CONDITION_XX,
    };
    Ok(SetOptionsV1 {
        expiration_ms,
        condition,
        keep_ttl: u8::from(options.keep_ttl),
    })
}

fn direction(direction: ListDirection) -> u32 {
    match direction {
        ListDirection::Left => LIST_LEFT,
        ListDirection::Right => LIST_RIGHT,
    }
}

fn string_slices(values: &[String]) -> Vec<ByteSliceV1> {
    values
        .iter()
        .map(|value| ByteSliceV1::from_slice(value.as_bytes()))
        .collect()
}

fn byte_slices(values: &[Bytes]) -> Vec<ByteSliceV1> {
    values
        .iter()
        .map(|value| ByteSliceV1::from_slice(value))
        .collect()
}

fn status_result(status: i32, error: OwnedBufferV1, operation: &str) -> CacheResult<()> {
    let message = take_buffer(error)
        .map(|value| String::from_utf8_lossy(&value).into_owned())
        .unwrap_or_else(|buffer_error| buffer_error.to_string());
    if status == STATUS_OK {
        return Ok(());
    }
    let details = if message.is_empty() {
        format!("dynamic provider {operation} failed with status {status}")
    } else {
        format!("dynamic provider {operation}: {message}")
    };
    Err(match status {
        STATUS_TIMEOUT => CacheError::Timeout(details),
        STATUS_UNAVAILABLE => CacheError::Unavailable(details),
        STATUS_AUTHENTICATION => CacheError::Authentication(details),
        STATUS_PERMISSION_DENIED => CacheError::PermissionDenied(details),
        STATUS_INVALID_ARGUMENT => CacheError::InvalidArgument(details),
        STATUS_INVALID_DATA => CacheError::InvalidData(details),
        STATUS_CROSS_SLOT => CacheError::CrossSlot(details),
        STATUS_READ_ONLY => CacheError::ReadOnly(details),
        STATUS_UNSUPPORTED_OPERATION => CacheError::UnsupportedOperation(details),
        _ => CacheError::Internal(details),
    })
}

fn take_bool(value: u8, operation: &str) -> CacheResult<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(CacheError::InvalidData(format!(
            "dynamic provider {operation} returned invalid boolean {other}"
        ))),
    }
}

fn take_optional_buffer(value: OptionalOwnedBufferV1) -> CacheResult<Option<Bytes>> {
    match value.present {
        0 => Ok(None),
        1 => take_buffer(value.value).map(Some),
        other => Err(CacheError::InvalidData(format!(
            "dynamic provider returned invalid optional flag {other}"
        ))),
    }
}

fn take_buffer(buffer: OwnedBufferV1) -> CacheResult<Bytes> {
    if buffer.len == 0 {
        return Ok(Bytes::new());
    }
    if buffer.data.is_null() {
        return Err(CacheError::InvalidData(
            "dynamic provider returned a null buffer".into(),
        ));
    }
    let value = Bytes::copy_from_slice(unsafe { slice::from_raw_parts(buffer.data, buffer.len) });
    unsafe { host_dealloc(buffer.data, buffer.len, align_of::<u8>()) };
    Ok(value)
}

fn take_buffer_array(array: OwnedBufferArrayV1) -> CacheResult<Vec<Bytes>> {
    if array.len == 0 {
        return Ok(Vec::new());
    }
    if array.items.is_null() {
        return Err(CacheError::InvalidData(
            "dynamic provider returned a null buffer array".into(),
        ));
    }
    let items = unsafe { slice::from_raw_parts(array.items, array.len) }.to_vec();
    unsafe {
        host_dealloc(
            array.items.cast::<u8>(),
            array.len * size_of::<OwnedBufferV1>(),
            align_of::<OwnedBufferV1>(),
        )
    };
    items.into_iter().map(take_buffer).collect()
}

fn take_optional_buffer_array(
    array: OptionalOwnedBufferArrayV1,
) -> CacheResult<Vec<Option<Bytes>>> {
    if array.len == 0 {
        return Ok(Vec::new());
    }
    if array.items.is_null() {
        return Err(CacheError::InvalidData(
            "dynamic provider returned a null optional buffer array".into(),
        ));
    }
    let items = unsafe { slice::from_raw_parts(array.items, array.len) }.to_vec();
    unsafe {
        host_dealloc(
            array.items.cast::<u8>(),
            array.len * size_of::<OptionalOwnedBufferV1>(),
            align_of::<OptionalOwnedBufferV1>(),
        )
    };
    items.into_iter().map(take_optional_buffer).collect()
}

fn take_script_value(value: ScriptValueV1) -> CacheResult<ScriptValue> {
    match value.kind {
        SCRIPT_NULL => Ok(ScriptValue::Null),
        SCRIPT_INTEGER => Ok(ScriptValue::Integer(value.integer)),
        SCRIPT_BYTES => take_buffer(value.bytes).map(|value| ScriptValue::Bytes(value.to_vec())),
        SCRIPT_BOOLEAN => take_bool(value.boolean, "execute_script").map(ScriptValue::Boolean),
        SCRIPT_ARRAY => {
            if value.items_len == 0 {
                return Ok(ScriptValue::Array(Vec::new()));
            }
            if value.items.is_null() {
                return Err(CacheError::InvalidData(
                    "dynamic provider returned a null script array".into(),
                ));
            }
            let items = unsafe { slice::from_raw_parts(value.items, value.items_len) }.to_vec();
            unsafe {
                host_dealloc(
                    value.items.cast::<u8>(),
                    value.items_len * size_of::<ScriptValueV1>(),
                    align_of::<ScriptValueV1>(),
                )
            };
            items
                .into_iter()
                .map(take_script_value)
                .collect::<CacheResult<Vec<_>>>()
                .map(ScriptValue::Array)
        }
        other => Err(CacheError::InvalidData(format!(
            "dynamic provider returned invalid script value kind {other}"
        ))),
    }
}

fn close_handle(api: ProviderApiV1, handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let mut error = OwnedBufferV1::default();
    if let Some(close) = api.close {
        unsafe {
            close(handle, &mut error);
        }
        let _ = take_buffer(error);
    }
}

unsafe extern "C" fn host_alloc(size: usize, alignment: usize) -> *mut u8 {
    if size == 0 {
        return ptr::null_mut();
    }
    match Layout::from_size_align(size, alignment) {
        Ok(layout) => unsafe { alloc(layout) },
        Err(_) => ptr::null_mut(),
    }
}

unsafe extern "C" fn host_dealloc(data: *mut u8, size: usize, alignment: usize) {
    if data.is_null() || size == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, alignment) {
        unsafe { dealloc(data, layout) };
    }
}
