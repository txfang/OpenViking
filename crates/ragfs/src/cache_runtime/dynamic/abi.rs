use std::ffi::{c_char, c_void};

pub(super) const ABI_VERSION_V1: u32 = 1;
pub(super) const STATUS_OK: i32 = 0;
pub(super) const STATUS_NOT_FOUND: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct OvSlice {
    pub(super) ptr: *const u8,
    pub(super) len: usize,
}

impl OvSlice {
    pub(super) fn new(value: &[u8]) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct OvBuffer {
    pub(super) ptr: *mut u8,
    pub(super) len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct OvPutOptions {
    pub(super) ttl_ms: u64,
    pub(super) has_ttl: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct OvEntry {
    pub(super) key: OvSlice,
    pub(super) value: OvSlice,
    pub(super) ttl_ms: u64,
    pub(super) has_ttl: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct OvScriptRequest {
    pub(super) script_id: OvSlice,
    pub(super) keys: *const OvSlice,
    pub(super) key_count: usize,
    pub(super) args: *const OvSlice,
    pub(super) arg_count: usize,
}

pub(super) type InitFn = unsafe extern "C" fn(OvSlice, *mut *mut c_void) -> i32;
pub(super) type GetFn = unsafe extern "C" fn(*mut c_void, OvSlice, *mut OvBuffer) -> i32;
pub(super) type PutFn =
    unsafe extern "C" fn(*mut c_void, OvSlice, OvSlice, *const OvPutOptions) -> i32;
pub(super) type DeleteFn = unsafe extern "C" fn(*mut c_void, OvSlice) -> i32;
pub(super) type ExistsFn = unsafe extern "C" fn(*mut c_void, OvSlice, *mut u8) -> i32;
pub(super) type BatchGetFn =
    unsafe extern "C" fn(*mut c_void, *const OvSlice, usize, *mut OvBuffer) -> i32;
pub(super) type BatchPutFn = unsafe extern "C" fn(*mut c_void, *const OvEntry, usize) -> i32;
pub(super) type BatchDeleteFn = unsafe extern "C" fn(*mut c_void, *const OvSlice, usize) -> i32;
pub(super) type ExecuteScriptFn =
    unsafe extern "C" fn(*mut c_void, *const OvScriptRequest, *mut OvBuffer) -> i32;
pub(super) type HealthFn = unsafe extern "C" fn(*mut c_void) -> i32;
pub(super) type FreeBufferFn = unsafe extern "C" fn(*mut OvBuffer);
pub(super) type CloseFn = unsafe extern "C" fn(*mut c_void);
pub(super) type LastErrorFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct OvCacheProviderV1 {
    pub(super) abi_version: u32,
    pub(super) struct_size: u32,
    pub(super) init: Option<InitFn>,
    pub(super) get: Option<GetFn>,
    pub(super) put: Option<PutFn>,
    pub(super) delete_key: Option<DeleteFn>,
    pub(super) exists: Option<ExistsFn>,
    pub(super) batch_get: Option<BatchGetFn>,
    pub(super) batch_put: Option<BatchPutFn>,
    pub(super) batch_delete: Option<BatchDeleteFn>,
    pub(super) execute_script: Option<ExecuteScriptFn>,
    pub(super) health: Option<HealthFn>,
    pub(super) free_buffer: Option<FreeBufferFn>,
    pub(super) close: Option<CloseFn>,
    pub(super) last_error: Option<LastErrorFn>,
}

pub(super) type ProviderEntryV1 = unsafe extern "C" fn() -> *const OvCacheProviderV1;
