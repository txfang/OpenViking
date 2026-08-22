//! Errors returned by the unified cache runtime.

/// Result type returned by CacheRuntime operations.
pub type CacheResult<T> = std::result::Result<T, CacheError>;

/// Provider-independent cache runtime error.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// The configured provider cannot be reached.
    #[error("cache provider unavailable: {0}")]
    Unavailable(String),
    /// A provider operation exceeded its deadline.
    #[error("cache provider operation timed out: {0}")]
    Timeout(String),
    /// A provider returned malformed data.
    #[error("invalid cache data: {0}")]
    InvalidData(String),
    /// A caller supplied an invalid argument.
    #[error("invalid cache argument: {0}")]
    InvalidArgument(String),
    /// The selected provider does not implement a named script.
    #[error("unsupported cache script: {0}")]
    UnsupportedScript(String),
    /// A dynamic provider uses an incompatible ABI.
    #[error("cache provider ABI mismatch: {0}")]
    AbiMismatch(String),
    /// The runtime has already been closed.
    #[error("cache runtime is closed")]
    Closed,
    /// A synchronous facade was called from an asynchronous runtime thread.
    #[error("synchronous cache calls are not allowed inside a Tokio runtime")]
    InvalidExecutionContext,
    /// An internal provider or executor failure occurred.
    #[error("cache runtime internal error: {0}")]
    Internal(String),
}
