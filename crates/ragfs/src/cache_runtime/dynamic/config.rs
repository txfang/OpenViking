use std::path::PathBuf;

/// Configuration for one dynamically loaded cache provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicProviderConfig {
    /// Absolute path to the provider library.
    pub library_path: PathBuf,
    /// Provider-owned JSON configuration passed unchanged to `create`.
    pub params_json: String,
}
