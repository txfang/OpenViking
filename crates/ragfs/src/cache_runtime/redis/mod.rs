//! Built-in Redis CacheRuntime provider.

mod client;
mod config;
mod provider;

use client::RedisClient;
pub use config::RedisProviderConfig;
pub(crate) use provider::RedisProvider;
