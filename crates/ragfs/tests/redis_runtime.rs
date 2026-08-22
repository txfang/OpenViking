use bytes::Bytes;
use ragfs::cache_runtime::{AsyncCacheRuntime, CacheRuntime, PutOptions, RedisProviderConfig};
use std::time::Duration;

fn config() -> Option<RedisProviderConfig> {
    let endpoint = std::env::var("REDIS_URL").ok()?;
    Some(RedisProviderConfig {
        endpoints: vec![endpoint],
        key_prefix: String::new(),
        connect_timeout_ms: 30_000,
        command_timeout_ms: 1_000,
        default_ttl_seconds: 60,
        ..RedisProviderConfig::default()
    })
}

fn key(test_name: &str, value: &str) -> String {
    format!(
        "ragfs-runtime-test:{}:{test_name}:{value}",
        std::process::id()
    )
}

#[tokio::test]
async fn redis_runtime_preserves_primitive_and_batch_semantics() {
    let Some(config) = config() else {
        return;
    };
    let runtime = CacheRuntime::redis(config).await.unwrap();
    let missing = key("contract", "missing");
    let one = key("contract", "one");
    let two = key("contract", "two");

    assert_eq!(runtime.get(&missing).await.unwrap(), None);
    runtime
        .batch_put(vec![
            (one.clone(), Bytes::from_static(b"1")),
            (two.clone(), Bytes::from_static(b"2")),
        ])
        .await
        .unwrap();
    assert_eq!(
        runtime
            .batch_get(&[two.clone(), missing, one.clone()])
            .await
            .unwrap(),
        vec![
            Some(Bytes::from_static(b"2")),
            None,
            Some(Bytes::from_static(b"1")),
        ]
    );
    runtime.batch_delete(&[one.clone(), two]).await.unwrap();
    assert!(!runtime.exists(&one).await.unwrap());
    runtime.close().await.unwrap();
}

#[tokio::test]
async fn redis_runtime_preserves_default_ttl_and_per_write_override() {
    let Some(mut config) = config() else {
        return;
    };
    config.default_ttl_seconds = 30;
    let runtime = CacheRuntime::redis(config).await.unwrap();
    let ttl_key = key("ttl", "ttl");

    runtime
        .put(
            &ttl_key,
            Bytes::from_static(b"short"),
            PutOptions {
                ttl: Some(Duration::from_millis(100)),
            },
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(runtime.get(&ttl_key).await.unwrap(), None);
    runtime.close().await.unwrap();
}
