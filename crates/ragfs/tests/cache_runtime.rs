use bytes::Bytes;
use ragfs::cache_runtime::{
    AsyncCacheRuntime, CacheError, CacheRuntime, MemoryMockProvider, PutOptions, ScriptRequest,
    SyncCacheRuntime,
};
use std::sync::Arc;

#[tokio::test]
async fn async_runtime_supports_the_primitive_contract() {
    let runtime = CacheRuntime::memory();

    assert_eq!(runtime.get("missing").await.unwrap(), None);
    assert!(!runtime.exists("missing").await.unwrap());

    runtime
        .put("a", Bytes::from_static(b"one"), PutOptions::default())
        .await
        .unwrap();
    assert_eq!(
        runtime.get("a").await.unwrap(),
        Some(Bytes::from_static(b"one"))
    );
    assert!(runtime.exists("a").await.unwrap());

    runtime
        .batch_put(vec![
            ("b".to_string(), Bytes::from_static(b"two")),
            ("c".to_string(), Bytes::from_static(b"three")),
        ])
        .await
        .unwrap();
    assert_eq!(
        runtime
            .batch_get(&["c".to_string(), "missing".to_string(), "b".to_string()])
            .await
            .unwrap(),
        vec![
            Some(Bytes::from_static(b"three")),
            None,
            Some(Bytes::from_static(b"two")),
        ]
    );

    runtime
        .batch_delete(&["a".to_string(), "c".to_string()])
        .await
        .unwrap();
    runtime.delete("b").await.unwrap();
    assert_eq!(
        runtime
            .batch_get(&["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .unwrap(),
        vec![None, None, None]
    );
}

#[tokio::test]
async fn unknown_script_returns_an_explicit_error() {
    let runtime = CacheRuntime::memory();
    let error = runtime
        .execute_script(ScriptRequest {
            script_id: "queuefs.unknown.v1".to_string(),
            keys: Vec::new(),
            args: Vec::new(),
        })
        .await
        .unwrap_err();

    assert!(matches!(error, CacheError::UnsupportedScript(_)));
}

#[test]
fn sync_and_async_facades_share_one_provider_instance() {
    let runtime = CacheRuntime::memory();
    let sync = runtime.sync_facade();

    sync.put(
        "shared",
        Bytes::from_static(b"value"),
        PutOptions::default(),
    )
    .unwrap();

    let async_runtime = runtime.clone();
    let value = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move { async_runtime.get("shared").await.unwrap() })
    })
    .join()
    .unwrap();

    assert_eq!(value, Some(Bytes::from_static(b"value")));
}

#[tokio::test]
async fn close_rejects_new_operations() {
    let runtime = CacheRuntime::memory();
    runtime.close().await.unwrap();

    assert!(matches!(runtime.get("key").await, Err(CacheError::Closed)));
}

#[tokio::test]
async fn controlled_memory_provider_is_only_accessed_through_runtime() {
    let provider = Arc::new(MemoryMockProvider::new());
    let runtime = CacheRuntime::memory_with_provider(Arc::clone(&provider));

    runtime
        .put(
            "observed",
            Bytes::from_static(b"value"),
            PutOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(provider.keys().await, vec!["observed".to_string()]);

    provider.set_unavailable(true);
    assert!(matches!(
        runtime.get("observed").await,
        Err(CacheError::Unavailable(_))
    ));
}
