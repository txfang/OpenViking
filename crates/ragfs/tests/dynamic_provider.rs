use bytes::Bytes;
use ragfs::cache_runtime::{
    AsyncCacheRuntime, CacheError, CacheRuntime, DynamicProviderConfig, PutOptions, ScriptRequest,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn compile_fixture(name: &str, definitions: &[&str]) -> PathBuf {
    let output_dir = tempfile::tempdir().unwrap().keep();
    let library = output_dir.join(format!(
        "{}{}{}",
        std::env::consts::DLL_PREFIX,
        name,
        std::env::consts::DLL_SUFFIX
    ));
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dynamic_provider/provider.c");
    let mut command = Command::new("cc");
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
    } else {
        command.args(["-shared", "-fPIC"]);
    }
    for definition in definitions {
        command.arg(format!("-D{definition}"));
    }
    let status = command
        .arg(source)
        .arg("-o")
        .arg(&library)
        .status()
        .unwrap();
    assert!(status.success());
    library
}

async fn runtime(library_path: PathBuf) -> std::sync::Arc<CacheRuntime> {
    CacheRuntime::dynamic(DynamicProviderConfig {
        library_path,
        params_json: "{}".into(),
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn dynamic_provider_runs_primitive_batch_and_script_contract() {
    let runtime = runtime(compile_fixture("provider", &[])).await;

    assert_eq!(runtime.get("missing").await.unwrap(), None);
    runtime
        .put("one", Bytes::from_static(b"1"), PutOptions::default())
        .await
        .unwrap();
    assert!(runtime.exists("one").await.unwrap());
    runtime
        .batch_put(vec![
            ("two".into(), Bytes::from_static(b"2")),
            ("three".into(), Bytes::from_static(b"3")),
        ])
        .await
        .unwrap();
    assert_eq!(
        runtime
            .batch_get(&["three".into(), "missing".into(), "one".into()])
            .await
            .unwrap(),
        vec![
            Some(Bytes::from_static(b"3")),
            None,
            Some(Bytes::from_static(b"1")),
        ]
    );
    let result = runtime
        .execute_script(ScriptRequest {
            script_id: "runtime.test.echo.v1".into(),
            keys: vec!["one".into()],
            args: vec![Bytes::from_static(b"argument")],
        })
        .await
        .unwrap();
    assert_eq!(result.payload, Bytes::from_static(b"argument"));
    runtime
        .batch_delete(&["one".into(), "two".into(), "three".into()])
        .await
        .unwrap();
    assert!(!runtime.exists("one").await.unwrap());
    runtime.close().await.unwrap();
    assert!(matches!(runtime.get("one").await, Err(CacheError::Closed)));
}

#[tokio::test]
async fn dynamic_provider_rejects_missing_symbol_and_abi_mismatch() {
    let missing = CacheRuntime::dynamic(DynamicProviderConfig {
        library_path: compile_fixture("missing", &["OMIT_ENTRY"]),
        params_json: "{}".into(),
    })
    .await;
    let missing = match missing {
        Ok(_) => panic!("missing entry unexpectedly loaded"),
        Err(error) => error,
    };
    assert!(matches!(missing, CacheError::AbiMismatch(_)));

    let mismatch = CacheRuntime::dynamic(DynamicProviderConfig {
        library_path: compile_fixture("mismatch", &["ABI_VERSION=99"]),
        params_json: "{}".into(),
    })
    .await;
    let mismatch = match mismatch {
        Ok(_) => panic!("ABI mismatch unexpectedly loaded"),
        Err(error) => error,
    };
    assert!(matches!(mismatch, CacheError::AbiMismatch(_)));
}

#[tokio::test]
async fn dynamic_provider_close_waits_for_inflight_blocking_call() {
    let runtime = runtime(compile_fixture("slow", &[])).await;
    let reader = {
        let runtime = runtime.clone();
        tokio::spawn(async move { runtime.get("slow").await.unwrap() })
    };
    tokio::time::sleep(Duration::from_millis(25)).await;
    let started = Instant::now();
    runtime.close().await.unwrap();

    assert_eq!(reader.await.unwrap(), Some(Bytes::from_static(b"slow")));
    assert!(started.elapsed() >= Duration::from_millis(100));
}
