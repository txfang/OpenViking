#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef ABI_VERSION
#define ABI_VERSION 1
#endif

typedef struct { const uint8_t *ptr; size_t len; } OvSlice;
typedef struct { uint8_t *ptr; size_t len; } OvBuffer;
typedef struct { OvSlice key; OvSlice value; uint64_t ttl_ms; uint8_t has_ttl; } OvEntry;
typedef struct { uint64_t ttl_ms; uint8_t has_ttl; } OvPutOptions;
typedef struct {
    OvSlice script_id;
    const OvSlice *keys;
    size_t key_count;
    const OvSlice *args;
    size_t arg_count;
} OvScriptRequest;

typedef struct OvCacheProviderV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t (*init)(OvSlice, void **);
    int32_t (*get)(void *, OvSlice, OvBuffer *);
    int32_t (*put)(void *, OvSlice, OvSlice, const OvPutOptions *);
    int32_t (*delete_key)(void *, OvSlice);
    int32_t (*exists)(void *, OvSlice, uint8_t *);
    int32_t (*batch_get)(void *, const OvSlice *, size_t, OvBuffer *);
    int32_t (*batch_put)(void *, const OvEntry *, size_t);
    int32_t (*batch_delete)(void *, const OvSlice *, size_t);
    int32_t (*execute_script)(void *, const OvScriptRequest *, OvBuffer *);
    int32_t (*health)(void *);
    void (*free_buffer)(OvBuffer *);
    void (*close)(void *);
    const char *(*last_error)(void *);
} OvCacheProviderV1;

typedef struct { char key[64]; uint8_t value[256]; size_t len; uint8_t used; } Item;
typedef struct { Item items[32]; char error[128]; } Provider;

static int equal(OvSlice value, const char *text) {
    size_t len = strlen(text);
    return value.len == len && memcmp(value.ptr, text, len) == 0;
}

static Item *find(Provider *provider, OvSlice key) {
    for (size_t i = 0; i < 32; i++) {
        if (provider->items[i].used && strlen(provider->items[i].key) == key.len &&
            memcmp(provider->items[i].key, key.ptr, key.len) == 0) return &provider->items[i];
    }
    return NULL;
}

static int32_t init_provider(OvSlice config, void **handle) {
    (void)config;
    *handle = calloc(1, sizeof(Provider));
    return *handle ? 0 : -1;
}

static int32_t get_value(void *handle, OvSlice key, OvBuffer *out) {
    Provider *provider = handle;
    if (equal(key, "slow")) {
        struct timespec delay = {0, 150000000};
        nanosleep(&delay, NULL);
        out->ptr = malloc(4);
        memcpy(out->ptr, "slow", 4);
        out->len = 4;
        return 0;
    }
    Item *item = find(provider, key);
    if (!item) return 1;
    out->ptr = malloc(item->len);
    memcpy(out->ptr, item->value, item->len);
    out->len = item->len;
    return 0;
}

static int32_t put_value(void *handle, OvSlice key, OvSlice value, const OvPutOptions *options) {
    (void)options;
    Provider *provider = handle;
    Item *item = find(provider, key);
    if (!item) {
        for (size_t i = 0; i < 32; i++) if (!provider->items[i].used) { item = &provider->items[i]; break; }
    }
    if (!item || key.len >= sizeof(item->key) || value.len > sizeof(item->value)) return -1;
    memset(item, 0, sizeof(*item));
    memcpy(item->key, key.ptr, key.len);
    memcpy(item->value, value.ptr, value.len);
    item->len = value.len;
    item->used = 1;
    return 0;
}

static int32_t delete_value(void *handle, OvSlice key) {
    Item *item = find(handle, key);
    if (item) item->used = 0;
    return 0;
}

static int32_t exists_value(void *handle, OvSlice key, uint8_t *out) {
    *out = find(handle, key) != NULL;
    return 0;
}

static void append(char **cursor, const char *text) {
    size_t len = strlen(text);
    memcpy(*cursor, text, len);
    *cursor += len;
}

static int32_t batch_get_value(void *handle, const OvSlice *keys, size_t count, OvBuffer *out) {
    char *json = malloc(8192), *cursor = json;
    append(&cursor, "[");
    for (size_t i = 0; i < count; i++) {
        if (i) append(&cursor, ",");
        Item *item = find(handle, keys[i]);
        if (!item) { append(&cursor, "null"); continue; }
        append(&cursor, "[");
        for (size_t j = 0; j < item->len; j++) {
            if (j) append(&cursor, ",");
            cursor += sprintf(cursor, "%u", item->value[j]);
        }
        append(&cursor, "]");
    }
    append(&cursor, "]");
    out->ptr = (uint8_t *)json;
    out->len = (size_t)(cursor - json);
    return 0;
}

static int32_t batch_put_value(void *handle, const OvEntry *entries, size_t count) {
    for (size_t i = 0; i < count; i++) {
        OvPutOptions options = { entries[i].ttl_ms, entries[i].has_ttl };
        if (put_value(handle, entries[i].key, entries[i].value, &options) != 0) return -1;
    }
    return 0;
}

static int32_t batch_delete_value(void *handle, const OvSlice *keys, size_t count) {
    for (size_t i = 0; i < count; i++) delete_value(handle, keys[i]);
    return 0;
}

static int32_t execute_script_value(void *handle, const OvScriptRequest *request, OvBuffer *out) {
    (void)handle;
    if (!equal(request->script_id, "runtime.test.echo.v1") || request->arg_count == 0) return -1;
    out->ptr = malloc(request->args[0].len);
    memcpy(out->ptr, request->args[0].ptr, request->args[0].len);
    out->len = request->args[0].len;
    return 0;
}

static int32_t health_provider(void *handle) { return handle ? 0 : -1; }
static void free_buffer_value(OvBuffer *buffer) { free(buffer->ptr); buffer->ptr = NULL; buffer->len = 0; }
static void close_provider(void *handle) { free(handle); }
static const char *last_error_provider(void *handle) { return handle ? ((Provider *)handle)->error : "provider init failed"; }

#ifndef OMIT_ENTRY
static const OvCacheProviderV1 API = {
    ABI_VERSION, sizeof(OvCacheProviderV1), init_provider, get_value, put_value,
    delete_value, exists_value, batch_get_value, batch_put_value, batch_delete_value,
    execute_script_value, health_provider, free_buffer_value, close_provider, last_error_provider
};

#ifdef _WIN32
__declspec(dllexport)
#endif
const OvCacheProviderV1 *openviking_cache_provider_v1(void) { return &API; }
#else
int fixture_without_provider_entry(void) { return 0; }
#endif
