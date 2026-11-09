# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source-aware memory removal and stale-extraction protection.

Call cleanup while holding the user evidence-index lease. The memory tree
lease excludes normal updater writes; updater source checks run under their
card leases, so an extraction started before a deletion cannot resurrect it.
"""
from __future__ import annotations

from copy import deepcopy

from openviking.session.memory.dataclass import MemoryFile
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils, bump_memory_version
from openviking_cli.exceptions import ConflictError


def prune_fields(fields, invalid):
    """Remove invalid supports, rebuilding prose only from surviving records."""
    result = deepcopy(fields)
    changed = False
    records = []
    for fact in fields.get("fact_records", []):
        ids = fact.get("source_turn_ids", [])
        kept = [sid for sid in ids if not invalid(sid, fact)]
        if kept != ids:
            changed = True
            if not kept:
                continue
            fact = dict(fact, source_turn_ids=kept,
                        source_quotes={sid: quote for sid, quote in fact.get("source_quotes", {}).items()
                                       if sid in kept})
        records.append(fact)
    ids = fields.get("source_turn_ids", [])
    kept_ids = [sid for sid in ids if not invalid(sid, None)]
    changed |= kept_ids != ids
    if not changed:
        return fields
    if not records and fields.get("fact_records") and fields.get("memory_type") != "entities":
        return None
    if not records and not kept_ids:
        return None
    result["fact_records"] = records
    if "source_turn_ids" in fields:
        result["source_turn_ids"] = kept_ids
    # Do not retain a merged summary that may still contain removed assertions.
    quotes = list(dict.fromkeys(quote for fact in records
                               for quote in fact.get("source_quotes", {}).values()))
    result["content"] = "\n".join(quotes)
    if not quotes:
        result["content"] = str(fields.get("name", ""))
    if "summary" in result:
        result["summary"] = result["content"]
    if "aliases" in result:
        result["aliases"] = [fields["name"]] if fields.get("name") else []
    return result


async def guard_source_update(fs, metadata, uri, ctx, *, index=None):
    """Validate revoked supports at the final write boundary, under card locks."""
    if not metadata.get("fact_records") and not metadata.get("source_turn_ids"):
        return metadata
    from openviking.retrieve.evidence_retriever import load_index
    from openviking.retrieve.memory_sources import verify_fact
    from openviking.session.memory.fact_records import alias_in_text

    root = uri.split("/memories/", 1)[0]
    if index is None:
        index = await load_index(fs, root + "/.index/evidence.json", ctx)
    if not index.source_invalidations:
        return metadata
    cache = {}

    def invalid(sid, fact):
        if sid not in index.source_invalidations:
            return False
        turns = index.resolve_sources([sid])
        if not turns:
            return True
        if fact is not None:
            return not verify_fact(fact, index, cache)
        aliases = metadata.get("aliases") or [metadata.get("name", "")]
        return metadata.get("memory_type") == "entities" and not any(
            alias_in_text(str(alias), turns[0]["text"], metadata.get("category"))
            for alias in aliases if alias
        )

    return prune_fields(metadata, invalid)


async def cleanup_source_memories(fs, index, index_uri, index_lease, source_ids, ctx,
                                  *, vikingdb=None):
    """Prune affected cards and refresh their indexes; failures remain retryable.

    The durable URI journal is saved before changing any card. A retry also
    refreshes already-written cards whose vector enqueue previously failed.
    """
    from openviking.session.memory.memory_type_registry import get_default_registry
    from openviking.session.memory.memory_updater import MemoryUpdater

    if not source_ids and not index.pending_memory_cleanup:
        return {"updated": 0, "deleted": 0}
    root = index_uri.rsplit("/.index/", 1)[0] + "/memories"
    if not await fs.exists(root, ctx=ctx) and not index.pending_memory_cleanup:
        return {"updated": 0, "deleted": 0}
    lease = await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path(root, ctx=ctx), timeout_secs=30)
    try:
        if not await fs.exists(root, ctx=ctx):
            pending = index.pending_memory_cleanup
            if any(not uri.startswith(root + "/") for uri in pending):
                raise ValueError("Cleanup journal contains an out-of-scope memory URI")
            await fs._delete_from_vector_store(
                sorted(set(pending) | {uri.rsplit("/", 1)[0] for uri in pending}), ctx=ctx)
            index.pending_memory_cleanup = []
            index._dirty = True
            await index.save(fs, index_uri, ctx, lease_ref=index_lease)
            return {"updated": 0, "deleted": 0}
        listing = await fs.glob("**/*.md", uri=root, ctx=ctx, force_fs=True)
        if listing.get("truncated"):
            raise ConflictError("Memory listing was truncated; cleanup was not applied")
        cards = {}
        for uri in listing.get("matches", []):
            if not uri.startswith(root + "/") or uri.rsplit("/", 1)[-1].startswith("."):
                continue
            cards[uri] = MemoryFileUtils.read(await fs.read_file(uri, ctx=ctx), uri=uri)
        removed = set(source_ids)
        plans = {}
        deleted = {uri for uri in index.pending_memory_cleanup if uri not in cards}
        for uri, card in cards.items():
            fields = card.to_metadata()
            pruned = prune_fields(fields, lambda sid, fact: sid in removed)
            if pruned is not fields:
                plans[uri] = pruned
                if pruned is None:
                    deleted.add(uri)
        # Relations to removed cards must disappear from surviving endpoints.
        for uri, card in cards.items():
            if uri in deleted:
                continue
            fields = plans.get(uri, card.to_metadata())
            for key in ("links", "backlinks"):
                links = fields.get(key, [])
                kept = [link for link in links
                        if link.get("from_uri") not in deleted and link.get("to_uri") not in deleted]
                if kept != links:
                    fields = dict(fields, **{key: kept})
                    plans[uri] = fields
        pending = set(index.pending_memory_cleanup) | set(plans)
        if not pending:
            return {"updated": 0, "deleted": 0}
        if any(not uri.startswith(root + "/") for uri in pending):
            raise ValueError("Cleanup journal contains an out-of-scope memory URI")
        index.pending_memory_cleanup = sorted(pending)
        index._dirty = True
        await index.save(fs, index_uri, ctx, lease_ref=index_lease)
        for uri, fields in plans.items():
            if fields is None:
                await fs.rm(uri, ctx=ctx, lease_ref=lease)
            else:
                card = MemoryFile.from_parsed(uri=uri, parsed=fields)
                bump_memory_version(card)
                await fs.write_file(uri, MemoryFileUtils.write(card), ctx=ctx, lease_ref=lease)
        updater = MemoryUpdater(registry=get_default_registry(), vikingdb=vikingdb,
                                transaction_handle=lease, strict_index_errors=True)
        updater._viking_fs = fs
        for uri in sorted(pending):
            await fs._delete_from_vector_store([uri], ctx=ctx)
            if await fs.exists(uri, ctx=ctx):
                refreshed = await MemoryUpdater.refresh_file_embedding(
                    viking_fs=fs, vikingdb=vikingdb, uri=uri,
                    memory_type=updater.memory_type_from_uri(uri), ctx=ctx, strict=True)
                if vikingdb is not None and not refreshed:
                    raise RuntimeError("Memory embedding queue is unavailable")
        for directory in sorted({uri.rsplit("/", 1)[0] for uri in pending}):
            # Remove stale directory previews, including their vector records.
            await fs.rm(directory + "/.overview.md", ctx=ctx, lease_ref=lease)
            await fs.rm(directory + "/.abstract.md", ctx=ctx, lease_ref=lease)
            await fs._delete_from_vector_store([directory], ctx=ctx)
            if await fs.exists(directory, ctx=ctx):
                memory_type = updater.memory_type_from_uri(directory)
                schema = updater._registry.get(memory_type)
                if schema and schema.overview_template:
                    if not await updater.generate_overview(memory_type, directory, ctx, lease_ref=lease):
                        raise RuntimeError("Failed to rebuild memory overview")
        index.pending_memory_cleanup = []
        index._dirty = True
        await index.save(fs, index_uri, ctx, lease_ref=index_lease)
        return {"updated": sum(fields is not None for fields in plans.values()),
                "deleted": sum(fields is None for fields in plans.values())}
    finally:
        await fs._async_agfs.pathlock_release(lease)
