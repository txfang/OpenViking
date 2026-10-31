# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Authorized, bounded memory hints whose assertions can be checked in source turns."""
from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

from openviking.retrieve.evidence_retriever import EvidenceIndex, EventNode, extract_keys
from openviking.session.memory.fact_records import make_fact_records
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils


@lru_cache(maxsize=1024)
def _regenerated_facts(text, speaker, source_id, date):
    """Cache immutable extraction results by complete input, never by URI alone."""
    from openviking.session.memory.deterministic_extractor import _resolve_dates_in_text
    ts = datetime.fromisoformat(date.replace("Z", "+00:00")) if date else None
    return frozenset(json.dumps(f, sort_keys=True, ensure_ascii=False) for f in
                     make_fact_records(text, speaker, source_id, ts, _resolve_dates_in_text))


def verify_fact(fact, index: EvidenceIndex, cache=None) -> bool:
    """Re-extract from the authorized source, rather than trusting card metadata."""
    if not isinstance(fact, dict):
        return False
    ids = fact.get("source_turn_ids")
    if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str):
        return False
    cache = {} if cache is None else cache
    if ids[0] not in cache:
        turns = index.resolve_sources(ids)
        regenerated = frozenset()
        if len(turns) == 1:
            turn = turns[0]
            try:
                regenerated = _regenerated_facts(turn["text"], turn["speaker"], ids[0], turn.get("date"))
            except (ValueError, TypeError):
                return False
        cache[ids[0]] = regenerated
    return json.dumps(fact, sort_keys=True, ensure_ascii=False) in cache[ids[0]]


def _valid_fact_shape(fact) -> bool:
    return (isinstance(fact, dict)
            and all(isinstance(fact.get(key), str) for key in
                    ("id", "subject", "predicate", "object", "speaker", "assertion_type"))
            and isinstance(fact.get("source_turn_ids"), list)
            and len(fact["source_turn_ids"]) == 1
            and isinstance(fact["source_turn_ids"][0], str))


@dataclass
class MemorySources:
    facts: list[dict] = field(default_factory=list)
    events: list[EventNode] = field(default_factory=list)
    metadata_by_id: dict[str, dict] = field(default_factory=dict)
    selected_facts: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=lambda: dict(cards=0, rejected_facts=0, errors=[], truncated=False))

    def candidates(self, question: str, index: EvidenceIndex, limit: int) -> list[dict]:
        """Rank source-backed hints, then return only original evidence candidates."""
        _, query_keys = extract_keys(question)
        assistant_question = bool(re.search(
            r"\b(?:you|your)\b.{0,40}\b(?:suggest\w*|recommend\w*|mention\w*|say|said|list\w*|tell|told|give|gave|offer\w*)\b"
            r"|你.{0,20}(?:建议|推荐|提到|提过|说过|列出)", question, re.I))
        requested_speaker = "assistant" if assistant_question else "user"
        ranked = []
        # Reuse source lexical features: do not reparse the entire fact corpus
        # with NLP on every query. Explicit subjects supply resolved pronouns.
        for fact in self.facts:
            if not _valid_fact_shape(fact):
                self.diagnostics["rejected_facts"] += 1
                continue
            if fact.get("speaker") != requested_speaker:
                continue
            if assistant_question and fact["id"] not in self.metadata_by_id:
                continue
            turns = index.resolve_sources(fact["source_turn_ids"])
            keys = set().union(*(set(t.get("keys", [])) for t in turns))
            subject_keys = set(re.findall(r"\w+", fact["subject"].lower()))
            keys.update(subject_keys)
            keys.add(fact["predicate"])
            overlap = query_keys & keys
            if not (overlap - subject_keys):
                continue
            score = sum(math.log1p(index._n_turns / (len(index._lex_inv.get(k, [])) + 1)) for k in overlap)
            ranked.append((score, fact["id"], fact))
        hits = {}
        cache = {}
        verified = 0
        self.selected_facts = []
        for score, _, fact in sorted(ranked, key=lambda row: (-row[0], row[1])):
            if not verify_fact(fact, index, cache):
                self.diagnostics["rejected_facts"] += 1
                continue
            verified += 1
            if len(self.selected_facts) < 12:
                metadata = self.metadata_by_id.get(fact["id"], {})
                self.selected_facts.append(dict(
                    id=fact["id"], subject=fact["subject"][:100],
                    predicate=fact["predicate"], object=fact["object"][:200],
                    quantity=metadata.get("quantity"), scope=metadata.get("scope"),
                    time_expression=metadata.get("time_expression"),
                    event_time=fact.get("time"), recorded_at=fact.get("recorded_at"),
                    speaker=fact.get("speaker"), source_turn_ids=fact["source_turn_ids"],
                ))
            for turn in index.resolve_sources(fact["source_turn_ids"]):
                tid = f"{turn['episode']}:{turn['turn']}"
                hits.setdefault(tid, dict(turn, turn_id=tid, score=float(score)))
            if len(hits) >= limit:
                break
        self.diagnostics["verified_facts"] = verified
        return list(hits.values())[:limit]


async def load_memory_sources(fs, index, user_uri, ctx, *, use_memories=False, use_events=False):
    """Read cards only through request-authorized VikingFS, with bounded I/O.

    Legacy summaries without verified records never supply candidates. Event
    membership comes from exact message IDs, not directory dates or proximity.
    """
    from openviking.session.memory.deterministic_extractor import is_completed_event

    result = MemorySources()
    cache = {}
    result.diagnostics["records_checked"] = 0
    seen = set()
    roots = (["entities", "facts"] if use_memories else []) + (["events"] if use_events else [])
    semaphore = asyncio.Semaphore(8)
    for category in roots:
        root = f"{user_uri.rstrip('/')}/memories/{category}"
        try:
            listing = await fs.glob(pattern="**/*.md", uri=root, ctx=ctx, force_fs=True, node_limit=2001)
        except Exception as exc:
            result.diagnostics["errors"].append(dict(stage="glob", error_type=type(exc).__name__))
            continue
        paths = sorted({p for p in listing.get("matches", []) if isinstance(p, str)
                        and p.startswith(root + "/") and not p.rsplit("/", 1)[-1].startswith(".")})
        result.diagnostics["truncated"] |= len(paths) > 2000 or bool(listing.get("truncated"))

        async def read(path):
            async with semaphore:
                try:
                    raw = await fs.read_file(path, ctx=ctx)
                    if len(raw) > 2_000_000:
                        result.diagnostics["truncated"] = True
                        return path, {}
                    return path, MemoryFileUtils.read(raw, uri=path).extra_fields
                except Exception as exc:
                    result.diagnostics["errors"].append(dict(stage="read", error_type=type(exc).__name__))
                    return path, {}

        def consume(path, fields):
            result.diagnostics["cards"] += 1
            records = fields.get("fact_records", [])
            if not isinstance(records, list):
                result.diagnostics["rejected_facts"] += 1
                return
            for fact in records:
                result.diagnostics["records_checked"] += 1
                if result.diagnostics["records_checked"] > 20000:
                    result.diagnostics["truncated"] = True
                    return
                if not _valid_fact_shape(fact):
                    result.diagnostics["rejected_facts"] += 1
                    continue
                if category in {"entities", "facts"}:
                    if fact["id"] not in seen and fact["assertion_type"] != "question":
                        seen.add(fact["id"])
                        result.facts.append(fact)
                    if category == "facts":
                        result.metadata_by_id[fact["id"]] = {
                            "quantity": fields.get("quantity"), "scope": fields.get("scope"),
                            "time_expression": fields.get("time_expression"),
                        }
                    continue
                if not verify_fact(fact, index, cache):
                    result.diagnostics["rejected_facts"] += 1
                    continue
                event_type = fields.get("event_type")
                if not isinstance(event_type, str):
                    result.diagnostics["rejected_facts"] += 1
                    continue
                quote = next(iter(fact["source_quotes"].values()))
                if not is_completed_event(fact, event_type):
                    continue
                turns = index.resolve_sources(fact["source_turn_ids"])
                _, keys = extract_keys(quote)
                result.events.append(EventNode(
                    id=f"{path}#{fact['id']}", date=fact["time"] if fact["time_precision"] == "day" else "",
                    name=quote, ents={fact["subject"].lower()}, keys=keys,
                    member_turns=[f"{t['episode']}:{t['turn']}" for t in turns],
                ))

        for offset in range(0, min(len(paths), 2000), 8):
            cards = await asyncio.gather(*(read(p) for p in paths[offset:offset + 8]))
            for path, fields in cards:
                await asyncio.to_thread(consume, path, fields)
                if result.diagnostics["records_checked"] > 20000:
                    break
            if result.diagnostics["records_checked"] > 20000:
                break
    result.diagnostics.update(facts=len(result.facts), events=len(result.events))
    return result
