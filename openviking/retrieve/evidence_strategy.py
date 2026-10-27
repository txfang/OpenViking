# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Bounded, opt-in coverage retrieval over an already authorized evidence index."""
from __future__ import annotations

import asyncio
import math
import os
import re
from functools import lru_cache
from collections.abc import Awaitable, Callable
from typing import Any

from openviking.retrieve.evidence_retriever import EvidenceIndex, extract_keys, extract_labeled_ents, _normalize_date


@lru_cache(maxsize=1)
def configured_event_cap() -> float | None:
    """Fix one cap per process. 'none' explicitly selects the uncapped ablation."""
    raw = os.environ.get("OV_EVENT_CAP", "1.0").strip().lower()
    if raw == "none":
        return None
    cap = float(raw)
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("OV_EVENT_CAP must be finite and nonnegative, or 'none'")
    return cap


def apply_events(index, hits, events, question, top_k, cap):
    """Boost existing candidates; exact date matches may supply missing sources."""
    _, keys = extract_keys(question)
    labeled = extract_labeled_ents(question)
    dates = {d for value in labeled.get("DATE", []) if (d := _normalize_date(value))}
    dates.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question))
    ranked = []
    for event in events:
        overlap = keys & event.keys
        people = {p for p in event.ents if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", question.lower())}
        date_match = bool(event.date and event.date in dates)
        # A protagonist name alone must not activate unrelated events.
        if not overlap and not date_match:
            continue
        score = len(overlap) + len(people) + 3 * date_match
        ranked.append((score, event.id, event, date_match))
    rows = {h["turn_id"]: dict(h) for h in hits}
    diagnostics = dict(cap=cap, loaded_ids=[e.id for e in events], matched=[], boosts={}, added_ids=[])
    protected = []
    for score, _, event, date_match in sorted(ranked, key=lambda r: (-r[0], r[1]))[:top_k]:
        diagnostics["matched"].append(dict(id=event.id, date=event.date, member_turns=event.member_turns, date_match=date_match))
        for turn in index.get_turns(event.member_turns):
            tid = f"{turn['episode']}:{turn['turn']}"
            if tid not in rows:
                if not date_match:
                    continue
                rows[tid] = dict(turn, turn_id=tid, score=0., query_branches=[])
                diagnostics["added_ids"].append(tid)
            # Apply cap to the total per-member boost, in units of one RRF vote.
            proposed = diagnostics["boosts"].get(tid, 0.) * 61 + .5 * score
            boost = (proposed if cap is None else min(proposed, cap)) / 61
            rows[tid]["score"] += boost - diagnostics["boosts"].get(tid, 0.)
            diagnostics["boosts"][tid] = boost
            if date_match and tid not in protected:
                protected.append(tid)
    ordered = sorted(rows.values(), key=lambda row: (-row["score"], row["turn_id"]))
    return ordered, protected, diagnostics


def _body(turn: dict) -> str:
    text = re.sub(r"^\s*\[[^]]+\]:\s*", "", turn.get("text", ""))
    return text.split("[Image URL:", 1)[0].strip()


def build_evidence_groups(
    index: EvidenceIndex, hits: list[dict], *, max_group_turns: int = 4
) -> list[dict]:
    """Preserve adjacent questions and replies, including chunk boundaries."""
    groups = []
    for hit in hits:
        ep, anchor = hit["episode"], hit["turn"]
        current = index.get_turns([f"{ep}:{anchor}"])
        if not current:
            continue
        members = {anchor: current[0]}
        previous = index.get_turns([f"{ep}:{anchor-1}"])
        if previous and "?" in _body(previous[0]):
            members[anchor-1] = previous[0]
        # Close the anchor's own question, not every new question introduced
        # by its reply; that would recursively absorb the whole conversation.
        if "?" in _body(current[0]):
            following = index.get_turns([f"{ep}:{anchor+1}"])
            if following and following[0].get("speaker") != current[0].get("speaker"):
                members[anchor+1] = following[0]
        if len(members) > max_group_turns:
            continue
        turns = [members[i] for i in sorted(members)]
        groups.append(dict(anchor_id=hit["turn_id"], turn_ids=[f"{ep}:{t['turn']}" for t in turns],
                           turns=turns, score=hit.get("score", 0),
                           query_branches=hit.get("query_branches", [0]), reason="question_answer" if len(turns)>1 else "direct"))
    return groups


def select_evidence_groups(groups: list[dict], *, top_k: int) -> list[dict]:
    """Select whole groups; reward uncovered query branches and episodes."""
    selected: dict[str, dict] = {}
    branches: set[int] = set()
    episodes: dict[str, int] = {}
    pending = list(groups)
    while pending and len(selected) < top_k:
        def priority(group):
            ep = group["turns"][0]["episode"]
            novelty = len(set(group["query_branches"]) - branches)
            score = group["score"] * (1 + .25 * novelty) / (1 + .15 * episodes.get(ep, 0))
            return (not group.get("date_protected", False), -score, group["anchor_id"])
        pending.sort(key=priority)
        group = pending.pop(0)
        additions = [(tid, t) for tid, t in zip(group["turn_ids"], group["turns"]) if tid not in selected]
        if not additions or len(selected) + len(additions) > top_k:
            continue
        selected.update(additions)
        branches.update(group["query_branches"])
        ep = group["turns"][0]["episode"]
        episodes[ep] = episodes.get(ep, 0) + 1
    return list(selected.values())


def plan_collection_queries(question: str) -> list[str]:
    """Generic relation expansions; no dataset identities or answers."""
    q = question.lower()
    if re.match(r"\s*(when|how long)\b", q):
        return []
    # Retain the original subject phrase, rather than guessing a speaker ID.
    subjects = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    subjects = [s for s in subjects if s not in {"What", "Where", "Who", "Which", "How", "In"}]
    if not subjects:
        return []
    target = " and ".join(subjects)
    expansions = [
        (r"\bpets?\b", ("pets animals names called", "adopted another new cat dog named")),
        (r"\bbooks?\b", ("books read titles reading", "recommended book finished reading")),
        (r"\bpaint(?:ed|ings?)?\b", ("painted artwork picture subject", "made art latest work image")),
        (r"\b(bought|purchased|items)\b", ("bought purchased items", "got new things shopping")),
        (r"\b(destress|de-stress|relax)\b", ("relax unwind relieve stress", "hobby therapy calming peaceful")),
        (r"\bactivities\b", ("activities hobbies recreation", "family children together trips classes outdoors")),
        (r"\b(camp|camped|camping)\b", ("camping trips places locations", "family trip stayed outdoors")),
        (r"\b(events|participated|attended)\b", ("participated attended joined events", "volunteered program community meetings")),
        (r"\b(instruments?|artists?|bands?)\b", ("music played performed instruments", "concert band musician performance")),
        (r"\bhow many times\b", ("actual visits trips dates", "went visited again another trip")),
    ]
    for pattern, phrases in expansions:
        if re.search(pattern, q):
            return [f"{question} {target} {phrase}" for phrase in phrases]
    return []


def fuse_candidates(batches: list[list[dict]]) -> list[dict]:
    merged: dict[str, dict] = {}
    for branch, batch in enumerate(batches):
        seen = set()
        for rank, hit in enumerate(batch, 1):
            tid = hit["turn_id"]
            if tid in seen:
                continue
            seen.add(tid)
            row = merged.setdefault(tid, dict(hit, score=0., query_branches=[]))
            row["score"] += 1 / (60 + rank)
            row["query_branches"].append(branch)
    return sorted(merged.values(), key=lambda r: (-r["score"], r["turn_id"]))


async def retrieve_coverage(
    index: EvidenceIndex, question: str, *,
    embed: Callable[[str], Awaitable[list[float] | None]],
    top_k: int = 15, candidate_limit: int = 50, max_supplementary_queries: int = 2,
    memory_sources=None, event_top_k: int = 2, event_cap: float | None = 1.,
) -> tuple[list[dict], dict[str, Any]]:
    """One initial retrieval plus at most two collection supplements.

    All searches use the same authorized index. Supplemental work has one
    shared 20 second deadline; completed results survive a timeout.
    """
    diagnostics: dict[str, Any] = dict(retrieval_calls=0, queries=[], errors=[])

    async def retrieve(query):
        vector = None
        try:
            vector = await embed(query)
        except Exception as exc:
            diagnostics["errors"].append(dict(stage="embedding", error_type=type(exc).__name__))
        diagnostics["retrieval_calls"] += 1
        candidates = await asyncio.to_thread(index.retrieve_candidates, query, limit=candidate_limit, q_vec=vector)
        diagnostics["queries"].append(dict(query=query, semantic_used=any("semantic_rank" in c for c in candidates),
            candidate_ids=[c["turn_id"] for c in candidates]))
        return candidates

    batches = [await retrieve(question)]
    budget = max(0, min(max_supplementary_queries, 2))

    async def supplement():
        known = {" ".join(question.lower().split())}
        for query in plan_collection_queries(question):
            key = " ".join(query.lower().split())
            if key in known:
                continue
            if diagnostics["retrieval_calls"] >= 1 + budget:
                break
            known.add(key)
            candidates = await retrieve(query)
            batches.append(candidates)

    if budget:
        try:
            await asyncio.wait_for(supplement(), timeout=20.)
        except TimeoutError:
            diagnostics["errors"].append(dict(stage="supplement", error_type="TimeoutError"))
    if memory_sources is not None:
        memory_hits = await asyncio.to_thread(memory_sources.candidates, question, index, candidate_limit)
        if memory_hits:
            batches.append(memory_hits)
        diagnostics["memories"] = dict(memory_sources.diagnostics, candidate_ids=[h["turn_id"] for h in memory_hits])
    hits = fuse_candidates(batches)
    protected = []
    if memory_sources is not None and memory_sources.events:
        hits, protected, diagnostics["events"] = await asyncio.to_thread(
            apply_events, index, hits, memory_sources.events, question, event_top_k, event_cap)
    groups = build_evidence_groups(index, hits)
    # Exact-date groups reserve room first, but cannot exceed the hard budget.
    if protected:
        for group in groups:
            if group["anchor_id"] in protected:
                group["date_protected"] = True
    evidence = select_evidence_groups(groups, top_k=top_k)
    diagnostics["candidate_count"] = len(hits)
    diagnostics["selected_ids"] = [f"{t['episode']}:{t['turn']}" for t in evidence]
    return evidence, diagnostics
