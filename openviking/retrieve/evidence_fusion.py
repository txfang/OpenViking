# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Opt-in graph and temporal candidates over one authorized evidence snapshot."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from openviking.retrieve.evidence_graph import extract_graph_entity_signals, parse_recorded_time


@dataclass(frozen=True)
class AuxHit:
    turn_id: str
    channel: str
    raw_score: float
    source_window_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceIntent:
    name: str
    rho: float
    channel: str | None


_RELATION = re.compile(r"\b(relationship|related|relation|connect|between|affect|involve|belong|associated|whose|who is)\b|关系|关联|属于|涉及", re.I)
_OPEN = re.compile(r"\b(?:open\s+(?:items|tasks|issues|tickets)|pending|unfinished|outstanding|still need|todo|remaining|remains)\b|待办|未完成|还需|尚未", re.I)
_MONTHS = {name: i for i, name in enumerate(("january", "february", "march", "april", "may", "june",
                                               "july", "august", "september", "october", "november", "december"), 1)}
_MONTH_PATTERN = "|".join(_MONTHS)


def _date_span(question: str) -> tuple[datetime, datetime] | None:
    m = re.search(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", question)
    if m:
        try:
            day = datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
            return day, day.replace(hour=23, minute=59, second=59)
        except ValueError:
            return None
    m = re.search(rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b", question, re.I)
    if m:
        month, year = _MONTHS[m[1].lower()], int(m[2])
    else:
        m = re.search(r"(?<!\d)(\d{4})年\s*(\d{1,2})月", question)
        if not m:
            return None
        year, month = int(m[1]), int(m[2])
    try:
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        next_month = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta
        return start, next_month - timedelta(seconds=1)
    except ValueError:
        return None


def classify_evidence_intent(question: str) -> EvidenceIntent:
    if _date_span(question):
        return EvidenceIntent("HISTORICAL_DECISIONS", .7, "time")
    if _OPEN.search(question):
        return EvidenceIntent("OPEN_ITEMS", .7, "time")
    if _RELATION.search(question):
        return EvidenceIntent("ENTITY_RELATIONS", .3, "graph")
    return EvidenceIntent("OTHER", 1., None)


def graph_candidates(index, question: str, *, max_seeds: int = 5, per_seed: int = 5) -> list[AuxHit]:
    if index.graph_unavailable or not index.graph["windows"]:
        return []
    seeds = sorted({s.key for s in extract_graph_entity_signals(question)},
                   key=lambda key: (-len(key.split(":", 1)[1]), key))[:max_seeds]
    if not seeds:
        return []
    matches: dict[str, AuxHit] = {}
    from openviking.retrieve.evidence_retriever import _STOP, extract_keys
    query_keys = extract_keys(question)[1]
    query_terms = {w for w in re.findall(r"[A-Za-z]{4,}", question.lower()) if w not in _STOP}
    for seed in seeds:
        edges = [index.graph["appears_in"][key] for key in index._graph_inv.get(seed, ())
                 if index.graph["appears_in"][key]["weight"] >= .7]
        # Very common names are poor graph seeds; keep exact rare names.
        if index._n_turns >= 10 and sum(len(e["turn_ids"]) for e in edges) > index._n_turns * .4:
            continue
        edges.sort(key=lambda e: (-e["weight"], e["window_id"]))
        count = 0
        for edge in edges:
            for tid in edge["turn_ids"]:
                if tid not in index._turn_by_id:
                    continue
                hit = AuxHit(tid, "graph", edge["weight"], (edge["window_id"],))
                if tid not in matches or hit.raw_score > matches[tid].raw_score:
                    matches[tid] = hit
                count += 1
                if count >= per_seed:
                    break
            if count >= per_seed:
                break
            for adjacent in index._graph_adj_inv.get(edge["window_id"], ()):
                if adjacent.get("scope") != "same_session":
                    continue
                neighbor = (adjacent["to_window_id"] if adjacent["from_window_id"] == edge["window_id"]
                            else adjacent["from_window_id"] if adjacent["to_window_id"] == edge["window_id"] else None)
                if neighbor is None:
                    continue
                window = index.graph["windows"].get(neighbor)
                if not window:
                    continue
                for tid in window["turn_ids"]:
                    turn = index._turn_by_id.get(tid)
                    if turn is None:
                        continue
                    terms = set(re.findall(r"[A-Za-z]{4,}", turn["text"].lower()))
                    if not (query_keys & set(turn.get("keys", [])) or query_terms & terms):
                        continue
                    hit = AuxHit(tid, "graph", edge["weight"] * .5, (neighbor,))
                    if tid not in matches:
                        matches[tid] = hit
                    count += 1
                    if count >= per_seed:
                        break
                if count >= per_seed:
                    break
    return sorted(matches.values(), key=lambda h: (-h.raw_score, h.turn_id))[:max_seeds * per_seed]


def time_candidates(index, question: str, *, now: datetime, content_turn_ids: set[str],
                    limit: int = 5) -> list[AuxHit]:
    if index.graph_unavailable:
        return []
    span = _date_span(question)
    open_items = bool(_OPEN.search(question)) and span is None
    if span is None and not open_items:
        return []
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    rows = []
    for wid, window in index.graph["windows"].items():
        start = parse_recorded_time(window.get("when_start"))
        end = parse_recorded_time(window.get("when_end"))
        if start is None or end is None:
            continue
        lo, hi = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if span:
            if hi < span[0] or lo > span[1]:
                continue
            distance = 0
        else:
            distance = max(0, (now - hi).total_seconds() / 86400)
        score = math.exp(-distance / 30)
        for tid in window["turn_ids"]:
            if tid in index._turn_by_id and (not open_items or tid in content_turn_ids):
                rows.append(AuxHit(tid, "time", score, (wid,)))
    by_id = {hit.turn_id: hit for hit in sorted(rows, key=lambda h: (-h.raw_score, h.turn_id))}
    return sorted(by_id.values(), key=lambda h: (-h.raw_score, h.turn_id))[:limit]


def fuse_auxiliary(content_hits: list[dict], graph_hits: list[AuxHit], time_hits: list[AuxHit],
                   intent: EvidenceIntent, *, candidate_limit: int) -> tuple[list[dict], dict]:
    auxiliary = graph_hits if intent.channel == "graph" else time_hits if intent.channel == "time" else []
    if not auxiliary:
        return content_hits, {"intent": intent.name, "rho": 1., "selected_channels": ["content"],
                              "fallback_reason": "no_matching_auxiliary"}
    max_content = max((h.get("score", 0.) for h in content_hits), default=0.)
    max_aux = max(h.raw_score for h in auxiliary)
    rows = {h["turn_id"]: dict(h, channels=list(h.get("channels", ["content"]))) for h in content_hits}
    for row in rows.values():
        row["score"] = intent.rho * row.get("score", 0.) / max_content if max_content else 0.
    for hit in auxiliary:
        row = rows.setdefault(hit.turn_id, {"turn_id": hit.turn_id, "query_branches": [0],
                                            "score": 0., "channels": []})
        row["score"] += (1 - intent.rho) * hit.raw_score / max_aux
        if hit.channel not in row["channels"]:
            row["channels"].append(hit.channel)
    ordered = sorted(rows.values(), key=lambda row: (-row["score"], row["turn_id"]))[:max(1, min(candidate_limit, 50))]
    return ordered, {"intent": intent.name, "rho": intent.rho,
                     "selected_channels": ["content", intent.channel], "fallback_reason": None}
