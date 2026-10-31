# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic, source-backed entity links for archived evidence windows."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone


_CODE = re.compile(
    r"(?<![\w.])(?:[\w./-]+\.(?:py|ts|tsx|js|jsx|md|json|yaml|yml|sh|toml|rs|go)"
    r"|[A-Za-z_][\w-]*(?:/[A-Za-z_][\w-]*)+|https?://[^\s\"'<>]+)"
    r"(?![\w.])", re.I,
)
_TITLE = re.compile(
    r"(?<!\w)(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+[A-Z][a-z]+\b"
    r"|[\u4e00-\u9fa5]{1,3}(?:总|经理|老师|医生|教授)"
)
_CATEGORIES = {"PERSON": "person", "ORG": "organization", "GPE": "place",
               "LOC": "place", "NORP": "group", "WORK_OF_ART": "work"}
_NOISE = {"inc", "ltd", "what", "where", "who", "which", "how", "when", "the", "a", "an"}
MAX_SIGNALS_PER_TURN = 64
MAX_ENTITIES_PER_WINDOW = 128
MAX_GRAPH_EDGES = 100000


@dataclass(frozen=True)
class GraphEntitySignal:
    key: str
    category: str
    canonical: str
    family: str
    snippet: str


def canonical_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).strip(".,;:!?\"'` ")


def extract_graph_entity_signals(text: str) -> list[GraphEntitySignal]:
    """Use independent code regex and NER; co-occurrence is never an identity vote."""
    signals: list[GraphEntitySignal] = []
    seen: set[tuple[str, str]] = set()
    code_matches = list(_CODE.finditer(text))
    for match in code_matches:
        if len(signals) >= MAX_SIGNALS_PER_TURN:
            break
        canonical = canonical_name(match.group())
        if canonical and len(canonical) <= 120 and ("code:" + canonical, "regex") not in seen:
            seen.add(("code:" + canonical, "regex"))
            signals.append(GraphEntitySignal("code:" + canonical, "code", canonical,
                                             "regex", text[max(0, match.start()-40):match.end()+40][:120]))
    for match in _TITLE.finditer(text):
        if len(signals) >= MAX_SIGNALS_PER_TURN:
            break
        canonical = canonical_name(match.group())
        key = "person:" + canonical
        if len(canonical) <= 120 and (key, "title") not in seen:
            seen.add((key, "title"))
            signals.append(GraphEntitySignal(key, "person", canonical, "title",
                                             text[max(0, match.start()-40):match.end()+40][:120]))
    # Local import avoids coupling the index serializer to the spaCy loader.
    from openviking.retrieve.evidence_retriever import extract_labeled_ents
    for label, values in extract_labeled_ents(text).items():
        category = _CATEGORIES.get(label)
        if category is None:
            continue
        for value in sorted(values):
            if len(signals) >= MAX_SIGNALS_PER_TURN:
                return signals
            canonical = canonical_name(value)
            if not canonical or canonical in _NOISE or not 2 <= len(canonical) <= 120:
                continue
            key = category + ":" + canonical
            if (key, "ner") in seen:
                continue
            pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
            match = next((m for m in re.finditer(pattern, text, re.I)
                          if not any(code.start() <= m.start() and m.end() <= code.end()
                                     for code in code_matches)), None)
            if match is None:
                continue
            seen.add((key, "ner"))
            snippet = text[max(0, match.start()-40):match.end()+40][:120]
            signals.append(GraphEntitySignal(key, category, canonical, "ner", snippet))
    return signals


def window_id(archive_uri: str) -> str:
    return hashlib.sha256(archive_uri.rstrip("/").encode()).hexdigest()[:24]


def parse_recorded_time(value: str) -> str | None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # OpenViking's existing archive timestamps may be naive. Interpret
        # them consistently as UTC rather than losing the entire time view.
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, AttributeError):
        return None


def empty_graph() -> dict:
    return {"windows": {}, "appears_in": {}, "adjacent": {},
            "pending_archive_uris": [], "needs_rebuild": False}


def build_edges(index, window_ids: list[str], *, min_votes: int = 2,
                min_agreement: float = 1.0) -> dict:
    """Recompute bounded derived edges from source turns; replay cannot add weight."""
    if min_votes < 1 or not 0 < min_agreement <= 1:
        raise ValueError("Invalid graph confirmation threshold")
    windows = index.graph["windows"]
    changed = set(window_ids)
    appearances: dict[str, dict] = {
        key: dict(edge) for key, edge in index.graph["appears_in"].items()
        if edge["window_id"] in windows and edge["window_id"] not in changed
    }
    support: dict[str, set[str]] = {}
    for edge in appearances.values():
        support.setdefault(edge["entity_key"], set()).add(edge["window_id"])
    for wid in sorted(changed):
        window = windows.get(wid)
        if window is None:
            continue
        window["entity_truncated"] = False
        window_entity_count = 0
        for tid in window["turn_ids"]:
            turn = index._turn_by_id.get(tid)
            if turn is None:
                continue
            for signal in extract_graph_entity_signals(turn["text"]):
                edge_key = wid + "|" + signal.key
                if edge_key not in appearances and window_entity_count >= MAX_ENTITIES_PER_WINDOW:
                    window["entity_truncated"] = True
                    break
                if edge_key not in appearances:
                    window_entity_count += 1
                edge = appearances.setdefault(edge_key, {
                    "window_id": wid, "entity_key": signal.key,
                    "category": signal.category, "canonical": signal.canonical,
                    "turn_ids": [], "source_ids": [], "signals": [],
                    "snippet": signal.snippet, "weight": .5,
                    "support_window_ids": [], "reason": "",
                })
                if tid not in edge["turn_ids"]:
                    edge["turn_ids"].append(tid)
                source = turn.get("source_id")
                if source and source not in edge["source_ids"]:
                    edge["source_ids"].append(source)
                if signal.family not in edge["signals"]:
                    edge["signals"].append(signal.family)
                support.setdefault(signal.key, set()).add(wid)
            if window["entity_truncated"]:
                break
    by_name: dict[str, dict[str, set[str]]] = {}
    for edge in appearances.values():
        by_name.setdefault(edge["canonical"], {}).setdefault(edge["category"], set()).add(edge["window_id"])
    for edge in appearances.values():
        supporting = sorted(support[edge["entity_key"]])
        edge["support_window_ids"] = supporting[:5]
        edge["support_window_count"] = len(supporting)
        votes = len(supporting) + max(0, len(edge["signals"]) - 1)
        categories = by_name[edge["canonical"]]
        agreement = len(categories[edge["category"]]) / sum(len(windows) for windows in categories.values())
        edge["agreement"] = agreement
        edge["conflict_categories"] = sorted(categories) if len(categories) > 1 else []
        confirmed = votes >= min_votes and agreement >= min_agreement
        edge["weight"] = (.95 if confirmed and len(supporting) >= 3
                          else .85 if confirmed
                          else .7 if "regex" in edge["signals"] else .5)
        edge["reason"] = (f"signals={','.join(sorted(edge['signals']))}; "
                          f"window_count={len(supporting)}; "
                          f"windows={','.join(supporting[:5])}; turns={','.join(edge['turn_ids'][:5])}")[:300]
    if len(appearances) > MAX_GRAPH_EDGES:
        raise ValueError("Evidence graph edge limit exceeded")
    adjacent = {}
    by_session: dict[str, list[tuple[str, dict]]] = {}
    for wid, window in windows.items():
        by_session.setdefault(window["session_uri"], []).append((wid, window))
    for items in by_session.values():
        def order(row):
            name = row[1]["archive_uri"].rsplit("/", 1)[-1]
            match = re.fullmatch(r"archive_(\d+)", name)
            return (int(match[1]) if match else -1, row[1]["archive_uri"])
        items.sort(key=order)
        for (left, _), (right, _) in zip(items, items[1:]):
            adjacent[left + "|" + right] = {"from_window_id": left, "to_window_id": right,
                                           "weight": .5, "scope": "same_session"}
    timed = []
    for wid, window in windows.items():
        start = parse_recorded_time(window["when_start"])
        end = parse_recorded_time(window["when_end"])
        if start and end:
            timed.append((datetime.fromisoformat(start), datetime.fromisoformat(end), wid, window))
    timed.sort()
    for (left_start, left_end, left, left_window), (right_start, _, right, right_window) in zip(timed, timed[1:]):
        if (left_window["session_uri"] != right_window["session_uri"]
                and left_window["episode"] != right_window["episode"]
                and left_start < right_start and left_end <= right_start
                and (right_start - left_end).total_seconds() <= 86400):
            adjacent[left + "|" + right] = {"from_window_id": left, "to_window_id": right,
                                           "weight": .3, "scope": "cross_episode"}
    return {"appears_in": appearances, "adjacent": adjacent}
