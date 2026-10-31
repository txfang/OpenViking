# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Source-backed personal assertions that do not depend on named entities."""
from __future__ import annotations

import re
from datetime import datetime

from openviking.session.memory.fact_records import _parse, make_fact_records


_PERSONAL = re.compile(r"\b(?:I|I've|I'm|I'd|I'll|my|mine|we|we've|our)\b|(?:我|我们|我的|我们的|咱们)", re.I)
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30,
}
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")(?![A-Za-z0-9_])", re.I)
_AFTER_NUMBER = re.compile(r"^\s*-?\s*(?:more\s+|new\s+|different\s+|additional\s+)?"
                           r"([a-z][\w-]*(?:\s+[a-z][\w-]*){0,3})", re.I)
_STOP = {"since", "so", "from", "for", "at", "by", "in", "on", "to", "and", "but", "when", "which", "that"}
_TIME = re.compile(r"\b(?:a|one|two|three|four|\d+)\s+(?:days?|weeks?|months?|years?)\s+ago\b"
                   r"|\b(?:today|yesterday|last\s+(?:week|month|year))\b"
                   r"|\b\d{1,2}\s*(?:am|pm)\b"
                   r"|(?:[一二两三四五六七八九十百\d]+)个?(?:天|周|星期|月|年)前"
                   r"|(?:昨天|前天|上周|上个月|去年)", re.I)
_SUGGESTION = re.compile(r"\b(?:suggest|recommend|alternative|other\s+terms|options?|examples?)\b", re.I)
_LOW_VALUE = re.compile(r"\b(?:I(?:'m| am)?\s+(?:looking for|wondering|hoping for)|I\s+(?:think|guess|wonder|need)\b)", re.I)
_CONVERSATION_OBJECT = re.compile(
    r"\b(?:these|those|this|that|your)\s+(?:ideas?|suggestions?|options?|advice|help|responses?|answers?|recommendations?)\b", re.I)
_LIST_ITEM = re.compile(r"^\s*\d+[.)]\s+(.+?)(?:\s+[-–—:]\s+|$)")
_QUOTED_TITLE = re.compile(r"[\"“]([^\"”]{2,70})[\"”]")


def _quantity(quote: str) -> dict | None:
    match = _NUMBER.search(quote)
    if not match:
        return None
    token = match.group(1).casefold()
    value = _NUMBER_WORDS.get(token)
    if value is None:
        value = float(token) if "." in token else int(token)
    tail = re.split(r"\b(?:since|so far|from|for|at|by|in|on|to|and|but|when|which|that|where|who|while|with)\b|[,.!?]",
                    quote[match.end():], maxsplit=1, flags=re.I)[0]
    noun = _AFTER_NUMBER.match(tail)
    words = noun.group(1).split() if noun else []
    words = [word for word in words if word.casefold() not in _STOP]
    if not words:
        chinese = re.match(r"\s*([\u4e00-\u9fff]{1,10})", tail)
        if chinese:
            words = [chinese.group(1)]
    if not words:
        before = quote[:match.start()]
        count = re.search(r"\b((?:[a-z]+\s+){0,3}(?:count|total))\s+(?:to|of)\s*$", before, re.I)
        words = count.group(1).split() if count else []
        words = [word for word in words if word.casefold() not in {"my", "the", "total"}]
    obj = " ".join(words).strip(".,!? ") or None
    if re.search(r"\b(?:more|additional|another)\b|(?:新增|又增加|再添|额外增加)", quote, re.I):
        status = "increment"
    elif re.search(r"\b(?:since|so far|total|count|altogether|already|completed|finished)\b|(?:累计|总共|一共|至今|目前|现在|已完成)", quote, re.I):
        status = "total"
    else:
        status = "observation"
    return {"value": value, "object": obj, "status": status}


def _scope(quote: str) -> str | None:
    chinese = re.search(r"(?:自|从)[^，。！？]{1,70}?(?:以来|开始)|(?:今年|去年|本月|上月)", quote)
    if chinese:
        return chinese.group(0)
    match = re.search(r"\b(since\s+[^,.!?]+|in\s+my\s+[^,.!?]+)", quote, re.I)
    if not match:
        return None
    return re.split(r"\b(?:which|where|when|and|but)\b", match.group(1), maxsplit=1, flags=re.I)[0].strip()[:100]


def select_user_facts(text: str, source_id: str, recorded_at: datetime | None,
                      speaker: str = "user") -> list[dict]:
    """Keep explicit first-person assertions; never promote questions to facts."""
    from openviking.session.memory.deterministic_extractor import _resolve_dates_in_text

    selected = []
    for fact in make_fact_records(text, speaker, source_id, recorded_at, _resolve_dates_in_text):
        quote = fact["source_quotes"][source_id]
        if not _PERSONAL.search(quote) or fact["assertion_type"] in {"question", "conditional", "uncertain", "report"}:
            continue
        time = _TIME.search(quote)
        quantity = _quantity(quote)
        if _LOW_VALUE.search(quote) and not (quantity or time):
            continue
        if _CONVERSATION_OBJECT.search(quote) and not (quantity or time):
            continue
        if fact["predicate"] == "states" and not re.search(r"[\u4e00-\u9fff]", quote):
            doc = _parse(quote)
            if doc is not None:
                if not any(token.pos_ in {"VERB", "AUX"} for token in doc):
                    continue
                personal_subject = any(
                    token.dep_ in {"nsubj", "nsubjpass"} and token.lower_ in {"i", "we"}
                    for token in doc
                )
                owned_subject = any(
                    token.dep_ in {"nsubj", "nsubjpass"}
                    and any(child.lower_ in {"my", "our"} for child in token.children)
                    for token in doc
                )
                if not (personal_subject or owned_subject):
                    continue
        start = text.find(quote)
        context = ""
        if start > 0 and (re.search(r"\b(?:it|its|their|this|that|those|these)\b", quote, re.I)
                          or quantity):
            prefix = text[:start].strip()
            context = prefix[-200:] if prefix else ""
        selected.append({"fact": fact, "quantity": quantity, "scope": _scope(quote),
                         "time_expression": time.group(0) if time else None,
                         "speaker": speaker, "content": quote, "context": context})
    return selected


def select_assistant_facts(text: str, previous_user_text: str, source_id: str,
                           recorded_at: datetime | None) -> list[dict]:
    """Keep compact, source-backed suggestions and named examples from replies."""
    if not (_SUGGESTION.search(previous_user_text) or
            re.search(r"\b(?:other alternatives|other options|some suggestions)\b", text[:180], re.I)):
        return []
    from openviking.session.memory.deterministic_extractor import _resolve_dates_in_text

    facts = make_fact_records(text, "assistant", source_id, recorded_at, _resolve_dates_in_text)
    rows = []
    for line in text.splitlines():
        match = _LIST_ITEM.match(line)
        if not match:
            continue
        label = match.group(1).strip(" *\t.!")
        if not 2 <= len(label) <= 75 or len(label.split()) > 8:
            continue
        fact = next((f for f in facts if label.casefold() in f["source_quotes"][source_id].casefold()), None)
        if fact:
            rows.append({"fact": fact, "speaker": "assistant", "label": label,
                         "content": f"Assistant suggested: {label}. User asked: {previous_user_text[:160]}"})
    if 2 <= len(rows) <= 12:
        return rows
    rows = []
    for fact in facts:
        quote = fact["source_quotes"][source_id]
        if _QUOTED_TITLE.search(quote) and re.search(r"\b(?:example|cartoon|film|book|title|called|such as)\b", quote, re.I):
            rows.append({"fact": fact, "speaker": "assistant", "label": None,
                         "content": f"Assistant mentioned: {quote[:300]}. User asked: {previous_user_text[:160]}"})
    return rows[:4]
