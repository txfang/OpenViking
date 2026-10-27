# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Conservative, source-backed assertions; no inferred inverse relationships."""
from __future__ import annotations

import hashlib
import json
import re

FACT_VERSION = 1
_NAME = r"[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2}"
_KIN = r"son|daughter|mother|mom|father|dad|wife|husband|brother|sister"
_KIN_NORMAL = {'mom': 'mother', 'dad': 'father'}


def alias_in_text(alias: str, text: str) -> bool:
    alias = alias.replace('_', ' ').strip()
    if not alias:
        return False
    if re.search(r'[\u4e00-\u9fff]', alias):
        return alias in text
    return bool(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text, re.I))


def assertion_type(sentence: str) -> str:
    if sentence.rstrip().endswith(('?', '？')):
        return 'question'
    if re.search(r'\b(?:said|says|told|heard|thought|believe[ds]?|according to)\b|说过|说道|说：|听说|以为', sentence, re.I):
        return 'report'
    if re.search(r'\b(?:if|unless|would)\b|如果|假如|要是', sentence, re.I):
        return 'conditional'
    if re.search(r'\b(?:might|may|could|perhaps|maybe|possibly)\b|可能|也许|大概', sentence, re.I):
        return 'uncertain'
    if re.search(r"\b(not|never|haven't|hasn't|didn't|don't|doesn't|no longer)\b|没有|并非|不是|未曾", sentence, re.I):
        return 'negation'
    if re.search(r"\b(will|plan(?:s|ned)? to|going to|want to|hope to|next month|tomorrow)\b|计划|打算|准备|明天|下周", sentence, re.I):
        return 'plan'
    if re.search(r'\b(usually|always|every|once or twice a year)\b|通常|经常|每年|每周', sentence, re.I):
        return 'habit'
    return 'fact'


def _relation(sentence: str, speaker: str):
    # A vocative such as "Caroline, my son ..." is never a relation target.
    match = re.search(rf"\b({_NAME})\s+is\s+(?:not\s+)?my\s+({_KIN})\b", sentence)
    if match and speaker:
        return speaker, 'has_' + _KIN_NORMAL.get(match[2], match[2]), match[1]
    match = re.search(rf"\b[Mm]y\s+({_KIN})\s+(?:(?:is|named|called)\s+)?({_NAME})\b", sentence)
    if match and speaker:
        return speaker, 'has_' + _KIN_NORMAL.get(match[1], match[1]), match[2]
    match = re.search(rf"\b({_NAME}|I)\s+(?:is|am)\s+(?:not\s+)?({_NAME})['’]s\s+({_KIN})\b", sentence)
    if match:
        subject = speaker if match[1] == 'I' else match[1]
        return subject, _KIN_NORMAL.get(match[3], match[3]) + '_of', match[2]
    match = re.search(r'我(?:的)?(儿子|女儿|妈妈|母亲|父亲|爸爸)(?:叫|是)([\u4e00-\u9fff]{2,4})(?=[，。！？\s]|$)', sentence)
    if match and speaker:
        kin={'儿子':'son','女儿':'daughter','妈妈':'mother','母亲':'mother','父亲':'father','爸爸':'father'}
        return speaker, 'has_' + kin[match[1]], match[2]
    match = re.search(rf'\bmy\s+({_KIN})\b', sentence, re.I)
    if match and speaker:
        term = _KIN_NORMAL.get(match[1].lower(), match[1].lower())
        return speaker, 'has_' + term, f"{speaker}'s {term}"
    return None


def _action(sentence, speaker, inherited_subject=None):
    """Only explicit main-clause subjects; do not resolve third-person pronouns."""
    if re.search(r'[\u4e00-\u9fff]', sentence):
        match = re.match(r'^(我|我们|[\u4e00-\u9fff]{2,4}?)(?:昨天|今天|前天)?(去|到|飞|搬到|搬去|参加|见|庆祝|住院|做)(?:了|过)(.+?)[。！？]*$', sentence)
        if match:
            actor = speaker if match[1] == '我' else (speaker + ' and companions' if match[1] == '我们' else match[1])
            return actor, match[2], match[3].rstrip('。！？')
        return None
    from openviking.retrieve.evidence_retriever import _get_nlp
    nlp = _get_nlp()
    if nlp is None:
        return None
    doc = nlp(sentence)
    root = next((t for t in doc if t.dep_ == 'ROOT' and t.pos_ in {'VERB', 'AUX'}), None)
    if root is None:
        return None
    subjects = [t for t in root.children if t.dep_ == 'nsubj']
    # Small English models may label both a vocative and the actual actor as
    # subjects. A comma-separated leading name must not win over the actor.
    if len(subjects) > 1 and any(t.text == ',' for t in doc[:subjects[-1].i]):
        subjects = subjects[-1:]
    subject = subjects[0] if len(subjects) == 1 else None
    if subject is None and inherited_subject and re.match(r'^(?:will|would|did|have|had|was|went|moved|visited)\b', sentence, re.I):
        action = _action('I ' + sentence, speaker)
        return (inherited_subject, action[1], action[2]) if action else None
    if subject is None:
        return None
    name = doc[subject.left_edge.i:subject.right_edge.i + 1].text
    if name.lower() == 'i':
        name = speaker
    elif name.lower() == 'we':
        name = speaker + ' and companions'
    elif name.lower().startswith('my '):
        kin = _relation(name, speaker)
        name = kin[2] if kin else speaker + "'s " + name[3:]
    elif subject.pos_ != 'PROPN':
        return None
    complement = doc[root.i + 1:].text.rstrip('.!?。！？').strip()
    if not name or not complement:
        return None
    return name, root.lemma_.lower(), complement


def make_fact_records(text, speaker, source_id, recorded_at, resolve_dates):
    """Keep every sentence. Ambiguous relations remain attributed reports.

    Time may be unknown; message time must not silently become event time.
    Quotes remain verbatim so a reader can verify them against original text.
    """
    if not source_id:
        return []
    body = re.sub(r'^\s*\[[^]]+\]:?\s*', '', text).split('[Image URL:', 1)[0]
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[。！？])|[;；\n]+', body)
    clauses = []
    for sentence in sentences:
        # Only separate coordinated clauses with explicit subjects/modals.
        # Reports/conditionals retain the entire scope as an attributed report.
        parts = []
        offset = 0
        if assertion_type(sentence) not in {'report', 'conditional'}:
            for match in re.finditer(r',?\s+(?:but|and)\s+', sentence):
                left = sentence[offset:match.start()]
                action = _action(left.strip(), speaker)
                # Both sides need an action. "Alice and Bob went ..." and
                # "Luna and Oliver!" are coordinated names, not two clauses.
                right = _action(sentence[match.end():].strip(), speaker, action[0] if action else None)
                if action and right:
                    parts.append(left)
                    offset = match.end()
        parts.append(sentence[offset:])
        inherited = None
        for part in parts:
            action = _action(part.strip(), speaker, inherited) if part.strip() else None
            clauses.append((part, action))
            if action:
                inherited = action[0]
    result = []
    for raw, action in clauses:
        quote = raw.strip()
        if not quote:
            continue
        _, dates = resolve_dates(quote, recorded_at)
        precision = 'day' if dates else 'unknown'
        time = dates[0] if len(dates) == 1 else None
        if re.search(r'\b(last|this|next) year\b|去年|今年|明年', quote, re.I):
            precision = 'year'; time = time[:4] if time else None
        elif re.search(r'\b(last|this|next) week\b|上周|下周|本周', quote, re.I):
            precision = 'week'
        elif re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', quote, re.I):
            precision = 'month'; time = time[:7] if time else None
        kind = assertion_type(quote)
        relations = []
        if kind not in {'question', 'report', 'conditional', 'uncertain'}:
            relations = [r for r in (_relation(quote, speaker), action) if r]
        for subject, predicate, obj in dict.fromkeys(relations or [(speaker, 'states', quote)]):
            if not subject:
                continue
            record = dict(version=FACT_VERSION, subject=subject, predicate=predicate,
                          object=obj, time=time, time_precision=precision, assertion_type=kind,
                          recorded_at=recorded_at.isoformat() if recorded_at else None,
                          source_turn_ids=[source_id], source_quotes={source_id:quote})
            record['id'] = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
            result.append(record)
    return result


def merge_fact_records(old, new):
    """Deduplicate identities without truncating the authoritative facts."""
    merged = {}
    for item in list(old or []) + list(new or []):
        if isinstance(item, dict) and item.get('version') == FACT_VERSION and item.get('id'):
            merged[item['id']] = item
    return list(merged.values())
