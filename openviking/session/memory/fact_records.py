# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Conservative, source-backed assertions; no inferred inverse relationships."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

FACT_VERSION = 2
_NAME = r"[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2}"
_KIN = r"son|daughter|mother|mom|father|dad|wife|husband|brother|sister"
_KIN_NORMAL = {'mom': 'mother', 'dad': 'father'}


@lru_cache(maxsize=1024)
def _parse(text):
    from openviking.retrieve.evidence_retriever import _get_nlp
    nlp = _get_nlp()
    return nlp(text) if nlp is not None else None


def alias_in_text(alias: str, text: str, category: str | None = None) -> bool:
    alias = alias.replace('_', ' ').strip()
    if not alias:
        return False
    if re.search(r'[\u4e00-\u9fff]', alias):
        return alias in text
    matches = list(re.finditer(r'(?<![\w-])' + re.escape(alias) + r'(?![\w-])', text, re.I))
    if not matches:
        return False
    doc = _parse(text)
    if doc is None:
        return text.strip().casefold() == alias.casefold()
    for match in matches:
        spans = [e for e in doc.ents if e.start_char < match.end() and e.end_char > match.start()]
        if any(e.start_char < match.start() or e.end_char > match.end() for e in spans):
            continue
        if any(e.label_ in {'DATE', 'TIME', 'MONEY', 'QUANTITY', 'CARDINAL', 'ORDINAL'} for e in spans):
            continue
        if category == 'person' and any(e.label_ != 'PERSON' for e in spans):
            continue
        tokens = [t for t in doc if match.start() <= t.idx < match.end()]
        if not tokens:
            continue
        # NER can miss part of a multiword name; adjacent proper names still
        # form one mention, unless an explicit alias was supplied separately.
        first, last = tokens[0], tokens[-1]
        if (first.i and doc[first.i-1].pos_ == 'PROPN') or (last.i+1 < len(doc) and doc[last.i+1].pos_ == 'PROPN'):
            continue
        return True
    return False


def assertion_type(sentence: str) -> str:
    if sentence.rstrip().endswith(('?', '？')):
        return 'question'
    if re.search(r'\b(?:said|says|told|heard|thought|believe[ds]?|according to)\b|说过|说道|说：|听说|以为', sentence, re.I):
        return 'report'
    if re.search(r'\b(?:if|unless|would)\b|如果|假如|要是', sentence, re.I):
        return 'conditional'
    if re.search(r'\b(?:might|may|could|perhaps|maybe|possibly)\b|可能|也许|大概', sentence, re.I):
        return 'uncertain'
    if re.search(r"\b(?:not|never|cannot|[a-z]+n['’]t|no longer)\b|没有|并非|不是|未曾", sentence, re.I):
        return 'negation'
    if re.search(r"\b(will|plan(?:s|ned)? to|going to|want to|hope to|next month|tomorrow)\b|计划|打算|准备|明天|下周", sentence, re.I):
        return 'plan'
    if re.search(r'\b(usually|always|every|once or twice a year)\b|通常|经常|每年|每周', sentence, re.I):
        return 'habit'
    return 'fact'


def _relation(sentence: str, speaker: str):
    match = re.search(r'我(?:的)?(儿子|女儿|妈妈|母亲|父亲|爸爸)(?:叫|是)([\u4e00-\u9fff]{2,4})(?=[，。！？\s]|$)', sentence)
    if match and speaker:
        kin={'儿子':'son','女儿':'daughter','妈妈':'mother','母亲':'mother','父亲':'father','爸爸':'father'}
        return speaker, 'has_' + kin[match[1]], match[2]
    body, _ = _without_vocative(sentence)
    doc = _parse(body)
    if doc is None:
        return None
    for token in doc:
        if token.lower_ not in _KIN.split('|') or token.dep_ == 'poss':
            continue
        if any(c.dep_ in {'amod', 'compound'} for c in token.children):
            continue
        if token.i and doc[token.i-1].text == '-':
            continue
        tail = body[token.idx + len(token.text):]
        term = _KIN_NORMAL.get(token.lower_, token.lower_)
        if tail.startswith('-'):
            if not re.match(r'-in-law(?![\w-])', tail):
                continue
            term += '_in_law'
        if re.match(r"['’]s\b", tail):
            continue
        owners = [c for c in token.children if c.dep_ == 'poss']
        if len(owners) != 1:
            continue
        owner = speaker if owners[0].lower_ == 'my' else _proper_name(owners[0])
        if not owner:
            continue
        names = [_proper_name(c) for c in token.children if c.dep_ == 'appos']
        names = [n for n in names if n]
        root = token.head
        if token.dep_ == 'attr' and root.lemma_ == 'be':
            subjects = [c for c in root.children if c.dep_ == 'nsubj']
            if len(subjects) == 1:
                subject = speaker if subjects[0].lower_ == 'i' else _proper_name(subjects[0])
                if subject:
                    if owners[0].lower_ == 'my':
                        return owner, 'has_' + term, subject
                    return subject, term + '_of', owner
        if token.dep_ == 'nsubj' and root.lemma_ == 'be':
            names += [_proper_name(c) for c in root.children if c.dep_ == 'attr' and _proper_name(c)]
        if len(names) == 1:
            return owner, 'has_' + term, names[0]
        if token.dep_ in {'nsubj', 'dobj', 'pobj', 'ROOT'} and not names:
            return owner, 'has_' + term, f"{owner}'s {term.replace('_', '-')}"
    return None


def _proper_name(token):
    if token.pos_ != 'PROPN' or token.ent_type_ in {'DATE', 'TIME'}:
        return None
    parts = [t for t in token.subtree if t == token or t.dep_ in {'compound', 'flat'}]
    return ' '.join(t.text for t in sorted(parts, key=lambda t: t.i))


def _without_vocative(sentence):
    match = re.match(rf'^({_NAME}),\s+(.+)$', sentence)
    if match and match[1] not in {'I', 'We'}:
        doc = _parse(match[2])
        if doc is not None and any(t.dep_ == 'nsubj' for t in doc) and not any(
            t.text == ',' and t.i < doc[:].root.i for t in doc
        ):
            return match[2], [match[1]]
    return sentence, []


def _action(sentence, speaker, inherited_subject=None, antecedent=None):
    """Resolve only explicit subjects or a unique preceding local antecedent."""
    if re.search(r'[\u4e00-\u9fff]', sentence):
        match = re.match(r'^(我|我们|[\u4e00-\u9fff]{2,4}?)(?:昨天|今天|前天)?(去|到|飞|搬到|搬去|参加|见|庆祝|住院|做)(?:了|过)(.+?)[。！？]*$', sentence)
        if match:
            actor = speaker if match[1] == '我' else (speaker + ' and companions' if match[1] == '我们' else match[1])
            return actor, match[2], match[3].rstrip('。！？')
        return None
    sentence, _ = _without_vocative(sentence)
    doc = _parse(sentence)
    if doc is None:
        return None
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
    elif name.lower() in {'he', 'she', 'it'}:
        if not antecedent:
            return None
        name = antecedent['name']
    elif name.lower().startswith('my '):
        named = [_proper_name(t) for t in subject.subtree if t.dep_ == 'appos' and _proper_name(t)]
        kin = _relation(name, speaker)
        name = named[0] if len(named) == 1 else (kin[2] if kin else speaker + "'s " + name[3:])
    elif subject.pos_ != 'PROPN':
        return None
    elif not any(t.dep_ == 'conj' for t in subject.subtree):
        name = _proper_name(subject)
    complement = doc[root.i + 1:].text.rstrip('.!?。！？').strip()
    if not name or not complement:
        return None
    return name, root.lemma_.lower(), complement


def _local_antecedent(sentence):
    """No state across source messages; replace rather than accumulate context."""
    if assertion_type(sentence) != 'fact' or '"' in sentence or '“' in sentence:
        return None
    body, _ = _without_vocative(sentence)
    doc = _parse(body)
    if doc is None:
        return None
    if any(t.dep_ == 'nsubj' and t.pos_ == 'PRON' for t in doc):
        return None
    contraction = re.fullmatch(rf"({_NAME})['’]s\s+(.+?)[.!]?", body)
    if contraction:
        description = _parse(contraction[2])
        if description is not None and any(t.pos_ == 'ADJ' for t in description) and all(
            t.pos_ in {'ADJ', 'ADV', 'PUNCT', 'SPACE'} for t in description
        ):
            return dict(name=contraction[1], quote=sentence)
    names = {_proper_name(t) for t in doc if t.pos_ == 'PROPN' and t.dep_ != 'compound'} - {None}
    if len(names) != 1:
        return None
    name = next(iter(names))
    # A subject name, including "Oliver's hilarious", can introduce a referent.
    # An incidental name in an object/report cannot silently become the actor.
    if not re.match(re.escape(name) + r"(?:\s|['’]s\s)", body):
        return None
    if any(t.dep_ in {'dobj', 'pobj', 'dative'} and t.pos_ in {'NOUN', 'PRON', 'PROPN'} for t in doc):
        return None
    if not any(t.dep_ == 'nsubj' and _proper_name(t) == name for t in doc):
        if not (doc[:].root.pos_ == 'ADJ' and re.match(re.escape(name) + r"['’]s\s", body)):
            return None
    return dict(name=name, quote=sentence)


def _roles(sentence, speaker, action, antecedent):
    body, addressees = _without_vocative(sentence)
    roles = dict(actor=action[0] if action else None, patient=None, patient_owner=None,
                 location=None, location_owner=None, addressees=addressees)
    resolutions = []
    doc = _parse(body)
    if doc is None or not action:
        return roles, resolutions
    root = doc[:].root
    subjects = [t for t in root.children if t.dep_ == 'nsubj']
    if antecedent and any(t.lower_ in {'he', 'she', 'it'} for t in subjects):
        resolutions.append(dict(mention=subjects[0].text, resolved_to=action[0], antecedent_quote=antecedent['quote']))

    def nominal(token):
        owner = None
        possessors = [c for c in token.children if c.dep_ == 'poss']
        if len(possessors) == 1:
            poss = possessors[0]
            if poss.lower_ == 'my':
                owner = speaker
            elif poss.lower_ in {'his', 'her', 'its'}:
                # Same-clause possessives are only resolved when no second named
                # participant offers another plausible antecedent.
                names = {_proper_name(t) for t in doc if t.pos_ == 'PROPN' and t.dep_ != 'compound'} - {None}
                pronouns = {t.lower_ for t in subjects}
                matching = {'his': 'he', 'her': 'she', 'its': 'it'}[poss.lower_]
                candidates = set(names)
                if antecedent:
                    candidates.add(antecedent['name'])
                compatible = not (pronouns & {'he', 'she', 'it'}) or matching in pronouns
                if compatible and len(candidates) == 1:
                    owner = next(iter(candidates))
                    resolutions.append(dict(mention=poss.text, resolved_to=owner,
                                            antecedent_quote=antecedent['quote'] if antecedent else sentence))
            else:
                owner = _proper_name(poss)
        phrase = ' '.join(t.text for t in sorted(token.subtree, key=lambda t:t.i)
                          if t == token or t.dep_ in {'amod', 'compound'})
        return phrase, owner

    patients = [t for t in root.children if t.dep_ in {'dobj', 'obj'}]
    if len(patients) == 1:
        roles['patient'], roles['patient_owner'] = nominal(patients[0])
    locations = [c for prep in root.children if prep.dep_ == 'prep' and prep.lower_ in {'in', 'on', 'at', 'under', 'inside', 'behind', 'near'}
                 for c in prep.children if c.dep_ == 'pobj' and not any(t.ent_type_ in {'DATE', 'TIME'} for t in c.subtree)]
    if len(locations) == 1:
        roles['location'], roles['location_owner'] = nominal(locations[0])
    return roles, resolutions


def make_fact_records(text, speaker, source_id, recorded_at, resolve_dates):
    """Keep every sentence. Ambiguous relations remain attributed reports.

    Time may be unknown; message time must not silently become event time.
    Quotes remain verbatim so a reader can verify them against original text.
    """
    if not source_id:
        return []
    body = re.sub(r'^\s*\[[^]]+\]:?\s*', '', text).split('[Image URL:', 1)[0]
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[。！？])|[;；\n]+', body)
    quoted_spans = []
    for match in re.finditer(r'''"[^"]*"|“[^”]*”|(?<!\w)‘.*?’(?!\w)|(?<!\w)'(?:[^']|'(?=\w))*?'(?!\w)''', body):
        inner = _parse(match.group()[1:-1])
        # Quoted titles are nominal mentions, not necessarily reported speech.
        if assertion_type(body) == 'report' or inner is None or any(
            t.pos_ in {'VERB', 'AUX'} or t.lower_ in {'i', 'my', 'we', 'our'} for t in inner
        ):
            quoted_spans.append((match.start(), match.end()))
    clauses = []
    antecedent = None
    sentence_offset = 0
    for sentence in sentences:
        if not sentence.strip():
            continue
        start = body.find(sentence, sentence_offset)
        sentence_offset = start + len(sentence)
        quoted = any(a < sentence_offset and b > start for a, b in quoted_spans)
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
            action = _action(part.strip(), speaker, inherited, antecedent) if part.strip() else None
            clauses.append((part, action, antecedent, quoted))
            if action:
                inherited = action[0]
        antecedent = None if quoted else _local_antecedent(sentence.strip())
    result = []
    for raw, action, antecedent, quoted in clauses:
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
        kind = 'report' if quoted else assertion_type(quote)
        relations = []
        if kind not in {'question', 'report', 'conditional', 'uncertain'}:
            relations = [r for r in (_relation(quote, speaker), action) if r]
        for subject, predicate, obj in dict.fromkeys(relations or [(speaker, 'states', quote)]):
            if not subject:
                continue
            roles, resolutions = _roles(quote, speaker, action if predicate != 'states' else None, antecedent)
            if predicate.startswith('has_') or predicate.endswith('_of'):
                roles.update(actor=subject, patient=obj, patient_owner=None, location=None, location_owner=None)
                resolutions = []
            record = dict(version=FACT_VERSION, subject=subject, predicate=predicate, speaker=speaker,
                          roles=roles, resolution_evidence=resolutions,
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
        if isinstance(item, dict) and item.get('version') in {1, FACT_VERSION} and item.get('id'):
            merged[item['id']] = item
    return list(merged.values())
