# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Evidence index retriever for OpenViking.
Zero-Mem / EdgeMem style minimal index: entity-context anchors + lexical keys
+ co-occurrence hyperedges + temporal hierarchy pointing at raw session turns.
Design notes (see project_zero_mem_optimization memory):
- Raw context stays byte-faithful in sessions/<sid>/history/archive_*/messages.jsonl
  (1 line = 1 message). This index NEVER mirrors or rewrites raw text; it only
  stores lexical/entity keys and episode structure so retrieval can point back
  at the original turn.
- Episode merging relies on deterministic session_id encoding from the import
  script: ``sample{sid}_s{sess_num}_c{chunk_idx}``. Chunks of the same episode
  are merged by chunk index so the episode window stays contiguous even though
  the 8-message chunk boundary splits the conversation.
- Zero LLM calls at build/retrieve time: spaCy NER + deterministic lexical
  features only. The final answer reader is the only LLM step.
- Fail-open: any failure degrades to returning an empty evidence list; the
  caller falls back to the legacy ``find`` path.
Version 2 adds an optional semantic channel (E-plan): turn texts are embedded
once at write time (``emb`` field, 384-d bge dense vectors, zero LLM / zero
token) and ``retrieve(use_semantic=True)`` runs the ALG-A algorithm — lexical
+ entity additive scoring as before, then semantic cosine anchors (top-K by
similarity) join the episode-window / co-occurrence anchor sets, and the
semantic anchors plus their ±closure_radius neighbors are force-included in
the final package (Zero-Mem Eq.14 "Local Neighbors" / EdgeMem episode
activation: closure is inclusion, not scoring). Any turn without ``emb``, a
missing query vector, or an unconfigured embedder degrades to the version-1
four-channel path byte-for-byte.
"""
import glob
import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Deterministic lexical / entity extraction (ported from index_proto.py)
# ---------------------------------------------------------------------------
_STOP = set(
    """i me my we our you your he his she her it its they their them a an the and or
    but if then so of in on at to for with from by as is are was were be been being
    have has had do does did will would can could should shall may might must about
    into over under again further once here there all any both each few more most
    other some such no nor not only own same than too very s t can now just don
    really lot lots yeah oh hey wow thanks thank great good going go get got want
    gonna im thats dont know tell let us """.split()
)
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "NORP", "WORK_OF_ART"}
_KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
# Event hyperedge helpers ------------------------------------------------------
# Deterministic event-card date normalization (probe phase, zero LLM).
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
def _normalize_date(text: str) -> Optional[str]:
    """Normalize a DATE entity to ``YYYY-MM-DD`` (or None when unparseable)."""
    s = (text or "").strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.fullmatch(r"([a-z]+) (\d{1,2}),? (\d{4})", s)  # may 8 2023 / may 8, 2023
    if m:
        mn = _MONTHS.get(m.group(1)) or _MONTH_ABBR.get(m.group(1))
        if mn:
            return f"{m.group(3)}-{mn:02d}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{4})", s)  # 8 may 2023
    if m:
        mn = _MONTHS.get(m.group(2)) or _MONTH_ABBR.get(m.group(2))
        if mn:
            return f"{m.group(3)}-{mn:02d}-{int(m.group(1)):02d}"
    return None
@dataclass
class EventNode:
    """One event hyperedge derived from a deterministic event card."""
    id: str
    date: str  # YYYY-MM-DD (card directory date)
    name: str
    ents: Set[str] = field(default_factory=set)  # participants (lowercased)
    keys: Set[str] = field(default_factory=set)  # lexical keys of the summary
    member_turns: List[str] = field(default_factory=list)  # episode:turn ids
    @property
    def has_members(self) -> bool:
        return bool(self.member_turns)
_nlp = None
def _get_nlp():
    """Lazy-load the English spaCy model. Returns None on failure (fail-open)."""
    global _nlp
    if _nlp is not None:
        return _nlp or None
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("[evidence] spaCy en_core_web_sm unavailable: %s", exc)
        _nlp = False
    return _nlp or None
def extract_keys(text: str) -> Tuple[List[str], Set[str]]:
    """Return (entities, lexical keys) for one turn text.
    Deterministic, zero LLM. Mirrors index_proto.extract_keys: spaCy NER
    entities (1-4 tokens, whitelisted labels) + lemma unigrams/bigrams.
    """
    nlp = _get_nlp()
    if nlp is None:
        return [], set()
    try:
        doc = nlp(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[evidence] spaCy parse failed: %s", exc)
        return [], set()
    ents = []
    for e in doc.ents:
        if e.label_ in _ENTITY_LABELS:
            t = e.text.lower().strip()
            if 1 <= len(t.split()) <= 4:
                ents.append(t)
    toks = [t for t in doc if t.is_alpha and t.text.lower() not in _STOP and len(t.text) > 1]
    lemmas = [t.lemma_.lower() for t in toks if t.pos_ in _KEEP_POS and len(t.lemma_) > 2]
    bigrams = [f"{lemmas[i]}_{lemmas[i + 1]}" for i in range(len(lemmas) - 1)]
    keys = set(lemmas) | set(bigrams)
    return ents, keys
def extract_labeled_ents(text: str) -> Dict[str, Set[str]]:
    """One spaCy pass returning whitelisted entities grouped by label.
    Used by the event hyperedge channel to distinguish DATE from PERSON
    entities at query time (``extract_keys`` deliberately drops labels).
    Deterministic, zero LLM.
    """
    nlp = _get_nlp()
    if nlp is None:
        return {}
    try:
        doc = nlp(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[evidence] labeled entity parse failed: %s", exc)
        return {}
    out: Dict[str, Set[str]] = defaultdict(set)
    for e in doc.ents:
        t = e.text.strip().lower()
        if t and e.label_ in _ENTITY_LABELS:
            out[e.label_].add(t)
    return dict(out)
_EPISODE_RE = re.compile(r"^sample(?P<sid>[0-9_]+)_s(?P<sess>\d+)_c(?P<chunk>\d+)$")

def parse_episode_key(session_id: str) -> Optional[Tuple[str, str, int]]:
    """Parse a chunk session_id into (episode_key, session_label, chunk_idx).
    Accepts ``sample{sid}_s{sess}_c{chunk}`` encoded by import_to_ov.py.
    Returns None for any non-encoded session id (fail-open: skip indexing).
    """
    m = _EPISODE_RE.match(session_id or "")
    if not m:
        return None
    sid, sess, chunk = m.group("sid"), m.group("sess"), int(m.group("chunk"))
    return (f"sample{sid}_s{sess}", f"s{sess}", chunk)
# ---------------------------------------------------------------------------
# Evidence index
# ---------------------------------------------------------------------------
class EvidenceIndex:
    """Lexical + entity inverted index over raw session turns.
    Layout on disk (``{user_space}/.index/evidence.json``)::
        {
          "version": 2,
          "episodes": {
            "sample0_s2": {
              "session": "s2",
              "turns": [
                {"episode": "sample0_s2", "turn": 0, "speaker": "...",
                 "text": "...", "date": "...", "uri": "viking://user/.../sample0_s2_c1",
                 "keys": [...], "ents": [...], "emb": [...384 floats...]},
                ...
              ]
            }
          }
        }
    ``emb`` is optional: version-1 indexes and turns written without an
    embedder simply lack it, and retrieval degrades to the four-channel path.
    Inverted maps are rebuilt from episodes at load time (kept in-memory only,
    not persisted, to avoid drift).
    """
    VERSION = 3
    DEFAULT_EPI_WINDOW = 4
    DEFAULT_TOP_K = 15
    DEFAULT_SEM_ANCHOR_TOP = 3
    DEFAULT_CLOSURE_RADIUS = 1
    def __init__(self) -> None:
        self._episodes: Dict[str, Dict[str, Any]] = {}
        self._lex_inv: Dict[str, Set[str]] = defaultdict(set)  # key -> turn id
        self._ent_inv: Dict[str, Set[str]] = defaultdict(set)  # entity -> turn id
        self._turn_keys: Dict[str, Set[str]] = {}  # turn id -> keys
        self._turn_by_id: Dict[str, Dict[str, Any]] = {}  # turn id -> turn dict
        self._source_inv: Dict[str, Set[str]] = defaultdict(set)
        self._n_turns = 0
        self._dirty = False
        # Event hyperedge layer (probe phase, Option A: derived from cards at
        # load time). Empty by default; retrieve(use_events=True) only acts
        # when events are attached, so the default output stays byte-identical.
        self._events: List[EventNode] = []
        self._event_by_id: Dict[str, EventNode] = {}
    # -- events (probe phase: deterministic card parsing) ------------------
    def load_events_from_memories(self, memories_dir: str) -> int:
        """Parse deterministic event cards into event hyperedge nodes.
        Zero LLM. For every ``memories/events/YYYY/MM/DD/*.md`` card (skipping
        hidden files such as ``.overview.md``) read ``MEMORY_FIELDS``, derive
        the date from the directory, participants from the ``参与者：`` summary
        prefix, and lexical keys from the summary text. Member turns are the
        turns whose date matches the card date and whose keys overlap the
        event keys (approximate membership, Option A).
        """
        try:
            cards = sorted(
                glob.glob(f"{memories_dir.rstrip('/')}/events/**/*.md", recursive=True)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[evidence] event card glob failed: %s", exc)
            return 0
        loaded = 0
        for card in cards:
            if os.path.basename(card).startswith("."):
                continue
            try:
                with open(card, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                continue
            m = re.search(r"<!--\s*MEMORY_FIELDS(.*?)-->", text, re.S)
            if not m:
                continue
            try:
                fields = json.loads(m.group(1))
            except (ValueError, TypeError):
                continue
            name = str(fields.get("event_name", "") or "").strip()
            summary = str(fields.get("summary", "") or "")
            if not name:
                continue
            parts = card.replace("\\", "/").split("/")
            if len(parts) < 4:
                continue
            year, month, day = parts[-4], parts[-3], parts[-2]
            date = f"{year}-{month}-{day}"
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                continue
            ents: Set[str] = set()
            pm = re.search(r"参与者[:：]\s*([^。；;\n]+)", summary)
            if pm:
                for p in re.split(r"[、,，]", pm.group(1)):
                    p = p.strip()
                    if p:
                        ents.add(p.lower())
            _, keys = extract_keys(summary)
            node = EventNode(
                id=f"{date}::{name}",
                date=date,
                name=name,
                ents=ents,
                keys=keys,
            )
            for ep, epd in self._episodes.items():
                for turn in epd.get("turns", []):
                    if str(turn.get("date", ""))[:10] != date:
                        continue
                    if set(turn.get("keys", ())) & keys:
                        node.member_turns.append(self._turn_id(ep, turn["turn"]))
            if not node.has_members:
                continue
            self._events.append(node)
            self._event_by_id[node.id] = node
            loaded += 1
        logger.info("[evidence] loaded %d event hyperedge nodes", loaded)
        return loaded
    # -- identity helpers ------------------------------------------------
    @staticmethod
    def _turn_id(episode: str, turn: int) -> str:
        return f"{episode}:{turn}"
    @staticmethod
    def _turn_label(turn: Dict[str, Any]) -> str:
        return f"{turn['episode'].rsplit('_', 1)[-1]}:{turn['turn']}"  # s2:7
    # -- build (write-time incremental) ----------------------------------
    def append_chunk(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        user_space: str = "",
        embeddings: Optional[List[Optional[List[float]]]] = None,
    ) -> bool:
        """Incrementally index one archived chunk (called after commit).
        ``messages`` must be the archived message list, each with
        ``{"speaker": ..., "text": ..., "date": ...}``. The session_id encodes
        the episode + chunk index so multiple chunks of one episode merge into
        a single contiguous turn list.
        ``embeddings`` optionally aligns with ``messages`` (same order): the
        dense vector for each message's text, or None for texts that were not
        embedded. Stored as the turn's ``emb`` field for the ALG-A semantic
        channel. Missing embeddings degrade retrieval to four channels.
        Ordinary sessions append by stable message identity; encoded imports retain turn numbering.
        """
        parsed = parse_episode_key(session_id)
        episode_key, session_label, chunk_idx = parsed or (session_id, session_id, 1)
        episode = self._episodes.setdefault(
            episode_key,
            {"session": session_label, "turns": []},
        )
        base = ((chunk_idx - 1) * 8 if parsed else
                max((t["turn"] for t in episode["turns"]), default=-1) + 1)
        for i, m in enumerate(messages):
            text = m.get("text", "")
            speaker = m.get("speaker", "")
            date = m.get("date", "")
            if not text:
                continue
            if not parsed and m.get("id") and self._source_inv.get(m["id"]):
                continue
            turn_no = base + i if parsed else base
            tid = self._turn_id(episode_key, turn_no)
            if tid in self._turn_by_id:
                # Already indexed (duplicate append / re-import of the same
                # chunk). Skip so the episode window and inverted maps stay
                # exactly one copy per turn.
                continue
            ents, keys = extract_keys(text)
            turn = {
                "episode": episode_key,
                "turn": turn_no,
                "speaker": speaker,
                "text": text,
                "date": date,
                "uri": self._chunk_uri(user_space, session_id),
                "keys": sorted(keys),
                "ents": sorted(ents),
            }
            if m.get("id"):
                turn["source_id"] = m["id"]
            if embeddings is not None and i < len(embeddings):
                emb = embeddings[i]
                if emb:
                    turn["emb"] = [float(v) for v in emb]
            episode["turns"].append(turn)
            self._index_turn(turn)
            self._n_turns += 1
            if not parsed:
                base += 1
        self._dirty = True
        return True
    def _index_turn(self, turn: Dict[str, Any]) -> None:
        tid = self._turn_id(turn["episode"], turn["turn"])
        keys = set(turn["keys"])
        ents = set(turn["ents"])
        self._turn_keys[tid] = keys
        self._turn_by_id[tid] = turn
        if turn.get("source_id"):
            self._source_inv[turn["source_id"]].add(tid)
        for k in keys:
            self._lex_inv[k].add(tid)
        for e in ents:
            self._ent_inv[e].add(tid)
    def resolve_sources(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Resolve exact original message identities; ambiguous identities fail closed."""
        tids = []
        for source_id in source_ids:
            matches = self._source_inv.get(source_id, set())
            if len(matches) != 1:
                return []
            tids.extend(matches)
        return self.get_turns(list(dict.fromkeys(tids)))

    @staticmethod
    def _chunk_uri(user_space: str, session_id: str) -> str:
        if not user_space:
            return f"viking://user/{session_id}"
        return f"{user_space.rstrip('/')}/sessions/{session_id}"
    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.VERSION, "episodes": self._episodes}
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceIndex":
        index = cls()
        episodes = data.get("episodes", {}) if isinstance(data, dict) else {}
        for episode_key, ep in episodes.items():
            if not isinstance(ep, dict):
                continue
            # Keep the episode dict as-is (keys/ents stay JSON-safe lists);
            # _index_turn converts to sets for the inverted maps.
            index._episodes[episode_key] = ep
            turns = ep.get("turns", []) if isinstance(ep.get("turns"), list) else []
            for turn in turns:
                index._index_turn(turn)
                index._n_turns += 1
        return index
    async def save(self, viking_fs: Any, uri: str, ctx: Any) -> None:
        """Persist the index to ``uri`` (e.g. ``{user_space}/.index/evidence.json``)."""
        if not self._dirty:
            return
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=1)
        try:
            await viking_fs.write_file(uri=uri, content=payload, ctx=ctx)
            self._dirty = False
        except Exception as exc:  # pragma: no cover - fail-open
            logger.warning("[evidence] index save failed: %s", exc)
    @classmethod
    async def load(cls, viking_fs: Any, uri: str, ctx: Any) -> "EvidenceIndex":
        try:
            content = await viking_fs.read_file(uri=uri, ctx=ctx)
        except Exception:
            return cls()
        try:
            data = json.loads(content)
        except (TypeError, ValueError) as exc:
            logger.warning("[evidence] index load parse failed: %s", exc)
            return cls()
        return cls.from_dict(data)
    # -- retrieval -------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        epi_window: int = DEFAULT_EPI_WINDOW,
        propagate: bool = True,
        use_semantic: bool = False,
        q_vec: Optional[List[float]] = None,
        sem_anchor_top: int = DEFAULT_SEM_ANCHOR_TOP,
        closure_radius: int = DEFAULT_CLOSURE_RADIUS,
        use_events: bool = False,
        event_top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Rank raw turns for ``query`` and return an evidence package.
        Ported from index_proto.retrieve: lexical (df-filtered) + entity
        signals, dedup-session top-3 primary evidence, episode window
        expansion (±epi_window neighbors), co-occurrence propagation, then a
        deterministic merge. Returned items keep raw text so the caller can
        build a prompt without a second read.
        With ``use_semantic=True`` and a ``q_vec`` the ALG-A path runs instead:
        semantic cosine anchors (top-``sem_anchor_top`` turns with an ``emb``)
        join the window/co-occurrence anchor sets, and the anchors plus their
        ±``closure_radius`` neighbors are force-included in the package
        (Zero-Mem Eq.14 / EdgeMem inclusion closure). If no turn has an
        ``emb`` or ``q_vec`` is missing, this is the version-1 path unchanged.
        With ``use_events=True`` and event nodes attached (see
        ``load_events_from_memories``), event hyperedges contribute a fourth
        anchor source: events scored by date / participant / lexical-key
        overlap, member turns boosted, and date-matched event members
        force-included (EdgeMem Time-hyperedge style). Off by default: when no
        event matches, the output is byte-identical to the event-less path.
        """
        if not self._n_turns or not query.strip():
            return []
        n = self._n_turns
        q_ents, q_keys = extract_keys(query)
        idf = lambda df: math.log(n / (df + 1)) + 1.0
        # Low-discrimination key filter (mirrors index_proto):
        # unigram df>25% dropped; entities df>40% dropped (protagonist names).
        q_keys = {
            k
            for k in q_keys
            if "_" in k or len(self._lex_inv.get(k, ())) <= n * 0.25
        }
        q_ents = {e for e in q_ents if len(self._ent_inv.get(e, ())) <= n * 0.40}
        score: Dict[str, float] = defaultdict(float)
        for k in q_keys:
            df = len(self._lex_inv.get(k, ()))
            if df == 0:
                continue
            w = idf(df) * (1.5 if "_" in k else 1.0)
            for tid in self._lex_inv[k]:
                score[tid] += w
        for e in q_ents:
            df = len(self._ent_inv.get(e, ()))
            if df == 0:
                continue
            w = idf(df) * 1.2
            for tid in self._ent_inv[e]:
                score[tid] += w
        if not score:
            return []
        # Primary evidence: top-3 across distinct episodes (deterministic).
        primary = []
        seen_ep = set()
        for tid in sorted(score, key=lambda t: -score[t]):
            ep = tid.split(":")[0]
            if ep in seen_ep:
                continue
            seen_ep.add(ep)
            primary.append(tid)
            if len(primary) >= 3:
                break
        # Semantic anchors (ALG-A): top-K turns by cosine against the query
        # vector. Empty when the semantic channel is off / unavailable.
        sem_anchors: List[str] = []
        if use_semantic and q_vec is not None:
            sem_anchors = self._sem_anchors(q_vec, sem_anchor_top)
        sem_active = bool(sem_anchors)
        anchors = list(dict.fromkeys(primary + sem_anchors))
        lex_ch = sorted(score, key=lambda t: -score[t])[:top_k]
        lex_set = set(lex_ch)
        # Episode window expansion: neighbors of the anchor turns. The ALG-A
        # path applies the bonus directly (validated on the pilot harness);
        # the legacy path keeps the v1 lex_set skip for byte-identical output.
        if sem_active:
            for anchor in anchors:
                ep, idx_s = anchor.rsplit(":", 1)
                idx = int(idx_s)
                for j in range(max(0, idx - epi_window), idx + epi_window + 1):
                    oid = self._turn_id(ep, j)
                    if oid not in self._turn_by_id:
                        continue
                    d = abs(j - idx)
                    bonus = 1.0 - d / (epi_window + 1)
                    score[oid] = max(score.get(oid, 0.0), bonus)
        else:
            epi_cand: Dict[str, float] = {}
            for tid in anchors:
                ep, idx_s = tid.rsplit(":", 1)
                idx = int(idx_s)
                for j in range(max(0, idx - epi_window), idx + epi_window + 1):
                    oid = self._turn_id(ep, j)
                    if oid not in self._turn_by_id:
                        continue
                    if oid in lex_set:
                        continue
                    d = abs(j - idx)
                    bonus = 1.0 - d / (epi_window + 1)
                    epi_cand[oid] = max(epi_cand.get(oid, 0.0), bonus)
            for oid, bonus in sorted(epi_cand.items(), key=lambda kv: -kv[1]):
                score[oid] = max(score.get(oid, 0.0), bonus)
        # Co-occurrence propagation: turns sharing a discriminative key with
        # an anchor turn.
        if propagate:
            for tid in anchors:
                disc = {
                    k
                    for k in self._turn_keys.get(tid, ())
                    if "_" in k or len(self._lex_inv.get(k, ())) <= n * 0.25
                }
                if not disc:
                    continue
                for ep, epd in self._episodes.items():
                    for turn in epd.get("turns", []):
                        oid = self._turn_id(ep, turn["turn"])
                        if oid in score:
                            continue
                        shared = set(turn.get("keys", ())) & disc
                        if shared:
                            score[oid] = max(score.get(oid, 0.0), 0.4 * len(shared))
        # Event hyperedge channel (zero LLM, off by default): events scored by
        # date / participant / lexical-key overlap; top-K event members get a
        # boost, date-matched event members are force-included below.
        event_anchors: List[str] = []
        if use_events and self._events:
            labeled = extract_labeled_ents(query)
            q_dates = set()
            for dt in labeled.get("DATE", ()):
                norm = _normalize_date(dt)
                if norm:
                    q_dates.add(norm)
            q_persons = labeled.get("PERSON", set())
            ev_scores: Dict[str, float] = {}
            for ev in self._events:
                s = 0.0
                date_hit = ev.date in q_dates
                if date_hit:
                    s += 3.0
                shared = q_keys & ev.keys
                # Protagonist names match dozens of events; a pure participant
                # hit is not enough on its own (requires a key overlap too).
                if ev.ents & q_persons and shared:
                    s += 1.5
                if shared:
                    s += sum(idf(len(self._lex_inv.get(k, ()))) for k in shared) * 1.2
                if s > 0:
                    ev_scores[ev.id] = s
            for ev_id in sorted(ev_scores, key=lambda e: -ev_scores[e])[:event_top_k]:
                ev = self._event_by_id[ev_id]
                s_e = ev_scores[ev_id]
                boost = 0.5 * s_e
                cap = os.environ.get("OV_EVENT_CAP")
                if cap:
                    boost = min(boost, float(cap))
                date_hit = ev.date in q_dates
                for tid in ev.member_turns:
                    if tid not in self._turn_by_id:
                        continue
                    if date_hit:
                        # Time-hyperedge: date-matched members join the package
                        # regardless of prior signal (force-included below).
                        score[tid] = max(score.get(tid, 0.0), boost)
                        event_anchors.append(tid)
                    elif tid in score:
                        # Non-date matches only boost turns that already carry
                        # a signal; never inject cold member turns.
                        score[tid] = max(score[tid], boost)
        ranked_ids = sorted(score, key=lambda t: -score[t])[:top_k]
        # ALG-A closure: force-include semantic anchors + their ±radius
        # neighbors, displacing the lowest-ranked non-closure members.
        if sem_active and closure_radius >= 0:
            closure = set()
            for anchor in sem_anchors:
                ep, idx_s = anchor.rsplit(":", 1)
                idx = int(idx_s)
                for j in range(max(0, idx - closure_radius), idx + closure_radius + 1):
                    oid = self._turn_id(ep, j)
                    if oid in self._turn_by_id:
                        closure.add(oid)
            for tid in closure:
                if tid not in ranked_ids:
                    for cand in reversed(ranked_ids):
                        if cand not in closure:
                            ranked_ids.remove(cand)
                            break
                    ranked_ids.append(tid)
            ranked_ids = sorted(ranked_ids, key=lambda t: -score[t])
        # Event hyperedge closure: date-matched event members are
        # force-included (inclusion, not scoring; mirrors the semantic closure).
        if event_anchors:
            eclosure = set(event_anchors)
            for tid in eclosure:
                if tid not in ranked_ids:
                    for cand in reversed(ranked_ids):
                        if cand not in eclosure:
                            ranked_ids.remove(cand)
                            break
                    ranked_ids.append(tid)
            ranked_ids = sorted(ranked_ids, key=lambda t: -score[t])
        return [self._turn_by_id[tid] for tid in ranked_ids]
    def get_turns(self, turn_ids: List[str]) -> List[Dict[str, Any]]:
        """Read evidence from this index only, without exposing stored vectors."""
        return [
            {k: v for k, v in self._turn_by_id[tid].items() if k != "emb"}
            for tid in turn_ids if tid in self._turn_by_id
        ]

    def retrieve_candidates(
        self, query: str, *, limit: int, q_vec: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Fuse independent lexical and semantic ranks before context expansion.

        Used by the opt-in coverage strategy. Unlike legacy retrieval, a
        lexical miss must not suppress an otherwise valid semantic channel.
        """
        if not query.strip() or not self._n_turns:
            return []
        limit = max(1, min(limit, 50))
        ents, keys = extract_keys(query)
        scores: Dict[str, float] = defaultdict(float)
        for key in keys:
            ids = self._lex_inv.get(key, ())
            if not ids or ("_" not in key and len(ids) > max(1, self._n_turns * .25)):
                continue
            weight = math.log(self._n_turns / (len(ids) + 1)) + 1
            for tid in ids:
                scores[tid] += weight * (1.5 if "_" in key else 1)
        lexical_ids = set(scores)
        for ent in ents:
            ids = self._ent_inv.get(ent, ())
            if not ids or len(ids) > max(1, self._n_turns * .4):
                continue
            for tid in ids:
                # When a relation/topic matched, entity presence may refine
                # those hits but must not turn unrelated mentions into a
                # second ranked channel that overwhelms specific evidence.
                if lexical_ids and tid not in lexical_ids:
                    continue
                scores[tid] += math.log(self._n_turns / (len(ids) + 1)) + 1
        lexical = sorted(scores, key=lambda tid: (-scores[tid], tid))[:limit]
        semantic = self._sem_anchors(q_vec, limit) if q_vec is not None else []
        ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        for channel, ids in (("lexical_rank", lexical), ("semantic_rank", semantic)):
            for rank, tid in enumerate(ids, 1):
                ranks[tid][channel] = rank
        result = []
        for tid, channels in ranks.items():
            row = self.get_turns([tid])[0]
            row.update(turn_id=tid, score=sum(1 / (60 + r) for r in channels.values()), **channels)
            result.append(row)
        return sorted(result, key=lambda r: (-r["score"], r["turn_id"]))[:limit]

    def _sem_anchors(
        self,
        q_vec: List[float],
        top_k: int,
    ) -> List[str]:
        """Top-``top_k`` turns by cosine(q_vec, emb), over embedded turns only.
        Deterministic; numpy imported lazily so a missing dependency degrades
        to the four-channel path instead of raising.
        """
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("[evidence] numpy unavailable for semantic channel: %s", exc)
            return []
        try:
            q = np.asarray(q_vec, dtype=np.float32)
            qn = float(np.linalg.norm(q))
            if qn == 0:
                return []
            sims = []
            for tid, turn in self._turn_by_id.items():
                emb = turn.get("emb")
                if not emb:
                    continue
                v = np.asarray(emb, dtype=np.float32)
                c = float(q @ v / (qn * float(np.linalg.norm(v)) + 1e-9))
                sims.append((c, tid))
            sims.sort(key=lambda kv: -kv[0])
            return [tid for _, tid in sims[:top_k]]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[evidence] semantic anchor computation failed: %s", exc)
            return []

# ---------------------------------------------------------------------------
# Request-authorized loader
# ---------------------------------------------------------------------------
async def load_index(viking_fs: Any, uri: str, ctx: Any) -> "EvidenceIndex":
    """Read with current authorization and current persisted state.

    Mutable URI-only caches can leak across accounts or bypass revocation.
    The session commit hook serializes read->append->save; loading fresh also
    prevents stale snapshots from overwriting another caller's appended turns.
    """
    return await EvidenceIndex.load(viking_fs, uri, ctx)
