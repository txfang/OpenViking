# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared language routing and model cache for deterministic NLP."""

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)
_CJK_RE = re.compile(r"[\u4e00-\u9fa5]")


def detect_language(text: str) -> str:
    """Use the extractor's CJK-ratio rule for both writes and queries."""
    if not text.strip():
        return "en"
    return "zh" if len(_CJK_RE.findall(text)) / len(text) >= 0.1 else "en"


@lru_cache(maxsize=4)
def load_spacy_model(model_name: str):
    """Load once per model name, including failures until process restart."""
    try:
        import spacy

        nlp = spacy.load(model_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("spaCy model %s unavailable; using NLP fallback: %s", model_name, exc)
        return None
    logger.info("Loaded spaCy model %s (%s)", model_name, nlp.meta.get("version"))
    return nlp
