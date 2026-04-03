from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException, status


_TEXT_TRIM_RE = re.compile(r"\s+")

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}\b")
_ID_LIKE_RE = re.compile(r"\b\d{8,16}\b")

_DOC_TYPE_RULES: list[tuple[str, set[str]]] = [
    ("invoice", {"invoice", "amount", "gst", "total"}),
    ("resume", {"experience", "skills", "education"}),
    ("legal", {"agreement", "terms", "party"}),
]


def _lazy_spacy_model():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install it during build: python -m spacy download en_core_web_sm"
        )


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = _lazy_spacy_model()
    return _NLP


def _get_sentiment_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


_SENTIMENT = None


def _sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _detect_document_type(text: str) -> str:
    lowered = (text or "").lower()
    for doc_type, keywords in _DOC_TYPE_RULES:
        if any(k in lowered for k in keywords):
            return doc_type
    return "general"


def _detect_sensitive_data(text: str) -> dict[str, Any]:
    lowered = text or ""
    types: list[str] = []

    if _EMAIL_RE.search(lowered):
        types.append("email")
    if _PHONE_RE.search(lowered):
        types.append("phone")
    if _ID_LIKE_RE.search(lowered):
        types.append("id_number")

    return {
        "has_sensitive": len(types) > 0,
        "types": types,
    }


def _entities_confidence(entities: dict[str, list[str]]) -> float:
    # Heuristic: more entities => higher confidence, capped.
    total = sum(len(v) for v in entities.values())
    return max(0.0, min(1.0, total / 20.0))


def _sentiment_confidence(compound: float) -> float:
    # Heuristic: compound magnitude is confidence-like in [0..1].
    return max(0.0, min(1.0, abs(compound)))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        k = x.strip()
        if not k:
            continue
        key = k.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(k)
    return out


def _simple_summary(text: str, min_words: int = 100, max_words: int = 200) -> str:
    cleaned = _TEXT_TRIM_RE.sub(" ", text).strip()
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    picked: list[str] = []
    word_count = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        s_words = len(s.split())
        if not picked and s_words > max_words:
            # Very long single sentence: hard-trim to max_words.
            return " ".join(s.split()[:max_words]).strip()

        if word_count + s_words > max_words:
            break

        picked.append(s)
        word_count += s_words

        # Stop as soon as we reach the target minimum, but never exceed max_words.
        if word_count >= min_words:
            break

    if not picked:
        # Fallback: first max_words words.
        return " ".join(cleaned.split()[:max_words]).strip()

    summary = " ".join(picked).strip()
    # Final safety cap.
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]).strip()

    return summary


def analyze_text(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty text")

    nlp = _get_nlp()
    doc = nlp(text)

    entities: dict[str, list[str]] = {
        "PERSON": [],
        "ORG": [],
        "DATE": [],
        "MONEY": [],
        "GPE": [],
    }

    for ent in doc.ents:
        label = ent.label_
        if label in entities:
            entities[label].append(ent.text)

    for k in list(entities.keys()):
        entities[k] = _dedupe_preserve_order(entities[k])

    global _SENTIMENT
    if _SENTIMENT is None:
        _SENTIMENT = _get_sentiment_analyzer()

    scores = _SENTIMENT.polarity_scores(text)
    compound = float(scores.get("compound", 0.0))
    sentiment = _sentiment_label(compound)

    summary = _simple_summary(text)

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to generate summary",
        )

    doc_type = _detect_document_type(text)
    sensitive_data = _detect_sensitive_data(text)
    confidence = {
        "entities": _entities_confidence(entities),
        "sentiment": _sentiment_confidence(compound),
    }

    return {
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment,
        "type": doc_type,
        "sensitive_data": sensitive_data,
        "confidence": confidence,
    }
