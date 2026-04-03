from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException, status


_TEXT_TRIM_RE = re.compile(r"\s+")


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


def _simple_summary(text: str, max_sentences: int = 3, max_chars: int = 900) -> str:
    cleaned = _TEXT_TRIM_RE.sub(" ", text).strip()
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    picked = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        picked.append(s)
        if len(picked) >= max_sentences:
            break

    summary = " ".join(picked).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
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
    sentiment = _sentiment_label(scores.get("compound", 0.0))

    summary = _simple_summary(text)

    if not summary:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to generate summary",
        )

    return {
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment,
    }
