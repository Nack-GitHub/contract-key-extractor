"""English rules: regex word segmentation and keyword checks for CUAD-style spans."""
from __future__ import annotations

import re

from ..schema import Span
from . import LangRules

# words = alphanumeric runs (with internal hyphens/apostrophes/dots: ``X-Y``, ``Inc.``) or single punctuation
_WORD_RE = re.compile(r"\w+(?:[-'’.]\w+)*|[^\w\s]", re.UNICODE)

_DIGIT_RE = re.compile(r"\d")
# a clause-level term span (CUAD ``Expiration Date``) may describe the term without a literal date
_TERM_WORD_RE = re.compile(r"year|month|day|week|term|expir|terminat|anniversar|effective|commence", re.IGNORECASE)
_JURISDICTION_RE = re.compile(r"court|law|arbitra|tribunal|jurisdiction", re.IGNORECASE)
_PAYMENT_RE = re.compile(r"month|quarter|annual|invoice", re.IGNORECASE)


def segment_words(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def validate_span(span: Span) -> bool:
    """Keyword-level plausibility of ``span.text`` for its type (after the shared generic checks)."""
    t = span.text.strip()
    if span.label in ("AGREEMENT_DATE", "START_DATE", "END_DATE"):
        if len(t.split()) <= 4:  # a bare date expression must parse
            from ..postprocess import normalize_date
            return bool(_DIGIT_RE.search(t)) and normalize_date(t) is not None
        return bool(_DIGIT_RE.search(t)) or bool(_TERM_WORD_RE.search(t))
    if span.label in ("CONTRACT_VALUE", "PENALTY_RATE"):
        return bool(_DIGIT_RE.search(t))
    if span.label == "PAYMENT_DUE":
        return bool(_DIGIT_RE.search(t)) or bool(_PAYMENT_RE.search(t))
    if span.label == "JURISDICTION":
        return bool(_JURISDICTION_RE.search(t))
    if span.label == "PARTY":
        return len(t) >= 2 and re.search(r"[^\W\d_]", t) is not None  # at least one letter
    return True


RULES = LangRules(lang="en", segment_words=segment_words, validate_span=validate_span, romanize=lambda s: s)
