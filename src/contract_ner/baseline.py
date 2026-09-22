"""Regex-only baseline for the English/CUAD track.

Same interface as :class:`contract_ner.predict.Extractor` (``lang``,
``entity_types``, ``predict`` / ``predict_with_confidence``) so it can be
scored by ``evaluate_documents`` on the same split with the same metrics.
Rules follow CUAD's annotation conventions as far as a pattern can: parties
are the legal names and quoted defined terms in the preamble, the agreement /
effective dates are the dated-as-of expressions there, and expiration and
governing law are whole sentences. It knows nothing about the label space of
the model beyond these five types, and gives every span confidence 1.0.
"""
from __future__ import annotations

import re

from .schema import Span

ENTITY_TYPES: tuple[str, ...] = ("PARTY", "AGREEMENT_DATE", "START_DATE", "END_DATE", "JURISDICTION")

_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_DAY = r"\d{1,2}\s*(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"
# "April 18, 2018" · "18 April 2018" · "5th day of December, 2007" · "December 2007" · "04/18/2018"
_DATE_RE = re.compile(
    rf"(?:{_DAY}\s+(?:day\s+of\s+)?{_MONTH},?\s*{_YEAR}"
    rf"|{_MONTH}\s+{_DAY},?\s*{_YEAR}"
    rf"|{_MONTH},?\s+{_YEAR}"
    rf"|\d{{1,2}}/\d{{1,2}}/(?:\d{{4}}|\d{{2}}))\b",
    re.IGNORECASE,
)
_AGREEMENT_CUE_RE = re.compile(
    r"\b(?:dated|made|entered\s+into|executed|signed|agreed)\b(?:\s+and\s+entered\s+into)?(?:\s+(?:as\s+of|on|this|the|effective(?:\s+as)?(?:\s+of)?))?\s*[:,]?\s*",
    re.IGNORECASE,
)
_EFFECTIVE_CUE_RE = re.compile(r"\beffective\s+(?:as\s+of|on|from|date\s+of|date\s+is)?\s*[:,]?\s*", re.IGNORECASE)
_PREAMBLE_END_RE = re.compile(r"\b(?:WHEREAS|RECITALS|WITNESSETH|BACKGROUND)\b|\n\s*(?:1|ARTICLE\s+I|Section\s+1)[.\s]", re.IGNORECASE)

_SUFFIX = (r"(?i:Inc\.?|Incorporated|Ltd\.?|Limited|LLC|L\.L\.C\.?|Corp\.?|Corporation|Company|Co\.|plc|GmbH|AG|"
           r"S\.A\.?|N\.V\.?|B\.V\.?|L\.P\.?|LP|LLP|Pty\.?|Pte\.?|N\.A\.?|FSB|Trust|Bank|Partnership|Association|Group)")
# a run of capitalised words (with &, digits, initials) ending in a legal-form suffix; "a Delaware corporation"
# after the article is the entity-form descriptor, not a name
_LEGAL_NAME_RE = re.compile(
    rf"(?<!\ba )(?<!\ban )\b((?:[A-Z][\w&'\-.]*|\d[\w\-]*)(?:[ ,]+(?:[A-Z][\w&'\-.]*|\d[\w\-]*|&|and|of|de|the)){{0,7}}?[ ,]*{_SUFFIX})(?=[\s,.;:)(\"]|$)"
)
# ("Company") · (the "Licensee") · (hereinafter "Supplier") — the quoted term with its quotes, as CUAD annotates it
_DEFINED_TERM_RE = re.compile(r"\(\s*(?:the\s+|hereinafter\s+|hereinafter\s+referred\s+to\s+as\s+)?([\"“][A-Z][^\"”]{0,40}[\"”])")
# defined terms that name the contract, a date or a deliverable rather than a party
_NOT_A_PARTY_RE = re.compile(
    r"\b(?:Agreement|Contract|Date|Term|Services?|Site|Products?|Business|Notes|Software|Schedule|Plan|Program|Territory|Work|Order|Fee|Period)\b",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(r"\b(?:by\s+and\s+)?(?:between|among|amongst)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^.;\n][^.;]*[.;]?")
# the sentence must be about the agreement's own term, not a licence, a warranty or a record-keeping duty
_END_CUE_RE = re.compile(
    r"^(?:the\s+)?(?:initial\s+)?terms?\s+(?:of\s+this|of\s+the|will|shall)\b|^this\s+\w*\s*(?:agreement|contract)\s+(?:shall|will|is|as amended)\b"
    r"|\bterm\s+of\s+this\s+(?:agreement|contract)\b|\b(?:agreement|contract)\s+(?:shall|will)\s+(?:terminate|expire|continue|remain|be\s+in\s+effect|commence)\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty)\s*(?:\(\d+\)\s*)?(?:years?|months?|days?)\b",
    re.IGNORECASE,
)
_LAW_RE = re.compile(r"\bgoverned\s+by\b|\bconstrued\s+(?:and\s+enforced\s+)?in\s+accordance\s+with\b|\bgoverning\s+law\b", re.IGNORECASE)
_LAW_OBJECT_RE = re.compile(r"\blaws?\s+of\b|\bEnglish\s+law\b|\blaws?\s+in\s+force\b", re.IGNORECASE)


def _preamble(text: str) -> str:
    m = _PREAMBLE_END_RE.search(text, 200)
    return text[: m.start()] if m else text[:4000]


def _sentences(text: str):
    for m in _SENTENCE_RE.finditer(text):
        s = m.group(0)
        if s.strip():
            yield m.start(), s


class RegexExtractor:
    """Pattern-only stand-in for the neural :class:`Extractor`."""

    name = "regex-baseline"
    lang = "en"
    entity_types = ENTITY_TYPES

    def predict_with_confidence(self, text: str) -> list[tuple[Span, float]]:
        spans = self.predict(text)
        return [(sp, 1.0) for sp in spans]

    def predict(self, text: str) -> list[Span]:
        out: list[Span] = []
        pre = _preamble(text)
        out += self._parties(pre)
        out += self._preamble_dates(pre)
        out += self._clauses(text)
        out.sort(key=lambda s: (s.start, s.end))
        return _dedupe(out)

    # ------------------------------------------------------------------ rules
    def _parties(self, pre: str) -> list[Span]:
        # CUAD marks the *defining* mention: names from the "between ... and ..." sentence onwards, once each
        cue = _BETWEEN_RE.search(pre)
        lo = cue.start() if cue else 0
        spans: list[Span] = []
        seen: set[str] = set()
        for m in _LEGAL_NAME_RE.finditer(pre, lo):
            key = _WS_RE.sub(" ", m.group(1)).casefold().rstrip(".")
            if key in seen:
                continue
            seen.add(key)
            spans.append(Span(m.start(1), m.end(1), "PARTY", m.group(1)))
        for m in _DEFINED_TERM_RE.finditer(pre, lo):
            if not _NOT_A_PARTY_RE.search(m.group(1)):
                spans.append(Span(m.start(1), m.end(1), "PARTY", m.group(1)))
        return spans

    def _preamble_dates(self, pre: str) -> list[Span]:
        spans: list[Span] = []
        dates = list(_DATE_RE.finditer(pre))
        if not dates:
            return spans

        def nearest_after(cue: re.Match, window: int = 40) -> re.Match | None:
            return next((d for d in dates if cue.end() <= d.start() <= cue.end() + window), None)

        agreement = next((d for c in _AGREEMENT_CUE_RE.finditer(pre) if (d := nearest_after(c))), None)
        if agreement is None:
            agreement = dates[0]
        spans.append(Span(agreement.start(), agreement.end(), "AGREEMENT_DATE", agreement.group(0)))
        effective = next((d for c in _EFFECTIVE_CUE_RE.finditer(pre) if (d := nearest_after(c))), None)
        if effective is not None and effective.start() != agreement.start():
            spans.append(Span(effective.start(), effective.end(), "START_DATE", effective.group(0)))
        return spans

    def _clauses(self, text: str) -> list[Span]:
        spans: list[Span] = []
        got_end = got_law = False
        for start, s in _sentences(text):
            body = s.strip()
            if not body or len(body) > 600:
                continue
            off = start + (len(s) - len(s.lstrip()))
            if not got_end and _END_CUE_RE.search(body) and (_DATE_RE.search(body) or _DURATION_RE.search(body)):
                spans.append(Span(off, off + len(body), "END_DATE", body))
                got_end = True
            elif not got_law and _LAW_RE.search(body) and _LAW_OBJECT_RE.search(body):
                spans.append(Span(off, off + len(body), "JURISDICTION", body))
                got_law = True
            if got_end and got_law:
                break
        return spans


def _dedupe(spans: list[Span]) -> list[Span]:
    """Drop spans nested inside an earlier span of the same type (``Inc.`` inside ``Acme, Inc.``)."""
    kept: list[Span] = []
    for sp in spans:
        if any(k.label == sp.label and k.start <= sp.start and sp.end <= k.end for k in kept):
            continue
        kept.append(sp)
    return kept
