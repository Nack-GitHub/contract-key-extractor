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
# an end-of-term clause must say something about ending or duration; digits alone ("Act of 1976") do not qualify
_END_WORD_RE = re.compile(
    r"year|month|day|week|term|expir|terminat|anniversar|until|through|period|duration|renew|\bend|perpetu|for (so|as) long|ceas",
    re.IGNORECASE,
)
# a party span that is nothing but a legal-form suffix is a fragment of a name, not a name
_SUFFIX_ONLY_RE = re.compile(
    r"^(inc|incorporated|ltd|limited|llc|l\.l\.c|corp|corporation|plc|gmbh|ag|s\.a|n\.v|b\.v|l\.p|lp|llp|gp|co|pty|sa|srl|s\.r\.l)\.?$",
    re.IGNORECASE,
)
_QUOTE_EDGE_RE = re.compile(r"^[\"'“”‘’(\[\s]+|[\"'“”‘’)\]\s.,]+$")
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
        if span.label == "END_DATE":
            return bool(_END_WORD_RE.search(t))
        return bool(_DIGIT_RE.search(t)) or bool(_TERM_WORD_RE.search(t))
    if span.label in ("CONTRACT_VALUE", "PENALTY_RATE"):
        return bool(_DIGIT_RE.search(t))
    if span.label == "PAYMENT_DUE":
        return bool(_DIGIT_RE.search(t)) or bool(_PAYMENT_RE.search(t))
    if span.label == "JURISDICTION":
        return bool(_JURISDICTION_RE.search(t))
    if span.label == "PARTY":
        if len(t) < 2 or re.search(r"[^\W\d_]", t) is None:  # at least one letter
            return False
        core = _QUOTE_EDGE_RE.sub("", t)
        if _SUFFIX_ONLY_RE.match(core):
            return False
        # an unquoted, all-lowercase single word ("together", "anonyme") is never a legal name;
        # quoted lowercase defined terms ("you", "we") are kept
        return not (core == t and len(core.split()) == 1 and core.islower())
    return True


# ---------------------------------------------------------------- surface forms
# The buckets a label's spans fall into. Chosen from what CUAD actually contains
# (see ``contract-ner profile``): every bucket here holds at least ~1 % of the
# training spans of its label, and the ones a rule-based stage can never reach
# ("clause-no-date", "parties-boilerplate") are named so they can be counted out.
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
          r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_F_MONTH_D_Y = re.compile(rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*(?:19|20)\d{{2}}", re.IGNORECASE)
_F_D_MONTH_Y = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_MONTH}\.?,?\s*(?:19|20)\d{{2}}", re.IGNORECASE)
_F_MONTH_Y = re.compile(rf"\b{_MONTH},?\s+(?:19|20)\d{{2}}", re.IGNORECASE)
_F_NUMERIC = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")
_F_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_F_BLANK = re.compile(r"_{2,}")
_F_DURATION = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|sixty|ninety)\b"
    r"[^.;]{0,20}?\b(?:year|month|day|week|anniversar)", re.IGNORECASE)
_F_OPEN_ENDED = re.compile(r"until\b|unless\s+terminated|terminat|perpetu|indefinit|so\s+long\s+as|renew", re.IGNORECASE)
_F_GOVERNED = re.compile(r"govern|construed", re.IGNORECASE)
_F_LAWS_OF = re.compile(r"laws?\s+of\b|law\s+of\s+the\b", re.IGNORECASE)
_F_FORUM = re.compile(r"court|arbitrat|tribunal|venue|jurisdiction", re.IGNORECASE)
_F_SUFFIX = re.compile(
    r"\b(?:Inc|Incorporated|Ltd|Limited|LLC|L\.L\.C|Corp|Corporation|Company|Co|plc|GmbH|AG|S\.A|N\.V|B\.V|"
    r"L\.P|LP|LLP|Pty|Pte|Trust|Bank|Partnership|Association|Group|Holdings|Systems|Technologies)\b\.?",
    re.IGNORECASE)
_F_BOILERPLATE = re.compile(r"referred\s+to|collectively|the\s+[\"\u201c]?Part(?:y|ies)", re.IGNORECASE)
_F_QUOTED = re.compile(r"^[(\s]*[\"\u201c\u2018\'].{1,40}[\"\u201d\u2019\'][)\s]*$")
_F_PERSON = re.compile(r"^(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+$")


def _date_form(t: str) -> str:
    if _F_MONTH_D_Y.search(t):
        return "month-d-y"          # "June 21, 1999" — CUAD's house style
    if _F_D_MONTH_Y.search(t):
        return "d-month-y"          # "23rd day of September, 2016"
    if _F_NUMERIC.search(t):
        return "numeric"            # "03/01/05", "3-31-16"
    if _F_MONTH_Y.search(t):
        return "month-y"            # "June, 2004"
    if _F_BLANK.search(t):
        return "placeholder-blank"  # "____ day of ______, 2004" — unsigned template
    if _F_YEAR.search(t):
        return "year-only"
    return "clause-no-date"         # "as of the Effective Date" — no value to normalise


def span_form(span: Span) -> str:
    """Coarse surface form of ``span``; see :data:`FORMS` for the buckets per label."""
    t = " ".join(span.text.split())
    if span.label in ("AGREEMENT_DATE", "START_DATE"):
        return _date_form(t)
    if span.label == "END_DATE":
        if _F_MONTH_D_Y.search(t) or _F_D_MONTH_Y.search(t) or _F_NUMERIC.search(t):
            return "explicit-date"
        if _F_DURATION.search(t):
            return "duration"
        if _F_OPEN_ENDED.search(t):
            return "open-ended"
        return "other"
    if span.label == "JURISDICTION":
        if _F_GOVERNED.search(t):
            return "governed-by-clause"
        if _F_FORUM.search(t):
            return "forum-court"
        if _F_LAWS_OF.search(t):
            return "laws-of-phrase"
        return "place-only" if len(t.split()) <= 4 else "other"
    if span.label == "PARTY":
        if _F_BOILERPLATE.search(t):
            return "parties-boilerplate"
        if _F_QUOTED.match(t):
            return "quoted-alias"
        if _F_SUFFIX.search(t):
            return "legal-name-caps" if t.upper() == t else "legal-name"
        if t.upper() == t and len(t) > 3:
            return "caps-no-suffix"
        if _F_PERSON.match(t):
            return "person"
        return "other"
    return "unclassified"


#: buckets ``span_form`` can return per label, in reporting order
FORMS: dict[str, tuple[str, ...]] = {
    "PARTY": ("legal-name", "legal-name-caps", "caps-no-suffix", "quoted-alias", "parties-boilerplate", "person", "other"),
    "AGREEMENT_DATE": ("month-d-y", "d-month-y", "numeric", "month-y", "year-only", "placeholder-blank", "clause-no-date"),
    "START_DATE": ("month-d-y", "d-month-y", "numeric", "month-y", "year-only", "placeholder-blank", "clause-no-date"),
    "END_DATE": ("explicit-date", "duration", "open-ended", "other"),
    "JURISDICTION": ("governed-by-clause", "forum-court", "laws-of-phrase", "place-only", "other"),
}


RULES = LangRules(lang="en", segment_words=segment_words, validate_span=validate_span, romanize=lambda s: s,
                  span_form=span_form, span_forms=FORMS)
