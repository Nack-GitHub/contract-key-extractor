"""Slot-level evaluation: is each cell of the metadata table right?

The span metrics in :mod:`evaluate` score *mentions*; CUAD annotates one
defining mention per contract, so a second, equally correct governing-law
clause or a repeated party name counts against the model there. What a user
sees is the table produced by :func:`postprocess.assemble` — one value per
slot — and that is what this module scores. Gold slot values are derived from
the gold spans with the same normalisers the pipeline uses, so a clause is
"right" when it normalises to the same value, not when it is the same string.

Slot definitions (``SLOTS``):

* ``Parties`` — the set of legal names (defined-term aliases such as
  ``"Company"`` and CUAD's "referred to as the Parties" boilerplate are
  dropped on both sides). Scored as a set per document; names match when
  their normalised forms are equal or one contains the other.
* ``Agreement_Date`` — ISO date of the AGREEMENT_DATE span(s).
* ``Start_Date`` — ISO date of a dated START_DATE span; when the gold
  effective-date clause carries no date the slot falls back to the agreement
  date, mirroring the ``inferred_from`` rule in ``assemble``.
* ``End_Date`` — ISO date when the clause names one, otherwise a duration
  key such as ``3 year`` (so "three (3) years from the Effective Date" and
  "a period of 3 years" agree), otherwise the raw text.
* ``Jurisdiction`` — the place named by the governing-law clause
  (``normalize_jurisdiction(...)["english"]``); CUAD's category is
  "Governing Law", so the prediction is taken from ``Governing_Law`` and
  falls back to ``Jurisdiction``.

A slot with no gold value and no prediction is a true negative and is not
counted. A wrong value counts as both a false positive and a false negative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .postprocess import (
    _BOILERPLATE_RE,
    _party_key,
    group_parties,
    normalize_date,
    normalize_jurisdiction,
)
from .schema import Document, Span

SLOTS = ("Parties", "Agreement_Date", "Start_Date", "End_Date", "Jurisdiction")

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "eighteen": 18, "twenty": 20, "twenty-four": 24, "thirty": 30, "thirty-six": 36,
    "sixty": 60, "ninety": 90,
}
_DURATION_RE = re.compile(
    r"(?:(?P<n>\d+)|\b(?P<w>" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")\b)"
    r"(?:\s*\(\s*\d+\s*\))?[\s-]*(?P<unit>years?|months?|weeks?|days?)\b",
    re.IGNORECASE,
)
_PLACE_NOISE_RE = re.compile(r"\b(the|state|commonwealth|province|republic|kingdom|district|federal|of)\b", re.IGNORECASE)
_NAME_NOISE_RE = re.compile(r"[\"'“”‘’.,()\[\]]")


# punctuation a clause may or may not carry at its edges, and which never changes the value the
# clause states: "…shall continue indefinitely." and "…shall continue indefinitely" are one value
_EDGE_PUNCTUATION_RE = re.compile(r"^[\s\"'“”‘’(\[.,;:]+|[\s\"'“”‘’)\].,;:]+$")


def value_key(text: str) -> str:
    """Fallback key when a slot value cannot be normalised: collapsed, case-folded text.

    Clause-level slots (an end-of-term clause, a governing-law clause) fall back to
    their own text when nothing normalises out of them, so the comparison has to
    ignore what a span boundary can differ by without changing the value: the
    surrounding quotes and the full stop at the end of the sentence.
    """
    return _EDGE_PUNCTUATION_RE.sub("", _party_key(text))


def name_key(text: str) -> str:
    return re.sub(r"\s+", " ", _NAME_NOISE_RE.sub("", value_key(text))).strip()


def names_match(a: str, b: str) -> bool:
    """Normalised legal names are the same party when equal or one contains the other."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and short in long_


def party_names(spans: list[Span]) -> list[str]:
    """Distinct legal names among PARTY spans: quoted aliases and boilerplate are dropped."""
    groups = group_parties([(sp, 1.0) for sp in spans if sp.label == "PARTY"])
    names = []
    for g in groups:
        if g.quoted or _BOILERPLATE_RE.search(g.surface):
            continue
        k = name_key(g.surface)
        if k and not any(names_match(k, n) for n in names):
            names.append(k)
    return names


def date_key(text: str, lang: str = "en") -> str:
    return normalize_date(text, lang) or value_key(text)


def end_date_key(text: str, lang: str = "en") -> str:
    iso = normalize_date(text, lang)
    if iso:
        return iso
    m = _DURATION_RE.search(text)
    if m:
        n = int(m.group("n")) if m.group("n") else _NUM_WORDS[m.group("w").lower()]
        return f"{n} {m.group('unit').lower().rstrip('s')}"
    return value_key(text)


def jurisdiction_key(text: str) -> str:
    place = normalize_jurisdiction(text)["english"]
    if not place:
        return value_key(text)
    place = re.sub(r"^courts of |\s+law$", "", place)
    return re.sub(r"\s+", " ", _PLACE_NOISE_RE.sub("", place)).strip().casefold()


def gold_slots(doc: Document) -> dict[str, set[str] | list[str]]:
    """Slot values implied by the gold spans (a set: CUAD occasionally tags two dates)."""
    ents = doc.entities
    by = lambda label: [e for e in ents if e.label == label]  # noqa: E731
    lang = doc.lang
    agreement = {date_key(e.text, lang) for e in by("AGREEMENT_DATE")}
    start = {normalize_date(e.text, lang) for e in by("START_DATE") if normalize_date(e.text, lang)}
    return {
        "Parties": party_names(ents),
        "Agreement_Date": agreement,
        "Start_Date": start or agreement,
        "End_Date": {end_date_key(e.text, lang) for e in by("END_DATE")},
        "Jurisdiction": {jurisdiction_key(e.text) for e in by("JURISDICTION")},
    }


def pred_slots(rec: dict, spans: list[Span]) -> dict[str, str | list[str] | None]:
    """Slot values of an assembled record (``spans`` are the validated PARTY spans behind it)."""
    def text(k):
        v = rec.get(k)
        return v.get("text") if v else None

    lang = rec.get("lang", "en")
    j = text("Governing_Law") or text("Jurisdiction")
    return {
        "Parties": party_names(spans),
        "Agreement_Date": date_key(text("Agreement_Date"), lang) if text("Agreement_Date") else None,
        "Start_Date": date_key(text("Start_Date"), lang) if text("Start_Date") else None,
        "End_Date": end_date_key(text("End_Date"), lang) if text("End_Date") else None,
        "Jurisdiction": jurisdiction_key(j) if j else None,
    }


@dataclass
class SlotScorer:
    tp: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLOTS})
    fp: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLOTS})
    fn: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLOTS})
    n_gold: dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLOTS})  # docs with a gold value
    errors: list[dict] = field(default_factory=list)

    def add(self, doc_id: str, gold: dict, pred: dict, max_errors: int = 200) -> None:
        # parties: set match with fuzzy name equality
        g_names, p_names = list(gold["Parties"]), list(pred["Parties"])
        if g_names:
            self.n_gold["Parties"] += 1
        unmatched = list(g_names)
        for p in p_names:
            hit = next((g for g in unmatched if names_match(p, g)), None)
            if hit is None:
                self.fp["Parties"] += 1
            else:
                unmatched.remove(hit)
                self.tp["Parties"] += 1
        self.fn["Parties"] += len(unmatched)
        if (unmatched or len(p_names) > len(g_names) - len(unmatched)) and len(self.errors) < max_errors:
            self.errors.append({"doc": doc_id, "slot": "Parties", "gold": g_names, "pred": p_names})

        for s in SLOTS[1:]:
            g, p = gold[s], pred[s]
            if g:
                self.n_gold[s] += 1
            if not g and p is None:
                continue
            if g and p in g:
                self.tp[s] += 1
                continue
            if p is not None:
                self.fp[s] += 1
            if g:
                self.fn[s] += 1
            if len(self.errors) < max_errors:
                self.errors.append({"doc": doc_id, "slot": s, "gold": sorted(g), "pred": p})

    def result(self) -> dict:
        out = {}
        for s in SLOTS:
            out[s] = _prf(self.tp[s], self.fp[s], self.fn[s])
            out[s]["docs_with_gold"] = self.n_gold[s]
        out["micro"] = _prf(sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values()))
        return out


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}
