"""Build English contract NER documents from the Kaggle *Legal Contract Dataset*.

The Kaggle release (``changethan/legal-contract-dataset``) re-shapes CUAD into
``all_reshaped_clauses.csv`` (``filename, clause_type, clause_text``) and ships
the full contract text in ``complete.json``. The clause spans lost their
character offsets in that re-shaping, so we project them back into the full
text by string search. Only the CUAD categories that correspond to a field in
our schema are used:

======================  ===============
CUAD category           entity type
======================  ===============
Parties                 PARTY
Agreement Date          AGREEMENT_DATE
Effective Date          START_DATE
Expiration Date         END_DATE
Governing Law           JURISDICTION
======================  ===============

CUAD has no clean "contract value", "payment due" or "penalty rate" category
(``Liquidated Damages`` is a whole-clause span present in ~60 contracts), so
the English model covers five of the eight target fields. Alignment quality on
the retained categories is 92–100 % (see ``tests/test_cuad.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..schema import Document, Span

CATEGORY_MAP = {
    "Parties": "PARTY",
    "Agreement Date": "AGREEMENT_DATE",
    "Effective Date": "START_DATE",
    "Expiration Date": "END_DATE",
    "Governing Law": "JURISDICTION",
}

_PRIORITY = {"AGREEMENT_DATE": 0, "PARTY": 1, "END_DATE": 2, "JURISDICTION": 3, "START_DATE": 4}
_PAGE_RE = re.compile(r"\s*\(Page \d+\)\s*$")
_QUOTE_CLASS = {
    '"': "[\"“”]", "“": "[\"“”]", "”": "[\"“”]",
    "'": "['‘’]", "‘": "['‘’]", "’": "['‘’]",
}


@dataclass
class AlignStats:
    spans_total: int = 0
    spans_aligned: int = 0
    per_type: dict | None = None

    def rate(self) -> float:
        return self.spans_aligned / self.spans_total if self.spans_total else 0.0


def _stem(name: str) -> str:
    return os.path.splitext(name)[0].casefold()


def split_spans(cell: str) -> list[str]:
    """``'A (Page 1)\\n\\nB (Page 2)'`` → ``['A', 'B']``."""
    out = []
    for part in str(cell).split("\n\n"):
        part = _PAGE_RE.sub("", part).strip()
        if part:
            out.append(part)
    return out


def span_regex(span: str) -> re.Pattern:
    """Whitespace/quote/case-insensitive pattern locating ``span`` in raw text."""
    toks = span.replace("\xa0", " ").split()
    pieces = []
    for t in toks:
        pieces.append("".join(_QUOTE_CLASS.get(c) or re.escape(c) for c in t))
    return re.compile(r"\s+".join(pieces), re.IGNORECASE)


def _occurrences(text: str, span: str, limit: int = 50) -> list[tuple[int, int]]:
    pat = span_regex(span)
    return [(m.start(), m.end()) for m in pat.finditer(text)][:limit]


def align_contract(text: str, clauses: list[tuple[str, str]]) -> tuple[list[Span], int, int]:
    """Project ``(clause_type, clause_text)`` rows onto ``text``.

    Returns (spans, n_spans_tried, n_spans_found). Parties are anchored to the
    first legal-name occurrence so that aliases such as ``"Company"`` are picked
    from the preamble rather than from a random later mention.
    """
    candidates: list[tuple[str, str]] = []  # (label, span text)
    for ctype, cell in clauses:
        label = CATEGORY_MAP.get(ctype)
        if not label:
            continue
        for sp in split_spans(cell):
            candidates.append((label, sp))

    tried = found = 0
    # anchor: first occurrence of the longest PARTY span
    anchor = None
    party_spans = sorted((s for l, s in candidates if l == "PARTY"), key=len, reverse=True)
    for sp in party_spans:
        occ = _occurrences(text, sp, limit=1)
        if occ:
            anchor = occ[0][0]
            break

    picked: list[Span] = []
    for label, sp in candidates:
        tried += 1
        occ = _occurrences(text, sp)
        if not occ:
            continue
        found += 1
        if label == "PARTY" and anchor is not None:
            s, e = min(occ, key=lambda o: abs(o[0] - anchor))
        else:
            s, e = occ[0]
        picked.append(Span(s, e, label, text[s:e]))

    # resolve overlaps: keep longer spans; on identical spans prefer AGREEMENT_DATE
    # over START_DATE (CUAD's "Effective Date" is usually the signing date again;
    # post-processing falls back to Agreement_Date when no Start_Date is found)
    picked.sort(key=lambda x: (x.start, -(x.end - x.start), _PRIORITY.get(x.label, 9)))
    kept: list[Span] = []
    for sp in picked:
        if kept and sp.overlaps(kept[-1]):
            if (sp.end - sp.start) > (kept[-1].end - kept[-1].start):
                kept[-1] = sp
            continue
        kept.append(sp)
    return kept, tried, found


def load_kaggle(raw_dir: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    raw_dir = Path(raw_dir)
    with open(raw_dir / "complete.json", encoding="utf-8") as f:
        items = json.load(f)
    full = {_stem(it["file_name"]): it["full_text"] for it in items}
    df = pd.read_csv(raw_dir / "all_reshaped_clauses.csv")
    df["stem"] = df["filename"].map(_stem)
    return full, df


def build_documents(raw_dir: str | Path) -> tuple[list[Document], AlignStats]:
    full, df = load_kaggle(raw_dir)
    docs: list[Document] = []
    stats = AlignStats(per_type={})
    for stem, grp in df.groupby("stem", sort=True):
        text = full.get(stem)
        if text is None:
            continue
        clauses = list(zip(grp["clause_type"], grp["clause_text"]))
        spans, tried, found = align_contract(text, clauses)
        stats.spans_total += tried
        stats.spans_aligned += found
        for sp in spans:
            stats.per_type[sp.label] = stats.per_type.get(sp.label, 0) + 1
        if not spans:
            continue
        docs.append(Document(id=f"cuad-{_short_hash(stem)}", text=text, entities=spans, lang="en",
                             source="cuad", meta={"file": stem}))
    return docs, stats


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def split_documents(docs: list[Document], val_frac: float = 0.1, test_frac: float = 0.1
                    ) -> tuple[list[Document], list[Document], list[Document]]:
    """Deterministic contract-level split keyed on the document id hash."""
    train, val, test = [], [], []
    for d in docs:
        h = int(hashlib.sha1(d.id.encode()).hexdigest(), 16) % 1000 / 1000
        if h < test_frac:
            test.append(d)
        elif h < test_frac + val_frac:
            val.append(d)
        else:
            train.append(d)
    return train, val, test
