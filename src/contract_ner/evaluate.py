"""Document-level evaluation: precision / recall / F1 per entity type.

Unlike the window-level numbers printed during training, this runs the full
inference path (windowing + stitching + BIO decoding) on whole documents and
scores exact entity matches with ``seqeval``, so it reflects what the
extractor actually returns. A relaxed, span-overlap based score is reported
alongside because a one-token boundary slip (e.g. keeping/dropping ``พ.ศ.``)
is much less harmful for downstream normalisation than a missed entity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from .predict import Extractor
from .schema import ENTITY_TYPES, Document, Span, spans_to_bio
from .tokenization import segment_words


@dataclass
class EvalResult:
    n_docs: int
    strict: dict  # seqeval exact-match
    relaxed: dict  # overlap-based per type
    report: str
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_docs": self.n_docs, "strict": self.strict, "relaxed": self.relaxed, "report": self.report,
                "errors": self.errors}


def evaluate_documents(extractor: Extractor, docs: list[Document], max_errors: int = 200) -> EvalResult:
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    tp = {t: 0 for t in ENTITY_TYPES}
    fp = {t: 0 for t in ENTITY_TYPES}
    fn = {t: 0 for t in ENTITY_TYPES}
    errors: list[dict] = []

    for d in docs:
        offsets = segment_words(d.text, extractor.lang)
        gold_tags = spans_to_bio(offsets, d.entities)
        wp = extractor.predict_words(d.text)
        assert wp.offsets == offsets
        y_true.append(gold_tags)
        y_pred.append(wp.tags)

        pred_spans = extractor.predict(d.text)
        matched_gold: set[int] = set()
        for ps in pred_spans:
            hit = None
            for gi, gs in enumerate(d.entities):
                if gi in matched_gold:
                    continue
                if gs.label == ps.label and gs.overlaps(ps):
                    hit = gi
                    break
            if hit is None:
                fp[ps.label] += 1
                if len(errors) < max_errors:
                    errors.append({"doc": d.id, "kind": "FP", "label": ps.label, "text": ps.text[:120]})
            else:
                matched_gold.add(hit)
                tp[ps.label] += 1
        for gi, gs in enumerate(d.entities):
            if gi not in matched_gold:
                fn[gs.label] += 1
                if len(errors) < max_errors:
                    errors.append({"doc": d.id, "kind": "FN", "label": gs.label, "text": gs.text[:120]})

    strict = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    relaxed = {}
    for t in ENTITY_TYPES:
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        relaxed[t] = {"precision": p, "recall": r, "f1": f, "support": tp[t] + fn[t]}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    relaxed["micro"] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": TP + FN}

    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return EvalResult(len(docs), strict, relaxed, report, errors)


def save_result(res: EvalResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)


def format_result(res: EvalResult, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"== {title} ({res.n_docs} docs) ==")
    s = res.strict
    lines.append(f"strict  (exact span, seqeval)  P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
    m = res.relaxed["micro"]
    lines.append(f"relaxed (overlap, per type)    P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    lines.append("")
    lines.append(res.report)
    return "\n".join(lines)
