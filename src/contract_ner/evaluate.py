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

from .postprocess import _party_key, validate_span
from .predict import Extractor
from .schema import ENTITY_TYPES, Document, spans_to_bio
from .tokenization import segment_words


@dataclass
class EvalResult:
    n_docs: int
    strict: dict  # seqeval exact-match
    relaxed: dict  # overlap-based per type
    fields: dict  # set-of-values per document and type
    report: str
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_docs": self.n_docs, "strict": self.strict, "relaxed": self.relaxed, "fields": self.fields,
                "report": self.report, "errors": self.errors}


def _value_key(text: str) -> str:
    """Normalise a surface string so repeated mentions collapse to one value."""
    return _party_key(text)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}


def evaluate_documents(extractor: Extractor, docs: list[Document], max_errors: int = 200,
                       validate: bool = True) -> EvalResult:
    """Score the extractor on ``docs``.

    With ``validate=True`` (the default) predictions pass through the same
    span-validation stage the JSON output uses, so the numbers describe the
    pipeline's output rather than the raw tagger.
    """
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    tp = {t: 0 for t in ENTITY_TYPES}
    fp = {t: 0 for t in ENTITY_TYPES}
    fn = {t: 0 for t in ENTITY_TYPES}
    ftp = {t: 0 for t in ENTITY_TYPES}
    ffp = {t: 0 for t in ENTITY_TYPES}
    ffn = {t: 0 for t in ENTITY_TYPES}
    errors: list[dict] = []

    for d in docs:
        offsets = segment_words(d.text, extractor.lang)
        gold_tags = spans_to_bio(offsets, d.entities)
        pred_spans = extractor.predict(d.text)
        if validate:
            pred_spans = [sp for sp in pred_spans if validate_span(sp)]
        y_true.append(gold_tags)
        y_pred.append(spans_to_bio(offsets, pred_spans))

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
        # field level: the set of distinct values per type (what ends up in the metadata table)
        for t in ENTITY_TYPES:
            gold_vals = {_value_key(e.text) for e in d.entities if e.label == t}
            pred_vals = {_value_key(p.text) for p in pred_spans if p.label == t}
            ftp[t] += len(gold_vals & pred_vals)
            ffp[t] += len(pred_vals - gold_vals)
            ffn[t] += len(gold_vals - pred_vals)

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
    fields = {t: _prf(ftp[t], ffp[t], ffn[t]) for t in ENTITY_TYPES}
    fields["micro"] = _prf(sum(ftp.values()), sum(ffp.values()), sum(ffn.values()))

    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return EvalResult(len(docs), strict, relaxed, fields, report, errors)


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
    f = res.fields["micro"]
    lines.append(f"field   (distinct values/doc)  P={f['precision']:.3f} R={f['recall']:.3f} F1={f['f1']:.3f}")
    lines.append("")
    lines.append(res.report)
    lines.append("field-level per type:")
    for t in ENTITY_TYPES:
        v = res.fields[t]
        lines.append(f"  {t:15s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")
    return "\n".join(lines)
