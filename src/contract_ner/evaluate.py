"""Document-level evaluation: precision / recall / F1 per entity type.

Unlike the window-level numbers printed during training, this runs the full
inference path (windowing + stitching + BIO decoding) on whole documents and
scores exact entity matches with ``seqeval``, so it reflects what the
extractor actually returns. A relaxed, span-overlap based score is reported
alongside because a one-token boundary slip (e.g. keeping/dropping ``พ.ศ.``)
is much less harmful for downstream normalisation than a missed entity. The
field view collapses repeated mentions to distinct values per document, and
the slot view (:mod:`contract_ner.slots`) scores the assembled metadata table
itself — one normalised value per cell — which is what a user of the output
sees.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from .augment import map_labels
from .lang import rules_for
from .postprocess import _party_key, assemble, validate_span
from .predict import Extractor
from .schema import ENTITY_TYPES, Document, Span, spans_to_bio
from .slots import SlotScorer, gold_slots, pred_slots
from .tokenization import segment_words


@dataclass
class EvalResult:
    n_docs: int
    strict: dict  # seqeval exact-match
    relaxed: dict  # overlap-based per type
    fields: dict  # set-of-values per document and type
    slots: dict  # one value per metadata-table cell, after assemble() and normalisation
    report: str
    errors: list[dict] = field(default_factory=list)
    slot_errors: list[dict] = field(default_factory=list)
    types: tuple[str, ...] = ENTITY_TYPES  # entity types the model was scored on
    forms: dict = field(default_factory=dict)
    """Recall per surface form of the gold spans (``lang.<language>.span_form``).

    One number per label hides that a label is several kinds of string: the
    tagger can find every ``June 21, 1999`` and no ``7/7/98`` and still look
    average. This is the view that says which form to fix, and it pairs with the
    dataset profile in :mod:`contract_ner.profile`, which counts the same forms
    in the training data."""

    def to_dict(self) -> dict:
        return {"n_docs": self.n_docs, "types": list(self.types), "strict": self.strict, "relaxed": self.relaxed,
                "fields": self.fields, "slots": self.slots, "forms": self.forms, "report": self.report,
                "errors": self.errors, "slot_errors": self.slot_errors}


def _value_key(text: str) -> str:
    """Normalise a surface string so repeated mentions collapse to one value."""
    return _party_key(text)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp + fn}


def _predict(extractor, text: str) -> list[tuple[Span, float]]:
    """``(span, confidence)`` pairs; extractors without confidences get 1.0."""
    if hasattr(extractor, "predict_with_confidence"):
        return list(extractor.predict_with_confidence(text))
    return [(sp, 1.0) for sp in extractor.predict(text)]


def evaluate_documents(extractor: Extractor, docs: list[Document], max_errors: int = 200,
                       validate: bool = True, types: tuple[str, ...] | None = None) -> EvalResult:
    """Score the extractor on ``docs``.

    With ``validate=True`` (the default) predictions pass through the same
    span-validation stage the JSON output uses, so the numbers describe the
    pipeline's output rather than the raw tagger. Scoring covers ``types``
    (default: the types the model was trained on); gold spans of other types
    are ignored rather than counted as misses. Gold labels are folded through
    the extractor's ``label_map`` (if any) so a model trained with
    ``START_DATE → AGREEMENT_DATE`` is scored against the same merged gold.
    """
    types = tuple(types) if types else tuple(extractor.entity_types)
    label_map = getattr(extractor, "label_map", None) or {}
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    tp = {t: 0 for t in types}
    fp = {t: 0 for t in types}
    fn = {t: 0 for t in types}
    ftp = {t: 0 for t in types}
    ffp = {t: 0 for t in types}
    ffn = {t: 0 for t in types}
    errors: list[dict] = []
    slot_scorer = SlotScorer()
    span_form = rules_for(extractor.lang).span_form
    form_found: dict[tuple[str, str], int] = {}
    form_missed: dict[tuple[str, str], int] = {}

    for d in docs:
        offsets = segment_words(d.text, extractor.lang)
        gold = [e for e in map_labels(d, label_map).entities if e.label in types]
        gold_tags = spans_to_bio(offsets, gold)
        with_conf = [(sp, c) for sp, c in _predict(extractor, d.text) if sp.label in types]
        pred_spans = [sp for sp, _ in with_conf]
        if validate:
            pred_spans = [sp for sp in pred_spans if validate_span(sp, extractor.lang)]
        # slot level: the table assemble() would show for this document vs. the table implied by gold
        rec = assemble(d.text, with_conf, lang=extractor.lang)
        slot_scorer.add(d.id, gold_slots(d), pred_slots(rec, [sp for sp in pred_spans if sp.label == "PARTY"]),
                        max_errors=max_errors)
        y_true.append(gold_tags)
        y_pred.append(spans_to_bio(offsets, pred_spans))

        matched_gold: set[int] = set()
        for ps in pred_spans:
            hit = None
            for gi, gs in enumerate(gold):
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
        for gi, gs in enumerate(gold):
            key = (gs.label, span_form(gs))
            if gi in matched_gold:
                form_found[key] = form_found.get(key, 0) + 1
            else:
                form_missed[key] = form_missed.get(key, 0) + 1
                fn[gs.label] += 1
                if len(errors) < max_errors:
                    errors.append({"doc": d.id, "kind": "FN", "label": gs.label, "text": gs.text[:120],
                                   "form": key[1]})
        # field level: the set of distinct values per type (what ends up in the metadata table)
        for t in types:
            gold_vals = {_value_key(e.text) for e in gold if e.label == t}
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
    for t in types:
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        relaxed[t] = {"precision": p, "recall": r, "f1": f, "support": tp[t] + fn[t]}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    relaxed["micro"] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": TP + FN}
    fields = {t: _prf(ftp[t], ffp[t], ffn[t]) for t in types}
    fields["micro"] = _prf(sum(ftp.values()), sum(ffp.values()), sum(ffn.values()))

    forms: dict[str, dict[str, dict]] = {}
    for (label, form) in sorted(set(form_found) | set(form_missed)):
        found, missed = form_found.get((label, form), 0), form_missed.get((label, form), 0)
        forms.setdefault(label, {})[form] = {
            "found": found, "missed": missed, "support": found + missed,
            "recall": found / (found + missed) if found + missed else 0.0,
        }

    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return EvalResult(len(docs), strict, relaxed, fields, slot_scorer.result(), report, errors,
                      slot_scorer.errors, types, forms)


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
    sl = res.slots["micro"]
    lines.append(f"slot    (table cells, normalised) P={sl['precision']:.3f} R={sl['recall']:.3f} F1={sl['f1']:.3f}")
    lines.append("")
    lines.append(res.report)
    lines.append("field-level per type:")
    for t in res.types:
        v = res.fields[t]
        lines.append(f"  {t:15s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")
    lines.append("recall per surface form (overlap match, gold spans grouped by how they are written):")
    for label in res.types:
        for form, v in sorted(res.forms.get(label, {}).items(), key=lambda kv: -kv[1]["support"]):
            lines.append(f"  {label:15s} {form:22s} R={v['recall']:.3f} "
                         f"({v['found']}/{v['support']})")
    lines.append("slot-level per table cell:")
    for k, v in res.slots.items():
        if k == "micro":
            continue
        lines.append(f"  {k:15s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
                     f"(docs with gold={v['docs_with_gold']})")
    return "\n".join(lines)
