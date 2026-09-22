"""Label-distribution analysis of an annotated dataset, run before any training.

A label in a contract corpus is not one kind of string. ``PARTY`` is a legal
name in the preamble, a quoted alias two lines later and a sentence of
"collectively referred to as the Parties" boilerplate; ``END_DATE`` is a date
in one contract and a thirty-word clause about renewal in the next. Training a
single tagger over all of them, and reporting one score per label, hides which
of those forms the model actually learns.

This module answers four questions about a dataset before it is used:

1. **How much of each label is there** — spans, documents containing the label,
   spans per document. A label present in 18 % of documents is not learned the
   same way as one present in 99 %.
2. **What shape are the spans** — length in words, so short-span types
   (a date) are not scored or trained as if they were clause-level types.
3. **Where do they sit** — position in the document, which decides whether
   dropping entity-free windows loses training signal evenly or not.
4. **Which surface forms occur, and do the splits agree** — the per-label
   buckets from ``lang.<language>.span_form``. A form that is common in the test
   split and rare in training is a predictable miss, and it is cheaper to see it
   here than after a two-hour fine-tune.

The report is written as JSON (machine-readable, for the work log and the
README tables) and printed as Markdown.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .lang import rules_for
from .schema import Document

#: position buckets, as a fraction of the document's length
_POSITION_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("first 5%", 0.0, 0.05),
    ("5-25%", 0.05, 0.25),
    ("25-50%", 0.25, 0.5),
    ("second half", 0.5, 1.01),
)


def _percentile(values: list[int], fraction: float) -> int:
    """The value below which ``fraction`` of ``values`` fall (nearest rank, no interpolation)."""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


@dataclass
class LabelProfile:
    """Everything the profile knows about one entity type in one split."""

    label: str
    n_spans: int
    n_documents_with_label: int
    coverage: float                      # documents containing the label / documents in the split
    spans_per_document: dict[str, int]   # "0", "1", "2", "3+" → number of documents
    words_median: float
    words_p90: int
    words_max: int
    chars_median: float
    position: dict[str, float]           # position bucket → fraction of the label's spans
    forms: dict[str, int]                # surface form → number of spans
    form_share: dict[str, float]
    form_examples: dict[str, str]
    forms_absent: list[str]              # forms this label can take that the split contains none of
    repeated_value_rate: float           # share of spans whose text repeats within the same document


@dataclass
class SplitProfile:
    """The profile of one dataset file."""

    name: str
    path: str
    lang: str
    n_documents: int
    n_spans: int
    chars_median: int
    entity_char_ratio: float             # characters inside a gold span / characters in the corpus
    labels: dict[str, LabelProfile] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["labels"] = {k: asdict(v) for k, v in self.labels.items()}
        return d


def profile_documents(docs: list[Document], name: str = "", path: str = "",
                      lang: str | None = None, types: tuple[str, ...] | None = None,
                      examples_per_form: int = 1) -> SplitProfile:
    """Measure label counts, span shapes, positions and surface forms over ``docs``."""
    if not docs:
        raise ValueError("no documents to profile")
    lang = lang or docs[0].lang
    rules = rules_for(lang)
    span_form = rules.span_form
    known_forms = rules.span_forms
    labels_present = types or tuple(dict.fromkeys(e.label for d in docs for e in d.entities))

    total_chars = sum(len(d.text) for d in docs)
    entity_chars = sum(e.end - e.start for d in docs for e in d.entities)
    split = SplitProfile(
        name=name or Path(path).stem, path=str(path), lang=lang, n_documents=len(docs),
        n_spans=sum(len(d.entities) for d in docs),
        chars_median=int(statistics.median(len(d.text) for d in docs)),
        entity_char_ratio=entity_chars / total_chars if total_chars else 0.0,
    )

    for label in labels_present:
        words: list[int] = []
        chars: list[int] = []
        positions: list[float] = []
        forms: Counter[str] = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        per_document: Counter[int] = Counter()
        repeated = 0
        documents_with_label = 0
        for d in docs:
            spans = [e for e in d.entities if e.label == label]
            per_document[len(spans)] += 1
            documents_with_label += bool(spans)
            mentions: Counter[str] = Counter(" ".join(e.text.split()).casefold() for e in spans)
            repeated += sum(count - 1 for count in mentions.values() if count > 1)
            for e in spans:
                text = " ".join(e.text.split())
                words.append(len(text.split()))
                chars.append(len(text))
                positions.append(e.start / max(1, len(d.text)))
                form = span_form(e)
                forms[form] += 1
                if len(examples[form]) < examples_per_form:
                    examples[form].append(text[:110])
        if not words:
            continue
        n = len(words)
        split.labels[label] = LabelProfile(
            label=label,
            n_spans=n,
            n_documents_with_label=documents_with_label,
            coverage=documents_with_label / len(docs),
            spans_per_document={
                "0": per_document[0], "1": per_document[1], "2": per_document[2],
                "3+": sum(documents for spans_in_document, documents in per_document.items()
                          if spans_in_document >= 3),
            },
            words_median=statistics.median(words),
            words_p90=_percentile(words, 0.9),
            words_max=max(words),
            chars_median=statistics.median(chars),
            position={
                bucket: sum(low <= position < high for position in positions) / n
                for bucket, low, high in _POSITION_BUCKETS
            },
            forms=dict(forms.most_common()),
            form_share={form: count / n for form, count in forms.most_common()},
            form_examples={form: shown[0] for form, shown in examples.items()},
            forms_absent=[form for form in known_forms.get(label, ()) if form not in forms],
            repeated_value_rate=repeated / n,
        )
    return split


# ------------------------------------------------------------------ drift between splits
@dataclass
class FormDrift:
    """One surface form's share in the reference split versus another split."""

    label: str
    form: str
    reference_share: float
    other_share: float
    other_count: int
    verdict: str  # "under-represented in training", "unseen in training", "balanced"


def compare_forms(reference: SplitProfile, other: SplitProfile, min_count: int = 2,
                  ratio: float = 2.0) -> list[FormDrift]:
    """Surface forms of ``other`` that training (``reference``) under-represents.

    A form is flagged when it holds at least ``min_count`` spans in ``other`` and
    its share there is ``ratio`` times its share in the reference split (or it
    does not occur in the reference split at all). Those are the misses that can
    be predicted before training rather than diagnosed after it.
    """
    drifts: list[FormDrift] = []
    for label, other_profile in other.labels.items():
        reference_profile = reference.labels.get(label)
        for form, count in other_profile.forms.items():
            if count < min_count:
                continue
            other_share = other_profile.form_share[form]
            reference_share = reference_profile.form_share.get(form, 0.0) if reference_profile else 0.0
            if reference_share == 0.0:
                verdict = "unseen in training"
            elif other_share >= ratio * reference_share:
                verdict = "under-represented in training"
            else:
                continue
            drifts.append(FormDrift(label, form, reference_share, other_share, count, verdict))
    return sorted(drifts, key=lambda d: (-d.other_count, d.label))


# ------------------------------------------------------------------ windowing diagnostics
@dataclass
class WindowDiagnostics:
    """What the configured window length and overlap do to this dataset.

    Long contracts are cut into fixed-length windows, so two settings have to be
    checked against the data rather than guessed: whether a window is long enough
    to hold a whole clause-level span, and how much of the corpus is windows with
    no entity in them at all (the ratio the training code samples down).
    """

    max_length: int
    stride: int
    n_windows: int
    windows_with_entity: int
    entity_window_share: float
    spans_cut_by_a_boundary: int
    spans_cut_share: float
    spans_longer_than_a_window: int
    subwords_per_span: dict[str, dict[str, float]]  # label → median / 90th percentile / max


def window_diagnostics(docs: list[Document], encoder, lang: str) -> WindowDiagnostics:
    """Encode ``docs`` the way training does and measure the window split."""
    from .schema import spans_to_bio

    n_windows = 0
    windows_with_entity = 0
    cut = 0
    too_long = 0
    n_spans = 0
    lengths: dict[str, list[int]] = defaultdict(list)
    o_id = encoder.label2id["O"]

    for d in docs:
        offsets = rules_for(lang).segment_words(d.text)
        spans = [sp for sp in d.entities if f"B-{sp.label}" in encoder.label2id]
        tags = spans_to_bio(offsets, spans)
        enc = encoder.encode(d.text, offsets, tags)
        covered: list[tuple[int, int]] = []
        for w in enc.windows:
            n_windows += 1
            if any(label not in (-100, o_id) for label in w.labels):
                windows_with_entity += 1
            real = [i for i in w.subword_index if i >= 0]
            if real:
                covered.append((min(real), max(real)))
        # a span is usable only if some single window holds all of its subwords
        word_of_subword = enc.subword_to_word
        for sp in spans:
            words = {i for i, (start, end) in enumerate(offsets) if start < sp.end and end > sp.start}
            subwords = [i for i, w in enumerate(word_of_subword) if w in words]
            if not subwords:
                continue
            n_spans += 1
            first, last = min(subwords), max(subwords)
            lengths[sp.label].append(last - first + 1)
            if last - first + 1 > encoder.body_len:
                too_long += 1
            if not any(low <= first and last <= high for low, high in covered):
                cut += 1

    return WindowDiagnostics(
        max_length=encoder.max_length,
        stride=encoder.stride,
        n_windows=n_windows,
        windows_with_entity=windows_with_entity,
        entity_window_share=windows_with_entity / n_windows if n_windows else 0.0,
        spans_cut_by_a_boundary=cut,
        spans_cut_share=cut / n_spans if n_spans else 0.0,
        spans_longer_than_a_window=too_long,
        subwords_per_span={
            label: {"median": statistics.median(v), "p90": _percentile(v, 0.9), "max": max(v)}
            for label, v in sorted(lengths.items())
        },
    )


def format_window_diagnostics(name: str, diagnostics: WindowDiagnostics) -> str:
    """The Markdown block for the windowing diagnostics of one split."""
    lines = [
        f"### Windowing — {name} (window {diagnostics.max_length} subwords, "
        f"overlap {diagnostics.stride})",
        "",
        f"* windows: {diagnostics.n_windows:,}, of which {diagnostics.windows_with_entity:,} "
        f"({diagnostics.entity_window_share:.1%}) contain at least one entity",
        f"* gold spans no single window holds in full: {diagnostics.spans_cut_by_a_boundary} "
        f"({diagnostics.spans_cut_share:.2%}); spans longer than a whole window: "
        f"{diagnostics.spans_longer_than_a_window}",
        "",
        "| Label | subwords per span (median / 90th pct / max) |",
        "|---|---|",
    ]
    for label, stats in diagnostics.subwords_per_span.items():
        lines.append(f"| {label} | {stats['median']:g} / {stats['p90']:g} / {stats['max']:g} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ rendering
def format_profile(split: SplitProfile) -> str:
    """The Markdown block for one split."""
    lines = [
        f"### {split.name} — {split.n_documents} documents, {split.n_spans} spans "
        f"(language: {split.lang})",
        "",
        f"Median document length: {split.chars_median:,} characters. "
        f"Characters inside a gold span: {split.entity_char_ratio:.2%} of the corpus "
        f"(this is how imbalanced the token-classification problem is).",
        "",
        "| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | "
        "words (median / 90th pct / max) | in first 5% of text | repeated mentions |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for label, profile in split.labels.items():
        per_doc = profile.spans_per_document
        lines.append(
            f"| {label} | {profile.n_spans} | {profile.n_documents_with_label} "
            f"({profile.coverage:.0%}) | {per_doc['0']} / {per_doc['1']} / {per_doc['2']} / {per_doc['3+']} | "
            f"{profile.words_median:g} / {profile.words_p90} / {profile.words_max} | "
            f"{profile.position['first 5%']:.0%} | {profile.repeated_value_rate:.0%} |"
        )
    lines.append("")
    lines.append("Surface forms per label:")
    lines.append("")
    for label, profile in split.labels.items():
        lines.append(f"* **{label}**")
        for form, count in profile.forms.items():
            example = profile.form_examples.get(form, "")
            lines.append(f"  * `{form}` — {count} ({profile.form_share[form]:.1%}) · e.g. “{example}”")
        if profile.forms_absent:
            lines.append("  * not present in this split: " + ", ".join(f"`{f}`" for f in profile.forms_absent))
    return "\n".join(lines)


def format_drift(reference_name: str, other_name: str, drifts: list[FormDrift]) -> str:
    """The Markdown block comparing one split's surface forms against training."""
    if not drifts:
        return (f"### {other_name} versus {reference_name}\n\n"
                "No surface form is materially under-represented in training.")
    lines = [
        f"### {other_name} versus {reference_name} — forms training under-represents",
        "",
        "| Label | form | share in " + reference_name + " | share in " + other_name +
        " | spans | verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    for d in drifts:
        lines.append(f"| {d.label} | `{d.form}` | {d.reference_share:.1%} | {d.other_share:.1%} | "
                     f"{d.other_count} | {d.verdict} |")
    return "\n".join(lines)


def build_report(splits: list[SplitProfile], reference: str | None = None,
                 windowing: dict[str, WindowDiagnostics] | None = None) -> str:
    """The full Markdown report: one section per split, the drift sections, the windowing section."""
    title = "# Label distribution\n"
    parts = [title, "\n\n".join(format_profile(s) for s in splits)]
    reference_split = next((s for s in splits if s.name == reference), splits[0] if splits else None)
    if reference_split is not None and len(splits) > 1:
        parts.append("## Surface-form drift between splits\n")
        parts.extend(
            format_drift(reference_split.name, s.name, compare_forms(reference_split, s))
            for s in splits if s is not reference_split
        )
    if windowing:
        parts.append("## Windowing\n")
        parts.extend(format_window_diagnostics(name, d) for name, d in windowing.items())
    return "\n\n".join(parts) + "\n"


def save_report(splits: list[SplitProfile], out_dir: str | Path, reference: str | None = None,
                stem: str = "label_distribution",
                windowing: dict[str, WindowDiagnostics] | None = None) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` into ``out_dir``; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    markdown_path = out_dir / f"{stem}.md"
    reference_split = next((s for s in splits if s.name == reference), splits[0] if splits else None)
    payload = {
        "splits": [s.to_dict() for s in splits],
        "drift": {
            s.name: [asdict(d) for d in compare_forms(reference_split, s)]
            for s in splits if reference_split is not None and s is not reference_split
        },
        "windowing": {name: asdict(d) for name, d in (windowing or {}).items()},
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(build_report(splits, reference, windowing), encoding="utf-8")
    return json_path, markdown_path
