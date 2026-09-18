"""Entity schema, BIO label space and the canonical document/span records.

The eight target fields from the project brief map onto seven entity types
(PARTY covers both Party A and Party B; the A/B assignment is done during
post-processing) plus AGREEMENT_DATE, which real contracts almost always carry
and which the model must learn to tell apart from START_DATE.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Iterator

ENTITY_TYPES: tuple[str, ...] = (
    "PARTY",
    "AGREEMENT_DATE",
    "START_DATE",
    "END_DATE",
    "CONTRACT_VALUE",
    "PAYMENT_DUE",
    "PENALTY_RATE",
    "JURISDICTION",
)

# Human-facing field names used in the structured JSON output (slide 12 of the deck).
OUTPUT_FIELDS: tuple[str, ...] = (
    "Party_A",
    "Party_B",
    "Agreement_Date",
    "Start_Date",
    "End_Date",
    "Contract_Value",
    "Payment_Due_Date",
    "Penalty_Rate",
    "Jurisdiction",
)

O_LABEL = "O"


def build_bio_labels(entity_types: Iterable[str] = ENTITY_TYPES) -> list[str]:
    """``["O", "B-PARTY", "I-PARTY", ...]`` in a stable order."""
    labels = [O_LABEL]
    for et in entity_types:
        labels.append(f"B-{et}")
        labels.append(f"I-{et}")
    return labels


BIO_LABELS: list[str] = build_bio_labels()
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: l for l, i in LABEL2ID.items()}


@dataclass
class Span:
    """A labelled character span ``text[start:end]`` (end exclusive)."""

    start: int
    end: int
    label: str
    text: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty/negative span {self.start}:{self.end} ({self.label})")
        if self.label not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {self.label!r}")

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    """One annotated contract: raw text plus character-level entity spans.

    This is the canonical on-disk format (JSONL, one document per line) and is
    the same shape that Label Studio / doccano exports produce, so hand-labelled
    contracts can be dropped in next to the synthetic ones.
    """

    id: str
    text: str
    entities: list[Span] = field(default_factory=list)
    lang: str = "th"
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entities = sorted(self.entities, key=lambda s: (s.start, s.end))
        for s in self.entities:
            if s.end > len(self.text):
                raise ValueError(f"{self.id}: span {s} exceeds text length {len(self.text)}")
            if not s.text:
                s.text = self.text[s.start : s.end]
        for a, b in zip(self.entities, self.entities[1:]):
            if a.overlaps(b):
                raise ValueError(f"{self.id}: overlapping spans {a} / {b}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "lang": self.lang,
            "source": self.source,
            "text": self.text,
            "entities": [s.to_dict() for s in self.entities],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            id=d["id"],
            text=d["text"],
            entities=[Span(**e) for e in d.get("entities", [])],
            lang=d.get("lang", "th"),
            source=d.get("source", "unknown"),
            meta=d.get("meta", {}),
        )


def spans_to_bio(word_offsets: list[tuple[int, int]], spans: list[Span]) -> list[str]:
    """Project character spans onto pre-segmented words → BIO tags.

    A word is *inside* a span when its character range overlaps the span. The
    first overlapping word gets ``B-``, the rest ``I-``. Words are expected in
    document order; spans are expected non-overlapping.
    """
    tags = [O_LABEL] * len(word_offsets)
    spans = sorted(spans, key=lambda s: s.start)
    wi = 0
    for sp in spans:
        first = True
        while wi < len(word_offsets) and word_offsets[wi][1] <= sp.start:
            wi += 1
        j = wi
        while j < len(word_offsets) and word_offsets[j][0] < sp.end:
            ws, we = word_offsets[j]
            if we > sp.start:  # overlap
                tags[j] = f"{'B' if first else 'I'}-{sp.label}"
                first = False
            j += 1
    return tags


def bio_to_spans(
    tags: list[str], word_offsets: list[tuple[int, int]], text: str | None = None
) -> list[Span]:
    """Decode word-level BIO tags back into character spans.

    Tolerant decoding: an ``I-X`` that does not continue an ``X`` entity opens
    a new one (the usual seqeval/IOB2 lenient behaviour), so a model that
    occasionally drops the ``B-`` still yields a usable span.
    """
    spans: list[Span] = []
    cur_label: str | None = None
    cur_start = cur_end = -1
    for tag, (ws, we) in zip(tags, word_offsets):
        if tag == O_LABEL or "-" not in tag:
            if cur_label is not None:
                spans.append(_mk(cur_start, cur_end, cur_label, text))
                cur_label = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or cur_label != label:
            if cur_label is not None:
                spans.append(_mk(cur_start, cur_end, cur_label, text))
            cur_label, cur_start, cur_end = label, ws, we
        else:
            cur_end = we
    if cur_label is not None:
        spans.append(_mk(cur_start, cur_end, cur_label, text))
    return spans


def _mk(start: int, end: int, label: str, text: str | None) -> Span:
    return Span(start, end, label, text[start:end] if text is not None else "")


def iter_entities(tags: list[str]) -> Iterator[tuple[str, int, int]]:
    """Yield ``(label, first_word_idx, last_word_idx_exclusive)`` from BIO tags."""
    n = len(tags)
    i = 0
    while i < n:
        if tags[i] == O_LABEL:
            i += 1
            continue
        label = tags[i].split("-", 1)[1]
        j = i + 1
        while j < n and tags[j] == f"I-{label}":
            j += 1
        yield label, i, j
        i = j
