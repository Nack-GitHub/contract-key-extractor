"""Training-time document transforms: label merging and date-format augmentation.

CUAD writes almost every agreement date as ``October 1, 2019``; the test split
also contains ``7/7/98``, ``3-31-16``, ``24 OCT 2011`` and ``September ___,
2019`` and the model tagged none of them. :func:`rewrite_dates` rewrites a
fraction of the dated spans in the training documents into such formats (the
underlying date is kept, so the gold value is unchanged) and shifts the
remaining offsets accordingly. :func:`map_labels` folds one label into another
(``START_DATE`` → ``AGREEMENT_DATE``: CUAD's effective-date spans are mostly
definitional clauses, and when they are a date the model cannot tell the two
apart from the surface alone). :func:`drop_dateless_spans` keeps a date label
as its own type but removes the dateless clauses from the training data, so the
tagger sees one consistent span shape per label.
"""
from __future__ import annotations

import random
from datetime import date

from .postprocess import normalize_date
from .schema import Document, Span

DATE_LABELS = ("AGREEMENT_DATE", "START_DATE")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_ORD = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}


def _ordinal(d: int) -> str:
    return f"{d}{_ORD.get(d, 'th')}"


# each format is a callable date -> surface; numeric forms are kept in the minority because page
# footers and section references look alike and most training windows never show them as negatives
_FORMATS: list[tuple[float, object]] = [
    (2, lambda d: f"{d.month}/{d.day}/{d.year % 100:02d}"),
    (2, lambda d: f"{d.month}/{d.day}/{d.year}"),
    (1, lambda d: f"{d.month:02d}/{d.day:02d}/{d.year}"),
    (1, lambda d: f"{d.month}-{d.day}-{d.year % 100:02d}"),
    (2, lambda d: f"{d.day} {_MONTHS[d.month - 1][:3].upper()} {d.year}"),
    (2, lambda d: f"{d.day} {_MONTHS[d.month - 1]} {d.year}"),
    (1, lambda d: f"{d.day} {_MONTHS[d.month - 1][:3]}. {d.year}"),
    (2, lambda d: f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"),
    (1, lambda d: f"{_MONTHS[d.month - 1][:3]}. {d.day}, {d.year}"),
    (1, lambda d: f"{_MONTHS[d.month - 1].upper()} {d.day}, {d.year}"),
    (1, lambda d: f"the {_ordinal(d.day)} day of {_MONTHS[d.month - 1]}, {d.year}"),
    (2, lambda d: f"{_MONTHS[d.month - 1]} {d.day}{_ORD.get(d.day, 'th')}, {d.year}"),
]
_WEIGHTS = [w for w, _ in _FORMATS]


def random_surface(d: date, rng: random.Random) -> str:
    (_, fmt), = rng.choices(_FORMATS, weights=_WEIGHTS, k=1)
    return fmt(d)


DATED_LABELS = ("AGREEMENT_DATE",)  # short date types: a span the pipeline could never normalise is dropped


def drop_dateless_spans(doc: Document, labels: tuple[str, ...] | list[str] | None) -> Document:
    """Remove spans of ``labels`` that contain no date the pipeline can normalise.

    A label whose gold mixes two kinds of string is two problems in one tagger.
    In this corpus ``START_DATE`` is a short date ("December 21, 2017") in three
    of five spans and a definitional clause ("shall become effective as of the
    first date above written") in the other two; the clause carries no value, is
    rejected by span validation at inference, and its 26-word average teaches the
    tagger a span shape the short dates never have. Dropping those spans from the
    *training* data leaves one consistent form to learn. The evaluation gold is
    never touched by this — a clause the model does not predict still counts as a
    miss there.
    """
    if not labels:
        return doc
    labels = tuple(labels)
    kept = [e for e in doc.entities
            if e.label not in labels or normalize_date(e.text, doc.lang) is not None]
    return Document(doc.id, doc.text, kept, doc.lang, doc.source, doc.meta)


def map_labels(doc: Document, label_map: dict[str, str] | None) -> Document:
    """Fold labels through ``label_map``; dateless spans of a short date type are dropped.

    Folding CUAD's ``Effective Date`` into ``AGREEMENT_DATE`` would otherwise
    teach the tagger that *"Effective Date" shall mean the date on which …* is
    an agreement date — span validation rejects such a span anyway, so it can
    neither be produced nor be a miss. Placeholders such as ``September ___,
    2019`` go the same way.
    """
    if not label_map:
        return doc
    ents = []
    for e in doc.entities:
        label = label_map.get(e.label, e.label)
        if label in DATED_LABELS and not normalize_date(e.text, doc.lang):
            continue
        ents.append(Span(e.start, e.end, label, e.text))
    return Document(doc.id, doc.text, ents, doc.lang, doc.source, doc.meta)


def rewrite_dates(doc: Document, prob: float, rng: random.Random, labels: tuple[str, ...] = DATE_LABELS) -> Document:
    """Rewrite each parsable date span of ``labels`` with probability ``prob``; offsets are re-based."""
    if prob <= 0:
        return doc
    out_text: list[str] = []
    out_ents: list[Span] = []
    cursor = 0
    shift = 0
    for e in doc.entities:  # sorted by start
        surface = e.text
        if e.label in labels and rng.random() < prob:
            iso = normalize_date(e.text, doc.lang)
            if iso and len(iso) == 10:
                surface = random_surface(date.fromisoformat(iso), rng)
        out_text.append(doc.text[cursor:e.start])
        out_text.append(surface)
        cursor = e.end
        start = e.start + shift
        shift += len(surface) - (e.end - e.start)
        out_ents.append(Span(start, start + len(surface), e.label, surface))
    out_text.append(doc.text[cursor:])
    return Document(doc.id, "".join(out_text), out_ents, doc.lang, doc.source, doc.meta)


def uppercase_party_names(doc: Document, prob: float, rng: random.Random,
                          labels: tuple[str, ...] = ("PARTY",)) -> Document:
    """Write a fraction of the ``labels`` spans in capitals, keeping every offset.

    Contracts name their parties in capitals about as often as in title case
    ("TELCOSTAR PTE, LTD."), but the two are not equally easy for a cased
    encoder: on the validation split the shipped model finds 83 % of the
    title-case names and 66 % of the capitalised ones. Capitals are the same
    length as the original, so the surface can be swapped in place and no offset
    moves.
    """
    if prob <= 0:
        return doc
    text = list(doc.text)
    entities = []
    for e in doc.entities:
        surface = e.text
        if e.label in labels and surface != surface.upper() and rng.random() < prob:
            surface = surface.upper()
            text[e.start:e.end] = surface
        entities.append(Span(e.start, e.end, e.label, surface))
    return Document(doc.id, "".join(text), entities, doc.lang, doc.source, doc.meta)


def prepare_training_documents(docs: list[Document], label_map: dict[str, str] | None,
                               date_augment: float, rng: random.Random,
                               dateless_labels_to_drop: tuple[str, ...] | list[str] | None = None,
                               party_case_augment: float = 0.0) -> list[Document]:
    """Apply every training-time transform, in the order the configuration implies."""
    prepared = []
    for d in docs:
        d = map_labels(d, label_map)
        d = drop_dateless_spans(d, dateless_labels_to_drop)
        d = rewrite_dates(d, date_augment, rng)
        d = uppercase_party_names(d, party_case_augment, rng)
        prepared.append(d)
    return prepared
