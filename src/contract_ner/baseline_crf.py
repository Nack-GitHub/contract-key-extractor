"""Conditional random field baseline: the classical sequence-labelling approach.

The regex baseline in :mod:`contract_ner.baseline` says how far hand-written
patterns go. This one says how far a statistical sequence model goes when it is
trained on the same annotation but has no pre-trained language representation:
a linear-chain conditional random field over hand-designed features (word shape,
capitalisation, suffixes, a small legal-vocabulary gazetteer, neighbouring words
and where the word sits in the contract). It was the standard approach to named
entity recognition before pre-trained encoders, so the difference between it and
the fine-tuned model is the part of the score that comes from pre-training
rather than from the annotation or the post-processing.

It shares the extractor interface (``lang``, ``entity_types``, ``predict`` /
``predict_with_confidence``), so the same evaluation, span validation and
metadata assembly run on top of it unchanged.

Training data comes from the same JSONL split as the encoder's; documents are
cut into sentences so a sequence stays a manageable length. Install with the
``baseline`` extra (``pip install -e ".[baseline]"``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .lang import rules_for
from .schema import ENTITY_TYPES, O_LABEL, Document, Span, spans_to_bio

# A sequence is one sentence: inference is linear in length, but whole contracts make training slow
# and every feature here is local anyway. The split must not fall inside a party name, so the full
# stop of a legal-form abbreviation ("Acme Holdings, Inc. and Beta Ltd.") does not end a sentence.
_SENTENCE_BREAK_RE = re.compile(r"^[.;:!?]$")
_ABBREVIATION = {
    "inc.", "ltd.", "co.", "corp.", "no.", "mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.",
    "l.p.", "l.l.c.", "s.a.", "n.v.", "b.v.", "pte.", "pty.",
}
_LEGAL_SUFFIX = {
    "inc", "inc.", "incorporated", "ltd", "ltd.", "limited", "llc", "corp", "corp.", "corporation",
    "company", "co", "co.", "plc", "gmbh", "ag", "lp", "llp", "pte", "pty", "nv", "bv", "sa",
}
_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept",
    "sep", "oct", "nov", "dec",
}
_LEGAL_CUE = {
    "agreement", "contract", "party", "parties", "between", "among", "dated", "effective",
    "governed", "construed", "laws", "law", "jurisdiction", "court", "arbitration", "term",
    "terminate", "terminates", "expire", "expires", "years", "year", "months", "month", "day",
    "days", "witnesseth", "hereby", "shall",
}


def _shape(word: str) -> str:
    """``"Acme"`` → ``"Xxxx"``, ``"1999"`` → ``"dddd"`` — collapsed to at most six characters."""
    shape = re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"\d", "d", word)))
    return re.sub(r"(.)\1{2,}", r"\1\1\1", shape)[:6]


def word_features(words: list[str], i: int, position: float) -> dict:
    """Features for one word: its own form, its neighbours and where it sits in the contract."""
    word = words[i]
    lower = word.lower()
    features = {
        "bias": 1.0,
        "word.lower": lower,
        "word.shape": _shape(word),
        "word.suffix3": lower[-3:],
        "word.prefix2": lower[:2],
        "word.is_upper": word.isupper(),
        "word.is_title": word.istitle(),
        "word.is_digit": word.isdigit(),
        "word.has_digit": any(c.isdigit() for c in word),
        "word.length": min(len(word), 12),
        "word.is_legal_suffix": lower in _LEGAL_SUFFIX,
        "word.is_month": lower in _MONTHS,
        "word.is_legal_cue": lower in _LEGAL_CUE,
        "word.is_punct": not any(c.isalnum() for c in word),
        "word.is_year": bool(re.fullmatch(r"(?:19|20)\d{2}", word)),
        # the label distribution is strongly positional: parties and the agreement date sit in the
        # opening, the governing-law clause almost always in the second half
        "doc.position": round(position, 1),
        "doc.in_opening": position < 0.05,
    }
    for offset in (-2, -1, 1, 2):
        neighbour_index = i + offset
        prefix = f"{offset:+d}"
        if 0 <= neighbour_index < len(words):
            neighbour = words[neighbour_index]
            features[f"{prefix}:word.lower"] = neighbour.lower()
            features[f"{prefix}:word.shape"] = _shape(neighbour)
            features[f"{prefix}:word.is_title"] = neighbour.istitle()
            features[f"{prefix}:word.is_legal_cue"] = neighbour.lower() in _LEGAL_CUE
            features[f"{prefix}:word.is_legal_suffix"] = neighbour.lower() in _LEGAL_SUFFIX
        else:
            features[f"{prefix}:boundary"] = True
    return features


@dataclass
class Sequence:
    """One sentence: the words' character offsets, their features and their tags."""

    offsets: list[tuple[int, int]]
    features: list[dict]
    tags: list[str]


def document_to_sequences(doc_text: str, offsets: list[tuple[int, int]], tags: list[str]) -> list[Sequence]:
    """Split a document's words into sentence-sized sequences, keeping offsets and tags aligned."""
    words = [doc_text[s:e] for s, e in offsets]
    total = max(1, len(doc_text))
    breaks = [0]
    for i, (start, end) in enumerate(offsets[:-1]):
        word = doc_text[start:end]
        between = doc_text[end:offsets[i + 1][0]]
        next_word = doc_text[offsets[i + 1][0]:offsets[i + 1][1]]
        blank_line = "\n" in between and between.count("\n") > 1
        ends_sentence = (
            _SENTENCE_BREAK_RE.match(word)
            or (word.endswith(".") and word.lower() not in _ABBREVIATION and len(word) > 2)
        )
        # a full stop only ends a sentence when what follows starts one
        if blank_line or (ends_sentence and between and (next_word[:1].isupper() or "\n" in between)):
            breaks.append(i + 1)
    breaks.append(len(offsets))
    sequences = []
    for a, b in zip(breaks, breaks[1:]):
        if a == b:
            continue
        chunk = words[a:b]
        sequences.append(Sequence(
            offsets=offsets[a:b],
            features=[word_features(chunk, i, offsets[a + i][0] / total) for i in range(len(chunk))],
            tags=tags[a:b],
        ))
    return sequences


def bio_to_spans(text: str, offsets: list[tuple[int, int]], tags: list[str]) -> list[Span]:
    """Decode word-level tags into character spans (a stray ``I-`` opens a span, as in the neural path)."""
    spans: list[Span] = []
    start = end = -1
    label = ""
    for (word_start, word_end), tag in zip(offsets, tags):
        if tag == O_LABEL:
            if label:
                spans.append(Span(start, end, label, text[start:end]))
                label = ""
            continue
        prefix, _, tag_label = tag.partition("-")
        if label and (prefix == "B" or tag_label != label):
            spans.append(Span(start, end, label, text[start:end]))
            label = ""
        if not label:
            start, label = word_start, tag_label
        end = word_end
    if label:
        spans.append(Span(start, end, label, text[start:end]))
    return spans


class CrfExtractor:
    """Trained conditional random field with the same interface as the neural extractor."""

    def __init__(self, model_dir: str | Path):
        import joblib

        model_dir = Path(model_dir)
        settings = json.loads((model_dir / "pipeline_config.json").read_text(encoding="utf-8"))
        self.crf = joblib.load(model_dir / "crf.joblib")
        self.lang: str = settings["lang"]
        self.name: str = settings["name"]
        self.entity_types: tuple[str, ...] = tuple(settings["entity_types"])
        self.label_map: dict[str, str] = settings.get("label_map") or {}

    def predict_with_confidence(self, text: str) -> list[tuple[Span, float]]:
        offsets = rules_for(self.lang).segment_words(text)
        if not offsets:
            return []
        sequences = document_to_sequences(text, offsets, [O_LABEL] * len(offsets))
        tags: list[str] = []
        confidences: list[float] = []
        for sequence in sequences:
            marginals = self.crf.predict_marginals_single(sequence.features)
            predicted = self.crf.predict_single(sequence.features)
            tags.extend(predicted)
            confidences.extend(marginals[i][tag] for i, tag in enumerate(predicted))
        spans = bio_to_spans(text, offsets, tags)
        # a span's confidence is the mean marginal probability of its words' tags, as for the encoder
        out: list[tuple[Span, float]] = []
        for span in spans:
            probabilities = [c for (start, end), c in zip(offsets, confidences)
                             if start >= span.start and end <= span.end]
            out.append((span, sum(probabilities) / len(probabilities) if probabilities else 0.0))
        return out

    def predict(self, text: str) -> list[Span]:
        return [span for span, _ in self.predict_with_confidence(text)]


def train_crf(train_docs: list[Document], out_dir: str | Path, lang: str = "en",
              entity_types: tuple[str, ...] = ENTITY_TYPES, name: str = "crf-baseline",
              max_iterations: int = 120, c1: float = 0.1, c2: float = 0.1,
              keep_empty_sequence_share: float = 0.15, seed: int = 42) -> dict:
    """Fit the conditional random field and write it next to a pipeline configuration.

    Entity-free sequences are sampled down the same way entity-free windows are
    for the encoder, so both models see a comparable balance of positives.
    """
    import random

    import joblib
    import sklearn_crfsuite

    rng = random.Random(seed)
    segment = rules_for(lang).segment_words
    features: list[list[dict]] = []
    labels: list[list[str]] = []
    for doc in train_docs:
        offsets = segment(doc.text)
        spans = [s for s in doc.entities if s.label in entity_types]
        tags = spans_to_bio(offsets, spans)
        for sequence in document_to_sequences(doc.text, offsets, tags):
            if all(t == O_LABEL for t in sequence.tags) and rng.random() > keep_empty_sequence_share:
                continue
            features.append(sequence.features)
            labels.append(sequence.tags)

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations,
                              all_possible_transitions=True)
    crf.fit(features, labels)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(crf, out_dir / "crf.joblib")
    settings = {"lang": lang, "name": name, "entity_types": list(entity_types), "model": "crf",
                "max_iterations": max_iterations, "c1": c1, "c2": c2,
                "keep_empty_sequence_share": keep_empty_sequence_share,
                "train_sequences": len(features)}
    (out_dir / "pipeline_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings
