"""Word segmentation and subword alignment.

Two layers of tokenisation are involved:

1. **Words** – the unit at which BIO tags live and at which we evaluate.
   Thai has no spaces, so words come from PyThaiNLP's ``newmm`` dictionary
   segmenter (the approach described in the project deck). English words come
   from a small regex.
2. **Subwords** – what the transformer consumes. We feed the *raw text* to the
   HF tokenizer and align subwords back to words through character offsets.
   Feeding the pre-segmented Thai words with ``is_split_into_words=True`` would
   insert an artificial ``▁`` marker in front of every Thai word (≈1.6× longer
   sequences that look nothing like the pre-training data), so offsets are the
   better bridge.

Every subword is mapped to the word containing its first character; the first
"real" subword of a word is the word's representative (it carries the label
during training and the prediction at inference). Long documents are cut into
overlapping windows of subwords so that nothing is truncated.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from pythainlp.tokenize import word_tokenize

from .schema import LABEL2ID, O_LABEL

_EN_WORD_RE = re.compile(r"\w+(?:[-'’.]\w+)*|[^\w\s]", re.UNICODE)
_ASCII_UPPER_RE = re.compile(r"[A-Z]+")


def lowercase_ascii(text: str) -> str:
    """Lower-case only ASCII letters, keeping the string length unchanged.

    ``str.lower`` can change the length of some non-ASCII characters, which
    would break character-offset alignment; contract text only needs the ASCII
    letters folded for the *uncased* WangchanBERTa vocabulary.
    """
    return _ASCII_UPPER_RE.sub(lambda m: m.group().lower(), text)


def segment_words(text: str, lang: str = "th") -> list[tuple[int, int]]:
    """Return ``(start, end)`` character offsets of the words in ``text``.

    Whitespace is never a word. Thai text is segmented with PyThaiNLP newmm;
    Latin runs inside Thai text are kept as whole tokens by newmm itself.
    """
    if lang == "th":
        return _segment_thai(text)
    if lang == "en":
        return [(m.start(), m.end()) for m in _EN_WORD_RE.finditer(text)]
    raise ValueError(f"unsupported lang {lang!r}")


def _segment_thai(text: str) -> list[tuple[int, int]]:
    toks = word_tokenize(text, engine="newmm", keep_whitespace=True)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for tok in toks:
        if not tok:
            continue
        # newmm preserves the surface string, so tokens tile the text in order.
        idx = text.find(tok, pos)
        if idx < 0:  # should not happen; fall back to sequential assumption
            idx = pos
        end = idx + len(tok)
        if not tok.isspace():
            offsets.append((idx, end))
        pos = end
    return offsets


@dataclass
class Window:
    """One model input: a slice of the document's subword sequence."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]  # -100 where no loss is computed
    # global subword indices (into the un-windowed sequence) for each position,
    # -1 for special tokens. Used to stitch predictions back together.
    subword_index: list[int]


@dataclass
class EncodedDocument:
    """A document tokenised into subwords with word alignment metadata."""

    word_offsets: list[tuple[int, int]]
    word_tags: list[str]
    # per subword: index of the word it belongs to (-1 = none / whitespace)
    subword_to_word: list[int]
    # per word: index of its representative subword (-1 if none)
    word_to_subword: list[int]
    windows: list[Window] = field(default_factory=list)


class Encoder:
    """Turns (text, word offsets, word tags) into training/inference windows."""

    def __init__(self, tokenizer, max_length: int = 256, stride: int = 64, lowercase: bool = False):
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer (offset mapping support) is required")
        self.tok = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.lowercase = lowercase
        self.prefix_ids, self.suffix_ids = _special_wrapper(tokenizer)
        self.n_special = len(self.prefix_ids) + len(self.suffix_ids)
        self.body_len = max_length - self.n_special
        if self.stride >= self.body_len:
            raise ValueError("stride must be smaller than the usable window length")

    # ------------------------------------------------------------------ encode
    def encode(
        self,
        text: str,
        word_offsets: list[tuple[int, int]],
        word_tags: list[str] | None = None,
    ) -> EncodedDocument:
        if word_tags is None:
            word_tags = [O_LABEL] * len(word_offsets)
        assert len(word_tags) == len(word_offsets)

        model_text = lowercase_ascii(text) if self.lowercase else text
        enc = self.tok(model_text, add_special_tokens=False, return_offsets_mapping=True)
        ids: list[int] = enc["input_ids"]
        offs: list[tuple[int, int]] = enc["offset_mapping"]

        sub2word, word2sub = _align(offs, word_offsets)
        labels = [-100] * len(ids)
        for w, s in enumerate(word2sub):
            if s >= 0:
                labels[s] = LABEL2ID[word_tags[w]]

        doc = EncodedDocument(
            word_offsets=word_offsets,
            word_tags=word_tags,
            subword_to_word=sub2word,
            word_to_subword=word2sub,
        )
        doc.windows = self._window(ids, labels)
        return doc

    def _window(self, ids: list[int], labels: list[int]) -> list[Window]:
        n = len(ids)
        windows: list[Window] = []
        start = 0
        while True:
            end = min(start + self.body_len, n)
            body_ids = ids[start:end]
            body_lab = labels[start:end]
            full_ids = self.prefix_ids + body_ids + self.suffix_ids
            lead, trail = len(self.prefix_ids), len(self.suffix_ids)
            full_lab = [-100] * lead + body_lab + [-100] * trail
            sub_idx = [-1] * lead + list(range(start, end)) + [-1] * trail
            windows.append(
                Window(
                    input_ids=full_ids,
                    attention_mask=[1] * len(full_ids),
                    labels=full_lab,
                    subword_index=sub_idx,
                )
            )
            if end >= n:
                break
            start = end - self.stride
        return windows


def _special_wrapper(tokenizer) -> tuple[list[int], list[int]]:
    """Special-token ids the tokenizer puts before/after a single sequence."""
    body = tokenizer("a", add_special_tokens=False)["input_ids"]
    full = tokenizer("a", add_special_tokens=True)["input_ids"]
    for lead in range(len(full) - len(body) + 1):
        if full[lead : lead + len(body)] == body:
            return full[:lead], full[lead + len(body) :]
    raise RuntimeError("could not locate body tokens inside the special-token wrapper")


def _align(
    sub_offsets: list[tuple[int, int]], word_offsets: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    """Map subwords → words (by first character) and pick representatives.

    A subword whose start offset equals the *next* subword's start offset is a
    bare SentencePiece ``▁`` marker; it is never chosen as a representative so
    that the word's prediction comes from a subword with actual content.
    """
    starts = [s for s, _ in word_offsets]
    ends = [e for _, e in word_offsets]
    sub2word = [-1] * len(sub_offsets)
    word2sub = [-1] * len(word_offsets)
    for i, (s, e) in enumerate(sub_offsets):
        if e <= s:
            continue  # zero-width (special) token
        w = bisect.bisect_right(starts, s) - 1
        if w < 0 or s >= ends[w]:
            continue  # character not inside any word (whitespace)
        sub2word[i] = w
        bare_marker = i + 1 < len(sub_offsets) and sub_offsets[i + 1][0] == s
        if word2sub[w] < 0 and not bare_marker:
            word2sub[w] = i
    # words whose only subword was a bare marker: fall back to that marker
    for i, w in enumerate(sub2word):
        if w >= 0 and word2sub[w] < 0:
            word2sub[w] = i
    return sub2word, word2sub
