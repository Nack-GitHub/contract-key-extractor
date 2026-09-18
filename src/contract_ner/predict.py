"""Run a fine-tuned model over a whole contract and decode entity spans.

Long contracts are split into overlapping windows; for every subword we keep
the logits from the window in which it sits furthest from the edges (where the
model has the most context), then read one prediction per word from its
representative subword and decode BIO → character spans.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .schema import ID2LABEL, O_LABEL, Span, bio_to_spans
from .tokenization import Encoder, EncodedDocument, segment_words


@dataclass
class WordPrediction:
    offsets: list[tuple[int, int]]
    tags: list[str]
    confidence: list[float]


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Extractor:
    """Text → entity spans, using a model directory produced by ``train``."""

    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 16):
        model_dir = Path(model_dir)
        with open(model_dir / "pipeline_config.json", encoding="utf-8") as f:
            pc = json.load(f)
        self.lang: str = pc["lang"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.device = pick_device(device)
        self.model.to(self.device).eval()
        self.encoder = Encoder(self.tokenizer, max_length=pc["max_length"], stride=pc["stride"],
                               lowercase=pc["lowercase"])
        self.batch_size = batch_size
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()} or ID2LABEL

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def _subword_logits(self, enc: EncodedDocument) -> np.ndarray:
        """Best-context logits for every global subword, shape (n_subwords, n_labels)."""
        n_sub = len(enc.subword_to_word)
        n_lab = self.model.config.num_labels
        logits = np.zeros((n_sub, n_lab), dtype=np.float32)
        best_margin = np.full(n_sub, -1, dtype=np.int32)
        windows = enc.windows
        for b in range(0, len(windows), self.batch_size):
            batch = windows[b : b + self.batch_size]
            max_len = max(len(w.input_ids) for w in batch)
            pad = self.tokenizer.pad_token_id or 0
            ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, w in enumerate(batch):
                ids[i, : len(w.input_ids)] = torch.tensor(w.input_ids)
                mask[i, : len(w.input_ids)] = 1
            out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
            out = out.float().cpu().numpy()
            for i, w in enumerate(batch):
                L = len(w.input_ids)
                for pos, g in enumerate(w.subword_index):
                    if g < 0:
                        continue
                    margin = min(pos, L - 1 - pos)
                    if margin > best_margin[g]:
                        best_margin[g] = margin
                        logits[g] = out[i, pos]
        return logits

    def predict_words(self, text: str) -> WordPrediction:
        offsets = segment_words(text, self.lang)
        if not offsets:
            return WordPrediction([], [], [])
        enc = self.encoder.encode(text, offsets)
        logits = self._subword_logits(enc)
        probs = _softmax(logits)
        pred_ids = probs.argmax(-1)

        tags: list[str] = []
        conf: list[float] = []
        used: set[int] = set()
        sub_starts = self._subword_starts(enc, text)
        for w, (ws, we) in enumerate(offsets):
            s = enc.word_to_subword[w]
            if s < 0:
                s = _covering_subword(sub_starts, ws)
            if s < 0:
                tags.append(O_LABEL)
                conf.append(1.0)
                continue
            tag = self.id2label[int(pred_ids[s])]
            if s in used and tag.startswith("B-"):
                tag = "I-" + tag[2:]  # this word shares a subword with the previous word
            used.add(s)
            tags.append(tag)
            conf.append(float(probs[s, pred_ids[s]]))
        return WordPrediction(offsets, tags, conf)

    def _subword_starts(self, enc: EncodedDocument, text: str) -> list[tuple[int, int]]:
        # re-derive subword char ranges from the tokenizer (cheap, keeps Encoder lean)
        model_text = text
        if self.encoder.lowercase:
            from .tokenization import lowercase_ascii
            model_text = lowercase_ascii(text)
        e = self.tokenizer(model_text, add_special_tokens=False, return_offsets_mapping=True)
        return e["offset_mapping"]

    def predict(self, text: str) -> list[Span]:
        wp = self.predict_words(text)
        spans = bio_to_spans(wp.tags, wp.offsets, text)
        # attach mean word confidence to each span via a parallel list
        conf_by_start = {}
        i = 0
        for sp in spans:
            words = [k for k, (ws, we) in enumerate(wp.offsets) if ws >= sp.start and we <= sp.end]
            conf_by_start[sp.start] = float(np.mean([wp.confidence[k] for k in words])) if words else 0.0
        self._last_confidence = conf_by_start
        return spans

    def predict_with_confidence(self, text: str) -> list[tuple[Span, float]]:
        spans = self.predict(text)
        return [(sp, self._last_confidence.get(sp.start, 0.0)) for sp in spans]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _covering_subword(sub_offsets: list[tuple[int, int]], char: int) -> int:
    best = -1
    for i, (s, e) in enumerate(sub_offsets):
        if s <= char < e:
            # prefer a content subword over a bare marker sharing the same start
            if best < 0 or (e - s) > (sub_offsets[best][1] - sub_offsets[best][0]):
                best = i
        elif s > char:
            break
    return best
