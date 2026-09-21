"""Fine-tune a token-classification encoder on BIO-tagged contract windows.

Driven by a YAML config (see ``configs/``). The same script trains the Thai
WangchanBERTa model on synthetic data and the English model on CUAD; only the
config differs. Metrics during training are window-level seqeval scores on the
validation set (fast, good enough for model selection); the document-level
evaluation with stitched windows lives in :mod:`contract_ner.evaluate`.
"""
from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .io import read_jsonl
from .schema import ENTITY_TYPES, O_LABEL, Document, build_bio_labels, spans_to_bio
from .tokenization import Encoder, segment_words


@dataclass
class TrainConfig:
    name: str
    model_name: str
    lang: str
    train: str
    val: str
    output_dir: str
    test: list[str] = field(default_factory=list)
    entity_types: list[str] | None = None  # subset of schema.ENTITY_TYPES the model is trained on (None = all)
    lowercase: bool = False
    per_word: bool = True  # tokenize word by word (no piece crosses a word boundary); see tokenization.Encoder
    max_length: int = 256
    stride: int = 64
    epochs: float = 3
    lr: float = 3e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    negative_window_keep: float = 1.0  # fraction of entity-free windows kept for training
    max_train_docs: int | None = None
    grad_accum: int = 1

    def __post_init__(self) -> None:
        if self.entity_types is not None:
            unknown = [t for t in self.entity_types if t not in ENTITY_TYPES]
            if unknown:
                raise ValueError(f"unknown entity types in config: {unknown}")
            if not self.entity_types:
                raise ValueError("entity_types must list at least one type (or be omitted)")

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(self.entity_types) if self.entity_types else ENTITY_TYPES

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)


# ------------------------------------------------------------------ features
def documents_to_windows(docs: list[Document], encoder: Encoder, lang: str,
                         negative_keep: float = 1.0, rng: random.Random | None = None) -> list[dict]:
    """Encode documents into training windows; spans of types outside the encoder's label space are ignored."""
    rng = rng or random.Random(0)
    o_id = encoder.label2id[O_LABEL]
    feats: list[dict] = []
    for d in docs:
        offsets = segment_words(d.text, lang)
        spans = [sp for sp in d.entities if f"B-{sp.label}" in encoder.label2id]
        tags = spans_to_bio(offsets, spans)
        enc = encoder.encode(d.text, offsets, tags)
        for w in enc.windows:
            has_entity = any(l not in (-100, o_id) for l in w.labels)
            if not has_entity and rng.random() > negative_keep:
                continue
            feats.append({"input_ids": w.input_ids, "attention_mask": w.attention_mask, "labels": w.labels})
    return feats


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, feats: list[dict]):
        self.feats = feats

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, i: int) -> dict:
        return self.feats[i]


def make_seqeval_metrics(id2label: dict[int, str]):
    def _seqeval_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true, y_pred = [], []
        for p_row, l_row in zip(preds, labels):
            t, p = [], []
            for pi, li in zip(p_row, l_row):
                if li == -100:
                    continue
                t.append(id2label[int(li)])
                p.append(id2label[int(pi)])
            y_true.append(t)
            y_pred.append(p)
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }

    return _seqeval_metrics


# ------------------------------------------------------------------ main entry
def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = build_bio_labels(cfg.types)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    encoder = Encoder(tokenizer, max_length=cfg.max_length, stride=cfg.stride, lowercase=cfg.lowercase,
                      per_word=cfg.per_word, label2id=label2id)

    train_docs = read_jsonl(cfg.train)
    if cfg.max_train_docs:
        train_docs = train_docs[: cfg.max_train_docs]
    val_docs = read_jsonl(cfg.val)
    rng = random.Random(cfg.seed)
    train_feats = documents_to_windows(train_docs, encoder, cfg.lang, cfg.negative_window_keep, rng)
    val_feats = documents_to_windows(val_docs, encoder, cfg.lang, 1.0)
    print(f"[{cfg.name}] train docs={len(train_docs)} windows={len(train_feats)} | "
          f"val docs={len(val_docs)} windows={len(val_feats)} | labels={len(labels)} ({', '.join(cfg.types)})")

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )

    steps_per_epoch = math.ceil(len(train_feats) / (cfg.batch_size * cfg.grad_accum))
    warmup_steps = int(cfg.warmup_ratio * steps_per_epoch * cfg.epochs)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
        dataloader_num_workers=0,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=WindowDataset(train_feats),
        eval_dataset=WindowDataset(val_feats),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_seqeval_metrics(id2label),
    )
    trainer.train()
    final_eval = trainer.evaluate()
    print(f"[{cfg.name}] best val (window-level): {final_eval}")

    # persist the final model + the pipeline settings the extractor needs
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    with open(out / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump({"lang": cfg.lang, "lowercase": cfg.lowercase, "per_word": cfg.per_word,
                   "max_length": cfg.max_length, "stride": cfg.stride, "base_model": cfg.model_name,
                   "name": cfg.name, "entity_types": list(cfg.types)}, f, indent=2)
    history = [h for h in trainer.state.log_history]
    with open(out / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg.__dict__, "final_eval": final_eval, "log_history": history}, f, indent=2)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)  # free disk; the final model is saved above
    return final_eval
