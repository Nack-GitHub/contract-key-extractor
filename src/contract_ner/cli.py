"""``contract-ner`` command line interface."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _cmd_download(args: argparse.Namespace) -> None:
    import shutil

    import kagglehub

    src = Path(kagglehub.dataset_download("changethan/legal-contract-dataset"))
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    print(f"copied {[p.name for p in dst.iterdir()]} → {dst}")


def _require_thai() -> None:
    import importlib.util

    if importlib.util.find_spec("pythainlp") is None:
        sys.exit('the Thai track needs the "thai" extra: pip install -e ".[thai]"')


def _cmd_build_thai(args: argparse.Namespace) -> None:
    _require_thai()
    from .data.thai_gold import load_gold_dir
    from .data.thai_synthetic import generate
    from .io import write_jsonl

    out = Path(args.out)
    train = generate(args.n_train, seed=args.seed, prefix="th-train")
    val = generate(args.n_val, seed=args.seed + 1, prefix="th-val")
    test_syn = generate(args.n_test, seed=args.seed + 2, prefix="th-test")
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    write_jsonl(test_syn, out / "test_synthetic.jsonl")
    gold = load_gold_dir(args.gold_dir)
    write_jsonl(gold, out / "test_gold.jsonl")
    blind = load_gold_dir(args.blind_dir) if Path(args.blind_dir).exists() else []
    write_jsonl(blind, out / "test_gold_blind.jsonl")
    print(f"thai: train={len(train)} val={len(val)} test_synthetic={len(test_syn)} test_gold={len(gold)} "
          f"test_gold_blind={len(blind)} → {out}")


def _cmd_build_cuad(args: argparse.Namespace) -> None:
    from .data.cuad import build_documents, split_documents
    from .io import write_jsonl

    docs, stats = build_documents(args.raw)
    train, val, test = split_documents(docs)
    out = Path(args.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    write_jsonl(test, out / "test.jsonl")
    print(f"cuad: {len(docs)} contracts, spans aligned {stats.spans_aligned}/{stats.spans_total} "
          f"({stats.rate():.1%}); per type {stats.per_type}")
    print(f"split train={len(train)} val={len(val)} test={len(test)} → {out}")


def _cmd_train(args: argparse.Namespace) -> None:
    from .train import TrainConfig, train

    cfg = TrainConfig.load(args.config)
    if args.max_train_docs:
        cfg.max_train_docs = args.max_train_docs
    if args.epochs is not None:
        cfg.epochs = args.epochs
    train(cfg)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_documents, format_result, save_result
    from .io import read_jsonl
    from .predict import Extractor

    ex = Extractor(args.model)
    for path in args.data:
        docs = read_jsonl(path)
        if args.limit:
            docs = docs[: args.limit]
        res = evaluate_documents(ex, docs, validate=not args.no_validate)
        print(format_result(res, title=str(path) + ("" if not args.no_validate else " [raw tagger, no validation]")))
        if args.out:
            name = Path(path).stem
            save_result(res, Path(args.out) / f"{Path(args.model).name}__{name}.json")


def _cmd_extract(args: argparse.Namespace) -> None:
    from .postprocess import assemble
    from .predict import Extractor

    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    ex = Extractor(args.model)
    spans = ex.predict_with_confidence(text)
    rec = assemble(text, spans, lang=ex.lang)
    if args.summary:
        print(json.dumps(rec["summary"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def _cmd_export_conll(args: argparse.Namespace) -> None:
    from .io import read_jsonl
    from .schema import spans_to_bio
    from .tokenization import segment_words

    docs = read_jsonl(args.data)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            offsets = segment_words(d.text, d.lang)
            tags = spans_to_bio(offsets, d.entities)
            f.write(f"# id = {d.id}\n")
            f.writelines(f"{d.text[s:e]}\t{t}\n" for (s, e), t in zip(offsets, tags))
            f.write("\n")
    print(f"wrote {len(docs)} docs → {args.out}")


def _cmd_sample(args: argparse.Namespace) -> None:
    _require_thai()
    from .data.thai_synthetic import generate

    for d in generate(args.n, seed=args.seed):
        print("=" * 80)
        print(d.text)
        print("-- entities:")
        for e in d.entities:
            print(f"   {e.label:15s} {e.text}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="contract-ner", description="Contract key-information extraction (NER)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download the Kaggle legal-contract dataset (anonymous access)")
    s.add_argument("--out", default="data/raw/kaggle")
    s.set_defaults(fn=_cmd_download)

    s = sub.add_parser("build-thai", help="generate synthetic Thai training data + load the gold set")
    s.add_argument("--out", default="data/processed/thai")
    s.add_argument("--gold-dir", default="data/thai_gold", help="hand-annotated dev/test contracts")
    s.add_argument("--blind-dir", default="data/thai_gold_blind", help="held-out contracts never used for iteration")
    s.add_argument("--n-train", type=int, default=3000)
    s.add_argument("--n-val", type=int, default=300)
    s.add_argument("--n-test", type=int, default=300)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=_cmd_build_thai)

    s = sub.add_parser("build-cuad", help="align CUAD clause spans into full text → BIO documents")
    s.add_argument("--raw", default="data/raw/kaggle")
    s.add_argument("--out", default="data/processed/cuad")
    s.set_defaults(fn=_cmd_build_cuad)

    s = sub.add_parser("train", help="fine-tune a token classifier from a YAML config")
    s.add_argument("--config", required=True)
    s.add_argument("--max-train-docs", type=int, default=None)
    s.add_argument("--epochs", type=float, default=None)
    s.set_defaults(fn=_cmd_train)

    s = sub.add_parser("evaluate", help="document-level P/R/F1 per entity type")
    s.add_argument("--model", required=True)
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--out", default="reports")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--no-validate", action="store_true", help="score raw tagger output without span validation")
    s.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("extract", help="extract structured metadata from a contract text file")
    s.add_argument("--model", required=True)
    s.add_argument("--file", help="text file (reads stdin when omitted)")
    s.add_argument("--out", help="write the full JSON record here")
    s.add_argument("--summary", action="store_true", help="print only the one-line-per-field summary")
    s.set_defaults(fn=_cmd_extract)

    s = sub.add_parser("export-conll", help="dump a JSONL dataset as word/BIO CoNLL for inspection")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_cmd_export_conll)

    s = sub.add_parser("sample", help="print a few generated Thai contracts with their annotations")
    s.add_argument("-n", type=int, default=2)
    s.add_argument("--seed", type=int, default=random.randint(0, 10**6))
    s.set_defaults(fn=_cmd_sample)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
