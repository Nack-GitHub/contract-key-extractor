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


def _profile_encoder(args: argparse.Namespace, lang: str, types):
    """The encoder the training config would use, so the profile can measure its windows.

    Returns ``None`` when no config was given or the tokenizer cannot be loaded
    (no network and nothing cached) — the rest of the profile does not need it.
    """
    if not args.config or args.no_windows:
        return None
    from transformers import AutoTokenizer

    from .schema import build_bio_labels
    from .tokenization import Encoder
    from .train import TrainConfig

    cfg = TrainConfig.load(args.config)
    labels = build_bio_labels(types or cfg.types)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    except Exception as exc:  # the profile is still useful without the windowing section
        print(f"[profile] windowing section skipped: could not load {cfg.model_name} ({exc})")
        return None
    return Encoder(tokenizer, max_length=cfg.max_length, stride=cfg.stride, lowercase=cfg.lowercase,
                   per_word=cfg.per_word, label2id={l: i for i, l in enumerate(labels)})


def _cmd_profile(args: argparse.Namespace) -> None:
    """Label-distribution report for a dataset, before anything is trained on it."""
    from .io import read_jsonl
    from .profile import build_report, profile_documents, save_report

    paths: list[str] = list(args.data)
    types = None
    lang = args.lang
    if args.config:
        from .train import TrainConfig

        cfg = TrainConfig.load(args.config)
        paths = [cfg.train, cfg.val, *cfg.test] + paths
        types = cfg.types
        lang = lang or cfg.lang
    if not paths:
        raise SystemExit("give --config or --data")

    splits = []
    windowing = {}
    encoder = _profile_encoder(args, lang, types)
    for path in paths:
        docs = read_jsonl(path)
        if args.limit:
            docs = docs[: args.limit]
        splits.append(profile_documents(docs, path=path, lang=lang, types=types,
                                        examples_per_form=args.examples))
        if encoder is not None:
            from .profile import window_diagnostics

            windowing[splits[-1].name] = window_diagnostics(docs, encoder, lang)
    print(build_report(splits, reference=splits[0].name, windowing=windowing))
    if args.out:
        json_path, markdown_path = save_report(splits, args.out, reference=splits[0].name, stem=args.stem,
                                               windowing=windowing)
        print(f"\nwrote {json_path} and {markdown_path}")


def _cmd_train(args: argparse.Namespace) -> None:
    from .train import TrainConfig, train

    cfg = TrainConfig.load(args.config)
    if args.max_train_docs:
        cfg.max_train_docs = args.max_train_docs
    if args.epochs is not None:
        cfg.epochs = args.epochs
    train(cfg)


def _load_extractor(model: str, min_confidence: float = 0.6):
    """A model directory, or ``regex`` for the pattern-only English baseline.

    A directory holding ``crf.joblib`` is loaded as the conditional random field
    baseline; anything else as a fine-tuned encoder.
    """
    if model == "regex":
        from .baseline import RegexExtractor
        return RegexExtractor()
    if (Path(model) / "crf.joblib").exists():
        from .baseline_crf import CrfExtractor
        return CrfExtractor(model)
    from .predict import Extractor
    return Extractor(model, min_confidence=min_confidence)


def _cmd_train_crf(args: argparse.Namespace) -> None:
    """Fit the conditional random field baseline on the same split as the encoder."""
    from .baseline_crf import train_crf
    from .io import read_jsonl
    from .train import TrainConfig

    cfg = TrainConfig.load(args.config)
    docs = read_jsonl(cfg.train)
    if args.max_train_docs:
        docs = docs[: args.max_train_docs]
    settings = train_crf(docs, args.out, lang=cfg.lang, entity_types=cfg.types, name=args.name,
                         max_iterations=args.max_iterations, seed=cfg.seed)
    print(f"wrote {args.out}: {settings['train_sequences']} training sequences, "
          f"{len(settings['entity_types'])} entity types")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate_documents, format_result, save_result
    from .io import read_jsonl

    ex = _load_extractor(args.model, args.min_confidence)
    for path in args.data:
        docs = read_jsonl(path)
        if args.limit:
            docs = docs[: args.limit]
        res = evaluate_documents(ex, docs, validate=not args.no_validate)
        print(format_result(res, title=str(path) + ("" if not args.no_validate else " [raw tagger, no validation]")))
        if args.out:
            name = Path(path).stem
            model_name = getattr(ex, "name", None) or Path(args.model).name
            save_result(res, Path(args.out) / f"{model_name}__{name}.json")


def _cmd_extract(args: argparse.Namespace) -> None:
    from .postprocess import assemble

    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    ex = _load_extractor(args.model, args.min_confidence)
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

    s = sub.add_parser("profile", help="label distribution of a dataset (counts, span shapes, surface forms)")
    s.add_argument("--config", help="training config; profiles its train / val / test files")
    s.add_argument("--data", nargs="*", default=[], help="extra JSONL datasets to profile")
    s.add_argument("--out", default="reports/profile", help="directory for the JSON and Markdown report")
    s.add_argument("--stem", default="label_distribution", help="base name of the written report files")
    s.add_argument("--lang", default=None, help="override the language of the documents")
    s.add_argument("--limit", type=int, default=None, help="profile only the first N documents of each file")
    s.add_argument("--examples", type=int, default=1, help="example spans to show per surface form")
    s.add_argument("--no-windows", action="store_true",
                   help="skip the windowing section (it loads the config's tokenizer)")
    s.set_defaults(fn=_cmd_profile)

    s = sub.add_parser("train", help="fine-tune a token classifier from a YAML config")
    s.add_argument("--config", required=True)
    s.add_argument("--max-train-docs", type=int, default=None)
    s.add_argument("--epochs", type=float, default=None)
    s.set_defaults(fn=_cmd_train)

    s = sub.add_parser("train-crf", help="fit the conditional random field baseline on a config's training split")
    s.add_argument("--config", required=True, help="training config; its train file, language and entity types are used")
    s.add_argument("--out", default="models/english-cuad-crf", help="directory for the fitted model")
    s.add_argument("--name", default="crf-baseline", help="model name recorded in the evaluation reports")
    s.add_argument("--max-iterations", type=int, default=120)
    s.add_argument("--max-train-docs", type=int, default=None)
    s.set_defaults(fn=_cmd_train_crf)

    s = sub.add_parser("evaluate", help="document-level P/R/F1 per entity type")
    s.add_argument("--model", required=True, help="model directory, or 'regex' for the pattern-only English baseline")
    s.add_argument("--data", nargs="+", required=True)
    s.add_argument("--out", default="reports")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--no-validate", action="store_true", help="score raw tagger output without span validation")
    s.add_argument("--min-confidence", type=float, default=0.6, help="drop spans below this mean label probability")
    s.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("extract", help="extract structured metadata from a contract text file")
    s.add_argument("--model", required=True, help="model directory, or 'regex' for the pattern-only English baseline")
    s.add_argument("--file", help="text file (reads stdin when omitted)")
    s.add_argument("--out", help="write the full JSON record here")
    s.add_argument("--summary", action="store_true", help="print only the one-line-per-field summary")
    s.add_argument("--min-confidence", type=float, default=0.6, help="drop spans below this mean label probability")
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
