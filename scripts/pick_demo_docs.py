#!/usr/bin/env python
"""Choose the contracts the demo shows and store them, with every model's record, in ``demo/samples/``.

Every held-out CUAD test contract is run through the three English models and
scored cell by cell against its gold table (the same keys as the slot-level
evaluation). Contracts are then picked for the moments the demo needs:

* ``cuad-clean``     – the shipped model fills every cell correctly (opening demo);
* ``cuad-parties``   – the model gets the parties right where both baselines do not
  (why pre-training matters);
* ``cuad-end-date``  – the model's end date is wrong (an honest error: the open
  End_Date selection problem);
* ``cuad-gold-gap``  – the model names a governing law the CUAD gold does not have
  (why the CUAD number is a floor);
* ``cuad-long``      – the longest contract of the split (windowing and speed).

The two example contracts in ``examples/`` and one Thai contract from the dev gold set
(never the blind set, which is kept for scoring only) are stored as well. Each
file keeps the record of every model that applies to it, so the server can answer
from it when a model cannot be loaded.

    python scripts/pick_demo_docs.py                       # needs the models in models/
    python scripts/pick_demo_docs.py --pick cuad-clean=cuad-0123456789   # override one choice
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from contract_ner.api import DEMO_MODELS, ModelRegistry, gold_check, run_model
from contract_ner.io import read_jsonl
from contract_ner.schema import Document
from contract_ner.slots import end_date_key

ROOT = Path(__file__).resolve().parents[1]
ENGLISH_MODELS = ("roberta", "crf", "regex")


def run_all(registry: ModelRegistry, doc: Document, model_ids: tuple[str, ...]) -> dict:
    """``{model: {"record", "seconds", "check"}}`` for every model that can run here."""
    entries = {e.id: e for e in DEMO_MODELS}
    out = {}
    for model_id in model_ids:
        if registry.missing_reason(model_id):
            print(f"  {model_id}: skipped ({registry.missing_reason(model_id)})")
            continue
        extractor = registry.get(model_id)
        t0 = time.perf_counter()
        record = run_model(extractor, doc.text, 0.6, entries[model_id].confidence_floor)
        seconds = round(time.perf_counter() - t0, 2)
        check = gold_check(record, doc, registry.label_map(model_id)) if doc.entities else None
        out[model_id] = {"record": record, "seconds": seconds, "check": check}
    return out


def verdicts(result: dict) -> dict[str, str]:
    return {c["slot"]: c["verdict"] for c in result["check"]}


def n_right(result: dict) -> int:
    return sum(v == "correct" for v in verdicts(result).values())


def all_right(result: dict) -> bool:
    return all(v in ("correct", "none") for v in verdicts(result).values())


def end_date_found_but_not_picked(doc: Document, result: dict) -> dict | None:
    """The valid END_DATE span that carries the gold value when the table shows another one."""
    cell = next(c for c in result["check"] if c["slot"] == "End_Date")
    if cell["verdict"] != "wrong":
        return None
    for e in result["record"]["entities"]:
        if e["label"] == "END_DATE" and e["valid"] and end_date_key(e["text"], doc.lang) in cell["gold"]:
            return e
    return None


def describe(results: dict) -> str:
    parts = []
    for model_id, r in results.items():
        v = verdicts(r)
        wrong = [s for s, x in v.items() if x not in ("correct", "none")]
        parts.append(f"{model_id}: {n_right(r)} right" + (f", off on {', '.join(wrong)}" if wrong else ""))
    return " · ".join(parts)


def pick(scored: list[tuple[Document, dict]], overrides: dict[str, str]) -> dict[str, tuple[Document, dict]]:
    by_id = {d.id: (d, r) for d, r in scored}
    medium = [(d, r) for d, r in scored if 8_000 <= len(d.text) <= 60_000]
    chosen: dict[str, tuple[Document, dict]] = {}

    def take(name: str, candidates, key):
        if name in overrides:
            chosen[name] = by_id[overrides[name]]
            return
        used = {d.id for d, _ in chosen.values()}
        candidates = [c for c in candidates if c[0].id not in used]
        if candidates:
            chosen[name] = min(candidates, key=key)

    rob = lambda r: r["roberta"]  # noqa: E731
    # opening demo: every cell right, at least four cells to show, a baseline slips somewhere, not too long
    take("cuad-clean",
         [(d, r) for d, r in medium if all_right(rob(r)) and n_right(rob(r)) >= 4],
         key=lambda c: (all_right(c[1].get("regex", c[1]["roberta"])), -n_right(rob(c[1])), len(c[0].text)))
    take("cuad-parties",
         [(d, r) for d, r in medium if verdicts(rob(r))["Parties"] == "correct"
          and all(verdicts(r[m])["Parties"] != "correct" for m in ("crf", "regex") if m in r)],
         key=lambda c: (-n_right(rob(c[1])), len(c[0].text)))
    # the open End_Date item: the right clause is among the model's spans, the selection takes another
    take("cuad-end-date",
         [(d, r) for d, r in scored if end_date_found_but_not_picked(d, rob(r)) and len(d.text) <= 200_000],
         key=lambda c: (-n_right(rob(c[1])), len(c[0].text)))
    take("cuad-gold-gap",
         [(d, r) for d, r in scored if verdicts(rob(r))["Jurisdiction"] == "extra"],
         key=lambda c: (-n_right(rob(c[1])), len(c[0].text)))
    take("cuad-long", scored, key=lambda c: -len(c[0].text))
    return chosen


TITLES = {
    "en-example": "Service agreement (short example)",
    "cuad-clean": "CUAD contract — every cell right",
    "cuad-parties": "CUAD contract — parties the baselines miss",
    "cuad-end-date": "CUAD contract — an end-date error",
    "cuad-gold-gap": "CUAD contract — a value the gold set lacks",
    "cuad-long": "CUAD contract — the longest in the test split",
    "th-example": "สัญญาจ้างบริการ (ตัวอย่างสั้น)",
    "th-gold": "สัญญาภาษาไทย — มีเฉลย (dev gold)",
}
ORDER = list(TITLES)


def talking_point(name: str, doc: Document, results: dict) -> str:
    chars = f"{len(doc.text):,} characters"
    if name == "en-example":
        return f"A short service agreement ({chars}) to show the table, the highlights and click-to-source."
    if name == "th-example":
        return f"The Thai example contract ({chars}): same API and page, Thai model, nine cells instead of six."
    if name == "th-gold":
        return f"A hand-annotated Thai contract ({chars}) from the dev gold set. {describe(results)}."
    if name == "cuad-long":
        secs = results["roberta"]["seconds"]
        return (f"The longest contract in the test split ({chars}, cut into overlapping 256-subword windows); "
                f"the shipped model took {secs} s when this sample was stored. {describe(results)}.")
    rob = results["roberta"]
    if name == "cuad-clean":
        lead = "Every cell the gold has is right for the shipped model."
    elif name == "cuad-parties":
        gold = next(c for c in rob["check"] if c["slot"] == "Parties")["gold"]
        lead = (f"The fine-tuned model finds both parties ({' / '.join(gold)}); the baselines do not — "
                "telling the defining mention of a name from every later one needs the whole preamble.")
    elif name == "cuad-end-date":
        found = end_date_found_but_not_picked(doc, rob)
        picked = rob["record"]["End_Date"]
        lead = (f"An honest error. The model does tag the right term clause (\"{found['text'][:90]}…\", "
                f"confidence {found['confidence']:.2f}), but the table takes a different, more confident one "
                f"(\"{picked['text'][:90]}…\", {picked['confidence']:.2f}). The fix belongs in the selection "
                "rule, not in training — the open End_Date item.")
    else:
        j = rob["record"].get("Governing_Law") or rob["record"]["Jurisdiction"]
        lead = (f"The model reads a governing-law clause (\"{j['text'][:110]}…\") where the CUAD gold has "
                "none, so it is scored as a false positive. Read the clause: the gold missed it — the CUAD "
                "number is a floor.")
    return f"{lead} ({chars}) {describe(results)}."


def to_sample(name: str, doc: Document, results: dict, source: str) -> dict:
    lang = doc.lang
    return {
        "id": name, "order": ORDER.index(name), "title": TITLES[name], "lang": lang, "source": source,
        "talking_point": talking_point(name, doc, results) if doc.entities or name.endswith("example") else "",
        "default_model": "thai" if lang == "th" else "roberta",
        "text": doc.text,
        "entities": [e.to_dict() for e in doc.entities],
        "cached": {m: {"record": r["record"], "seconds": r["seconds"]} for m, r in results.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--test", default="data/processed/cuad/test.jsonl")
    p.add_argument("--thai-gold", default="data/processed/thai/test_gold.jsonl")
    p.add_argument("--out", default="demo/samples")
    p.add_argument("--pick", nargs="*", default=[], help="name=document_id overrides")
    args = p.parse_args()
    overrides = dict(x.split("=", 1) for x in args.pick)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(ROOT)

    samples = []
    for name, path, lang in (("en-example", "examples/english_service_agreement.txt", "en"),
                             ("th-example", "examples/thai_service_contract.txt", "th")):
        doc = Document(name, (ROOT / path).read_text(encoding="utf-8"), [], lang, source=path)
        models = ENGLISH_MODELS if lang == "en" else ("thai",)
        samples.append(to_sample(name, doc, run_all(registry, doc, models), path))

    docs = read_jsonl(ROOT / args.test)
    scored = []
    for i, doc in enumerate(docs, 1):
        results = run_all(registry, doc, ENGLISH_MODELS)
        scored.append((doc, results))
        print(f"[{i}/{len(docs)}] {doc.id} {len(doc.text):>7,} chars  {describe(results)}")
    for name, (doc, results) in pick(scored, overrides).items():
        samples.append(to_sample(name, doc, results, f"CUAD test split, {doc.id}"))
        print(f"{name}: {doc.id}")

    thai_path = ROOT / args.thai_gold
    if thai_path.is_file() and not registry.missing_reason("thai"):
        thai = [(d, run_all(registry, d, ("thai",))) for d in read_jsonl(thai_path)]
        # a typical contract the model reads well, not the easiest one: most cells right, median length
        thai.sort(key=lambda c: (-n_right(c[1]["thai"]), abs(len(c[0].text) - 1500)))
        doc, results = thai[0]
        samples.append(to_sample("th-gold", doc, results, f"Thai dev gold set, {doc.id}"))
        print(f"th-gold: {doc.id}")

    for old in out.glob("*.json"):
        old.unlink()
    for s in samples:
        (out / f"{s['id']}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(samples)} samples → {out}")


if __name__ == "__main__":
    main()
