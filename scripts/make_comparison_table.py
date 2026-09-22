#!/usr/bin/env python
"""Collect every evaluation report in ``reports/`` into one comparison document.

Written for the project write-up: one table of the four document-level views per
model and split, one table per metric view, and the recall of every model on
every surface form, so the numbers in a report do not have to be copied out of
a dozen JSON files by hand.

    python scripts/make_comparison_table.py --reports reports --out reports/comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWS = ("strict", "relaxed", "field", "slot")


def load_reports(reports_dir: Path) -> list[dict]:
    """Every ``<model>__<split>.json`` in the directory, newest naming scheme only."""
    found = []
    for path in sorted(reports_dir.glob("*__*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model, _, split = path.stem.partition("__")
        found.append({"model": model, "split": split, "path": path, "data": data})
    return found


def _micro(data: dict, view: str) -> dict:
    if view == "strict":
        return data["strict"]
    return data[{"relaxed": "relaxed", "field": "fields", "slot": "slots"}[view]]["micro"]


def overview_table(reports: list[dict]) -> str:
    lines = ["| Model | Split | documents | strict F1 | relaxed F1 | field F1 | **slot F1** | slot precision | slot recall |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(reports, key=lambda r: (r["model"], r["split"])):
        cells = [f"{_micro(r['data'], view)['f1']:.3f}" for view in VIEWS]
        slot = _micro(r["data"], "slot")
        lines.append(f"| {r['model']} | {r['split']} | {r['data']['n_docs']} | " + " | ".join(cells)
                     + f" | {slot['precision']:.3f} | {slot['recall']:.3f} |")
    return "\n".join(lines)


def slot_cell_table(reports: list[dict]) -> str:
    cells = sorted({cell for r in reports for cell in r["data"].get("slots", {}) if cell != "micro"})
    lines = ["| Model | Split | " + " | ".join(cells) + " |",
             "|---|---|" + "---:|" * len(cells)]
    for r in sorted(reports, key=lambda r: (r["model"], r["split"])):
        row = [f"{r['data']['slots'][cell]['f1']:.2f}" if cell in r["data"]["slots"] else "–" for cell in cells]
        lines.append(f"| {r['model']} | {r['split']} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def form_tables(reports: list[dict]) -> str:
    """Recall per surface form, one table per split."""
    blocks = []
    for split in sorted({r["split"] for r in reports}):
        in_split = [r for r in reports if r["split"] == split and r["data"].get("forms")]
        if not in_split:
            continue
        keys = sorted({(label, form) for r in in_split for label, forms in r["data"]["forms"].items()
                       for form in forms})
        models = [r["model"] for r in sorted(in_split, key=lambda r: r["model"])]
        blocks.append(f"### {split}\n")
        blocks.append("| Label | Surface form | gold spans | " + " | ".join(models) + " |")
        blocks.append("|---|---|---:|" + "---:|" * len(models))
        for label, form in keys:
            support = max((r["data"]["forms"].get(label, {}).get(form, {}).get("support", 0)
                           for r in in_split), default=0)
            row = []
            for r in sorted(in_split, key=lambda r: r["model"]):
                entry = r["data"]["forms"].get(label, {}).get(form)
                row.append(f"{entry['found']}/{entry['support']}" if entry else "–")
            blocks.append(f"| {label} | `{form}` | {support} | " + " | ".join(row) + " |")
        blocks.append("")
    return "\n".join(blocks)


def share_versus_recall(reports: list[dict], profile_path: Path, split: str) -> str:
    """How much of each surface form the training data holds, next to the recall it gets.

    The two halves of the story in one table: the dataset profile says how often a
    form was seen, the evaluation says how often it was found.
    """
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    training = next((s for s in profile["splits"] if s["name"] == "train"), profile["splits"][0])
    in_split = sorted([r for r in reports if r["split"] == split and r["data"].get("forms")],
                      key=lambda r: r["model"])
    if not in_split:
        return ""
    models = [r["model"] for r in in_split]
    lines = [f"Training data: `{training['name']}` ({training['n_documents']} documents). "
             f"Recall measured on `{split}`.\n",
             "| Label | Surface form | share of the label in training | " + " | ".join(models) + " |",
             "|---|---|---:|" + "---:|" * len(models)]
    seen = {(label, form) for r in in_split for label, forms in r["data"]["forms"].items() for form in forms}
    for label, form in sorted(seen):
        share = training["labels"].get(label, {}).get("form_share", {}).get(form)
        row = []
        for r in in_split:
            entry = r["data"]["forms"].get(label, {}).get(form)
            row.append(f"{entry['found']}/{entry['support']}" if entry else "–")
        lines.append(f"| {label} | `{form}` | {f'{share:.1%}' if share is not None else 'absent'} | "
                     + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--out", default="reports/comparison.md")
    parser.add_argument("--profile", default="reports/profile/cuad_english.json",
                        help="dataset profile to join the per-form recall against")
    parser.add_argument("--profile-split", default="val",
                        help="split whose recall is compared with the training shares")
    args = parser.parse_args()

    reports = load_reports(Path(args.reports))
    document = [
        "# Model comparison\n",
        "Generated by `scripts/make_comparison_table.py` from the reports in `reports/`. "
        "Every number is document-level: whole contracts through windowing, stitching, decoding, "
        "span validation and normalisation.\n",
        "## All views, all models\n", overview_table(reports), "",
        "## Slot level, per table cell\n", slot_cell_table(reports), "",
        "## Recall per surface form\n", form_tables(reports),
    ]
    profile_path = Path(args.profile)
    if profile_path.exists():
        joined = share_versus_recall(reports, profile_path, args.profile_split)
        if joined:
            document += ["## Training share versus recall\n", joined, ""]
    out = Path(args.out)
    out.write_text("\n".join(document) + "\n", encoding="utf-8")
    print(f"wrote {out} from {len(reports)} reports")


if __name__ == "__main__":
    main()
