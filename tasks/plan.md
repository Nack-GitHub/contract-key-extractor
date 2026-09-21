# Implementation Plan: CUAD-first scope with a language seam for Thai

Spec: `SPEC.md`. Read-only analysis done on 2026-09-21.

## Overview

Make the English/CUAD track the primary deliverable (11-label model, no Thai dependencies needed
to install/run it) and move every language-specific piece behind a single `lang` seam so Thai
comes back with a YAML config + an optional extra. Finish by retraining the English model and
correcting the CUAD midterm doc from the new report.

## Dependency graph (what must exist before what)

```
schema.build_bio_labels(subset)            [exists]
    │
    ├── T1  config.entity_types → per-model label space
    │        train.py (num_labels, id2label, pipeline_config.json)
    │        tokenization.Encoder(label2id=...)   ← currently uses global LABEL2ID
    │        predict.Extractor (reads entity_types from pipeline_config)
    │        evaluate.evaluate_documents(types=extractor.entity_types)
    │
    ├── T2  lang/ package: LangRules for en / th
    │        tokenization.segment_words → lang.rules_for(lang).segment_words
    │        postprocess.validate_span / normalize_jurisdiction → per-lang
    │        (pythainlp import moves into lang/th.py + data/thai_synthetic.py only)
    │
    ├── T3  pyproject extras + test skips          ← needs T2 (imports are lazy)
    │
    ├── T4  retrain EN + new report                ← needs T1 (and T2/T3 for a clean env)
    │
    └── T5  docs (CUAD doc + README)               ← needs T4 for the numbers
```

T1 and T2 touch different functions and could be done in either order; T1 first because it is the
riskier change (touches training) and unblocks the 30-minute retrain early.

## Architecture decisions

- **Label space per model, stored beside the weights.** `pipeline_config.json` gains
  `entity_types`; `Extractor` already reads `id2label` from the HF config, so inference needs no
  new logic — only `evaluate` must stop iterating the global `ENTITY_TYPES`. Existing model
  directories (Thai, old EN) have no `entity_types` key → default to all `ENTITY_TYPES`, so
  `models/thai-wangchanberta` keeps working unchanged.
- **`Encoder` takes a `label2id` mapping** instead of importing the global. Default = full map, so
  every existing call site and test still works.
- **`LangRules` is a frozen dataclass, resolved lazily by `lang.rules_for(lang)`.** Only
  `lang/th.py` (and `data/thai_synthetic.py`) import pythainlp. `postprocess.py` keeps the shared
  normalisers (dates, money, rates are already bilingual regexes) and gets `lang` passed where it
  currently guesses from the text (`normalize_jurisdiction` → `_romanize` needs pythainlp only for
  Thai input).
- **`pythainlp` + `sentencepiece` + `protobuf` become the `[thai]` extra.** `cli.py` imports
  `thai_gold`/`thai_synthetic` only inside `build-thai` / `sample` (already function-local) — keep
  it that way. Thai tests use `pytest.importorskip("pythainlp")`.
- **Old 17-label English report is kept** in `reports/iteration2/` so the doc's v1→v3→final table
  can stay and a new "11-label" row is added.
- **No change to `docs/midterm_presentation.md`**, split hash, hyperparameters, or CUAD categories.

## Task list

### Phase 1 — label seam (core)
- [ ] Task 1: `entity_types` config → 11-label English training + config round-trip
- [ ] Task 2: evaluate/format over the model's own types

### Checkpoint A
- [ ] `python -m pytest -q` green (73 + new tests)
- [ ] `contract-ner evaluate --model models/thai-wangchanberta --data data/processed/thai/test_gold_blind.jsonl`
      still prints strict F1 0.792 (no `entity_types` key → all types)
- [ ] Kick off Task 5 retrain in the background (needs only Task 1)

### Phase 2 — language seam
- [ ] Task 3: `lang/` package; `segment_words` and span validation dispatched by lang
- [ ] Task 4: Thai deps → `[thai]` extra; tests skip without pythainlp

### Checkpoint B
- [ ] Clean venv `pip install -e ".[dev]"` → no pythainlp; `pytest -q` passes with Thai tests skipped
- [ ] `[dev,thai]` venv → all tests pass; Thai blind report reproduces

### Phase 3 — retrain + docs
- [ ] Task 5: retrain English model; new `reports/english-cuad-roberta__test.json`
- [ ] Task 6: correct `docs/midterm_presentation_cuad.md`
- [ ] Task 7: README → CUAD-first + "Extending to Thai"

### Checkpoint C (done)
- [ ] All SPEC success criteria 1–6 ticked
- [ ] Human review of doc diffs before commit

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Retrained 11-label model scores drift > ±0.02 from doc numbers | Med — doc tables must change | Same seed/hyperparams; keep old report; doc gets both rows, narrative unchanged |
| `evaluate.py` per-type loops use `ENTITY_TYPES` in 8 places — miss one → KeyError on subset | Med | Add test that evaluates a fake 2-type extractor; grep after edit |
| `pythainlp` imported at module top in `tokenization.py` and `postprocess.py` — moving it can break Thai romanisation path | Med | Lazy import inside `lang/th.py`; Thai postprocess tests stay in the `[thai]` env |
| Hidden pythainlp use in `thai_lexicon.py` or `io.py` | Low | grep showed none; Checkpoint B clean-venv test catches it |
| Old EN model dir loses compatibility when `Extractor` reads `entity_types` | Low | Key optional; default all types |
| M2 disk/RAM during retrain (memory note: disk limits) | Low | `train.py` already deletes checkpoints; move old model dir aside, don't duplicate |

## Parallelisation

- Task 5 (retrain, ~30 min wall clock) runs in the background right after Checkpoint A while
  Tasks 3–4 proceed.
- Task 6 and 7 are independent of each other but both wait for Task 5's numbers.

## Open questions

1. Retrain immediately after Checkpoint A (default) or hold until Checkpoint B? Default: immediately.
2. Keep the old EN model weights? Default: no — git-ignored and reproducible; only the report is kept.
