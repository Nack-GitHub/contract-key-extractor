# Spec: CUAD-first scope with a language seam for Thai

## Objective

Re-scope the repository so the **English / CUAD track is the primary, self-contained deliverable**
(matching `docs/midterm_presentation_cuad.md`), while keeping the Thai track intact and reachable
through a clean, config-driven language seam — no rewrite when Thai comes back.

User: the presenter (repo owner) and reviewers reading the midterm doc + repo. Success looks like:

* Someone can `pip install -e .` (no Thai deps), run `build-cuad → train → evaluate → extract` and
  get exactly the numbers in the CUAD doc.
* Every factual claim in `docs/midterm_presentation_cuad.md` matches the code.
* The English model is trained with an **11-label** space (B/I × 5 + O), not the shared 17.
* Adding Thai back = a YAML config + `pip install -e ".[thai]"`; no Python edits to the core.

### Why now

Four statements in the CUAD doc are currently untrue of the code (17 labels not 11; a text
"normalization" step that does not exist; alias demotion described as validation; jurisdiction
validation described as state/country matching). Fixing the label count properly *is* the seam
that future multi-language support needs, so the two goals share one change.

## Assumptions

1. Shared, language-agnostic core stays as-is: `schema.Document/Span`, JSONL format,
   `tokenization.Encoder` (per-word encoding + windowing), `predict.Extractor`, `evaluate`.
2. **One model per language** (RoBERTa for EN, WangchanBERTa for TH) — no multilingual XLM-R model.
3. The English model is **retrained** (~30 min on M2) so the 11-label claim is real; the reported
   numbers in the doc will be replaced by the retrained run's numbers (expected within noise).
4. Thai code, data, tests and reports are **not deleted**; they move behind an optional extra and a
   README section. `models/thai-wangchanberta` is untouched.
5. `docs/midterm_presentation.md` (bilingual version) is left unchanged; only the CUAD doc is edited.
6. No new runtime dependencies.

→ Correct any of these before implementation starts.

## Tech Stack

Python ≥ 3.10 · PyTorch ≥ 2.2 · transformers ≥ 4.45 (HF `Trainer`) · seqeval · pandas · PyYAML ·
python-dateutil · kagglehub. Thai only: pythainlp ≥ 5, sentencepiece, protobuf. Dev: pytest 8, ruff.

## Commands

```bash
pip install -e ".[dev]"            # CUAD-only install (after this change: no pythainlp)
pip install -e ".[dev,thai]"       # + Thai extras
contract-ner download              # Kaggle dataset → data/raw/kaggle
contract-ner build-cuad            # alignment → data/processed/cuad/{train,val,test}.jsonl
contract-ner train --config configs/english_cuad.yaml
contract-ner evaluate --model models/english-cuad-roberta --data data/processed/cuad/test.jsonl
contract-ner extract  --model models/english-cuad-roberta --file examples/english_service_agreement.txt --summary
python -m pytest -q                # unit tests; Thai tests skip when pythainlp is absent
ruff check src tests
```

## Project Structure

```
configs/                    YAML per model; gains `entity_types:` (the label seam)
src/contract_ner/
  schema.py                 ENTITY_TYPES (all known types) + build_bio_labels(subset)
  train.py                  label space from cfg.entity_types → saved in pipeline_config.json
  predict.py                label space from the model's config (already), lang from pipeline_config
  evaluate.py               per-type metrics over the model's types only
  tokenization.py           Encoder takes a label2id map instead of the global LABEL2ID
  postprocess.py            validate_span / normalisation dispatched by lang ({"en": …, "th": …})
  lang/                     NEW: en.py, th.py — language-specific rules (segmentation import,
                            validation regexes, normalisers); th.py imports pythainlp lazily
  data/cuad.py              unchanged
  data/thai_*.py            unchanged, importable only with the thai extra
tests/                      pytest; Thai tests `pytest.importorskip("pythainlp")`
docs/midterm_presentation_cuad.md   corrected claims + retrained numbers
README.md                   CUAD-first; Thai under "Extending to Thai (experimental)"
```

## Code Style

Match the existing code: type hints, dataclass configs, short module docstrings explaining *why*,
regexes compiled at module level, no abstractions beyond what two languages need.

```python
# src/contract_ner/lang/__init__.py
from dataclasses import dataclass
from collections.abc import Callable

@dataclass(frozen=True)
class LangRules:
    """Everything the pipeline needs that differs between languages."""
    segment_words: Callable[[str], list[tuple[int, int]]]
    validate_span: Callable[[Span], bool]
    normalise_jurisdiction: Callable[[str], dict]

def rules_for(lang: str) -> LangRules:
    if lang == "en":
        from .en import RULES
    elif lang == "th":
        from .th import RULES          # pulls in pythainlp only here
    else:
        raise ValueError(f"unsupported lang {lang!r}")
    return RULES
```

Conventions: `snake_case`, `UPPER_CASE` module constants, `_private` helpers, ruff line length 100.
Language-specific strings (Thai regexes, court tables) live in `lang/th.py`, never in shared modules.

## Testing Strategy

* Framework: pytest, `tests/` mirrors `src/contract_ner/` by topic.
* Unit: `build_bio_labels` with a subset; `Encoder` with a custom `label2id`; `rules_for("en")`
  validation cases (existing `test_postprocess.py` cases split by language); `pipeline_config.json`
  round-trip (train writes `entity_types`, `Extractor` reads it).
* Regression: `test_cuad.py` alignment counts unchanged (4,308 spans, 98.1 %).
* Integration (skipped without a model): `test_extractor_integration.py` on the retrained English
  model asserts `len(model.config.id2label) == 11` and the demo JSON fields.
* Thai tests keep passing with `[thai]` installed and **skip**, not fail, without it.
* Coverage target: no drop from current; all 73 tests green in the `[dev,thai]` environment.

## Boundaries

**Always**
* Run `python -m pytest -q` and `ruff check` before each commit; commit in small steps per task.
* Keep `models/thai-wangchanberta`, `data/thai_gold*`, `reports/thai-*` and Thai tests intact.
* Keep the JSONL document format and character-offset semantics unchanged.
* Every number changed in the CUAD doc must be copied from a file in `reports/`.

**Ask first**
* Retraining the English model (30 min, overwrites `models/english-cuad-roberta`; the old report
  is moved to `reports/iteration2/` first).
* Changing the split hash, hyperparameters, or the set of CUAD categories.
* Adding any dependency or editing `docs/midterm_presentation.md`.

**Never**
* Delete Thai code/data or rewrite git history.
* Change the label order of an already-trained model directory (would silently corrupt predictions).
* Commit model weights, generated data, or Kaggle raw files.

## Success Criteria

1. `configs/english_cuad.yaml` declares `entity_types: [PARTY, AGREEMENT_DATE, START_DATE, END_DATE, JURISDICTION]`;
   retrained model has `len(config.id2label) == 11`; `pipeline_config.json` lists the same types.
2. `contract-ner evaluate` on the retrained model writes `reports/english-cuad-roberta__test.json`
   with only the 5 types in `relaxed`/`fields` (no `support: 0` rows); strict/relaxed/field F1
   within ±0.02 of 0.584 / 0.735 / 0.657.
3. `pip install -e ".[dev]"` in a clean venv does not install pythainlp, and
   `python -m pytest -q` passes with Thai tests reported as skipped.
4. `pip install -e ".[dev,thai]"` → all tests pass, `contract-ner evaluate --model models/thai-wangchanberta`
   reproduces the committed Thai reports (blind strict F1 0.792).
5. `docs/midterm_presentation_cuad.md`: "11 label" is true; normalization row rewritten
   ("input text is not modified; whitespace/quote tolerance lives in the alignment regex");
   alias demotion moved to the Normalisation row; jurisdiction validation described as keyword
   matching (court/law/arbitra/tribunal/jurisdiction); §2.2 notes 510 → 499 contracts (11 with no
   aligned span dropped); results tables updated from the new report.
6. README leads with the English/CUAD track; Thai appears in one "Extending to Thai" section that
   states exactly what a new language needs: (a) an encoder + word segmenter, (b) `LangRules`,
   (c) annotated JSONL.

## Open Questions

1. Retrain now (needed for criterion 1–2, ~30 min) or ship the seam first and retrain later?
   Default: retrain as the last task.
2. Should `reports/` keep both the 17-label and 11-label English reports? Default: yes, old one
   under `reports/iteration2/`.
