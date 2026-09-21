# Tasks: CUAD-first scope with a language seam

Spec: `SPEC.md` · Plan: `tasks/plan.md`

## Phase 1 — label seam

- [ ] **Task 1: `entity_types` config → per-model label space**
  - Description: `TrainConfig.entity_types: list[str] | None` (None = all). `train.py` builds
    `labels = build_bio_labels(cfg.entity_types)` and passes its `label2id` to `Encoder` and
    `num_labels`/`id2label` to the model; writes `entity_types` into `pipeline_config.json`.
    `Encoder.__init__(..., label2id: dict[str,int] | None = None)` defaults to the global map.
    `Extractor` exposes `self.entity_types` (from pipeline_config, default `ENTITY_TYPES`).
    `configs/english_cuad.yaml` gets the 5-type list.
  - Acceptance:
    - [ ] `build_bio_labels(["PARTY","JURISDICTION"])` → 5 labels; `Encoder(label2id=…)` encodes a doc with only those tags
    - [ ] `TrainConfig.from_yaml("configs/english_cuad.yaml").entity_types` == the 5 CUAD types; Thai YAML → None
    - [ ] `Extractor` on `models/thai-wangchanberta` (no key) → `entity_types == ENTITY_TYPES`
  - Verify: `python -m pytest -q tests/test_tokenization.py tests/test_schema.py`; `ruff check src tests`
  - Files: `src/contract_ner/train.py`, `tokenization.py`, `predict.py`, `configs/english_cuad.yaml`, `tests/test_tokenization.py`
  - Dependencies: none · Scope: M

- [ ] **Task 2: evaluate over the model's own types**
  - Description: `evaluate_documents(...)` and `format_result(...)` iterate
    `extractor.entity_types` (passed as `types=`) instead of the global; `EvalResult` records `types`.
  - Acceptance:
    - [ ] Unit test with a stub extractor whose `entity_types=("PARTY","END_DATE")` → `relaxed`/`fields` contain exactly those keys + `micro`
    - [ ] `grep -n ENTITY_TYPES src/contract_ner/evaluate.py` → only the import/default
  - Verify: `python -m pytest -q tests/test_evaluate.py`; Thai blind evaluate prints strict F1 0.792
  - Files: `src/contract_ner/evaluate.py`, `tests/test_evaluate.py` (new)
  - Dependencies: Task 1 · Scope: S

### Checkpoint A
- [ ] `python -m pytest -q` green · `ruff check` clean
- [ ] Thai blind report reproduces (0.792 strict)
- [ ] Start Task 5 retrain in background: `contract-ner train --config configs/english_cuad.yaml`

## Phase 2 — language seam

- [ ] **Task 3: `lang/` package with `LangRules` for en / th**
  - Description: `src/contract_ner/lang/{__init__,en,th}.py`. `LangRules(segment_words,
    validate_span, romanize)`; `rules_for(lang)` imports lazily. Move `_EN_WORD_RE` → `en.py`;
    `thai_tokenizer`/`_segment_thai`/domain dictionary → `th.py`; `pythainlp.romanize` → `th.py`.
    `tokenization.segment_words(text, lang)` and `postprocess.validate_span(span, lang)` /
    `assemble(..., lang)` delegate. Thai-only regex branches in `validate_span` move to `th.py`;
    shared date/money/rate normalisers stay in `postprocess.py`. Public signatures keep defaults so
    existing callers/tests pass.
  - Acceptance:
    - [ ] `python -c "import contract_ner.tokenization, contract_ner.postprocess"` succeeds with pythainlp uninstalled (test via `sys.modules` block in a unit test)
    - [ ] `rules_for("en").validate_span` rejects `"Company"`-style alias? **No** — alias demotion stays in `group_parties`; test documents that validation only checks letters/length
    - [ ] All existing `test_postprocess.py` / `test_tokenization.py` cases pass unchanged
  - Verify: `python -m pytest -q`; `ruff check`
  - Files: `src/contract_ner/lang/__init__.py`, `lang/en.py`, `lang/th.py`, `tokenization.py`, `postprocess.py`
  - Dependencies: none (parallel to Task 1) · Scope: M (5 files)

- [ ] **Task 4: Thai deps → `[thai]` extra; tests skip without it**
  - Description: move `pythainlp`, `sentencepiece`, `protobuf` to
    `[project.optional-dependencies] thai`; `tests/test_thai_*.py` and Thai cases in
    `test_tokenization.py`/`test_postprocess.py` use `pytest.importorskip("pythainlp")`;
    `cli.py` `build-thai`/`sample` print a clear message if the extra is missing;
    `scripts/reproduce.sh` installs `[dev,thai]`.
  - Acceptance:
    - [ ] Clean venv `pip install -e ".[dev]"` → `pip show pythainlp` fails; `pytest -q` passes, Thai tests reported `s`
    - [ ] `[dev,thai]` venv → all tests pass
  - Verify: both venvs in scratchpad dir (`python -m venv /private/tmp/.../venv-en`)
  - Files: `pyproject.toml`, `tests/test_thai_gold.py`, `tests/test_thai_synthetic.py`, `tests/test_tokenization.py`, `src/contract_ner/cli.py`, `scripts/reproduce.sh`
  - Dependencies: Task 3 · Scope: M

### Checkpoint B
- [ ] Both venv runs above pass
- [ ] `contract-ner evaluate --model models/thai-wangchanberta …` in the `[thai]` venv reproduces blind 0.792

## Phase 3 — retrain + docs

- [ ] **Task 5: retrain English model (11 labels) + new report**
  - Description: `mv reports/english-cuad-roberta__test.json reports/iteration2/`; train with
    `configs/english_cuad.yaml`; `contract-ner evaluate --model models/english-cuad-roberta --data data/processed/cuad/test.jsonl --out reports/english-cuad-roberta__test.json`;
    regenerate `reports/example_english_service_agreement.json`.
  - Acceptance:
    - [ ] `len(AutoConfig.from_pretrained("models/english-cuad-roberta").id2label) == 11`
    - [ ] New report has 5 types + micro only; strict/relaxed/field F1 within ±0.02 of 0.584 / 0.735 / 0.657 (if not, record the actual numbers — doc narrative is about direction, not decimals)
    - [ ] `test_extractor_integration.py` passes and asserts the 11-label count
  - Verify: `python -m pytest -q tests/test_extractor_integration.py`
  - Files: `reports/…`, `tests/test_extractor_integration.py`
  - Dependencies: Task 1 (Task 4 for a clean env) · Scope: S (long wall clock ~30 min)

- [ ] **Task 6: correct `docs/midterm_presentation_cuad.md`**
  - Description: (a) "11 label" now true — keep; (b) Normalization row → "input text not modified,
    whitespace/quote tolerance is in the alignment regex"; (c) alias demotion moves from Span
    validation row to Normalisation row; (d) JURISDICTION validation = keyword match
    (court/law/arbitra/tribunal/jurisdiction); (e) §2.2(ง) note 510 → 499 contracts (11 with no
    aligned span dropped); (f) §4.2/4.3/6 numbers and error counts copied from the new report;
    (g) §3.3/§6 answer on `is_split_into_words` unchanged.
  - Acceptance:
    - [ ] Every numeric claim in §4 matches `reports/english-cuad-roberta__test.json` (checked with a small script)
    - [ ] No remaining claim about text normalization or state/country validation
  - Verify: script in scratchpad prints doc numbers vs report numbers side by side
  - Files: `docs/midterm_presentation_cuad.md`
  - Dependencies: Task 5 · Scope: XS

- [ ] **Task 7: README → CUAD-first + "Extending to Thai"**
  - Description: reorder so English/CUAD is the headline (install, quick start, results); Thai
    moves to one section "Extending to Thai (experimental)" that states what a new language needs:
    encoder + `LangRules` + annotated JSONL, plus `pip install -e ".[thai]"`. Keep Thai results table.
  - Acceptance:
    - [ ] Quick start block runs top-to-bottom in the `[dev]` venv
    - [ ] Thai section lists the three requirements verbatim from SPEC criterion 6
  - Verify: run quick-start commands in the en venv (download/build already cached → evaluate + extract)
  - Files: `README.md`
  - Dependencies: Task 5 · Scope: XS

### Checkpoint C
- [ ] SPEC success criteria 1–6 all true
- [ ] Human reviews doc diffs, then commit per phase
