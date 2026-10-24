# Contract Key-Information Extraction (NER on CUAD, extensible to Thai)

Reads a commercial contract as plain text and extracts the metadata a legal / finance / procurement
team asks for first — the parties, agreement / effective / expiration dates and the governing law —
into a JSON record, with Named Entity Recognition (a fine-tuned RoBERTa token classifier) followed
by rule-based span validation and normalisation. Every value keeps its character offsets and
confidence so a reviewer can check the draft extraction against the source text.

The primary track is **English, trained and evaluated on CUAD** (Hendrycks et al., 2021) via the
Kaggle [Legal Contract Dataset](https://www.kaggle.com/datasets/changethan/legal-contract-dataset).
The pipeline is language-agnostic behind one seam (`contract_ner.lang`); a **Thai** track
(WangchanBERTa on synthetic contracts + a hand-annotated gold set) lives in the same repo as an
experimental extension — see [Extending to Thai](#extending-to-thai-experimental).

| Track | Base model | Training data | Evaluation data | Install |
|---|---|---|---|---|
| **English** (primary) | `roberta-base` | CUAD spans re-aligned into full text (398 contracts) | 47 held-out CUAD contracts, 370 gold spans | `pip install -e ".[dev]"` (`[baseline]` for the conditional random field) |
| Thai (experimental) | `airesearch/wangchanberta-base-att-spm-uncased` | 3,000 synthetic, auto-annotated Thai contracts | 24 + 8 (blind) hand-annotated Thai contracts + 300 held-out synthetic | `pip install -e ".[dev,thai]"` |

## What it extracts

| Output field | Entity label | CUAD category | English model | Thai model |
|---|---|---|---|---|
| `Party_A`, `Party_B` | `PARTY` (A/B assigned by order of appearance) | `Parties` | ✅ | ✅ |
| `Agreement_Date` | `AGREEMENT_DATE` | `Agreement Date` | ✅ | ✅ |
| `Start_Date` | `START_DATE` | `Effective Date` | ✅ (inferred from `Agreement_Date`, or split from its candidates by context) | ✅ |
| `End_Date` | `END_DATE` | `Expiration Date` | ✅ | ✅ |
| `Jurisdiction` | `JURISDICTION` (court / arbitration / governing law) | `Governing Law` | ✅ | ✅ |
| `Contract_Value` | `CONTRACT_VALUE` | — (no short-span CUAD category) | — | ✅ |
| `Payment_Due_Date` | `PAYMENT_DUE` | — | — | ✅ |
| `Penalty_Rate` | `PENALTY_RATE` | — | — | ✅ |

Each model is trained on its own label space (`entity_types` in its YAML config): 11 BIO labels for
English (B/I × 5 + O), 17 for Thai.

```bash
contract-ner extract --model models/english-cuad-roberta-label-aware-augmented --file examples/english_service_agreement.txt --summary
```

```json
{
  "Party_A": "Northwind Analytics, Inc.",
  "Party_B": "Contoso Manufacturing Ltd.",
  "Agreement_Date": "2021-03-15",
  "Start_Date": "2021-03-15",
  "End_Date": "2024-03-14",
  "Jurisdiction": "State of New York"
}
```

The full record (`--summary` omitted) keeps every predicted span with its character offsets,
confidence, normalised value (ISO dates, court / governing-law place name) and a `valid` flag from the
span-validation stage; `Start_Date` carries `"inferred_from": "Agreement_Date"` when it was not
found separately. See `reports/example_english_service_agreement.json`.

## Results — English (RoBERTa fine-tuned on CUAD)

All numbers are document-level: whole contracts go through windowing, stitching, BIO decoding and
span validation, exactly as `contract-ner extract` does. Four views are reported, and every report
also carries **recall per surface form** (`forms`), which says *which way of writing a value* was
missed rather than only which type:

* **strict** – exact span match (seqeval, word level);
* **relaxed** – same type and overlapping span (a one-word boundary slip still counts);
* **field** – per document, the *set of distinct values* per type is compared (a party tagged five
  times is one value) — still on the raw span text, so a second, equally correct governing-law
  clause counts as a wrong value;
* **slot** – the metadata table itself: the record `assemble()` builds (one value per cell, after
  normalisation) against the table implied by the gold spans (`contract_ner.slots`). Parties are a
  set of legal names (aliases such as `"Company"` and the *referred to as the Parties* boilerplate
  are dropped on both sides), dates are ISO, end-of-term clauses reduce to a date or a duration
  (`3 year`), governing-law clauses to the place they name. This is what a user sees, and the
  number the project is judged on.

47 held-out CUAD contracts, 370 gold spans, five entity types (`reports/english-cuad-roberta__test.json`).
The slot column is the one to read: it is the metadata table, and the target for "usable with
review" is ≥ 0.80.

| Iteration | strict F1 | relaxed F1 | field F1 | **slot F1** |
|---|---:|---:|---:|---:|
| v1 — raw-text BPE encoding, no validation, 17 labels | 0.254 | 0.607 | 0.358 | — |
| v3 — per-word encoding (punctuation no longer glued to party names) | 0.514 | 0.654 | 0.603 | — |
| v4 — v3 + span validation (`reports/iteration2/`) | 0.584 | 0.735 | 0.657 | — |
| v5 — v4 retrained on the 5-type label space (11 labels) | 0.619 | 0.738 | 0.688 | — |
| v6 — v5 + stricter English span validation (name fragments, dateless "term" clauses) | 0.633 | 0.754 | 0.702 | 0.804\* |
| v7 — v6 + normalisation fixes, effective-date split, governing-law pick, confidence ≥ 0.6 (same weights) | 0.652 | 0.759 | 0.722 | 0.840 |
| v8 — v7 recipe reshaped by the label distribution (`configs/english_cuad_label_aware.yaml`) | 0.547 | 0.715 | 0.638 | 0.845 |
| **final** — v8 + date-format and party-capitalisation augmentation (`configs/english_cuad_label_aware_augmented.yaml`) | **0.545** | **0.720** | **0.640** | **0.855** |
| *DeBERTa-v3-base* under the merged recipe (`configs/english_cuad_deberta.yaml`) | *0.663* | *0.762* | *0.743* | *0.807* |
| *merged-label retrain* — `START_DATE` folded into `AGREEMENT_DATE` (`configs/english_cuad_merged.yaml`) | *0.630* | *0.806* | *0.691* | *0.839* |
| *conditional random field baseline* (`contract-ner train-crf`, `reports/crf-baseline__test.json`) | *0.509* | *0.586* | *0.576* | *0.690* |
| *regex-only baseline* (`--model regex`, `reports/regex-baseline__test.json`) | *0.497* | *0.705* | *0.545* | *0.762* |

\* v6 is the one row still measured before clause-level slot values were compared without their
edge punctuation (`slots.value_key`); every other row was re-scored under the current rule, which
left v7's test number unchanged at 0.840 and raised its validation number from 0.846 to 0.853.
`reports/comparison.md` holds all of them together with the validation split.

final: strict P / R = 0.475 / 0.641 · relaxed P / R = 0.627 / 0.846 · slot P / R = 0.867 / 0.843,
and on the validation split slot F1 = 0.874 against 0.853 for v7 — the recipe is ahead on both
splits. Strict and field F1 fall because the model now tags every mention of a party while CUAD's
gold holds only the defining one; the table it produces is what improved.

| Table cell | docs with gold | **final slot F1** (P / R) | v7 | conditional random field | regex |
|---|---:|---|---:|---:|---:|
| Parties (set of legal names) | 47 | **0.87** (0.89 / 0.85) | 0.84 | 0.61 | 0.80 |
| Agreement_Date | 42 | **0.83** (0.87 / 0.79) | 0.83 | 0.81 | 0.74 |
| Start_Date | 42 | **0.80** (0.84 / 0.76) | 0.75 | 0.59 | 0.68 |
| End_Date (date or duration) | 32 | **0.78** (0.73 / 0.84) | 0.84 | 0.72 | 0.55 |
| Jurisdiction (governing-law place) | 34 | **0.96** (0.94 / 0.97) | 0.96 | 0.96 | 0.88 |

Span-level per type (final model): relaxed / field F1 — PARTY 0.74 / 0.75 · AGREEMENT_DATE 0.59 / 0.64 ·
START_DATE 0.00 / 0.00 (10 gold spans, 4 of them dateless clauses the recipe drops) · END_DATE 0.73 / 0.37 ·
JURISDICTION 0.89 / 0.35; conditional random field PARTY 0.57 / 0.58 · AGREEMENT_DATE 0.53 / 0.55 ·
END_DATE 0.58 / 0.50 · JURISDICTION 0.88 / 0.81. The recall behind the table is reported per surface
form in every report (`forms`): on the test split the final model finds every `Month D, YYYY`
agreement date (18/18 against 16/18 before), 2 of the 5 written as digits (0/5 before), 74 of 90
title-case party names (69/90) and 32 of 35 written in capitals (29/35).

**Why the slot view moves so much more than the span views.** The two gaps are different things.
Span → field is the "one mention per contract" style of CUAD: 43 of the 77 surviving `PARTY` false
positives are exact repeats of a gold party elsewhere in the contract. Field → slot is
normalisation and selection: a second, equally correct governing-law clause (`The laws of the State
of New York govern this Agreement.` was a false positive when the gold clause was the longer one
four sections later), a clause in capitals, a party name trimmed by one token, `three (3) years`
versus `a period of 3 years`. Jurisdiction goes 0.76 → 0.96 on the same predictions. The remaining
slot errors are model errors or gold gaps: of the thirteen slots the shipped model predicts where CUAD has no gold (checked by hand), six
name a real value the annotation missed (two governing-law clauses — Nevada and the PRC — and three
term clauses), so the CUAD-gold number is a floor.

**Two baselines, same split, same metrics.** The pattern-only baseline
(`src/contract_ner/baseline.py`, `--model regex`) reads the preamble with patterns — legal names ending
in `Inc.` / `Ltd.` / …, quoted defined terms, the *dated as of* / *effective as of* dates — and takes
the first sentence about the agreement's term and the first *governed by … laws of* sentence. On
the preamble cells it is close to the model (Parties 0.80 vs 0.84, Agreement_Date 0.74 vs 0.83);
the gap opens on `End_Date` (0.55 vs 0.78 — the term clause is phrased a hundred ways) and on
`Jurisdiction` recall.

The second baseline is a **linear-chain conditional random field** over hand-designed features
(`src/contract_ner/baseline_crf.py`, `contract-ner train-crf`): word shape, capitalisation, affixes,
a legal-vocabulary gazetteer, the two words on either side and the position in the document — the
standard approach to named entity recognition before pre-trained encoders. It is trained on the same
contracts, decoded into the same spans and scored through the same span validation and normalisation,
so the distance between it and the fine-tuned encoder is what pre-training contributes, with
everything else held constant. It reaches slot F1 0.753 on validation and 0.690 on test against
0.874 / 0.855 for the encoder: it holds its own where a decision is local (`Jurisdiction` 0.96,
`Agreement_Date` 0.81 — a court name or a date is recognisable from the words around it) and falls
away where the decision needs the whole document (`Parties` 0.61 against 0.87, because the features
cannot tell the defining mention of a name from every later one).

**Merged-label retrain.** Folding CUAD's `Effective Date` into `AGREEMENT_DATE` (dropping the 37 %
of its spans that are dateless definitional clauses), rewriting a quarter of the training dates into
`7/7/98` / `3-31-16` / `24 OCT 2011` and keeping twice as many entity-free windows lifts span recall
(relaxed 0.759 → 0.806) but not the table (slot 0.840 → 0.828, i.e. two cells on 47 documents — the
same on the validation split, 0.846 vs 0.837). The numeric formats the test split misses (`7/7/98`,
`9/9/97`, `24 OCT 2011`) are still missed, and the extra numeric training dates make the tagger fire
on page-footer dates unless a confidence floor is applied. The simpler recipe therefore stays the
shipped one; the merged config is kept as `configs/english_cuad_merged.yaml`.

**DeBERTa-v3-base** under the same merged recipe (2 h 15 min on the M2; the checkpoint ships in
fp16 and has to be loaded as fp32 or fine-tuning diverges to NaN within 100 steps) has the best
span-level scores on both splits (strict 0.663 test / 0.729 val vs 0.652 / 0.692 for the shipped
model) and the best table on the validation split (slot 0.869 vs 0.846), but a lower table on
the test split (0.807): it misses nine agreement dates that RoBERTa finds. Two encoders 0.02 apart in
opposite directions on 47- and 54-document splits are not separable; the smaller, 1.5× faster
RoBERTa stays the shipped model and the DeBERTa run is the capacity reference in
`docs/worklog_slot_eval_and_retrain.md`.

Recall is high; precision is limited by CUAD's annotation style — only the *defining* mention of each
party is gold, so later mentions the model tags count as false positives at span level, and
`Expiration Date` / `Governing Law` are whole clauses whose exact boundaries are hard to reproduce.
CUAD's `Parties` gold also includes the *"X and Y are collectively referred to as the Parties"*
boilerplate (156 of 2,441 train spans) and `Effective Date` gold is often a definitional clause
without a date (37 of 102), so the validation stage cannot reject those forms without losing recall
— it only drops what can never be a value of the type. `Start_Date` is inferred as
`Agreement_Date` (flagged `inferred_from`) unless the agreement-date candidates name two different
dates, in which case the one introduced by *effective* / *commence* is the start date. The
narrative for the presentation is in `docs/midterm_presentation_cuad.md`; the work log for the
slot-level evaluation and the retrains is `docs/worklog_slot_eval_and_retrain.md`.

## Label distribution — the step before training

A label in a contract corpus is not one kind of string, and a single score per label hides that.
`contract-ner profile` measures the training data before anything is fine-tuned on it, and writes a
Markdown and a JSON report into `reports/profile/`:

```bash
contract-ner profile --config configs/english_cuad.yaml --out reports/profile --stem cuad_english
```

It answers four questions per split — how many spans a label has and in how many documents, how long
they are, where in the document they sit, and which **surface forms** they are written in — plus a
drift table that names the forms one split has and another does not, and a windowing section that
checks the configured window length against the spans it has to hold. The surface-form buckets are
part of the language seam (`lang/en.py`, `lang/th.py`), so a new language gets the same report, and
evaluation reuses them to report **recall per surface form** (`forms` in every report under
`reports/`).

What it found on CUAD (`reports/profile/cuad_english.md`, 398 training contracts):

| Label | spans | documents with it | length (words, median / 90th pct) | in first 5% of text | main surface forms |
|---|---:|---:|---|---:|---|
| `PARTY` | 1,953 | 100% | 2 / 6 | 82% | quoted alias 36% · legal name 34% · name in capitals 12% · "referred to as the Parties" 6% |
| `AGREEMENT_DATE` | 363 | 90% | 3 / 5 | 83% | `Month D, YYYY` 53% · `Dth day of Month` 25% · digits (`03/01/05`) 6% · unfilled blank 5% |
| `START_DATE` | 76 | **18%** | 10 / 26 | 28% | **no date at all 40%** · a date 60% |
| `END_DATE` | 361 | 80% | **31 / 66** | 8% | duration 44% · "until terminated" 27% · explicit date 22% |
| `JURISDICTION` | 346 | 83% | **29 / 51** | **0%** (93% in the second half) | "governed by / construed" 94% |

Three things follow directly, and they are why the recipe in `configs/english_cuad_label_aware.yaml`
differs from the original one:

* **Two problems share one tagger.** `PARTY` and `AGREEMENT_DATE` are short spans in the preamble;
  `END_DATE` and `JURISDICTION` are whole clauses in the second half. Exact-span F1 over both is not
  a meaningful average, which is the structural reason the strict and slot columns are so far apart.
* **`START_DATE` mixes two span shapes in 76 examples** — a three-word date and a 26-word
  definitional clause. Its dateless spans are dropped from the *training* data (never from the gold
  used for scoring), so the label has one shape to learn.
* **Entity-free windows are not interchangeable.** Over 90% of a contract carries no target entity,
  so those windows are sampled down; the profile shows the look-alikes (addresses, registration
  numbers, page headers) and four of five short-type gold spans are both concentrated in the opening,
  so windows from the first tenth of a document are now kept at 0.50 and the rest at 0.12 — the same
  total number of windows, harder negatives.

Characters inside a gold span are 1.03% of the corpus. The window settings themselves were checked
rather than assumed: at 256 subwords with 64 of overlap only 14 of 3,099 training spans (0.45%) fall
across a window boundary, so a longer window would cost training time and buy nothing.

The same command profiles the Thai track (`reports/profile/thai_wangchanberta.md`) and explains the
gap between its dev and blind gold sets: dates written with Thai numerals and no era marker are 3% of
the synthetic training spans and 25–29% of the blind set, and Latin-script party names never occur in
training at all but are 10% of the blind set's parties.

## Pipeline

```
 raw text ──► word segmentation ──► subword encoding ──► fine-tuned encoder ──► BIO decode ──► span validation ──► JSON
             (contract_ner.lang:     (offset-aligned,       (RoBERTa /            (word level,     + normalisation
              regex / PyThaiNLP)      overlapping windows)   WangchanBERTa)        lenient IOB2)   (dates, places, courts)
```

1. **Word segmentation** – the unit BIO tags live on. English uses a small regex (alphanumeric runs
   with internal hyphens / dots, single punctuation). Thai has no spaces, so `lang/th.py` uses
   PyThaiNLP `newmm` with a domain dictionary. The input text is never modified — every span must map
   back to a character offset in the original.
2. **Subword alignment** – each word is tokenised on its own through the tokenizer's Rust backend
   with the prefix marker (`Ġ` / `▁`) re-added only where a real space preceded the word. The pieces
   are the same as for raw text, but none can straddle a word boundary — with raw-text tokenisation
   BPE merges `"),` and SentencePiece merges `ของกฎหมาย`, and the entity's first/last word then has
   no piece of its own and loses its label (v1 → v3 above). (`is_split_into_words=True` avoids that
   too, but prepends a marker to *every* word, unlike pre-training.) The first content piece of each
   word carries the label and the prediction; pieces are mapped back to words through offsets.
3. **Windows** – long contracts (CUAD averages 52k characters ≈ 13k tokens) are cut into 256-subword
   windows with a 64-subword overlap; during training only 15 % of entity-free windows are kept
   (> 90 % of a contract has no target entity); at inference each subword takes the logits from the
   window where it is furthest from an edge.
4. **Fine-tuning** – standard token classification with the HF `Trainer`, YAML-driven (`configs/`):
   3 epochs, lr 3e-5, batch 16. The label space comes from the config's `entity_types` and is stored
   next to the weights (`pipeline_config.json`), together with an optional `label_map` (the merged
   English variant folds CUAD's `Effective Date` into `AGREEMENT_DATE`; evaluation folds the gold the
   same way). `date_augment` rewrites a fraction of the dated training spans into formats CUAD rarely
   uses — `7/7/98`, `3-31-16`, `24 OCT 2011`, `Month ___, YYYY` — with the offsets re-based
   (`contract_ner.augment`).
5. **Decoding + span validation** – word-level BIO → character spans; each span is checked against its
   type by surface form (a short date must parse; a jurisdiction must contain court / law /
   arbitration / tribunal / jurisdiction; a party must contain a letter, not be a stop-word or a
   bare legal-form suffix such as `Corp`; an end-of-term clause must mention ending or duration).
   This is the rule-based stand-in for the deck's "span classification" stage; invalid spans stay in
   the output (`valid: false`) but are never promoted to a summary field. Spans whose mean label
   probability is under 0.6 are dropped first (`--min-confidence`; the tagger emits one-token
   fragments such as `,` or `18/` at ≈ 0.5 that no surface rule catches — the floor was set on the
   validation split).
6. **Normalisation** – dates → ISO-8601 (`dateutil`; Buddhist-era and Thai numerals for Thai;
   all-numeric dates are month-first for English and day-first for Thai; `the first day of July,
   2017` and two-digit years parse), governing law / court → place name (also for all-caps clauses),
   amounts and rates → `{amount, currency, per}` (Thai fields).
   Party A/B = first two distinct legal names in order of appearance; repeated mentions merge, and
   short quoted defined terms such as `"Company"` and *referred to as the Parties* boilerplate are
   demoted to `Other_Parties` here. When the agreement-date candidates name two different dates,
   the one preceded by *effective* / *commence* becomes `Start_Date` and the earliest other one
   `Agreement_Date`; `Governing_Law` prefers a clause that says *governed by / construed* over one
   that merely mentions complying with laws.

## Data — CUAD via the Kaggle dataset

`all_reshaped_clauses.csv` lists CUAD clause spans per contract but without offsets;
`complete.json` has the full text. `src/contract_ner/data/cuad.py` re-locates each span with a
whitespace/quote/case-insensitive search (98.1 % of the 4,308 spans for the five categories used),
anchors party aliases to the preamble, resolves overlaps and splits 80/10/10 by contract (398 / 54 /
47; 11 of the 510 contracts have no aligned span of these five types and are dropped).
`clause_boundary_data.csv` (ContractNLI sentences, unlabelled) is not used.

Train-split spans: `PARTY` 1,953 · `AGREEMENT_DATE` 363 · `START_DATE` 76 · `END_DATE` 361 ·
`JURISDICTION` 346. CUAD's `Parties` annotation covers only the defining mention in the preamble;
`Expiration Date` and `Governing Law` are clause-level spans — both shape the scores above.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

contract-ner download              # Kaggle dataset (no credentials needed) → data/raw/kaggle
contract-ner build-cuad            # CUAD alignment                  → data/processed/cuad

contract-ner profile --config configs/english_cuad.yaml --out reports/profile --stem cuad_english   # label distribution first
contract-ner train --config configs/english_cuad_label_aware_augmented.yaml   # ~35 min on an M2 (MPS); the shipped recipe
contract-ner evaluate --model models/english-cuad-roberta-label-aware-augmented \
    --data data/processed/cuad/val.jsonl data/processed/cuad/test.jsonl --out reports   # --min-confidence 0.6 by default
contract-ner evaluate --model regex --data data/processed/cuad/test.jsonl --out reports   # pattern-only baseline
pip install -e ".[baseline]" && contract-ner train-crf --config configs/english_cuad.yaml --out models/english-cuad-crf
contract-ner evaluate --model models/english-cuad-crf --data data/processed/cuad/test.jsonl --out reports   # sequence-model baseline
contract-ner extract  --model models/english-cuad-roberta-label-aware-augmented --file my_contract.txt --out my_contract.json
python -m pytest -q                # unit tests (Thai tests skip without the "thai" extra)
```

`scripts/reproduce.sh` runs both tracks end to end. Model weights and generated data are git-ignored
(reproducible from the commands above); everything measured is committed under `reports/`:

| Path | What it holds |
|---|---|
| `reports/comparison.md` | every model × split × view in one place, with the slot table per cell and recall per surface form — regenerate with `python scripts/make_comparison_table.py` |
| `reports/<model>__<split>.json` | the full report: the four views, each false positive and miss with its text, `forms` (recall per surface form) and `slot_errors` (wrong table cells with the expected value) |
| `reports/profile/` | the label distribution of every split, in Markdown and JSON |
| `reports/training/` | the configuration, training loss and per-epoch validation score of each fine-tune, kept outside the git-ignored model directories |

Python API:

```python
from contract_ner.predict import Extractor
from contract_ner.postprocess import assemble

ex = Extractor("models/english-cuad-roberta-label-aware-augmented")
record = assemble(text, ex.predict_with_confidence(text), lang=ex.lang)
record["summary"]        # one value per field
record["entities"]       # every span with offsets, confidence and validity
```

## Demo — web page and HTTP API

```bash
pip install -e ".[demo]"                   # FastAPI + uvicorn; add ",thai" for the Thai contract
contract-ner serve                         # http://127.0.0.1:8000/   API docs: /docs
contract-ner serve --cached-only           # no model is loaded: demo contracts answer from stored records
```

One process serves a single static page (`src/contract_ner/web/`, no build step, works offline) and
the API behind it (`src/contract_ner/api.py`):

* **Extract** — the contract with every predicted span highlighted by label (dashed below 0.8
  confidence, dotted when span validation rejected it) next to the metadata table; clicking a cell
  scrolls to the span it came from. On a curated contract, *Compare with gold* outlines the gold
  spans and marks each cell right / wrong / missed / *not in gold*, with the same keys as the slot
  evaluation. Paste or upload a `.txt` contract, move the confidence floor, download the JSON record.
* **Compare models** — one contract through the regex baseline, the conditional random field and
  the fine-tuned RoBERTa (or the Thai model), the cells they disagree on marked.
* **Evaluation** — slot F1 by model against the 0.80 target, the English model across iterations,
  slot F1 per table cell; numbers read from `reports/`.

| Endpoint | Does |
|---|---|
| `GET /api/models` | demo models, whether their weights are present, test-split numbers |
| `GET /api/samples`, `GET /api/samples/{id}` | curated demo contracts (text, gold spans, talking point) |
| `POST /api/extract` | `{model, text \| sample_id, min_confidence}` → the `assemble()` record (+ `gold_check` for a curated contract) |
| `POST /api/extract/file` | the same for an uploaded UTF-8 `.txt` file |
| `POST /api/compare` | `{models, text \| sample_id}` → each model's table and the cells they disagree on |
| `GET /api/metrics` | slot-level numbers per model and the iteration history |

The demo contracts in `demo/samples/` are picked by `scripts/pick_demo_docs.py` from the CUAD test
split (all cells right · parties the baselines miss · the End_Date selection error · a governing law
the gold lacks · the longest contract), plus the two examples and one Thai dev-gold contract; each
keeps every model's record, so a model that fails to load on the day falls back to it
(`"source": "cache"`). `scripts/record_demo.py` drives the page in a headless browser and writes a
captioned walkthrough video and one screenshot per step to `demo/recording/` — the fallback when
the live demo cannot run. The presentation script is `docs/demo_script_th.md`.

## Extending to Thai (experimental)

The pipeline keeps everything language-specific behind `contract_ner.lang.LangRules`
(word segmentation, span-validation keywords, romanisation), resolved lazily by `rules_for(lang)`.
A new language needs three things:

1. **an encoder + word segmenter** for the language (a YAML config in `configs/` naming the base
   model, its `lang` and its `entity_types`);
2. **a `LangRules`** module in `src/contract_ner/lang/` (the toolkit it imports is only loaded when
   that language is used);
3. **annotated contracts in the same JSONL format** — `{"text", "entities": [{"start","end","label"}]}`,
   which is what Label Studio / doccano export.

Thai is the worked example. Install the extra and build the data:

```bash
pip install -e ".[dev,thai]"
contract-ner build-thai            # synthetic Thai data + gold set → data/processed/thai
contract-ner train --config configs/thai_wangchanberta.yaml     # ~35 min on an M2
contract-ner evaluate --model models/thai-wangchanberta \
    --data data/processed/thai/test_gold.jsonl data/processed/thai/test_gold_blind.jsonl
contract-ner extract  --model models/thai-wangchanberta --file examples/thai_service_contract.txt --summary
contract-ner sample -n 2           # peek at generated Thai contracts and their annotations
```

There is no public annotated Thai contract corpus, so the Thai model is bootstrapped from a
template-based synthetic corpus and evaluated on a small hand-written gold set:

* `src/contract_ner/data/thai_synthetic.py` renders contracts from templates with tracked offsets:
  9 contract families with their own role vocabulary, many surface forms per entity (Thai numerals,
  พ.ศ./ค.ศ., `ร้อยละ 15 ต่อปี` vs `15% ต่อเดือน`, `2.5 ล้านบาท`, `USD 48,000`, Latin names inside Thai
  text), and hard negatives for the "ambiguous numeric context" risk (signatories, addresses,
  registration numbers, statutory citations, VAT, deposits, loan interest, delivery deadlines).
* `data/thai_gold/` — 24 hand-written contracts (217 entities, inline `[[text|LABEL]]` mark-up) used
  for error analysis between iterations; `data/thai_gold_blind/` — 8 more (73 entities) written after
  the last iteration and never inspected.
* Thai has no spaces, so PyThaiNLP `newmm` defines the words BIO tags live on, with a domain
  dictionary that splits compounds gluing a non-entity word to an entity word (`เขตอำนาจศาล`,
  `ตามกฎหมาย`, `อัตราร้อยละ`) and adds contract vocabulary (`เที่ยวละ`, `ลงวันที่`, court names).
  WangchanBERTa is uncased, so ASCII letters are lower-cased before tokenisation.

Thai results (strict / relaxed / field / slot F1, `reports/thai-*.json`):

| Test set | docs | entities | strict P / R / F1 | relaxed F1 | field F1 | slot F1 |
|---|---:|---:|---|---:|---:|---:|
| **Blind gold** — written after all iteration, never inspected | 8 | 73 | 0.741 / 0.863 / **0.797** | 0.886 | 0.808 | 0.903 |
| Dev gold — used for error analysis between iterations | 24 | 217 | 0.885 / 0.922 / **0.903** | 0.953 | 0.902 | 0.964 |
| Synthetic held-out (same generator as training) | 300 | 2,662 | 0.997 / 0.993 / 0.995 | 0.997 | 0.994 | 1.000 |

Per type (strict F1, blind / dev): START_DATE 1.00 / 0.95 · END_DATE 0.88 / 0.98 · AGREEMENT_DATE 0.82 / 0.94 ·
CONTRACT_VALUE 0.94 / 0.85 · JURISDICTION 0.88 / 0.96 · PARTY 0.83 / 0.91 · PENALTY_RATE 0.67 / 0.88 ·
PAYMENT_DUE 0.42 / 0.78. How the score moved across iterations (dev gold, strict / relaxed F1):

| Iteration | What changed | dev gold |
|---|---|---|
| v1 | raw-text subword encoding, stock newmm dictionary, generator v1 | 0.733 / 0.846 |
| v1 + dictionary | same model, domain-adapted newmm dictionary (splits `เขตอำนาจศาล`) | 0.748 / 0.852 |
| v2 | generator v2 from the error analysis: `ลงวันที่`, explicit deadlines, role aliases, summary sheets, per-unit penalties, more hard negatives | 0.894 / 0.962 |
| v3 | per-word subword encoding (no piece straddles a word boundary) | 0.899 / 0.953 |
| **v4** | split `ตามกฎหมาย` / `อัตราร้อยละ` compounds, consistent penalty-span convention, evaluation after span validation | **0.897 / 0.946** (blind 0.792 / 0.881) |
| v4 + confidence ≥ 0.6 (same weights; the current reports) | drop spans whose mean label probability is under 0.6 | 0.903 / 0.953 (blind 0.797 / 0.886) |

Annotation conventions (applied identically in the generator and the gold set):

| Label | Span covers | Not covered |
|---|---|---|
| `PARTY` | every mention of a party's legal name | role words (ผู้ว่าจ้าง), signatories, witnesses, English name in parentheses |
| `*_DATE` | from the day number through the year (`1 มกราคม พ.ศ. 2569`) | the word วันที่ |
| `CONTRACT_VALUE` | amount + currency (+ adjacent period: `เดือนละ 45,000 บาท`) | amount in words, VAT remarks, instalments, deposits |
| `PAYMENT_DUE` | the whole timing condition (`ภายใน 30 วันนับจากวันที่ได้รับใบแจ้งหนี้`) | the verb ชำระ |
| `PENALTY_RATE` | rate/amount through the per-period word (`ร้อยละ 0.1 ของมูลค่าสัญญาต่อวัน`) | basis phrases after the period, penalty caps |
| `JURISDICTION` | court, arbitral institution or governing-law phrase | surrounding clause text |

The Thai model has never seen a real contract and the gold sets are small and written by the same
author as the templates, so even the blind score is optimistic for contracts in the wild; replacing
the synthetic corpus with real annotated contracts is the obvious next step, and `PAYMENT_DUE` /
`PENALTY_RATE` are where it would pay off first.

## Repository layout

```
configs/                 YAML training configs (english_cuad.yaml, english_cuad_label_aware.yaml, thai_wangchanberta.yaml)
data/thai_gold*/         hand-annotated Thai contracts (dev / blind), inline mark-up
examples/                sample contracts for `contract-ner extract`
reports/                 document-level evaluation reports (JSON); profile/ label distribution; iteration*/ earlier runs
scripts/reproduce.sh     end-to-end reproduction of both tracks
src/contract_ner/
  schema.py              entity types, BIO labels, Document / Span records
  lang/                  LangRules per language: en.py (regex), th.py (PyThaiNLP, imported lazily)
  tokenization.py        subword alignment (per-word encoding) and windowing
  data/cuad.py           CUAD alignment + split                  data/thai_synthetic.py  Thai generator
  data/thai_gold.py      [[text|LABEL]] mark-up parser           data/thai_lexicon.py    Thai vocab
  profile.py             label distribution before training      train.py                HF Trainer fine-tuning
  evaluate.py            strict/relaxed/field/slot P/R/F1 + recall per surface form
  augment.py             label folding + date-format augmentation slots.py               table-cell (slot) scoring
  predict.py             windowed inference + BIO decoding       postprocess.py          validation + normalisation + JSON
  baseline.py            regex-only English extractor (`--model regex`), scored by the same evaluator
  baseline_crf.py        conditional random field over hand-designed features, same interface
  cli.py                 `contract-ner` commands
  api.py                 demo HTTP API (`contract-ner serve`)    web/                    demo page (HTML/CSS/JS)
demo/samples/            curated demo contracts with every model's stored record
demo/recording/          walkthrough screenshots (and the git-ignored video) from scripts/record_demo.py
tests/                   pytest suite (Thai cases skip without the "thai" extra)
```

## Limitations and next steps

* **CUAD's annotation style is not an NER guideline** — only the defining party mention is gold,
  and `Expiration Date` / `Governing Law` are clause-level; strict scores understate what the
  metadata table gets right (hence the field-level metric). Re-annotating every party mention, or
  scoring only the preamble, would make strict precision meaningful.
* **`Effective Date` is rarely distinct from `Agreement Date`** in CUAD (10 test spans), so the
  model does not learn to separate them; `Start_Date` is inferred (flagged) or split from the
  agreement-date candidates by context, and it is the weakest cell (slot F1 0.75).
* **English covers five of eight fields** — CUAD has no clean contract-value / payment-due /
  penalty-rate spans. Its `Liquidated Damages` category is a whole-clause span and was left out.
  Short-span categories such as *Notice Period to Terminate Renewal* / *Renewal Term* could be added
  without new annotation.
* **A prompt-based large-language-model run is the missing baseline** — the pattern-only and the
  conditional random field baselines are in (`--model regex`, `contract-ner train-crf`); an
  instruction-following model on the same split would complete the comparison.
* **`End_Date` is where the shipped recipe gives something back** (0.84 → 0.78 on test). Its recall
  per surface form is unchanged, so the loss is in choosing between several correct term clauses in
  one contract, not in finding them — the next move for that cell is in the selection rules in
  `postprocess.py`, not another fine-tune.
* **Clause-level slot values are compared without their edge punctuation** (`slots.value_key`):
  five of twelve `End_Date` "errors" on the validation split differed from the gold only by the full
  stop closing the sentence. Every model in the tables above was re-scored under that rule; the
  starred rows in the iteration table predate it.
* **Folding `START_DATE` into `AGREEMENT_DATE` did not move the table** (slot 0.840 → 0.828, within
  the noise of 47 documents) although span recall rose; the recipe is kept as
  `configs/english_cuad_merged.yaml`. Numeric agreement dates (`7/7/98`, `24 OCT 2011`) remain the
  main `Agreement_Date` misses — CUAD has almost none in training, and augmentation alone taught
  the tagger to fire on page-footer dates instead.
* **Gold gaps** — of the thirteen slots the shipped model predicts where CUAD has no gold, six name a
  real value (governing-law clauses for Nevada and the PRC, three term clauses, one agreement date),
  one is borderline, and six are model errors (notice or cure periods picked as `End_Date`, an
  exclusivity expiry, a file-footer date, a wrong `Start_Date`); re-checking the 47 test contracts
  by hand would raise the ceiling.
* **Plain text in, no normalisation** — real contracts arrive as PDF / scans; OCR errors will lower
  the scores (spans are matched exactly, no fuzzy alias matching).
* **Synthetic Thai training data** — see [Extending to Thai](#extending-to-thai-experimental).
* Ideas from the deck's "what's next": layout-aware parsing of PDFs/DOCX, more entity types
  (termination, confidentiality terms), legal-domain encoders (`legal-bert`); `deberta-v3-base` was
  tried (best span scores, table not separable from RoBERTa on these splits).
