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
| **English** (primary) | `roberta-base` | CUAD spans re-aligned into full text (398 contracts) | 47 held-out CUAD contracts, 370 gold spans | `pip install -e ".[dev]"` |
| Thai (experimental) | `airesearch/wangchanberta-base-att-spm-uncased` | 3,000 synthetic, auto-annotated Thai contracts | 24 + 8 (blind) hand-annotated Thai contracts + 300 held-out synthetic | `pip install -e ".[dev,thai]"` |

## What it extracts

| Output field | Entity label | CUAD category | English model | Thai model |
|---|---|---|---|---|
| `Party_A`, `Party_B` | `PARTY` (A/B assigned by order of appearance) | `Parties` | ✅ | ✅ |
| `Agreement_Date` | `AGREEMENT_DATE` | `Agreement Date` | ✅ | ✅ |
| `Start_Date` | `START_DATE` | `Effective Date` | ✅ (falls back to `Agreement_Date`) | ✅ |
| `End_Date` | `END_DATE` | `Expiration Date` | ✅ | ✅ |
| `Jurisdiction` | `JURISDICTION` (court / arbitration / governing law) | `Governing Law` | ✅ | ✅ |
| `Contract_Value` | `CONTRACT_VALUE` | — (no short-span CUAD category) | — | ✅ |
| `Payment_Due_Date` | `PAYMENT_DUE` | — | — | ✅ |
| `Penalty_Rate` | `PENALTY_RATE` | — | — | ✅ |

Each model is trained on its own label space (`entity_types` in its YAML config): 11 BIO labels for
English (B/I × 5 + O), 17 for Thai.

```bash
contract-ner extract --model models/english-cuad-roberta --file examples/english_service_agreement.txt --summary
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
span validation, exactly as `contract-ner extract` does. Three views are reported:

* **strict** – exact span match (seqeval, word level);
* **relaxed** – same type and overlapping span (a one-word boundary slip still counts);
* **field** – per document, the *set of distinct values* per type is compared — what matters for a
  metadata table (a party tagged five times is one value).

47 held-out CUAD contracts, 370 gold spans, five entity types (`reports/english-cuad-roberta__test.json`):

| Iteration | strict F1 | relaxed F1 | field F1 |
|---|---:|---:|---:|
| v1 — raw-text BPE encoding, no validation, 17 labels | 0.254 | 0.607 | 0.358 |
| v3 — per-word encoding (punctuation no longer glued to party names) | 0.514 | 0.654 | 0.603 |
| v4 — v3 + span validation (`reports/iteration2/`) | 0.584 | 0.735 | 0.657 |
| **final** — v4 retrained on the 5-type label space (11 labels) | **0.619** | **0.738** | **0.688** |

strict P / R = 0.553 / 0.703 · relaxed P / R = 0.660 / 0.838

| Type | gold spans | relaxed F1 | field F1 |
|---|---:|---:|---:|
| PARTY | 244 | 0.77 | 0.73 |
| AGREEMENT_DATE | 42 | 0.73 | 0.66 |
| START_DATE | 10 | 0.00 | 0.00 |
| END_DATE | 38 | 0.64 | 0.61 |
| JURISDICTION | 36 | 0.89 | 0.76 |

Recall is high; precision is limited by CUAD's annotation style — only the *defining* mention of each
party is gold, so later mentions the model tags count as false positives at span level (86
of the 200 recorded errors are `PARTY` false positives), and `Expiration Date` / `Governing
Law` are whole clauses whose exact boundaries are hard to reproduce. CUAD's `Effective Date` usually
coincides with the agreement date (only 10 distinct `START_DATE` spans in the test split), which the
model never learns to separate; the JSON output therefore falls back to `Start_Date = Agreement_Date`
(flagged `inferred_from`). The narrative for the presentation is in
`docs/midterm_presentation_cuad.md`.

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
   next to the weights (`pipeline_config.json`).
5. **Decoding + span validation** – word-level BIO → character spans; each span is checked against its
   type by surface form (a short date must parse; a jurisdiction must contain court / law /
   arbitration / tribunal / jurisdiction; a party must contain a letter and not be a stop-word).
   This is the rule-based stand-in for the deck's "span classification" stage; invalid spans stay in
   the output (`valid: false`) but are never promoted to a summary field.
6. **Normalisation** – dates → ISO-8601 (`dateutil`; Buddhist-era and Thai numerals for Thai),
   governing law / court → place name, amounts and rates → `{amount, currency, per}` (Thai fields).
   Party A/B = first two distinct legal names in order of appearance; repeated mentions merge, and
   short quoted defined terms such as `"Company"` are demoted to `Other_Parties` here.

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

contract-ner train --config configs/english_cuad.yaml           # ~30 min on an M2 (MPS)
contract-ner evaluate --model models/english-cuad-roberta --data data/processed/cuad/test.jsonl --out reports
contract-ner extract  --model models/english-cuad-roberta --file my_contract.txt --out my_contract.json
python -m pytest -q                # unit tests (Thai tests skip without the "thai" extra)
```

`scripts/reproduce.sh` runs both tracks end to end. Model weights and generated data are git-ignored
(reproducible from the commands above); the evaluation reports in `reports/` are committed.

Python API:

```python
from contract_ner.predict import Extractor
from contract_ner.postprocess import assemble

ex = Extractor("models/english-cuad-roberta")
record = assemble(text, ex.predict_with_confidence(text), lang=ex.lang)
record["summary"]        # one value per field
record["entities"]       # every span with offsets, confidence and validity
```

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

Thai results (strict / relaxed / field F1, `reports/thai-*.json`):

| Test set | docs | entities | strict P / R / F1 | relaxed F1 | field F1 |
|---|---:|---:|---|---:|---:|
| **Blind gold** — written after all iteration, never inspected | 8 | 73 | 0.733 / 0.863 / **0.792** | 0.881 | 0.803 |
| Dev gold — used for error analysis between iterations | 24 | 217 | 0.873 / 0.922 / **0.897** | 0.946 | 0.896 |
| Synthetic held-out (same generator as training) | 300 | 2,662 | 0.997 / 0.993 / 0.995 | 0.997 | 0.994 |

Per type (strict F1, blind / dev): START_DATE 1.00 / 0.95 · END_DATE 0.88 / 0.98 · AGREEMENT_DATE 0.82 / 0.94 ·
CONTRACT_VALUE 0.94 / 0.85 · JURISDICTION 0.88 / 0.96 · PARTY 0.81 / 0.89 · PENALTY_RATE 0.67 / 0.88 ·
PAYMENT_DUE 0.42 / 0.77. How the score moved across iterations (dev gold, strict / relaxed F1):

| Iteration | What changed | dev gold |
|---|---|---|
| v1 | raw-text subword encoding, stock newmm dictionary, generator v1 | 0.733 / 0.846 |
| v1 + dictionary | same model, domain-adapted newmm dictionary (splits `เขตอำนาจศาล`) | 0.748 / 0.852 |
| v2 | generator v2 from the error analysis: `ลงวันที่`, explicit deadlines, role aliases, summary sheets, per-unit penalties, more hard negatives | 0.894 / 0.962 |
| v3 | per-word subword encoding (no piece straddles a word boundary) | 0.899 / 0.953 |
| **v4** | split `ตามกฎหมาย` / `อัตราร้อยละ` compounds, consistent penalty-span convention, evaluation after span validation | **0.897 / 0.946** (blind 0.792 / 0.881) |

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
configs/                 YAML training configs (english_cuad.yaml, thai_wangchanberta.yaml)
data/thai_gold*/         hand-annotated Thai contracts (dev / blind), inline mark-up
examples/                sample contracts for `contract-ner extract`
reports/                 document-level evaluation reports (JSON); iteration*/ keeps earlier runs
scripts/reproduce.sh     end-to-end reproduction of both tracks
src/contract_ner/
  schema.py              entity types, BIO labels, Document / Span records
  lang/                  LangRules per language: en.py (regex), th.py (PyThaiNLP, imported lazily)
  tokenization.py        subword alignment (per-word encoding) and windowing
  data/cuad.py           CUAD alignment + split                  data/thai_synthetic.py  Thai generator
  data/thai_gold.py      [[text|LABEL]] mark-up parser           data/thai_lexicon.py    Thai vocab
  train.py               HF Trainer fine-tuning                  evaluate.py             strict/relaxed/field P/R/F1
  predict.py             windowed inference + BIO decoding       postprocess.py          validation + normalisation + JSON
  cli.py                 `contract-ner` commands
tests/                   pytest suite (Thai cases skip without the "thai" extra)
```

## Limitations and next steps

* **CUAD's annotation style is not an NER guideline** — only the defining party mention is gold,
  and `Expiration Date` / `Governing Law` are clause-level; strict scores understate what the
  metadata table gets right (hence the field-level metric). Re-annotating every party mention, or
  scoring only the preamble, would make strict precision meaningful.
* **`Effective Date` is rarely distinct from `Agreement Date`** in CUAD (10 test spans), so the
  model does not learn to separate them; the fallback is flagged, not silent.
* **English covers five of eight fields** — CUAD has no clean contract-value / payment-due /
  penalty-rate spans. Its `Liquidated Damages` category is a whole-clause span and was left out.
  Short-span categories such as *Notice Period to Terminate Renewal* / *Renewal Term* could be added
  without new annotation.
* **No baselines yet** — a regex-only and a CRF + features baseline on the same split, and a
  prompt-based LLM run, would turn the "why contextual embeddings" argument into numbers.
* **Plain text in, no normalisation** — real contracts arrive as PDF / scans; OCR errors will lower
  the scores (spans are matched exactly, no fuzzy alias matching).
* **Synthetic Thai training data** — see [Extending to Thai](#extending-to-thai-experimental).
* Ideas from the deck's "what's next": layout-aware parsing of PDFs/DOCX, more entity types
  (termination, confidentiality terms), larger or legal-domain encoders (`roberta-large`,
  `deberta-v3`, `legal-bert`).
