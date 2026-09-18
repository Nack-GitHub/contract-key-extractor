# Contract Key-Information Extraction (Thai + English NER)

ระบบดึงข้อมูลสำคัญจากสัญญาอัตโนมัติ — อ่านข้อความสัญญาแล้วสกัด "ตัวแปร" สำคัญ
(คู่สัญญา วันที่ทำสัญญา/เริ่ม/สิ้นสุด มูลค่าสัญญา กำหนดชำระเงิน อัตราค่าปรับ เขตอำนาจศาล)
ออกมาเป็นตาราง Metadata (JSON) ด้วย Named Entity Recognition + span post-processing
ตามแนวทางในเอกสารนำเสนอ *Automated Key Information Extraction from Contracts*.

Two models share one code path:

| Track | Base model | Training data | Evaluation data |
|---|---|---|---|
| **Thai** (primary) | `airesearch/wangchanberta-base-att-spm-uncased` | 3,000 synthetic, auto-annotated Thai contracts | 24 hand-annotated Thai contracts (`data/thai_gold/`) + 300 held-out synthetic |
| **English** | `roberta-base` | CUAD spans re-aligned from the Kaggle [Legal Contract Dataset](https://www.kaggle.com/datasets/changethan/legal-contract-dataset) (398 contracts) | 47 held-out CUAD contracts |

> **About the dataset.** The Kaggle dataset is *English* (CUAD + ContractNLI). The project brief
> targets Thai contracts, and the deck (slide 5) already positions CUAD as an English reference for
> the entity schema, with Thai contracts to be self-collected and annotated. This repo therefore
> (1) uses the Kaggle data for what it is good at — real, labelled English contract spans — and
> (2) bootstraps the Thai model from a template-based synthetic corpus, evaluated on a small
> hand-written gold set. Replacing the synthetic corpus with real annotated Thai contracts
> (Label Studio / doccano exports drop straight into the same JSONL format) is the obvious next step.

## What it extracts

| Output field | Entity label | Thai model | English model (CUAD) |
|---|---|---|---|
| `Party_A`, `Party_B` | `PARTY` (A/B assigned by order of appearance) | ✅ | ✅ (`Parties`) |
| `Agreement_Date` | `AGREEMENT_DATE` | ✅ | ✅ (`Agreement Date`) |
| `Start_Date` | `START_DATE` | ✅ | ✅ (`Effective Date`; falls back to Agreement_Date) |
| `End_Date` | `END_DATE` | ✅ | ✅ (`Expiration Date`) |
| `Contract_Value` | `CONTRACT_VALUE` | ✅ | — (no CUAD category) |
| `Payment_Due_Date` | `PAYMENT_DUE` | ✅ | — |
| `Penalty_Rate` | `PENALTY_RATE` | ✅ | — |
| `Jurisdiction` | `JURISDICTION` (court / arbitration / governing law) | ✅ | ✅ (`Governing Law`) |

Example — the illustrative contract from slide 12 (`examples/thai_service_contract.txt`):

```bash
contract-ner extract --model models/thai-wangchanberta --file examples/thai_service_contract.txt --summary
```

```json
__THAI_SUMMARY__
```

The full record (`--summary` omitted) keeps every predicted span with its character offsets and
confidence, normalised values (ISO dates with พ.ศ.→ค.ศ. conversion, amount + currency, rate + period,
court name in English) and a `valid` flag from the span-validation stage, so a reviewer can verify
the draft extraction against the source text.

## Results

__RESULTS__

## Pipeline

```
 raw text ──► word segmentation ──► subword encoding ──► fine-tuned encoder ──► BIO decode ──► span validation ──► JSON
             (PyThaiNLP newmm /     (offset-aligned,       (WangchanBERTa /      (word level,     + normalisation
              regex for English)     overlapping windows)   RoBERTa)              lenient IOB2)   (dates, money, rates, courts)
```

1. **Word segmentation** – Thai has no spaces, so PyThaiNLP `newmm` defines the word units that BIO
   tags live on (the deck's approach). English uses a small regex tokenizer.
2. **Subword alignment** – the *raw text* is fed to the HF tokenizer and each subword is mapped to a
   word through character offsets. Feeding pre-segmented words with `is_split_into_words=True` would
   insert a `▁` marker before every Thai word (≈1.6× longer inputs that look nothing like the
   pre-training text). The first content subword of each word carries the label / prediction.
3. **Windows** – long contracts (CUAD averages 52k characters) are cut into 256-subword windows with a
   64-subword overlap; at inference each subword takes the logits from the window where it is
   furthest from an edge.
4. **Fine-tuning** – standard token classification (17 BIO labels) with the HF `Trainer`, YAML-driven
   (`configs/`). WangchanBERTa is uncased, so ASCII letters are lower-cased before tokenisation
   (uppercase Latin maps to `<unk>` otherwise — relevant for mixed Thai/English contracts).
5. **Decoding + span validation** – word-level BIO → character spans; each span is checked against its
   type (a date must parse, a jurisdiction must name a court/law/arbitral body, …). This is the
   rule-based stand-in for the deck's "span classification" stage; invalid spans are kept in the
   output but never promoted to a summary field.
6. **Normalisation** – Buddhist-era and Thai-numeral dates → ISO-8601, amounts → `{amount, currency,
   per}`, penalty rates → `{value, unit, per, basis}`, courts → English names, payment terms → short
   English descriptions. Party A/B = first two distinct legal names in order of appearance (short
   quoted aliases such as `"Company"` are demoted to `Other_Parties`).

## Data

### Thai — synthetic corpus + gold set

`src/contract_ner/data/thai_synthetic.py` renders contracts from templates with tracked offsets:

* 9 contract families (จ้างทำของ/บริการ, ก่อสร้าง, ซื้อขาย, เช่า, กู้ยืม, อนุญาตให้ใช้สิทธิ, รักษาความลับ,
  ตัวแทนจำหน่าย, จ้างแรงงาน) with their own role vocabulary (ผู้ว่าจ้าง/ผู้รับจ้าง, ผู้ให้เช่า/ผู้เช่า, …);
* many surface forms per entity: `1 มกราคม 2569`, `1 ม.ค. 2569`, `01/01/2569`, `1 เดือน มกราคม พ.ศ. 2569`,
  Thai numerals (`๑ มกราคม ๒๕๖๙`), `ร้อยละ 15 ต่อปี` vs `15% ต่อเดือน`, `เดือนละ 45,000 บาท`, `2.5 ล้านบาท`,
  `USD 48,000`, company / person / government-agency parties, Latin names inside Thai text;
* **hard negatives** for the "ambiguous numeric context" risk from the deck: signatories and witnesses
  (person names that are *not* parties), addresses, registration and contract numbers, statutory
  citations (มาตรา 224), VAT (ร้อยละ 7), deposits and retention, loan interest (not a penalty),
  delivery deadlines (not a payment date), insurance limits and unit prices (not the contract value);
* long numbered contracts with signature blocks and short single-paragraph excerpts.

`data/thai_gold/` holds 24 contracts written by hand in styles the generator does **not** produce
(government construction contract in Thai numerals, land sale, franchise, MOU, key-value summary sheet,
mixed Thai/English consulting agreement, …), annotated with an inline `[[text|LABEL]]` mark-up
(215 entities). The Thai numbers reported above are measured on this set.

Annotation conventions (applied identically in the generator and the gold set):

| Label | Span covers | Not covered |
|---|---|---|
| `PARTY` | every mention of a party's legal name | role words (ผู้ว่าจ้าง), signatories, witnesses, English name in parentheses |
| `*_DATE` | from the day number through the year (`1 มกราคม พ.ศ. 2569`) | the word วันที่ |
| `CONTRACT_VALUE` | amount + currency (+ adjacent period: `เดือนละ 45,000 บาท`) | amount in words, VAT remarks, instalments, deposits |
| `PAYMENT_DUE` | the whole timing condition (`ภายใน 30 วันนับจากวันที่ได้รับใบแจ้งหนี้`) | the verb ชำระ |
| `PENALTY_RATE` | rate/amount through the per-period word (`ร้อยละ 0.1 ของมูลค่าสัญญาต่อวัน`) | basis phrases after the period, penalty caps |
| `JURISDICTION` | court, arbitral institution or governing-law phrase | surrounding clause text |

### English — CUAD via the Kaggle dataset

`all_reshaped_clauses.csv` lists CUAD clause spans per contract but without offsets;
`complete.json` has the full text. `src/contract_ner/data/cuad.py` re-locates each span with a
whitespace/quote/case-insensitive search (98.1 % of the 4,308 spans for the five categories used),
anchors party aliases to the preamble, resolves overlaps and splits 80/10/10 by contract.
`clause_boundary_data.csv` (ContractNLI sentences, unlabelled) is not used.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

contract-ner build-thai            # synthetic Thai data + gold set  → data/processed/thai
contract-ner download              # Kaggle dataset (no credentials needed) → data/raw/kaggle
contract-ner build-cuad            # CUAD alignment                  → data/processed/cuad

contract-ner train --config configs/thai_wangchanberta.yaml     # ~35 min on an M2 (MPS)
contract-ner train --config configs/english_cuad.yaml           # ~30 min

contract-ner evaluate --model models/thai-wangchanberta --data data/processed/thai/test_gold.jsonl
contract-ner extract  --model models/thai-wangchanberta --file my_contract.txt --out my_contract.json
contract-ner sample -n 2           # peek at generated Thai contracts and their annotations
python -m pytest -q                # 70+ unit tests (+2 integration tests once a model exists)
```

`scripts/reproduce.sh` runs the whole sequence. Model weights and generated data are git-ignored
(reproducible from the commands above); the evaluation reports in `reports/` are committed.

Python API:

```python
from contract_ner.predict import Extractor
from contract_ner.postprocess import assemble

ex = Extractor("models/thai-wangchanberta")
record = assemble(text, ex.predict_with_confidence(text), lang="th")
record["summary"]        # one value per field
record["entities"]       # every span with offsets, confidence and validity
```

## Repository layout

```
configs/                 YAML training configs (Thai / English)
data/thai_gold/          hand-annotated Thai contracts (inline mark-up)
examples/                sample contracts for `contract-ner extract`
reports/                 document-level evaluation reports (JSON)
scripts/reproduce.sh     end-to-end reproduction
src/contract_ner/
  schema.py              entity types, BIO labels, Document / Span records
  tokenization.py        PyThaiNLP / regex word segmentation, subword alignment, windowing
  data/thai_synthetic.py synthetic Thai contract generator      data/thai_lexicon.py  vocab
  data/thai_gold.py      [[text|LABEL]] mark-up parser           data/cuad.py          CUAD alignment
  train.py               HF Trainer fine-tuning                  evaluate.py           seqeval P/R/F1
  predict.py             windowed inference + BIO decoding       postprocess.py        validation + JSON
  cli.py                 `contract-ner` commands
tests/                   pytest suite
```

## Limitations and next steps

* **Synthetic Thai training data.** The Thai model has never seen a real contract; the gold set is
  small (24 documents) and written by the same author as the templates, so the Thai numbers are an
  optimistic estimate of real-world performance. Collecting and annotating real contracts is the
  highest-value next step — the JSONL format (`{"text", "entities": [{"start","end","label"}]}`) is
  what Label Studio / doccano export.
* **Tokenisation errors propagate.** Entity boundaries snap to `newmm` words; a mis-segmented word
  can widen a span by a character or two (the relaxed metric shows how often this matters).
* **English covers five of eight fields** — CUAD has no clean contract-value / payment-due /
  penalty-rate spans. Its `Liquidated Damages` category is a whole-clause span and was left out.
* **Rule-based normalisation** handles the common Thai/English formats but is not exhaustive
  (e.g. payment terms that are neither a date, a day-of-month nor an "N days after X" pattern are
  returned verbatim).
* Ideas from the deck's "what's next": layout-aware parsing of PDFs/DOCX (this repo consumes plain
  text), more entity types (termination, confidentiality terms), and a prompt-based Thai LLM
  baseline to compare against fine-tuning.
