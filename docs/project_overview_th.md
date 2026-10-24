# Contract Key Extractor — เอกสารอธิบายโปรเจกต์ทั้งหมด (สำหรับทำ Presentation)

> ไฟล์นี้รวม **ทุกอย่างเกี่ยวกับโปรเจกต์นี้ไว้ที่เดียว**: สร้างมายังไง, workflow เป็นยังไง, ระบบทำงานยังไง, วัดผลยังไง, ได้ผลเท่าไร และมีข้อจำกัดอะไร
> ตัวเลขทุกตัวในเอกสารนี้ copy มาจากไฟล์จริงใน `reports/` และ `configs/` ไม่มีตัวเลขที่ประมาณเอง
> เอกสารที่เกี่ยวข้อง: [`docs/midterm_presentation_cuad.md`](midterm_presentation_cuad.md) (สคริปต์นำเสนอ), [`docs/worklog_slot_eval_and_retrain.md`](worklog_slot_eval_and_retrain.md), [`docs/worklog_label_distribution.md`](worklog_label_distribution.md), [`README.md`](../README.md)

---

## สารบัญ

1. [สรุปหนึ่งหน้า (Executive Summary)](#1-สรุปหนึ่งหน้า-executive-summary)
2. [ปัญหาและแรงจูงใจ](#2-ปัญหาและแรงจูงใจ)
3. [สถาปัตยกรรมภาพรวม](#3-สถาปัตยกรรมภาพรวม)
4. [ข้อมูล (Data)](#4-ข้อมูล-data)
5. [Pipeline ทำงานยังไง — ทีละขั้น](#5-pipeline-ทำงานยังไง--ทีละขั้น)
6. [การเทรน (Training)](#6-การเทรน-training)
7. [การวัดผล (Evaluation) — 4 มุมมอง](#7-การวัดผล-evaluation--4-มุมมอง)
8. [ผลการทดลอง](#8-ผลการทดลอง)
9. [Baseline สองตัวสำหรับเปรียบเทียบ](#9-baseline-สองตัวสำหรับเปรียบเทียบ)
10. [Language seam — รองรับหลายภาษา](#10-language-seam--รองรับหลายภาษา)
11. [Track ภาษาไทย](#11-track-ภาษาไทย)
12. [Workflow การพัฒนา — สร้างโปรเจกต์นี้มายังไง](#12-workflow-การพัฒนา--สร้างโปรเจกต์นี้มายังไง)
13. [Workflow การใช้งาน — คำสั่งตั้งแต่ต้นจนจบ](#13-workflow-การใช้งาน--คำสั่งตั้งแต่ต้นจนจบ)
14. [โครงสร้างไฟล์ในโปรเจกต์](#14-โครงสร้างไฟล์ในโปรเจกต์)
15. [คุณภาพโค้ดและการทดสอบ](#15-คุณภาพโค้ดและการทดสอบ)
16. [ข้อจำกัดและงานถัดไป](#16-ข้อจำกัดและงานถัดไป)
17. [แผนสไลด์ที่แนะนำ + ตัวเลขสำหรับแต่ละสไลด์](#17-แผนสไลด์ที่แนะนำ--ตัวเลขสำหรับแต่ละสไลด์)
18. [คำถามที่น่าจะโดนถาม (Q&A)](#18-คำถามที่น่าจะโดนถาม-qa)

---

## 1. สรุปหนึ่งหน้า (Executive Summary)

**โจทย์:** สัญญาเชิงพาณิชย์เป็น unstructured text ยาวเฉลี่ย ~52,000 ตัวอักษร (≈13,000 token) แต่สิ่งที่ฝ่ายกฎหมาย/บัญชี/จัดซื้อต้องการมีแค่ไม่กี่ช่อง → **แปลงสัญญาเป็นตาราง metadata (JSON) อัตโนมัติ**

**วิธี:** Named Entity Recognition (token classification ด้วย Transformer ที่ fine-tune) + ชั้น rule-based ทำ span validation / normalisation / การเลือกค่าเข้าตาราง

**สิ่งที่สกัดได้ (track อังกฤษ 5 ช่อง):** คู่สัญญา (Party A/B), วันลงนาม, วันเริ่มมีผล, วันสิ้นสุด/เงื่อนไขสิ้นสุด, กฎหมาย/เขตอำนาจศาลที่ใช้บังคับ

**Dataset:** CUAD (Contract Understanding Atticus Dataset) ผ่าน Kaggle *Legal Contract Dataset* → 499 สัญญาที่ใช้ได้ แบ่ง 398 / 54 / 47 (train / val / test) แบ่งระดับ "สัญญา" ไม่ใช่ระดับ span

**โมเดลที่ ship:** `roberta-base` fine-tuned, 11 BIO labels (B/I × 5 ประเภท + O), recipe `configs/english_cuad_label_aware_augmented.yaml`

**ผลลัพธ์หลัก (test split 47 สัญญา):**

| มุมมองการวัด | F1 |
|---|---:|
| strict (span ตรงเป๊ะ) | 0.545 |
| relaxed (span ซ้อนทับ + ประเภทถูก) | 0.720 |
| field (set ของค่าที่ไม่ซ้ำต่อเอกสาร) | 0.640 |
| **slot (ช่องในตาราง metadata จริง)** | **0.855** |

**ตัวเลขที่ต้องพูดบนเวที:** *slot F1 = 0.855* คือ "ช่องในตารางที่ผู้ใช้เห็นถูก 85.5%" ซึ่งผ่านเกณฑ์ ≥ 0.80 (ใช้งานได้โดยมีคน review) ส่วน strict F1 ต่ำเพราะ **วิธี annotate ของ CUAD** (tag แค่ mention แรกของคู่สัญญา, clause-level spans) ไม่ใช่เพราะโมเดลหาไม่เจอ

**Bonus:** มี track ภาษาไทย (WangchanBERTa + ข้อมูลสังเคราะห์ 3,000 สัญญา + gold set เขียนมือ) อยู่ใน repo เดียวกันหลัง language seam — blind gold strict F1 = 0.797 / slot F1 = 0.903

---

## 2. ปัญหาและแรงจูงใจ

### 2.1 ปัญหาคืออะไร

| ช่องข้อมูล | คำถามที่ตอบ | ตัวอย่างจาก CUAD |
|---|---|---|
| **Parties** | ใครเป็นคู่สัญญา | Diversinet Corp. / 2205925 Ontario Limited |
| **Agreement Date** | ลงนามเมื่อไร | July 1, 2020 |
| **Effective Date** | เริ่มมีผลเมื่อไร | (มักตรงกับวันลงนาม) |
| **Expiration Date** | หมดอายุเมื่อไร / ต่ออายุยังไง | "shall continue for a period of five (5) years" |
| **Governing Law** | ใช้กฎหมายรัฐ/ประเทศไหน | State of New York |

ในนิยามของ NLP นี่คือ **NLU** (unstructured → structured representation) และเป็นงาน **NER** โดยตรง: หา span ในข้อความ แล้วติด label ประเภท

### 2.2 ทำไมงานนี้ยากในเชิง NLP

1. **Lexical / contextual ambiguity** — `July 1, 2020` เป็น Agreement Date หรือ Effective Date หรือวันส่งมอบงวดแรก? ตัวเลขเหมือนกัน ต่างกันที่ประโยครอบข้าง · `"Company"` เป็น *defined term* (นามแฝง) ไม่ใช่ชื่อนิติบุคคล
2. **Long document** — สัญญายาวเกิน context window ของ encoder หลายเท่า ต้องทำ windowing + stitching
3. **Class imbalance สุดขั้ว** — ตัวอักษรที่อยู่ใน gold span คิดเป็นเพียง **1.03%** ของ corpus; >90% ของ window ไม่มี entity เลย
4. **หนึ่ง label ไม่ใช่สตริงชนิดเดียว** — `PARTY` เป็นทั้งชื่อนิติบุคคล, alias ในเครื่องหมายคำพูด, และ boilerplate "referred to as the Parties"; `END_DATE` เป็นทั้งวันที่และ clause ยาว 30 คำ
5. **ผลลัพธ์ต้องตรวจสอบย้อนกลับได้** — ทุกค่าต้องมี character offset + confidence เพื่อให้คน review ได้ (ไม่ใช่ generate ข้อความใหม่)

### 2.3 ทำไมไม่ใช้ LLM ไปเลย

ขอบเขตของโปรเจกต์นี้ตั้งใจ **ไม่ใช้ LLM** เพราะ (ก) ต้องการ character offset ที่ตรวจสอบได้ ไม่ใช่ข้อความที่ generate ขึ้นมา (ข) ต้องรันบนเครื่อง local (M2 MacBook) ได้ (ค) ต้องการวัดว่า pre-training ให้ประโยชน์เท่าไร โดยเทียบกับ baseline แบบ regex และ CRF บน split เดียวกัน — "prompt-based LLM baseline" ถูกระบุไว้เป็นงานที่ยังขาดใน Limitations

---

## 3. สถาปัตยกรรมภาพรวม

```
 raw text ──► word segmentation ──► subword encoding ──► fine-tuned encoder ──► BIO decode ──► span validation ──► JSON
             (contract_ner.lang:     (offset-aligned,       (RoBERTa /            (word level,     + normalisation
              regex / PyThaiNLP)      overlapping windows)   WangchanBERTa)        lenient IOB2)   (dates, places, courts)
```

หลักการออกแบบ 4 ข้อ:

1. **Core เป็นภาษากลาง (language-agnostic)** — `schema.py`, `tokenization.py`, `predict.py`, `evaluate.py` ไม่รู้จักภาษาใดภาษาหนึ่ง สิ่งที่ต่างกันระหว่างภาษาอยู่หลัง seam เดียวคือ `contract_ner.lang.LangRules`
2. **Label space ผูกกับโมเดล ไม่ใช่ผูกกับ repo** — แต่ละโมเดลประกาศ `entity_types` ใน YAML config แล้วเก็บไว้ข้าง weights (`pipeline_config.json`) → อังกฤษ 11 labels, ไทย 17 labels อยู่ร่วมกันได้
3. **ข้อความต้นฉบับห้ามถูกแก้** — ไม่มีขั้นตอน "normalise ข้อความ" ทุก span ต้อง map กลับไปยัง character offset เดิมได้ (ความยืดหยุ่นเรื่อง whitespace/quote อยู่ใน regex ตอน align ข้อมูล ไม่ใช่ในตัวข้อความ)
4. **Rule-based ทำเฉพาะงานที่ rule ทำได้ดีกว่า** — การ validate รูปแบบ, การแปลงเป็น ISO date, การเลือกค่าเข้าช่องตาราง — ไม่ใช่การหา entity

### โมดูลหลักและหน้าที่

| โมดูล | บรรทัด | หน้าที่ |
|---|---:|---|
| `schema.py` | 195 | ประเภท entity, BIO label space, `Document` / `Span`, แปลง span ↔ BIO |
| `lang/__init__.py` | 61 | `LangRules` dataclass + `rules_for(lang)` (import toolkit แบบ lazy) |
| `lang/en.py` | 154 | regex ตัดคำอังกฤษ, keyword validation, การจัดกลุ่ม surface form |
| `lang/th.py` | 194 | PyThaiNLP newmm + domain dictionary, กฎไทย, romanisation |
| `tokenization.py` | 262 | per-word subword encoding + windowing (หัวใจทางเทคนิค) |
| `train.py` | 246 | HF `Trainer` fine-tuning จาก YAML config |
| `predict.py` | 165 | windowed inference + stitching + BIO decoding + confidence |
| `postprocess.py` | 549 | span validation, normalisation (วันที่/เงิน/อัตรา/เขตอำนาจ), ประกอบ record |
| `slots.py` | 212 | วัดผลระดับ "ช่องในตาราง" |
| `evaluate.py` | 219 | strict / relaxed / field / slot + recall รายรูปแบบ |
| `profile.py` | 399 | วิเคราะห์การกระจายตัวของ label **ก่อน** เทรน |
| `augment.py` | 161 | label folding + date-format augmentation + party capitalisation |
| `data/cuad.py` | 191 | align CUAD clause → character span + แบ่ง split |
| `data/thai_synthetic.py` | 1,039 | generator สัญญาไทยสังเคราะห์พร้อม offset |
| `baseline.py` / `baseline_crf.py` | 163 / 244 | baseline regex และ CRF (interface เดียวกับ Extractor) |
| `cli.py` | 315 | คำสั่ง `contract-ner` ทั้ง 10 คำสั่ง |

---

## 4. ข้อมูล (Data)

### 4.1 CUAD ผ่าน Kaggle — และปัญหาที่ต้องแก้ก่อน

Kaggle release (`changethan/legal-contract-dataset`) มี 2 ไฟล์ที่ใช้:

* `all_reshaped_clauses.csv` — clause spans ต่อสัญญา (`filename, clause_type, clause_text`) แต่ **ไม่มี character offset** (หายไปตอน reshape)
* `complete.json` — ข้อความสัญญาฉบับเต็ม
* (`clause_boundary_data.csv` เป็นประโยค ContractNLI ไม่มี label — ไม่ได้ใช้)

**ปัญหา:** NER ต้องการ offset แต่ข้อมูลไม่มี → `src/contract_ner/data/cuad.py` ทำ **span re-alignment**:

1. สร้าง regex จาก clause text ที่ยืดหยุ่นเรื่อง whitespace / เครื่องหมายคำพูด / ตัวพิมพ์เล็กใหญ่
2. ค้นหาตำแหน่งในข้อความเต็ม (`_occurrences`, จำกัด 50 ตำแหน่ง)
3. **anchor alias ของคู่สัญญาไปที่ preamble** (ชื่อคู่สัญญาปรากฏหลายสิบครั้ง แต่ CUAD tag เฉพาะ mention แรก)
4. แก้ overlap ด้วยลำดับความสำคัญ `AGREEMENT_DATE > PARTY > END_DATE > JURISDICTION > START_DATE`
5. ตัด `(Page N)` ท้าย span ออก

**ผลการ align: 98.1% ของ 4,308 spans** ใน 5 category ที่ใช้ (regression test ล็อกตัวเลขนี้ไว้ใน `tests/test_cuad.py`)

### 4.2 Mapping CUAD category → entity type

| CUAD category | entity type ของเรา |
|---|---|
| Parties | `PARTY` |
| Agreement Date | `AGREEMENT_DATE` |
| Effective Date | `START_DATE` |
| Expiration Date | `END_DATE` |
| Governing Law | `JURISDICTION` |

CUAD **ไม่มี** category ที่สะอาดพอสำหรับ contract value / payment due / penalty rate (`Liquidated Damages` เป็น clause ทั้งย่อหน้า) → track อังกฤษจึงครอบคลุม 5 จาก 8 ช่องใน schema

### 4.3 การแบ่ง split

* แบ่ง **80/10/10 ระดับสัญญา** ด้วย hash ของ id (deterministic, ไม่มี leak ระหว่าง split)
* สัญญาทั้งหมด 510 ฉบับ → **11 ฉบับไม่มี span ที่ align ได้เลยใน 5 ประเภทนี้ → ตัดทิ้ง → เหลือ 499**
* ผลลัพธ์: **train 398 / val 54 / test 47**

| Split | เอกสาร | ตัวอักษรเฉลี่ย | spans | PARTY | AGREEMENT_DATE | START_DATE | END_DATE | JURISDICTION |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 398 | 51,476 | 3,099 | 1,953 | 363 | 76 | 361 | 346 |
| val | 54 | 49,218 | 408 | 244 | 50 | 16 | 47 | 51 |
| test | 47 | 62,073 | 370 | 244 | 42 | 10 | 38 | 36 |

### 4.4 การวิเคราะห์ label ก่อนเทรน (`contract-ner profile`)

นี่คือขั้นตอนที่ทำให้ recipe การเทรนเปลี่ยนไป — รันก่อน fine-tune ทุกครั้ง เขียนรายงานลง `reports/profile/`:

| Label | spans | เอกสารที่มี | ความยาว (คำ, median / p90) | อยู่ใน 5% แรกของเอกสาร | รูปแบบหลัก |
|---|---:|---:|---|---:|---|
| `PARTY` | 1,953 | 100% | 2 / 6 | 82% | alias ในเครื่องหมายคำพูด 36% · legal name 34% · ตัวพิมพ์ใหญ่ทั้งหมด 12% · boilerplate 6% |
| `AGREEMENT_DATE` | 363 | 90% | 3 / 5 | 83% | `Month D, YYYY` 53% · `Dth day of Month` 25% · ตัวเลข 6% · ช่องว่างไม่กรอก 5% |
| `START_DATE` | 76 | **18%** | 10 / 26 | 28% | **ไม่มีวันที่เลย 40%** · มีวันที่ 60% |
| `END_DATE` | 361 | 80% | **31 / 66** | 8% | duration 44% · "until terminated" 27% · วันที่ชัดเจน 22% |
| `JURISDICTION` | 346 | 83% | **29 / 51** | **0%** (93% อยู่ครึ่งหลัง) | "governed by / construed" 94% |

**3 ข้อสรุปที่นำไปแก้ recipe ตรง ๆ:**

1. **สองปัญหาใช้ tagger ตัวเดียวกัน** — `PARTY`/`AGREEMENT_DATE` เป็น span สั้นใน preamble ส่วน `END_DATE`/`JURISDICTION` เป็น clause ในครึ่งหลัง → การเฉลี่ย exact-span F1 ข้ามทั้งสองไม่มีความหมาย (นี่คือเหตุผลเชิงโครงสร้างที่ strict กับ slot ห่างกันมาก)
2. **`START_DATE` มี 2 รูปทรงใน 76 ตัวอย่าง** — วันที่ 3 คำ กับ clause นิยาม 26 คำ → ตัด span ที่ไม่มีวันที่ออกจาก **training data** (gold ที่ใช้วัดผลไม่แตะ) ให้ label มีรูปทรงเดียว
3. **window ที่ไม่มี entity ไม่เท่ากันทุกอัน** — negative ที่มีค่าคือช่วงต้นเอกสาร (ที่อยู่, เลขทะเบียน, page header ที่หน้าตาเหมือน entity) → เก็บ window จาก 10% แรกของเอกสารที่ 0.50 ส่วนที่เหลือ 0.12 (จำนวน window รวมเท่าเดิม แต่ negative ยากขึ้น)

นอกจากนี้ยังตรวจ window length: ที่ 256 subword overlap 64 มีเพียง **14 จาก 3,099 spans (0.45%)** ที่ตกคร่อมขอบ window → window ยาวกว่านี้เสียเวลาเทรนโดยไม่ได้อะไร

---

## 5. Pipeline ทำงานยังไง — ทีละขั้น

### ขั้น 1 — Word segmentation (`lang/en.py`, `lang/th.py`)

หน่วยที่ BIO tag อาศัยอยู่คือ "คำ" ไม่ใช่ subword

* **อังกฤษ:** regex `\w+(?:[-'’.]\w+)*|[^\w\s]` — กลุ่มตัวอักษร/ตัวเลขที่มี hyphen/apostrophe/จุดภายในได้ (`X-Y`, `Inc.`) หรือเครื่องหมายวรรคตอนเดี่ยว
* **ไทย:** ไม่มีช่องว่างระหว่างคำ → PyThaiNLP `newmm` + **domain dictionary** ที่แยกคำประสมซึ่งเอาคำที่ไม่ใช่ entity ไปติดกับคำที่เป็น entity (`เขตอำนาจศาล`, `ตามกฎหมาย`, `อัตราร้อยละ`) และเพิ่มคำศัพท์สัญญา (`เที่ยวละ`, `ลงวันที่`, ชื่อศาล)
* whitespace ไม่เคยเป็นคำ

### ขั้น 2 — Subword alignment แบบ per-word (หัวใจทางเทคนิค)

**ปัญหาที่แก้:** ถ้าโยนข้อความดิบเข้า tokenizer ทั้งก้อน BPE จะ merge `"),` เป็น piece เดียว และ SentencePiece จะ merge `ของกฎหมาย` เป็น piece เดียว → คำแรก/คำสุดท้ายของ entity **ไม่มี subword เป็นของตัวเอง** → label หายไป

**วิธีแก้:** tokenize ทีละคำ แล้วเติม prefix marker (`Ġ` / `▁`) กลับเข้าไป **เฉพาะตำแหน่งที่มีช่องว่างจริงนำหน้า** (`per_word_backend()` patch ตัว Rust backend: `Metaspace.prepend_scheme="never"`, ถอด `WhitespaceSplit` ออก)

ผลลัพธ์: pieces เหมือนกับตอน tokenize ข้อความดิบทุกประการ **ยกเว้น** ไม่มี piece ไหนคร่อมขอบคำได้

> **ทำไมไม่ใช้ `is_split_into_words=True`?** เพราะมันจะเติม marker ให้ **ทุกคำ** ซึ่งไม่เหมือนตอน pre-training (ภาษาไทยจะได้ sequence ยาวขึ้น ~1.6–2 เท่า) — คำถามนี้น่าจะโดนถามบนเวที มีคำตอบอยู่ใน `tokenization.py` docstring

**ผลกระทบเชิงตัวเลข:** เปลี่ยนจาก raw-text encoding เป็น per-word encoding ดัน strict F1 ของอังกฤษจาก **0.254 → 0.514** (v1 → v3)

จากนั้น: subword แต่ละตัว map ไปยังคำที่มีตัวอักษรแรกของมัน; subword "ตัวแทน" ของแต่ละคำ (ตัวแรกที่มีเนื้อหาจริง ไม่ใช่ bare `▁` marker) เป็นตัวที่ถือ label ตอนเทรนและถือ prediction ตอน inference

### ขั้น 3 — Windowing

* สัญญายาวเฉลี่ย 52k ตัวอักษร ≈ 13k token แต่ encoder รับได้ 512 → ตัดเป็น **window ละ 256 subword, overlap 64**
* ตอนเทรน: window ที่ไม่มี entity เลยถูก sample ลง (ตาม `negative_window_keep`)
* ตอน inference: subword แต่ละตัวเอา logits จาก **window ที่มันอยู่ห่างขอบมากที่สุด** (มี context รอบตัวมากที่สุด) — นี่คือขั้น "stitching"

### ขั้น 4 — Fine-tuning

Token classification มาตรฐานด้วย HF `Trainer` ขับเคลื่อนด้วย YAML (ดูหัวข้อ 6)

### ขั้น 5 — BIO decoding + Span validation

* decode แบบ **lenient IOB2**: `I-X` ที่ไม่ได้ต่อจาก `X` จะเปิด entity ใหม่ (โมเดลที่ลืม `B-` ยังได้ span ที่ใช้ได้)
* **ตัด span ที่ mean label probability < 0.6** ก่อน (`--min-confidence`, ค่า default 0.6) — tagger ปล่อย fragment เช่น `,` หรือ `18/` ออกมาที่ ≈0.5 ซึ่งไม่มีกฎ surface ไหนจับได้; threshold เลือกจาก **validation split เท่านั้น**
* **Span validation** (`lang/en.py:validate_span`) — ตรวจว่า surface form เป็นไปได้ไหมสำหรับ type นั้น:
  * วันที่สั้น (≤4 คำ) ต้อง parse เป็นวันที่ได้จริง
  * `END_DATE` ที่เป็น clause ต้องพูดถึงการสิ้นสุด/ระยะเวลา (`year|month|expir|terminat|until|renew|…`) — ตัวเลขอย่างเดียวไม่นับ ("Act of 1976")
  * `JURISDICTION` ต้องมีคำว่า **court / law / arbitra / tribunal / jurisdiction**
  * `PARTY` ต้องมีตัวอักษรอย่างน้อยหนึ่งตัว, ไม่ใช่ stop-word, ไม่ใช่ legal-form suffix เปล่า ๆ (`Corp`), ไม่ใช่คำเดี่ยวตัวพิมพ์เล็กที่ไม่ได้อยู่ในเครื่องหมายคำพูด
  * span ที่ไม่ผ่าน **ยังอยู่ใน output** (`"valid": false`) แต่ไม่ถูกเลื่อนขึ้นไปเป็นค่าในช่องสรุป — เพื่อให้คน review เห็น

### ขั้น 6 — Normalisation + การประกอบ record (`postprocess.py`)

* **วันที่ → ISO-8601** รองรับ: `Month D, YYYY`, `Dth day of Month`, `the first day of July, 2017` (สะกดเป็นคำ), `7/7/98` (ปี 2 หลัก pivot ที่ 70), `24 OCT 2011`, พ.ศ. (−543), เลขไทย (๐-๙); **ตัวเลขล้วนอ่านแบบ month-first สำหรับอังกฤษ / day-first สำหรับไทย**
* **Governing law / court → ชื่อสถานที่** (รองรับ clause ที่เป็นตัวพิมพ์ใหญ่ทั้งหมด)
* **จำนวนเงิน/อัตรา → `{amount, currency, per}`** (ใช้ใน track ไทย)
* **Party A/B** = สองชื่อนิติบุคคลแรกตามลำดับที่ปรากฏ; mention ซ้ำถูกรวมเข้าด้วยกัน; **alias สั้นในเครื่องหมายคำพูด (`"Company"`) และ boilerplate "referred to as the Parties" ถูกลดชั้นไปที่ `Other_Parties`**
* **การแยก Agreement vs Start date** — ถ้า candidate ของ agreement date ชี้ไปวันที่ต่างกันสองค่า ตัวที่มีคำว่า *effective* / *commence* นำหน้าภายใน 40 ตัวอักษรจะกลายเป็น `Start_Date` ส่วนตัวที่เหลือที่มาก่อนสุดเป็น `Agreement_Date`; ถ้ามีค่าเดียว `Start_Date` จะถูก **infer** จาก `Agreement_Date` พร้อมธง `"inferred_from"`
* **การเลือก Governing_Law** — เลือก clause ที่บอกว่า *governed by / construed* มากกว่า clause ที่แค่พูดถึงการปฏิบัติตามกฎหมาย

### ตัวอย่าง output จริง

```bash
contract-ner extract --model models/english-cuad-roberta-label-aware-augmented \
    --file examples/english_service_agreement.txt --summary
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

ถ้าไม่ใส่ `--summary` จะได้ record เต็มที่มี **ทุก span พร้อม character offset, confidence, ค่า normalised และธง `valid`** (ดู `reports/example_english_service_agreement.json`)

---

## 6. การเทรน (Training)

### 6.1 Config-driven ทั้งหมด

ทุกอย่างอยู่ใน YAML — ไม่มีการแก้โค้ดเพื่อเทรนโมเดลใหม่

```yaml
name: english-cuad-roberta-label-aware-augmented
model_name: roberta-base
lang: en
entity_types: [PARTY, AGREEMENT_DATE, START_DATE, END_DATE, JURISDICTION]   # → 11 BIO labels
dateless_labels_to_drop: [START_DATE]   # เฉพาะ training data; gold ที่ใช้วัดผลไม่แตะ
date_augment: 0.16
party_case_augment: 0.16
per_word: true
max_length: 256
stride: 64
epochs: 4
lr: 3.0e-5
batch_size: 16
warmup_ratio: 0.1
weight_decay: 0.01
seed: 42
negative_window_keep: 0.12                 # เนื้อเอกสาร
negative_window_keep_document_head: 0.50   # 10% แรก ที่ negative ยาก ๆ อยู่
document_head_fraction: 0.10
```

### 6.2 Hyperparameters ที่ใช้จริง

| ค่า | อังกฤษ (shipped) | อังกฤษ (baseline config) | ไทย | DeBERTa |
|---|---|---|---|---|
| base model | `roberta-base` | `roberta-base` | `wangchanberta-base-att-spm-uncased` | `deberta-v3-base` |
| entity types | 5 (11 labels) | 5 (11 labels) | 8 (17 labels) | 4 (9 labels) |
| epochs | 4 | 3 | 3 | 3 |
| learning rate | 3e-5 | 3e-5 | 3e-5 | 2e-5 |
| batch size | 16 | 16 | 16 | 8 × grad_accum 2 |
| window / stride | 256 / 64 | 256 / 64 | 256 / 64 | 256 / 64 |
| negative keep | 0.12 (head 0.50) | 0.15 | 1.0 | 0.3 |
| lowercase | false | false | **true** (base model เป็น uncased) | false |
| เวลาเทรนบน M2 (MPS) | ~35 นาที | ~30 นาที | ~35 นาที | ~2 ชม. 15 นาที |

### 6.3 รายละเอียดทางเทคนิคที่ควรรู้

* **`save_total_limit=1`, `load_best_model_at_end=True`, `metric_for_best_model="f1"`** — เลือก checkpoint ที่ดีที่สุดจาก validation (window-level seqeval F1)
* **ลบโฟลเดอร์ checkpoint ทิ้งหลังเทรนเสร็จ** เพื่อประหยัดพื้นที่ดิสก์บนเครื่อง local
* **โหลด model เป็น `torch.float32` เสมอ** — checkpoint ของ deberta-v3 ship มาเป็น fp16 ถ้า fine-tune ต่อโดยไม่มี loss scaling จะ diverge เป็น NaN ภายใน ~100 steps บน MPS
* **สิ่งที่เก็บไว้หลังเทรน** ข้าง weights: `pipeline_config.json` (lang, lowercase, per_word, max_length, stride, base_model, **entity_types**, **label_map**) + `train_log.json`; และ copy ของ config/loss/validation ต่อ epoch ไปเก็บไว้ที่ `reports/training/` (ซึ่งไม่ถูก git-ignore)

### 6.4 การ augment ข้อมูล (`augment.py`)

เพิ่มเข้ามาเพราะ **recall รายรูปแบบ** ชี้ให้เห็นว่าพลาดอะไร:

| เทคนิค | ทำอะไร | ทำไม |
|---|---|---|
| `dateless_labels_to_drop` | ตัด span ของ `START_DATE` ที่ไม่มีวันที่ออกจาก training data | 40% ของ `START_DATE` เป็น clause นิยามยาว 26 คำ ≠ วันที่ 3 คำ → label เดียวสองรูปทรง tagger เรียนไม่ได้ทั้งคู่ |
| `date_augment: 0.16` | เขียนวันที่ในข้อมูลเทรนใหม่เป็น `7/7/98`, `3-31-16`, `24 OCT 2011`, `Month ___, YYYY` (re-base offset ทั้งหมด) | วันที่แบบตัวเลขเป็น 5.5% ของ training spans และโมเดลเดิมหาไม่เจอเลยสักตัวบน validation |
| `party_case_augment: 0.16` | เขียนชื่อคู่สัญญาเป็นตัวพิมพ์ใหญ่ทั้งหมด (ความยาวเท่าเดิม → offset ไม่ขยับ) | ชื่อตัวพิมพ์ใหญ่ถูกหาเจอ 66% เทียบกับ 83% สำหรับชื่อ title-case |
| `label_map` | พับ label หนึ่งเข้าอีก label (`START_DATE → AGREEMENT_DATE`) | ใช้ใน experiment merged/DeBERTa; evaluation พับ gold แบบเดียวกันเพื่อความเป็นธรรม |

> **ข้อควรระวังที่บันทึกไว้:** การ augment วันที่แบบตัวเลขในอัตราสูง (0.25) ทำให้ tagger ไปยิงวันที่ใน page footer แทน — จึงลดเหลือ 0.16 และเก็บ negative window จากช่วงต้นเอกสารไว้เยอะ

---

## 7. การวัดผล (Evaluation) — 4 มุมมอง

ทุกตัวเลขวัด **ระดับเอกสาร**: สัญญาทั้งฉบับผ่าน windowing → stitching → BIO decoding → span validation เหมือนที่ `contract-ner extract` ทำจริง

| มุมมอง | นิยาม | ตอบคำถามอะไร |
|---|---|---|
| **strict** | span ตรงเป๊ะระดับคำ (seqeval) | tagger ตรงกับ annotation แค่ไหน |
| **relaxed** | ประเภทถูก + span ซ้อนทับกัน (พลาด boundary 1 คำยังนับถูก) | หา entity เจอไหม (ไม่สนขอบเขตเป๊ะ) |
| **field** | ต่อเอกสาร เทียบ **set ของค่าที่ไม่ซ้ำ** ต่อประเภท (คู่สัญญาที่ถูก tag 5 ครั้ง = 1 ค่า) — ยังเทียบข้อความดิบ | ยุบ mention ซ้ำแล้วดีขึ้นไหม |
| **slot** | **ตาราง metadata จริง** — record ที่ `assemble()` สร้าง (หนึ่งค่าต่อช่อง หลัง normalisation) เทียบกับตารางที่ได้จาก gold spans | **ผู้ใช้เห็นอะไร — นี่คือตัวเลขที่ตัดสินโปรเจกต์** |

### slot-level วัดยังไง (`slots.py`)

* **Gold ของ slot** สร้างจาก gold spans ด้วย normaliser ชุดเดียวกับ pipeline → clause ที่ normalise แล้วได้ค่าเดียวกันถือว่าถูก ไม่ต้องเป็น string เดียวกัน
* **Pred ของ slot** คือ record ที่ `assemble()` สร้างจริง
* นิยามช่อง:
  * `Parties` — set ของชื่อนิติบุคคล (ตัด alias และ boilerplate ออกทั้งสองฝั่ง); ชื่อตรงกันเมื่อ normalised form เท่ากันหรือฝั่งหนึ่งเป็น substring ของอีกฝั่ง
  * `Agreement_Date` / `Start_Date` — ISO date (ถ้า gold `START_DATE` เป็น clause ไม่มีตัวเลข จะ fallback ไป agreement date สะท้อนกฎ `inferred_from`)
  * `End_Date` — ISO date ถ้ามี ไม่งั้นเป็น duration key เช่น `3 year` (`three (3) years from the Effective Date` = `a period of 3 years`)
  * `Jurisdiction` — สถานที่จาก governing-law clause
* ช่องที่ gold ว่างและ pred ว่าง = true negative ไม่นับ; ค่าผิด = FP + FN
* **เทียบโดยไม่สนเครื่องหมายวรรคตอนที่ขอบ** (`slots.value_key`) — 5 จาก 12 "error" ของ `End_Date` บน validation ต่างจาก gold แค่จุดปิดประโยค

### recall รายรูปแบบ (`forms`)

ทุก report มีตารางนี้: gold spans ถูกจัดกลุ่มตาม **วิธีเขียน** แล้วรายงาน recall ต่อกลุ่ม — เพราะ "F1 ของ `AGREEMENT_DATE` = 0.59" ซ่อนความจริงว่าโมเดลหา `June 21, 1999` เจอทุกตัวแต่หา `7/7/98` ไม่เจอเลย

ตัวอย่างของโมเดลที่ ship (test split): เจอ agreement date แบบ `Month D, YYYY` **18/18** (เดิม 16/18) · แบบตัวเลข **2/5** (เดิม 0/5) · ชื่อคู่สัญญา title-case **74/90** (เดิม 69/90) · ชื่อตัวพิมพ์ใหญ่ **32/35** (เดิม 29/35)

---

## 8. ผลการทดลอง

### 8.1 เส้นทางการพัฒนา — test split (47 สัญญา, 370 gold spans)

| Iteration | strict F1 | relaxed F1 | field F1 | **slot F1** |
|---|---:|---:|---:|---:|
| v1 — raw-text BPE encoding, ไม่มี validation, 17 labels | 0.254 | 0.607 | 0.358 | — |
| v3 — per-word encoding (เครื่องหมายวรรคตอนไม่ติดกับชื่อคู่สัญญาแล้ว) | 0.514 | 0.654 | 0.603 | — |
| v4 — v3 + span validation | 0.584 | 0.735 | 0.657 | — |
| v5 — v4 retrain บน label space 5 ประเภท (11 labels) | 0.619 | 0.738 | 0.688 | — |
| v6 — v5 + span validation อังกฤษที่เข้มขึ้น | 0.633 | 0.754 | 0.702 | 0.804\* |
| v7 — v6 + normalisation, แยก effective date, เลือก governing law, confidence ≥ 0.6 (**weights เดิม**) | 0.652 | 0.759 | 0.722 | 0.840 |
| v8 — v7 + recipe ที่ reshape ตาม label distribution | 0.547 | 0.715 | 0.638 | 0.845 |
| **final — v8 + date/party augmentation** | **0.545** | **0.720** | **0.640** | **0.855** |
| *DeBERTa-v3-base (merged recipe)* | *0.663* | *0.762* | *0.743* | *0.807* |
| *merged-label retrain (START_DATE พับเข้า AGREEMENT_DATE)* | *0.630* | *0.806* | *0.691* | *0.839* |
| *CRF baseline* | *0.509* | *0.586* | *0.576* | *0.690* |
| *regex-only baseline* | *0.497* | *0.705* | *0.545* | *0.762* |

\* v6 เป็นแถวเดียวที่วัดก่อนกฎ "ไม่สนเครื่องหมายวรรคตอนที่ขอบ clause" แถวอื่นถูก re-score ใหม่ทั้งหมดด้วยเกณฑ์ปัจจุบัน

**Precision / Recall ของโมเดลที่ ship:** strict P/R = 0.475 / 0.641 · relaxed P/R = 0.627 / 0.846 · **slot P/R = 0.867 / 0.843**
บน validation split: slot F1 = **0.874** (เทียบ 0.853 ของ v7) → recipe ชนะทั้งสอง split

> **ประเด็นที่ต้องอธิบายบนเวที:** strict/field ของ final **ต่ำกว่า** v7 แต่ slot **สูงกว่า** เพราะโมเดลตอนนี้ tag คู่สัญญา *ทุก* mention ขณะที่ CUAD gold มีแค่ mention ที่นิยามไว้ → นับเป็น false positive ที่ระดับ span แต่ตารางที่ออกมาดีขึ้น

### 8.2 ผลระดับช่องในตาราง (slot) — ตัวเลขที่ควรขึ้นสไลด์

| ช่องในตาราง | เอกสารที่มี gold | **final slot F1 (P / R)** | v7 | CRF | regex |
|---|---:|---|---:|---:|---:|
| Parties (set ของชื่อนิติบุคคล) | 47 | **0.871** (0.891 / 0.853) | 0.84 | 0.61 | 0.80 |
| Agreement_Date | 42 | **0.825** (0.868 / 0.786) | 0.83 | 0.81 | 0.74 |
| Start_Date | 42 | **0.800** (0.842 / 0.762) | 0.75 | 0.59 | 0.68 |
| End_Date (วันที่หรือระยะเวลา) | 32 | **0.783** (0.730 / 0.844) | 0.84 | 0.72 | 0.55 |
| Jurisdiction (สถานที่ตามกฎหมายที่ใช้บังคับ) | 34 | **0.957** (0.943 / 0.971) | 0.96 | 0.96 | 0.88 |
| **micro** | — | **0.855** (0.867 / 0.843) | 0.840 | 0.690 | 0.762 |

### 8.3 ผลระดับ span รายประเภท (final model, test)

relaxed / field F1: `PARTY` 0.735 / 0.75 · `AGREEMENT_DATE` 0.589 / 0.64 · `START_DATE` **0.00 / 0.00** (gold แค่ 10 spans, 4 ตัวเป็น clause ไม่มีวันที่ที่ recipe ตัดทิ้ง) · `END_DATE` 0.731 / 0.37 · `JURISDICTION` 0.886 / 0.35

### 8.4 ทำไม slot ขยับมากกว่า span มาก — คำอธิบายที่ต้องเตรียมไว้

ช่องว่างสองช่วงเป็นคนละเรื่องกัน:

* **span → field** คือสไตล์ "หนึ่ง mention ต่อสัญญา" ของ CUAD — **43 จาก 77** false positive ของ `PARTY` ที่รอดมาเป็นการซ้ำของ gold party ที่อื่นในสัญญาเดียวกัน
* **field → slot** คือ normalisation กับการเลือกค่า — governing-law clause ตัวที่สองที่ถูกพอ ๆ กัน, clause ที่เป็นตัวพิมพ์ใหญ่, ชื่อคู่สัญญาที่ขาดไป 1 token, `three (3) years` เทียบกับ `a period of 3 years` → `Jurisdiction` ขยับจาก 0.76 → 0.96 **บน prediction ชุดเดียวกัน**
* error ที่เหลือคือ model error จริง หรือ **gold gap**: จาก 13 ช่องที่โมเดล final ทายมีค่าแต่ gold ว่าง (ตรวจด้วยมือครบทุกช่อง) มี **6 ช่องเป็นค่าจริงที่ CUAD ไม่ได้ annotate** (governing law ของ Nevada และ PRC, term clause 3 อัน, วันที่ทำสัญญา 1 อัน), 1 ช่องก้ำกึ่ง (ข้อ terminate ที่ไม่ใช่ระยะสัญญา) และ **6 ช่องเป็น model error จริง** (ระยะ notice/cure ถูกหยิบเป็น End_Date, วันหมดสิทธิ์ exclusivity, วันที่จาก footer ของไฟล์, Start_Date ที่ผิดวัน) → ตัวเลขบน CUAD gold เป็น **floor** ไม่ใช่เพดาน

### 8.5 Experiment ที่ทำแล้วไม่เอา (มีค่าเล่าบนเวที)

* **merged-label retrain** (พับ `START_DATE` เข้า `AGREEMENT_DATE` + date augment 0.25 + เก็บ negative window เยอะขึ้น 2 เท่า): span recall ดีขึ้นจริง (relaxed 0.759 → 0.806) แต่ **ตารางไม่ขยับ** (slot 0.840 → 0.828 = 2 ช่องจาก 47 เอกสาร) และรูปแบบตัวเลขที่ test พลาดก็ยังพลาดอยู่ → recipe ที่ง่ายกว่าชนะ
* **DeBERTa-v3-base** (2 ชม. 15 นาที บน M2): span score ดีที่สุดทั้งสอง split (strict 0.663 test / 0.729 val) และตารางดีที่สุดบน validation (slot 0.869) แต่ตารางบน test ต่ำกว่า (0.807 — พลาด agreement date 9 ตัวที่ RoBERTa หาเจอ) → **encoder สองตัวที่ห่างกัน 0.02 ในทิศทางตรงข้ามบน split 47/54 เอกสาร แยกกันไม่ออกทางสถิติ** → เก็บ RoBERTa ที่เล็กกว่าและเร็วกว่า 1.5 เท่าไว้

---

## 9. Baseline สองตัวสำหรับเปรียบเทียบ

ทั้งคู่ใช้ **interface เดียวกับ `Extractor`** (`lang`, `entity_types`, `predict_with_confidence`) จึงถูกวัดด้วย evaluator ตัวเดียวกัน บน split เดียวกัน ผ่าน span validation และ normalisation ชุดเดียวกัน — **ทุกอย่างคุมคงที่หมด เหลือแค่ตัวโมเดลที่ต่าง**

### 9.1 Regex-only (`--model regex`, `baseline.py`)

อ่าน preamble ด้วย pattern: ชื่อนิติบุคคลที่ลงท้ายด้วย `Inc.` / `Ltd.` / …, defined term ในเครื่องหมายคำพูด, วันที่แบบ *dated as of* / *effective as of*, ประโยคแรกที่พูดถึง term และประโยคแรกที่ *governed by … laws of*

**ผล:** slot F1 = 0.762 — ใกล้โมเดลมากในช่องที่อยู่ใน preamble (Parties 0.80 เทียบ 0.87, Agreement_Date 0.74 เทียบ 0.83) แต่ห่างมากที่ `End_Date` (0.55 เทียบ 0.78 — term clause เขียนได้เป็นร้อยแบบ) และที่ recall ของ `Jurisdiction`

### 9.2 Linear-chain CRF (`contract-ner train-crf`, `baseline_crf.py`)

feature ที่ออกแบบมือ: word shape (`"Acme"` → `Xxxx`), capitalisation, affix, gazetteer คำศัพท์กฎหมาย, สองคำซ้ายและขวา, ตำแหน่งในเอกสาร — **คือวิธีมาตรฐานของ NER ก่อนยุค pre-trained encoder** เทรนด้วยข้อมูลชุดเดียวกัน ตัดเป็นประโยค (ระวังไม่ให้จุดของ `Inc.` ตัดกลางชื่อคู่สัญญา)

**ผล:** slot F1 = 0.753 (val) / 0.690 (test) เทียบกับ 0.874 / 0.855 ของ encoder

**บทเรียน:** CRF สู้ได้ดีเมื่อการตัดสินใจเป็น **local** (`Jurisdiction` 0.96, `Agreement_Date` 0.81 — ชื่อศาลหรือวันที่ดูจากคำรอบ ๆ ก็รู้) แต่พังเมื่อต้องใช้ทั้งเอกสาร (`Parties` 0.61 เทียบ 0.87 เพราะ feature แยกไม่ออกว่า mention ไหนคือ mention ที่นิยามคู่สัญญา)
→ **ระยะห่างระหว่าง CRF กับ encoder คือส่วนที่ pre-training ให้มา** โดยคุมทุกอย่างอื่นคงที่

---

## 10. Language seam — รองรับหลายภาษา

ทุกอย่างที่ต่างกันระหว่างภาษาอยู่ใน `LangRules` เพียงจุดเดียว:

```python
@dataclass(frozen=True)
class LangRules:
    lang: str
    segment_words: Callable[[str], list[tuple[int, int]]]   # text → offsets ของคำ
    validate_span: Callable[[Span], bool]                   # กฎ keyword เฉพาะภาษา
    romanize: Callable[[str], str]                          # ชื่อสถานที่ → อักษรละติน
    span_forms: dict[str, tuple[str, ...]]                  # label → รูปแบบ surface ที่เป็นไปได้
    span_form: Callable[[Span], str]                        # span → รูปแบบ (ใช้ใน profile + evaluate)

@functools.cache
def rules_for(lang: str) -> LangRules:
    if lang == "en":  from .en import RULES
    elif lang == "th": from .th import RULES   # import PyThaiNLP ที่นี่เท่านั้น
    ...
```

**ผลที่ได้จริง:**
* `pip install -e ".[dev]"` (track อังกฤษ) **ไม่ลง PyThaiNLP เลย** — test ภาษาไทยขึ้นสถานะ `skipped` ไม่ใช่ `failed`
* `pip install -e ".[dev,thai]"` → ครบทั้งสอง track
* เพิ่มภาษาใหม่ = **3 อย่าง**: (1) encoder + word segmenter (YAML config ที่ระบุ base model, `lang`, `entity_types`), (2) โมดูล `LangRules` ใน `src/contract_ner/lang/`, (3) สัญญาที่ annotate แล้วในรูปแบบ JSONL เดียวกัน (`{"text", "entities": [{"start","end","label"}]}` — คือสิ่งที่ Label Studio / doccano export ออกมา) — **ไม่ต้องแก้ core ของ pipeline**

**Label space ก็แยกกันด้วย:** แต่ละโมเดลเก็บ `entity_types` ไว้ใน `pipeline_config.json` ข้าง weights → `Extractor` และ `evaluate` วนเฉพาะประเภทของโมเดลตัวนั้น (โมเดลเก่าที่ไม่มี key นี้ default เป็นทุกประเภท → backward compatible)

---

## 11. Track ภาษาไทย

### 11.1 ปัญหา: ไม่มี dataset สาธารณะ

ไม่มี corpus สัญญาภาษาไทยที่ annotate แล้วเปิดสาธารณะ → bootstrap ด้วย 2 ทาง:

1. **ข้อมูลสังเคราะห์** (`data/thai_synthetic.py`, 1,039 บรรทัด) — render สัญญาจาก template พร้อมติดตาม offset: **9 ตระกูลสัญญา** ที่มีคำศัพท์บทบาทของตัวเอง, หลาย surface form ต่อ entity (เลขไทย ๐-๙, พ.ศ./ค.ศ., `ร้อยละ 15 ต่อปี` vs `15% ต่อเดือน`, `2.5 ล้านบาท`, `USD 48,000`, ชื่อภาษาอังกฤษในข้อความไทย) และ **hard negative** สำหรับความเสี่ยง "บริบทตัวเลขกำกวม" (ผู้ลงนาม, ที่อยู่, เลขทะเบียน, การอ้างมาตรากฎหมาย, VAT, เงินมัดจำ, ดอกเบี้ยเงินกู้, กำหนดส่งมอบ)
2. **Gold set เขียนมือ** — `data/thai_gold/` 24 สัญญา (217 entities) ใช้ทำ error analysis ระหว่าง iteration; `data/thai_gold_blind/` **8 สัญญา (73 entities) เขียนหลัง iteration สุดท้ายและไม่เคยถูกเปิดดู** — ใช้ mark-up inline `[[ข้อความ|LABEL]]`

| Split ไทย | เอกสาร | spans |
|---|---:|---:|
| train (สังเคราะห์) | 3,000 | 26,880 |
| val (สังเคราะห์) | 300 | 2,547 |
| test สังเคราะห์ | 300 | 2,662 |
| dev gold (เขียนมือ) | 24 | 217 |
| **blind gold** | **8** | **73** |

### 11.2 ผลภาษาไทย

| Test set | เอกสาร | entities | strict P / R / F1 | relaxed F1 | field F1 | slot F1 |
|---|---:|---:|---|---:|---:|---:|
| **Blind gold** — เขียนหลัง iteration ทั้งหมด ไม่เคยเปิดดู | 8 | 73 | 0.741 / 0.863 / **0.797** | 0.886 | 0.808 | **0.903** |
| Dev gold — ใช้ทำ error analysis | 24 | 217 | 0.885 / 0.922 / **0.903** | 0.953 | 0.902 | 0.964 |
| Synthetic held-out (generator เดียวกับ training) | 300 | 2,662 | 0.997 / 0.993 / 0.995 | 0.997 | 0.994 | 1.000 |

**ช่องว่างระหว่าง synthetic 0.995 กับ blind 0.797 คือระยะห่างระหว่าง "ข้อมูลที่ generator สร้าง" กับ "ภาษาที่คนเขียนจริง"** — และเป็นเหตุผลที่ต้องมี blind set

รายประเภท (strict F1, blind / dev): `START_DATE` 1.00 / 0.95 · `END_DATE` 0.88 / 0.98 · `AGREEMENT_DATE` 0.82 / 0.94 · `CONTRACT_VALUE` 0.94 / 0.85 · `JURISDICTION` 0.88 / 0.96 · `PARTY` 0.83 / 0.91 · `PENALTY_RATE` 0.67 / 0.88 · `PAYMENT_DUE` 0.42 / 0.78

### 11.3 เส้นทาง iteration ของภาษาไทย (dev gold, strict / relaxed F1)

| Iteration | เปลี่ยนอะไร | dev gold |
|---|---|---|
| v1 | raw-text subword encoding, newmm dictionary มาตรฐาน, generator v1 | 0.733 / 0.846 |
| v1 + dictionary | โมเดลเดิม แต่ใช้ domain dictionary (แยก `เขตอำนาจศาล`) | 0.748 / 0.852 |
| v2 | generator v2 จาก error analysis: `ลงวันที่`, กำหนดเวลาชัดเจน, alias บทบาท, ตารางสรุป, ค่าปรับต่อหน่วย, hard negative เพิ่ม | 0.894 / 0.962 |
| v3 | per-word subword encoding | 0.899 / 0.953 |
| v4 | แยก `ตามกฎหมาย` / `อัตราร้อยละ`, มาตรฐาน span ของค่าปรับ, วัดผลหลัง span validation | 0.897 / 0.946 (blind 0.792 / 0.881) |
| v4 + confidence ≥ 0.6 | ตัด span ที่ mean probability < 0.6 (weights เดิม) | **0.903 / 0.953** (blind 0.797 / 0.886) |

### 11.4 อะไรที่ต่างจากอังกฤษในทางเทคนิค

* **ไม่มีช่องว่างระหว่างคำ** → PyThaiNLP `newmm` เป็นตัวกำหนดว่า "คำ" คืออะไร พร้อม domain dictionary ที่แยกคำประสม
* **WangchanBERTa เป็น uncased** → ต้อง lower-case เฉพาะ ASCII ก่อน tokenize (`lowercase_ascii` ไม่เปลี่ยนความยาวสตริง เพื่อไม่ให้ offset เพี้ยน)
* **พ.ศ. / เลขไทย** → `_fix_year` ลบ 543 เมื่อปี > 2400 เว้นแต่มี ค.ศ. กำกับ
* **วันที่ตัวเลขล้วนอ่านแบบ day-first** (ต่างจากอังกฤษที่ month-first)
* **การเก็บ negative window = 1.0** เพราะสัญญาไทยสังเคราะห์สั้น (~1,350 ตัวอักษร) ต่างจาก CUAD ที่ยาว 52k

### 11.5 ข้อจำกัดที่ต้องพูดตรง ๆ

โมเดลไทย **ไม่เคยเห็นสัญญาจริง** และ gold set เล็ก + เขียนโดยคนเดียวกับที่เขียน template → แม้แต่ blind score ก็ยัง optimistic สำหรับสัญญาในโลกจริง การแทนที่ corpus สังเคราะห์ด้วยสัญญาจริงที่ annotate แล้วคือก้าวถัดไปที่ชัดเจนที่สุด และ `PAYMENT_DUE` / `PENALTY_RATE` คือจุดที่จะได้ผลตอบแทนก่อน

---

## 12. Workflow การพัฒนา — สร้างโปรเจกต์นี้มายังไง

### 12.1 ประวัติ commit (8 commits, 18–22 ก.ย. 2026)

| วันที่ | commit | ทำอะไร |
|---|---|---|
| 2026-09-18 | `e37647f` | Pipeline แรก (ไทย + อังกฤษ): schema, tokenization, train, predict, postprocess, evaluate, CLI |
| 2026-09-18 | `76e1870` | **Iteration 1**: error analysis → domain dictionary, generator ที่รวยขึ้น |
| 2026-09-18 | `300ae7f` | **Per-word subword encoding** + blind gold set + การนับ boundary ที่ถูกต้อง |
| 2026-09-18 | `5698076` | โมเดลสุดท้าย, evaluation ที่ผ่าน validation, README พร้อมผล |
| 2026-09-21 | `d051f57` | **Label space ต่อโมเดล**: `entity_types` ใน config, Encoder/evaluate วนตามประเภทของโมเดล |
| 2026-09-21 | `b06f57e` | **Language seam**: `lang/{en,th}`, PyThaiNLP ย้ายไปอยู่หลัง extra `[thai]` |
| 2026-09-21 | `71bf0aa` | CUAD-first docs: retrain โมเดลอังกฤษเป็น 11 labels, แก้เอกสาร midterm, ปรับโครงสร้าง README |
| 2026-09-22 | `86f27df` | **Slot-level evaluation**, การวิเคราะห์ label distribution, baseline ตัวที่สอง (CRF) |

### 12.2 วิธีทำงาน: spec-driven development

แต่ละรอบใหญ่มี 3 ไฟล์ที่เขียน **ก่อน** แตะโค้ด:

1. **`SPEC.md`** — Objective, Assumptions (ที่ยืนยันกับผู้ใช้ก่อน), Tech Stack, Commands, Project Structure, Code Style, Testing Strategy, **Boundaries** (Always / Ask first / Never), **Success Criteria** ที่วัดได้, Open Questions
2. **`tasks/plan.md`** — dependency graph ว่าอะไรต้องมาก่อนอะไร, architecture decisions พร้อมเหตุผล, ตารางความเสี่ยง + วิธีลดความเสี่ยง, จุดที่ขนานกันได้
3. **`tasks/todo.md`** — task ย่อยพร้อม **acceptance criteria แบบ checkbox**, คำสั่ง verify, ไฟล์ที่แตะ, dependencies, scope (XS/S/M)

แล้วเดินตาม **Phase → Checkpoint**: จบแต่ละ Phase ต้องผ่าน checkpoint (test เขียว + ruff สะอาด + ตัวเลขเดิมยัง reproduce ได้) ก่อนไปต่อ

ตัวอย่าง Boundaries ที่ใช้จริง:
* **Always** — รัน `pytest -q` และ `ruff check` ก่อน commit ทุกครั้ง; commit เป็นก้อนเล็กต่อ task; ตัวเลขทุกตัวที่เปลี่ยนในเอกสารต้อง copy มาจากไฟล์ใน `reports/`
* **Ask first** — retrain โมเดล (30 นาที, ทับของเดิม), เปลี่ยน split hash / hyperparameter / ชุด category, เพิ่ม dependency
* **Never** — ลบโค้ด/ข้อมูลภาษาไทย, เขียน git history ใหม่, **เปลี่ยนลำดับ label ของโมเดลที่เทรนไปแล้ว** (จะทำให้ prediction เพี้ยนแบบเงียบ ๆ), commit ตัว weights หรือข้อมูลที่ generate ได้

### 12.3 วงจรการปรับปรุงที่ใช้จริง (ทำซ้ำหลายรอบ)

```
วัดผล → อ่าน error list ทีละตัว → ตั้งสมมติฐานว่าอะไรคือสาเหตุ
      → แยกว่าเป็นปัญหา "วิธีวัด" หรือ "โมเดล" → แก้ตัวที่ถูกที่สุดก่อน → วัดใหม่บน validation
      → ตกลง recipe แล้วค่อยแตะ test split
```

**ตัวอย่างที่ชัดที่สุด:** ตอนวินิจฉัยก่อนรอบ slot-level พบว่า
* FP ของ `JURISDICTION` 8 ตัว อย่างน้อย 6 ตัวเป็น governing-law clause ที่ **ถูกจริง** แต่ CUAD tag แค่ clause เดียวต่อสัญญา
* สัญญาเดียว (`df0521`) ให้ FP ของ `END_DATE` ถึง 9 ตัว เพราะมีหลาย supplement ที่แต่ละอันมีวันหมดอายุจริง
* FN ของ `AGREEMENT_DATE` เกือบทั้งหมดเป็นรูปแบบที่ train set แทบไม่มี (`7/7/98`, `24 OCT 2011`, `September ___, 2019`)
* 43 จาก 77 FP ของ `PARTY` เป็น mention ซ้ำของ gold

→ สรุปได้ว่า **ครึ่งหนึ่งของช่องว่างเป็นเรื่อง "วิธีวัด" ไม่ใช่โมเดล** → สร้าง slot-level metric ขึ้นมา → ได้ slot F1 = 0.804 ทันทีจาก **โมเดลตัวเดิม** → แล้วแก้ normalisation/selection ใน `postprocess.py` ดันเป็น 0.840 **โดยไม่ retrain เลย** → ค่อย retrain ทีหลัง (ได้ 0.855)

**บทเรียนที่ควรพูดบนเวที: gain ที่ใหญ่ที่สุดมาจากการวัดให้ถูกและชั้น post-processing ไม่ใช่จากการ retrain**

### 12.4 วินัยเรื่อง split

* **test split ถูกอ่านเป็นครั้งสุดท้ายเท่านั้น** — ทุกการตัดสินใจ (threshold 0.6, อัตรา augmentation, การเลือก recipe) ตัดสินบน **validation split** ล้วน
* ตัวเลขของ iteration เก่าเก็บไว้ที่ `reports/iteration1/`, `reports/iteration2/` ไม่ทับของใหม่
* ทุกแถวในตารางเปรียบเทียบถูก **re-score ใหม่ด้วยเกณฑ์ปัจจุบัน** (ยกเว้นแถวที่ติดดาว) เพื่อให้เทียบกันได้จริง

---

## 13. Workflow การใช้งาน — คำสั่งตั้งแต่ต้นจนจบ

### 13.1 ติดตั้ง

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # track อังกฤษ (ไม่ลง PyThaiNLP)
pip install -e ".[dev,thai]"       # + track ไทย
pip install -e ".[baseline]"       # + CRF baseline (sklearn-crfsuite)
```

### 13.2 End-to-end (track อังกฤษ)

```bash
contract-ner download              # Kaggle dataset (ไม่ต้องมี credential) → data/raw/kaggle
contract-ner build-cuad            # align CUAD → data/processed/cuad/{train,val,test}.jsonl

# อ่าน label distribution ก่อนเทรนเสมอ
contract-ner profile --config configs/english_cuad.yaml --out reports/profile --stem cuad_english

contract-ner train --config configs/english_cuad_label_aware_augmented.yaml   # ~35 นาทีบน M2 (MPS)

contract-ner evaluate --model models/english-cuad-roberta-label-aware-augmented \
    --data data/processed/cuad/val.jsonl data/processed/cuad/test.jsonl --out reports

contract-ner evaluate --model regex --data data/processed/cuad/test.jsonl --out reports        # baseline regex
contract-ner train-crf --config configs/english_cuad.yaml --out models/english-cuad-crf        # baseline CRF
contract-ner evaluate --model models/english-cuad-crf --data data/processed/cuad/test.jsonl --out reports

contract-ner extract --model models/english-cuad-roberta-label-aware-augmented \
    --file my_contract.txt --out my_contract.json

python -m pytest -q                # unit tests (test ไทย skip ถ้าไม่มี extra "thai")
```

`scripts/reproduce.sh` รันทั้งสอง track ตั้งแต่ต้นจนจบ (~1.5 ชม. บน Apple Silicon)

### 13.3 คำสั่ง CLI ทั้งหมด

| คำสั่ง | หน้าที่ |
|---|---|
| `download` | ดึง Kaggle legal-contract dataset (anonymous access) |
| `build-cuad` | align CUAD clause spans → BIO documents + แบ่ง split |
| `build-thai` | สร้างข้อมูลไทยสังเคราะห์ + โหลด gold set |
| `profile` | วิเคราะห์การกระจายตัวของ label (นับ, รูปทรง span, surface form, drift ระหว่าง split, ตรวจ window) |
| `train` | fine-tune token classifier จาก YAML config |
| `train-crf` | fit CRF baseline บน split เดียวกัน |
| `evaluate` | P/R/F1 ระดับเอกสาร ทั้ง 4 มุมมอง + recall รายรูปแบบ |
| `extract` | สกัด metadata จากไฟล์สัญญา (`--summary` = ย่อหนึ่งค่าต่อช่อง) |
| `export-conll` | dump JSONL เป็น CoNLL แบบ word/BIO เพื่อตรวจด้วยตา |
| `sample` | พิมพ์สัญญาไทยสังเคราะห์พร้อม annotation |

### 13.4 Python API

```python
from contract_ner.predict import Extractor
from contract_ner.postprocess import assemble

ex = Extractor("models/english-cuad-roberta-label-aware-augmented")
record = assemble(text, ex.predict_with_confidence(text), lang=ex.lang)
record["summary"]        # หนึ่งค่าต่อช่อง
record["entities"]       # ทุก span พร้อม offset, confidence และธง valid
```

### 13.5 สิ่งที่เก็บไว้ใน repo (และสิ่งที่ไม่เก็บ)

Model weights และข้อมูลที่ generate ได้ถูก **git-ignore** (สร้างใหม่ได้จากคำสั่งข้างบน) แต่ **ทุกอย่างที่วัดได้ถูก commit**:

| Path | เก็บอะไร |
|---|---|
| `reports/comparison.md` | ทุกโมเดล × split × มุมมอง ในที่เดียว พร้อมตาราง slot รายช่องและ recall รายรูปแบบ (สร้างใหม่ด้วย `python scripts/make_comparison_table.py`) |
| `reports/<model>__<split>.json` | report เต็ม: 4 มุมมอง, ทุก false positive และ miss พร้อมข้อความ, `forms`, `slot_errors` พร้อมค่าที่ควรจะเป็น |
| `reports/profile/` | label distribution ของทุก split (Markdown + JSON) |
| `reports/training/` | config, training loss และ validation score ต่อ epoch ของทุกการ fine-tune |
| `reports/iteration1/`, `iteration2/` | ผลของ iteration ก่อนหน้า (ไม่ถูกทับ) |

---

## 14. โครงสร้างไฟล์ในโปรเจกต์

```
configs/                 YAML training config (6 ตัว)
  english_cuad.yaml                          recipe แรก, จุดอ้างอิงที่ recipe อื่นวัดเทียบ
  english_cuad_label_aware.yaml              recipe ที่ label distribution เสนอ
  english_cuad_label_aware_augmented.yaml    ★ recipe ที่ ship
  english_cuad_merged.yaml                   experiment: พับ START_DATE
  english_cuad_deberta.yaml                  capacity check
  thai_wangchanberta.yaml                    track ไทย
data/
  raw/kaggle/            ไฟล์ดิบจาก Kaggle (git-ignored)
  processed/cuad/        train/val/test JSONL (git-ignored)
  processed/thai/        train/val/test_{synthetic,gold,gold_blind} (git-ignored)
  thai_gold/             24 สัญญาเขียนมือ mark-up [[text|LABEL]]  ← commit ไว้
  thai_gold_blind/       8 สัญญา blind                            ← commit ไว้
docs/
  midterm_presentation_cuad.md       สคริปต์นำเสนอ (CUAD-first)
  midterm_presentation.md            เวอร์ชันสองภาษา
  worklog_slot_eval_and_retrain.md   บันทึกงาน slot-level + retrain
  worklog_label_distribution.md      บันทึกงานวิเคราะห์ label distribution
  project_overview_th.md             ★ ไฟล์นี้
examples/                สัญญาตัวอย่างสำหรับ `contract-ner extract`
models/                  weights (git-ignored) — 6 โมเดล
reports/                 ผลการวัดทั้งหมด (commit ไว้)
scripts/
  reproduce.sh           รันทั้งสอง track ตั้งแต่ต้นจนจบ
  make_comparison_table.py  สร้าง reports/comparison.md
src/contract_ner/        (ดูตารางโมดูลในหัวข้อ 3)
tests/                   pytest 14 ไฟล์ / 82 test functions
SPEC.md                  spec ของรอบพัฒนาล่าสุด
tasks/plan.md            แผนการทำงาน + dependency graph + ความเสี่ยง
tasks/todo.md            task ย่อยพร้อม acceptance criteria
pyproject.toml           dependencies + extras + ruff + pytest config
```

---

## 15. คุณภาพโค้ดและการทดสอบ

* **82 test functions ใน 14 ไฟล์** — `tests/` สะท้อนโครงสร้าง `src/contract_ner/` ตามหัวข้อ
* ประเภทของ test:
  * **Unit** — `build_bio_labels` กับ subset, `Encoder` กับ `label2id` ที่กำหนดเอง, กฎ validation ของแต่ละภาษา, normalisation ของวันที่/เงิน/อัตรา
  * **Round-trip** — `train` เขียน `entity_types` ลง `pipeline_config.json` แล้ว `Extractor` อ่านกลับได้ถูก
  * **Regression** — `test_cuad.py` ล็อกตัวเลข alignment (4,308 spans, 98.1%) ไว้ไม่ให้ drift
  * **Integration** — `test_extractor_integration.py` รันโมเดลจริง ตรวจว่า `len(model.config.id2label) == 11` และ field ใน JSON demo ถูกต้อง (skip ถ้าไม่มีโมเดล)
  * **Isolation** — test ที่ block `pythainlp` ใน `sys.modules` เพื่อพิสูจน์ว่า core import ได้โดยไม่มี PyThaiNLP
  * **Skip ไม่ใช่ fail** — test ภาษาไทยใช้ `pytest.importorskip("pythainlp")`
* **ruff** (line-length 100, rules `E,F,W,I,UP,B,RUF`) รันก่อน commit ทุกครั้ง
* Convention: type hints ครบ, dataclass config, docstring สั้น ๆ ที่อธิบาย **เหตุผล** ไม่ใช่กลไก, compile regex ที่ระดับ module, ไม่สร้าง abstraction เกินกว่าที่สองภาษาต้องการ

---

## 16. ข้อจำกัดและงานถัดไป

### ทำได้ดี

* `Jurisdiction` — slot F1 **0.957** (clause governing-law เขียนซ้ำ ๆ กันแบบมี pattern ชัด)
* `Parties` — slot F1 **0.871** แม้ CUAD จะ tag แค่ mention ที่นิยาม
* วันที่แบบ `Month D, YYYY` — หาเจอ **18/18** บน test split
* ทั้ง pipeline รันบน MacBook M2 ได้ (เทรน ~35 นาที, ไม่ต้องใช้ GPU cloud)

### ล้มเหลว / อ่อนตรงไหน

| ปัญหา | รายละเอียด |
|---|---|
| **สไตล์ annotation ของ CUAD ไม่ใช่ NER guideline** | tag แค่ mention ที่นิยามคู่สัญญา; `Expiration Date`/`Governing Law` เป็น clause ทั้งย่อหน้า → strict score ต่ำกว่าความเป็นจริงที่ตารางทำได้ |
| **`Effective Date` แทบไม่ต่างจาก `Agreement Date`** | test มีแค่ 10 spans → โมเดลไม่เรียนรู้ที่จะแยก; `Start_Date` จึงถูก infer (มีธงกำกับ) และเป็นช่องที่อ่อนที่สุด |
| **อังกฤษครอบคลุม 5 จาก 8 ช่อง** | CUAD ไม่มี contract value / payment due / penalty rate ที่สะอาดพอ |
| **`End_Date` คือจุดที่ recipe ใหม่เสียคืน** | 0.84 → 0.78 บน test; recall รายรูปแบบไม่เปลี่ยน → ปัญหาอยู่ที่ **การเลือกระหว่าง term clause หลายอันที่ถูกทั้งคู่** ไม่ใช่การหาไม่เจอ → ก้าวถัดไปอยู่ใน `postprocess.py` ไม่ใช่การ fine-tune อีกรอบ |
| **วันที่แบบตัวเลข** (`7/7/98`, `24 OCT 2011`) | ยังเป็น miss หลักของ `Agreement_Date`; augmentation อย่างเดียวสอนให้ tagger ไปยิงวันที่ใน page footer แทน |
| **Gold gaps** | 13 ช่องที่ทายเกิน (โมเดล final): 6 เป็นค่าจริงที่ CUAD ไม่ได้ annotate, 1 ก้ำกึ่ง, 6 เป็น model error → ตรวจ 47 สัญญาด้วยมือจะยกเพดานขึ้น |
| **Plain text เข้าอย่างเดียว** | สัญญาจริงมาเป็น PDF/สแกน; OCR error จะทำให้คะแนนตก (span ถูก match แบบตรงตัว ไม่มี fuzzy matching) |
| **ข้อมูลไทยเป็นข้อมูลสังเคราะห์** | โมเดลไม่เคยเห็นสัญญาจริง; gold set เล็กและเขียนโดยคนเดียวกับ template |

### งานถัดไป (เรียงตามผลตอบแทน)

1. **Prompt-based LLM baseline บน split เดียวกัน** — เป็น baseline ตัวเดียวที่ยังขาด (มี regex + CRF แล้ว)
2. **แก้กฎการเลือกค่าของ `End_Date`** ใน `postprocess.py` (ไม่ต้อง retrain)
3. **Re-annotate ทุก mention ของคู่สัญญา** หรือวัดเฉพาะ preamble → ทำให้ strict precision มีความหมาย
4. **เพิ่ม CUAD category ที่เป็น span สั้น** (`Notice Period to Terminate Renewal`, `Renewal Term`) — ได้โดยไม่ต้อง annotate เพิ่ม
5. **สัญญาไทยจริงที่ annotate แล้ว** — โดยเฉพาะสำหรับ `PAYMENT_DUE` / `PENALTY_RATE`
6. **Layout-aware parsing ของ PDF/DOCX** และ **legal-domain encoder** (`legal-bert`)

---

## 17. แผนสไลด์ที่แนะนำ + ตัวเลขสำหรับแต่ละสไลด์

| # | สไลด์ | เนื้อหา / ตัวเลขที่ต้องมี | อ้างอิงในเอกสารนี้ |
|---|---|---|---|
| 1 | Title | ชื่อโปรเจกต์, งาน NER / Information Extraction, dataset CUAD | — |
| 2 | ปัญหา | ตาราง 5 ช่องข้อมูล + "สัญญายาวเฉลี่ย 52,000 ตัวอักษร" | §2.1 |
| 3 | ทำไมยากในเชิง NLP | 5 ข้อ: ambiguity, long doc, imbalance **1.03%**, หนึ่ง label หลายรูปทรง, ต้องตรวจสอบย้อนกลับได้ | §2.2 |
| 4 | Data | CUAD → 510 → **499** สัญญา, align **98.1%** ของ 4,308 spans, split **398/54/47** ระดับสัญญา | §4.1–4.3 |
| 5 | Label distribution (สไลด์ที่แยกโปรเจกต์นี้ออกจากงานทั่วไป) | ตาราง 5 label + 3 ข้อสรุปที่นำไปแก้ recipe | §4.4 |
| 6 | Pipeline | diagram 6 ขั้น | §3, §5 |
| 7 | เทคนิคเด็ด: per-word encoding | ปัญหา BPE merge `"),` → **strict F1 0.254 → 0.514** | §5 ขั้น 2 |
| 8 | Windowing | 256 subword / overlap 64, stitching เอา logits จาก window ที่ห่างขอบสุด, **0.45%** ของ span ตกคร่อมขอบ | §5 ขั้น 3, §4.4 |
| 9 | Training recipe | ตาราง hyperparameter + 3 เทคนิค augmentation พร้อมเหตุผล | §6 |
| 10 | Post-processing | validation → confidence ≥ 0.6 → normalisation → assemble | §5 ขั้น 5–6 |
| 11 | วิธีวัด 4 มุมมอง | ตาราง strict/relaxed/field/slot + เหตุผลว่าทำไม slot คือตัวจริง | §7 |
| 12 | ผลลัพธ์ | ตาราง iteration v1 → final + **slot F1 0.855** | §8.1 |
| 13 | ผลรายช่อง | ตาราง 5 ช่อง + เทียบ CRF / regex | §8.2 |
| 14 | ทำไม slot ≫ strict | 43/77 FP ของ PARTY เป็น mention ซ้ำ; Jurisdiction 0.76 → 0.96 บน prediction ชุดเดิม; **6/13 gold gaps เป็นค่าจริง** (ตรวจมือแล้ว) | §8.4 |
| 15 | Baselines | regex 0.762 / CRF 0.690 / encoder 0.855 → "ระยะห่างคือสิ่งที่ pre-training ให้" | §9 |
| 16 | บทเรียนจากการพัฒนา | "gain ใหญ่ที่สุดมาจากการวัดให้ถูก + post-processing ไม่ใช่การ retrain" (0.702 → 0.840 ด้วย weights เดิม) | §12.3 |
| 17 | Multi-language seam + ผลภาษาไทย | LangRules 3 อย่างที่ภาษาใหม่ต้องมี; blind gold strict **0.797** / slot **0.903** | §10, §11 |
| 18 | Limitations | 4–5 ข้อหลัก พูดตรง ๆ | §16 |
| 19 | Next steps | LLM baseline, End_Date selection, re-annotation, PDF layout | §16 |
| 20 | Demo | output JSON จริงพร้อม offset + confidence | §5 ตัวอย่าง output |

### ตัวเลขที่ต้องจำขึ้นใจ

```
499 สัญญา · 398/54/47 split · 98.1% alignment · 3,099 training spans
11 BIO labels (อังกฤษ) · 17 BIO labels (ไทย)
window 256 / stride 64 · confidence floor 0.6 · 1.03% ของตัวอักษรอยู่ใน span
slot F1 = 0.855 (test) / 0.874 (val)   ← ตัวเลขหลัก
strict 0.545 · relaxed 0.720 · field 0.640
baseline: regex 0.762 · CRF 0.690
ไทย blind: strict 0.797 · slot 0.903
เทรน ~35 นาที บน MacBook M2 (MPS)
```

---

## 18. คำถามที่น่าจะโดนถาม (Q&A)

**Q: ทำไม strict F1 ถึงแค่ 0.545 — ดูต่ำไปไหม?**
A: เพราะสไตล์การ annotate ของ CUAD ไม่ใช่ NER guideline — tag แค่ mention ที่นิยามคู่สัญญา ดังนั้น mention อื่นที่โมเดลหาเจอ (ซึ่งถูกต้อง) ถูกนับเป็น false positive: **43 จาก 77** FP ของ `PARTY` เป็นการซ้ำของ gold เอง และ `Expiration Date` / `Governing Law` เป็น clause ทั้งย่อหน้าที่ขอบเขตเป๊ะ ๆ ทำซ้ำได้ยาก ตัวเลขที่ตอบว่า "ระบบใช้ได้ไหม" คือ **slot F1 = 0.855**

**Q: ทำไมไม่ใช้ `is_split_into_words=True`?**
A: มันจะเติม prefix marker ให้ **ทุกคำ** ซึ่งไม่เหมือนข้อมูลตอน pre-training (ภาษาไทยจะได้ sequence ยาวขึ้น ~2 เท่า) วิธีของเราให้ pieces เหมือนการ tokenize ข้อความดิบทุกประการ ต่างแค่ไม่มี piece ไหนคร่อมขอบคำ

**Q: ทำไมเลือก threshold 0.6?**
A: เลือกจาก **validation split เท่านั้น** — เป็นจุดที่ strict และ relaxed ขึ้นทั้งสอง split และ slot-level เป็นกลาง; tagger ปล่อย fragment หนึ่ง token (`,`, `18/`) ที่ ≈0.5 ซึ่งไม่มีกฎ surface ไหนจับได้

**Q: DeBERTa ได้ span score ดีกว่า ทำไมไม่ใช้?**
A: มันดีกว่าบน span และบนตารางของ validation แต่ **แย่กว่าบนตารางของ test** (0.807 vs 0.855) — encoder สองตัวที่ห่างกัน 0.02 ในทิศทางตรงข้ามบน split ขนาด 47 กับ 54 เอกสาร แยกกันไม่ออกทางสถิติ จึงเลือกตัวที่เล็กกว่าและเร็วกว่า 1.5 เท่า และเก็บผล DeBERTa ไว้เป็น capacity reference

**Q: การ retrain ช่วยแค่ไหน เทียบกับการแก้ post-processing?**
A: จาก field F1 0.702 → slot F1 0.840 ทำได้ด้วย **weights ตัวเดิม** (แค่วัดให้ถูก + normalisation + selection + confidence floor); การ retrain ตามหลัง label distribution ดันจาก 0.840 → 0.855 — คือ **ประมาณ 10% ของ gain มาจาก retrain**

**Q: ข้อมูลไทยเป็นของสังเคราะห์ เชื่อถือได้ไหม?**
A: ไม่เชื่อ 100% — นี่คือเหตุผลที่มี **blind gold set** ที่เขียนหลัง iteration สุดท้ายและไม่เคยเปิดดู ช่องว่าง synthetic 0.995 → blind 0.797 คือระยะห่างระหว่าง generator กับภาษาที่คนเขียนจริง และเราระบุชัดว่าแม้ blind score ก็ยัง optimistic

**Q: ทำไมต้องมี CRF baseline?**
A: เพื่อแยกว่าคะแนนมาจากไหน — CRF เทรนด้วยข้อมูลชุดเดียวกัน decode เป็น span แบบเดียวกัน ผ่าน validation/normalisation ชุดเดียวกัน ต่างแค่ไม่มี pre-trained representation → **ระยะห่าง 0.690 vs 0.855 คือส่วนที่ pre-training ให้** และรูปแบบก็สอดคล้องกับทฤษฎี: CRF สู้ได้เมื่อการตัดสินใจเป็น local (Jurisdiction 0.96) แต่พังเมื่อต้องใช้ทั้งเอกสาร (Parties 0.61)

**Q: ระบบนี้ใช้แทนคนได้เลยไหม?**
A: ไม่ — ออกแบบให้เป็น **draft extraction ที่คน review** ทุกค่ามี character offset + confidence + ธง `valid` เพื่อให้ตรวจกลับไปที่ต้นฉบับได้ เกณฑ์ ≥ 0.80 ที่ตั้งไว้คือ "ใช้งานได้โดยมีคน review" ไม่ใช่ "ใช้แทนคน"
