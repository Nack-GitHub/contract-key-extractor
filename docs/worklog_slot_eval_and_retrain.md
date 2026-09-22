# บันทึกงาน — slot-level evaluation, normalisation และการ retrain (21–22 ก.ย. 2026)

**โจทย์:** ตาราง metadata ที่ผู้ใช้เห็นต้องถูกอย่างน้อย 80 % ของช่อง (field/slot F1 ≥ 0.80 = "ใช้แทนคนได้โดยมี review")
ก่อนเริ่มงานรอบนี้ตัวเลขที่ใกล้เคียงที่สุดคือ field F1 = 0.702 บน test split ของ CUAD (47 สัญญา)
ขอบเขตที่ตกลงกัน: Tier 1–3 ของข้อเสนอ (วัดให้ถูก → แก้ model ราคาถูก → เพิ่ม capacity) **ไม่ใช้ LLM**

**ผลสรุปสั้น ๆ**

| | strict F1 | relaxed F1 | field F1 | **slot F1** |
|---|---:|---:|---:|---:|
| ก่อนเริ่ม (v6: validation ใหม่) | 0.633 | 0.754 | 0.702 | 0.804 |
| **หลังงานรอบนี้ — weights เดิม + post-processing/normalisation ใหม่ + confidence ≥ 0.6** | **0.652** | **0.759** | **0.722** | **0.840** |
| retrain แบบรวม START_DATE + date augmentation (`english_cuad_merged.yaml`) | 0.630 | 0.806 | 0.691 | 0.828 |
| DeBERTa-v3-base ภายใต้ recipe เดียวกับ merged (`english_cuad_deberta.yaml`) | 0.663 | 0.762 | 0.743 | 0.807 (val 0.869) |
| regex-only baseline | 0.497 | 0.705 | 0.545 | 0.762 |

ถึงเป้า 0.80 ที่ slot level ด้วย **model ตัวเดิม** — gain ทั้งหมดมาจากชั้นวัดผล/normalisation/selection ไม่ใช่จาก retrain
ยังไม่ commit อะไรทั้งสิ้น (ทุกอย่างอยู่ใน working tree)

---

## 1. วินิจฉัยก่อนลงมือ — gap ครึ่งหนึ่งเป็น "วิธีวัด" ไม่ใช่ model

ไล่ error list ของ report เดิมทีละตัว:

| type | relaxed | field | สิ่งที่พบใน error list |
|---|---:|---:|---|
| JURISDICTION | 0.886 | 0.759 | FP 8 ตัว อย่างน้อย 6 ตัวเป็น governing-law clause ที่ถูกจริง (`The laws of the State of New York govern this Agreement.`) แต่ CUAD tag แค่ clause เดียวต่อสัญญา |
| END_DATE | 0.667 | 0.634 | สัญญาเดียว (`df0521`) ให้ FP 9 ตัว เพราะมีหลาย supplement ที่แต่ละอันมี expiry จริง |
| AGREEMENT_DATE | 0.729 | 0.659 | FN เกือบทั้งหมดเป็น format ที่ train set แทบไม่มี: `7/7/98`, `3-31-16`, `24 OCT 2011`, `September ___, 2019` |
| PARTY | 0.791 | 0.748 | 43/77 FP เป็น mention ซ้ำของ gold |

และ field metric เดิม (`evaluate.py`) เทียบ *ข้อความดิบของ span* เป็น set — ไม่ได้ผ่าน `normalize_date` / `normalize_jurisdiction`
และไม่ได้เทียบ "ช่องในตาราง" หลัง `assemble()` เลือกค่าเดียว จึงลงโทษคำตอบที่ถูกแต่ต่างประโยค

## 2. Tier 1 — วัดตารางจริง: `contract_ner.slots`

ไฟล์ใหม่ [`src/contract_ner/slots.py`](../src/contract_ner/slots.py) ต่อเข้า `evaluate_documents()` (ยังคง strict/relaxed/field ไว้ครบ; view ใหม่ชื่อ **slot**)

* **Gold ของ slot** สร้างจาก gold spans ด้วย normaliser ตัวเดียวกับ pipeline — clause ที่ normalise แล้วได้ค่าเดียวกันถือว่าถูก ไม่ใช่ต้องเป็น string เดียวกัน
* **Pred ของ slot** คือ record ที่ `assemble()` สร้างจริง (หนึ่งค่าต่อช่อง)
* นิยามช่อง:
  * `Parties` — set ของชื่อนิติบุคคล (ตัด alias ในเครื่องหมายคำพูดเช่น `"Company"` และ boilerplate *referred to as the Parties* ทั้งสองฝั่ง); ชื่อตรงกันเมื่อ normalised form เท่ากันหรือฝั่งหนึ่งเป็น substring ของอีกฝั่ง (รับ boundary เพี้ยน 1 token)
  * `Agreement_Date`, `Start_Date` — ISO date; ถ้า gold `START_DATE` เป็น clause นิยามไม่มีตัวเลข slot จะ fallback ไป agreement date (สะท้อนกฎ `inferred_from` ของ `assemble`)
  * `End_Date` — ISO date ถ้ามีวันที่ ไม่งั้นเป็น duration key เช่น `3 year` (`three (3) years from the Effective Date` = `a period of 3 years`)
  * `Jurisdiction` — สถานที่จาก governing-law clause (`normalize_jurisdiction(...)["english"]`); pred ใช้ช่อง `Governing_Law` ก่อน `Jurisdiction` เพราะ category ของ CUAD คือ *Governing Law*
* การนับ: ช่องที่ gold ว่างและ pred ว่าง = true negative ไม่นับ; ค่าผิด = FP + FN; ค่าถูกเมื่ออยู่ใน set ของ gold (CUAD บางสัญญา tag วันที่สองค่า)
* error list ของ slot แยกเก็บที่ `slot_errors` ใน report JSON

ผลบน model เดิมทันทีที่วัดแบบนี้: slot F1 = **0.804** (Jurisdiction 0.899, Parties 0.858, Agreement_Date 0.795, End_Date 0.841, Start_Date 0.486)

### 2.1 Gold gaps ที่พบ (ไม่ได้แก้ data)

จาก 9 ช่องที่ pred มีค่าแต่ gold ว่าง อ่านต้นฉบับแล้ว 5 ช่องเป็นค่าจริงที่ CUAD ไม่ได้ annotate:
`7e39c8` governing law = Nevada, `023dce` governing law = PRC, `b0967a` "continue in effect for 2 years following the Launch Date",
`254532` "continue in force until terminated…", `2a803f` "This Remarketing Agreement shall terminate (i)…"
→ ตัวเลขบน CUAD gold เป็น floor; ไม่ได้แก้ไฟล์ gold เพื่อให้เทียบกับงานอื่นบน CUAD ได้

## 3. Tier 1½ — normalisation / selection ใน `postprocess.py` (product code, ไม่ต้อง retrain)

| การแก้ | เหตุผลจาก error list | ผลต่อ slot F1 |
|---|---|---:|
| `normalize_date(text, lang)`: ปี 2 หลัก (`7/7/98`), ลำดับ M/D/Y สำหรับอังกฤษ (D/M/Y สำหรับไทย หรือเมื่อปีเป็น พ.ศ.), เลขลำดับเป็นคำ (`the first day of July, 2017`, `twenty-first`), กันเลข section (`5.1.2`) | gold ที่ normalise ไม่ได้ทำให้เทียบไม่ตรง | 0.804 → 0.813 |
| `normalize_jurisdiction`: clause ตัวพิมพ์ใหญ่ทั้งหมด, apostrophe (`People's Republic of China`), `English law`, เลิกใช้ `str.isascii()` | clause ของ CUAD หลายอันเป็น ALL CAPS | (รวมข้างล่าง) |
| `Governing_Law` เลือก clause ที่มี *governed / construed / interpreted* ก่อน clause ที่แค่ "comply with the laws of the United States" | `df0521` เลือกผิด clause | 0.813 → 0.821 (Jurisdiction 0.957) |
| `split_effective_date()`: ถ้า candidate ของ AGREEMENT_DATE มีวันที่ต่างกัน ≥ 2 ค่า ตัวที่ 40 ตัวอักษรก่อนหน้ามี *effective / commence* เป็น `Start_Date`, ตัวแรกที่เหลือเป็น `Agreement_Date` | model tag ทั้ง signing date และ effective date เป็น AGREEMENT_DATE | 0.821 → 0.831 (Start_Date 0.533 → 0.613) |
| `Agreement_Date` / `Start_Date` prefer candidate ที่ parse เป็นวันที่ได้ก่อน clause นิยาม | v2 เลือก `"Effective Date" shall mean…` แทนวันที่ | — |
| `Extractor(min_confidence=0.6)` + `--min-confidence` ใน CLI: ตัด span ที่ mean label probability < 0.6 | tagger ปล่อย fragment สั้น ๆ (`,`, `04`, `18/`) ที่ conf ≈ 0.5 ซึ่ง rule จับไม่หมด; เลือก 0.6 บน **val split** (strict/relaxed ขึ้นทั้ง val และ test, slot เป็นกลางถึงบวก; 0.7 ดีกว่าบน val แต่แย่กว่าบน test — split เล็กเกินกว่าจะ tune ละเอียด) | 0.831 → **0.840** |

ทุก rule ถูกเช็คกับ gold ทั้ง 3 split ก่อนว่าไม่ชนกับ annotation style ของ CUAD

## 4. Tier 2 — retrain: รวม START_DATE เข้า AGREEMENT_DATE + date-format augmentation

โค้ดใหม่ [`src/contract_ner/augment.py`](../src/contract_ner/augment.py) ผ่าน config `label_map` และ `date_augment` (บันทึกลง `pipeline_config.json`; `evaluate` fold gold ด้วย map เดียวกัน)

* `map_labels`: `START_DATE → AGREEMENT_DATE` และ **ทิ้ง** span ของ short-date type ที่ `normalize_date` อ่านไม่ได้ (clause นิยาม 37/102 และ placeholder `September ___, 2019`) — สิ่งที่ validation ไม่มีวันปล่อยผ่านไม่ควรเป็น positive ตอน train
* `rewrite_dates`: เขียนวันที่ใน gold ใหม่เป็น format สุ่ม (`7/1/17`, `1 JUN 2020`, `Jul. 1, 2020`, `the 22nd day of August, 2005`, …) แล้วเลื่อน offset ที่ตามมา (มี test ว่า `text[start:end] == span.text` ทุก span)

| run | recipe | val slot F1 (th 0.6) | test slot F1 | test relaxed F1 | หมายเหตุ |
|---|---|---:|---:|---:|---|
| v1 (shipped) | 5 types, ไม่ augment | 0.846 | **0.840** | 0.759 | |
| v2 | merge (ไม่ทิ้ง dateless), augment 0.4, neg-window 0.15 | 0.841 | 0.828 | 0.716 | tagger ยิงวันที่ใน page footer (`04/18/18`) ทั่วเอกสาร: AGREEMENT_DATE raw spans 199 vs 109; เลือก clause นิยามเป็น Agreement_Date |
| v3 = `english_cuad_merged.yaml` | merge + ทิ้ง dateless, augment 0.25, neg-window 0.3 | 0.837 | 0.828 | **0.806** | span recall ดีขึ้นชัด แต่ตารางเท่าเดิม |

ข้อสรุป: การรวม label ทำให้ span-level ดีขึ้น (relaxed +0.05) แต่ slot F1 ต่างกัน 0.012 = 2 ช่องจาก ~200 ช่อง ซึ่งอยู่ในระดับ noise ของ split 47 เอกสาร
(val บอกทางเดียวกัน) FN เดิมอย่าง `7/7/98`, `9/9/97`, `24 OCT 2011` ยังไม่ได้คืนมา — augmentation สอนให้รู้จัก format แต่ไม่ได้สอน "ตำแหน่งใน preamble"
เพราะ negative windows ส่วนใหญ่ถูกทิ้ง จึง**ยังคง v1 เป็น model ที่ ship** (recipe ง่ายกว่า ผลเท่ากัน) และเก็บ v3 ไว้เป็น `models/english-cuad-roberta-merged` + config เผื่อ seed อื่น/ข้อมูลเพิ่ม

## 5. Tier 3 — DeBERTa-v3-base (capacity check)

`configs/english_cuad_deberta.yaml` = recipe ของ v3 ทุกอย่าง เปลี่ยนแค่ encoder (`microsoft/deberta-v3-base`, lr 2e-5, batch 8 × grad_accum 2)
encoder ต่อเข้า per-word tokenisation ได้ทันที (Metaspace backend เหมือน WangchanBERTa)

**บั๊กที่เจอระหว่างทาง:** run แรกและ trial ที่ lr 1e-5 diverge เป็น NaN ภายใน ~100 step (loss 12 → 264 → nan) ทั้งที่ forward pass บน MPS ตรงกับ CPU
สาเหตุ: checkpoint `deberta-v3-base` เก็บเป็น **fp16** และ transformers 5 โหลดตาม dtype ของ checkpoint → fine-tune fp16 โดยไม่มี loss scaling
(`roberta-base` เป็น fp32 จึงไม่เคยเจอ) แก้ที่ `train.py` / `predict.py`: `from_pretrained(..., dtype=torch.float32)` — sanity check 12 step บน MPS: loss 1.82 → 0.44

**ผล** (2 ชม. 15 นาที บน M2, window-level val F1 0.618 vs 0.563 ของ RoBERTa merged):

| | strict | relaxed | field | slot | slot per cell (Parties / Agr / Start / End / Juris) |
|---|---:|---:|---:|---:|---|
| test (47) | **0.663** | 0.762 | **0.743** | 0.807 | 0.82 / 0.69 / 0.69 / 0.84 / 0.96 |
| val (54) | **0.729** | 0.830 | **0.805** | **0.869** | 0.85 / 0.82 / 0.81 / 0.91 / 0.98 |
| RoBERTa shipped — test | 0.652 | 0.759 | 0.722 | **0.840** | 0.84 / 0.83 / 0.75 / 0.84 / 0.96 |
| RoBERTa shipped — val | 0.692 | 0.804 | 0.758 | 0.846 | 0.83 / 0.83 / 0.76 / 0.88 / 0.95 |

อ่านผล: DeBERTa ดีกว่าที่ span level ทั้งสอง split และดีกว่าที่ตารางบน val (+0.023) แต่แย่กว่าที่ตารางบน test (−0.033)
เพราะพลาด Agreement_Date 9 สัญญาที่ RoBERTa หาเจอ (ส่วนใหญ่เป็นวันที่ format ตัวเลข + สัญญาที่ preamble ยาว) — ต่างกัน 0.02–0.03 คนละทิศบน split ขนาด 47/54 เอกสาร
คือแยกไม่ออก **จึงยังคง RoBERTa เป็น model ที่ ship** (เล็กกว่า 3×, inference เร็วกว่า ~1.5×) และเก็บ DeBERTa ไว้เป็น capacity reference
(`models/english-cuad-deberta`, 701 MB, `reports/english-cuad-deberta__test.json`)
ถ้าจะตัดสินให้ขาดต้องใช้ 3 seed หรือ cross-validation ทั้ง 499 สัญญา (≈ 10 ชม. บน M2)

## 6. ผลข้างเคียงกับ track ภาษาไทย

`min_confidence=0.6` และ slot view ใช้กับทุก model จึง regenerate report ไทยใหม่ (weights เดิม):
dev gold strict 0.897 → 0.903 / slot 0.964, blind gold 0.792 → 0.797 / slot 0.903, synthetic 0.995 / slot 1.000 — README อัปเดตแล้ว

## 7. ไฟล์ที่เปลี่ยน

* ใหม่: `src/contract_ner/slots.py`, `src/contract_ner/augment.py`, `configs/english_cuad_merged.yaml`, `configs/english_cuad_deberta.yaml`,
  `tests/test_slots.py`, `reports/english-cuad-roberta-merged__test.json`, `reports/english-cuad-deberta__test.json`, เอกสารนี้
* แก้: `evaluate.py` (slot view, `label_map`, รับ extractor ที่มีแค่ `predict`), `postprocess.py` (normalisers, `split_effective_date`, governing-law pick),
  `predict.py` (`min_confidence`, `label_map`, โหลด fp32), `train.py` (`label_map`, `date_augment`, โหลด fp32), `cli.py` (`--min-confidence`),
  `configs/english_cuad.yaml` (คอมเมนต์อธิบาย recipe), `README.md`, `scripts/reproduce.sh`, `tests/test_postprocess.py`, `tests/test_schema.py`,
  reports ทั้งหมดใน `reports/*.json` (regenerate ด้วย threshold + slot view)
* model dirs (git-ignored): `models/english-cuad-roberta` (shipped, weights เดิม), `models/english-cuad-roberta-merged`, `models/english-cuad-deberta`
* tests: `rtk proxy python -m pytest -q` → ผ่านทั้งหมด; `ruff check src tests` สะอาด

## 8. สิ่งที่ยังไม่ได้ทำ / ข้อเสนอถัดไป

* **ยังไม่ commit** — รอสั่ง
* ทางไป 0.90: (ก) แก้ gold gaps 5 ช่อง + review ที่เหลือของ 47 สัญญา (ceiling ปัจจุบันกดอยู่), (ข) `Start_Date` เป็นช่องที่อ่อนสุด (0.75) — ต้องการ annotation ของ effective date ที่เป็นวันที่จริง ๆ มากกว่า 10 span,
  (ค) numeric date ใน preamble: ทำ negative augmentation (ใส่วันที่ format เดียวกันใน footer/body เป็น O) แทนที่จะเพิ่ม positive อย่างเดียว, (ง) Tier 4 (LLM verifier/baseline) ที่ตกลงว่ายังไม่ทำ
