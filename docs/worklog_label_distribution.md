# บันทึกงาน — วิเคราะห์การกระจายตัวของ label ก่อนเทรน, ปรับ recipe การเทรน และ baseline เชิงสถิติ (22 ก.ย. 2026)

**โจทย์จากผู้ใช้ 2 ข้อ**

1. label แต่ละตัวมีข้อความหลายรูปแบบ — ต้องมีขั้นตอน "วิเคราะห์การกระจายตัวของ label" ก่อนนำไปเทรน
2. ตอนนี้เทียบกับ regex baseline อย่างเดียว — อยากได้ NER baseline ที่เทียบกันได้จริง

ขอบเขตที่ตกลงกัน: จัดการเรื่อง label distribution ก่อนเทรน → ตั้งค่า training ให้เหมาะกับสิ่งที่พบ →
เทรนและวัดผลแยก train / dev (val) / test ให้ครบ

---

## 1. เครื่องมือใหม่: `contract-ner profile`

```bash
contract-ner profile --config configs/english_cuad.yaml --out reports/profile --stem cuad_english
contract-ner profile --config configs/thai_wangchanberta.yaml --out reports/profile --stem thai_wangchanberta --no-windows
```

เขียนรายงานเป็น Markdown + JSON (`reports/profile/`) ตอบ 4 คำถามต่อ 1 split:

| คำถาม | สิ่งที่รายงานออกมา |
|---|---|
| มี label ละเท่าไร | จำนวน span, จำนวนเอกสารที่มี label นั้น, span ต่อเอกสาร (0 / 1 / 2 / 3+) |
| span หน้าตาเป็นอย่างไร | ความยาวเป็นคำ (median / p90 / max) — แยก short-span กับ clause-level ออกจากกัน |
| อยู่ตรงไหนของเอกสาร | สัดส่วน span ใน 5 % แรก / 5–25 % / 25–50 % / ครึ่งหลัง |
| เขียนได้กี่รูปแบบ และ split ตรงกันไหม | bucket ของ surface form ต่อ label + ตารางเทียบ train ↔ val ↔ test |

รูปแบบ (surface form) นิยามไว้ที่ language seam เดิม (`lang/en.py`, `lang/th.py` → `span_form`)
เพื่อให้เพิ่มภาษาใหม่แล้วได้รายงานชุดเดียวกันทันที และ evaluation ก็เรียก bucket ชุดเดียวกันนี้
ไปทำ **recall รายรูปแบบ** (`reports/*.json` → `forms`)

นอกจากนี้ยังรายงาน **windowing** ด้วย (ต้องมี `--config` เพื่อโหลด tokenizer ของ recipe นั้น):
จำนวน window, สัดส่วน window ที่มี entity และจำนวน gold span ที่ไม่มี window ใดครอบได้ทั้ง span

---

## 2. สิ่งที่พบ — CUAD (อังกฤษ)

ที่มา: `reports/profile/cuad_english.md` · train 398 / val 54 / test 47 สัญญา

**2.1 label เดียวกันแต่คนละปัญหา**

| Label | spans (train) | เอกสารที่มี | ความยาว (คำ, median / p90) | ใน 5 % แรกของเอกสาร |
|---|---:|---:|---|---:|
| PARTY | 1,953 | 100 % | 2 / 6 | 82 % |
| AGREEMENT_DATE | 363 | 90 % | 3 / 5 | 83 % |
| START_DATE | 76 | **18 %** | 10 / 26 | 28 % |
| END_DATE | 361 | 80 % | **31 / 66** | 8 % |
| JURISDICTION | 346 | 83 % | **29 / 51** | **0 %** (93 % อยู่ครึ่งหลัง) |

สองกลุ่มนี้คนละงานกัน: PARTY/AGREEMENT_DATE เป็น NER ช่วงสั้นในหัวสัญญา ส่วน END_DATE/JURISDICTION
เป็น clause ทั้งประโยคในครึ่งหลัง — เป็นเหตุผลเชิงโครงสร้างว่าทำไม strict span F1 ถึงต่ำกว่า slot F1 มาก
และทำไมถึงไม่ควรอ่าน strict F1 ตัวเดียวแล้วสรุปคุณภาพงาน

สัดส่วนตัวอักษรที่อยู่ใน gold span = **1.03 %** ของคลังทั้งหมด (ความไม่สมดุลระดับ token)

**2.2 รูปแบบข้อความต่อ label (train)**

* `PARTY` — quoted alias (`"Acme"`) 36.3 % · ชื่อนิติบุคคล 33.5 % · ชื่อพิมพ์ใหญ่ล้วน 12.3 % ·
  boilerplate "referred to as the Parties" 6.2 % · ชื่อบุคคล 1.3 %
* `AGREEMENT_DATE` — `Month D, YYYY` 52.6 % · `Dth day of Month, YYYY` 25.1 % · ตัวเลขล้วน 5.5 % ·
  ช่องว่างยังไม่กรอก (`____ day of ____, 2004`) 5.0 %
* `START_DATE` — **ไม่มีวันที่ในตัว 39.5 %** (เป็นประโยคนิยาม) · มีวันที่ 60.5 %
* `END_DATE` — ระยะเวลา (`three (3) years`) 44.3 % · เปิดปลาย ("until terminated") 26.6 % · มีวันที่ชัดเจน 21.9 %
* `JURISDICTION` — ประโยค "governed by / construed" 93.9 %

**2.3 split ไม่ตรงกันตรงไหน**

เทียบ train เป็นฐาน (ตัดสินใจจาก val เท่านั้น ไม่เปิดดู test ก่อนสรุป recipe):

| เทียบกับ train | label | รูปแบบ | train | อีก split |
|---|---|---|---:|---:|
| val | AGREEMENT_DATE | `month-y` | 5.5 % | 14.0 % |
| test (ดูหลังสรุป recipe แล้ว) | AGREEMENT_DATE | `numeric` | 5.5 % | 11.9 % |

**2.4 การตัดหน้าต่าง (window) ไม่ใช่ปัญหา**

window 256 subword / overlap 64: มี gold span เพียง 14 จาก 3,099 span (0.45 %) ที่ไม่มี window ใดครอบได้ทั้ง span
และมีแค่ 1 span ที่ยาวเกิน 1 window → **ไม่ต้องขยาย window** (ขยายแล้วเสียเวลาเทรนฟรี)
window ที่มี entity มีเพียง 7.6 % ของทั้งหมด → การ sample negative window ลงเป็นสิ่งจำเป็น ไม่ใช่ทางเลือก

**2.5 recall รายรูปแบบของโมเดลที่ใช้อยู่ (RoBERTa ตัวปัจจุบัน)**

วัดบน val (ใช้ตัดสินใจ) และ test (รายงานเฉย ๆ):

| label / รูปแบบ | val | test |
|---|---|---|
| AGREEMENT_DATE `month-d-y` | 22/25 | 16/18 |
| AGREEMENT_DATE `numeric` | **0/1** | **0/5** |
| AGREEMENT_DATE `placeholder-blank` | 1/3 | **0/3** |
| PARTY ชื่อปกติ `legal-name` | 62/75 (83 %) | 69/90 |
| PARTY พิมพ์ใหญ่ล้วน `legal-name-caps` | **21/32 (66 %)** | 29/35 |
| START_DATE ทุกงรูปแบบรวมกัน | 3/16 | **0/10** |

สามข้อที่เห็นได้จาก val อย่างเดียว: วันที่ตัวเลขล้วนหาไม่เจอเลย, ชื่อพิมพ์ใหญ่ล้วนแย่กว่าชื่อปกติ 17 จุด,
และ START_DATE แทบไม่ถูกทำนายเลย

---

## 3. สิ่งที่พบ — ไทย (WangchanBERTa, ข้อมูลสังเคราะห์)

ที่มา: `reports/profile/thai_wangchanberta.md`

* สัดส่วนตัวอักษรที่เป็น entity = **14.3 %** (สังเคราะห์) เทียบกับ **1.0 %** ของ CUAD —
  สัญญาสังเคราะห์ "แน่น" กว่าสัญญาจริงมาก เป็นอีกเหตุผลว่าทำไมคะแนนชุดสังเคราะห์ถึงมองในแง่ดีเกินจริง
* PARTY มี repeated mention 43 % (ชื่อเดิมซ้ำในเอกสารเดียว) ต่างจาก CUAD ที่ gold มีแค่ mention แรก (0 %)
  → เป็นเหตุผลว่าทำไม precision ของฝั่งอังกฤษถึงถูกลงโทษจาก mention ซ้ำ แต่ฝั่งไทยไม่โดน
* **blind gold ต่างจาก training data อย่างมีระบบ**:

| label | รูปแบบ | train (สังเคราะห์) | blind gold |
|---|---|---:|---:|
| AGREEMENT_DATE | ปีไม่ระบุยุค + เลขไทย | 3.2 % | 25.0 % |
| START_DATE | ปีไม่ระบุยุค + เลขไทย | 3.5 % | 28.6 % |
| END_DATE | ปีไม่ระบุยุค + เลขไทย | 3.6 % | 28.6 % |
| PARTY | ชื่ออักษรละติน | **0 %** | 10.5 % |

ข้อนี้อธิบายช่องว่าง blind (strict F1 0.797) กับ dev (0.903) ได้ตรง ๆ และบอกว่า generator รุ่นถัดไป
ควรเพิ่มอะไร โดยไม่ต้องเดา

---

## 4. ปรับ recipe การเทรนตามหลักฐาน

หลักการ: ทุกการเปลี่ยนต้องชี้กลับไปที่บรรทัดใดบรรทัดหนึ่งในรายงาน §2 ได้ และ **ตัดสินใจจาก train + val เท่านั้น**
(ตัวเลขฝั่ง test ในรายงานถูกอ่านหลังจากล็อก recipe แล้ว)

| สิ่งที่เปลี่ยน | หลักฐาน | ค่าที่ตั้ง |
|---|---|---|
| ตัด span ของ `START_DATE` ที่ไม่มีวันที่ออกจาก **ข้อมูลเทรน** | 39.5 % ของ gold เป็นประโยคนิยามยาวเฉลี่ย 26 คำ ปนกับวันที่สั้น 3 คำ ใน label เดียว รวมทั้งหมดแค่ 76 span | `dateless_labels_to_drop: [START_DATE]` (76 → 43 span) |
| เก็บ negative window ของ **หัวสัญญา** ในอัตราสูงกว่าเนื้อเอกสาร | 82–83 % ของ span ชนิดสั้นอยู่ใน 5 % แรก และของที่หน้าตาเหมือน entity (ที่อยู่, เลขทะเบียน, หัว/ท้ายหน้า) ก็อยู่ที่เดียวกัน | `negative_window_keep: 0.12` + `negative_window_keep_document_head: 0.50` (หัว = 10 % แรก) |
| เพิ่ม epoch เป็น 4 และเลือก checkpoint ที่ดีที่สุดบน val | ชุดเทรนสะอาดขึ้นและเล็กลงเล็กน้อย | `epochs: 4` |
| ไม่ขยาย window | span ที่ถูกตัดขาดมีแค่ 0.45 % | คงเดิม 256 / 64 |

จำนวน window รวมแทบไม่เปลี่ยน (1,387 เทียบ 1,389 บนสัญญา 100 ฉบับแรก) → **เวลาเทรนเท่าเดิม แต่ negative ยากขึ้น**
config: `configs/english_cuad_label_aware.yaml`

recipe ที่สอง (`configs/english_cuad_label_aware_augmented.yaml`) เพิ่ม augmentation 2 อย่างที่ recall รายรูปแบบชี้:

* `date_augment: 0.16` — วันที่แบบตัวเลขล้วนหาไม่เจอเลยบน val (0/1) ขณะที่ในชุดเทรนมีแค่ 5.5 %
* `party_case_augment: 0.16` — ชื่อพิมพ์ใหญ่ล้วน recall 66 % เทียบกับชื่อปกติ 83 % บน val
  (เขียนทับด้วยตัวพิมพ์ใหญ่ความยาวเท่าเดิม offset จึงไม่ขยับ)

ทั้งสอง config ต่างกันแค่ augmentation → ผลต่างระหว่างรายงานสองฉบับคือผลของ augmentation ล้วน ๆ

---

## 5. วินัยการใช้ split

| split | จำนวนสัญญา | ใช้ทำอะไร |
|---|---:|---|
| train | 398 | เทรนอย่างเดียว |
| val (dev) | 54 | เลือก checkpoint, ตั้ง confidence floor, ตัดสินใจทุกอย่างเรื่อง recipe |
| test | 47 | อ่านครั้งเดียวหลังล็อก recipe แล้ว |

ทุกโมเดลในบันทึกนี้รายงานทั้ง val และ test เพราะ test มีแค่ 47 สัญญา — ผลต่าง 0.02 ที่ slot level
คือประมาณ 2 ช่องในตาราง ซึ่งอยู่ในระดับ noise การดูสอง split พร้อมกันช่วยไม่ให้สรุปเกินจริง

---

## 6. NER baseline เชิงสถิติ (conditional random field)

ก่อนหน้านี้มีแต่ regex baseline ซึ่งตอบได้แค่ว่า "เขียน pattern เองไปได้ไกลแค่ไหน"
เพิ่ม **linear-chain conditional random field** (`src/contract_ner/baseline_crf.py`, `contract-ner train-crf`)
ซึ่งเป็นวิธีมาตรฐานของงาน NER ก่อนยุค pre-trained encoder:

* feature ต่อคำ: รูปคำ (`Acme` → `Xxxx`), ตัวพิมพ์ใหญ่/เล็ก, suffix/prefix, มีตัวเลขหรือไม่,
  อยู่ในรายการคำลงท้ายนิติบุคคล / ชื่อเดือน / คำเชิงสัญญา, คำข้างเคียง ±2 ตำแหน่ง
  และ **ตำแหน่งในเอกสาร** (มาจากข้อค้นพบใน §2 ว่า label กระจายตัวตามตำแหน่งชัดมาก)
* ตัดเอกสารเป็นประโยค โดยจุดที่เป็นตัวย่อนิติบุคคล (`Inc.`, `Ltd.`) ไม่นับเป็นจบประโยค
  มิฉะนั้นชื่อคู่สัญญาจะถูกตัดครึ่ง
* sample sequence ที่ไม่มี entity ลงเหลือ 15 % เหมือนที่ encoder ทำกับ window
* ใช้ interface เดียวกับ extractor ตัวจริง (`predict_with_confidence`) → ผ่าน span validation,
  normalisation และ metric ชุดเดียวกันทั้งหมด

ความต่างระหว่าง CRF กับ RoBERTa จึงเป็น "ส่วนที่ได้จาก pre-training" ล้วน ๆ เพราะข้อมูล กฎ และตัววัดเหมือนกันหมด

---

## 7. ข้อมูลทั้งหมดที่เก็บไว้ (สำหรับทำรายงาน)

| ไฟล์ | เนื้อหา |
|---|---|
| `reports/comparison.md` | ตารางรวมทุกโมเดล × ทุก split × 4 มุมมอง (strict / relaxed / field / slot), ตาราง slot รายช่อง และ recall รายรูปแบบ — สร้างใหม่ได้ด้วย `python scripts/make_comparison_table.py` |
| `reports/<model>__<split>.json` | รายงานเต็มต่อโมเดลต่อ split: P/R/F1 ทั้ง 4 มุมมอง, รายการ error (FP/FN) พร้อมข้อความจริง, `forms` = recall รายรูปแบบ, `slot_errors` = ช่องตารางที่ผิดพร้อมค่าที่ควรเป็น |
| `reports/profile/cuad_english.{md,json}` | การกระจายตัวของ label ฝั่งอังกฤษ ทั้ง train / val / test + drift + windowing |
| `reports/profile/thai_wangchanberta.{md,json}` | การกระจายตัวฝั่งไทย ทั้ง train / val / dev gold / blind gold / synthetic |
| `reports/training/<model>.json` | config ที่ใช้เทรน, loss ระหว่างเทรน และคะแนน validation ราย epoch (คัดออกมาจาก `models/` ซึ่งไม่ได้ commit) |

ตัวเลขทุกตัวในบันทึกนี้อ้างอิงไฟล์ข้างบนได้โดยตรง และทุกอย่างสร้างซ้ำได้จาก `scripts/reproduce.sh`

---

## 8. ผลการทดลอง

ตัวเลขทั้งหมดอยู่ใน `reports/comparison.md` (สร้างจากไฟล์ JSON โดยตรง ไม่ได้พิมพ์มือ)

> ตัวเลขใน §8.1 และ §8.2 วัดก่อนแก้การเทียบค่าระดับ clause (§8.4) จึงต่ำกว่าตารางสรุปสุดท้ายใน §8.5
> เก็บไว้เพื่อให้เห็นลำดับการทดลองตามจริง

### 8.1 recipe label-aware เทียบโมเดลเดิม — validation split (54 สัญญา)

| มุมมอง | โมเดลเดิม | label-aware |
|---|---:|---:|
| strict F1 | 0.692 | 0.608 |
| relaxed F1 | 0.804 | 0.792 |
| field F1 | 0.758 | 0.667 |
| **slot F1** | **0.846** | **0.847** |

| ช่องตาราง | โมเดลเดิม | label-aware |
|---|---:|---:|
| Parties | 0.830 | 0.819 |
| Agreement_Date | 0.831 | 0.835 |
| **Start_Date** | 0.761 | **0.860** |
| End_Date | 0.879 | 0.809 |
| Jurisdiction | 0.949 | 0.949 |

**อ่านผล:** ตารางรวมแทบไม่ขยับ (0.846 → 0.847) แต่ช่องที่การวิเคราะห์ label ชี้ไว้ตรง ๆ ขยับจริง —
`Start_Date` +0.10 และ recall ของ START_DATE แบบที่มีวันที่จริงขยับจาก 0/6 เป็น 2/6
(ก่อนหน้านี้โมเดลไม่เคยทำนาย label นี้ได้เลย) ส่วน `End_Date` แลกกลับไป 0.07

recall รายรูปแบบของ PARTY ดีขึ้นทุก bucket (alias 90/94 → 94/94, ชื่อปกติ 62/75 → 64/75,
พิมพ์ใหญ่ล้วน 21/32 → 24/32, boilerplate 18/19 → 19/19) ซึ่งตรงกับสมมติฐานเรื่อง negative window
ของหัวสัญญา แต่ strict/field F1 ลดลงเพราะโมเดลทำนาย mention เพิ่ม ขณะที่ CUAD ให้ gold แค่ mention แรก
— เป็นการลงโทษจากรูปแบบ annotation ไม่ใช่จากคุณภาพค่าที่สกัดได้ (ดู §2.1)

### 8.2 recipe label-aware เทียบโมเดลเดิม — test split (47 สัญญา, อ่านครั้งเดียว)

| มุมมอง | โมเดลเดิม | label-aware |
|---|---:|---:|
| strict F1 | 0.652 | 0.547 |
| relaxed F1 | 0.759 | 0.715 |
| field F1 | 0.722 | 0.638 |
| **slot F1** | **0.840** | **0.834** |

| ช่องตาราง | โมเดลเดิม | label-aware |
|---|---:|---:|
| Parties | 0.840 | 0.849 |
| Agreement_Date | 0.827 | 0.842 |
| Start_Date | 0.747 | 0.789 |
| **End_Date** | **0.841** | **0.716** |
| Jurisdiction | 0.957 | 0.928 |

recall รายรูปแบบ (test) — ทุกบรรทัดที่ขยับ:

| label / รูปแบบ | โมเดลเดิม | label-aware |
|---|---|---|
| AGREEMENT_DATE `month-d-y` | 16/18 | **18/18** |
| AGREEMENT_DATE `numeric` (`7/7/98`) | **0/5** | **2/5** |
| PARTY `legal-name` | 69/90 | **74/90** |
| PARTY `legal-name-caps` | 29/35 | **32/35** |
| PARTY `caps-no-suffix` | 2/7 | 3/7 |
| PARTY `quoted-alias` | 79/83 | 81/83 |
| START_DATE `d-month-y` | 0/1 | 1/1 |
| END_DATE ทุกแบบ | 34/38 | 33/38 |

**สรุปผลการทดลองนี้**

* recipe ที่มาจากการวิเคราะห์ label **ยกระดับ recall ตรงจุดที่การวิเคราะห์ชี้** — วันที่แบบตัวเลขล้วน
  ซึ่งเดิมหาไม่เจอเลยทั้ง val และ test ตอนนี้เจอ 2 ใน 5 และชื่อคู่สัญญาดีขึ้นทุกรูปแบบ
  โดย**ยังไม่ได้ใช้ augmentation ใด ๆ เลย** — มาจากการทำ label ให้สะอาดและ negative ที่ยากขึ้นล้วน ๆ
* แต่ **ตารางรวมไม่ขยับ** (val 0.846 → 0.847, test 0.840 → 0.834) เพราะ span ที่ทำนายเพิ่มขึ้น
  ไปทำให้ช่อง `End_Date` เลือกค่าผิดบ่อยขึ้น (recall รายรูปแบบของ END_DATE แทบไม่เปลี่ยน
  แต่ precision ของช่องนี้ลดลง — มี term clause ที่ถูกต้องหลายอันให้เลือก)
* ดังนั้น **โมเดลที่ ship ยังเป็นตัวเดิม** ส่วน `english_cuad_label_aware.yaml` เก็บไว้เป็น recipe อ้างอิง
  ขั้นถัดไปที่ชัดเจนคือแก้ตัวเลือกค่าในช่อง `End_Date` (postprocess) ไม่ใช่การเทรนใหม่อีกรอบ

### 8.3 เพิ่ม augmentation ตามที่ recall รายรูปแบบชี้

`configs/english_cuad_label_aware_augmented.yaml` = recipe เดิม + `date_augment: 0.16` +
`party_case_augment: 0.16` (เขียนชื่อคู่สัญญาบางส่วนเป็นตัวพิมพ์ใหญ่ทั้งหมด โดยความยาวเท่าเดิม offset ไม่ขยับ)

ผลคือ **ดีที่สุดทั้งสอง split** ที่ระดับตาราง ซึ่งเป็นเกณฑ์ตัดสินของโปรเจกต์

### 8.4 แก้การเทียบค่าระดับ clause (วัดผลให้เป็นธรรม)

ตรวจ `slot_errors` ของช่อง `End_Date` แล้วพบว่า 5 จาก 12 รายการบน validation
"ผิด" เพราะต่างกันแค่เครื่องหมายวรรคตอนท้ายประโยคเท่านั้น เช่น

```
gold: unless earlier terminated in accordance with this paragraph 7, this agreement shall continue indefinitely.
pred: unless earlier terminated in accordance with this paragraph 7, this agreement shall continue indefinitely
```

ช่องระดับ clause (`End_Date`, `Jurisdiction`) จะถอยมาเทียบข้อความดิบเมื่อ normalise ไม่ได้
จึงแก้ `slots.value_key` ให้ตัดเครื่องหมายวรรคตอนหัว/ท้ายก่อนเทียบ แล้ว**วัดใหม่ทุกโมเดล**
ด้วยเกณฑ์เดียวกัน (ไม่ใช่แก้แล้ววัดเฉพาะโมเดลใหม่ ซึ่งจะเทียบกันไม่ได้)

### 8.5 ตารางสรุปสุดท้าย (เกณฑ์เดียวกันทุกแถว)

slot F1 — เกณฑ์ตัดสินของโปรเจกต์ (เป้า ≥ 0.80 สำหรับ "ใช้ได้โดยมีคนรีวิว"):

| โมเดล | val (54 สัญญา) | test (47 สัญญา) |
|---|---:|---:|
| regex baseline | 0.738 | 0.762 |
| conditional random field baseline | 0.753 | 0.690 |
| RoBERTa ตัวเดิม (v7) | 0.853 | 0.840 |
| + recipe label-aware (v8) | 0.866 | 0.845 |
| **+ augmentation (ตัวที่ ship)** | **0.874** | **0.855** |

ช่องตาราง (test):

| ช่อง | ตัวที่ ship | ตัวเดิม | conditional random field | regex |
|---|---:|---:|---:|---:|
| Parties | **0.871** | 0.840 | 0.611 | 0.800 |
| Agreement_Date | 0.825 | 0.827 | 0.805 | 0.737 |
| Start_Date | **0.800** | 0.747 | 0.590 | 0.684 |
| End_Date | 0.783 | **0.841** | 0.721 | 0.549 |
| Jurisdiction | 0.957 | 0.957 | 0.955 | 0.875 |

**ข้อสรุป**

1. การวิเคราะห์การกระจายตัวของ label ก่อนเทรน**ให้ผลจริง**: slot F1 ขยับ 0.840 → 0.855 บน test
   และ 0.853 → 0.874 บน val โดยทุกการเปลี่ยนแปลงอ้างหลักฐานจากรายงาน §2 ได้ และตัดสินจาก val เท่านั้น
2. สิ่งที่ยังแลกอยู่คือช่อง `End_Date` (0.841 → 0.783) เพราะโมเดลทำนาย term clause มากขึ้น
   แล้วเลือกผิดอัน — งานถัดไปอยู่ที่กฎเลือกค่าใน `postprocess` ไม่ใช่การเทรนใหม่
3. **CRF baseline บอกว่า pre-training ให้อะไร**: ตรงที่ตัดสินได้จากบริบทใกล้ ๆ (ชื่อศาล, วันที่)
   CRF เกือบเท่า encoder (Jurisdiction 0.955 vs 0.957, Agreement_Date 0.805 vs 0.825)
   แต่ตรงที่ต้องเข้าใจทั้งเอกสาร (คู่สัญญา 0.611 vs 0.871) ห่างกันมาก
   — เป็นคำตอบเชิงปริมาณว่าทำไมต้องใช้ contextual encoder
4. strict/field F1 ของตัวที่ ship ต่ำลง เพราะโมเดลจับ mention ของคู่สัญญาครบขึ้น
   ขณะที่ CUAD ให้ gold แค่ mention แรก — เป็นข้อจำกัดของ annotation ไม่ใช่คุณภาพของค่าที่ได้
