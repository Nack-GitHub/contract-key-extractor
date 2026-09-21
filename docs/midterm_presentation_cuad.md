# Midterm Presentation — Key Information Extraction from Legal Contracts with NER (CUAD)

**หัวข้อ:** NER / Information Extraction — สกัดข้อมูลสำคัญจากสัญญาเชิงพาณิชย์ด้วย Named Entity Recognition
**Dataset:** CUAD (Contract Understanding Atticus Dataset) ผ่าน Kaggle *Legal Contract Dataset*
**ประเภท:** Application + case study จากโปรเจกต์ที่ทำเอง (`contract-key-extractor`)
**โครงสร้าง:** Problem → Data → Method → Results → Limitations → Q&A

> ตัวเลขทั้งหมดในเอกสารนี้มาจากการทดลองของผู้นำเสนอเอง (`reports/english-cuad-roberta__test.json`; รอบ v4 ก่อน retrain อยู่ใน `reports/iteration2/`) ยกเว้นส่วนที่ระบุว่าเป็นของผู้สร้าง dataset (Hendrycks et al., 2021)

---

## 1. Problem and Motivation

### ปัญหาคืออะไร

สัญญาเชิงพาณิชย์ (distribution, license, service, supply, …) เป็น **unstructured text** ยาวหลายสิบหน้า แต่คำถามที่ฝ่ายกฎหมาย/บัญชี/จัดซื้อถามบ่อยที่สุดมีไม่กี่ข้อ:

| ช่องข้อมูล | คำถามที่ตอบ | ตัวอย่างจาก CUAD |
|---|---|---|
| **Parties** | ใครเป็นคู่สัญญา | Diversinet Corp. / 2205925 Ontario Limited |
| **Agreement Date** | ลงนามเมื่อไร | July 1, 2020 |
| **Effective Date** | เริ่มมีผลเมื่อไร | (มักตรงกับวันลงนาม) |
| **Expiration Date** | หมดอายุเมื่อไร / ต่ออายุอย่างไร | "...shall continue for a period of five (5) years" |
| **Governing Law** | ใช้กฎหมายรัฐ/ประเทศไหน | State of New York |

งานคือแปลง **ข้อความสัญญา → ตาราง metadata (JSON)** โดยอัตโนมัติ — ในนิยามของคลาสคือ **NLU** (unstructured → structured representation) และเป็นงาน **NER** โดยตรง: ระบุ span ในข้อความแล้วติด label ประเภท

### ทำไมน่าสนใจในเชิง NLP

ปัญหานี้รวม "ความยากของภาษา" (Section 1 ของคลาส) ไว้เกือบครบ:

1. **Lexical ambiguity — คำเดียวกัน ความหมายต่างกันตามบริบท**
   - วันที่ `July 1, 2020` เป็น *Agreement Date* หรือ *Effective Date* หรือวันที่ส่งมอบงวดแรก — ตัวเลขเหมือนกัน ต่างกันที่ประโยครอบข้าง
   - ชื่อองค์กร `Bank of America` เป็นคู่สัญญา หรือเป็นธนาคารที่ระบุไว้ในเงื่อนไขการชำระเงิน
   - `"Company"`, `"Licensor"` เป็น *defined term* (นามแฝงของคู่สัญญา) ไม่ใช่ชื่อนิติบุคคล
   → เหตุผลหลักที่ต้องใช้ **contextual embedding** ไม่ใช่ pattern matching
2. **Structural ambiguity — span ควรครอบถึงไหน**
   `This Agreement shall commence on the Effective Date and [continue for five (5) years unless earlier terminated]` — Expiration Date คือทั้ง clause หรือแค่ `five (5) years` (attachment problem ในระดับ span)
3. **Language variation ในเอกสารกฎหมาย** — ตัวพิมพ์ใหญ่ทั้งคำ (`TWENTIETH CENTURY FOX LICENSING & MERCHANDISING`), เครื่องหมายคำพูดหลายแบบ (`"`, `“`), การขึ้นบรรทัดจากการแปลง PDF, ตัวเลขเขียนซ้ำ `five (5)`
4. **เอกสารยาวมาก** — สัญญา CUAD ยาวเฉลี่ย ~52,000 ตัวอักษร (≈ 13,000 token) เกินขีดจำกัด 512 token ของ BERT-family ~25 เท่า และ **> 90 % ของเอกสารไม่มี entity เป้าหมายเลย** (ปัญหา class imbalance)

### ใช้ที่ไหนในโลกจริง

- **Contract review / due diligence** ใน M&A — ผู้สร้าง CUAD ระบุว่าการตรวจสัญญาด้วยคนเป็นงานที่แพงและซ้ำซาก จึงสร้าง dataset นี้เพื่อให้ AI ช่วย
- Contract lifecycle management: ค้นสัญญาที่กำลังหมดอายุ, ตรวจว่า governing law ตรง policy
- ผลลัพธ์เป็น **draft ให้คนตรวจ** (human-in-the-loop) — ระบบเก็บ offset, confidence และ flag ความถูกต้องของทุก span ให้ผู้ตรวจย้อนดูต้นฉบับได้

---

## 2. Data or Dataset

### 2.1 ที่มา

| | |
|---|---|
| ต้นฉบับ | **CUAD** — The Atticus Project, Hendrycks et al. (NeurIPS 2021 Datasets & Benchmarks) |
| ที่ใช้ | Kaggle `changethan/legal-contract-dataset` ซึ่งรวม CUAD + ContractNLI (ส่วน ContractNLI ไม่ได้ใช้ในงานนี้) |
| ชนิดข้อมูล | สัญญาเชิงพาณิชย์จริง **510 ฉบับ** จาก SEC EDGAR filings (บริษัทมหาชนสหรัฐฯ ต้องยื่นสัญญาสำคัญเป็นเอกสารสาธารณะ) ~25 ประเภทสัญญา |
| Annotation | ผู้เชี่ยวชาญด้านกฎหมาย annotate **41 ประเภท clause** (~13,000 annotation) โดยมี guideline ที่ผ่านการตรวจหลายชั้น |
| ไฟล์ใน Kaggle | `complete.json` — ข้อความเต็มของสัญญา; `all_reshaped_clauses.csv` — ข้อความ clause แต่ละประเภทต่อสัญญา |

### 2.2 ลักษณะสำคัญที่มีผลต่อการออกแบบ

**(ก) ไม่มี character offset** — CSV เก็บเฉพาะ *ข้อความ* clause ไม่ใช่ตำแหน่ง แต่ NER แบบ token classification ต้องรู้ว่า span อยู่ตรงไหนของเอกสาร → ต้องเขียนขั้นตอน **re-alignment**: ค้นหา clause ในข้อความเต็มโดยไม่สนช่องว่าง/ขึ้นบรรทัด/ตัวพิมพ์/เครื่องหมายคำพูด → หาเจอ **98.1 % ของ 4,308 span** ใน 5 ประเภทที่ใช้

**(ข) Annotation guideline ของ CUAD ไม่ใช่ NER guideline** — มีผลต่อการตีความคะแนน (§4):
- `Parties` ติดเฉพาะ **mention ที่นิยามคู่สัญญา** ในย่อหน้าเปิด — ชื่อเดียวกันที่ปรากฏอีก 30 ครั้งในสัญญา *ไม่ถูกติด*
- `Expiration Date` และ `Governing Law` เป็น **span ระดับ clause** ทั้งประโยค ไม่ใช่ entity สั้น ๆ
- `Effective Date` มักเป็น **วันเดียวกับ** `Agreement Date` (ใน test split มี START_DATE ที่แตกต่างจริงเพียง 10 span)

**(ค) เลือกใช้ 5 จาก 41 ประเภท** — เฉพาะที่เป็น "ค่า" สกัดเป็นช่อง metadata ได้:

| CUAD category | Label ในงานนี้ | span ใน train |
|---|---|---:|
| Parties | `PARTY` | 1,953 |
| Agreement Date | `AGREEMENT_DATE` | 363 |
| Effective Date | `START_DATE` | 76 |
| Expiration Date | `END_DATE` | 361 |
| Governing Law | `JURISDICTION` | 346 |

ประเภทอย่าง *Non-Compete*, *Change of Control*, *Liquidated Damages* เป็น clause ยาวทั้งย่อหน้า เหมาะกับ clause classification มากกว่า NER จึงไม่ใช้

**(ง) การแบ่งข้อมูล** — จาก 510 ฉบับ เหลือ **499** ที่มี span ของ 5 ประเภทนี้อย่างน้อยหนึ่งรายการและ align ได้ (11 ฉบับที่ไม่มีถูกตัดออก) แล้วแบ่งระดับ *สัญญา* (ไม่ให้เอกสารเดียวกันอยู่ทั้ง train และ test) แบบ deterministic จาก hash ของ id:

| split | สัญญา |
|---|---:|
| train | 398 |
| validation | 54 |
| test | 47 (370 gold span) |

**Input / Output ของระบบ:** input = ไฟล์ข้อความสัญญา (plain text); output = JSON ที่มี `summary` (ค่าเดียวต่อช่อง) และ `entities` (ทุก span พร้อม offset, confidence, ค่า normalised, flag `valid`)

---

## 3. Method, Model, or Tool

### 3.1 แนวคิดหลัก: Token classification ด้วย Transformer + rule-based post-processing

**Fine-tune pre-trained encoder** (`roberta-base`, 125M parameters) ให้ทำ **BIO tagging** ระดับคำ (11 label = B/I × 5 ชนิด + O; ชุด label ถูกกำหนดจาก `entity_types` ใน config ของแต่ละโมเดล) แล้วต่อด้วยกฎตรวจสอบและ normalise ค่า

ทำไมเป็น NER ไม่ใช่ text classification: ต้องการ *ตำแหน่งและข้อความ* ของค่า ไม่ใช่แค่รู้ว่าสัญญามี governing law clause หรือไม่
ทำไมไม่เป็น extractive QA แบบ CUAD ต้นฉบับ: QA ต้องรัน 1 ครั้งต่อ 1 คำถาม (41 ครั้งต่อสัญญา); NER รันครั้งเดียวได้ทุกชนิดพร้อมกัน และให้ทุก mention ไม่ใช่แค่คำตอบเดียว

### 3.2 Pipeline (เชื่อมกับลำดับ preprocessing ในคลาส)

```
raw text → normalization → tokenization (word) → subword encoding → windowing
        → RoBERTa encoder → BIO decoding → span validation → normalisation → JSON
```

| ขั้น | ทำอะไร | เชื่อมกับเนื้อหาคลาส |
|---|---|---|
| **Normalization** | **จงใจไม่แก้ข้อความ input เลย** — ไม่รวมช่องว่าง ไม่ lower-case (RoBERTa เป็น cased และตัวพิมพ์ใหญ่เป็นสัญญาณของชื่อนิติบุคคล) เพราะทุก span ต้องชี้กลับไปที่ offset ในต้นฉบับได้; ความทนต่อช่องว่าง/ขึ้นบรรทัด/เครื่องหมายคำพูดจาก PDF อยู่ที่ขั้น *re-alignment* ของ gold (§2.2 ก) ไม่ใช่ที่ input ของโมเดล | Section 2: normalization — ต้องเลือกว่าอะไรคือ "noise" อะไรคือ "signal" และที่นี่ "signal" รวมถึงตำแหน่งตัวอักษร |
| **Word tokenization** | regex แยกตาม whitespace และ punctuation — คำ = หน่วยที่ BIO tag ติด | Section 2: regex tokenization ภาษาอังกฤษ |
| **Subword encoding** | ตัดแต่ละคำผ่าน BPE ของ RoBERTa **ทีละคำ** โดยเติม prefix `Ġ` เฉพาะที่มีช่องว่างจริงนำหน้า → ไม่มี subword ข้ามขอบคำ (ถ้าตัดจากข้อความดิบ BPE จะรวม `"),` กับตัวอักษรท้ายชื่อบริษัทเป็นชิ้นเดียว → คำสุดท้ายของ entity ไม่มี piece ของตัวเอง label หาย) piece แรกของแต่ละคำถือ label; piece ที่เหลือถูก mask ตอน train | Section 2: subword tokenization แก้ OOV (ชื่อบริษัทแปลก ๆ, ศัพท์กฎหมาย) |
| **Windowing** | หน้าต่าง 256 subword, overlap 64; ตอน train **สุ่มเก็บ 15 % ของหน้าต่างที่ไม่มี entity** (แก้ imbalance); ตอน inference แต่ละ token ใช้ logits จากหน้าต่างที่มันอยู่ไกลขอบที่สุด | รับเอกสาร 13,000 token |
| **Encoder + classifier** | fine-tune 3 epoch, lr 3e-5, batch 16 ด้วย HF `Trainer` | Section 3: contextual embeddings — เวกเตอร์ของ `Bank of America` ต่างกันเมื่ออยู่ในย่อหน้าเปิดกับในเงื่อนไขชำระเงิน |
| **BIO decoding** | piece → คำ → span ตัวอักษร (IOB2 แบบผ่อนปรน: `I-` ที่ไม่มี `B-` นำหน้าเริ่ม span ใหม่) | |
| **Span validation** | ตรวจ *surface form* ของแต่ละ span ตามชนิด: วันที่สั้น ๆ ต้อง parse ได้ (span ระดับ clause ต้องมีตัวเลขหรือคำอย่าง year/term/expire), jurisdiction ต้องมี keyword court/law/arbitration/tribunal/jurisdiction, party ต้องมีตัวอักษรอย่างน้อยหนึ่งตัวและไม่ใช่ stop-word; span ที่ไม่ผ่านยังอยู่ใน output (`valid: false`) แต่ไม่ถูกยกขึ้น summary | rule-based stand-in ของ "span classification" |
| **Normalisation** | วันที่ → ISO-8601 (`dateutil`), Party A/B = นิติบุคคล 2 ชื่อแรกตามลำดับที่ปรากฏ — mention ซ้ำถูกรวมเป็นค่าเดียว และ defined term สั้น ๆ ในเครื่องหมายคำพูด (`"Company"`, `"Licensor"`) ถูกลดชั้นเป็น `Other_Parties` ในขั้นนี้ (ไม่ใช่ที่ validation), `Start_Date` fallback = `Agreement_Date` พร้อม flag `inferred_from` | NLU → structured output |

**สิ่งที่จงใจไม่ทำ (และเหตุผล):** ไม่ลบ stopword และไม่ทำ stemming/lemmatization — เพราะ NER ต้องรักษาลำดับและทุกคำใน span (`State **of** New York`, `five (5) years **from the** Effective Date`) และ offset ต้องกลับไปหาต้นฉบับได้ — หลักเดียวกับข้อควรระวัง "ห้ามลบ stopword ใน sentiment/n-gram" ในคลาส

### 3.3 ทำไมวิธีนี้เหมาะกับปัญหา

| ทางเลือก | ข้อจำกัดต่อปัญหานี้ |
|---|---|
| **Regex / rule-based** | จับวันที่และ `State of X` ได้ แต่ **ตัดสินชนิดไม่ได้** — วันที่ 3 ตัวในย่อหน้าเปิดมี pattern เดียวกัน; ชื่อบริษัทไม่มี pattern เลย |
| **BoW / TF-IDF + classifier** | sparse representation ที่ทิ้งลำดับคำ — ระบุ *ตำแหน่ง* span ไม่ได้ และไม่รู้ว่า `Corp.` กับ `Inc.` มีบทบาทเหมือนกัน |
| **Static embedding (Word2Vec) + CRF** | เวกเตอร์ของ `Date` เป็นค่าเดียวเสมอ (polysemy) — แยก Agreement/Effective/Expiration ไม่ได้จากคำเอง ต้องอาศัย feature รอบข้างที่ออกแบบมือ |
| **Contextual (RoBERTa) + fine-tune** ✅ | เวกเตอร์เปลี่ยนตามบริบททั้งประโยค; pre-train บนข้อความอังกฤษ 160 GB ทำให้รู้จักโครงสร้างประโยคกฎหมายอยู่แล้ว fine-tune ด้วย 398 สัญญาได้ |
| **Prompt LLM (zero-shot)** | ทำได้ แต่ควบคุม offset/format ยาก, ต้นทุนต่อสัญญา 13k token สูง, ข้อมูลออกนอกองค์กร — เหมาะเป็น baseline เปรียบเทียบ (ยังไม่ได้ทำ) |

เหตุผลที่เลือก **RoBERTa-base**: เป็น encoder มาตรฐานสำหรับ token classification ภาษาอังกฤษ, cased (รักษาสัญญาณตัวพิมพ์ใหญ่), ขนาด base fine-tune บนเครื่องส่วนตัว (Apple M2, ~30 นาที) ได้; ผู้สร้าง CUAD เองก็ใช้ RoBERTa/DeBERTa เป็น baseline

---

## 4. Results and Findings

### 4.1 วิธีวัด

วัดระดับ *เอกสารทั้งฉบับ* บน 47 สัญญา test (ผ่าน windowing, stitching, validation จริงเหมือนตอนใช้งาน) 3 มุมมอง:
- **strict** — span ตรงเป๊ะทั้งขอบและชนิด (seqeval)
- **relaxed** — ชนิดถูกและ span ซ้อนทับกัน (เลื่อนขอบยังนับ)
- **field** — ต่อเอกสาร เทียบ *เซตของค่าที่ต่างกัน* ต่อชนิด (คู่สัญญาถูกติด 5 ครั้ง = 1 ค่า) — สิ่งที่ผู้ใช้ตาราง metadata สนใจจริง

### 4.2 ผลรวม (ผลของผู้นำเสนอ)

| รอบ | สิ่งที่เปลี่ยน | strict F1 | relaxed F1 | field F1 |
|---|---|---:|---:|---:|
| v1 | BPE จากข้อความดิบ, ไม่มี validation | 0.254 | 0.607 | 0.358 |
| v3 | ตัด subword ทีละคำ | 0.514 | 0.654 | 0.603 |
| v4 | v3 + span validation (label space ร่วมกับโมเดลไทย 17 label; `reports/iteration2/`) | 0.584 | 0.735 | 0.657 |
| **final** | v4 train ใหม่ด้วย label space 5 ชนิด (11 label) | **0.619** | **0.738** | **0.688** |

strict P / R (final) = 0.553 / 0.703 · relaxed P / R = 0.660 / 0.838

รายชนิด (final):

| ชนิด | gold span | relaxed F1 | field F1 |
|---|---:|---:|---:|
| PARTY | 244 | 0.77 | 0.73 |
| AGREEMENT_DATE | 42 | 0.73 | 0.66 |
| START_DATE | 10 | 0.00 | 0.00 |
| END_DATE | 38 | 0.64 | 0.61 |
| JURISDICTION | 36 | 0.89 | 0.76 |

### 4.3 การตีความ

**(1) การกระโดดใหญ่สุดมาจาก tokenization ไม่ใช่โมเดล** — v1 → v3 (+0.26 strict) เปลี่ยนแค่วิธีตัด subword โมเดล/ข้อมูล/hyperparameter เหมือนเดิมทุกอย่าง ใน v1 ชื่อบริษัทที่ตามด้วย `"),` หรือ `,` จะเสียคำสุดท้ายของ span ไปเพราะ BPE รวม `.` ท้าย `Inc.` กับเครื่องหมายถัดไปเป็นชิ้นเดียว — ตรงกับกฎ "Garbage features → Unreliable results" ในคลาส: representation พังก่อนถึงโมเดล

**(2) relaxed recall = 0.84 แต่ strict = 0.62 → โมเดล "หาเจอ" แต่ "ขอบไม่ตรง"** — และขอบที่ไม่ตรงส่วนใหญ่มาจาก annotation style ของ CUAD:
- FP `JURISDICTION`: `shall be in any court of competent jurisdiction located in San Francisco, California.` — โมเดลติดทั้ง clause ตาม style ของ train แต่ gold ของสัญญานี้ติดสั้นกว่า
- FP `END_DATE`: `Subject to satisfaction of the conditions precedent set forth in Section 10 hereof, the GOB Sale shall commence at each…` — Expiration Date ใน CUAD เป็นประโยคยาว ขอบตรงเป๊ะยาก

**(3) Precision ของ PARTY ต่ำเพราะนิยาม gold ไม่ใช่เพราะโมเดลผิด** — จาก error ระดับ span ~220 รายการ (FP 160 / FN 60) **91 เป็น FP PARTY** ตัวอย่าง:
- `Fox Entertainment Group, Inc.`, `Twentieth Century Fox Film Corporation` — ชื่อคู่สัญญาจริงที่ปรากฏ *ครั้งหลัง* ในสัญญา ซึ่ง CUAD ไม่ติด (ติดเฉพาะย่อหน้าเปิด) → ถูกต้องในเชิงใช้งาน แต่นับเป็น FP ระดับ span
- `"Company"`, `"Licensee"`, `"Recipient"` — defined term ที่โมเดลติด → ขั้น normalisation ลดชั้นออกจาก `Party_A/B` แล้ว แต่ยังนับเป็น FP ใน strict
- นี่คือเหตุผลที่รายงาน **field F1** (0.73 สำหรับ PARTY) ควบคู่ — เมื่อยุบเป็นเซตค่าต่อเอกสาร ผลกระทบจาก mention ซ้ำหายไป

**(3b) ตัด label ที่ไม่มีข้อมูลออกช่วยเล็กน้อย** — v4 → final เปลี่ยนแค่ label space จาก 17 (ร่วมกับโมเดลไทย) เป็น 11 (5 ชนิดที่ CUAD มีจริง) seed/hyperparameter เท่าเดิม: strict +0.035, field +0.031 (JURISDICTION field 0.58 → 0.76, END_DATE field 0.49 → 0.61) ส่วนหนึ่งเป็น run-to-run noise ของ test set 47 ฉบับ แต่ทิศทางสอดคล้องกับการที่ classifier head ไม่ต้องเผื่อ 6 class ที่ไม่เคยเห็น

**(4) START_DATE = 0 คือข้อจำกัดของข้อมูล ไม่ใช่โมเดล** — `Effective Date` ใน CUAD ตรงกับ `Agreement Date` เกือบทุกครั้ง โมเดลจึงเรียนรู้ว่า "วันที่ในย่อหน้าเปิด = AGREEMENT_DATE" (10 span ที่ต่างจริงไม่พอให้แยก) error pattern ชัดมาก: ทุก FN START_DATE มี FP AGREEMENT_DATE ข้อความเดียวกันคู่กัน (`November 1, 2019`, `July 1, 2020`) ระบบจึง fallback `Start_Date = Agreement_Date` และ flag ว่า inferred

**(5) เทียบกับผู้สร้าง dataset (Hendrycks et al., 2021)** — รายงานว่าโมเดลที่ดีที่สุด (DeBERTa-xlarge, extractive QA) ได้ **AUPR ≈ 47.8 %** เฉลี่ยทั้ง 41 ประเภท และสรุปว่างานนี้ "ยากแม้สำหรับโมเดลใหญ่" — ผลของผู้นำเสนอชี้ทิศทางเดียวกัน แต่ **เทียบตัวเลขตรง ๆ ไม่ได้**: คนละ task formulation (QA vs NER), คนละ metric (AUPR vs F1), คนละ subset (41 vs 5 ประเภท), คนละ split — หลักเดียวกับข้อควรระวังเรื่องเปรียบเทียบ perplexity ข้าม tokenizer/dataset ในคลาส

### 4.4 ตัวอย่าง output จริง

`examples/english_service_agreement.txt` (สัญญาที่เขียนขึ้นเพื่อ demo ไม่อยู่ใน CUAD):

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

record เต็ม (`reports/example_english_service_agreement.json`) เก็บทุก span พร้อม offset และ confidence เช่น `{"label": "PARTY", "text": "Northwind Analytics, Inc.", "start": 52, "end": 77, "confidence": 0.98, "valid": true}` และ `Start_Date` มี `"inferred_from": "Agreement_Date"`

---

## 5. Limitations

### ทำได้ดี
- **Recall สูง** (relaxed 0.84) — หา entity เจอเกือบหมด เหมาะกับ human-in-the-loop (คนตัด FP ออกง่ายกว่าหา FN)
- JURISDICTION relaxed 0.89 / field 0.76, PARTY field 0.73 บนสัญญาจริงที่ไม่เคยเห็น
- รับเอกสาร 13,000 token ได้ด้วย windowing; รัน 47 สัญญาบน laptop ได้ในไม่กี่นาที
- output ตรวจสอบย้อนกลับได้ (offset + confidence + valid flag)

### ล้มเหลวตรงไหน
1. **ขอบ span ของ clause ยาว** (Expiration Date, Governing Law) — strict F1 ต่ำเพราะ gold เองไม่มีขอบที่สม่ำเสมอระหว่างสัญญา
2. **Effective Date แยกจาก Agreement Date ไม่ได้** — ข้อมูลไม่มีตัวอย่างเพียงพอ
3. **Defined terms และ mention ซ้ำ** — โมเดลติดทุกครั้งที่ชื่อปรากฏ (ซึ่งเป็นพฤติกรรม NER ปกติ) แต่ขัดกับ guideline ของ CUAD → ต้องพึ่ง validation + field-level aggregation
4. **ขนาด test set เล็ก** — 47 สัญญา / 370 span; START_DATE มีแค่ 10 → ตัวเลขรายชนิดมีช่วงความเชื่อมั่นกว้าง
5. **ครอบคลุม 5 ช่อง** — ไม่มีมูลค่าสัญญา, กำหนดชำระ, ค่าปรับ เพราะ CUAD ไม่มี category เหล่านี้ในรูป span สั้น (`Liquidated Damages` เป็นทั้งย่อหน้า)

### สมมติฐาน/ข้อจำกัดที่ผู้ใช้ควรรู้
- รับ **plain text** — สัญญาจริงเป็น PDF/สแกน ต้อง OCR ก่อน และ OCR error จะทำให้ผลตก (ไม่มี fuzzy matching)
- Train บนสัญญา **บริษัทมหาชนสหรัฐฯ** จาก SEC — สัญญาสไตล์อื่น (UK, สัญญาระหว่างบุคคล, ภาษาอื่น) เป็น distribution shift
- Party A/B กำหนดจาก *ลำดับที่ปรากฏ* ไม่ใช่บทบาท (Licensor/Licensee)
- ผลลัพธ์เป็น **draft** ควรมีคนตรวจก่อนใช้ในทางกฎหมาย

### สิ่งที่ปรับปรุงได้
| การปรับปรุง | คาดว่าจะช่วยอะไร |
|---|---|
| เพิ่ม **baseline**: regex-only และ CRF + hand-crafted features บน split เดียวกัน | แสดงหลักฐานเชิงตัวเลขว่า contextual embedding จำเป็น (ตอนนี้เป็นการอ้างเหตุผล) |
| **Re-annotate PARTY ให้ติดทุก mention** (หรือประเมินเฉพาะย่อหน้าเปิด) | ทำให้ strict precision สะท้อนความสามารถจริง |
| เพิ่ม CUAD category ที่เป็น span สั้น: *Notice Period to Terminate Renewal*, *Renewal Term* | ครอบคลุมช่อง metadata มากขึ้นโดยไม่ต้อง annotate ใหม่ |
| **Prompt-based LLM baseline** บน test split เดียวกัน | เทียบต้นทุน/ความแม่นยำกับ fine-tuning |
| โมเดลใหญ่ขึ้น (`roberta-large`, `deberta-v3`) หรือ legal-domain pre-trained (`legal-bert`) | ผู้สร้าง CUAD พบว่าโมเดลใหญ่ + pre-train เพิ่มบนสัญญาช่วยชัดเจน |
| Levenshtein / fuzzy alias matching สำหรับคู่สัญญา | ทนต่อ OCR error และรวม `Fox Entertainment Group, Inc.` กับ `Fox Entertainment Group` เป็นค่าเดียว |
| ขยายไปภาษาอื่น (โมเดลต่อภาษา, pipeline เดียวกัน) | โค้ดแยกส่วนที่ขึ้นกับภาษาไว้ที่ word segmentation + validation keywords (`contract_ner.lang`); ภาษาใหม่ต้องการ (1) encoder ของภาษานั้น (2) กฎภาษาชุดหนึ่ง (3) ข้อมูล annotate ใน JSONL format เดียวกัน — ภาษาไทยมีต้นแบบใน repo แล้ว (WangchanBERTa + ข้อมูลสังเคราะห์) |

---

## 6. Q&A — คำถามที่คาดว่าจะถูกถาม

**Q: ทำไมไม่ใช้ regex อย่างเดียว วันที่กับ "State of X" ก็มี pattern ชัด?**
A: regex หา *candidate* ได้ แต่ตัดสิน *ชนิด* ไม่ได้ — ย่อหน้าเปิดของสัญญามักมีวันที่ 2–3 ตัว (ลงนาม, มีผล, สัญญาฉบับก่อน) pattern เดียวกันหมด และชื่อบริษัทไม่มี pattern เลย ระบบนี้ใช้ regex ในขั้น validation เพื่อ *กรอง* span ที่โมเดลเสนอ (v3 → v4 +0.07 strict) ไม่ใช่เพื่อหา

**Q: ทำไมไม่ใช้ BoW/TF-IDF?**
A: เป็น representation ระดับ *เอกสาร* ที่ทิ้งลำดับคำ ตอบได้ว่า "สัญญานี้มี governing law clause ไหม" แต่ตอบไม่ได้ว่า "อยู่ตรงไหน ข้อความว่าอะไร" งานนี้ต้องการ representation ระดับ *token* ที่ขึ้นกับบริบท

**Q: strict F1 0.62 ถือว่าดีหรือแย่?**
A: ต้องดูคู่กับ relaxed recall 0.84 และ field F1 0.69 — โมเดลหาเจอ แต่ขอบและนิยาม FP ถูกกำหนดโดย annotation style ของ CUAD ที่ไม่ใช่ NER guideline ผู้สร้าง CUAD เองรายงาน AUPR ~48 % และเรียกงานนี้ว่ายาก ตัวเลขที่ "ใช้งานได้จริง" คือ field F1 เพราะผู้ใช้เห็นตารางค่าต่อเอกสาร ไม่ใช่ span

**Q: ทำไมต้องตัด subword ทีละคำ ต่างจาก `is_split_into_words=True` ของ HuggingFace อย่างไร?**
A: `is_split_into_words` เติม `Ġ` (space marker) ให้ *ทุก* คำรวมทั้ง punctuation ที่ติดคำก่อนหน้า → subword ต่างจากตอน pre-train วิธีในงานนี้เติม `Ġ` เฉพาะที่มีช่องว่างจริง piece จึงเหมือน raw text ทุกประการ แต่ไม่มี piece ข้ามขอบคำ — จุดเดียวนี้ทำให้ strict F1 0.25 → 0.51

**Q: ทำไม START_DATE เป็น 0 ไม่ถือว่าโมเดลล้มเหลว?**
A: เพราะใน CUAD, Effective Date = Agreement Date เกือบทุกฉบับ ใน test มี START_DATE ที่ต่างจริง 10 span จาก 370 โมเดลเรียนรู้ pattern ที่ถูกต้องตามข้อมูล (วันที่ในย่อหน้าเปิด = วันลงนาม) และระบบ fallback ให้พร้อม flag — ถ้าจะแก้ต้องมีข้อมูลที่สองวันนี้ต่างกันมากกว่านี้

**Q: ถ้าใช้ LLM (GPT/Claude) แทนล่ะ?**
A: ควรเป็น baseline ถัดไป — ข้อแลกเปลี่ยนคือ ต้นทุนต่อสัญญา (13k token × จำนวนสัญญา), ควบคุม offset/format ยาก, ข้อมูลสัญญาออกนอกองค์กร, และ reproducibility fine-tuned encoder ตัวเล็กรันในเครื่องได้ 100 % ให้ confidence ต่อ span และ deterministic

**Q: ข้อมูลจาก Kaggle ต่างจาก CUAD ต้นฉบับไหม?**
A: เนื้อหาเดียวกันแต่ถูก reshape เป็น CSV ที่ **ไม่มี offset** (ต้นฉบับเป็น SQuAD-style JSON ที่มี `answer_start`) จึงต้องทำ re-alignment เอง ได้ 98.1 % ที่หายไป 1.9 % คือ span ที่ข้อความใน CSV ไม่ตรงกับ `complete.json` (ช่องว่าง/อักขระพิเศษจาก PDF)
