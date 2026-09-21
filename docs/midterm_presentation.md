# Midterm Presentation — Automated Key Information Extraction from Contracts (NER / Information Extraction)

**หัวข้อ:** NER / Information Extraction — การสกัดข้อมูลสำคัญจากสัญญาภาษาไทยและอังกฤษ
**ประเภท:** Application + case study จากโปรเจกต์ที่ทำเอง (`contract-key-extractor`)
**โครงสร้าง:** Problem → Data → Method → Results → Limitations → Q&A

> ตัวเลขทั้งหมดในเอกสารนี้มาจากการทดลองของผู้นำเสนอเอง (`reports/*.json` ในโปรเจกต์) ยกเว้นส่วนที่ระบุว่าเป็นข้อมูลของผู้สร้าง dataset (CUAD)

---

## 1. Problem and Motivation

### ปัญหาคืออะไร

สัญญา (จ้างทำของ, เช่า, ซื้อขาย, กู้ยืม, NDA, …) เป็น **unstructured text** ที่ยาวหลายหน้า แต่ข้อมูลที่ฝ่ายกฎหมาย/บัญชี/จัดซื้อต้องใช้จริงมีไม่กี่ช่อง:

| ช่องข้อมูล | ตัวอย่าง |
|---|---|
| คู่สัญญา (Party A / B) | บริษัท เอบีซี จำกัด / บริษัท เอ็กซ์วายแซด จำกัด |
| วันที่ทำสัญญา / เริ่ม / สิ้นสุด | 1 มกราคม 2569 – 31 ธันวาคม 2569 |
| มูลค่าสัญญา | 2,500,000 บาท |
| กำหนดชำระเงิน | ภายใน 30 วันนับจากวันที่ได้รับใบแจ้งหนี้ |
| อัตราค่าปรับ | ร้อยละ 0.1 ของมูลค่าสัญญาต่อวัน |
| เขตอำนาจศาล / กฎหมายที่ใช้บังคับ | ศาลแพ่งกรุงเทพใต้ |

งานคือแปลง **ข้อความสัญญา → ตาราง metadata (JSON)** โดยอัตโนมัติ — เป็นงาน **NLU** ตามนิยามในคลาส (unstructured → structured representation) และเป็นงาน **NER** โดยตรง: หา span ของข้อความแล้วติด label ประเภท

### ทำไมน่าสนใจในเชิง NLP

ปัญหานี้รวม "ความยากของภาษา" (Section 1 ของคลาส) ไว้เกือบครบ:

1. **Lexical ambiguity** — คำเดียวกันแต่ label ต่างกันตามบริบท
   - `ร้อยละ 7` (ภาษีมูลค่าเพิ่ม → ไม่ใช่ entity) vs `ร้อยละ 15 ต่อเดือน` (ค่าปรับ → `PENALTY_RATE`)
   - `วันที่ 15 ของทุกเดือน` (กำหนดชำระ) vs `วันที่ 15 มีนาคม 2569` (วันเริ่มสัญญา) vs วันส่งมอบงาน (ไม่ใช่ entity)
   - ตัวเลข `2,500,000 บาท` อาจเป็นมูลค่าสัญญา, เงินประกัน, วงเงินประกันภัย หรือราคาต่อหน่วย
   → เหตุผลหลักที่ต้องใช้ **contextual embedding** ไม่ใช่ pattern matching
2. **Structural ambiguity** — span ควรครอบถึงไหน: `ชำระ [ภายใน 30 วัน นับจากวันที่ได้รับใบแจ้งหนี้]` คำว่า "นับจาก…" ขยายกำหนดชำระหรือเป็นประโยคใหม่ (attachment problem)
3. **Language variation** — ภาษาไทยไม่มีช่องว่างระหว่างคำ, ปี พ.ศ./ค.ศ., ตัวเลขไทย `๑๒ กุมภาพันธ์ ๒๕๖๗`, ตัวย่อ `ม.ค.`, และสัญญาสองภาษา (clause อังกฤษปนไทย)
4. **เอกสารยาวมาก** — สัญญา CUAD ยาวเฉลี่ย ~52,000 ตัวอักษร เกินขีดจำกัด 512 token ของ BERT หลายสิบเท่า

### ใช้ที่ไหนในโลกจริง

- Contract management / due diligence: ค้นสัญญาที่กำลังหมดอายุ, รวมมูลค่าสัญญาทั้งบริษัท
- ฝ่ายบัญชี: ดึงกำหนดชำระและค่าปรับไปตั้ง reminder
- ฝ่ายกฎหมาย: ตรวจว่าเขตอำนาจศาลตรงกับ policy หรือไม่
- ผลลัพธ์เป็น **draft ให้คนตรวจ** (human-in-the-loop) ไม่ใช่การตัดสินใจอัตโนมัติ — ระบบจึงเก็บ offset, ค่าความมั่นใจ และ flag ความถูกต้องของทุก span ไว้ให้ผู้ตรวจย้อนดูต้นฉบับได้

---

## 2. Data or Dataset

### 2.1 ภาษาอังกฤษ — CUAD (Contract Understanding Atticus Dataset)

| | |
|---|---|
| แหล่งที่มา | Kaggle: `changethan/legal-contract-dataset` (รวม CUAD + ContractNLI) — CUAD ต้นฉบับโดย The Atticus Project (Hendrycks et al., 2021) |
| ชนิดข้อมูล | สัญญาเชิงพาณิชย์จริงจาก SEC EDGAR filings เป็นภาษาอังกฤษ **510 ฉบับ** ที่ผู้เชี่ยวชาญด้านกฎหมาย annotate 41 ประเภท clause |
| รูปแบบใน Kaggle | `complete.json` (ข้อความเต็ม) + `all_reshaped_clauses.csv` (ข้อความ clause ต่อสัญญา **โดยไม่มีตำแหน่ง/offset**) |
| ที่ใช้ในงานนี้ | 398 ฉบับ (หลังกรองที่ align ได้), 5 ประเภทที่เป็น span สั้น: Parties, Agreement Date, Effective Date, Expiration Date, Governing Law |

**ลักษณะสำคัญที่มีผลต่อการออกแบบ:**
- ข้อมูล **ไม่มี character offset** → ต้องเขียนขั้นตอน re-alignment (ค้นหา clause ในข้อความเต็มแบบไม่สนช่องว่าง/เครื่องหมายคำพูด/ตัวพิมพ์) ได้ **98.1 % ของ 4,308 span**
- CUAD annotate **เฉพาะ mention ที่นิยามคู่สัญญา** (ในย่อหน้าเปิด) ไม่ได้ติดทุกครั้งที่ชื่อปรากฏ → มีผลต่อการตีความ precision (ดู §4)
- `Expiration Date` และ `Governing Law` เป็น **span ระดับ clause ยาว ๆ** ไม่ใช่ entity สั้น ๆ
- **ไม่มี** มูลค่าสัญญา / กำหนดชำระ / อัตราค่าปรับ ในรูป span ที่สะอาด → ภาษาอังกฤษครอบคลุมได้ 5 จาก 8 ช่อง

### 2.2 ภาษาไทย — ไม่มี dataset สาธารณะ → สร้างเอง

**ข้อเท็จจริงที่ต้องพูดตรง ๆ:** ไม่มีชุดข้อมูลสัญญาภาษาไทยที่ annotate NER เปิดเผยสาธารณะ ผู้นำเสนอจึง bootstrap ด้วย 2 ส่วน:

| ชุด | จำนวน | ที่มา | ใช้ทำอะไร |
|---|---:|---|---|
| Synthetic corpus | 3,000 สัญญา | template generator (`thai_synthetic.py`) — 9 ตระกูลสัญญา, สุ่มชื่อ/วันที่/จำนวนเงินหลายรูปแบบ, มี offset ของ entity แม่นยำ | train / validation |
| Dev gold | 24 สัญญา / 217 entity | เขียนด้วยมือในสไตล์ที่ generator ไม่ได้สร้าง (สัญญาราชการตัวเลขไทย, MOU, แฟรนไชส์, สัญญาสองภาษา) annotate ด้วย `[[text|LABEL]]` | error analysis ระหว่างรอบพัฒนา |
| **Blind gold** | 8 สัญญา / 73 entity | เขียนหลังจบการพัฒนาทั้งหมด **ไม่เคยเปิดดูผล** จนถึงการวัดครั้งสุดท้าย | ตัวเลขที่รายงานเป็นหลัก |

ลักษณะสำคัญของ synthetic corpus:
- surface form หลากหลาย: `1 มกราคม 2569`, `1 ม.ค. 2569`, `01/01/2569`, `๑ มกราคม ๒๕๖๙`, `ร้อยละ 15 ต่อปี` vs `15% ต่อเดือน`, `2.5 ล้านบาท`, `USD 48,000`
- **hard negatives** ที่จงใจใส่เพื่อสอนโมเดลแยก ambiguity: ผู้ลงนาม/พยาน (ชื่อคนที่ไม่ใช่คู่สัญญา), เลขทะเบียน, VAT 7 %, เงินประกัน, ดอกเบี้ยเงินกู้ (ไม่ใช่ค่าปรับ), กำหนดส่งมอบ (ไม่ใช่กำหนดชำระ)

**Input / Output ของระบบ:** input = ไฟล์ข้อความสัญญา (plain text); output = JSON ที่มี `summary` (ค่าเดียวต่อช่อง) และ `entities` (ทุก span พร้อม offset, confidence, ค่า normalised, flag `valid`)

---

## 3. Method, Model, or Tool

### 3.1 แนวคิดหลัก: Token classification ด้วย Transformer + rule-based post-processing

ใช้ **fine-tuned pre-trained encoder** (WangchanBERTa สำหรับไทย, RoBERTa สำหรับอังกฤษ) ทำ **BIO tagging** ระดับคำ แล้วต่อด้วยกฎตรวจสอบและ normalise ค่า

ทำไมเป็น NER แบบ token classification ไม่ใช่ text classification: เราต้องการ *ตำแหน่ง* ของข้อความ ไม่ใช่แค่รู้ว่าเอกสารมีค่าปรับหรือไม่

### 3.2 Pipeline (เชื่อมกับลำดับ preprocessing ในคลาส)

```
raw text → normalization → word segmentation → subword encoding → windowing
        → Transformer encoder → BIO decoding → span validation → normalisation → JSON
```

| ขั้น | ทำอะไร | เชื่อมกับเนื้อหาคลาส |
|---|---|---|
| **Normalization** | lower-case เฉพาะตัวอักษร ASCII (WangchanBERTa เป็น uncased — ตัวพิมพ์ใหญ่ Latin จะกลายเป็น `<unk>`) | Section 2: normalization |
| **Word segmentation (ไทย)** | PyThaiNLP `newmm` + **พจนานุกรมเฉพาะโดเมน**: แยกคำประสมที่กลืน entity (`เขตอำนาจศาล` → `เขตอำนาจ|ศาล`, `ตามกฎหมาย`, `อัตราร้อยละ`) และเพิ่มคำที่ newmm ตัดพัง (`เที่ยวละ`, `ลงวันที่`, ชื่อศาล) | Section 2: Thai tokenization — ไทยไม่มีช่องว่าง ต้องใช้ dictionary-based engine |
| **Word segmentation (อังกฤษ)** | regex แยก whitespace/punctuation | Section 2: regex |
| **Subword encoding** | ตัดแต่ละคำผ่าน SentencePiece/BPE **ทีละคำ** เพื่อไม่ให้ subword ข้ามขอบคำ (ถ้าตัดจากข้อความดิบ SentencePiece จะรวม `ของกฎหมาย` เป็นชิ้นเดียว → คำแรกของ entity ไม่มี piece ของตัวเอง label หาย) | Section 2: subword tokenization แก้ OOV |
| **Windowing** | หน้าต่าง 256 subword, overlap 64; ตอน inference แต่ละ token ใช้ logits จากหน้าต่างที่มันอยู่ไกลขอบที่สุด | รับมือเอกสารยาว 52k ตัวอักษร |
| **Encoder + classifier** | 17 label (B/I × 8 ชนิด + O) fine-tune ด้วย HF Trainer | Section 3: contextual embeddings — เวกเตอร์ของ `ร้อยละ` ต่างกันตามบริบท |
| **Span validation** | ตรวจแต่ละ span ตามชนิด: วันที่ต้อง parse ได้, jurisdiction ต้องมีคำว่าศาล/กฎหมาย/อนุญาโตตุลาการ, … span ที่ไม่ผ่านยังอยู่ใน output แต่ไม่ถูกยกขึ้น summary | rule-based stand-in ของ "span classification" |
| **Normalisation** | พ.ศ./ตัวเลขไทย → ISO-8601, จำนวนเงิน → `{amount, currency, per}`, ค่าปรับ → `{value, unit, per}`, ชื่อศาล → อังกฤษ; Party A/B = นิติบุคคล 2 ชื่อแรกตามลำดับที่ปรากฏ | NLU → structured output |

**สิ่งที่จงใจไม่ทำ (และเหตุผล):** ไม่ลบ stopword และไม่ทำ stemming/lemmatization — เพราะ NER ต้องรักษาลำดับและทุกคำใน span (`ภายใน 30 วัน **นับจาก** วันที่…`) การลบคำเชื่อมจะทำให้ขอบ span พังและ offset กลับไปหาต้นฉบับไม่ได้ — หลักเดียวกับข้อควรระวัง "ห้ามลบ stopword ใน sentiment/n-gram" ในคลาส

### 3.3 ทำไมวิธีนี้เหมาะกับปัญหา

| ทางเลือก | ข้อจำกัดต่อปัญหานี้ |
|---|---|
| **Regex / rule-based** | จับรูปแบบตายตัว (วันที่, จำนวนเงิน) ได้ แต่ **แยก ambiguity ไม่ได้** — `ร้อยละ 7` กับ `ร้อยละ 15` มี pattern เดียวกัน ต้องอาศัยบริบทรอบข้าง |
| **BoW / TF-IDF + classifier** | เป็น sparse representation ที่ทิ้งลำดับคำ — "Dog bites man" = "Man bites dog" — ใช้ระบุ *ตำแหน่ง* span ไม่ได้ และมองไม่เห็นบริบท |
| **Static embedding (Word2Vec) + CRF** | เวกเตอร์ของ `วันที่` เป็นค่าเดียวเสมอไม่ว่าจะเป็นวันชำระหรือวันเริ่มสัญญา (polysemy) |
| **Contextual (BERT-family) + fine-tune** ✅ | เวกเตอร์เปลี่ยนตามบริบท → แก้ polysemy ได้; pre-training บนภาษาไทยจำนวนมากทำให้ fine-tune ด้วยข้อมูลไม่มากได้ |
| **Prompt LLM (zero-shot)** | ทำได้แต่ควบคุม offset/ความสม่ำเสมอยาก, ต้นทุนต่อเอกสารสูง, ข้อมูลสัญญาส่งออกนอกองค์กร — เหมาะเป็น baseline เปรียบเทียบ (ยังไม่ได้ทำ) |

เหตุผลที่เลือก **WangchanBERTa** (`airesearch/wangchanberta-base-att-spm-uncased`): pre-train ด้วยข้อความไทยขนาดใหญ่, tokenizer SentencePiece ทำงานกับข้อความไม่มีช่องว่างได้, ขนาด base รัน fine-tune บนเครื่องส่วนตัว (Apple M2, ~35 นาที) ได้

---

## 4. Results and Findings

### 4.1 วิธีวัด

วัดระดับ *เอกสารทั้งฉบับ* (ผ่าน windowing, stitching, validation จริงเหมือนตอนใช้งาน) 3 มุมมอง:
- **strict** — span ตรงเป๊ะ (seqeval)
- **relaxed** — ชนิดถูกและ span ซ้อนทับกัน (เลื่อนขอบ 1 คำยังนับ)
- **field** — ต่อเอกสาร เทียบ *เซตของค่าที่ต่างกัน* ต่อชนิด (คู่สัญญาถูกติด 5 ครั้ง = 1 ค่า) — สิ่งที่ผู้ใช้ตาราง metadata สนใจจริง

### 4.2 ภาษาไทย (ผลของผู้นำเสนอ)

| ชุดทดสอบ | เอกสาร | entity | strict P / R / F1 | relaxed F1 | field F1 |
|---|---:|---:|---|---:|---:|
| **Blind gold** (ตัวเลขที่ยึดเป็นหลัก) | 8 | 73 | 0.733 / 0.863 / **0.792** | 0.881 | 0.803 |
| Dev gold (ใช้ทำ error analysis → มองโลกในแง่ดี) | 24 | 217 | 0.873 / 0.922 / 0.897 | 0.946 | 0.896 |
| Synthetic held-out (generator เดียวกับ train) | 300 | 2,662 | 0.997 / 0.993 / 0.995 | 0.997 | 0.994 |

รายชนิด (strict F1, blind / dev): START_DATE 1.00 / 0.95 · END_DATE 0.88 / 0.98 · AGREEMENT_DATE 0.82 / 0.94 · CONTRACT_VALUE 0.94 / 0.85 · JURISDICTION 0.88 / 0.96 · PARTY 0.81 / 0.89 · PENALTY_RATE 0.67 / 0.88 · **PAYMENT_DUE 0.42 / 0.77**

**การตีความ:**
- ช่องว่าง synthetic 0.995 → blind 0.792 คือ **distribution shift** ระหว่างข้อมูลที่ generator สร้างกับสัญญาที่เขียนอิสระ — วัด "ความจำ template" ไม่ได้ ต้องวัดบนข้อมูลที่โมเดลไม่เคยเห็นสไตล์
- relaxed สูงกว่า strict ~0.09 → ข้อผิดพลาดจำนวนมากคือ **ขอบ span คลาดเคลื่อน 1 คำ** ไม่ใช่หาไม่เจอ — ต้นตอมาจาก word segmentation (ขอบ entity ตกได้เฉพาะขอบคำของ newmm) ตรงกับที่คลาสบอกว่า tokenization ไทยเป็นปัญหาต้นน้ำ
- recall (0.86) สูงกว่า precision (0.73) → โมเดล "ติดเกิน" มากกว่า "ติดขาด" — เป็นทิศทางที่เหมาะกับ human-in-the-loop (คนตัดออกง่ายกว่าหาเพิ่ม)
- ช่องที่ **มีรูปแบบชัด** (วันที่, ศาล, จำนวนเงิน) ถ่ายทอดจาก synthetic ไปข้อมูลจริงได้ดี; ช่องที่ **free-form** (กำหนดชำระ, ค่าปรับ) ต่ำสุด — เป็นจุดที่ข้อมูล annotate จริงจะช่วยมากที่สุด

**การพัฒนาแบบวนรอบ (dev gold strict / relaxed F1):**

| รอบ | สิ่งที่เปลี่ยน | ผล |
|---|---|---|
| v1 | ตัด subword จากข้อความดิบ, พจนานุกรม newmm มาตรฐาน | 0.733 / 0.846 |
| + dictionary | เพิ่มพจนานุกรมโดเมน (แยก `เขตอำนาจศาล`) — โมเดลเดิม | 0.748 / 0.852 |
| v2 | error analysis → generator เพิ่ม `ลงวันที่`, ตารางสรุป, ค่าปรับต่อหน่วย, hard negatives | 0.894 / 0.962 |
| v3 | ตัด subword ทีละคำ (ไม่มี piece ข้ามขอบคำ) | 0.899 / 0.953 |
| v4 | แยกคำประสม `ตามกฎหมาย`/`อัตราร้อยละ`, ประเมินหลัง validation | 0.897 / 0.946 (blind 0.792 / 0.881) |

ข้อค้นพบ: กระโดดใหญ่สุด (v1 → v2, +0.16) มาจาก **การแก้ข้อมูล** ไม่ใช่การแก้โมเดล — สอดคล้องกับกฎ "Garbage in → Garbage features" ในคลาส

**ตัวอย่างข้อผิดพลาดจาก blind set:**
- `เป็นรายไตรมาส` ถูกติดเป็น PAYMENT_DUE (FP) — รูปแบบความถี่การชำระที่ generator ไม่มี
- `15 พฤศจิกายน 2568` ใน "หนังสือแจ้งต่ออายุ" ถูกติดเป็น AGREEMENT_DATE — จริง ๆ เป็นวันที่ออกหนังสือ (ambiguity ระดับเอกสาร)
- `เทศบาลตำบลแม่ริม` → โมเดลติดแค่ `แม่ริม` (dev) — ขอบคำหน่วยงานราชการ

### 4.3 ภาษาอังกฤษ — CUAD (ผลของผู้นำเสนอ)

47 สัญญา held-out, 370 gold span, 5 ชนิด

| รอบ | strict F1 | relaxed F1 | field F1 |
|---|---:|---:|---:|
| v1 — BPE จากข้อความดิบ, ไม่มี validation | 0.254 | 0.607 | 0.358 |
| v3 — ตัด subword ทีละคำ (เครื่องหมายวรรคตอนไม่ติดชื่อบริษัท) | 0.514 | 0.654 | 0.603 |
| **final** — + span validation | **0.584** | **0.735** | **0.657** |

รายชนิด (field F1): PARTY 0.72 · AGREEMENT_DATE 0.68 · JURISDICTION 0.58 · END_DATE 0.49 · START_DATE 0.00

**การตีความ (แยกจากผลของผู้สร้าง CUAD):**
- ตัวเลข strict ต่ำกว่าไทยมาก แต่ **relaxed recall = 0.87** → โมเดล *หาเจอ* แต่ *ขอบไม่ตรง* เพราะ `Expiration Date` / `Governing Law` ใน CUAD เป็น clause ยาวทั้งประโยค
- precision ต่ำเพราะ CUAD ติด label เฉพาะ mention นิยาม; mention ครั้งต่อ ๆ ไปที่โมเดลติด (ซึ่งถูกต้องในเชิงการใช้งาน) นับเป็น FP → **annotation guideline ของ dataset กำหนดว่าตัวเลขแปลว่าอะไร**
- START_DATE = 0 เพราะ `Effective Date` ใน CUAD ตรงกับ `Agreement Date` เกือบทุกครั้ง (ทั้ง test split มีเพียง 10 span ที่แตกต่าง) → โมเดลแยกไม่ได้ ระบบจึง fallback `Start_Date = Agreement_Date` และ flag ไว้
- **ผู้สร้าง CUAD รายงาน** (Hendrycks et al., 2021) ว่า DeBERTa-xlarge ได้ AUPR ≈ 47.8 % บน 41 ประเภท ซึ่งชี้เหมือนกันว่างานนี้ยากแม้สำหรับโมเดลใหญ่ — ตัวเลขนั้นเป็น extractive-QA บนคนละ metric จึง **เทียบตรงกับผลข้างต้นไม่ได้** (หลักเดียวกับข้อควรระวังเรื่องเปรียบเทียบ perplexity ข้าม tokenizer ในคลาส)

### 4.4 ตัวอย่าง output จริง

สัญญาจ้างบริการภาษาไทย (`examples/thai_service_contract.txt`):

```json
{
  "Party_A": "บริษัท เอบีซี จำกัด",
  "Party_B": "บริษัท เอ็กซ์วายแซด จำกัด",
  "Agreement_Date": null,
  "Start_Date": "2026-01-01",
  "End_Date": "2026-12-31",
  "Contract_Value": "2,500,000 THB",
  "Payment_Due_Date": "day 5 of each month",
  "Penalty_Rate": "15% per month",
  "Jurisdiction": "Bangkok Civil Court"
}
```

สัญญาสองภาษา (ไทย/อังกฤษ) จาก dev gold — โมเดลไทยได้ 8/9 ช่อง (พลาดคู่สัญญาชื่ออังกฤษ `Siam Digital Ventures Co., Ltd.`), โมเดลอังกฤษได้เฉพาะช่องที่เป็นอังกฤษ → ทั้งสองโมเดลเสริมกัน แต่ยังไม่มีโมเดลเดียวที่รับ code-switching ได้ (ดู §5)

---

## 5. Limitations

### ทำได้ดี
- ช่องที่มีรูปแบบชัด (วันที่ทุกรูปแบบรวมตัวเลขไทย/พ.ศ., จำนวนเงิน, ศาล) — F1 0.85–1.00 บนข้อมูลที่ไม่เคยเห็น
- แยก hard negatives ที่ออกแบบไว้ได้ (VAT, เงินประกัน, ผู้ลงนาม)
- รับเอกสารยาวเกิน 512 token ได้ด้วย windowing
- output ตรวจสอบย้อนกลับได้ (offset + confidence + valid flag)

### ล้มเหลวตรงไหน
1. **ข้อมูล train ภาษาไทยเป็น synthetic ทั้งหมด** — โมเดลไม่เคยเห็นสัญญาจริงแม้แต่ฉบับเดียว และ gold set เขียนโดยคนเดียวกับที่เขียน template → แม้ blind score ก็ยังน่าจะสูงกว่าความเป็นจริงในสัญญาที่มีการใช้ภาษาแปลก ๆ ชุดทดสอบเล็ก (8 ฉบับ) ช่วงความเชื่อมั่นกว้าง
2. **ข้อผิดพลาดของ word segmentation ส่งต่อไปทั้ง pipeline** — ขอบ entity ตกได้เฉพาะขอบคำของ newmm คำประสมที่พจนานุกรมไม่รู้จักทำให้ span กว้าง/แคบไป 1 คำ (ช่องว่าง strict vs relaxed ≈ 0.09)
3. **ช่อง free-form** (กำหนดชำระ, ค่าปรับ) ต่ำสุด — ภาษามีหลายแบบเกินกว่า template จะครอบคลุม
4. **ภาษาอังกฤษครอบคลุม 5/8 ช่อง** และขอบ clause ยาวใน CUAD ทำให้ strict F1 ต่ำ
5. **สัญญาสองภาษา** — แต่ละโมเดลเห็นภาษาเดียว WangchanBERTa ตัดคำอังกฤษเป็นชิ้นเล็กมาก (`agreement` → `a·gre·e·ment`) RoBERTa ไม่รู้จักไทยเลย
6. **Ambiguity ระดับเอกสาร** — วันที่ออกหนังสือแจ้งต่ออายุ vs วันที่ทำสัญญา ต้องเข้าใจว่าเอกสารนี้คืออะไร ซึ่ง NER ระดับ span ไม่พอ

### สมมติฐาน/ข้อจำกัดที่ผู้ใช้ควรรู้
- รับ **plain text** เท่านั้น — สัญญาจริงเป็น PDF สแกน ต้องผ่าน OCR ก่อน และ OCR error จะทำให้ผลตก (ระบบไม่มี fuzzy matching)
- Party A/B กำหนดจาก *ลำดับที่ปรากฏ* ไม่ใช่บทบาท (ผู้ว่าจ้าง/ผู้รับจ้าง)
- Normalisation เป็นกฎที่ครอบคลุมรูปแบบที่พบบ่อย ค่าที่ไม่เข้ากฎจะคืนข้อความดิบ
- ผลลัพธ์เป็น **draft** ควรมีคนตรวจก่อนใช้ในทางกฎหมาย/การเงิน

### สิ่งที่ปรับปรุงได้
| การปรับปรุง | คาดว่าจะช่วยอะไร |
|---|---|
| เก็บและ annotate สัญญาไทยจริง (Label Studio/doccano → JSONL รูปแบบเดิม) | แก้ข้อ 1 และ 3 โดยตรง — คุ้มค่าที่สุด |
| เพิ่ม baseline: regex-only และ CRF + hand-crafted features | แสดงหลักฐานว่า contextual embedding จำเป็นจริง (ตอนนี้เป็นการอ้างเหตุผล ยังไม่มีตัวเลขเทียบ) |
| Tokenizer ablation: newmm vs attacut vs deepcut | วัดว่าข้อ 2 ส่งผลต่อ strict F1 เท่าไร |
| โมเดล multilingual ตัวเดียว (XLM-R) train ไทย synthetic + CUAD รวมกัน + template สองภาษา | รับสัญญาสองภาษา (ข้อ 5) |
| Levenshtein / fuzzy alias matching สำหรับคู่สัญญา | ทนต่อ OCR error และการเว้นวรรคต่างกัน (`บริษัท เอบีซี` vs `บริษัทเอบีซี`) |
| Prompt-based Thai LLM baseline | เทียบต้นทุน/ความแม่นยำกับ fine-tuning |
| Layout-aware input (PDF/DOCX) | ใช้ตำแหน่งบนหน้า (ตารางสรุป, ลายเซ็น) เป็นสัญญาณเพิ่ม |

---

## 6. Q&A — คำถามที่คาดว่าจะถูกถาม

**Q: ทำไมไม่ใช้ regex อย่างเดียว วันที่กับจำนวนเงินก็มี pattern ชัด?**
A: regex หา *candidate* ได้ แต่ตัดสิน *ชนิด* ไม่ได้ — `ร้อยละ 7` (VAT) และ `ร้อยละ 15 ต่อเดือน` (ค่าปรับ) มี pattern เดียวกัน ต้องดูบริบท ("ภาษี" vs "ค่าปรับ/ผิดนัด") ระบบนี้ใช้ regex ในขั้น validation เพื่อ *กรอง* span ที่โมเดลเสนอ ไม่ใช่เพื่อหา

**Q: ทำไมไม่ใช้ BoW/TF-IDF?**
A: เป็น representation ที่ทิ้งลำดับคำ ใช้ระบุตำแหน่ง span ไม่ได้ และเป็น sparse vector ที่ไม่รู้ว่า `ศาลแพ่ง` กับ `ศาลจังหวัด` ใกล้กัน งานนี้ต้องการ representation ระดับ token ที่ขึ้นกับบริบท

**Q: ข้อมูล train เป็น synthetic แล้วเชื่อถือได้แค่ไหน?**
A: เชื่อได้ในระดับ proof-of-concept — จึงรายงาน 3 ชุดแยกกัน และยึด blind set ที่ไม่เคยเปิดดูเป็นหลัก (0.79 ไม่ใช่ 0.99) ข้อจำกัดคือ gold เขียนโดยคนเดียวกับ template; ขั้นถัดไปคือข้อมูลจริง

**Q: ทำไมคะแนนอังกฤษต่ำกว่าไทย ทั้งที่ CUAD เป็นข้อมูลจริง?**
A: เพราะ annotation guideline ต่างกัน — CUAD ติดเฉพาะ mention นิยามและ clause ยาว ส่วน guideline ไทยติดทุก mention และ span สั้น relaxed recall ของอังกฤษอยู่ที่ 0.87 แสดงว่าโมเดล *หาเจอ* ปัญหาอยู่ที่ขอบและนิยาม FP — ตัวเลขข้ามชุดข้อมูลเทียบกันตรง ๆ ไม่ได้

**Q: ทำไมต้องตัด subword ทีละคำ?**
A: SentencePiece ตัดจากข้อความดิบจะรวม `ของกฎหมาย` เป็นชิ้นเดียว → คำว่า `กฎหมาย` (คำแรกของ entity) ไม่มี subword ของตัวเอง label จึงหายไป การตัดทีละคำทำให้ทุกคำมี piece แรกที่ถือ label ได้ (strict F1 อังกฤษ 0.25 → 0.51 จากการแก้จุดนี้จุดเดียว)

**Q: ถ้าใช้ LLM (GPT/Claude) แทนล่ะ?**
A: ทำได้และควรเป็น baseline — ข้อแลกเปลี่ยนคือ ต้นทุนต่อเอกสาร, ควบคุม offset/format ยาก, ข้อมูลสัญญาออกนอกองค์กร, และ reproducibility fine-tuned encoder ตัวเล็กรันในเครื่องได้ 100 % และให้ confidence ต่อ span

**Q: รับสัญญาสองภาษาได้ไหม?**
A: ได้บางส่วน (โมเดลไทย 8/9 ช่องบนสัญญาผสม) แต่ยังไม่มีโมเดลเดียวที่ออกแบบสำหรับ code-switching — ทางแก้คือ XLM-R train ด้วยข้อมูลรวมสองภาษา
