# สคริปต์เดโม — Contract Key-Information Extractor

> เอกสารนี้คือ "บทพูด + ลำดับคลิก" สำหรับเดโมสดประมาณ 13 นาที ใช้คู่กับหน้าเว็บ `contract-ner serve`
> ตัวเลขทุกตัวมาจาก `reports/` และจากสัญญาเดโมใน `demo/samples/` (ค่าในตารางที่ระบุในสคริปต์ตรวจแล้วว่าตรงกับผลจริงของโมเดล)
> ภาพหน้าจอแต่ละขั้นอยู่ใน `demo/recording/screenshots/` · วิดีโอสำรอง `demo/recording/demo_walkthrough.mp4`

---

## ประโยคหลักที่ต้องการให้คนดูจำ

> **"ระบบนี้ไม่ได้แทนคนตรวจสัญญา แต่เปลี่ยนงานจาก 'อ่านสัญญาทั้งฉบับ' เป็น 'ตรวจ 6 ช่อง ที่ทุกช่องคลิกแล้วกระโดดไปที่ต้นฉบับได้'"**
> หลักฐาน: slot F1 = **0.855** บน 47 สัญญาที่โมเดลไม่เคยเห็น ผ่านเป้า "ใช้ได้เมื่อมีคนตรวจ" ≥ 0.80

---

## เตรียมก่อนวันจริง (checklist)

| | สิ่งที่ต้องทำ | คำสั่ง / วิธี |
|---|---|---|
| ☐ | ติดตั้ง extra ของเดโม (ครั้งเดียว) | `pip install -e ".[demo,thai,baseline]"` |
| ☐ | เปิด server ก่อนขึ้นพูด (โหลดโมเดลล่วงหน้า ~3 วินาที) | `contract-ner serve --preload roberta crf thai` |
| ☐ | เปิด `http://127.0.0.1:8000/` แล้วกด Extract หนึ่งรอบเพื่อ warm-up | — |
| ☐ | เปลี่ยนเป็นธีมสว่าง (ปุ่ม ◐ มุมขวาบน) — โปรเจกเตอร์อ่านง่ายกว่า | ค่าที่เลือกจะถูกจำไว้ในเบราว์เซอร์ |
| ☐ | ซูมเบราว์เซอร์ 110–125% ถ้าห้องใหญ่ | Cmd + |
| ☐ | ปิด notification / Slack / อีเมล | — |
| ☐ | ทดสอบโหมดสำรอง: ปิด server แล้วเปิดใหม่ด้วย `--cached-only` ว่ายังทำงาน | `contract-ner serve --cached-only` |
| ☐ | เปิดวิดีโอสำรองไว้ในแท็บ/แอปอีกตัว | `demo/recording/demo_walkthrough.mp4` (55 วินาที มีคำบรรยายในภาพ) |
| ☐ | ถ้าแก้โค้ดหลังจากนี้ ให้อัดวิดีโอ + screenshot ใหม่ | `python scripts/record_demo.py` (ต้องเปิด server ไว้) |

**คีย์ลัดบนหน้าเว็บ:** `1` / `2` / `3` = สลับแท็บ Extract / Compare / Evaluation · `N` = เปิด/ปิด presenter notes (talking point ของสัญญาที่เลือก — **ปิดไว้ตอนแชร์จอ** ถ้าไม่อยากให้คนดูเห็น)

---

## ลำดับการนำเสนอ (≈ 13 นาที)

### 0:00 – 1:00 · เปิดเรื่อง (Hook)

**ทำ:** แท็บ Extract → เลือก *CUAD contract — the longest in the test split* (338,211 ตัวอักษร) → เลื่อน scroll ยาว ๆ ให้เห็นว่ายาวแค่ไหน (ยังไม่ต้องกด Extract)

**พูด:**
> "นี่คือสัญญาจริงจากชุดข้อมูล CUAD ยาว 3 แสนกว่าตัวอักษร ฝ่ายจัดซื้อหรือฝ่ายกฎหมายต้องการรู้แค่ไม่กี่อย่าง — คู่สัญญาเป็นใคร เซ็นวันไหน มีผลวันไหน หมดอายุเมื่อไร และใช้กฎหมายที่ไหน ถามว่า governing law อยู่ตรงไหนในเอกสารนี้? … นี่คือปัญหาที่โปรเจกต์นี้แก้"

### 1:00 – 3:00 · เดโม 1: กรณีที่ทำงานได้ดี

**ทำ:**
1. เลือก *Service agreement (short example)* → กด **Extract** (ใช้เวลา < 1 วินาที)
2. ชี้ตารางด้านขวา: Party A/B, Agreement date, Start date (ป้าย *inferred from Agreement Date*), End date, Jurisdiction พร้อม confidence
3. **คลิกช่อง Jurisdiction / governing law** → หน้าจอกระโดดไปที่ clause ข้อ 5 และกระพริบ
4. ชี้สีไฮไลต์: แต่ละสีคือ label; เส้นใต้ประ = confidence < 0.8; ช่องที่ต่ำกว่า 0.8 จะมีธง ⚑ *review*

**พูด:**
> "ทุกค่าในตารางเก็บตำแหน่งตัวอักษร (offset) และ confidence ไว้ตลอด pipeline คนตรวจจึงคลิกแล้วเห็นต้นฉบับทันที — นี่คือเหตุผลที่ระบบ 'ใช้ได้เมื่อมีคนตรวจ' ไม่ใช่กล่องดำ"

**ต่อด้วย:** เลือก *CUAD contract — every cell right* → Extract → ติ๊ก **Compare with gold**
- มุมขวาบนของตารางขึ้น **5 / 5 slots match gold**
- ชี้ Start date = **2008-01-01** ไม่เท่ากับ Agreement date **2007-12-05** → "สัญญานี้เซ็นวันหนึ่ง มีผลอีกวันหนึ่ง กฎ effective-date split แยกให้ถูก" (ใน Compare จะเห็นว่า baseline ทั้งสองตอบ Start date ผิดเป็นวันเซ็น)

### 3:00 – 6:00 · ระบบทำงานยังไง (สไลด์)

ใช้สไลด์จาก `docs/project_overview_th.md` §17 (สไลด์ 3, 6, 7, 8, 10) — pipeline 6 ขั้น, ทำไมยาก (1.03% ของตัวอักษรเป็น entity, label เดียวหลายรูปทรง), per-word encoding (strict F1 0.254 → 0.514), windowing 256/64

> เคล็ดลับ: ถ้าต้องการโชว์ว่า windowing ทำงานจริง กลับไปที่สัญญายาวสุดแล้วกด Extract — ใช้ประมาณ **7.5 วินาที** สำหรับ 338k ตัวอักษร (ตารางขึ้น Party A = INTERNATIONAL BUSINESS MACHINES CORPORATION)

### 6:00 – 8:00 · เดโม 2: เทียบโมเดล

**ทำ:** กด `2` (Compare models) → เลือก *CUAD contract — parties the baselines miss* → กด **Compare**

**สิ่งที่จะเห็น:**

| ช่อง | Pattern-only | CRF | RoBERTa |
|---|---|---|---|
| Party B | Trust ✗ | Trust ✗ | **PFS Funds** ✓ |
| ช่องอื่น (วันที่, End date, Massachusetts) | ✓ | ✓ | ✓ |
| รวม | 4 / 5 | 4 / 5 | **5 / 5** |

**พูด:**
> "ทั้งสามโมเดลผ่าน validation และ normalisation ชุดเดียวกัน ต่างกันแค่ตัว tagger — baseline ทั้งสองหยิบคำว่า 'Trust' ที่ซ้ำอยู่ในเอกสารเป็นคู่สัญญา ส่วนโมเดลที่ fine-tune แล้วรู้ว่า defining mention คือ 'PFS Funds' — การแยก mention แรกที่นิยามชื่อออกจาก mention ซ้ำ ๆ ต้องใช้บริบททั้ง preamble ซึ่ง feature ที่ออกแบบด้วยมือมองไม่เห็น"

### 8:00 – 9:30 · หน้า Evaluation

**ทำ:** กด `3`

- Tile: **0.855** (slot F1, 47 สัญญา) · precision/recall **0.87 / 0.84** · **+0.093** เหนือ regex · ไทย blind **0.903**
- กราฟแท่ง: regex 0.762 · CRF 0.690 · RoBERTa 0.855 เทียบเส้นเป้า 0.80
- กราฟเส้น (hover จุด v7): "v6 → v7 ใช้ weights ชุดเดิม คะแนนตารางขึ้นจาก 0.804 เป็น 0.840 ด้วย normalisation + selection rules + confidence floor — **บทเรียนใหญ่ที่สุดคือวัดให้ถูกและ post-process ให้ดี ไม่ใช่ retrain**"
- กราฟรายช่อง: CRF ดีเท่า RoBERTa บน Jurisdiction (0.96) แต่ตกบน Parties (0.61 vs 0.87); regex ใกล้ ๆ บนช่องใน preamble แต่ตกบน End date (0.55 vs 0.78)

### 9:30 – 11:00 · เดโม 3: พูดตรง ๆ เรื่องจุดอ่อน

**ก. error จริง** — แท็บ Extract → *CUAD contract — an end-date error* → Extract → ติ๊ก Compare with gold → ตารางขึ้น **3 / 5**
- ใน End date: ตารางเลือก clause *"The license granted hereby shall commence upon the delivery…"* (confidence 0.99) ✗ — gold คือ **2014-11-30**
- เลื่อนเอกสารขึ้นไปช่วงต้น จะเห็น clause *"…end on November 30, 2014 (the "Initial Term")"* มีไฮไลต์สีเหลือง **และ** กรอบประของ gold → "โมเดลเจอ clause ที่ถูกแล้ว (0.94) แต่กฎเลือกหยิบอันที่ confidence สูงกว่า — ต้องแก้ที่ selection rule ไม่ใช่เทรนใหม่ (เป็น open item ใน worklog)"
- ช่อง Party A ขึ้น *party set differs from gold* และบอกว่า **also tagged as a party: corporation oriental financial group** → เศษจาก signature block ที่ถูกนับเป็นคู่สัญญาเกิน

**ข. gold gap** — *CUAD contract — a value the gold set lacks* → Extract → ติ๊ก Compare with gold → **คลิกช่อง Jurisdiction**
- ช่องขึ้น **? not in gold** และ gold = *(no gold value)*
- เอกสารกระโดดไปข้อ 9 *Governing Law*: "…governed by … the laws of the People's Republic of China…"
> "โมเดลถูก แต่ CUAD ไม่ได้ annotate ไว้ ระบบนับเป็น false positive — ตัวเลขบน CUAD จึงเป็นค่าต่ำสุด ไม่ใช่เพดาน"

> ⚠️ อย่าอ้างตัวเลข "5 จาก 9 ช่องเป็น gold gap" กับโมเดลตัวสุดท้าย — ตัวเลขนั้นนับจากโมเดล v7 ส่วนโมเดล final มี 13 ช่องที่ทายเกินโดย gold ว่างและยังไม่ได้ตรวจด้วยมือ (ดู `docs/worklog_demo_app.md`)

### 11:00 – 12:00 · เดโม 4: ภาษาไทย

**ทำ:** เลือก *สัญญาภาษาไทย — มีเฉลย (dev gold)* (โมเดลเปลี่ยนเป็น WangchanBERTa ให้อัตโนมัติ) → Extract → ติ๊ก Compare with gold → **5 / 5**
- ชี้ "1 ตุลาคม 2568" → **2025-10-01** (แปลง พ.ศ. → ค.ศ.), "28 พฤษภาคม 2569" → **2026-05-28**
- ภาษาไทยมีช่องเพิ่ม: Contract value **4,280,000 THB**, Payment due, Penalty rate **0.1% per day of contract price**, ศาลปกครองนครราชสีมา → *Nakhon Ratchasima Administrative Court*

**พูด:**
> "API เดียวกัน หน้าเดียวกัน ที่เปลี่ยนคือโมเดลและ `LangRules` ของภาษา — ชุด blind gold ภาษาไทยที่เขียนหลังจบการพัฒนาและไม่เคยเปิดดู ได้ slot F1 0.903"

(สัญญาไทยที่โชว์มาจาก **dev gold** ไม่ใช่ blind set — blind set เก็บไว้ใช้วัดผลอย่างเดียว)

### 12:00 – 13:00 · ข้อจำกัด + งานถัดไป → Q&A

ใช้สไลด์ 18–19 จาก `docs/project_overview_th.md`: End_Date selection, gold gaps / re-annotation, PDF layout, LLM baseline

---

## แผนสำรอง

| ถ้าเกิด… | ทำแบบนี้ |
|---|---|
| โมเดลโหลดไม่ขึ้น / เครื่องช้า | เปิด server ใหม่ด้วย `contract-ner serve --cached-only` — สัญญาเดโมทุกฉบับตอบจาก record ที่เก็บไว้ (ป้าย *stored record* สีเหลืองในกล่อง Run) ส่วน Compare ก็ใช้ได้ |
| โมเดลหายไปตัวเดียว (เช่น Thai extra ไม่ได้ติดตั้ง) | ไม่ต้องทำอะไร — API จะ fallback ไปใช้ record ที่เก็บไว้ของตัวนั้นเอง และขึ้น toast บอกเหตุผล |
| server / เบราว์เซอร์ล่ม | เปิด `demo/recording/demo_walkthrough.mp4` (55 วินาที มีคำบรรยาย) แล้วพากย์ทับ |
| ไม่มีเวลาเดโมสด | ใช้ภาพ `demo/recording/screenshots/01…09` เป็นสไลด์ เรียงตามสคริปต์นี้ |
| ไม่มีอินเทอร์เน็ต | ไม่กระทบ — หน้าเว็บ ฟอนต์ และโมเดลอยู่ในเครื่องทั้งหมด |

**หมายเหตุ:** ถ้าเลื่อน slider confidence ไปค่าอื่นที่ไม่ใช่ 0.60 ระบบจะต้องรันโมเดลจริง (record ที่เก็บไว้ทำที่ 0.60) — ในโหมด `--cached-only` จึงอย่าขยับ slider (และ paste/upload ข้อความใหม่ไม่ได้ — หน้าเว็บจะขึ้นข้อความแจ้งว่า server รันแบบ cached-only)

---

## คำถามที่อาจโดนถามระหว่างเดโม

- **"ถ้าเอาสัญญาของเราเองมาลองล่ะ?"** → ปุ่ม *Paste text…* หรือ *Upload .txt* (UTF-8, ≤ 4 MB) — ยังไม่รับ PDF โดยตั้งใจ เพราะการอ่าน layout ของ PDF เป็นข้อจำกัดที่ประกาศไว้ ถ้าแปลงเป็น text ก่อนก็ใช้ได้
- **"ทำไมช่อง Party ถูกทั้งสองช่องแต่ขึ้น differs?"** → parties ถูกให้คะแนนเป็น *เซต* ของชื่อตามกติกา slot evaluation ชื่อที่ถูกแท็กเกินมา (เช่นเศษจาก signature block) ก็นับเป็นผิด — ดูบรรทัด *also tagged as a party*
- **"confidence มาจากไหน?"** → ค่าเฉลี่ยของ probability ของ label บนคำใน span; floor 0.6 เลือกจาก validation split
- **"เร็วแค่ไหน?"** → บน MacBook M2 (MPS): สัญญาขนาดกลาง 8–17k ตัวอักษร ≈ 0.2–0.4 วินาที, สัญญายาวสุด 338k ตัวอักษร ≈ 7.5 วินาที
- **"API ใช้ยังไง?"** → เปิด `http://127.0.0.1:8000/docs` (Swagger ที่ FastAPI สร้างให้) — `POST /api/extract` รับ `{"model": "roberta", "text": "..."}`
