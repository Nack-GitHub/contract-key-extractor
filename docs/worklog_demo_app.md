# Work log — Demo app (API + หน้าเว็บ + สคริปต์เดโม)

> **Branch:** `demo-app` (แตกจาก `main` ที่ `86f27df`) · **ยังไม่ได้ push** · commit ทั้งหมดอยู่ใน branch นี้
> สคริปต์สำหรับพูดตอนเดโมอยู่ที่ [`docs/demo_script_th.md`](demo_script_th.md) — เอกสารนี้คือรายงานว่าทำอะไรไป, ตัดสินใจอะไรแทนไปบ้าง และเจออะไรระหว่างทาง

---

## 1. สรุปสั้น

ทำครบทั้ง 4 ส่วนตามแผนที่คุยกันไว้ (A API · B front-end · C คัดสัญญาเดโม · D กันเดโมพัง) และเพิ่มสคริปต์นำเสนอ:

| ส่วน | ผลลัพธ์ | ไฟล์หลัก |
|---|---|---|
| A. API | FastAPI 7 endpoint + คำสั่ง `contract-ner serve` + extra `[demo]` | `src/contract_ner/api.py`, `cli.py`, `pyproject.toml` |
| B. Front-end | หน้าเดียว 3 แท็บ (Extract / Compare models / Evaluation) ไม่มี build step ทำงาน offline | `src/contract_ner/web/{index.html,app.js,style.css}` |
| C. สัญญาเดโม | 8 ฉบับ แต่ละฉบับเก็บ record ของทุกโมเดลไว้ + talking point ที่สร้างจากข้อมูลจริง | `scripts/pick_demo_docs.py`, `demo/samples/*.json` |
| D. กันพัง | fallback ไปใช้ record ที่เก็บไว้อัตโนมัติ, โหมด `--cached-only`, วิดีโอ 55 วินาที + screenshot 9 ภาพ | `scripts/record_demo.py`, `demo/recording/` |
| สคริปต์พูด | ลำดับคลิก + บทพูด 13 นาที + checklist + แผนสำรอง + Q&A | `docs/demo_script_th.md` |
| เทสต์ | `tests/test_api.py` 10 เทสต์ (ใช้โมเดล regex ไม่ต้องมี weights) — ทั้งชุด **142 passed**, `ruff` ผ่าน | |

**ลองเปิดดูได้เลย:**

```bash
git checkout demo-app
pip install -e ".[demo,thai,baseline]"
contract-ner serve --preload roberta crf thai      # แล้วเปิด http://127.0.0.1:8000/
```

ถ้าไม่อยากรันอะไรเลย ให้ดูภาพใน `demo/recording/screenshots/` หรือวิดีโอ `demo/recording/demo_walkthrough.mp4` (วิดีโอไม่ได้ commit เพราะไฟล์ใหญ่ แต่อยู่ในเครื่องแล้ว)

---

## 2. สิ่งที่สร้าง — รายละเอียด

### 2.1 API (`src/contract_ner/api.py`)

| Endpoint | ทำอะไร |
|---|---|
| `GET /api/health` | สถานะ + โมเดลที่โหลดแล้ว |
| `GET /api/models` | โมเดลเดโม 4 ตัว (roberta, crf, regex, thai), มี weights ไหม, ตัวเลข slot F1 จาก `reports/` |
| `GET /api/samples` / `GET /api/samples/{id}` | รายการสัญญาเดโม / ตัวสัญญา + gold spans + talking point |
| `POST /api/extract` | `{model, text \| sample_id, min_confidence}` → record จาก `assemble()` เหมือน CLI ทุกอย่าง + `gold_check` รายช่องถ้าเป็นสัญญาที่มีเฉลย |
| `POST /api/extract/file` | อัปโหลด `.txt` (UTF-8, ≤ 4 MB) |
| `POST /api/compare` | สัญญาเดียวผ่านหลายโมเดล + รายชื่อช่องที่โมเดลเห็นไม่ตรงกัน |
| `GET /api/metrics` | ตัวเลขทั้งหมดของหน้า Evaluation |

- **`gold_check` ใช้ key ชุดเดียวกับ slot evaluation** (`slots.gold_slots` / `pred_slots` / `names_match`) — ช่องที่หน้าเว็บบอกว่า "matches gold" คือช่องที่ report นับว่าถูกจริง มีเทสต์ยืนยัน (`test_gold_check_agrees_with_the_slot_scorer`)
- โหลดโมเดลแบบ lazy + cache ไว้ใน process (มี lock กันโหลดซ้ำ) และ `--preload` ให้โหลดตอน start
- record ที่ได้จาก API **เหมือนกับ `contract-ner extract` ทุกตัวอักษร** เพราะเรียก `predict_with_confidence` + `assemble` ตัวเดียวกัน

### 2.2 หน้าเว็บ (`src/contract_ner/web/`)

- **Extract:** ไฮไลต์ทุก span ในเอกสารตามสีของ label (เส้นทึบ = confidence ≥ 0.8, เส้นประ = ต่ำกว่า, เส้นจุด = ถูก validation ตัดทิ้ง ซ่อนไว้เป็นค่าเริ่มต้น) · ตาราง metadata ด้านขวา คลิกช่องแล้วกระโดดไปที่ span และกระพริบ · hover ที่ไฮไลต์ดู confidence/offset · legend กดซ่อน/แสดงทีละ label · slider confidence floor · paste / upload `.txt` · ดาวน์โหลด JSON
- **Compare with gold:** กรอบประรอบ gold span ในเอกสาร + ป้ายในทุกช่อง: ✓ matches gold / ✗ differs / – missed / ? not in gold พร้อมค่า gold และสำหรับ parties บอกชื่อที่ "tagged เกิน" และ "หาไม่เจอ"
- **Compare models:** ตารางช่อง × โมเดล, แถบเหลืองที่แถวที่โมเดลเห็นไม่ตรงกัน, รวม "slots matching gold" ต่อโมเดล
- **Evaluation:** 4 tile + กราฟ 3 ตัว (slot F1 ต่อโมเดลเทียบเส้นเป้า 0.80, เส้น iteration v1 → final, แท่งรายช่อง) ทุกกราฟมี tooltip และ "Table view" — สีใช้ palette ที่ผ่านการตรวจ colour-blind (ลำดับสีคงที่ต่อ label; ป้ายถูก/ผิดใช้สีสถานะคู่กับไอคอนและคำเสมอ)
- ธีมสว่าง/มืด (ปุ่ม ◐, จำค่าไว้), presenter notes (`N`), คีย์ลัด `1/2/3`, ตรวจที่ความกว้าง 375 px แล้วไม่มี horizontal scroll

### 2.3 สัญญาเดโม (`scripts/pick_demo_docs.py` → `demo/samples/`)

สคริปต์รันโมเดลอังกฤษทั้ง 3 ตัวบน CUAD test split ทั้ง 47 ฉบับ (≈ 90 วินาที) แล้วเลือกสัญญาตามบทบาทในเดโม:

| id | สัญญา | ทำไมเลือก (ตรวจผลจริงแล้ว) |
|---|---|---|
| `en-example` | `examples/english_service_agreement.txt` | สั้น อ่านง่าย ใช้เปิดเดโม |
| `cuad-clean` | cuad-64fad3d4a4 (8,852 ตัวอักษร) | RoBERTa ถูก 5/5; CRF ผิด Parties + Start date, regex ผิด Start date (มีผล 2008-01-01 ≠ เซ็น 2007-12-05) |
| `cuad-parties` | cuad-14d0b0acff (15,052) | RoBERTa 5/5; baseline ทั้งสองเอา "Trust" เป็น Party B แทน "PFS Funds" |
| `cuad-end-date` | cuad-9f4882d859 (159,336) | โมเดล **แท็ก clause ที่ถูกแล้ว** (0.94) แต่ selection หยิบ clause license (0.99) — ตรงกับ open item End_Date |
| `cuad-gold-gap` | cuad-fd51023dce (17,917) | ทุกโมเดลอ่าน governing law = PRC ได้ แต่ CUAD ไม่มี gold |
| `cuad-long` | cuad-0428df0521 (338,211) | ยาวที่สุดใน split ใช้โชว์ windowing (≈ 7.5 วินาที) |
| `th-example` | `examples/thai_service_contract.txt` | ตัวอย่างไทย 9 ช่อง |
| `th-gold` | th-gold-008 จาก **dev gold** (1,355) | โมเดลไทยถูก 5/5 |

ถ้าอยากเปลี่ยนฉบับไหน: `python scripts/pick_demo_docs.py --pick cuad-clean=<doc_id>`

### 2.4 กันเดโมพัง

1. **Fallback อัตโนมัติ:** ถ้าโมเดลโหลดไม่ได้ (ไม่มี weights / ไม่ได้ติดตั้ง Thai extra) API จะตอบจาก record ที่เก็บไว้และติดป้าย `"source": "cache"` (หน้าเว็บขึ้น *stored record* + toast บอกเหตุผล)
2. **`contract-ner serve --cached-only`:** ไม่โหลดโมเดลเลย — ทดสอบแล้วว่า server ขึ้นทันทีและตอบสัญญายาวสุดได้ (`loaded: []`); ข้อความที่ paste เข้ามาหรือ confidence floor อื่นที่ไม่ใช่ 0.6 จะถูกปฏิเสธ (503) แทนที่จะแอบโหลดโมเดล
3. **วิดีโอ + screenshot:** `scripts/record_demo.py` ใช้ Playwright เปิดหน้าเว็บแบบ headless เดินตามสคริปต์ 9 ขั้นพร้อมคำบรรยายในภาพ → `demo/recording/demo_walkthrough.{webm,mp4}` และ `screenshots/01…09.png` (1440×900, ธีมสว่าง) อัดใหม่ได้ทุกเมื่อที่แก้หน้าเว็บ

---

## 3. สิ่งที่ผมตัดสินใจแทนไปก่อน (ขอให้ตรวจ)

| # | เรื่อง | ตัดสินใจ | เหตุผล / ถ้าไม่เห็นด้วย |
|---|---|---|---|
| 1 | เทคโนโลยี front-end | **HTML/CSS/JS ล้วน** ให้ FastAPI serve | ไม่ต้องลง Node, ไม่มี build, รันได้แม้ไม่มีเน็ต, อยู่ใน repo เดียว — ถ้าอยากได้ React ภายหลัง API ไม่ต้องแก้ |
| 2 | Dependencies | extra ใหม่ `[demo]` = fastapi, uvicorn, python-multipart, httpx · **Playwright ไม่ได้ใส่ใน extra** | Playwright + Chromium (~95 MB ใน `~/Library/Caches/ms-playwright`) ใช้แค่ตอนอัดวิดีโอ ผมติดตั้งลง `.venv` แล้ว |
| 3 | ภาษาบนหน้าเว็บ | **อังกฤษ** | ให้ตรงกับ README / ชื่อ field / CUAD; บทพูดภาษาไทยอยู่ใน `docs/demo_script_th.md` |
| 4 | Commit ตัวสัญญาเดโม | commit `demo/samples/` (668 KB) | `data/processed/` ถูก git-ignore — ถ้าไม่ commit เดโมจะพังบนเครื่องอื่น; CUAD เป็น CC BY 4.0 |
| 5 | Confidence floor | โหลด encoder ที่ floor 0 แล้วกรองต่อ request; CRF/regex ไม่มี floor | ทำให้ slider ทำงานได้ และตรงกับวิธีที่ report วัด (CLI ก็ไม่ใส่ floor ให้ CRF) |
| 6 | ช่อง Jurisdiction (อังกฤษ) | หน้าเว็บโชว์ **Governing_Law ก่อน** แล้วค่อย Jurisdiction | ตรงกับ slot evaluation (`pred_slots`) — ⚠️ ต่างจาก `summary["Jurisdiction"]` ของ CLI ซึ่งใช้ `Jurisdiction` (อาจเลือก clause ศาลแทน clause กฎหมาย) ตรวจแล้วว่าในสัญญาเดโมทุกฉบับ ค่าที่หน้าเว็บแสดงตรงกับ summary ของ CLI (ถ้าไม่มี Governing_Law ก็ใช้ Jurisdiction) |
| 7 | สัญญาไทยที่มีเฉลย | ใช้จาก **dev gold** ไม่ใช้ blind | ตามกติกาเดิม: blind set ใช้วัดผลอย่างเดียว ห้ามเอามาดู error |
| 8 | สัญญา End_Date | เลือกกรณี "หาเจอแต่เลือกผิด" แทน "หาไม่เจอ" | เล่าเรื่องได้ชัดกว่าและตรงกับ open item ใน worklog รอบที่แล้ว (รอบแรกสคริปต์เลือก cuad-dccf9671d5 ซึ่งเป็นแบบหาไม่เจอ ผมเปลี่ยนเกณฑ์) |
| 9 | PDF | **ไม่รับ** PDF รับแค่ `.txt` | การอ่าน layout PDF เป็นข้อจำกัดที่ประกาศไว้ โชว์แล้วพังเสียมากกว่าได้ |
| 10 | ความปลอดภัย / deploy | bind `127.0.0.1` เท่านั้น, ไม่มี auth/Docker | เป็นเดโมบนเครื่องตัวเอง; ถ้าจะเปิดให้คนอื่นในห้องเข้า ใช้ `--host 0.0.0.0` ได้แต่ไม่มีการป้องกันใด ๆ |
| 11 | ประวัติ iteration | hard-code ตัวเลข v1–final ใน `api.py` (ยกมาจาก README) | v1–v5 ไม่มี report ที่วัด slot ไว้ จึงอ่านจาก `reports/` ไม่ได้ |
| 12 | วิดีโอ | git-ignore `*.webm` / `*.mp4`, commit แค่ screenshot (1.9 MB) | วิดีโอ ~6 MB ต่อไฟล์ สร้างใหม่ได้จากสคริปต์ |
| 13 | `.claude/launch.json` | สร้างไว้ให้ปุ่ม preview ในแอป Claude แต่ **ไม่ได้ commit** | เป็นค่าตั้งของเครื่อง ไม่ใช่ของโปรเจกต์ |
| 14 | `docs/project_overview_th.md` | **ไม่ได้แตะ** (ยังเป็น untracked เหมือนเดิม) | เป็นไฟล์ของคุณ |

---

## 4. สิ่งที่เจอระหว่างทำ (findings)

1. **ตัวเลข "5 จาก 9 gold gaps" เป็นของโมเดล v7 ไม่ใช่ final** — โมเดล final มี **13 ช่อง** ที่ทายมีค่าแต่ gold ว่าง (`slot_errors` ใน `reports/english-cuad-roberta-label-aware-augmented__test.json`) และยังไม่มีใครตรวจด้วยมือ ใน `docs/project_overview_th.md` สไลด์ 14 กับ README §Limitations ยังอ้าง 5/9 — ถ้าจะพูดในสไลด์ ควรระบุว่าเป็นของ v7 หรือตรวจ 13 ช่องนี้ใหม่ (งานไม่เกินชั่วโมง ส่วนใหญ่เป็น End_Date clause) ผมเลยเลี่ยงตัวเลขนี้ในคำบรรยายวิดีโอ
2. **Boilerplate รั่วเข้าเซต parties:** `_BOILERPLATE_RE` ใน `postprocess.py` จับ *"each a Party and together the Parties"* ได้ แต่ **ไม่จับ** `a "Party" and together the "Parties"` (เจอในสัญญา cuad-long — ทำให้ Parties ของฉบับนั้นนับผิด) แก้ได้ด้วยการเพิ่ม `together the` ใน regex แต่**ผมไม่ได้แก้** เพราะจะเปลี่ยนตัวเลขของทุก report ต้อง re-score ใหม่ทั้งหมด — ควรทำเป็นงานแยก
3. **เศษจาก signature block ถูกนับเป็นคู่สัญญา** (เช่น `CORPORATION ORIENTAL FINANCIAL GROUP` ในสัญญา cuad-end-date) — เป็นเหตุผลหนึ่งที่ช่อง Parties ผิดทั้งที่ Party A/B ถูก หน้าเว็บจึงแสดงบรรทัด "also tagged as a party" และกล่อง "Other names tagged as parties" ให้เห็นตรง ๆ
4. **ความเร็ว** (M2, MPS, โมเดลโหลดแล้ว): โหลด RoBERTa 1.3 วินาที · สัญญา 8–18k ตัวอักษร 0.2–0.4 วินาที · 33k ตัวอักษร 1.1 วินาที · 338k ตัวอักษร 7.5 วินาที · CRF ≤ 1.9 วินาที · regex ~0 — เดโมสดไม่ต้องมี queue หรือ async

---

## 5. การตรวจสอบที่ทำแล้ว

- `pytest`: **142 passed** (เดิม 132 + API 10) · `ruff check src tests scripts`: ผ่าน
- เปิดหน้าเว็บจริงใน browser แล้วไล่ทุกแท็บ: dark + light, desktop + 375 px (ไม่มี horizontal scroll), คลิก cell → กระโดดไปที่ span, tooltip, Compare 3 โมเดล, Evaluation ทุกกราฟ, สัญญาไทย
- `--cached-only`: server ขึ้นโดยไม่โหลดโมเดล และตอบสัญญายาวสุดจาก record ได้ · `HEAD /` ตอบ 200
- ทุกข้อความใน talking point / บทพูดที่อ้างค่าในตาราง ตรวจกับ record จริงแล้ว
- วิดีโอ walkthrough 55 วินาที (มีคำบรรยาย) อัดสำเร็จ และเปิดดู screenshot ทุกภาพแล้ว

---

## 6. สิ่งที่ยังไม่ได้ทำ / แนะนำต่อ

- [ ] ซ้อมเดโมจริงตาม `docs/demo_script_th.md` จับเวลา (ผมไม่ได้ซ้อมพูด)
- [ ] ตรวจ 13 ช่อง gold-gap ของโมเดล final ด้วยมือ ถ้าจะใช้ตัวเลขนี้ในสไลด์ (finding 1)
- [ ] แก้ boilerplate regex + re-score ทุก report เป็นงานแยก (finding 2)
- [ ] ถ้าต้องการวิดีโอที่มีเสียงพากย์: ใช้ `demo_walkthrough.mp4` เป็นฐาน หรือปรับ `--pause` ให้ช้าลงแล้วอัดใหม่
- [ ] Merge `demo-app` เข้า `main` เมื่อพอใจ (ผมไม่ได้ merge และไม่ได้ push)

## 7. Commits ใน branch นี้

ดูด้วย `git log --oneline main..demo-app`:

1. Demo HTTP API, curated demo contracts with stored records
2. Demo web page: extraction with click-to-source, model comparison, evaluation charts
3. Walkthrough recorder, presentation script, work log and README section
4. Cached-only mode never loads a model
