# Label distribution


### train — 3000 documents, 26880 spans (language: th)

Median document length: 1,598 characters. Characters inside a gold span: 14.31% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 10066 | 2851 (95%) | 149 / 0 / 908 / 1943 | 3 / 5 / 6 | 12% | 43% |
| AGREEMENT_DATE | 2851 | 2851 (95%) | 149 / 2851 / 0 / 0 | 3 / 4 / 5 | 36% | 0% |
| START_DATE | 2442 | 2442 (81%) | 558 / 2442 / 0 / 0 | 3 / 4 / 5 | 0% | 0% |
| END_DATE | 2137 | 2137 (71%) | 863 / 2137 / 0 / 0 | 3 / 4 / 5 | 0% | 0% |
| CONTRACT_VALUE | 2338 | 2338 (78%) | 662 / 2338 / 0 / 0 | 2 / 3 / 3 | 0% | 0% |
| PAYMENT_DUE | 2090 | 2090 (70%) | 910 / 2090 / 0 / 0 | 3 / 4 / 6 | 0% | 0% |
| PENALTY_RATE | 2383 | 2268 (76%) | 732 / 2153 / 115 / 0 | 3 / 3 / 3 | 0% | 0% |
| JURISDICTION | 2573 | 2254 (75%) | 746 / 1935 / 319 / 0 | 1 / 2 / 3 | 0% | 0% |

Surface forms per label:

* **PARTY**
  * `company` — 6842 (68.0%) · e.g. “บริษัท คลีน เกษตร จำกัด”
  * `person` — 1447 (14.4%) · e.g. “นาย สุดา สุวรรณ”
  * `other` — 974 (9.7%) · e.g. “ว่าที่ร้อยตรีพิมพ์ใจ อินทร์แก้ว”
  * `organization` — 803 (8.0%) · e.g. “มหาวิทยาลัยราชภัฏนครปฐม”
  * not present in this split: `latin-name`
* **AGREEMENT_DATE**
  * `year-without-era/arabic-numerals` — 1553 (54.5%) · e.g. “21 ม.ค. 2566”
  * `buddhist-era/arabic-numerals` — 696 (24.4%) · e.g. “30 มกราคม พ.ศ. 2568”
  * `numeric` — 306 (10.7%) · e.g. “14/11/2565”
  * `christian-era/arabic-numerals` — 136 (4.8%) · e.g. “20 กันยายน ค.ศ. 2022”
  * `year-without-era/thai-numerals` — 91 (3.2%) · e.g. “๒๐ กุมภาพันธ์ ๒๕๖๗”
  * `buddhist-era/thai-numerals` — 57 (2.0%) · e.g. “๒๓ กันยายน พ.ศ. ๒๕๗๐”
  * `christian-era/thai-numerals` — 12 (0.4%) · e.g. “๒ สิงหาคม ค.ศ. ๒๐๒๒”
  * not present in this split: `no-date`
* **START_DATE**
  * `year-without-era/arabic-numerals` — 1334 (54.6%) · e.g. “26 มิถุนายน 2566”
  * `buddhist-era/arabic-numerals` — 611 (25.0%) · e.g. “21 มกราคม พ.ศ. 2566”
  * `numeric` — 263 (10.8%) · e.g. “๒๐/๐๒/๒๕๖๗”
  * `christian-era/arabic-numerals` — 109 (4.5%) · e.g. “1 ตุลาคม ค.ศ. 2021”
  * `year-without-era/thai-numerals` — 85 (3.5%) · e.g. “๘ ตุลาคม ๒๐๒๗”
  * `buddhist-era/thai-numerals` — 31 (1.3%) · e.g. “๔ พฤศจิกายน พ.ศ. ๒๕๖๒”
  * `christian-era/thai-numerals` — 9 (0.4%) · e.g. “๑๑ กุมภาพันธ์ ค.ศ. ๒๐๒๑”
  * not present in this split: `no-date`
* **END_DATE**
  * `year-without-era/arabic-numerals` — 1156 (54.1%) · e.g. “20 เม.ย. 2566”
  * `buddhist-era/arabic-numerals` — 508 (23.8%) · e.g. “31 ธันวาคม พ.ศ. 2565”
  * `numeric` — 253 (11.8%) · e.g. “31-12-2562”
  * `christian-era/arabic-numerals` — 103 (4.8%) · e.g. “31 ธันวาคม ค.ศ. 2023”
  * `year-without-era/thai-numerals` — 77 (3.6%) · e.g. “๗ ตุลาคม ๒๕๗๒”
  * `buddhist-era/thai-numerals` — 34 (1.6%) · e.g. “๑๙ พฤษภาคม พ.ศ. ๒๕๖๗”
  * `christian-era/thai-numerals` — 6 (0.3%) · e.g. “๒๙ กุมภาพันธ์ ค.ศ. ๒๐๒๔”
  * not present in this split: `duration`, `no-date`
* **CONTRACT_VALUE**
  * `baht-digits` — 1734 (74.2%) · e.g. “5,480,000 บาท”
  * `per-period` — 473 (20.2%) · e.g. “407,000.00 บาท ต่อเดือน”
  * `foreign-currency` — 131 (5.6%) · e.g. “USD 304,390,000”
  * not present in this split: `amount-in-words`
* **PAYMENT_DUE**
  * `within-n-days` — 1074 (51.4%) · e.g. “ภายใน 7 วันนับจากวันสิ้นเดือน”
  * `recurring-day` — 568 (27.2%) · e.g. “ทุกสิ้นเดือน”
  * `other` — 448 (21.4%) · e.g. “ทุก 1 เดือน”
* **PENALTY_RATE**
  * `percent-word` — 934 (39.2%) · e.g. “ร้อยละ 10 ต่อปี”
  * `percent-sign` — 809 (33.9%) · e.g. “๑๘% ของมูลค่าสัญญาต่อปี”
  * `fixed-amount` — 640 (26.9%) · e.g. “10,000 บาท ต่อการฝ่าฝืนแต่ละครั้ง”
* **JURISDICTION**
  * `court` — 1480 (57.5%) · e.g. “ศาลจังหวัดสงขลา”
  * `governing-law` — 709 (27.6%) · e.g. “กฎหมายแห่งราชอาณาจักรไทย”
  * `arbitration` — 384 (14.9%) · e.g. “สำนักงานอนุญาโตตุลาการ สภาหอการค้าแห่งประเทศไทย”

### val — 300 documents, 2547 spans (language: th)

Median document length: 1,469 characters. Characters inside a gold span: 14.37% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 951 | 281 (94%) | 19 / 0 / 112 / 169 | 3 / 5 / 6 | 13% | 41% |
| AGREEMENT_DATE | 281 | 281 (94%) | 19 / 281 / 0 / 0 | 3 / 4 / 5 | 30% | 0% |
| START_DATE | 229 | 229 (76%) | 71 / 229 / 0 / 0 | 3 / 4 / 5 | 0% | 0% |
| END_DATE | 208 | 208 (69%) | 92 / 208 / 0 / 0 | 3 / 4 / 5 | 0% | 0% |
| CONTRACT_VALUE | 221 | 221 (74%) | 79 / 221 / 0 / 0 | 2 / 3 / 3 | 0% | 0% |
| PAYMENT_DUE | 197 | 197 (66%) | 103 / 197 / 0 / 0 | 3 / 4 / 6 | 1% | 0% |
| PENALTY_RATE | 224 | 215 (72%) | 85 / 206 / 9 / 0 | 3 / 3 / 3 | 0% | 0% |
| JURISDICTION | 236 | 215 (72%) | 85 / 194 / 21 / 0 | 1 / 2 / 3 | 0% | 0% |

Surface forms per label:

* **PARTY**
  * `company` — 612 (64.4%) · e.g. “บริษัท ฟาร์มา จำกัด (มหาชน)”
  * `person` — 143 (15.0%) · e.g. “ดร. ชัยวัฒน์ รักไทย”
  * `other` — 110 (11.6%) · e.g. “ว่าที่ร้อยตรีอรทัย ใจดี”
  * `organization` — 86 (9.0%) · e.g. “องค์การบริหารส่วนจังหวัดภูเก็ต”
  * not present in this split: `latin-name`
* **AGREEMENT_DATE**
  * `year-without-era/arabic-numerals` — 159 (56.6%) · e.g. “15 มิถุนายน 2569”
  * `buddhist-era/arabic-numerals` — 61 (21.7%) · e.g. “1 เมษายน พ.ศ. 2568”
  * `numeric` — 31 (11.0%) · e.g. “7-10-2562”
  * `christian-era/arabic-numerals` — 14 (5.0%) · e.g. “1 ธันวาคม ค.ศ. 2020”
  * `year-without-era/thai-numerals` — 9 (3.2%) · e.g. “๑๔ กุมภาพันธ์ ๒๕๖๙”
  * `buddhist-era/thai-numerals` — 7 (2.5%) · e.g. “๒๕ เดือน มกราคม พ.ศ. ๒๕๖๖”
  * not present in this split: `no-date`
* **START_DATE**
  * `year-without-era/arabic-numerals` — 126 (55.0%) · e.g. “15 กรกฎาคม 2569”
  * `buddhist-era/arabic-numerals` — 62 (27.1%) · e.g. “6 ธันวาคม พ.ศ. 2562”
  * `numeric` — 17 (7.4%) · e.g. “13-08-2566”
  * `christian-era/arabic-numerals` — 12 (5.2%) · e.g. “10 กุมภาพันธ์ ค.ศ. 2019”
  * `buddhist-era/thai-numerals` — 7 (3.1%) · e.g. “๒๔ กุมภาพันธ์ พ.ศ. ๒๕๖๖”
  * `year-without-era/thai-numerals` — 4 (1.7%) · e.g. “๑ มีนาคม ๒๕๖๙”
  * `christian-era/thai-numerals` — 1 (0.4%) · e.g. “๑ มีนาคม ค.ศ. ๒๐๒๔”
  * not present in this split: `no-date`
* **END_DATE**
  * `year-without-era/arabic-numerals` — 121 (58.2%) · e.g. “8 เมษายน 2564”
  * `buddhist-era/arabic-numerals` — 48 (23.1%) · e.g. “31 พฤษภาคม พ.ศ. 2564”
  * `numeric` — 24 (11.5%) · e.g. “04/02/2569”
  * `year-without-era/thai-numerals` — 6 (2.9%) · e.g. “๑๗ April ๒๐๒๘”
  * `christian-era/arabic-numerals` — 5 (2.4%) · e.g. “22 พฤษภาคม ค.ศ. 2029”
  * `buddhist-era/thai-numerals` — 4 (1.9%) · e.g. “๒๘ กุมภาพันธ์ พ.ศ. ๒๕๗๔”
  * not present in this split: `duration`, `no-date`
* **CONTRACT_VALUE**
  * `baht-digits` — 160 (72.4%) · e.g. “5,411,000 บาท”
  * `per-period` — 51 (23.1%) · e.g. “ในอัตราเดือนละ 70,000 บาท”
  * `foreign-currency` — 10 (4.5%) · e.g. “22,663,000 เยน”
  * not present in this split: `amount-in-words`
* **PAYMENT_DUE**
  * `within-n-days` — 90 (45.7%) · e.g. “ภายใน 45 วันหลังจากวันได้รับใบแจ้งหนี้ถูกต้องครบถ้วน”
  * `recurring-day` — 57 (28.9%) · e.g. “ทุกเดือน เดือนละ 30,000 บาท”
  * `other` — 50 (25.4%) · e.g. “ในวันที่ 6 พฤศจิกายน 2569”
* **PENALTY_RATE**
  * `percent-sign` — 82 (36.6%) · e.g. “0.5% ต่อปี”
  * `percent-word` — 77 (34.4%) · e.g. “ร้อยละ 10 ของมูลค่าสัญญา”
  * `fixed-amount` — 65 (29.0%) · e.g. “คนละ 2,000 บาทต่อชั่วโมง”
* **JURISDICTION**
  * `court` — 130 (55.1%) · e.g. “ศาลจังหวัดขอนแก่น”
  * `governing-law` — 70 (29.7%) · e.g. “กฎหมายไทย”
  * `arbitration` — 36 (15.3%) · e.g. “สถาบันอนุญาโตตุลาการ สำนักงานศาลยุติธรรม”

### test_gold — 24 documents, 217 spans (language: th)

Median document length: 1,005 characters. Characters inside a gold span: 18.35% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 57 | 24 (100%) | 0 / 0 / 18 / 6 | 3 / 5 / 6 | 14% | 14% |
| AGREEMENT_DATE | 23 | 23 (96%) | 1 / 23 / 0 / 0 | 3 / 4 / 4 | 13% | 0% |
| START_DATE | 19 | 19 (79%) | 5 / 19 / 0 / 0 | 3 / 3 / 3 | 0% | 0% |
| END_DATE | 22 | 22 (92%) | 2 / 22 / 0 / 0 | 3 / 3 / 3 | 0% | 0% |
| CONTRACT_VALUE | 23 | 23 (96%) | 1 / 23 / 0 / 0 | 2 / 3 / 3 | 0% | 0% |
| PAYMENT_DUE | 24 | 23 (96%) | 1 / 22 / 1 / 0 | 3 / 4 / 4 | 0% | 0% |
| PENALTY_RATE | 24 | 23 (96%) | 1 / 22 / 1 / 0 | 3 / 3 / 3 | 0% | 0% |
| JURISDICTION | 25 | 20 (83%) | 4 / 15 / 5 / 0 | 1 / 1 / 2 | 0% | 0% |

Surface forms per label:

* **PARTY**
  * `company` — 35 (61.4%) · e.g. “บริษัท เอบีซี จำกัด”
  * `person` — 12 (21.1%) · e.g. “นางสาวพรทิพย์ ศรีวิไล”
  * `organization` — 7 (12.3%) · e.g. “เทศบาลตำบลแม่ริม”
  * `latin-name` — 2 (3.5%) · e.g. “Siam Digital Ventures Co., Ltd.”
  * `other` — 1 (1.8%) · e.g. “กิจการร่วมค้า เอสทีเอ็น-ไทยวัฒน์”
* **AGREEMENT_DATE**
  * `year-without-era/arabic-numerals` — 19 (82.6%) · e.g. “9 สิงหาคม 2567”
  * `buddhist-era/arabic-numerals` — 2 (8.7%) · e.g. “25 มีนาคม พ.ศ. 2568”
  * `year-without-era/thai-numerals` — 1 (4.3%) · e.g. “๑๒ กุมภาพันธ์ ๒๕๖๗”
  * `numeric` — 1 (4.3%) · e.g. “01/04/2569”
  * not present in this split: `buddhist-era/thai-numerals`, `christian-era/arabic-numerals`, `no-date`
* **START_DATE**
  * `year-without-era/arabic-numerals` — 17 (89.5%) · e.g. “1 มกราคม 2569”
  * `year-without-era/thai-numerals` — 1 (5.3%) · e.g. “๑๓ กุมภาพันธ์ ๒๕๖๗”
  * `numeric` — 1 (5.3%) · e.g. “15/04/2569”
  * not present in this split: `buddhist-era/arabic-numerals`, `buddhist-era/thai-numerals`, `christian-era/arabic-numerals`, `no-date`
* **END_DATE**
  * `year-without-era/arabic-numerals` — 20 (90.9%) · e.g. “31 ธันวาคม 2569”
  * `year-without-era/thai-numerals` — 1 (4.5%) · e.g. “๑๒ พฤษภาคม ๒๕๖๗”
  * `numeric` — 1 (4.5%) · e.g. “14/10/2569”
  * not present in this split: `buddhist-era/arabic-numerals`, `buddhist-era/thai-numerals`, `christian-era/arabic-numerals`, `duration`, `no-date`
* **CONTRACT_VALUE**
  * `baht-digits` — 15 (65.2%) · e.g. “2,500,000 บาท”
  * `per-period` — 6 (26.1%) · e.g. “เดือนละ 45,000 บาท”
  * `foreign-currency` — 1 (4.3%) · e.g. “USD 48,000”
  * `amount-in-words` — 1 (4.3%) · e.g. “4.5 ล้านบาท”
* **PAYMENT_DUE**
  * `within-n-days` — 18 (75.0%) · e.g. “ภายใน ๓๐ วันนับถัดจากวันที่ตรวจรับงานถูกต้องครบถ้วน”
  * `recurring-day` — 6 (25.0%) · e.g. “ทุกวันที่ 5 ของเดือน”
  * not present in this split: `other`
* **PENALTY_RATE**
  * `percent-word` — 12 (50.0%) · e.g. “ร้อยละ ๐.๒๕ ของค่าจ้างตามสัญญาต่อวัน”
  * `fixed-amount` — 8 (33.3%) · e.g. “วันละ 500 บาท”
  * `percent-sign` — 4 (16.7%) · e.g. “15% ต่อเดือน”
* **JURISDICTION**
  * `court` — 18 (72.0%) · e.g. “ศาลแพ่งกรุงเทพฯ”
  * `governing-law` — 5 (20.0%) · e.g. “กฎหมายไทย”
  * `arbitration` — 2 (8.0%) · e.g. “สถาบันอนุญาโตตุลาการ สำนักงานศาลยุติธรรม”

### test_gold_blind — 8 documents, 73 spans (language: th)

Median document length: 1,183 characters. Characters inside a gold span: 18.08% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 19 | 8 (100%) | 0 / 0 / 6 / 2 | 3 / 5 / 5 | 11% | 16% |
| AGREEMENT_DATE | 8 | 8 (100%) | 0 / 8 / 0 / 0 | 3 / 3 / 3 | 25% | 0% |
| START_DATE | 7 | 7 (88%) | 1 / 7 / 0 / 0 | 3 / 3 / 3 | 0% | 0% |
| END_DATE | 7 | 7 (88%) | 1 / 7 / 0 / 0 | 3 / 3 / 3 | 0% | 0% |
| CONTRACT_VALUE | 8 | 8 (100%) | 0 / 8 / 0 / 0 | 2 / 3 / 3 | 0% | 0% |
| PAYMENT_DUE | 8 | 8 (100%) | 0 / 8 / 0 / 0 | 3 / 4 / 4 | 0% | 0% |
| PENALTY_RATE | 8 | 8 (100%) | 0 / 8 / 0 / 0 | 3 / 5 / 5 | 0% | 0% |
| JURISDICTION | 8 | 7 (88%) | 1 / 6 / 1 / 0 | 1 / 2 / 2 | 0% | 0% |

Surface forms per label:

* **PARTY**
  * `company` — 9 (47.4%) · e.g. “บริษัท เน็ตเวิร์ค โปร โซลูชั่น จำกัด”
  * `person` — 5 (26.3%) · e.g. “นายกิตติพงษ์ อรุณรัตน์”
  * `organization` — 3 (15.8%) · e.g. “สมาคมกีฬาฟุตบอลจังหวัดขอนแก่น”
  * `latin-name` — 2 (10.5%) · e.g. “Sunrise Holidays Pte. Ltd.”
  * not present in this split: `other`
* **AGREEMENT_DATE**
  * `year-without-era/arabic-numerals` — 6 (75.0%) · e.g. “20 ธันวาคม 2567”
  * `year-without-era/thai-numerals` — 2 (25.0%) · e.g. “๕ มีนาคม ๒๕๖๘”
  * not present in this split: `buddhist-era/arabic-numerals`, `buddhist-era/thai-numerals`, `christian-era/arabic-numerals`, `numeric`, `no-date`
* **START_DATE**
  * `year-without-era/arabic-numerals` — 5 (71.4%) · e.g. “1 มกราคม 2569”
  * `year-without-era/thai-numerals` — 2 (28.6%) · e.g. “๑๗ มีนาคม ๒๕๖๘”
  * not present in this split: `buddhist-era/arabic-numerals`, `buddhist-era/thai-numerals`, `christian-era/arabic-numerals`, `numeric`, `no-date`
* **END_DATE**
  * `year-without-era/arabic-numerals` — 5 (71.4%) · e.g. “31 ธันวาคม 2569”
  * `year-without-era/thai-numerals` — 2 (28.6%) · e.g. “๓๐ พฤศจิกายน ๒๕๖๘”
  * not present in this split: `buddhist-era/arabic-numerals`, `buddhist-era/thai-numerals`, `christian-era/arabic-numerals`, `numeric`, `duration`, `no-date`
* **CONTRACT_VALUE**
  * `baht-digits` — 4 (50.0%) · e.g. “๑๒,๗๕๐,๐๐๐ บาท”
  * `per-period` — 3 (37.5%) · e.g. “ปีละ 1,260,000 บาท”
  * `foreign-currency` — 1 (12.5%) · e.g. “USD 96,000”
  * not present in this split: `amount-in-words`
* **PAYMENT_DUE**
  * `within-n-days` — 8 (100.0%) · e.g. “ภายใน 30 วันนับจากวันที่ได้รับใบแจ้งหนี้”
  * not present in this split: `recurring-day`, `other`
* **PENALTY_RATE**
  * `fixed-amount` — 4 (50.0%) · e.g. “ชั่วโมงละ 2,000 บาท”
  * `percent-word` — 4 (50.0%) · e.g. “ร้อยละ 10 ของราคาซื้อขาย”
  * not present in this split: `percent-sign`
* **JURISDICTION**
  * `court` — 6 (75.0%) · e.g. “ศาลจังหวัดพัทยา”
  * `governing-law` — 1 (12.5%) · e.g. “กฎหมายไทย”
  * `arbitration` — 1 (12.5%) · e.g. “สถาบันอนุญาโตตุลาการ (THAC)”

### test_synthetic — 300 documents, 2662 spans (language: th)

Median document length: 1,611 characters. Characters inside a gold span: 14.16% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 1017 | 289 (96%) | 11 / 0 / 94 / 195 | 3 / 5 / 6 | 13% | 43% |
| AGREEMENT_DATE | 289 | 289 (96%) | 11 / 289 / 0 / 0 | 3 / 4 / 5 | 34% | 0% |
| START_DATE | 247 | 247 (82%) | 53 / 247 / 0 / 0 | 3 / 4 / 5 | 0% | 0% |
| END_DATE | 206 | 206 (69%) | 94 / 206 / 0 / 0 | 3 / 4 / 5 | 0% | 0% |
| CONTRACT_VALUE | 238 | 238 (79%) | 62 / 238 / 0 / 0 | 2 / 3 / 3 | 0% | 0% |
| PAYMENT_DUE | 213 | 213 (71%) | 87 / 213 / 0 / 0 | 3 / 4 / 6 | 0% | 0% |
| PENALTY_RATE | 217 | 210 (70%) | 90 / 203 / 7 / 0 | 3 / 3 / 3 | 0% | 0% |
| JURISDICTION | 235 | 210 (70%) | 90 / 185 / 25 / 0 | 1 / 2 / 3 | 0% | 0% |

Surface forms per label:

* **PARTY**
  * `company` — 684 (67.3%) · e.g. “บริษัท กรุ๊ป จำกัด”
  * `person` — 150 (14.7%) · e.g. “นางสาว ศิริพร สุขสันต์”
  * `other` — 117 (11.5%) · e.g. “ว่าที่ร้อยตรี ณัฐพล พงษ์พันธ์”
  * `organization` — 66 (6.5%) · e.g. “เทศบาลนครนครราชสีมา”
  * not present in this split: `latin-name`
* **AGREEMENT_DATE**
  * `year-without-era/arabic-numerals` — 146 (50.5%) · e.g. “1 มิถุนายน 2019”
  * `buddhist-era/arabic-numerals` — 75 (26.0%) · e.g. “17 พฤศจิกายน พ.ศ. 2567”
  * `numeric` — 34 (11.8%) · e.g. “29/01/2562”
  * `christian-era/arabic-numerals` — 18 (6.2%) · e.g. “7 มีนาคม ค.ศ. 2022”
  * `year-without-era/thai-numerals` — 8 (2.8%) · e.g. “๑ มิ.ย. ๒๕๗๐”
  * `buddhist-era/thai-numerals` — 7 (2.4%) · e.g. “๑๓ มิถุนายน พ.ศ. ๒๕๖๘”
  * `christian-era/thai-numerals` — 1 (0.3%) · e.g. “๑ กันยายน ค.ศ. ๒๐๒๓”
  * not present in this split: `no-date`
* **START_DATE**
  * `year-without-era/arabic-numerals` — 139 (56.3%) · e.g. “1 มิ.ย. 2562”
  * `buddhist-era/arabic-numerals` — 58 (23.5%) · e.g. “1 สิงหาคม พ.ศ. 2563”
  * `numeric` — 22 (8.9%) · e.g. “12-01-2570”
  * `christian-era/arabic-numerals` — 15 (6.1%) · e.g. “1 พฤศจิกายน ค.ศ. 2025”
  * `year-without-era/thai-numerals` — 10 (4.0%) · e.g. “๑ มิถุนายน ๒๐๒๗”
  * `buddhist-era/thai-numerals` — 3 (1.2%) · e.g. “๑๓ เดือน มิถุนายน พ.ศ. ๒๕๖๘”
  * not present in this split: `no-date`
* **END_DATE**
  * `year-without-era/arabic-numerals` — 118 (57.3%) · e.g. “31 พฤษภาคม 2563”
  * `buddhist-era/arabic-numerals` — 49 (23.8%) · e.g. “16 เดือน มิถุนายน พ.ศ. 2563”
  * `numeric` — 19 (9.2%) · e.g. “28/02/2568”
  * `christian-era/arabic-numerals` — 9 (4.4%) · e.g. “6 พฤศจิกายน ค.ศ. 2021”
  * `year-without-era/thai-numerals` — 8 (3.9%) · e.g. “๑๒ มิถุนายน ๒๕๗๒”
  * `buddhist-era/thai-numerals` — 2 (1.0%) · e.g. “๓๑ พฤษภาคม พ.ศ. ๒๕๗๒”
  * `christian-era/thai-numerals` — 1 (0.5%) · e.g. “๓๐ กันยายน ค.ศ. ๒๐๒๗”
  * not present in this split: `duration`, `no-date`
* **CONTRACT_VALUE**
  * `baht-digits` — 174 (73.1%) · e.g. “6,525,000.00 บาท”
  * `per-period` — 51 (21.4%) · e.g. “352,000 เหรียญสหรัฐ ต่อเดือน”
  * `foreign-currency` — 13 (5.5%) · e.g. “20,385,000 ยูโร”
  * not present in this split: `amount-in-words`
* **PAYMENT_DUE**
  * `within-n-days` — 113 (53.1%) · e.g. “ภายในวันที่ 28 ของแต่ละเดือน”
  * `recurring-day` — 54 (25.4%) · e.g. “ล่วงหน้าทุกต้นเดือน”
  * `other` — 46 (21.6%) · e.g. “ทุกวันศุกร์สุดท้ายของเดือน”
* **PENALTY_RATE**
  * `percent-sign` — 80 (36.9%) · e.g. “วันละ 1%”
  * `percent-word` — 76 (35.0%) · e.g. “ร้อยละ 10 ต่อวัน”
  * `fixed-amount` — 61 (28.1%) · e.g. “1,000 บาท ต่อวัน”
* **JURISDICTION**
  * `court` — 139 (59.1%) · e.g. “ศาลจังหวัดนครศรีธรรมราช”
  * `governing-law` — 54 (23.0%) · e.g. “กฎหมายของประเทศไทย”
  * `arbitration` — 42 (17.9%) · e.g. “สำนักงานอนุญาโตตุลาการ สภาหอการค้าแห่งประเทศไทย”

## Surface-form drift between splits


### val versus train — forms training under-represents

| Label | form | share in train | share in val | spans | verdict |
|---|---|---:|---:|---:|---|
| START_DATE | `buddhist-era/thai-numerals` | 1.3% | 3.1% | 7 | under-represented in training |

### test_gold versus train — forms training under-represents

| Label | form | share in train | share in test_gold | spans | verdict |
|---|---|---:|---:|---:|---|
| PARTY | `latin-name` | 0.0% | 3.5% | 2 | unseen in training |

### test_gold_blind versus train — forms training under-represents

| Label | form | share in train | share in test_gold_blind | spans | verdict |
|---|---|---:|---:|---:|---|
| AGREEMENT_DATE | `year-without-era/thai-numerals` | 3.2% | 25.0% | 2 | under-represented in training |
| END_DATE | `year-without-era/thai-numerals` | 3.6% | 28.6% | 2 | under-represented in training |
| PARTY | `latin-name` | 0.0% | 10.5% | 2 | unseen in training |
| START_DATE | `year-without-era/thai-numerals` | 3.5% | 28.6% | 2 | under-represented in training |

### test_synthetic versus train

No surface form is materially under-represented in training.
