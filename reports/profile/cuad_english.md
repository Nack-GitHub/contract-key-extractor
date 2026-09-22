# Label distribution


### train — 398 documents, 3099 spans (language: en)

Median document length: 33,871 characters. Characters inside a gold span: 1.03% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 1953 | 397 (100%) | 1 / 1 / 16 / 380 | 2 / 6 / 88 | 82% | 0% |
| AGREEMENT_DATE | 363 | 360 (90%) | 38 / 357 / 3 / 0 | 3 / 5 / 18 | 83% | 0% |
| START_DATE | 76 | 71 (18%) | 327 / 66 / 5 / 0 | 10 / 26 / 43 | 28% | 0% |
| END_DATE | 361 | 320 (80%) | 78 / 295 / 19 / 6 | 31 / 66 / 204 | 8% | 0% |
| JURISDICTION | 346 | 330 (83%) | 68 / 317 / 11 / 2 | 29 / 51 / 114 | 0% | 0% |

Surface forms per label:

* **PARTY**
  * `quoted-alias` — 709 (36.3%) · e.g. “"i-Escrow"”
  * `legal-name` — 654 (33.5%) · e.g. “Collectible Concepts Group, Inc.”
  * `legal-name-caps` — 240 (12.3%) · e.g. “I-ESCROW, INC.”
  * `other` — 152 (7.8%) · e.g. “The University of Texas M. D. Anderson Cancer Center”
  * `parties-boilerplate` — 121 (6.2%) · e.g. “(the "Parties" or "Joint Venturers" if referred to collectively, or the "Party" or Joint Venturer" if referred”
  * `caps-no-suffix` — 51 (2.6%) · e.g. “SIEMENS AKTIENGESELLSCHAFT”
  * `person` — 26 (1.3%) · e.g. “Gregory P. Spivy”
* **AGREEMENT_DATE**
  * `month-d-y` — 191 (52.6%) · e.g. “June 21, 1999”
  * `d-month-y` — 91 (25.1%) · e.g. “23rd day of September, 2016”
  * `month-y` — 20 (5.5%) · e.g. “1s t day of June, 2004.”
  * `numeric` — 20 (5.5%) · e.g. “03/01/05”
  * `placeholder-blank` — 18 (5.0%) · e.g. “this ___ day of _________, 2004”
  * `year-only` — 15 (4.1%) · e.g. “[ ] day of [ ], 2020 (”
  * `clause-no-date` — 8 (2.2%) · e.g. “this Agreement as of the Effective Date.”
* **START_DATE**
  * `month-d-y` — 30 (39.5%) · e.g. “December 21, 2017”
  * `clause-no-date` — 30 (39.5%) · e.g. “This Agreement shall become effective as of the first date above written”
  * `d-month-y` — 9 (11.8%) · e.g. “Signed and effective this 25th day of May, 1999.”
  * `placeholder-blank` — 3 (3.9%) · e.g. “________________, 2000”
  * `numeric` — 3 (3.9%) · e.g. “Contract start: 04-01-06”
  * `year-only` — 1 (1.3%) · e.g. “IN WITNESS WHEREOF the parties hereto have hereunto executed this Agreement on the 1st day of JUNE , 2006 , bu”
  * not present in this split: `month-y`
* **END_DATE**
  * `duration` — 160 (44.3%) · e.g. “The term of this Agreement shall continue for one (1) year following the Launch Date, unless earlier terminate”
  * `open-ended` — 96 (26.6%) · e.g. “Term: This Study Order will continue until the Study is completed, which is expected to be ________ (__) month”
  * `explicit-date` — 79 (21.9%) · e.g. “The Joint Venture shall commence on the 1st of March, 2003, and shall be effective until February 28, 2004 unl”
  * `other` — 26 (7.2%) · e.g. “This Agreement is effective as of the date hereof and shall remain in full force and effect for the period (th”
* **JURISDICTION**
  * `governed-by-clause` — 325 (93.9%) · e.g. “This Agreement will be governed and construed in accordance with the laws of the State of California without g”
  * `laws-of-phrase` — 11 (3.2%) · e.g. “This Agreement shall be interpreted under the laws of the State of Nevada.”
  * `forum-court` — 7 (2.0%) · e.g. “Any lawsuit relating to any matter arising under this Agreement may be initiated in a State or Federal Court l”
  * `place-only` — 2 (0.6%) · e.g. “South Dakota”
  * `other` — 1 (0.3%) · e.g. “Effective legislation of the Republic of Kazakhstan shall apply to any relations of the Parties arising out of”

### val — 54 documents, 408 spans (language: en)

Median document length: 30,088 characters. Characters inside a gold span: 1.06% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 244 | 54 (100%) | 0 / 0 / 1 / 53 | 2 / 6 / 21 | 88% | 0% |
| AGREEMENT_DATE | 50 | 49 (91%) | 5 / 48 / 1 / 0 | 3 / 6 / 13 | 86% | 0% |
| START_DATE | 16 | 15 (28%) | 39 / 14 / 1 / 0 | 7.5 / 43 / 43 | 25% | 0% |
| END_DATE | 47 | 44 (81%) | 10 / 41 / 3 / 0 | 31 / 69 / 87 | 9% | 0% |
| JURISDICTION | 51 | 49 (91%) | 5 / 47 / 2 / 0 | 28 / 56 / 63 | 0% | 0% |

Surface forms per label:

* **PARTY**
  * `quoted-alias` — 94 (38.5%) · e.g. “'the Manufacturer'”
  * `legal-name` — 75 (30.7%) · e.g. “Armstrong Flooring, Inc.”
  * `legal-name-caps` — 32 (13.1%) · e.g. “ORGANIC PREPARATIONS INC.”
  * `other` — 19 (7.8%) · e.g. “"Licensing" and together with Seller, "Arizona"”
  * `parties-boilerplate` — 19 (7.8%) · e.g. “(each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Pa”
  * `person` — 4 (1.6%) · e.g. “Celebrity Speakers”
  * `caps-no-suffix` — 1 (0.4%) · e.g. “ANDY NORTH”
* **AGREEMENT_DATE**
  * `month-d-y` — 25 (50.0%) · e.g. “December 31, 2018”
  * `d-month-y` — 10 (20.0%) · e.g. “1st day of August, 2013”
  * `month-y` — 7 (14.0%) · e.g. “15t h day of January 2018”
  * `placeholder-blank` — 3 (6.0%) · e.g. “____ day of ______________ 2013”
  * `year-only` — 2 (4.0%) · e.g. “2018”
  * `clause-no-date` — 2 (4.0%) · e.g. “[Date]”
  * `numeric` — 1 (2.0%) · e.g. “13/07/05”
* **START_DATE**
  * `clause-no-date` — 8 (50.0%) · e.g. “This agreement commences upon execution of this document.”
  * `month-d-y` — 6 (37.5%) · e.g. “April 17, 2017”
  * `d-month-y` — 1 (6.2%) · e.g. “"Commencement Date" means: 1st January 2013.”
  * `numeric` — 1 (6.2%) · e.g. “25/05/2008”
  * not present in this split: `month-y`, `year-only`, `placeholder-blank`
* **END_DATE**
  * `duration` — 22 (46.8%) · e.g. “This agreement is for a term of ten (10) years.”
  * `explicit-date` — 12 (25.5%) · e.g. “The term of this Agreement shall terminate on May 31, 2015, unless sooner terminated.”
  * `open-ended` — 11 (23.4%) · e.g. “Unless earlier terminated pursuant to the provisions hereof, the term of this Agreement and the licenses and o”
  * `other` — 2 (4.3%) · e.g. “The Agreement shall commence on the Effective Date and remain in full force and effect for an initial term of ”
* **JURISDICTION**
  * `governed-by-clause` — 48 (94.1%) · e.g. “This agreement shall be governed by the Laws of England (English common and statutory Law).”
  * `laws-of-phrase` — 3 (5.9%) · e.g. “This Agreement shall be interpreted and enforced in accordance with the laws of the State of California as app”
  * not present in this split: `forum-court`, `place-only`, `other`

### test — 47 documents, 370 spans (language: en)

Median document length: 33,577 characters. Characters inside a gold span: 0.80% of the corpus (this is how imbalanced the token-classification problem is).

| Label | spans | documents with it | spans per document (0 / 1 / 2 / 3+) | words (median / 90th pct / max) | in first 5% of text | repeated mentions |
|---|---:|---:|---|---|---:|---:|
| PARTY | 244 | 47 (100%) | 0 / 0 / 2 / 45 | 2 / 5 / 55 | 80% | 0% |
| AGREEMENT_DATE | 42 | 42 (89%) | 5 / 42 / 0 / 0 | 3 / 5 / 6 | 83% | 0% |
| START_DATE | 10 | 9 (19%) | 38 / 8 / 1 / 0 | 3 / 35 / 35 | 70% | 0% |
| END_DATE | 38 | 32 (68%) | 15 / 26 / 6 / 0 | 29.5 / 65 / 100 | 8% | 0% |
| JURISDICTION | 36 | 34 (72%) | 13 / 32 / 2 / 0 | 29 / 56 / 120 | 3% | 0% |

Surface forms per label:

* **PARTY**
  * `legal-name` — 90 (36.9%) · e.g. “Ability Computer & Software Industries Ltd”
  * `quoted-alias` — 83 (34.0%) · e.g. “"Provider"”
  * `legal-name-caps` — 35 (14.3%) · e.g. “TELCOSTAR PTE, LTD.”
  * `other` — 14 (5.7%) · e.g. “each and both of them "Recipient"”
  * `parties-boilerplate` — 14 (5.7%) · e.g. “Each of the foregoing parties is referred to herein as a "Party" and together as the "Parties".”
  * `caps-no-suffix` — 7 (2.9%) · e.g. “DISTRIBUTOR”
  * `person` — 1 (0.4%) · e.g. “Bruce Jenner”
* **AGREEMENT_DATE**
  * `month-d-y` — 18 (42.9%) · e.g. “October 1, 2019”
  * `d-month-y` — 9 (21.4%) · e.g. “5th day of December, 2007”
  * `numeric` — 5 (11.9%) · e.g. “7/7/98”
  * `month-y` — 4 (9.5%) · e.g. “first day of July, 2017 (”
  * `placeholder-blank` — 3 (7.1%) · e.g. “September ___, 2019”
  * `clause-no-date` — 2 (4.8%) · e.g. “[●]”
  * `year-only` — 1 (2.4%) · e.g. “November 19t h, 2012”
* **START_DATE**
  * `month-d-y` — 4 (40.0%) · e.g. “November 1, 2019”
  * `clause-no-date` — 4 (40.0%) · e.g. “This Agreement shall become effective when it shall have been executed by all parties and upon receipt of all ”
  * `numeric` — 1 (10.0%) · e.g. “7/1/98”
  * `d-month-y` — 1 (10.0%) · e.g. “27 January 2020”
  * not present in this split: `month-y`, `year-only`, `placeholder-blank`
* **END_DATE**
  * `duration` — 20 (52.6%) · e.g. “The term of this Agreement shall begin on the Effective Date and continue in effect for a period of five (5) y”
  * `explicit-date` — 9 (23.7%) · e.g. “This Agreement be deemed effective as of the Effective Date, Agreement and shall terminate on December 31, 202”
  * `open-ended` — 8 (21.1%) · e.g. “This Agreement between HEMISPHERX and SCIEN shall be in effect beginning the last date of execution set forth ”
  * `other` — 1 (2.6%) · e.g. “Party A and Party B agree and confirm that the term of cooperation under this Agreement shall commence from th”
* **JURISDICTION**
  * `governed-by-clause` — 34 (94.4%) · e.g. “This Agreement and any claim, controversy or dispute arising out of or related to this Agreement, any of the t”
  * `laws-of-phrase` — 2 (5.6%) · e.g. “This Agreement is made under and subject to the provision of the substantive laws of the State of New York, wi”
  * not present in this split: `forum-court`, `place-only`, `other`

## Surface-form drift between splits


### val versus train — forms training under-represents

| Label | form | share in train | share in val | spans | verdict |
|---|---|---:|---:|---:|---|
| AGREEMENT_DATE | `month-y` | 5.5% | 14.0% | 7 | under-represented in training |

### test versus train — forms training under-represents

| Label | form | share in train | share in test | spans | verdict |
|---|---|---:|---:|---:|---|
| AGREEMENT_DATE | `numeric` | 5.5% | 11.9% | 5 | under-represented in training |
| AGREEMENT_DATE | `clause-no-date` | 2.2% | 4.8% | 2 | under-represented in training |

## Windowing


### Windowing — train (window 256 subwords, overlap 64)

* windows: 22,673, of which 1,713 (7.6%) contain at least one entity
* gold spans no single window holds in full: 14 (0.45%); spans longer than a whole window: 1

| Label | subwords per span (median / 90th pct / max) |
|---|---|
| AGREEMENT_DATE | 4 / 7 / 27 |
| END_DATE | 39 / 84 / 272 |
| JURISDICTION | 33 / 61 / 149 |
| PARTY | 5 / 14 / 107 |
| START_DATE | 12 / 37 / 49 |

### Windowing — val (window 256 subwords, overlap 64)

* windows: 2,945, of which 241 (8.2%) contain at least one entity
* gold spans no single window holds in full: 3 (0.74%); spans longer than a whole window: 0

| Label | subwords per span (median / 90th pct / max) |
|---|---|
| AGREEMENT_DATE | 4.5 / 8 / 17 |
| END_DATE | 38 / 89 / 122 |
| JURISDICTION | 32 / 61 / 72 |
| PARTY | 5 / 12 / 32 |
| START_DATE | 11.5 / 46 / 61 |

### Windowing — test (window 256 subwords, overlap 64)

* windows: 3,206, of which 187 (5.8%) contain at least one entity
* gold spans no single window holds in full: 1 (0.27%); spans longer than a whole window: 0

| Label | subwords per span (median / 90th pct / max) |
|---|---|
| AGREEMENT_DATE | 5 / 8 / 10 |
| END_DATE | 35.5 / 75 / 141 |
| JURISDICTION | 31 / 76 / 135 |
| PARTY | 5 / 12 / 88 |
| START_DATE | 5 / 40 / 40 |
