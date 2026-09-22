import importlib.util

import pytest

from contract_ner.postprocess import (
    assemble,
    group_parties,
    normalize_date,
    normalize_jurisdiction,
    normalize_money,
    normalize_payment_due,
    normalize_rate,
    pick_parties,
    validate_span,
)
from contract_ner.schema import Span

# Thai cases need the ``thai`` extra (``pip install -e ".[thai]"``); they skip, not fail, without it.
thai = pytest.mark.skipif(importlib.util.find_spec("pythainlp") is None, reason="pythainlp not installed")


@pytest.mark.parametrize("text,iso", [
    ("1 มกราคม 2569", "2026-01-01"),
    ("15 ธันวาคม พ.ศ. 2568", "2025-12-15"),
    ("๒๙ ก.ย. ๒๕๖๘", "2025-09-29"),
    ("01/04/2569", "2026-04-01"),
    ("2-12-2562", "2019-12-02"),
    ("1 เดือน มกราคม พ.ศ. 2569", "2026-01-01"),
    ("12 มีนาคม ค.ศ. 2026", "2026-03-12"),
    ("15 May 2025", "2025-05-15"),
    ("June 8, 2010", "2010-06-08"),
    ("8th day of May 2014", "2014-05-08"),
    ("December 31st, 2019", "2019-12-31"),
    ("มกราคม 2569", "2026-01"),
    ("วันที่ 22 ธันวาคม 2568", "2025-12-22"),
    ("ไม่ใช่วันที่", None),
    ("31 กุมภาพันธ์ 2569", None),
])
def test_normalize_date(text, iso):
    assert normalize_date(text) == iso


@pytest.mark.parametrize("text,amount,cur,per", [
    ("2,500,000 บาท", 2_500_000, "THB", None),
    ("๔๕,๐๑๒,๐๐๐.- บาท", 45_012_000, "THB", None),
    ("2.5 ล้านบาท", 2_500_000, "THB", None),
    ("USD 48,000", 48_000, "USD", None),
    ("เดือนละ 45,000 บาท", 45_000, "THB", "month"),
    ("38,000 บาท ต่อเดือน", 38_000, "THB", "month"),
    ("ปีละ 1,850,000 บาท", 1_850_000, "THB", "year"),
    ("1,890,000.00 บาท", 1_890_000, "THB", None),
])
def test_normalize_money(text, amount, cur, per):
    m = normalize_money(text)
    assert m["amount"] == amount and m["currency"] == cur and m["per"] == per


def test_normalize_money_without_number():
    assert normalize_money("บาท") is None


@pytest.mark.parametrize("text,display", [
    ("15% ต่อเดือน", "15% per month"),
    ("ร้อยละ ๐.๒๕ ของค่าจ้างตามสัญญาต่อวัน", "0.25% per day of contract value"),
    ("วันละ 500 บาท", "500 THB per day"),
    ("500,000 บาท", "500,000 THB (fixed)"),
    ("1.25% per month", "1.25% per month"),
    ("2,000,000 บาท ต่อการฝ่าฝืนแต่ละครั้ง", "2,000,000 THB per occurrence"),
    ("คนละ 500 บาทต่อวัน", "500 THB per person per day"),
    ("ร้อยละ 3 ต่อปี", "3% per year"),
])
def test_normalize_rate(text, display):
    assert normalize_rate(text)["display"] == display


@pytest.mark.parametrize("text,expected", [
    ("ทุกวันที่ 5 ของเดือน", "day 5 of each month"),
    ("ภายใน ๓๐ วันนับถัดจากวันที่ตรวจรับงานถูกต้องครบถ้วน", "within 30 days of acceptance of the work/goods"),
    ("ภายในวันที่ 10 ของเดือนถัดไป", "day 10 of the following month"),
    ("ทุกสิ้นเดือน", "end of each month"),
    ("ภายในวันที่ 30 มิถุนายน 2569", "by 2026-06-30"),
    ("ภายใน 15 วันทำการนับจากวันที่ได้รับใบแจ้งหนี้", "within 15 business days of receipt of invoice"),
    ("within 30 days of invoice", "within 30 days of receipt of invoice"),
])
def test_normalize_payment_due(text, expected):
    assert normalize_payment_due(text) == expected


@thai
def test_normalize_jurisdiction():
    assert normalize_jurisdiction("ศาลแพ่งกรุงเทพฯ") == {"kind": "court", "english": "Bangkok Civil Court"}
    assert normalize_jurisdiction("ศาลจังหวัดเชียงใหม่")["english"] == "Chiang Mai Provincial Court"
    assert normalize_jurisdiction("สถาบันอนุญาโตตุลาการ (THAC)")["kind"] == "arbitration"
    assert normalize_jurisdiction("กฎหมายไทย") == {"kind": "governing_law", "english": "Thai law"}
    j = normalize_jurisdiction("This Agreement shall be governed by the laws of the State of New York, without regard to")
    assert j == {"kind": "governing_law", "english": "State of New York"}
    assert normalize_jurisdiction("governed by the laws of England and Wales.")["english"] == "England and Wales"
    assert normalize_jurisdiction("the laws of the Province of Ontario and the federal laws of Canada")["english"] == "Province of Ontario"
    assert normalize_jurisdiction("exclusive jurisdiction of the courts located in New York County, New York")["english"] == "courts of New York County, New York"


@thai
def test_validate_span_rejects_impossible_surface_forms():
    assert validate_span(Span(0, 5, "START_DATE", "1 มกราคม 2569"))
    assert not validate_span(Span(0, 5, "START_DATE", "สัญญานี้"))
    assert not validate_span(Span(0, 5, "CONTRACT_VALUE", "บาท"))
    assert validate_span(Span(0, 5, "JURISDICTION", "ศาลแพ่ง"))
    assert not validate_span(Span(0, 5, "JURISDICTION", "กรุงเทพมหานคร"))
    assert not validate_span(Span(0, 5, "PARTY", "1"))
    for frag in ("*", ",", "and", "THE", ".", "<"):
        assert not validate_span(Span(0, 5, "PARTY", frag)), frag
    assert not validate_span(Span(0, 5, "AGREEMENT_DATE", "18/18"))
    assert not validate_span(Span(0, 5, "END_DATE", "December"))
    assert validate_span(Span(0, 5, "END_DATE", "This Agreement shall continue in force for a period of twenty (20) years"))
    assert validate_span(Span(0, 5, "END_DATE", "The term shall commence on the Effective Date and continue until terminated"))
    assert validate_span(Span(0, 5, "PARTY", "GA"))  # short but a real name is possible


def _sp(text, start, label, conf=0.9):
    return Span(start, start + len(text), label, text), conf


@thai
def test_assemble_thai_deck_example():
    text = ("สัญญาฉบับนี้ทำขึ้นระหว่าง บริษัท เอบีซี จำกัด (“ผู้ว่าจ้าง”) และ บริษัท เอ็กซ์วายแซด จำกัด (“ผู้รับจ้าง”) "
            "มีผลตั้งแต่วันที่ 1 มกราคม 2569 ถึงวันที่ 31 ธันวาคม 2569 มูลค่าสัญญารวม 2,500,000 บาท กำหนดชำระทุกวันที่ 5 ของเดือน "
            "หากผิดนัดชำระคิดค่าปรับ 15% ต่อเดือน ข้อพิพาทอยู่ภายใต้เขตอำนาจศาลแพ่งกรุงเทพฯ")
    spans = [
        _sp("บริษัท เอบีซี จำกัด", text.index("บริษัท เอบีซี"), "PARTY"),
        _sp("บริษัท เอ็กซ์วายแซด จำกัด", text.index("บริษัท เอ็กซ์"), "PARTY"),
        _sp("1 มกราคม 2569", text.index("1 มกราคม"), "START_DATE"),
        _sp("31 ธันวาคม 2569", text.index("31 ธันวาคม"), "END_DATE"),
        _sp("2,500,000 บาท", text.index("2,500,000"), "CONTRACT_VALUE"),
        _sp("ทุกวันที่ 5 ของเดือน", text.index("ทุกวันที่"), "PAYMENT_DUE"),
        _sp("15% ต่อเดือน", text.index("15%"), "PENALTY_RATE"),
        _sp("ศาลแพ่งกรุงเทพฯ", text.index("ศาลแพ่ง"), "JURISDICTION"),
    ]
    rec = assemble(text, spans, lang="th")
    assert rec["summary"] == {
        "Party_A": "บริษัท เอบีซี จำกัด",
        "Party_B": "บริษัท เอ็กซ์วายแซด จำกัด",
        "Agreement_Date": None,
        "Start_Date": "2026-01-01",
        "End_Date": "2026-12-31",
        "Contract_Value": "2,500,000 THB",
        "Payment_Due_Date": "day 5 of each month",
        "Penalty_Rate": "15% per month",
        "Jurisdiction": "Bangkok Civil Court",
    }
    assert rec["Party_A"]["start"] == text.index("บริษัท เอบีซี")
    assert all(e["valid"] for e in rec["entities"])


def test_assemble_demotes_aliases_and_falls_back_to_agreement_date():
    text = 'between Acme Holdings Inc. ("Acme") and Beta Corp ("Beta"), dated June 8, 2010. Governed by the laws of Texas.'
    spans = [
        _sp("Acme Holdings Inc.", text.index("Acme Holdings"), "PARTY"),
        _sp('"Acme"', text.index('"Acme"'), "PARTY"),
        _sp("Beta Corp", text.index("Beta Corp"), "PARTY"),
        _sp('"Beta"', text.index('"Beta"'), "PARTY"),
        _sp("June 8, 2010", text.index("June"), "AGREEMENT_DATE"),
        _sp("Governed by the laws of Texas", text.index("Governed"), "JURISDICTION"),
        _sp("no digits here", 0, "END_DATE", 0.4),  # must be flagged invalid
    ]
    rec = assemble(text, spans, lang="en")
    assert rec["Party_A"]["text"] == "Acme Holdings Inc."
    assert rec["Party_B"]["text"] == "Beta Corp"
    assert rec["Other_Parties"] == ["Acme", "Beta"]
    assert rec["Agreement_Date"]["normalized"] == "2010-06-08"
    assert rec["Start_Date"]["inferred_from"] == "Agreement_Date"
    assert rec["End_Date"] is None
    assert rec["Governing_Law"]["normalized"]["kind"] == "governing_law"
    invalid = [e for e in rec["entities"] if not e["valid"]]
    assert len(invalid) == 1 and invalid[0]["label"] == "END_DATE"


def test_party_grouping_merges_repeated_mentions():
    groups = group_parties([_sp("บริษัท ก จำกัด", 0, "PARTY"), _sp("บริษัท  ก จำกัด", 100, "PARTY", 0.5),
                            _sp("นาย ข", 50, "PARTY")])
    assert [g.surface for g in groups] == ["บริษัท ก จำกัด", "นาย ข"]
    assert len(groups[0].mentions) == 2 and groups[0].confidence == 0.9
    a, b, rest = pick_parties(groups)
    assert (a.surface, b.surface, rest) == ("บริษัท ก จำกัด", "นาย ข", [])


@thai
def test_thai_person_party_is_not_an_alias():
    text = "ข้าพเจ้า นางสาวพรทิพย์ ศรีวิไล ผู้ให้เช่า ตกลงให้ บริษัท คาเฟ่ดีไซน์ จำกัด ผู้เช่า"
    spans = [_sp("นางสาวพรทิพย์ ศรีวิไล", text.index("นางสาว"), "PARTY"),
             _sp("บริษัท คาเฟ่ดีไซน์ จำกัด", text.index("บริษัท"), "PARTY")]
    rec = assemble(text, spans, lang="th")
    assert rec["Party_A"]["text"] == "นางสาวพรทิพย์ ศรีวิไล"
    assert rec["Party_B"]["text"] == "บริษัท คาเฟ่ดีไซน์ จำกัด"
    assert rec["Other_Parties"] == []


def test_unquoted_short_latin_alias_is_demoted():
    text = "between Skype Communications S.A. (Skype) and Big Company Limited"
    spans = [_sp("Skype Communications S.A.", 8, "PARTY"), _sp("Skype", text.index("(Skype") + 1, "PARTY"),
             _sp("Big Company Limited", text.index("Big"), "PARTY")]
    rec = assemble(text, spans, lang="en")
    assert (rec["Party_A"]["text"], rec["Party_B"]["text"]) == ("Skype Communications S.A.", "Big Company Limited")
    assert rec["Other_Parties"] == ["Skype"]


def test_english_validation_rejects_name_fragments_and_dateless_term_clauses():
    # legal-form suffix on its own is a fragment of a name; a lowercase common word is not a name
    for frag in ("Corp", "LTD", "GP", "S.A.", '"Corporation"', "together", "anonyme", "Affiliates"):
        assert not validate_span(Span(0, 5, "PARTY", frag), lang="en"), frag
    # ...but CUAD-style quoted defined terms and real short names stay
    for ok in ('"Company"', '"you"', "IREYA B.V", "Bruce Jenner", "Party A", "DISTRIBUTOR"):
        assert validate_span(Span(0, 5, "PARTY", ok), lang="en"), ok
    # an END_DATE clause must talk about ending or duration; a year alone is not enough
    assert not validate_span(Span(0, 5, "END_DATE", "Antitrust Improvements Act of 1976, as amended."), lang="en")
    assert not validate_span(Span(0, 5, "END_DATE", "Addendum, set forth in Section 7.2."), lang="en")
    assert validate_span(Span(0, 5, "END_DATE", "The Agreement shall end on the End Date without further action"), lang="en")
    assert validate_span(Span(0, 5, "END_DATE", "shall continue for so long as Photronics holds the shares"), lang="en")
    assert validate_span(Span(0, 5, "END_DATE", "December 31, 2020"), lang="en")


def test_parties_boilerplate_never_becomes_party_a_or_b():
    text = ('NICE and Contractor are collectively referred to as the Parties. '
            'NICE Ltd. ("NICE") and Acme Corp. ("Contractor")')
    spans = [_sp("NICE and Contractor are collectively referred to as the Parties.", 0, "PARTY", 0.95),
             _sp("NICE Ltd.", text.index("NICE Ltd."), "PARTY"), _sp("Acme Corp.", text.index("Acme"), "PARTY")]
    rec = assemble(text, spans, lang="en")
    assert (rec["Party_A"]["text"], rec["Party_B"]["text"]) == ("NICE Ltd.", "Acme Corp.")
    assert rec["Other_Parties"][0].startswith("NICE and Contractor")


def test_assemble_splits_signing_and_effective_dates_from_merged_agreement_dates():
    from contract_ner.postprocess import assemble

    text = ('This Agreement is entered into on October 1, 2019 and is made effective as of November 1, 2019 '
            '(the "Effective Date"), by and between Acme Corp. and Beta Inc.')
    spans = [(Span(text.index(d), text.index(d) + len(d), "AGREEMENT_DATE", d), c)
             for d, c in (("October 1, 2019", 0.9), ("November 1, 2019", 0.85))]
    s = assemble(text, spans, lang="en")["summary"]
    assert (s["Agreement_Date"], s["Start_Date"]) == ("2019-10-01", "2019-11-01")
    # a single date is both the agreement date and the inferred start date
    rec = assemble(text, spans[:1], lang="en")
    assert rec["summary"]["Start_Date"] == "2019-10-01" and rec["Start_Date"]["inferred_from"] == "Agreement_Date"


@pytest.mark.parametrize("text,iso", [
    ("7/7/98", "1998-07-07"),
    ("3-31-16", "2016-03-31"),
    ("Dated 05/05/98", "1998-05-05"),
    ("4/8/2020", "2020-04-08"),          # English: month first
    ("24 OCT 2011", "2011-10-24"),
    ("the first day of July, 2017 (", "2017-07-01"),
    ("the twenty-first day of May 2003", "2003-05-21"),
    ("Section 5.1.2", None),
    ("September ___, 2019", None),
])
def test_normalize_date_english_numeric_and_ordinal_forms(text, iso):
    assert normalize_date(text, "en") == iso


def test_normalize_date_reads_thai_numeric_dates_day_first():
    assert normalize_date("4/8/2568", "th") == "2025-08-04"
    assert normalize_date("4/8/2568") == "2025-08-04"  # a BE year marks a Thai date even without Thai script


def test_normalize_jurisdiction_handles_caps_apostrophes_and_adjective_law():
    from contract_ner.postprocess import normalize_jurisdiction

    assert normalize_jurisdiction("SHALL BE GOVERNED BY THE LAWS OF THE STATE OF NEW YORK (INCLUDING")["english"] == "STATE OF NEW YORK"
    assert normalize_jurisdiction("the laws of the People's Republic of China that")["english"] == "People's Republic of China"
    assert normalize_jurisdiction("construed in accordance with English law, save")["english"] == "English law"
