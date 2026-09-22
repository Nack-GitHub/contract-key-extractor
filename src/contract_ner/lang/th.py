"""Thai rules: PyThaiNLP ``newmm`` segmentation with a contract-domain dictionary.

This is the only module in the core pipeline that imports PyThaiNLP; install
the ``thai`` extra (``pip install -e ".[thai]"``) to use it.
"""
from __future__ import annotations

import functools
import re

from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import Tokenizer
from pythainlp.transliterate import romanize as _pythainlp_romanize
from pythainlp.util import Trie

from ..data.thai_lexicon import PLACE_EN
from ..schema import Span
from . import LangRules

# Domain adaptation of the newmm dictionary. Entity boundaries can only fall on
# word boundaries, so compounds that glue a non-entity word to an entity word
# (``เขตอำนาจศาล`` → jurisdiction + court) are split, and contract vocabulary
# that newmm otherwise breaks into unhelpful pieces (``เที่ยว|ละ``) is added.
DOMAIN_REMOVE = {"เขตอำนาจศาล", "ตามกฎหมาย", "อัตราร้อยละ", "ในอัตรา"}
DOMAIN_ADD = {
    "เขตอำนาจ", "เดือนละ", "วันละ", "ปีละ", "เที่ยวละ", "ครั้งละ", "คนละ", "ชั่วโมงละ", "งวดละ", "เล่มละ",
    "เครื่องละ", "ต่อเดือน", "ต่อวัน", "ต่อปี", "ต่อครั้ง", "ต่อเที่ยว", "บาทถ้วน", "นับถัดจาก", "นับแต่",
    "ผู้ให้เช่า", "ผู้เช่า", "ผู้กู้", "ผู้ให้กู้", "ผู้อนุญาต", "ผู้รับอนุญาต", "ผู้รับเหมา", "ผู้จัดจำหน่าย",
    "ตัวแทนจำหน่าย", "ผู้ให้สิทธิ", "ผู้รับสิทธิ", "ผู้เปิดเผยข้อมูล", "ผู้รับข้อมูล", "คู่สัญญา", "เบี้ยปรับ",
    "ดอกเบี้ยผิดนัด", "หนังสือค้ำประกัน", "ภาษีมูลค่าเพิ่ม", "ใบกำกับภาษี", "ใบวางบิล", "ใบสั่งซื้อ",
    "อนุญาโตตุลาการ", "สถาบันอนุญาโตตุลาการ", "ศาลแพ่งกรุงเทพใต้", "ศาลทรัพย์สินทางปัญญาและการค้าระหว่างประเทศกลาง",
    "ศาลแรงงานกลาง", "ศาลปกครองกลาง", "ศาลล้มละลายกลาง", "ศาลภาษีอากรกลาง", "ลงวันที่", "ลงนาม",
}


@functools.lru_cache(maxsize=1)
def thai_tokenizer() -> Tokenizer:
    """newmm tokenizer with the contract-domain dictionary (built once)."""
    words = (set(thai_words()) - DOMAIN_REMOVE) | DOMAIN_ADD
    return Tokenizer(custom_dict=Trie(words), engine="newmm", keep_whitespace=True)


def segment_words(text: str) -> list[tuple[int, int]]:
    toks = thai_tokenizer().word_tokenize(text)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for tok in toks:
        if not tok:
            continue
        # newmm preserves the surface string, so tokens tile the text in order.
        idx = text.find(tok, pos)
        if idx < 0:  # should not happen; fall back to sequential assumption
            idx = pos
        end = idx + len(tok)
        if not tok.isspace():
            offsets.append((idx, end))
        pos = end
    return offsets


_DIGIT_RE = re.compile(r"[\d๐-๙]")
_TERM_WORD_RE = re.compile(r"ปี|เดือน|วัน|year|month|day|term|expir|terminat|effective|commence", re.IGNORECASE)
_JURISDICTION_RE = re.compile(r"ศาล|กฎหมาย|อนุญาโตตุลาการ|court|law|arbitra|tribunal", re.IGNORECASE)
_PAYMENT_RE = re.compile(r"สิ้นเดือน|ต้นเดือน|ราย|ทุก|month|quarter|annual|invoice", re.IGNORECASE)


def validate_span(span: Span) -> bool:
    """Keyword-level plausibility of ``span.text`` for its type (mixed Thai/English contracts are common)."""
    t = span.text.strip()
    if span.label in ("AGREEMENT_DATE", "START_DATE", "END_DATE"):
        if len(t.split()) <= 4:
            from ..postprocess import normalize_date
            return bool(_DIGIT_RE.search(t)) and normalize_date(t) is not None
        return bool(_DIGIT_RE.search(t)) or bool(_TERM_WORD_RE.search(t))
    if span.label in ("CONTRACT_VALUE", "PENALTY_RATE"):
        return bool(_DIGIT_RE.search(t))
    if span.label == "PAYMENT_DUE":
        return bool(_DIGIT_RE.search(t)) or bool(_PAYMENT_RE.search(t))
    if span.label == "JURISDICTION":
        return bool(_JURISDICTION_RE.search(t))
    if span.label == "PARTY":
        return len(t) >= 2 and re.search(r"[^\W\d_]", t) is not None
    return True


def romanize(thai: str) -> str:
    if thai in PLACE_EN:
        return PLACE_EN[thai]
    try:
        return _pythainlp_romanize(thai, engine="royin").title()
    except Exception:  # pragma: no cover - transliteration is best-effort
        return thai


# ---------------------------------------------------------------- surface forms
# Buckets for the label-distribution profile (``contract-ner profile``) and the
# per-form recall in evaluation. Thai contracts write the same value many ways —
# a date in Buddhist or Christian era with Thai or Arabic numerals, an amount in
# digits or in words, a penalty as ร้อยละ or % — and the generator's coverage of
# each form is exactly what the gold sets test.
_THAI_NUM_RE = re.compile(r"[๐-๙]")
_BUDDHIST_ERA_RE = re.compile(r"พ\.?ศ\.?")
_CHRISTIAN_ERA_RE = re.compile(r"ค\.?ศ\.?|\bA\.?D\.?")
_SLASH_DATE_RE = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}")
_THAI_MONTH_RE = re.compile(r"มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.")
_FOREIGN_CCY_RE = re.compile(r"USD|EUR|JPY|GBP|CNY|\$|ดอลลาร์|ยูโร|เยน|หยวน", re.IGNORECASE)
_MILLION_RE = re.compile(r"ล้าน|แสน|หมื่น|พัน")
_PER_PERIOD_RE = re.compile(r"ละ|ต่อ(?:เดือน|วัน|ปี|ครั้ง|เที่ยว|ชั่วโมง|งวด)")
_PERCENT_WORD_RE = re.compile(r"ร้อยละ")
_PERCENT_SIGN_RE = re.compile(r"%|เปอร์เซ็นต์")
_WITHIN_RE = re.compile(r"ภายใน")
_EVERY_RE = re.compile(r"ทุก(?:วันที่|สิ้นเดือน|เดือน)|สิ้นเดือน|ต้นเดือน")
_COMPANY_RE = re.compile(r"บริษัท|ห้างหุ้นส่วน|หจก|บมจ|จำกัด")
_PERSON_RE = re.compile(r"^\s*(?:นาย|นาง|นางสาว|น\.ส\.|ดร\.)")
_ORG_RE = re.compile(r"มหาวิทยาลัย|สำนักงาน|กรม|กระทรวง|สมาคม|มูลนิธิ|องค์การ|สหกรณ์|โรงพยาบาล|เทศบาล")
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARBITRATION_RE = re.compile(r"อนุญาโตตุลาการ|arbitrat", re.IGNORECASE)
_COURT_RE = re.compile(r"ศาล|court", re.IGNORECASE)


def _date_form(t: str) -> str:
    if _SLASH_DATE_RE.search(t):
        return "numeric"                       # 01/01/2569
    era = "buddhist-era" if _BUDDHIST_ERA_RE.search(t) else ("christian-era" if _CHRISTIAN_ERA_RE.search(t) else "year-without-era")
    numerals = "thai-numerals" if _THAI_NUM_RE.search(t) else "arabic-numerals"
    if not _THAI_MONTH_RE.search(t) and not re.search(r"[\d๐-๙]", t):
        return "no-date"                       # a clause with no date in it
    return f"{era}/{numerals}"


def span_form(span: Span) -> str:
    """Coarse surface form of ``span``; see :data:`FORMS` for the buckets per label."""
    t = " ".join(span.text.split())
    if span.label in ("AGREEMENT_DATE", "START_DATE", "END_DATE"):
        if span.label == "END_DATE" and not _THAI_MONTH_RE.search(t) and re.search(r"(?:ปี|เดือน|วัน)", t):
            return "duration"                  # กำหนด 2 ปี — a term length, not a date
        return _date_form(t)
    if span.label == "CONTRACT_VALUE":
        if _FOREIGN_CCY_RE.search(t):
            return "foreign-currency"
        if _MILLION_RE.search(t):
            return "amount-in-words"           # 2.5 ล้านบาท
        if _PER_PERIOD_RE.search(t):
            return "per-period"                # เดือนละ 45,000 บาท
        return "baht-digits"
    if span.label == "PENALTY_RATE":
        if _PERCENT_WORD_RE.search(t):
            return "percent-word"              # ร้อยละ 0.1 ต่อวัน
        if _PERCENT_SIGN_RE.search(t):
            return "percent-sign"              # 15% ต่อเดือน
        return "fixed-amount"                  # วันละ 1,000 บาท
    if span.label == "PAYMENT_DUE":
        if _WITHIN_RE.search(t):
            return "within-n-days"
        if _EVERY_RE.search(t):
            return "recurring-day"
        return "other"
    if span.label == "JURISDICTION":
        if _ARBITRATION_RE.search(t):
            return "arbitration"
        if _COURT_RE.search(t):
            return "court"
        return "governing-law"
    if span.label == "PARTY":
        if _COMPANY_RE.search(t):
            return "company"
        if _PERSON_RE.search(t):
            return "person"
        if _ORG_RE.search(t):
            return "organization"
        if _LATIN_RE.search(t):
            return "latin-name"
        return "other"
    return "unclassified"


#: buckets ``span_form`` can return per label, in reporting order
FORMS: dict[str, tuple[str, ...]] = {
    "PARTY": ("company", "person", "organization", "latin-name", "other"),
    "AGREEMENT_DATE": ("buddhist-era/arabic-numerals", "buddhist-era/thai-numerals", "christian-era/arabic-numerals", "numeric",
                       "year-without-era/arabic-numerals", "no-date"),
    "START_DATE": ("buddhist-era/arabic-numerals", "buddhist-era/thai-numerals", "christian-era/arabic-numerals", "numeric",
                       "year-without-era/arabic-numerals", "no-date"),
    "END_DATE": ("buddhist-era/arabic-numerals", "buddhist-era/thai-numerals", "christian-era/arabic-numerals", "numeric",
                   "duration", "no-date"),
    "CONTRACT_VALUE": ("baht-digits", "per-period", "amount-in-words", "foreign-currency"),
    "PAYMENT_DUE": ("within-n-days", "recurring-day", "other"),
    "PENALTY_RATE": ("percent-word", "percent-sign", "fixed-amount"),
    "JURISDICTION": ("court", "arbitration", "governing-law"),
}


RULES = LangRules(lang="th", segment_words=segment_words, validate_span=validate_span, romanize=romanize,
                  span_form=span_form, span_forms=FORMS)
