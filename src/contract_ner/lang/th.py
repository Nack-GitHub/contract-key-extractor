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


RULES = LangRules(lang="th", segment_words=segment_words, validate_span=validate_span, romanize=romanize)
