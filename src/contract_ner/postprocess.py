"""Span validation and normalisation → structured contract metadata.

The token classifier gives us labelled character spans. This module turns them
into the JSON record from the project deck (Party_A/B, dates as ISO-8601,
money with currency, penalty rates, jurisdiction) and performs a light
span-level validation pass: a span whose surface form cannot possibly be an
instance of its type (a "date" with no digits, a "jurisdiction" that names no
court/law/arbitral body, …) is demoted so it never reaches the summary fields.

Everything here is rule-based and deliberately conservative: the raw span text
and its offsets are always preserved so a human reviewer can verify the draft
(the deck's "human verification still required" note).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from dateutil import parser as du_parser
from pythainlp.transliterate import romanize

from .data.thai_lexicon import EN_MONTHS, PLACE_EN, THAI_MONTHS, THAI_MONTHS_ABBR
from .schema import Span

# ------------------------------------------------------------------ digits / numbers
_THAI_TO_ASCII = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def ascii_digits(s: str) -> str:
    return s.translate(_THAI_TO_ASCII)


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _first_number(s: str) -> float | None:
    m = _NUM_RE.search(ascii_digits(s))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


# ------------------------------------------------------------------ dates
_MONTH_LOOKUP: dict[str, int] = {}
for i, (full, abbr, en) in enumerate(zip(THAI_MONTHS, THAI_MONTHS_ABBR, EN_MONTHS), start=1):
    _MONTH_LOOKUP[full] = i
    _MONTH_LOOKUP[abbr] = i
    _MONTH_LOOKUP[abbr.replace(".", "")] = i
    _MONTH_LOOKUP[en.lower()] = i
    _MONTH_LOOKUP[en[:3].lower()] = i
_MONTH_LOOKUP["พฤศจิกายน"] = 11
_MONTH_LOOKUP["sept"] = 9

_MONTH_ALT = "|".join(sorted((re.escape(m) for m in _MONTH_LOOKUP), key=len, reverse=True))
_TH_DATE_RE = re.compile(
    rf"(?P<d>\d{{1,2}})\s*(?:เดือน\s*)?(?P<m>{_MONTH_ALT})\s*(?:พ\.?ศ\.?|ค\.?ศ\.?)?\s*(?P<y>\d{{4}})",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"(?P<d>\d{1,2})[/\-.](?P<m>\d{1,2})[/\-.](?P<y>\d{4})")
_EN_MDY_RE = re.compile(rf"(?P<m>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(rf"(?P<m>{_MONTH_ALT})\s*(?:พ\.?ศ\.?)?\s*(?P<y>\d{{4}})", re.IGNORECASE)
_CE_MARK_RE = re.compile(r"ค\.?ศ\.?")


def _fix_year(y: int, text: str) -> int:
    """Buddhist-era years (พ.ศ.) are 543 ahead of the common era."""
    if _CE_MARK_RE.search(text):
        return y
    return y - 543 if y > 2400 else y


def normalize_date(text: str) -> str | None:
    """Thai/English date expression → ISO ``YYYY-MM-DD`` (or ``YYYY-MM``)."""
    t = ascii_digits(text)
    m = _TH_DATE_RE.search(t) or _EN_MDY_RE.search(t)
    if m:
        mon = _MONTH_LOOKUP.get(m.group("m").lower())
        if mon:
            return _iso(int(m.group("d")), mon, _fix_year(int(m.group("y")), t))
    m = _NUMERIC_DATE_RE.search(t)
    if m:
        return _iso(int(m.group("d")), int(m.group("m")), _fix_year(int(m.group("y")), t))
    # English free text ("8th day of May 2014", "the 1st of June, 2019")
    if re.search(r"\d{4}", t) and re.search(r"[A-Za-z]{3,}", t) and re.search(r"(?<!\d)\d{1,2}(?!\d)", t):
        try:
            d = du_parser.parse(re.sub(r"\bday of\b", "", t, flags=re.I), fuzzy=True, default=datetime(1900, 1, 1))
            if d.year != 1900:
                return d.date().isoformat()
        except (ValueError, OverflowError):
            pass
    m = _MONTH_YEAR_RE.search(t)
    if m:
        mon = _MONTH_LOOKUP.get(m.group("m").lower())
        if mon:
            return f"{_fix_year(int(m.group('y')), t):04d}-{mon:02d}"
    return None


def _iso(d: int, m: int, y: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


# ------------------------------------------------------------------ money / rates
_CURRENCY = [
    (re.compile(r"บาท|THB|฿"), "THB"),
    (re.compile(r"ดอลลาร์สหรัฐ|เหรียญสหรัฐ|ดอลลาร์|USD|US\$|\$"), "USD"),
    (re.compile(r"ยูโร|EUR|€"), "EUR"),
    (re.compile(r"เยน|JPY|¥"), "JPY"),
    (re.compile(r"ปอนด์|GBP|£"), "GBP"),
    (re.compile(r"หยวน|CNY|RMB"), "CNY"),
    (re.compile(r"ริงกิต|MYR"), "MYR"),
    (re.compile(r"ดอลลาร์สิงคโปร์|SGD"), "SGD"),
]
_PER = [
    (re.compile(r"ต่อวัน|วันละ|per day|/day|daily"), "day"),
    (re.compile(r"ต่อเดือน|เดือนละ|รายเดือน|per month|/month|monthly"), "month"),
    (re.compile(r"ต่อปี|ปีละ|รายปี|per annum|per year|/year|annually|p\.a\."), "year"),
    (re.compile(r"ต่อสัปดาห์|สัปดาห์ละ|per week|weekly"), "week"),
    (re.compile(r"ต่อครั้ง|ครั้งละ|per occurrence|per breach|each (?:breach|occurrence)|แต่ละครั้ง"), "occurrence"),
    (re.compile(r"ต่อเที่ยว|เที่ยวละ|per trip"), "trip"),
    (re.compile(r"ต่อชั่วโมง|ชั่วโมงละ|per hour|hourly"), "hour"),
]
_PER_PERSON = re.compile(r"คนละ|ต่อคน|per person|per head")
_MILLION = re.compile(r"ล้าน")


def _per(s: str) -> str | None:
    for rx, unit in _PER:
        if rx.search(s):
            return unit
    return None


def normalize_money(text: str) -> dict | None:
    t = ascii_digits(text)
    amount = _first_number(t)
    if amount is None:
        return None
    if _MILLION.search(t):
        amount *= 1_000_000
    currency = next((code for rx, code in _CURRENCY if rx.search(t)), None)
    per = _per(t)
    out = {"amount": amount, "currency": currency, "per": per}
    if _PER_PERSON.search(t):
        out["per_person"] = True
    out["display"] = _money_display(amount, currency, per, out.get("per_person", False))
    return out


def _money_display(amount: float, currency: str | None, per: str | None, per_person: bool = False) -> str:
    num = f"{amount:,.0f}" if float(amount).is_integer() else f"{amount:,.2f}"
    s = f"{num} {currency or ''}".strip()
    if per_person:
        s += " per person"
    if per:
        s += f" per {per}"
    return s


_PERCENT_RE = re.compile(r"(?:ร้อยละ\s*(\d+(?:\.\d+)?))|(?:(\d+(?:\.\d+)?)\s*(?:%|เปอร์เซ็นต์|percent))", re.IGNORECASE)
_BASIS = [
    (re.compile(r"มูลค่าสัญญา|ค่าจ้างตามสัญญา|contract (?:value|price)"), "contract value"),
    (re.compile(r"ค่าจ้าง|ราคาพัสดุ|ราคาสินค้า|มูลค่างาน|มูลค่าของ"), "contract price"),
    (re.compile(r"ค้างชำระ|ผิดนัด|outstanding|overdue|unpaid"), "outstanding amount"),
]


def normalize_rate(text: str) -> dict | None:
    t = ascii_digits(text)
    per = _per(t)
    basis = next((b for rx, b in _BASIS if rx.search(t)), None)
    m = _PERCENT_RE.search(t)
    if m:
        value = float(m.group(1) or m.group(2))
        out = {"value": value, "unit": "percent", "per": per, "basis": basis}
        out["display"] = f"{value:g}%" + (f" per {per}" if per else "") + (f" of {basis}" if basis else "")
        return out
    money = normalize_money(t)
    if money:
        out = {"value": money["amount"], "unit": money["currency"] or "amount", "per": per, "basis": basis}
        if money.get("per_person"):
            out["per_person"] = True
        out["display"] = money["display"] + ("" if per else " (fixed)")
        return out
    return None


# ------------------------------------------------------------------ payment due
_EVENT_EN = [
    (re.compile(r"ใบแจ้งหนี้|ใบวางบิล|invoice"), "receipt of invoice"),
    (re.compile(r"ตรวจรับ"), "acceptance of the work/goods"),
    (re.compile(r"ส่งมอบ"), "delivery"),
    (re.compile(r"ลงนาม"), "signing of the contract"),
    (re.compile(r"สิ้นไตรมาส"), "end of the quarter"),
    (re.compile(r"จัดงาน"), "the event"),
    (re.compile(r"ได้รับสินค้า"), "receipt of the goods"),
]
_EVERY_DAY_RE = re.compile(r"(?:ทุก|ภายใน|ไม่เกิน|ก่อน)?\s*วันที่\s*(\d{1,2})\s*ของ\s*(?:ทุก|แต่ละ)?\s*เดือน\s*(ถัดไป)?")
_WITHIN_RE = re.compile(r"ภายใน\s*(\d{1,3})\s*(?:\([^)]*\)\s*)?(วันทำการ|วัน)")
_END_MONTH_RE = re.compile(r"สิ้นเดือน|วันทำการสุดท้ายของเดือน")
_ADVANCE_RE = re.compile(r"ล่วงหน้า")
_EN_WITHIN_RE = re.compile(r"within\s+(\d+)\s+(business\s+)?days", re.IGNORECASE)


def normalize_payment_due(text: str) -> str:
    t = ascii_digits(text)
    m = _EVERY_DAY_RE.search(t)
    if m:
        s = f"day {int(m.group(1))} of each month"
        if m.group(2):
            s = f"day {int(m.group(1))} of the following month"
        if _ADVANCE_RE.search(t):
            s += " (in advance)"
        return s
    m = _WITHIN_RE.search(t) or _EN_WITHIN_RE.search(t)
    if m:
        n = int(m.group(1))
        unit = "business days" if (m.group(2) or "").strip() in ("วันทำการ", "business") else "days"
        ev = next((e for rx, e in _EVENT_EN if rx.search(t)), None)
        return f"within {n} {unit}" + (f" of {ev}" if ev else "")
    if _END_MONTH_RE.search(t):
        return "end of each month"
    iso = normalize_date(t)
    if iso:
        prefix = "by " if re.search(r"ภายใน|ไม่เกิน|ก่อน|by|before", t) else "on "
        return prefix + iso
    if re.search(r"ทุกเดือน|รายเดือน|monthly", t):
        return "monthly"
    if re.search(r"รายไตรมาส|ทุกไตรมาส|quarterly", t):
        return "quarterly"
    if re.search(r"รายปี|ทุกปี|annually|yearly", t):
        return "annually"
    return text.strip()


# ------------------------------------------------------------------ jurisdiction
_COURT_EN = {
    "ศาลแพ่ง": "Civil Court (Bangkok)",
    "ศาลแพ่งกรุงเทพฯ": "Bangkok Civil Court",
    "ศาลแพ่งกรุงเทพมหานคร": "Bangkok Civil Court",
    "ศาลแพ่งกรุงเทพใต้": "Civil Court, South Bangkok",
    "ศาลแพ่งธนบุรี": "Thonburi Civil Court",
    "ศาลแพ่งตลิ่งชัน": "Taling Chan Civil Court",
    "ศาลแพ่งพระโขนง": "Phra Khanong Civil Court",
    "ศาลแพ่งมีนบุรี": "Min Buri Civil Court",
    "ศาลทรัพย์สินทางปัญญาและการค้าระหว่างประเทศกลาง": "Central Intellectual Property and International Trade Court",
    "ศาลแรงงานกลาง": "Central Labour Court",
    "ศาลปกครองกลาง": "Central Administrative Court",
    "ศาลล้มละลายกลาง": "Central Bankruptcy Court",
    "ศาลภาษีอากรกลาง": "Central Tax Court",
    "กฎหมายไทย": "Thai law",
    "กฎหมายแห่งราชอาณาจักรไทย": "Laws of the Kingdom of Thailand",
    "กฎหมายของประเทศไทย": "Laws of Thailand",
}
def _romanize(thai: str) -> str:
    if thai in PLACE_EN:
        return PLACE_EN[thai]
    try:
        return romanize(thai, engine="royin").title()
    except Exception:  # pragma: no cover - transliteration is best-effort
        return thai


_PROV_COURT_RE = re.compile(r"^ศาล(จังหวัด|แขวง|ปกครอง)(.+)$")
_ARBITRATION_RE = re.compile(r"อนุญาโตตุลาการ|arbitrat|THAC", re.IGNORECASE)
_GOV_LAW_RE = re.compile(r"กฎหมาย|laws? of|governed by", re.IGNORECASE)


def normalize_jurisdiction(text: str) -> dict:
    t = text.strip()
    kind = "arbitration" if _ARBITRATION_RE.search(t) else "court" if ("ศาล" in t or re.search(r"court", t, re.I)) \
        else "governing_law" if _GOV_LAW_RE.search(t) else "other"
    english = _COURT_EN.get(t)
    if not english:
        m = _PROV_COURT_RE.match(t)
        if m:
            level = {"จังหวัด": "Provincial Court", "แขวง": "Kwaeng (District) Court", "ปกครอง": "Administrative Court"}[m.group(1)]
            english = f"{_romanize(m.group(2))} {level}"
        elif kind == "arbitration" and "สำนักงานศาลยุติธรรม" in t:
            english = "Thai Arbitration Institute, Office of the Judiciary"
        elif "THAC" in t:
            english = "Thailand Arbitration Center (THAC)"
    return {"kind": kind, "english": english}


# ------------------------------------------------------------------ span validation
_DIGIT_RE = re.compile(r"[\d๐-๙]")


def validate_span(span: Span) -> bool:
    """Cheap type-consistency check on the surface form (span classification stage)."""
    t = span.text.strip()
    if not t:
        return False
    if span.label in ("AGREEMENT_DATE", "START_DATE", "END_DATE"):
        return bool(_DIGIT_RE.search(t)) and normalize_date(t) is not None
    if span.label in ("CONTRACT_VALUE", "PENALTY_RATE"):
        return bool(_DIGIT_RE.search(t))
    if span.label == "PAYMENT_DUE":
        return bool(_DIGIT_RE.search(t)) or bool(re.search(r"สิ้นเดือน|ต้นเดือน|ราย|ทุก|month|quarter|annual|invoice", t))
    if span.label == "JURISDICTION":
        return bool(re.search(r"ศาล|กฎหมาย|อนุญาโตตุลาการ|court|law|arbitra|tribunal|jurisdiction", t, re.I))
    if span.label == "PARTY":
        return len(t) >= 2 and not t.replace(",", "").replace(".", "").isdigit()
    return True


# ------------------------------------------------------------------ assembly
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|ltd|limited|llc|l\.l\.c|corp|corporation|company|co|plc|gmbh|ag|s\.a|n\.v|b\.v|l\.p|lp|llp|pty|sa|srl|s\.r\.l|kk|k\.k)\b\.?",
    re.IGNORECASE,
)
_QUOTE_STRIP_RE = re.compile(r"^[\"'“”‘’(\[\s]+|[\"'“”‘’)\]\s]+$")


def _party_key(text: str) -> str:
    return re.sub(r"\s+", " ", _QUOTE_STRIP_RE.sub("", text)).strip().casefold()


@dataclass
class PartyGroup:
    key: str
    surface: str
    first: int
    mentions: list[Span] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def looks_like_alias(self) -> bool:
        words = self.surface.split()
        return len(words) <= 2 and not _LEGAL_SUFFIX_RE.search(self.surface)


def group_parties(spans: list[tuple[Span, float]]) -> list[PartyGroup]:
    groups: dict[str, PartyGroup] = {}
    for sp, conf in spans:
        k = _party_key(sp.text)
        if not k:
            continue
        g = groups.get(k)
        if g is None:
            g = groups[k] = PartyGroup(k, _QUOTE_STRIP_RE.sub("", sp.text).strip(), sp.start)
        g.mentions.append(sp)
        g.confidence = max(g.confidence, conf)
    return sorted(groups.values(), key=lambda g: g.first)


def pick_parties(groups: list[PartyGroup]) -> tuple[PartyGroup | None, PartyGroup | None, list[PartyGroup]]:
    """First two *legal names* in order of appearance; short aliases are demoted."""
    names = [g for g in groups if not g.looks_like_alias]
    aliases = [g for g in groups if g.looks_like_alias]
    ordered = names + aliases if names else groups
    a = ordered[0] if ordered else None
    b = ordered[1] if len(ordered) > 1 else None
    rest = ordered[2:]
    return a, b, rest


def _span_record(sp: Span, conf: float, **extra) -> dict:
    rec = {"text": sp.text, "start": sp.start, "end": sp.end, "confidence": round(conf, 4)}
    rec.update(extra)
    return rec


def _pick_primary(cands: list[tuple[Span, float]], prefer=None) -> tuple[Span, float] | None:
    if not cands:
        return None
    if prefer:
        preferred = [c for c in cands if prefer(c[0])]
        if preferred:
            cands = preferred
    # a clearly more confident span wins, otherwise the first mention
    return sorted(cands, key=lambda c: (-round(c[1], 1), c[0].start))[0]


def assemble(text: str, spans: list[tuple[Span, float]], lang: str = "th") -> dict:
    """Build the structured record from ``(span, confidence)`` predictions."""
    valid = [(sp, c) for sp, c in spans if validate_span(sp)]
    invalid = [(sp, c) for sp, c in spans if not validate_span(sp)]
    by_label: dict[str, list[tuple[Span, float]]] = {}
    for sp, c in valid:
        by_label.setdefault(sp.label, []).append((sp, c))

    out: dict = {}
    a, b, others = pick_parties(group_parties(by_label.get("PARTY", [])))
    for key, g in (("Party_A", a), ("Party_B", b)):
        out[key] = None if g is None else {
            "text": g.surface, "confidence": round(g.confidence, 4), "mentions": len(g.mentions),
            "start": g.mentions[0].start, "end": g.mentions[0].end,
        }
    out["Other_Parties"] = [g.surface for g in others]

    for key, label in (("Agreement_Date", "AGREEMENT_DATE"), ("Start_Date", "START_DATE"), ("End_Date", "END_DATE")):
        p = _pick_primary(by_label.get(label, []))
        out[key] = None if p is None else _span_record(p[0], p[1], normalized=normalize_date(p[0].text))
    if out["Start_Date"] is None and out["Agreement_Date"] is not None:
        out["Start_Date"] = {**out["Agreement_Date"], "inferred_from": "Agreement_Date"}

    p = _pick_primary(by_label.get("CONTRACT_VALUE", []))
    out["Contract_Value"] = None if p is None else _span_record(p[0], p[1], normalized=normalize_money(p[0].text))

    p = _pick_primary(by_label.get("PAYMENT_DUE", []))
    out["Payment_Due_Date"] = None if p is None else _span_record(p[0], p[1], normalized=normalize_payment_due(p[0].text))

    p = _pick_primary(by_label.get("PENALTY_RATE", []))
    out["Penalty_Rate"] = None if p is None else _span_record(p[0], p[1], normalized=normalize_rate(p[0].text))

    p = _pick_primary(by_label.get("JURISDICTION", []), prefer=lambda s: normalize_jurisdiction(s.text)["kind"] in ("court", "arbitration"))
    out["Jurisdiction"] = None if p is None else _span_record(p[0], p[1], normalized=normalize_jurisdiction(p[0].text))
    out["Governing_Law"] = None
    for sp, c in by_label.get("JURISDICTION", []):
        if normalize_jurisdiction(sp.text)["kind"] == "governing_law":
            out["Governing_Law"] = _span_record(sp, c, normalized=normalize_jurisdiction(sp.text))
            break

    out["entities"] = [
        {**_span_record(sp, c), "label": sp.label, "valid": True} for sp, c in valid
    ] + [
        {**_span_record(sp, c), "label": sp.label, "valid": False} for sp, c in invalid
    ]
    out["entities"].sort(key=lambda e: e["start"])
    out["lang"] = lang
    out["summary"] = summary_line(out)
    return out


def summary_line(rec: dict) -> dict:
    """Compact one-value-per-field view (what goes into a metadata table)."""
    def text(k):
        v = rec.get(k)
        return v.get("text") if v else None

    def norm(k, sub=None):
        v = rec.get(k)
        if not v:
            return None
        n = v.get("normalized")
        if isinstance(n, dict):
            return n.get(sub) or v.get("text")
        return n or v.get("text")

    return {
        "Party_A": text("Party_A"),
        "Party_B": text("Party_B"),
        "Agreement_Date": norm("Agreement_Date"),
        "Start_Date": norm("Start_Date"),
        "End_Date": norm("End_Date"),
        "Contract_Value": norm("Contract_Value", "display"),
        "Payment_Due_Date": norm("Payment_Due_Date"),
        "Penalty_Rate": norm("Penalty_Rate", "display"),
        "Jurisdiction": norm("Jurisdiction", "english"),
    }
