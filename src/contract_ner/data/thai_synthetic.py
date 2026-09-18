"""Synthetic, automatically annotated Thai contracts.

No public Thai contract NER corpus exists (the deck plans to self-collect and
label contracts), so we bootstrap a training set from templates. The generator
is deliberately noisy in the *right* ways:

* many surface forms per entity (Thai/Buddhist-era dates, abbreviations, Thai
  numerals, ``ร้อยละ`` vs ``%``, ``เดือนละ`` vs ``ต่อเดือน`` ...);
* nine contract families with their own role vocabulary;
* hard negatives that look like entities but are not: signatories and
  witnesses (person names that are *not* parties), addresses, registration
  and contract numbers, statutory citations, VAT, deposits/retention, loan
  interest (not a penalty), delivery deadlines (not a payment due date),
  insurance and unit prices (not the contract value);
* both long numbered contracts and short single-paragraph excerpts.

Each document is emitted with character-level spans, so the same pipeline
that consumes hand-labelled data consumes these.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from pythainlp.util import bahttext

from ..schema import Document, Span
from . import thai_lexicon as L

Segment = tuple[str, str | None]  # (surface text, entity label or None)


# ===================================================================== builder
class _Doc:
    """Accumulates text segments while tracking entity offsets."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[Span] = []
        self.pos = 0

    def add(self, text: str, label: str | None = None) -> None:
        if not text:
            return
        if label:
            self.spans.append(Span(self.pos, self.pos + len(text), label, text))
        self.parts.append(text)
        self.pos += len(text)

    def extend(self, segs: list[Segment]) -> None:
        for t, l in segs:
            self.add(t, l)

    def text(self) -> str:
        return "".join(self.parts)


def render(template: str, fills: dict[str, Segment | list[Segment]]) -> list[Segment]:
    """Fill ``{NAME}`` placeholders; each fill is a segment or list of segments."""
    out: list[Segment] = []
    i = 0
    while i < len(template):
        j = template.find("{", i)
        if j < 0:
            out.append((template[i:], None))
            break
        k = template.find("}", j)
        if k < 0:
            raise ValueError(f"unclosed placeholder in template: {template!r}")
        if j > i:
            out.append((template[i:j], None))
        name = template[j + 1 : k]
        val = fills[name]
        if isinstance(val, list):
            out.extend(val)
        else:
            out.append(val)
        i = k + 1
    return out


# ===================================================================== sampling helpers
@dataclass
class Party:
    name: str
    kind: str  # company | person | org
    role: str
    signatory: str = ""  # person who signs for a company/org (never a party)
    position: str = ""
    address: str = ""
    regno: str = ""


@dataclass
class Ctx:
    """Everything sampled for one contract, so clauses stay mutually consistent."""

    rng: random.Random
    ctype: dict
    title: str
    A: Party
    B: Party
    agreement: date
    start: date
    end: date | None
    months: int
    value: int | None
    currency: str
    thai_digits: bool = False
    fields: dict = field(default_factory=dict)  # gold normalised values, for tests


def _person_name(rng: random.Random, title: bool = True) -> str:
    t = rng.choice(L.PERSON_TITLES) if title else ""
    sep = " " if (t and rng.random() < 0.25) else ""
    return f"{t}{sep}{rng.choice(L.FIRST_NAMES)} {rng.choice(L.LAST_NAMES)}"


def _company_name(rng: random.Random) -> str:
    n = rng.choice([1, 1, 2, 2, 2, 3])
    cores = rng.sample(L.COMPANY_CORES, n)
    if rng.random() < 0.12:
        cores[0] = rng.choice(L.COMPANY_CORES_LATIN)
    core = " ".join(cores)
    form = rng.random()
    if form < 0.55:
        return f"บริษัท {core} จำกัด"
    if form < 0.72:
        return f"บริษัท {core} จำกัด (มหาชน)"
    if form < 0.84:
        return f"ห้างหุ้นส่วนจำกัด {core}"
    if form < 0.92:
        return f"บริษัท {core} (ประเทศไทย) จำกัด"
    return f"หจก. {core}"


def _address(rng: random.Random) -> str:
    no = f"{rng.randint(1, 999)}" + (f"/{rng.randint(1, 99)}" if rng.random() < 0.5 else "")
    if rng.random() < 0.6:
        d = rng.choice(L.BANGKOK_DISTRICTS)
        return (f"เลขที่ {no} ถนน{rng.choice(L.ROADS)} แขวง{d} เขต{d} กรุงเทพมหานคร "
                f"{rng.randint(10100, 10900)}")
    p = rng.choice(L.PROVINCES[1:])
    return (f"เลขที่ {no} หมู่ {rng.randint(1, 15)} ตำบล{rng.choice(L.BANGKOK_DISTRICTS)} "
            f"อำเภอเมือง{p} จังหวัด{p} {rng.randint(10000, 96000)}")


def _regno(rng: random.Random) -> str:
    return "0" + "".join(str(rng.randint(0, 9)) for _ in range(12))


def _make_party(rng: random.Random, kind: str, role: str) -> Party:
    if kind == "company":
        name = _company_name(rng)
        pos = rng.choice(["กรรมการผู้จัดการ", "กรรมการผู้มีอำนาจลงนาม", "ผู้จัดการทั่วไป", "ประธานเจ้าหน้าที่บริหาร",
                          "หุ้นส่วนผู้จัดการ", "ผู้รับมอบอำนาจ"])
        return Party(name, kind, role, _person_name(rng), pos, _address(rng), _regno(rng))
    if kind == "org":
        name = rng.choice(L.ORG_NAMES)
        pos = rng.choice(["อธิบดี", "ผู้อำนวยการ", "อธิการบดี", "ผู้ว่าการ", "นายกเทศมนตรี", "ปลัด", "ผู้รับมอบอำนาจ",
                          "รองอธิการบดี ปฏิบัติการแทนอธิการบดี"])
        return Party(name, kind, role, _person_name(rng), pos, _address(rng), "")
    return Party(_person_name(rng), "person", role, "", "", _address(rng), "")


def _party_kinds(rng: random.Random, ctype_key: str) -> tuple[str, str]:
    r = rng.random()
    if ctype_key == "employment":
        return ("company" if r < 0.85 else "org"), "person"
    if ctype_key == "loan":
        return rng.choice([("company", "person"), ("person", "person"), ("company", "company"), ("person", "company")])
    if ctype_key == "lease":
        return rng.choice([("company", "company"), ("person", "company"), ("company", "person"), ("person", "person")])
    if ctype_key in ("construction", "service", "sale"):
        return ("org" if r < 0.35 else "company"), ("company" if rng.random() < 0.85 else "person")
    return rng.choice([("company", "company"), ("company", "company"), ("company", "person"), ("org", "company")])


# ------------------------------------------------------------------ dates
def _be(d: date) -> int:
    return d.year + 543


def fmt_thai_date(rng: random.Random, d: date) -> str:
    """One of the many ways a date is written in a Thai contract."""
    m = d.month - 1
    r = rng.random()
    if r < 0.42:
        return f"{d.day} {L.THAI_MONTHS[m]} {_be(d)}"
    if r < 0.62:
        return f"{d.day} {L.THAI_MONTHS[m]} พ.ศ. {_be(d)}"
    if r < 0.72:
        return f"{d.day} {L.THAI_MONTHS_ABBR[m]} {_be(d)}"
    if r < 0.80:
        return f"{d.day:02d}/{d.month:02d}/{_be(d)}"
    if r < 0.86:
        return f"{d.day} เดือน {L.THAI_MONTHS[m]} พ.ศ. {_be(d)}"
    if r < 0.91:
        return f"{d.day} {L.THAI_MONTHS[m]} ค.ศ. {d.year}"
    if r < 0.95:
        return f"{d.day} {L.EN_MONTHS[m]} {d.year}"
    if r < 0.98:
        return f"{d.day}-{d.month:02d}-{_be(d)}"
    return f"{d.day} {L.THAI_MONTHS[m]} {d.year}"


def _sample_dates(rng: random.Random) -> tuple[date, date, date | None, int]:
    start = date(2019, 1, 1) + timedelta(days=rng.randint(0, 365 * 9))
    if rng.random() < 0.55:
        start = start.replace(day=1)
    lag = rng.choice([0, 0, 0, 1, 3, 7, 14, 15, 30, 45, 60])
    agreement = start - timedelta(days=lag)
    months = rng.choice([3, 6, 6, 12, 12, 12, 18, 24, 24, 36, 36, 48, 60])
    end = _add_months(start, months) - timedelta(days=1)
    if rng.random() < 0.18:
        end = None  # duration only, no explicit end date
    return agreement, start, end, months


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y += d.year
    m += 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def _duration_text(rng: random.Random, months: int) -> str:
    if months % 12 == 0 and rng.random() < 0.85:
        y = months // 12
        return rng.choice([f"{y} ปี", f"{y} ปี", f"{y} ({_thai_num_word(y)}) ปี"])
    return rng.choice([f"{months} เดือน", f"{months} ({_thai_num_word(months)}) เดือน"])


_NUM_WORDS = {1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่", 5: "ห้า", 6: "หก", 7: "เจ็ด", 8: "แปด", 9: "เก้า", 10: "สิบ",
              12: "สิบสอง", 18: "สิบแปด", 24: "ยี่สิบสี่", 36: "สามสิบหก", 48: "สี่สิบแปด", 60: "หกสิบ"}


def _thai_num_word(n: int) -> str:
    return _NUM_WORDS.get(n, str(n))


# ------------------------------------------------------------------ money
def _sample_amount(rng: random.Random, key: str) -> int:
    lo, hi = {
        "service": (50_000, 20_000_000), "construction": (500_000, 500_000_000), "sale": (20_000, 50_000_000),
        "lease": (5_000, 500_000), "loan": (50_000, 20_000_000), "license": (100_000, 30_000_000),
        "nda": (200_000, 5_000_000), "distribution": (500_000, 50_000_000), "employment": (12_000, 250_000),
    }[key]
    x = int(rng.uniform(lo ** 0.5, hi ** 0.5) ** 2)  # skew towards the lower end
    unit = 1000 if x >= 100_000 else 500 if x >= 20_000 else 100
    return max(lo, (x // unit) * unit)


def fmt_money(rng: random.Random, amount: int, currency: str = "THB") -> tuple[str, str]:
    """Return ``(entity_span, trailing_text_outside_span)``."""
    if currency != "THB":
        cur = {"USD": rng.choice(["ดอลลาร์สหรัฐ", "เหรียญสหรัฐ", "USD"]), "EUR": rng.choice(["ยูโร", "EUR"]),
               "JPY": rng.choice(["เยน", "JPY"])}[currency]
        if cur.isascii():
            return f"{cur} {amount:,}", ""
        return f"{amount:,} {cur}", ""
    r = rng.random()
    if amount % 1_000_000 == 0 and amount >= 1_000_000 and r < 0.10:
        return f"{amount // 1_000_000} ล้านบาท", ""
    if amount % 100_000 == 0 and amount >= 1_000_000 and r < 0.16:
        return f"{amount / 1_000_000:g} ล้านบาท", ""
    if r < 0.45:
        num = f"{amount:,}"
    elif r < 0.75:
        num = f"{amount:,}.00"
    elif r < 0.82:
        num = f"{amount:,}.-"
    else:
        num = f"{amount:,}"
    span = f"{num} บาท"
    trail = ""
    t = rng.random()
    if t < 0.40:
        trail = f" ({bahttext(amount)})"
    elif t < 0.55:
        trail = rng.choice([" (รวมภาษีมูลค่าเพิ่มแล้ว)", " ซึ่งรวมภาษีมูลค่าเพิ่มร้อยละ 7 แล้ว",
                            " (ไม่รวมภาษีมูลค่าเพิ่ม)", " ถ้วน"])
    return span, trail


# ===================================================================== party blocks
def _alias(rng: random.Random, role: str, side: str) -> str:
    q = rng.choice([('"', '"'), ("“", "”"), ("“", "”")])
    tail = rng.choice(["", "", f" {side}"])
    r = rng.random()
    if r < 0.40:
        return f' ({q[0]}{role}{q[1]}){tail}'
    if r < 0.70:
        return f" ซึ่งต่อไปในสัญญานี้เรียกว่า {q[0]}{role}{q[1]}{tail}"
    if r < 0.88:
        return f" ซึ่งต่อไปนี้จะเรียกว่า {q[0]}{role}{q[1]}{tail}"
    return f" ซึ่งต่อไปในสัญญานี้จะเรียกว่า {q[0]}{role}{q[1]}{tail}"


def party_block(rng: random.Random, p: Party, side: str, detail: bool) -> list[Segment]:
    segs: list[Segment] = [(p.name, "PARTY")]
    if detail:
        if p.kind == "company":
            bits = []
            if rng.random() < 0.75:
                bits.append(f"ทะเบียนนิติบุคคลเลขที่ {p.regno}")
            if rng.random() < 0.25:
                bits.append(rng.choice([f"ซึ่งจดทะเบียนเป็นนิติบุคคล ณ สำนักงานทะเบียนหุ้นส่วนบริษัทจังหวัด{rng.choice(L.PROVINCES[1:])}",
                                        "จดทะเบียนเป็นนิติบุคคลตามกฎหมายไทย ณ สำนักงานทะเบียนหุ้นส่วนบริษัทกรุงเทพมหานคร",
                                        "ซึ่งจดทะเบียนเป็นนิติบุคคลตามประมวลกฎหมายแพ่งและพาณิชย์"]))
            if rng.random() < 0.85:
                bits.append(rng.choice(["สำนักงานตั้งอยู่", "สำนักงานใหญ่ตั้งอยู่", "มีภูมิลำเนาอยู่"]) + p.address)
            if rng.random() < 0.85:
                bits.append(f"โดย {p.signatory} {p.position}")
            if bits:
                segs.append((" " + " ".join(bits), None))
        elif p.kind == "org":
            bits = []
            if rng.random() < 0.6:
                bits.append("ตั้งอยู่" + p.address)
            bits.append(f"โดย {p.signatory} {p.position}")
            segs.append((" " + " ".join(bits), None))
        else:
            bits = []
            if rng.random() < 0.2:
                bits.append(f"พนักงานรหัส {rng.choice(['EMP', 'E', 'ID'])}-{rng.randint(1000, 99999):05d} ตำแหน่ง{rng.choice(['เจ้าหน้าที่บัญชี', 'วิศวกร', 'พนักงานขาย', 'ผู้จัดการแผนก', 'ช่างเทคนิค'])}")
            if rng.random() < 0.6:
                bits.append(f"อายุ {rng.randint(22, 70)} ปี")
            if rng.random() < 0.5:
                cid = f"{rng.randint(1,8)}-{rng.randint(1000,9999)}-{rng.randint(10000,99999)}-{rng.randint(10,99)}-{rng.randint(0,9)}"
                bits.append(f"บัตรประจำตัวประชาชนเลขที่ {cid}")
            if rng.random() < 0.8:
                bits.append("อยู่บ้าน" + p.address)
            if bits:
                segs.append((" " + " ".join(bits), None))
    segs.append((_alias(rng, p.role, side), None))
    return segs


# ===================================================================== clauses
def preamble(ctx: Ctx) -> list[Segment]:
    rng = ctx.rng
    ad = (fmt_thai_date(rng, ctx.agreement), "AGREEMENT_DATE")
    place = rng.choice([f"ณ {rng.choice(L.PROVINCES)}", "ณ สำนักงานของ" + ctx.A.role, f"ณ จังหวัด{rng.choice(L.PROVINCES[1:])}",
                        "ณ กรุงเทพมหานคร", f"ณ ที่ทำการ{ctx.A.role}", ""])
    detail = rng.random() < 0.75
    fills = {
        "AD": ad, "PLACE": (place, None),
        "A": party_block(rng, ctx.A, "ฝ่ายหนึ่ง", detail),
        "B": party_block(rng, ctx.B, "อีกฝ่ายหนึ่ง", detail),
        "TITLE": (ctx.title, None),
        "BARE_PLACE": (rng.choice([rng.choice(L.PROVINCES), "สำนักงานของ" + ctx.A.role, f"จังหวัด{rng.choice(L.PROVINCES[1:])}"]), None),
    }
    t = rng.choice([
        "สัญญาฉบับนี้ทำขึ้น {PLACE} เมื่อวันที่ {AD} ระหว่าง {A} กับ {B}",
        "{TITLE}ฉบับนี้ทำขึ้น {PLACE} เมื่อวันที่ {AD} ระหว่าง {A} ฝ่ายหนึ่ง กับ {B} อีกฝ่ายหนึ่ง",
        "สัญญานี้ทำขึ้นเมื่อวันที่ {AD} {PLACE} ระหว่าง {A} และ {B}",
        "สัญญาฉบับนี้ทำขึ้น ณ วันที่ {AD} ระหว่าง {A} กับ {B}",
        "{TITLE}นี้ทำขึ้นเมื่อวันที่ {AD} โดยมีคู่สัญญาคือ {A} และ {B}",
        "ทำที่ {BARE_PLACE}\nวันที่ {AD}\n\n{TITLE}ฉบับนี้ทำขึ้นระหว่าง {A} กับ {B}",
        "สัญญาฉบับนี้ทำขึ้นระหว่าง {A} กับ {B} เมื่อวันที่ {AD}",
        "ข้าพเจ้า {A} ตกลงทำสัญญากับ {B} เมื่อวันที่ {AD} {PLACE}",
        "ระหว่าง {A} กับ {B} ได้ทำ{TITLE}ตามสัญญาเลขที่ {NO} ลงวันที่ {AD}",
        "{TITLE} เลขที่ {NO} ลงวันที่ {AD}\nระหว่าง {A} กับ {B}",
        "{TITLE}\nคู่สัญญาฝ่ายที่หนึ่ง: {A}\nคู่สัญญาฝ่ายที่สอง: {B}\nวันที่ลงนาม: {AD}",
        "{TITLE}\n\n{A} และ {B} ตกลงทำสัญญานี้เมื่อวันที่ {AD} โดยมีข้อความดังต่อไปนี้",
    ])
    fills["NO"] = (f"{rng.choice(['', 'สจ.', 'กสพ.', 'จ.', 'ซ.'])}{rng.randint(1, 400)}/{_be(ctx.agreement)}", None)
    segs = render(t, fills)
    # fix double spaces produced by an empty PLACE
    return [(s.replace("  ", " "), l) for s, l in segs]


def clause_scope(ctx: Ctx) -> list[Segment]:
    rng, k = ctx.rng, ctx.ctype["key"]
    A, B = ctx.A.role, ctx.B.role
    if k in ("service", "construction"):
        w = rng.choice(L.WORK_ITEMS)
        return render(rng.choice([
            "{A}ตกลงจ้างและ{B}ตกลงรับจ้าง{W} ตามรายละเอียดขอบเขตงานที่ระบุในเอกสารแนบท้ายสัญญา ผนวก 1",
            "{B}ตกลงรับจ้างดำเนิน{W} ให้แล้วเสร็จตามแบบและรายการที่{A}กำหนด",
            "{A}ตกลงว่าจ้าง{B}ให้ดำเนิน{W} โดย{B}ต้องปฏิบัติงานให้ถูกต้องครบถ้วนตามมาตรฐานวิชาชีพ",
        ]), {"A": (A, None), "B": (B, None), "W": (w, None)})
    if k == "sale":
        g = rng.choice(L.GOODS)
        n = rng.randint(2, 500)
        return render(rng.choice([
            "{B}ตกลงขายและ{A}ตกลงซื้อ{G} จำนวน {N} ชุด ตามรายละเอียดในใบเสนอราคาเลขที่ QT-{Q}",
            "{A}ตกลงซื้อ{G} จำนวน {N} เครื่อง จาก{B} ในราคาต่อหน่วย {U} บาท",
            "{B}ตกลงขาย{G}ให้แก่{A} ตามรายการและปริมาณที่ระบุในเอกสารแนบท้าย",
        ]), {"A": (A, None), "B": (B, None), "G": (g, None), "N": (f"{n:,}", None),
             "Q": (f"{rng.randint(1000, 9999)}/{_be(ctx.agreement)}", None),
             "U": (f"{rng.randint(5, 900) * 100:,}", None)})
    if k == "lease":
        return render(rng.choice([
            "{A}ตกลงให้เช่าและ{B}ตกลงเช่า{P} เพื่อใช้ประกอบกิจการ{X}",
            "{A}ตกลงให้{B}เช่า{P} ตั้งอยู่{ADDR}",
        ]), {"A": (A, None), "B": (B, None), "P": (rng.choice(L.PROPERTIES), None),
             "X": (rng.choice(["ร้านอาหาร", "สำนักงาน", "คลังสินค้า", "ร้านค้าปลีก", "โรงงาน", "คลินิก"]), None),
             "ADDR": (_address(rng), None)})
    if k == "loan":
        return render(rng.choice([
            "{B}ได้กู้ยืมเงินจาก{A}และได้รับเงินไปครบถ้วนแล้วในวันทำสัญญานี้",
            "{A}ตกลงให้{B}กู้ยืมเงินเพื่อใช้{X} โดยได้ส่งมอบเงินกู้ให้แก่{B}ครบถ้วนแล้ว",
        ]), {"A": (A, None), "B": (B, None), "X": (rng.choice(["หมุนเวียนในกิจการ", "ซื้อเครื่องจักร", "ลงทุนขยายกิจการ", "เป็นทุนการศึกษา"]), None)})
    if k == "license":
        return render(rng.choice([
            "{A}ตกลงอนุญาตให้{B}ใช้{X}ภายในประเทศไทยแต่เพียงผู้เดียว",
            "{A}อนุญาตให้{B}ใช้{X}ตามขอบเขตที่กำหนดในสัญญานี้ โดยไม่สามารถโอนสิทธิให้ผู้อื่น",
        ]), {"A": (A, None), "B": (B, None), "X": (rng.choice(["เครื่องหมายการค้า", "ซอฟต์แวร์", "ลิขสิทธิ์ในผลงาน", "สิทธิบัตรเลขที่ " + str(rng.randint(10000, 99999)) + " ", "แฟรนไชส์"]), None)})
    if k == "nda":
        return render(rng.choice([
            "{B}ตกลงรักษาข้อมูลที่ได้รับจาก{A}ไว้เป็นความลับ และจะไม่เปิดเผยแก่บุคคลภายนอกโดยไม่ได้รับความยินยอมเป็นลายลักษณ์อักษร",
            "คู่สัญญาตกลงแลกเปลี่ยนข้อมูลอันเป็นความลับเพื่อวัตถุประสงค์ในการ{X} และตกลงเก็บรักษาข้อมูลดังกล่าวตามเงื่อนไขของสัญญานี้",
        ]), {"A": (A, None), "B": (B, None), "X": (rng.choice(["ศึกษาความเป็นไปได้ในการร่วมลงทุน", "พัฒนาผลิตภัณฑ์ร่วมกัน", "ประเมินธุรกิจ"]), None)})
    if k == "distribution":
        return render("{A}แต่งตั้ง{B}เป็น{X}สินค้าประเภท{G}ในเขตพื้นที่{P}",
                      {"A": (A, None), "B": (B, None), "X": (rng.choice(["ตัวแทนจำหน่าย", "ผู้จัดจำหน่ายแต่เพียงผู้เดียว"]), None),
                       "G": (rng.choice(L.GOODS), None), "P": (rng.choice(["ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ", "ประเทศไทย", "กรุงเทพมหานครและปริมณฑล", "ภาคใต้"]), None)})
    # employment
    return render(rng.choice([
        "{A}ตกลงจ้าง{B}เข้าทำงานในตำแหน่ง{X} สังกัดฝ่าย{D}",
        "{B}ตกลงทำงานให้แก่{A}ในตำแหน่ง{X} โดยมีระยะเวลาทดลองงาน {T} วัน",
    ]), {"A": (A, None), "B": (B, None), "X": (rng.choice(["วิศวกรซอฟต์แวร์", "ผู้จัดการฝ่ายขาย", "นักบัญชี", "พนักงานขับรถ", "เจ้าหน้าที่การตลาด", "ผู้จัดการโครงการ"]), None),
         "D": (rng.choice(["ทรัพยากรบุคคล", "การตลาด", "เทคโนโลยีสารสนเทศ", "บัญชีและการเงิน", "ปฏิบัติการ"]), None),
         "T": (str(rng.choice([90, 119, 120])), None)})


def clause_term(ctx: Ctx) -> list[Segment]:
    rng = ctx.rng
    sd = (fmt_thai_date(rng, ctx.start), "START_DATE")
    dur = (_duration_text(rng, ctx.months), None)
    if ctx.end is not None:
        ed = (fmt_thai_date(rng, ctx.end), "END_DATE")
        t = rng.choice([
            "สัญญานี้มีผลบังคับใช้ตั้งแต่วันที่ {SD} ถึงวันที่ {ED}",
            "สัญญานี้มีกำหนดระยะเวลา {DUR} นับตั้งแต่วันที่ {SD} และสิ้นสุดในวันที่ {ED}",
            "ระยะเวลาของสัญญาเริ่มตั้งแต่วันที่ {SD} จนถึงวันที่ {ED} รวมระยะเวลา {DUR}",
            "คู่สัญญาตกลงให้สัญญานี้มีผลตั้งแต่วันที่ {SD} และมีผลสิ้นสุดลงในวันที่ {ED} เว้นแต่จะมีการบอกเลิกสัญญาก่อนกำหนด",
            "สัญญาฉบับนี้มีผลใช้บังคับตั้งแต่วันที่ {SD} เป็นต้นไป และจะสิ้นสุดลงในวันที่ {ED}",
            "อายุสัญญา {DUR} เริ่มวันที่ {SD} สิ้นสุดวันที่ {ED}",
            "{B}ตกลงเริ่มปฏิบัติงานตั้งแต่วันที่ {SD} และต้องดำเนินการให้แล้วเสร็จภายในวันที่ {ED}",
            "กำหนดเวลา{X}ตั้งแต่วันที่ {SD} ถึงวันที่ {ED} เป็นระยะเวลา {DUR}",
            "{B}ต้องเริ่มทำงานที่รับจ้างภายในวันที่ {SD} และจะต้องทำงานให้แล้วเสร็จบริบูรณ์ภายในวันที่ {ED} รวมเป็นระยะเวลา {DAYS} วัน",
            "ระยะเวลาสัญญา: {SD} – {ED} ({DUR})",
            "{B}ต้องส่งมอบ{DELIV} ณ {PLACE} ภายในวันที่ {ED} โดยเริ่มดำเนินการตั้งแต่วันที่ {SD}",
            "กำหนดส่งมอบ: ภายใน {DAYS} วันนับถัดจากวันลงนามในสัญญา คือภายในวันที่ {ED} ณ {PLACE}",
            "เริ่มงานวันที่ {SD} และกำหนดแล้วเสร็จวันที่ {ED}",
        ])
        fills = {"SD": sd, "ED": ed, "DUR": dur, "B": (ctx.B.role, None),
                 "X": (rng.choice(["เช่า", "จ้าง", "ให้สิทธิ", "จ้างงาน", "ของสัญญา"]), None),
                 "DAYS": (str((ctx.end - ctx.start).days + 1), None),
                 "DELIV": (rng.choice(["และติดตั้งเครื่องจักร", "สินค้าทั้งหมด", "งานงวดสุดท้าย", "เอกสารและระบบทั้งหมด"]), None),
                 "PLACE": (rng.choice([f"โรงงานของ{ctx.A.role} นิคมอุตสาหกรรม{rng.choice(['อมตะซิตี้', 'มาบตาพุด', 'บางปู', 'ลาดกระบัง', 'นวนคร'])} จังหวัด{rng.choice(L.PROVINCES[1:])}",
                                       f"คลังพัสดุ {rng.choice(L.ORG_NAMES)}", f"สำนักงานของ{ctx.A.role}", f"ศูนย์กระจายสินค้าจังหวัด{rng.choice(L.PROVINCES[1:])}"]), None)}
    else:
        t = rng.choice([
            "สัญญานี้มีกำหนดระยะเวลา {DUR} นับตั้งแต่วันที่ {SD} เป็นต้นไป",
            "สัญญาฉบับนี้มีผลใช้บังคับตั้งแต่วันที่ {SD} และมีอายุ {DUR}",
            "คู่สัญญาตกลงให้สัญญานี้มีผลตั้งแต่วันที่ {SD} มีกำหนด {DUR} และอาจต่ออายุได้อีกคราวละ {DUR} โดยความยินยอมของทั้งสองฝ่าย",
            "ให้สัญญานี้มีผลนับแต่วันที่ {SD} จนกว่างานจะแล้วเสร็จ แต่ทั้งนี้ต้องไม่เกิน {DUR}",
        ])
        fills = {"SD": sd, "DUR": dur}
    segs = render(t, fills)
    if rng.random() < 0.3:
        segs.append((" " + rng.choice([
            "ทั้งนี้ ให้ส่งมอบงานงวดสุดท้ายภายใน 30 วันนับจากวันสิ้นสุดสัญญา",
            "หากคู่สัญญาฝ่ายใดประสงค์จะต่ออายุสัญญา ให้แจ้งอีกฝ่ายทราบล่วงหน้าไม่น้อยกว่า 60 วัน",
            "การต่ออายุสัญญาต้องทำเป็นหนังสือลงนามโดยคู่สัญญาทั้งสองฝ่าย",
        ]), None))
    return segs


def clause_value(ctx: Ctx) -> list[Segment]:
    rng, k = ctx.rng, ctx.ctype["key"]
    assert ctx.value is not None
    span, trail = fmt_money(rng, ctx.value, ctx.currency)
    V = (span, "CONTRACT_VALUE")
    T = (trail, None)
    A, B = (ctx.A.role, None), (ctx.B.role, None)
    if k == "lease":
        per = rng.random()
        if per < 0.5:
            V = ("เดือนละ " + span, "CONTRACT_VALUE")
            t = rng.choice(["{B}ตกลงชำระค่าเช่าในอัตรา{V}{T}", "ค่าเช่ากำหนดไว้{V}{T}", "อัตราค่าเช่า {V}{T}"])
        else:
            V = (span + " ต่อเดือน", "CONTRACT_VALUE")
            t = rng.choice(["{B}ตกลงชำระค่าเช่าให้แก่{A}เป็นเงิน {V}{T}", "ค่าเช่าเป็นจำนวนเงิน {V}{T}"])
        segs = render(t, {"A": A, "B": B, "V": V, "T": T})
        if rng.random() < 0.5:
            dep = rng.choice([1, 2, 3])
            segs.append((f" และ{ctx.B.role}ตกลงวางเงินประกันการเช่าเท่ากับค่าเช่า {dep} เดือน เป็นเงิน {ctx.value * dep:,} บาท ในวันทำสัญญา", None))
        return segs
    if k == "employment":
        V = (rng.choice(["เดือนละ ", "ในอัตราเดือนละ "]) + span, "CONTRACT_VALUE")
        return render(rng.choice(["{A}ตกลงจ่ายค่าจ้างให้แก่{B}{V}{T}", "อัตราค่าจ้าง {V}{T} โดยจ่ายเดือนละหนึ่งครั้ง"]),
                      {"A": A, "B": B, "V": V, "T": T})
    if k == "loan":
        rate = rng.choice(["3", "5", "7.5", "10", "12", "15"])
        t = rng.choice([
            "{B}ได้กู้ยืมเงินจาก{A}เป็นจำนวน {V}{T} โดยตกลงเสียดอกเบี้ยในอัตราร้อยละ {R} ต่อปี",
            "จำนวนเงินกู้ {V}{T} อัตราดอกเบี้ยร้อยละ {R} ต่อปี คิดจากเงินต้นคงเหลือ",
            "เงินต้นตามสัญญานี้มีจำนวน {V}{T} ดอกเบี้ยร้อยละ {R} ต่อปี",
        ])
        return render(t, {"A": A, "B": B, "V": V, "T": T, "R": (rate, None)})
    if k == "nda":
        return render("ข้อมูลอันเป็นความลับภายใต้สัญญานี้มีมูลค่าประเมินเบื้องต้น {V}{T}", {"V": V, "T": T})
    t = rng.choice([
        "มูลค่าสัญญารวมทั้งสิ้น {V}{T}",
        "{A}ตกลงจ่ายค่าจ้างให้แก่{B}เป็นจำนวนเงินทั้งสิ้น {V}{T}",
        "ราคาค่าจ้างตามสัญญานี้เป็นเงิน {V}{T}",
        "คู่สัญญาตกลงราคาค่า{X}ทั้งสิ้นเป็นเงิน {V}{T}",
        "มูลค่าของสัญญานี้เท่ากับ {V}{T}",
        "ค่าตอบแทนตามสัญญานี้มีจำนวน {V}{T} ซึ่ง{A}จะชำระให้แก่{B}ตามงวดงานที่กำหนดในข้อถัดไป",
        "ราคา{X}รวมภาษีมูลค่าเพิ่มเป็นเงินทั้งสิ้น {V}{T}",
        "{B}ตกลงรับค่าจ้างในวงเงิน {V}{T} เป็นราคาเหมารวม",
        "วงเงินตามสัญญา: {V}{T}",
        "งบประมาณรวมทั้งสิ้น {V}{T}",
        "{A}ตกลงสนับสนุนทุนแก่{B}เป็นจำนวนเงิน {V}{T} ตลอดโครงการ",
        "ค่าบริการรายปีเป็นเงิน {V}{T}",
        "ประมาณการค่า{X}ตลอดอายุสัญญาไม่เกิน {V}{T}",
        "{A}ตกลงจ่ายและ{B}ตกลงรับเงินค่าจ้างจำนวนเงิน {V}{T} ซึ่งได้รวมภาษีมูลค่าเพิ่มจำนวน {VAT} บาท ตลอดจนภาษีอากรอื่น ๆ และค่าใช้จ่ายทั้งปวงด้วยแล้ว",
        "ค่า{X}ทั้งสิ้น {V}{T} ชำระเงินมัดจำร้อยละ {DEP} ของราคา คิดเป็นเงิน {DEPAMT} บาท ในวันลงนามสัญญา และเงินส่วนที่เหลือจำนวน {REST} บาท ชำระเมื่อส่งมอบ",
    ])
    if k == "sale":
        t = rng.choice([
            "ราคาสินค้าที่ตกลงซื้อขายรวมเป็นเงินทั้งสิ้น {V}{T}",
            "{A}ตกลงชำระราคาสินค้าให้แก่{B}เป็นเงิน {V}{T}",
            "มูลค่าการซื้อขายตามสัญญานี้เท่ากับ {V}{T}",
            "ราคาซื้อขายรวม {V}{T}",
        ])
    dep = rng.choice([10, 20, 30, 50])
    segs = render(t, {"A": A, "B": B, "V": V, "T": T,
                      "X": (rng.choice(["จ้าง", "สินค้า", "ก่อสร้าง", "บริการ", "สิทธิ", "จ้างเหมา", "ขนส่ง"]), None),
                      "VAT": (f"{ctx.value * 7 / 107:,.2f}", None), "DEP": (str(dep), None),
                      "DEPAMT": (f"{ctx.value * dep // 100:,}", None), "REST": (f"{ctx.value - ctx.value * dep // 100:,}", None)})
    if rng.random() < 0.35:
        bank = rng.choice(L.BANKS)
        segs.append((" " + rng.choice([
            "ทั้งนี้ ราคาดังกล่าวเป็นราคาคงที่ตลอดอายุสัญญา",
            f"โดย{ctx.A.role}จะหักเงินประกันผลงานไว้ร้อยละ 5 ของค่าจ้างแต่ละงวด",
            f"ราคาต่อหน่วย {rng.randint(1, 900) * 100:,} บาท จำนวน {rng.randint(2, 900)} หน่วย",
            f"{ctx.B.role}ต้องวางหลักประกันสัญญาเป็นจำนวนร้อยละ 5 ของมูลค่าสัญญา",
            f"{ctx.B.role}ได้นำหนังสือค้ำประกันของ{bank} จำกัด (มหาชน) สาขา{rng.choice(L.BANGKOK_DISTRICTS)} เลขที่ {rng.randint(1, 9999):04d}/{_be(ctx.agreement)} เป็นจำนวนเงิน {ctx.value // 20:,} บาท (ร้อยละ 5 ของราคาค่าจ้าง) มามอบให้แก่{ctx.A.role}เพื่อเป็นหลักประกันการปฏิบัติตามสัญญา",
            f"ราคา{rng.choice(['เครื่องละ', 'เล่มละ', 'ชุดละ', 'ตารางเมตรละ'])} {rng.randint(1, 900) * 100:,} บาท",
            f"อัตราค่าขนส่งเที่ยวละ {rng.randint(5, 90) * 100:,} บาท",
            f"ไม่รวมงบโฆษณาซึ่ง{ctx.A.role}จะชำระตรงให้แก่แพลตฟอร์มเอง โดยกำหนดงบขั้นต่ำเดือนละ {rng.randint(1, 50) * 10000:,} บาท",
            f"(เดือนละ {ctx.value // 12:,} บาท)",
        ]), None))
    return segs


def payment_due_phrase(rng: random.Random, ctx: Ctx) -> tuple[Segment, str]:
    """Return the PAYMENT_DUE span plus a normalised description for meta."""
    r = rng.random()
    if r < 0.22:
        d = rng.choice([1, 5, 5, 10, 15, 20, 25, 28, 30])
        s = rng.choice([f"ทุกวันที่ {d} ของเดือน", f"ทุกวันที่ {d} ของทุกเดือน", f"ภายในวันที่ {d} ของทุกเดือน",
                        f"ภายในวันที่ {d} ของแต่ละเดือน", f"ไม่เกินวันที่ {d} ของทุกเดือน"])
        return (s, "PAYMENT_DUE"), f"day {d} of each month"
    if r < 0.50:
        n = rng.choice([7, 7, 15, 15, 30, 30, 30, 45, 60, 90])
        ev = rng.choice(["ได้รับใบแจ้งหนี้", "ได้รับใบแจ้งหนี้ถูกต้องครบถ้วน", "ส่งมอบงานแต่ละงวด", "ตรวจรับงานเรียบร้อย",
                         "ได้รับสินค้า", "ลงนามในสัญญา", "ได้รับใบวางบิล", "ตรวจรับมอบงานถูกต้องครบถ้วน",
                         f"{ctx.A.role}ได้รับมอบงานถูกต้อง", "คณะกรรมการตรวจรับงานได้ตรวจรับงานงวดนั้นเรียบร้อยแล้ว",
                         "สิ้นไตรมาส", "จัดงาน", "ส่งมอบสินค้า", "ได้รับใบแจ้งหนี้และรายงานการปฏิบัติงานประจำเดือน",
                         "สิ้นเดือน", "ออกใบแจ้งหนี้", "ตรวจรับ"])
        unit = rng.choice(["วัน", "วัน", "วันทำการ"])
        s = rng.choice([f"ภายใน {n} {unit}นับจากวันที่{ev}", f"ภายใน {n} {unit}นับแต่วันที่{ev}",
                        f"ภายใน {n} {unit}หลังจาก{ev}", f"ภายใน {n} ({_thai_num_word(n) if n in _NUM_WORDS else n}) {unit}นับถัดจากวันที่{ev}",
                        f"ภายใน {n} {unit}นับจากวัน{ev}", f"ภายใน {n} {unit} นับจากวันที่{ev}", f"ภายใน {n} {unit}หลังจากวัน{ev}"])
        return (s, "PAYMENT_DUE"), f"within {n} {'business days' if unit == 'วันทำการ' else 'days'} of {ev}"
    if r < 0.62:
        s = rng.choice(["ทุกสิ้นเดือน", "ภายในสิ้นเดือน", "ทุกวันสิ้นเดือน", "ภายในวันทำการสุดท้ายของเดือน",
                        "ทุกวันที่ 1 ของเดือนถัดไป"])
        return (s, "PAYMENT_DUE"), "end of month"
    if r < 0.80:
        end = ctx.end or _add_months(ctx.start, ctx.months)
        due = ctx.start + timedelta(days=rng.randint(10, max(11, (end - ctx.start).days)))
        s = rng.choice(["ภายในวันที่ ", "ไม่เกินวันที่ ", "ในวันที่ "]) + fmt_thai_date(rng, due)
        return (s, "PAYMENT_DUE"), due.isoformat()
    n = rng.choice([2, 3, 4, 6, 12])
    s = rng.choice([f"ทุก {12 // n if n <= 12 else 1} เดือน", f"เป็นราย{rng.choice(['ไตรมาส', 'เดือน', 'ปี'])}",
                    "ล่วงหน้าทุกต้นเดือน", "ทุกวันศุกร์สุดท้ายของเดือน",
                    f"ทุกเดือน เดือนละ {rng.randint(1, 50) * 1000:,} บาท",
                    f"ภายในเดือน{rng.choice(L.THAI_MONTHS)}ของแต่ละปี",
                    f"ล่วงหน้าภายในวันที่ {rng.randint(1, 28)} {rng.choice(L.THAI_MONTHS)}ของทุกปี"])
    return (s, "PAYMENT_DUE"), s


def clause_payment(ctx: Ctx) -> list[Segment]:
    rng, k = ctx.rng, ctx.ctype["key"]
    A, B = (ctx.A.role, None), (ctx.B.role, None)
    PD, desc = payment_due_phrase(rng, ctx)
    ctx.fields["payment_due"] = desc
    t = rng.choice([
        "{A}ตกลงชำระเงินให้แก่{B}{PD}",
        "กำหนดชำระเงิน{PD}",
        "{A}จะชำระค่า{X}ให้แก่{B}{PD} โดยโอนเงินเข้าบัญชี{BANK} เลขที่บัญชี {ACC}",
        "การชำระเงินให้กระทำ{PD} ผ่านบัญชีธนาคารที่{B}แจ้งไว้",
        "{B}ต้องออกใบแจ้งหนี้ให้{A} และ{A}ต้องชำระเงิน{PD}",
        "กำหนดชำระค่า{X}{PD} หากวันครบกำหนดตรงกับวันหยุด ให้เลื่อนไปชำระในวันทำการถัดไป",
        "{A}ตกลงชำระเงินตามสัญญา{PD}",
        "{A}ตกลงชำระเงินล่วงหน้าร้อยละ {ADV} ในวันลงนามสัญญา และส่วนที่เหลือ{PD}",
        "{A}ตกลงชำระเงินมัดจำร้อยละ {ADV} ของราคาสินค้าในวันลงนามสัญญา และชำระส่วนที่เหลือ{PD}",
        "กำหนดจ่ายเงิน: ราย{PER} {PD}",
        "{A}จะจ่ายค่า{X}{PD} หลังจากได้รับใบแจ้งหนี้และใบลงเวลาปฏิบัติงาน",
        "{B}ยินยอมให้{A}หักจากเงินเดือน{PD} จนกว่าจะครบถ้วน",
    ])
    if k in ("service", "construction", "sale") and rng.random() < 0.45 and ctx.value:
        n = rng.choice([2, 3, 4])
        share = ctx.value // n
        t = rng.choice([
            "แบ่งชำระเป็น {N} งวด งวดที่ 1 จำนวน {S} บาท ชำระ{PD} งวดที่เหลือชำระตามความคืบหน้าของงาน",
            "{A}ตกลงแบ่งจ่ายค่าจ้างเป็น {N} งวด งวดละ {S} บาท โดยชำระ{PD}",
            "การชำระเงินแบ่งเป็น {N} งวด งวดแรกเป็นเงิน {S} บาท ชำระ{PD}",
        ])
        return render(t, {"A": A, "B": B, "PD": PD, "N": (str(n), None), "S": (f"{share:,}", None)})
    return render(t, {"A": A, "B": B, "PD": PD, "X": (rng.choice(["จ้าง", "สินค้า", "เช่า", "บริการ", "ตอบแทน", "ธรรมเนียม"]), None),
                      "BANK": (rng.choice(L.BANKS), None), "ADV": (str(rng.choice([10, 20, 30, 40, 50])), None),
                      "PER": (rng.choice(["เดือน", "ไตรมาส", "งวด"]), None),
                      "ACC": ("-".join(str(rng.randint(100, 999)) for _ in range(3)) + f"-{rng.randint(0, 9)}", None)})


def penalty_phrase(rng: random.Random) -> tuple[Segment, str]:
    r = rng.random()
    per_map = {"วัน": "day", "เดือน": "month", "ปี": "year"}
    if r < 0.30:
        rate = rng.choice(["0.1", "0.2", "0.25", "0.5", "1", "1.5", "2", "3", "5", "7.5", "10", "15"])
        per = rng.choice(["วัน", "วัน", "เดือน", "ปี"])
        basis = rng.choice(["มูลค่าสัญญา", "ค่าจ้าง", "ค่าจ้างตามสัญญา", "ราคาสินค้า", "ราคาพัสดุที่ยังไม่ได้รับมอบ", "ค่าจ้างรายเดือน",
                            "มูลค่างานที่ยังไม่ส่งมอบ", "ยอดที่ค้างชำระ"])
        s = rng.choice([f"ร้อยละ {rate} ต่อ{per}", f"ร้อยละ {rate} ต่อ{per}", f"อัตราร้อยละ {rate} ต่อ{per}",
                        f"ร้อยละ {rate} ของ{basis}ต่อ{per}", f"ร้อยละ {rate} ของ{basis}ต่อ{per}"])
        return (s, "PENALTY_RATE"), f"{rate}% per {per_map[per]}"
    if r < 0.55:
        rate = rng.choice(["0.1", "0.2", "0.5", "1", "1.5", "2", "5", "7.5", "10", "15", "18"])
        per = rng.choice(["วัน", "เดือน", "เดือน", "ปี"])
        s = rng.choice([f"{rate}% ต่อ{per}", f"{rate} % ต่อ{per}", f"{rate}% ของมูลค่าสัญญาต่อ{per}", f"{rate}% ต่อ{per}",
                        f"{rate}% per {per_map[per]}", f"{per}ละ {rate}% ของ{rng.choice(['ค่าจ้างตามสัญญา', 'มูลค่าสัญญา'])}"])
        return (s, "PENALTY_RATE"), f"{rate}% per {per_map[per]}"
    if r < 0.72:
        amt = rng.choice([500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000])
        per = rng.choice(["วัน", "วัน", "เดือน", "ครั้ง", "เที่ยว", "ชั่วโมง"])
        per_en = {**per_map, "ครั้ง": "occurrence", "เที่ยว": "trip", "ชั่วโมง": "hour"}[per]
        s = rng.choice([f"{per}ละ {amt:,} บาท", f"{amt:,} บาท ต่อ{per}", f"{amt:,} บาทต่อ{per}", f"อัตรา{per}ละ {amt:,} บาท",
                        f"คนละ {amt:,} บาทต่อ{per}", f"{amt:,} บาทต่อคนต่อ{per}"])
        return (s, "PENALTY_RATE"), f"{amt} THB per {per_en}"
    if r < 0.88:
        rate = rng.choice(["0.1", "0.2", "0.5", "1"])
        s = rng.choice([f"วันละ {rate}%", f"วันละร้อยละ {rate}", f"วันละ {rate} %"])
        return (s, "PENALTY_RATE"), f"{rate}% per day"
    amt = rng.choice([10000, 50000, 100000, 500000, 1000000, 2000000])
    s = rng.choice([f"{amt:,} บาท ต่อครั้ง", f"{amt:,} บาทต่อการผิดสัญญาแต่ละครั้ง", f"{amt:,} บาท", f"{amt:,} บาท ต่อการฝ่าฝืนแต่ละครั้ง",
                    f"ร้อยละ {rng.choice([10, 20, 25])} ของมูลค่าสัญญา"])
    return (s, "PENALTY_RATE"), f"{amt} THB fixed"


def clause_penalty(ctx: Ctx) -> list[Segment]:
    rng = ctx.rng
    A, B = (ctx.A.role, None), (ctx.B.role, None)
    P, desc = penalty_phrase(rng)
    ctx.fields["penalty"] = desc
    t = rng.choice([
        "หาก{B}ผิดนัดไม่ส่งมอบงานตามกำหนด {B}ยินยอมให้{A}ปรับเป็นรายวันในอัตรา{P} นับถัดจากวันครบกำหนดจนถึงวันที่ส่งมอบงานแล้วเสร็จ",
        "หากผิดนัดชำระ คิดค่าปรับ{P}",
        "ในกรณีที่{A}ชำระเงินล่าช้ากว่ากำหนด {A}ตกลงชำระดอกเบี้ยผิดนัดในอัตรา{P} ของจำนวนเงินที่ค้างชำระ",
        "หาก{B}ผิดสัญญาข้อหนึ่งข้อใด {B}ยอมเสียค่าปรับให้แก่{A}เป็นเงิน{P}",
        "ค่าปรับกรณีผิดนัด: {P}",
        "หากมีการชำระเงินล่าช้า {A}ยินยอมเสียเบี้ยปรับ{P} จนกว่าจะชำระครบถ้วน",
        "กรณีส่งมอบล่าช้า {B}ต้องชำระค่าปรับในอัตรา{P} ทั้งนี้ ค่าปรับรวมต้องไม่เกินร้อยละ 10 ของมูลค่าสัญญา",
        "หาก{B}ไม่ปฏิบัติตามสัญญา {B}ตกลงชดใช้ค่าเสียหายเป็นเบี้ยปรับ{P} โดยไม่ต้องพิสูจน์ความเสียหาย",
        "ผู้ผิดนัดชำระเงินต้องเสียดอกเบี้ยระหว่างผิดนัดในอัตรา{P} ตามประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 224",
        "หาก{B}ส่งมอบ{X}ล่าช้า {A}มีสิทธิปรับ{B}{P} จนกว่าจะส่งมอบครบถ้วน",
        "กรณีส่งมอบล่าช้า {B}ต้องชำระค่าปรับในอัตรา{P} ของมูลค่างานที่ยังไม่ส่งมอบ",
        "หากผิดนัด {B}จะต้องชำระค่าปรับให้แก่{A}เป็นรายวันในอัตรา{P} แต่ต้องไม่ต่ำกว่าวันละ {FLOOR} บาท นับถัดจากวันที่กำหนดแล้วเสร็จตามสัญญาจนถึงวันที่ทำงานแล้วเสร็จจริง",
        "หาก{B}ฝ่าฝืนข้อตกลงนี้ {B}ตกลงชำระค่าเสียหายที่กำหนดไว้ล่วงหน้าเป็นจำนวน {P} ทั้งนี้ ไม่ตัดสิทธิ{A}ที่จะเรียกร้องค่าเสียหายที่แท้จริงเพิ่มเติม",
        "อัตราค่าปรับ: {P} กรณี{X}ล่าช้า และปรับ {SECOND} กรณีจัดพนักงานไม่ครบ",
        "หาก{B}ไม่สามารถให้บริการได้ในวันใด {B}ต้องจัดหาสิ่งทดแทน มิฉะนั้นยินยอมให้{A}หักค่าบริการและปรับ{P}",
        "ค่าปรับ: ในอัตรา{P}",
        "หากส่งล่าช้ากว่ากำหนดเวลาในแต่ละเที่ยวเกิน {HRS} ชั่วโมง ยินยอมให้ปรับ{P}",
    ])
    second, _ = penalty_phrase(rng)
    return render(t, {"A": A, "B": B, "P": P, "X": (rng.choice(["งาน", "สินค้า", "ของ", "งวดงาน", "ส่งมอบ"]), None),
                      "FLOOR": (str(rng.choice([100, 200, 500, 1000])), None), "SECOND": second,
                      "HRS": (str(rng.choice([1, 2, 3, 4])), None)})


def clause_jurisdiction(ctx: Ctx) -> list[Segment]:
    rng = ctx.rng
    r = rng.random()
    if r < 0.62:
        J = (rng.choice(L.COURTS), "JURISDICTION")
        ctx.fields["jurisdiction"] = J[0]
        t = rng.choice([
            "หากมีข้อพิพาทเกิดขึ้นจากสัญญานี้ คู่สัญญาตกลงให้{J}เป็นศาลที่มีเขตอำนาจในการพิจารณาคดี",
            "ข้อพิพาทอยู่ภายใต้เขตอำนาจ{J}",
            "ข้อพิพาทใด ๆ อันเกิดจากหรือเกี่ยวเนื่องกับสัญญานี้ ให้เสนอต่อ{J}",
            "คู่สัญญาตกลงให้{J}เป็นศาลที่มีเขตอำนาจแต่เพียงศาลเดียว",
            "ในกรณีที่มีข้อพิพาท คู่สัญญาตกลงยื่นฟ้องต่อ{J}",
            "คู่สัญญาตกลงให้สัญญานี้อยู่ภายใต้บังคับของ{LAW} และให้{J}เป็นศาลที่มีอำนาจพิจารณาข้อพิพาท",
            "สัญญานี้ให้ตีความตาม{LAW} ข้อพิพาทที่เกิดขึ้นให้อยู่ในเขตอำนาจของ{J}",
            "หากตกลงกันไม่ได้ ให้นำข้อพิพาทขึ้นสู่{J}",
        ])
        law = (rng.choice(L.GOVERNING_LAW), "JURISDICTION")
        return render(t, {"J": J, "LAW": law})
    if r < 0.82:
        J = (rng.choice(L.ARBITRATION), "JURISDICTION")
        ctx.fields["jurisdiction"] = J[0]
        t = rng.choice([
            "ข้อพิพาทที่เกิดขึ้นจากสัญญานี้ให้ระงับโดยการอนุญาโตตุลาการของ{J} ตามข้อบังคับของสถาบันดังกล่าว",
            "คู่สัญญาตกลงให้เสนอข้อพิพาทต่อ{J} และให้คำชี้ขาดของอนุญาโตตุลาการเป็นที่สุด",
            "หากไม่สามารถตกลงกันได้ภายใน 30 วัน ให้เสนอข้อพิพาทต่อ{J} โดยใช้ภาษาไทยในการดำเนินกระบวนพิจารณา",
        ])
        return render(t, {"J": J})
    law = (rng.choice(L.GOVERNING_LAW), "JURISDICTION")
    ctx.fields["jurisdiction"] = law[0]
    t = rng.choice([
        "สัญญานี้อยู่ภายใต้บังคับและการตีความตาม{LAW}",
        "ให้ใช้{LAW}บังคับแก่สัญญานี้",
        "สัญญาฉบับนี้ให้เป็นไปตาม{LAW}ทุกประการ",
        "การระงับข้อพิพาท: ตามพระราชบัญญัติ{ACT} พ.ศ. {ACTYEAR} และให้เป็นไปตาม{LAW}",
        "กฎหมายที่ใช้บังคับ: {LAW}",
    ])
    return render(t, {"LAW": law, "ACT": (rng.choice(["การจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ", "ว่าด้วยข้อสัญญาที่ไม่เป็นธรรม", "คุ้มครองข้อมูลส่วนบุคคล", "อนุญาโตตุลาการ"]), None),
                      "ACTYEAR": (str(rng.choice([2540, 2545, 2560, 2562])), None)})


# ------------------------------------------------------------------ distractor clauses (no entities)
def clause_filler(ctx: Ctx) -> list[Segment]:
    rng = ctx.rng
    k = ctx.ctype["key"]
    A, B = ctx.A.role, ctx.B.role
    n1, n2 = rng.choice([7, 15, 30, 45, 60, 90, 120, 180]), rng.choice([1, 2, 3, 5])
    WORK = {"service", "construction", "sale", "distribution", "license"}
    ANY = None
    pool: list[tuple[set | None, str]] = [
        ({"service", "construction"}, f"{B}ต้องส่งมอบงานให้แล้วเสร็จภายใน {n1} วันนับถัดจากวันลงนามในสัญญา"),
        ({"sale", "distribution"}, f"{B}ต้องส่งมอบสินค้าให้แก่{A} ณ สถานที่ที่{A}กำหนด ภายใน {n1} วันนับจากวันที่ได้รับใบสั่งซื้อ"),
        ({"service", "construction"}, f"{B}รับประกันคุณภาพงานเป็นระยะเวลา {n2} ปี นับจากวันที่{A}ตรวจรับงานงวดสุดท้าย"),
        ({"sale", "distribution"}, f"{B}รับประกันความชำรุดบกพร่องของสินค้าเป็นเวลา {n1} วันนับแต่วันส่งมอบ"),
        (ANY, f"คู่สัญญาตกลงรักษาข้อมูลอันเป็นความลับของอีกฝ่ายไว้เป็นเวลา {n2} ปีนับจากวันสิ้นสุดสัญญา"),
        (ANY, f"คู่สัญญาฝ่ายใดฝ่ายหนึ่งมีสิทธิบอกเลิกสัญญาโดยแจ้งเป็นหนังสือให้อีกฝ่ายทราบล่วงหน้าไม่น้อยกว่า {n1} วัน"),
        (ANY, f"หากคู่สัญญาฝ่ายใดผิดสัญญาข้อใดข้อหนึ่ง และไม่แก้ไขภายใน {n1} วันนับแต่วันที่ได้รับหนังสือเตือน อีกฝ่ายมีสิทธิบอกเลิกสัญญาได้ทันที"),
        (ANY, "ในกรณีที่เกิดเหตุสุดวิสัย เช่น อุทกภัย อัคคีภัย การจลาจล หรือโรคระบาด ทำให้คู่สัญญาฝ่ายใดไม่สามารถปฏิบัติตามสัญญาได้ คู่สัญญาฝ่ายนั้นไม่ต้องรับผิด"),
        (WORK, f"{B}ต้องจัดให้มีประกันภัยความรับผิดต่อบุคคลภายนอกในวงเงินไม่น้อยกว่า {rng.choice([1, 2, 5, 10, 20]):,},000,000 บาท ตลอดอายุสัญญา"),
        (WORK, f"{B}ต้องวางหลักประกันการปฏิบัติตามสัญญาเป็นจำนวนร้อยละ {rng.choice([5, 10])} ของมูลค่าสัญญา ในรูปหนังสือค้ำประกันของธนาคาร"),
        ({"service", "construction"}, f"{A}จะหักเงินประกันผลงานไว้ร้อยละ {rng.choice([5, 10])} ของเงินค่าจ้างแต่ละงวด และจะคืนให้เมื่อพ้นกำหนดระยะเวลารับประกัน"),
        (WORK | {"lease"}, f"ราคาตามสัญญานี้ยังไม่รวมภาษีมูลค่าเพิ่มร้อยละ 7 ซึ่ง{A}จะชำระเพิ่มเติมตามใบกำกับภาษี"),
        (WORK | {"lease"}, f"{A}มีสิทธิหักภาษี ณ ที่จ่ายในอัตราร้อยละ {rng.choice([1, 3])} ตามที่กฎหมายกำหนด"),
        (WORK, f"{B}ตกลงให้ส่วนลดร้อยละ {rng.choice([5, 10, 15])} ในกรณีที่{A}ชำระเงินก่อนกำหนด"),
        (ANY, f"การแจ้งหรือการบอกกล่าวใด ๆ ให้ทำเป็นหนังสือส่งไปยังที่อยู่ของคู่สัญญาตามที่ระบุไว้ข้างต้น หรือทางอีเมล {rng.choice(['contract', 'legal', 'admin', 'info'])}@{rng.choice(['example', 'company', 'corp', 'group'])}.co.th"),
        (ANY, f"ติดต่อประสานงานได้ที่หมายเลขโทรศัพท์ 0{rng.randint(2, 9)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"),
        (ANY, f"{B}จะโอนสิทธิหรือหน้าที่ตามสัญญานี้ให้แก่บุคคลอื่นมิได้ เว้นแต่ได้รับความยินยอมเป็นหนังสือจาก{A}"),
        (ANY, "สัญญานี้ทำขึ้นเป็นสองฉบับมีข้อความถูกต้องตรงกัน คู่สัญญาได้อ่านและเข้าใจข้อความโดยตลอดแล้ว จึงลงลายมือชื่อไว้เป็นสำคัญต่อหน้าพยาน"),
        (WORK, f"บรรดาเอกสารแนบท้ายสัญญา ได้แก่ ผนวก 1 ขอบเขตงาน ผนวก 2 ใบเสนอราคาเลขที่ {rng.randint(100, 999)}/{_be(ctx.agreement)} ให้ถือเป็นส่วนหนึ่งของสัญญานี้"),
        (ANY, f"การแก้ไขเปลี่ยนแปลงสัญญาต้องทำเป็นหนังสือลงนามโดยคู่สัญญาทั้งสองฝ่าย ตามนัยมาตรา {rng.choice([193, 224, 369, 383, 387])} แห่งประมวลกฎหมายแพ่งและพาณิชย์"),
        ({"service", "construction", "employment"}, f"{B}ต้องปฏิบัติตามกฎหมายแรงงาน กฎหมายความปลอดภัย และระเบียบของ{A}อย่างเคร่งครัด"),
        ({"service", "employment", "license"}, f"ผลงานและทรัพย์สินทางปัญญาที่เกิดขึ้นจากการปฏิบัติงานตามสัญญานี้ให้ตกเป็นของ{A}แต่เพียงผู้เดียว"),
        ({"service", "construction"}, f"{A}มีสิทธิตรวจสอบความคืบหน้าของงานได้ตลอดเวลา และ{B}ต้องรายงานความก้าวหน้าทุก {rng.choice([7, 15, 30])} วัน"),
        ({"lease"}, f"เงินประกันการเช่าจะคืนให้{B}ภายใน {n1} วันนับจากวันส่งมอบทรัพย์สินที่เช่าคืน โดยหักค่าเสียหาย (ถ้ามี)"),
        ({"lease"}, f"{B}ตกลงชำระค่าสาธารณูปโภค ค่าน้ำ ค่าไฟฟ้า และค่าส่วนกลางเดือนละ {rng.randint(5, 60) * 100:,} บาท เอง"),
        ({"loan"}, f"{B}ตกลงผ่อนชำระเงินต้นพร้อมดอกเบี้ยเป็นรายเดือน เดือนละ {rng.randint(5, 200) * 1000:,} บาท จนกว่าจะครบถ้วน"),
        ({"loan"}, f"{B}ได้นำ{rng.choice(['ที่ดินโฉนดเลขที่ ' + str(rng.randint(1000, 99999)), 'รถยนต์ทะเบียน ' + rng.choice(['กข', 'ฌฒ', '1กก']) + ' ' + str(rng.randint(100, 9999)) + ' กรุงเทพมหานคร'])}มาเป็นหลักประกันการชำระหนี้ตามสัญญานี้"),
        ({"sale", "lease", "loan"}, f"ค่าธรรมเนียมการโอนและค่าอากรแสตมป์ให้{rng.choice([A, B, 'คู่สัญญาทั้งสองฝ่ายออกคนละครึ่ง'])}เป็นผู้รับผิดชอบ"),
        ({"employment"}, f"{B}จะได้รับสวัสดิการตามระเบียบของ{A} รวมถึงประกันสังคมและกองทุนสำรองเลี้ยงชีพในอัตราร้อยละ {rng.choice([3, 5])} ของค่าจ้าง"),
        ({"employment"}, f"ในระหว่างการจ้าง {B}ต้องไม่ประกอบกิจการอันเป็นการแข่งขันกับ{A} และภายใน {n2} ปีหลังพ้นสภาพการจ้าง"),
        ({"employment"}, f"{B}มีสิทธิลาพักผ่อนประจำปีได้ปีละ {rng.choice([6, 8, 10, 12, 15])} วันทำงาน โดยได้รับค่าจ้าง"),
        ({"nda"}, f"ข้อมูลอันเป็นความลับไม่รวมถึงข้อมูลที่เป็นที่รู้กันทั่วไป หรือข้อมูลที่{B}ได้รับมาโดยชอบจากบุคคลภายนอก"),
        ({"nda"}, f"เมื่อสิ้นสุดสัญญา {B}ต้องส่งคืนหรือทำลายเอกสารและข้อมูลทั้งหมดที่ได้รับจาก{A}ภายใน {n1} วัน"),
        ({"license", "distribution"}, f"{B}ต้องรายงานยอดขายให้{A}ทราบทุกไตรมาส และยินยอมให้{A}ตรวจสอบบัญชีที่เกี่ยวข้องได้"),
        ({"license", "distribution"}, f"{B}ตกลงชำระค่าสิทธิ (Royalty) ในอัตราร้อยละ {rng.choice([3, 5, 8, 10])} ของยอดขายสุทธิ"),
        (ANY, f"ผู้ประสานงานฝ่าย{A}: {_person_name(rng)} โทร. 0{rng.randint(2, 9)}{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)} ต่อ {rng.randint(100, 9999)}"),
        (ANY, f"ให้จัดงาน ณ ห้อง{rng.choice(['บอลรูม', 'ประชุมใหญ่', 'แกรนด์ฮอลล์'])} โรงแรม{rng.choice(['แกรนด์ ไฮแอท เอราวัณ', 'เซ็นทาราแกรนด์', 'แชงกรี-ลา', 'ดุสิตธานี', 'แมริออท'])} {rng.choice(L.PROVINCES)} ในวันที่ {fmt_thai_date(rng, ctx.start + timedelta(days=rng.randint(30, 200)))}"),
        (WORK, f"สถานที่ส่งมอบ: {rng.choice(['ศูนย์การค้าเซ็นทรัล', 'นิคมอุตสาหกรรมอมตะซิตี้', 'คลังสินค้าบางนา', 'ศูนย์ราชการฯ แจ้งวัฒนะ', 'ท่าเรือแหลมฉบัง'])} จังหวัด{rng.choice(L.PROVINCES)}"),
        ({"sale", "distribution", "license"}, f"สินค้าภายใต้ตราสินค้า “{rng.choice(['GreenLife', 'ฉ่ำ', 'Nova X', 'สยามเฟรช', 'ProMax', 'บ้านสวน'])}” รุ่น {rng.choice(['SR', 'TX', 'BM', 'KX'])}-{rng.randint(100, 999)}{rng.choice(['T', 'X', 'S', ''])} จำนวน {rng.randint(1, 500)} ชุด"),
        ({"sale", "lease", "loan"}, f"ให้ดำเนินการจดทะเบียน ณ สำนักงานที่ดินจังหวัด{rng.choice(L.PROVINCES[1:])} สาขา{rng.choice(L.BANGKOK_DISTRICTS)}"),
        ({"service", "employment", "construction"}, f"เวลาปฏิบัติงาน วัน{rng.choice(['จันทร์', 'อังคาร'])}ถึงวัน{rng.choice(['ศุกร์', 'เสาร์'])} เวลา {rng.choice(['08.00', '08.30', '09.00'])} – {rng.choice(['17.00', '17.30', '18.00'])} น. พักกลางวัน 1 ชั่วโมง"),
        (WORK, f"งานตามสัญญานี้เป็นส่วนหนึ่งของโครงการ{rng.choice(['ก่อสร้างประตูระบายน้ำคลองบางไผ่', 'พัฒนาระบบทะเบียนออนไลน์', 'ปรับปรุงภูมิทัศน์ริมธาร', 'บ้านสวนริมธาร', 'ถนนสายบ้านขอนตาล'])} ประจำปีงบประมาณ {_be(ctx.start)}"),
        (ANY, f"เริ่มงวดแรกในเดือน{rng.choice(L.THAI_MONTHS)} {_be(ctx.start)} และงวดสุดท้ายในเดือน{rng.choice(L.THAI_MONTHS)} {_be(ctx.start) + rng.choice([1, 2])}"),
        (ANY, f"หนังสือฉบับนี้อ้างถึงหนังสือของ{rng.choice(L.ORG_NAMES)} ที่ {rng.choice(['ศธ', 'กค', 'มท', 'สธ'])} {rng.randint(100, 999)}/{rng.randint(100, 9999)} ลงวันที่ {fmt_thai_date(rng, ctx.agreement - timedelta(days=rng.randint(5, 60)))}"),
    ]
    ok = [t for types, t in pool if types is None or k in types]
    return [(rng.choice(ok), None)]


def signature_block(ctx: Ctx) -> list[Segment]:
    rng = ctx.rng
    dots = rng.choice(["......................................", "________________________", "..............................."])
    segs: list[Segment] = []
    for p in (ctx.A, ctx.B):
        segs.append((f"ลงชื่อ {dots} {p.role}\n", None))
        if p.kind == "person":
            segs.append(("      (", None))
            segs.append((p.name, "PARTY"))
            segs.append((")\n", None))
        else:
            if rng.random() < 0.6:
                segs.append(("      ", None))
                segs.append((p.name, "PARTY"))
                segs.append((f"\n      โดย {p.signatory} {p.position}\n", None))
            else:
                segs.append((f"      ({p.signatory})\n      {p.position} ", None))
                segs.append((p.name, "PARTY"))
                segs.append(("\n", None))
    for _ in range(rng.choice([0, 1, 2, 2])):
        segs.append((f"ลงชื่อ {dots} พยาน\n      ({_person_name(rng)})\n", None))
    return segs


def body_mention(ctx: Ctx) -> list[Segment]:
    """An occasional sentence that names a party in the body of the contract."""
    rng = ctx.rng
    p = rng.choice([ctx.A, ctx.B])
    t = rng.choice([
        "{N} ในฐานะ{R} ตกลงปฏิบัติตามเงื่อนไขทั้งหมดในสัญญานี้",
        "ทั้งนี้ {N} รับรองว่ามีอำนาจโดยชอบในการเข้าทำสัญญาฉบับนี้",
        "{N} ({R}) รับทราบและยอมรับเงื่อนไขข้างต้นทุกประการ",
    ])
    return render(t, {"N": (p.name, "PARTY"), "R": (p.role, None)})


# ===================================================================== document assembly
_NUM_STYLES = ["ข้อ {n}. ", "ข้อ {n} ", "{n}. ", "ข้อที่ {n} ", "{n}) "]


_EN_ROLES = {
    "service": [("Client", "Contractor"), ("Client", "Service Provider"), ("Customer", "Supplier")],
    "construction": [("Employer", "Contractor"), ("Owner", "Contractor")],
    "sale": [("Buyer", "Seller"), ("Purchaser", "Vendor")],
    "lease": [("Lessor", "Lessee"), ("Landlord", "Tenant")],
    "loan": [("Lender", "Borrower")],
    "license": [("Licensor", "Licensee"), ("Franchisor", "Franchisee")],
    "nda": [("Discloser", "Recipient"), ("Disclosing Party", "Receiving Party")],
    "distribution": [("Principal", "Distributor"), ("Advertiser", "Agency"), ("Manufacturer", "Distributor")],
    "employment": [("Company", "Employee"), ("Employer", "Employee")],
}
_GENERIC_ALIAS = {"company": ["บริษัท", "ผู้ให้บริการ", "ลูกค้า"], "org": ["มหาวิทยาลัย", "กรม", "หน่วยงาน", "โรงพยาบาล", "เทศบาล", "สหกรณ์"],
                  "person": ["พนักงาน", "ผู้ซื้อ", "ผู้ขาย"]}


def _sample_ctx(rng: random.Random) -> Ctx:
    ctype = rng.choice(L.CONTRACT_TYPES)
    title = rng.choice(ctype["titles"])
    ra, rb = rng.choice(ctype["roles"])
    if rng.random() < 0.10:
        ra, rb = rng.choice(_EN_ROLES[ctype["key"]])
    ka, kb = _party_kinds(rng, ctype["key"])
    A, B = _make_party(rng, ka, ra), _make_party(rng, kb, rb)
    if A.name == B.name:
        B = _make_party(rng, kb, rb)
    for p in (A, B):  # sometimes the defined term is a generic noun rather than the role
        if rng.random() < 0.12:
            if p.kind == "org":
                p.role = next((a for a in _GENERIC_ALIAS["org"] if p.name.startswith(a)), "หน่วยงาน")
            else:
                p.role = rng.choice(_GENERIC_ALIAS[p.kind])
    agreement, start, end, months = _sample_dates(rng)
    value = _sample_amount(rng, ctype["key"])
    if ctype["key"] == "nda" and rng.random() < 0.7:
        value = None
    cur = "THB" if rng.random() < 0.94 else rng.choice(["USD", "USD", "EUR", "JPY"])
    ctx = Ctx(rng, ctype, title, A, B, agreement, start, end, months, value, cur)
    ctx.fields = {
        "party_a": A.name, "party_b": B.name, "agreement_date": agreement.isoformat(),
        "start_date": start.isoformat(), "end_date": end.isoformat() if end else None,
        "contract_value": value, "currency": cur,
    }
    return ctx


def generate_document(rng: random.Random, doc_id: str) -> Document:
    ctx = _sample_ctx(rng)
    d = _Doc()
    style = rng.random()
    if style < 0.25:
        _excerpt(ctx, d)
        source = "synthetic-excerpt"
    elif style < 0.35:
        _summary_sheet(ctx, d)
        source = "synthetic-summary"
    else:
        _full_contract(ctx, d)
        source = "synthetic-full"
    text = d.text()
    if rng.random() < 0.06:  # government-style documents with Thai numerals
        text = text.translate(L.THAI_DIGITS)
        ctx.thai_digits = True
        for s in d.spans:
            s.text = s.text.translate(L.THAI_DIGITS)
    meta = {"contract_type": ctx.ctype["key"], "thai_digits": ctx.thai_digits, **ctx.fields}
    return Document(id=doc_id, text=text, entities=d.spans, lang="th", source=source, meta=meta)


def _full_contract(ctx: Ctx, d: _Doc) -> None:
    rng = ctx.rng
    d.add(ctx.title)
    if rng.random() < 0.6:
        d.add(f"\nสัญญาเลขที่ {rng.randint(1, 400):03d}/{_be(ctx.agreement)}")
    d.add("\n\n")
    d.extend(preamble(ctx))
    if rng.random() < 0.6:
        d.add("\n\n" + rng.choice([
            "คู่สัญญาทั้งสองฝ่ายได้ตกลงกันมีข้อความดังต่อไปนี้",
            "ทั้งสองฝ่ายตกลงทำสัญญากันโดยมีข้อความดังต่อไปนี้",
            "โดยที่คู่สัญญามีความประสงค์จะทำสัญญากัน จึงตกลงกันดังนี้",
            "คู่สัญญาได้ตกลงกันดังมีข้อความต่อไปนี้",
        ]))
    d.add("\n\n")
    clauses: list = [clause_scope, clause_term]
    if ctx.value is not None:
        clauses.append(clause_value)
    if ctx.value is not None and rng.random() < 0.85:
        clauses.append(clause_payment)
    if rng.random() < 0.85:
        clauses.append(clause_penalty)
    n_filler = rng.randint(1, 5)
    fillers = [clause_filler] * n_filler
    if rng.random() < 0.25:
        fillers.append(body_mention)
    # keep scope first and jurisdiction near the end; shuffle the middle
    middle = clauses[1:] + fillers
    rng.shuffle(middle)
    ordered = [clauses[0]] + middle
    if rng.random() < 0.85:
        ordered.append(clause_jurisdiction)
    if rng.random() < 0.3:
        ordered.append(clause_filler)
    style = rng.choice(_NUM_STYLES)
    for i, fn in enumerate(ordered, 1):
        num = style.format(n=i)
        d.add(num)
        if rng.random() < 0.5:
            d.add(_clause_heading(rng, fn) )
        d.extend(fn(ctx))
        d.add("\n" if rng.random() < 0.6 else "\n\n")
    d.add("\n")
    d.extend(signature_block(ctx))


_HEADINGS = {
    "clause_scope": ["ขอบเขตของงาน ", "วัตถุประสงค์ ", "ข้อตกลง ", "รายละเอียดการจ้าง "],
    "clause_term": ["ระยะเวลาของสัญญา ", "อายุสัญญา ", "กำหนดเวลา ", "ระยะเวลา "],
    "clause_value": ["ค่าจ้างและมูลค่าสัญญา ", "ราคา ", "มูลค่าสัญญา ", "ค่าตอบแทน "],
    "clause_payment": ["การชำระเงิน ", "กำหนดชำระเงิน ", "เงื่อนไขการชำระเงิน ", "งวดการจ่ายเงิน "],
    "clause_penalty": ["ค่าปรับ ", "เบี้ยปรับ ", "การผิดสัญญาและค่าปรับ ", "ค่าปรับกรณีผิดนัด "],
    "clause_jurisdiction": ["กฎหมายที่ใช้บังคับและการระงับข้อพิพาท ", "ข้อพิพาท ", "เขตอำนาจศาล ", "การระงับข้อพิพาท "],
    "clause_filler": ["ข้อตกลงอื่น ๆ ", "เงื่อนไขทั่วไป ", "หน้าที่ของคู่สัญญา ", "การรับประกัน ", "การบอกเลิกสัญญา ", "การส่งมอบ ", "หลักประกัน "],
    "body_mention": ["คำรับรอง ", "ข้อตกลงเพิ่มเติม "],
}


def _clause_heading(rng: random.Random, fn) -> str:
    return rng.choice(_HEADINGS.get(fn.__name__, ["ข้อตกลง "]))


def _summary_sheet(ctx: Ctx, d: _Doc) -> None:
    """Key/value metadata sheet as found in contract registers and cover memos."""
    rng = ctx.rng
    sep = rng.choice([": ", " : ", ":\t"])
    d.add(rng.choice(["สรุปสาระสำคัญของสัญญา", "ข้อมูลสัญญา", "ใบสรุปสัญญา", ctx.title]) + "\n")
    if rng.random() < 0.7:
        d.add(f"ชื่อสัญญา{sep}{ctx.title}" + (f" ประจำปีงบประมาณ {_be(ctx.start)}" if rng.random() < 0.3 else "") + "\n")
    if rng.random() < 0.5:
        d.add(f"เลขที่สัญญา{sep}{rng.randint(1, 400)}/{_be(ctx.agreement)}\n")
    d.add(f"{rng.choice(['คู่สัญญาฝ่าย' + ctx.A.role, ctx.A.role, 'ฝ่ายที่หนึ่ง'])}{sep}")
    d.add(ctx.A.name, "PARTY")
    d.add("\n")
    d.add(f"{rng.choice(['คู่สัญญาฝ่าย' + ctx.B.role, ctx.B.role, 'ฝ่ายที่สอง'])}{sep}")
    d.add(ctx.B.name, "PARTY")
    d.add("\n")
    d.add(f"{rng.choice(['วันที่ลงนาม', 'วันที่ทำสัญญา', 'ลงวันที่', 'วันที่'])}{sep}")
    d.add(fmt_thai_date(rng, ctx.agreement), "AGREEMENT_DATE")
    d.add("\n")
    d.add(f"{rng.choice(['ระยะเวลาสัญญา', 'ระยะเวลา', 'อายุสัญญา'])}{sep}")
    d.add(fmt_thai_date(rng, ctx.start), "START_DATE")
    if ctx.end is not None:
        d.add(rng.choice([" – ", " ถึง ", " - ", " ถึงวันที่ "]))
        d.add(fmt_thai_date(rng, ctx.end), "END_DATE")
        if rng.random() < 0.5:
            d.add(f" ({_duration_text(rng, ctx.months)})")
    else:
        d.add(f" เป็นเวลา {_duration_text(rng, ctx.months)}")
    d.add("\n")
    if ctx.value is not None:
        span, trail = fmt_money(rng, ctx.value, ctx.currency)
        d.add(f"{rng.choice(['วงเงินตามสัญญา', 'มูลค่าสัญญา', 'วงเงิน', 'ราคา'])}{sep}")
        d.add(span, "CONTRACT_VALUE")
        d.add(trail)
        if rng.random() < 0.3:
            d.add(f" (เดือนละ {ctx.value // ctx.months:,} บาท)")
        d.add("\n")
        pd, _desc = payment_due_phrase(rng, ctx)
        d.add(f"{rng.choice(['กำหนดจ่ายเงิน', 'การชำระเงิน', 'กำหนดชำระ'])}{sep}" + rng.choice(["", "รายเดือน ", "งวดเดียว "]))
        d.add(*pd)
        d.add("\n")
    if rng.random() < 0.85:
        pen, _ = penalty_phrase(rng)
        d.add(f"{rng.choice(['อัตราค่าปรับ', 'ค่าปรับ', 'เบี้ยปรับ'])}{sep}")
        d.add(*pen)
        d.add(rng.choice(["", " กรณีส่งมอบล่าช้า", " กรณีผิดนัดชำระ"]) + "\n")
    if rng.random() < 0.85:
        d.add(f"{rng.choice(['เขตอำนาจศาล', 'ศาลที่มีเขตอำนาจ', 'การระงับข้อพิพาท'])}{sep}")
        d.add(rng.choice(L.COURTS + L.ARBITRATION), "JURISDICTION")
        d.add("\n")
    if rng.random() < 0.6:
        d.add(f"ผู้ประสานงาน{sep}{_person_name(rng)} โทร. 0{rng.randint(2, 9)}{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}\n")
    if rng.random() < 0.4:
        d.add(f"ผู้บันทึก{sep}{_person_name(rng)} ({rng.choice(['นิติกร', 'เจ้าหน้าที่พัสดุ', 'ผู้จัดการฝ่ายกฎหมาย'])})\n")


def _excerpt(ctx: Ctx, d: _Doc) -> None:
    """A short paragraph in the style of the deck's illustrative example."""
    rng = ctx.rng
    parts = []
    if rng.random() < 0.8:
        parts.append(preamble)
    pool = [clause_term]
    if ctx.value is not None:
        pool += [clause_value, clause_payment]
    pool += [clause_penalty, clause_jurisdiction, clause_filler]
    n = rng.randint(1, 4)
    chosen = rng.sample(pool, min(n, len(pool)))
    chosen.sort(key=lambda f: [clause_term, clause_value, clause_payment, clause_penalty, clause_filler, clause_jurisdiction].index(f))
    parts += chosen
    sep = rng.choice([" ", " ", "\n"])
    for i, fn in enumerate(parts):
        if i:
            d.add(sep)
        d.extend(fn(ctx))


def generate(n: int, seed: int = 42, prefix: str = "th-syn") -> list[Document]:
    rng = random.Random(seed)
    return [generate_document(rng, f"{prefix}-{i:05d}") for i in range(n)]
