import collections

from contract_ner.data.thai_synthetic import fmt_money, fmt_thai_date, generate, render
from contract_ner.postprocess import normalize_date, normalize_money
from contract_ner.schema import ENTITY_TYPES
import random
from datetime import date


def test_render_placeholders_and_nesting():
    segs = render("A {X} B {Y}", {"X": ("x", "PARTY"), "Y": [("y1", None), ("y2", "END_DATE")]})
    assert segs == [("A ", None), ("x", "PARTY"), (" B ", None), ("y1", None), ("y2", "END_DATE")]


def test_generation_is_deterministic_and_consistent():
    a = generate(20, seed=123)
    b = generate(20, seed=123)
    assert [d.text for d in a] == [d.text for d in b]
    assert [d.text for d in generate(5, seed=124)] != [d.text for d in a[:5]]


def test_spans_match_text_and_labels_valid():
    docs = generate(150, seed=7)
    for d in docs:
        for e in d.entities:
            assert d.text[e.start:e.end] == e.text
            assert e.label in ENTITY_TYPES
            assert e.text.strip() == e.text, f"span has outer whitespace: {e.text!r}"
    counts = collections.Counter(e.label for d in docs for e in d.entities)
    for t in ENTITY_TYPES:
        assert counts[t] > 20, f"{t} under-represented: {counts[t]}"


def test_metadata_agrees_with_normalised_spans():
    """The generator's gold ISO dates / amounts must round-trip through the normalisers."""
    checked = 0
    for d in generate(200, seed=99):
        if d.meta.get("thai_digits"):
            continue
        for e in d.entities:
            if e.label == "START_DATE":
                assert normalize_date(e.text) == d.meta["start_date"], (e.text, d.meta["start_date"])
                checked += 1
            elif e.label == "AGREEMENT_DATE":
                assert normalize_date(e.text) == d.meta["agreement_date"], e.text
            elif e.label == "END_DATE" and d.meta["end_date"]:
                assert normalize_date(e.text) == d.meta["end_date"], e.text
            elif e.label == "CONTRACT_VALUE" and d.meta["currency"] == "THB":
                m = normalize_money(e.text)
                assert m and m["amount"] == d.meta["contract_value"], (e.text, d.meta["contract_value"])
    assert checked > 100


def test_thai_digit_documents_keep_offsets():
    docs = [d for d in generate(300, seed=3) if d.meta.get("thai_digits")]
    assert docs, "expected some Thai-numeral documents"
    import re
    assert any(re.search(r"[๐-๙]", d.text) for d in docs)
    for d in docs:
        assert not re.search(r"[0-9]", d.text), "ASCII digits must all be converted"
        for e in d.entities:
            assert d.text[e.start:e.end] == e.text


def test_date_and_money_formatters():
    rng = random.Random(0)
    d = date(2026, 1, 1)
    forms = {fmt_thai_date(rng, d) for _ in range(200)}
    assert len(forms) >= 6
    for f in forms:
        assert normalize_date(f) == "2026-01-01", f
    for _ in range(100):
        span, trail = fmt_money(rng, 2_500_000)
        assert normalize_money(span)["amount"] == 2_500_000, span
        assert trail == "" or trail.startswith((" ", "(")) or trail == " ถ้วน"
