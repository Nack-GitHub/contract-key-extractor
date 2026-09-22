"""Slot-level evaluation: one value per metadata-table cell, compared after normalisation."""
import random

from contract_ner.augment import map_labels, rewrite_dates
from contract_ner.evaluate import evaluate_documents
from contract_ner.schema import Document, Span
from contract_ner.slots import SlotScorer, end_date_key, gold_slots, jurisdiction_key, party_names

TEXT = (
    'This Agreement is entered into on 3/31/16 (the "Effective Date") by and between Acme Holdings, Inc. '
    '("Acme") and Beta Systems Ltd. ("Beta"). Acme Holdings, Inc. and Beta Systems Ltd. are referred to as the '
    '"Parties". The term of this Agreement shall be three (3) years from the Effective Date. '
    "THIS AGREEMENT SHALL BE GOVERNED BY THE LAWS OF THE STATE OF NEW YORK. "
    "The laws of the State of New York govern this Agreement."
)


def _span(needle, label, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = TEXT.index(needle, start + 1)
    return Span(start, start + len(needle), label, needle)


GOLD = Document(id="d1", text=TEXT, lang="en", entities=[
    _span("3/31/16", "AGREEMENT_DATE"),
    _span("Acme Holdings, Inc.", "PARTY"),
    _span('"Acme"', "PARTY"),
    _span("Beta Systems Ltd.", "PARTY"),
    _span('Acme Holdings, Inc. and Beta Systems Ltd. are referred to as the "Parties"', "PARTY"),
    _span("The term of this Agreement shall be three (3) years from the Effective Date.", "END_DATE"),
    _span("THIS AGREEMENT SHALL BE GOVERNED BY THE LAWS OF THE STATE OF NEW YORK.", "JURISDICTION"),
])


class _Stub:
    lang = "en"
    entity_types = ("PARTY", "AGREEMENT_DATE", "END_DATE", "JURISDICTION")

    def __init__(self, spans):
        self.spans = spans
        self.label_map = {"START_DATE": "AGREEMENT_DATE"}

    def predict_with_confidence(self, text):
        return [(sp, 0.9) for sp in self.spans]

    def predict(self, text):
        return list(self.spans)


def test_gold_slots_normalise_values_and_drop_aliases_and_boilerplate():
    g = gold_slots(GOLD)
    assert g["Parties"] == ["acme holdings inc", "beta systems ltd"]
    assert g["Agreement_Date"] == {"2016-03-31"}
    assert g["Start_Date"] == {"2016-03-31"}  # no dated effective-date clause → falls back to the agreement date
    assert g["End_Date"] == {"3 year"}
    assert g["Jurisdiction"] == {"new york"}


def test_keys_agree_across_surface_forms():
    assert end_date_key("shall continue for a period of 3 years") == end_date_key("for three (3) years thereafter")
    assert end_date_key("shall expire on December 31, 2020, unless") == "2020-12-31"
    assert jurisdiction_key("The laws of the State of New York govern this Agreement.") == "new york"
    assert jurisdiction_key("governed by the laws of the People's Republic of China") == jurisdiction_key(
        "GOVERNED BY THE LAWS OF THE PEOPLE'S REPUBLIC OF CHINA")
    assert jurisdiction_key("construed in accordance with English law") == "english"
    assert party_names([Span(0, 4, "PARTY", '"Co"'), Span(5, 15, "PARTY", "Acme Corp.")]) == ["acme corp"]


def test_slot_scoring_accepts_a_different_but_equivalent_mention():
    # a second, equally correct governing-law clause and a boundary-trimmed party name are right at slot level
    preds = [
        _span("3/31/16", "AGREEMENT_DATE"),
        _span("Acme Holdings, Inc.", "PARTY"),
        _span("Beta Systems Ltd", "PARTY"),
        _span("three (3) years from the Effective Date", "END_DATE"),
        _span("The laws of the State of New York govern this Agreement.", "JURISDICTION"),
    ]
    res = evaluate_documents(_Stub(preds), [GOLD])
    assert res.relaxed["JURISDICTION"]["recall"] == 0.0  # span level: the other clause was tagged
    assert res.slots["Jurisdiction"]["f1"] == 1.0  # slot level: same governing law
    assert res.slots["micro"]["f1"] == 1.0
    assert res.slot_errors == []


def test_slot_scoring_counts_wrong_and_missing_values():
    sc = SlotScorer()
    sc.add("d", {"Parties": ["acme corp"], "Agreement_Date": {"2020-01-01"}, "Start_Date": {"2020-01-01"},
                 "End_Date": set(), "Jurisdiction": {"delaware"}},
           {"Parties": ["acme corp", "zeta llc"], "Agreement_Date": "2020-02-01", "Start_Date": None,
            "End_Date": "2 year", "Jurisdiction": "delaware"})
    r = sc.result()
    assert (r["Parties"]["precision"], r["Parties"]["recall"]) == (0.5, 1.0)
    assert r["Agreement_Date"]["f1"] == 0.0  # wrong value: FP and FN
    assert r["Start_Date"]["recall"] == 0.0  # missing
    assert r["End_Date"]["precision"] == 0.0  # predicted where gold has nothing
    assert r["Jurisdiction"]["f1"] == 1.0
    assert {e["slot"] for e in sc.errors} == {"Parties", "Agreement_Date", "Start_Date", "End_Date"}


def test_label_map_folds_gold_and_date_rewrite_keeps_offsets_aligned():
    doc = Document(id="d2", text="Effective as of October 1, 2019 between Acme Corp. and Beta Inc.", lang="en", entities=[
        Span(16, 31, "START_DATE"), Span(40, 50, "PARTY"), Span(55, 64, "PARTY"),
    ])
    merged = map_labels(doc, {"START_DATE": "AGREEMENT_DATE"})
    assert [e.label for e in merged.entities] == ["AGREEMENT_DATE", "PARTY", "PARTY"]
    rng = random.Random(3)
    seen = set()
    for _ in range(20):
        r = rewrite_dates(merged, 1.0, rng)
        assert all(r.text[e.start:e.end] == e.text for e in r.entities)
        assert [e.text for e in r.entities][1:] == ["Acme Corp.", "Beta Inc."]
        seen.add(r.entities[0].text)
    assert len(seen) > 3 and any("/" in s for s in seen)
    # gold START_DATE counts as a hit for a model that only predicts AGREEMENT_DATE
    res = evaluate_documents(_Stub([Span(16, 31, "AGREEMENT_DATE")]), [doc], validate=False)
    assert res.relaxed["AGREEMENT_DATE"]["f1"] == 1.0


def test_clause_slots_ignore_the_punctuation_a_span_boundary_may_carry():
    """An end-of-term clause is the same value with or without its closing full stop."""
    from contract_ner.slots import end_date_key, jurisdiction_key, value_key

    clause = "unless earlier terminated under paragraph 7, this agreement shall continue indefinitely"
    assert end_date_key(clause) == end_date_key(clause + ".")
    assert value_key('"the term of this agreement,"') == value_key("the term of this agreement")
    # a clause that still normalises is unaffected
    assert end_date_key("shall terminate on May 31, 2015.") == "2015-05-31"
    assert jurisdiction_key("governed by the laws of the State of New York.") == "new york"
