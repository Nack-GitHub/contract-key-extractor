"""Regex-only English baseline: same interface as the neural Extractor, scored by the same evaluator."""
from contract_ner.baseline import RegexExtractor
from contract_ner.evaluate import evaluate_documents
from contract_ner.postprocess import assemble
from contract_ner.schema import Document, Span

PREAMBLE = (
    'MASTER SERVICES AGREEMENT\n\nThis Master Services Agreement (the "Agreement") is made and entered into as of '
    'March 15, 2021 (the "Effective Date"), by and between Northwind Analytics, Inc., a Delaware corporation '
    '("Provider"), and Contoso Manufacturing Ltd., a company organised under the laws of England ("Customer").\n\n'
    "WHEREAS Provider offers analytics services;\n\n"
    "1. Term. The term of this Agreement shall commence on the Effective Date and continue for three (3) years, "
    "unless earlier terminated.\n\n"
    "2. Governing Law. This Agreement shall be governed by and construed in accordance with the laws of the "
    "State of New York.\n"
)


def _by_label(spans):
    out = {}
    for sp in spans:
        out.setdefault(sp.label, []).append(sp.text)
    return out


def test_regex_baseline_finds_preamble_parties_dates_and_clauses():
    got = _by_label(RegexExtractor().predict(PREAMBLE))
    assert got["PARTY"] == ["Northwind Analytics, Inc.", '"Provider"', "Contoso Manufacturing Ltd.", '"Customer"']
    assert got["AGREEMENT_DATE"] == ["March 15, 2021"]
    assert "START_DATE" not in got  # same date as the agreement date → not reported twice
    assert got["END_DATE"][0].startswith("The term of this Agreement shall commence")
    assert got["JURISDICTION"][0].startswith("This Agreement shall be governed by")


def test_regex_baseline_skips_entity_descriptors_and_non_party_defined_terms():
    texts = [sp.text for sp in RegexExtractor().predict(PREAMBLE) if sp.label == "PARTY"]
    assert "Delaware corporation" not in texts
    assert '"Agreement"' not in texts and '"Effective Date"' not in texts


def test_regex_baseline_plugs_into_evaluate_and_assemble():
    ex = RegexExtractor()
    doc = Document(id="d1", text=PREAMBLE, entities=[
        Span(PREAMBLE.index("Northwind"), PREAMBLE.index("Northwind") + len("Northwind Analytics, Inc."), "PARTY",
             "Northwind Analytics, Inc."),
        Span(PREAMBLE.index("March 15"), PREAMBLE.index("March 15") + len("March 15, 2021"), "AGREEMENT_DATE",
             "March 15, 2021"),
    ])
    res = evaluate_documents(ex, [doc])
    assert res.relaxed["AGREEMENT_DATE"]["recall"] == 1.0
    assert res.relaxed["PARTY"]["recall"] == 1.0
    rec = assemble(PREAMBLE, ex.predict_with_confidence(PREAMBLE), lang=ex.lang)
    assert rec["summary"]["Party_A"] == "Northwind Analytics, Inc."
    assert rec["summary"]["Party_B"] == "Contoso Manufacturing Ltd."
    assert rec["summary"]["Agreement_Date"] == "2021-03-15"
