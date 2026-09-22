from contract_ner.evaluate import evaluate_documents, format_result
from contract_ner.schema import ENTITY_TYPES, Document, Span


class _StubExtractor:
    """Predicts a fixed list of spans per document; scored over two entity types only."""

    lang = "en"
    entity_types = ("PARTY", "END_DATE")

    def __init__(self, preds):
        self.preds = preds

    def predict(self, text):
        return self.preds[text]


def test_evaluate_scores_only_the_models_types():
    text = "Acme Corp. and Beta Inc. dated June 8, 2010 expires December 31, 2012 under Delaware law."
    doc = Document(id="d1", text=text, lang="en", entities=[
        Span(text.index("Acme"), text.index("Corp.") + 5, "PARTY"),
        Span(text.index("Beta"), text.index("Inc.") + 4, "PARTY"),
        Span(text.index("June"), text.index("2010") + 4, "AGREEMENT_DATE"),   # not in the model's types
        Span(text.index("December"), text.index("2012") + 4, "END_DATE"),
    ])
    preds = {text: [
        Span(text.index("Acme"), text.index("Corp.") + 5, "PARTY"),
        Span(text.index("December"), text.index("2012") + 4, "END_DATE"),
        Span(text.index("June"), text.index("2010") + 4, "AGREEMENT_DATE"),   # ignored, not a false positive
    ]}
    res = evaluate_documents(_StubExtractor(preds), [doc], validate=False)

    assert res.types == ("PARTY", "END_DATE")
    assert set(res.relaxed) == {"PARTY", "END_DATE", "micro"}
    assert set(res.fields) == {"PARTY", "END_DATE", "micro"}
    assert res.relaxed["PARTY"]["support"] == 2 and res.relaxed["END_DATE"]["support"] == 1
    assert res.relaxed["micro"]["precision"] == 1.0  # the AGREEMENT_DATE prediction was not counted
    assert res.strict["recall"] == 2 / 3
    assert {e["label"] for e in res.errors} == {"PARTY"}  # only Beta Inc. is missed
    out = format_result(res, "stub")
    assert "AGREEMENT_DATE" not in out.split("field-level per type:")[1]
    assert res.to_dict()["types"] == ["PARTY", "END_DATE"]


def test_evaluate_defaults_to_all_types_for_legacy_models():
    class Legacy(_StubExtractor):
        entity_types = ENTITY_TYPES

    text = "Acme Corp."
    doc = Document(id="d", text=text, lang="en", entities=[Span(0, 10, "PARTY")])
    res = evaluate_documents(Legacy({text: [Span(0, 10, "PARTY")]}), [doc], validate=False)
    assert set(res.relaxed) == set(ENTITY_TYPES) | {"micro"}


def test_recall_is_reported_per_surface_form():
    """The gold spans are grouped by how they are written, so a miss can be attributed to a form."""
    text = ('Acme Corp. and BETA SYSTEMS INC. signed on June 8, 2010; a second contract on 7/7/98 '
            'expires December 31, 2012.')
    doc = Document(id="d1", text=text, lang="en", entities=[
        Span(text.index("Acme"), text.index("Corp.") + 5, "PARTY"),
        Span(text.index("BETA"), text.index("INC.") + 4, "PARTY"),
        Span(text.index("December"), text.index("2012") + 4, "END_DATE"),
    ])
    preds = {text: [
        Span(text.index("Acme"), text.index("Corp.") + 5, "PARTY"),      # title-case name: found
        Span(text.index("December"), text.index("2012") + 4, "END_DATE"),
    ]}                                                                    # the capitalised name is missed
    res = evaluate_documents(_StubExtractor(preds), [doc], validate=False)

    assert res.forms["PARTY"]["legal-name"] == {"found": 1, "missed": 0, "support": 1, "recall": 1.0}
    assert res.forms["PARTY"]["legal-name-caps"] == {"found": 0, "missed": 1, "support": 1, "recall": 0.0}
    assert res.forms["END_DATE"]["explicit-date"]["recall"] == 1.0
    assert [e["form"] for e in res.errors if e["kind"] == "FN"] == ["legal-name-caps"]
    assert "legal-name-caps" in format_result(res, "stub")
