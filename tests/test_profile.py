"""Label-distribution profile: surface-form buckets, per-split statistics and drift."""
import random

import pytest

from contract_ner.augment import drop_dateless_spans, prepare_training_documents
from contract_ner.lang import rules_for
from contract_ner.profile import compare_forms, profile_documents
from contract_ner.schema import Document, Span

TEXT = (
    'This Agreement is made as of June 21, 1999 by and between Acme Holdings, Inc. ("Acme") and '
    "BETA SYSTEMS LTD. The term of this Agreement shall continue for three (3) years. "
    "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."
)


def _span(needle, label):
    start = TEXT.index(needle)
    return Span(start, start + len(needle), label, needle)


def _doc(doc_id="d1", spans=None):
    return Document(id=doc_id, text=TEXT, lang="en", entities=spans or [
        _span("June 21, 1999", "AGREEMENT_DATE"),
        _span("Acme Holdings, Inc.", "PARTY"),
        _span('"Acme"', "PARTY"),
        _span("BETA SYSTEMS LTD.", "PARTY"),
        _span("The term of this Agreement shall continue for three (3) years.", "END_DATE"),
        _span("This Agreement shall be governed by and construed in accordance with the laws of the "
              "State of New York.", "JURISDICTION"),
    ])


def _bare(label, text):
    """A span carrying only a label and its surface string, for the form classifiers."""
    return Span(0, len(text), label, text)


def test_english_surface_forms_separate_the_kinds_of_string_a_label_holds():
    form = rules_for("en").span_form
    assert form(_bare("AGREEMENT_DATE", "June 21, 1999")) == "month-d-y"
    assert form(_bare("AGREEMENT_DATE", "7/7/98")) == "numeric"
    assert form(_bare("AGREEMENT_DATE", "September ___, 2019")) == "placeholder-blank"
    assert form(_bare("START_DATE", "shall become effective as of the date written above")) == "clause-no-date"
    assert form(_bare("PARTY", "Acme Holdings, Inc.")) == "legal-name"
    assert form(_bare("PARTY", "ACME HOLDINGS, INC.")) == "legal-name-caps"
    assert form(_bare("PARTY", '"Acme"')) == "quoted-alias"
    assert form(_bare("PARTY", 'Acme and Beta are referred to as the "Parties"')) == "parties-boilerplate"
    assert form(_bare("END_DATE", "continue for three (3) years")) == "duration"
    assert form(_bare("END_DATE", "shall terminate on May 31, 2015")) == "explicit-date"
    assert form(_bare("JURISDICTION", "governed by the laws of New York")) == "governed-by-clause"


def test_profile_counts_spans_shapes_and_forms():
    profile = profile_documents([_doc("a"), _doc("b")], name="train")
    assert profile.n_documents == 2
    party = profile.labels["PARTY"]
    assert party.n_spans == 6
    assert party.coverage == 1.0
    assert party.spans_per_document["3+"] == 2
    assert party.forms == {"legal-name": 2, "quoted-alias": 2, "legal-name-caps": 2}
    assert profile.labels["END_DATE"].words_median > party.words_median  # clause-level versus short span
    assert sum(profile.labels["PARTY"].position.values()) == pytest.approx(1.0)
    assert 0 < profile.entity_char_ratio < 1


def test_compare_forms_flags_what_training_under_represents():
    train = profile_documents([_doc("a")], name="train")
    numeric = TEXT.index("June 21, 1999")
    other = _doc("b", spans=[
        Span(numeric, numeric + 6, "AGREEMENT_DATE", "7/7/98"),
        Span(numeric + 7, numeric + 13, "AGREEMENT_DATE", "3-31-16"),
    ])
    test = profile_documents([other], name="test")
    drift = compare_forms(train, test)
    assert [(d.label, d.form, d.verdict) for d in drift] == [
        ("AGREEMENT_DATE", "numeric", "unseen in training"),
    ]


def test_dateless_spans_are_dropped_from_training_only_for_the_named_labels():
    clause = "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."
    doc = _doc("a", spans=[
        _span("June 21, 1999", "START_DATE"),
        Span(TEXT.index(clause), TEXT.index(clause) + len(clause), "START_DATE",
             "shall become effective as of the date written above"),
        _span("Acme Holdings, Inc.", "PARTY"),
    ])
    kept = drop_dateless_spans(doc, ["START_DATE"])
    assert [e.label for e in kept.entities] == ["START_DATE", "PARTY"]
    assert kept.entities[0].text == "June 21, 1999"
    assert drop_dateless_spans(doc, None).entities == doc.entities  # no labels named, nothing dropped


def test_prepare_training_documents_applies_the_drop():
    doc = _doc("a", spans=[Span(0, 4, "START_DATE", "This")])
    prepared = prepare_training_documents([doc], None, 0.0, random.Random(0), ["START_DATE"])
    assert prepared[0].entities == []


@pytest.mark.parametrize("label,text,expected", [
    ("PARTY", "บริษัท สยามเทค จำกัด", "company"),
    ("PARTY", "นายสมชาย ใจดี", "person"),
    ("AGREEMENT_DATE", "1 มกราคม พ.ศ. 2569", "buddhist-era/arabic-numerals"),
    ("AGREEMENT_DATE", "๑ มกราคม พ.ศ. ๒๕๖๙", "buddhist-era/thai-numerals"),
    ("PENALTY_RATE", "ร้อยละ 0.1 ต่อวัน", "percent-word"),
    ("PENALTY_RATE", "15% ต่อเดือน", "percent-sign"),
    ("CONTRACT_VALUE", "2.5 ล้านบาท", "amount-in-words"),
    ("PAYMENT_DUE", "ภายใน 30 วันนับจากวันที่ได้รับใบแจ้งหนี้", "within-n-days"),
    ("JURISDICTION", "ศาลแพ่งกรุงเทพใต้", "court"),
])
def test_thai_surface_forms(label, text, expected):
    pytest.importorskip("pythainlp")
    assert rules_for("th").span_form(_bare(label, text)) == expected


def test_window_diagnostics_report_what_the_window_settings_do(tmp_path):
    """A stub encoder is enough: the diagnostics only need windows and the alignment metadata."""
    from transformers import AutoTokenizer

    from contract_ner.profile import window_diagnostics
    from contract_ner.schema import build_bio_labels
    from contract_ner.tokenization import Encoder

    labels = build_bio_labels(["PARTY", "AGREEMENT_DATE", "END_DATE", "JURISDICTION"])
    try:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    except Exception:  # pragma: no cover - no cached tokenizer and no network
        pytest.skip("roberta-base tokenizer unavailable")
    encoder = Encoder(tokenizer, max_length=64, stride=16, label2id={l: i for i, l in enumerate(labels)})
    diagnostics = window_diagnostics([_doc("a")], encoder, "en")
    assert diagnostics.n_windows >= 1
    assert diagnostics.windows_with_entity >= 1
    assert diagnostics.spans_cut_by_a_boundary == 0  # the document is shorter than one window
    assert diagnostics.subwords_per_span["JURISDICTION"]["median"] > diagnostics.subwords_per_span["PARTY"]["median"]


def test_uppercase_party_names_keeps_every_offset():
    from contract_ner.augment import uppercase_party_names

    doc = _doc("a")
    augmented = uppercase_party_names(doc, 1.0, random.Random(0))
    assert len(augmented.text) == len(doc.text)
    for before, after in zip(doc.entities, augmented.entities):
        assert (before.start, before.end, before.label) == (after.start, after.end, after.label)
        assert augmented.text[after.start:after.end] == after.text
        if before.label == "PARTY":
            assert after.text == before.text.upper()
        else:
            assert after.text == before.text
