import pytest

from contract_ner.schema import (
    BIO_LABELS,
    ENTITY_TYPES,
    Document,
    Span,
    bio_to_spans,
    iter_entities,
    spans_to_bio,
)


def test_bio_label_space():
    assert BIO_LABELS[0] == "O"
    assert len(BIO_LABELS) == 1 + 2 * len(ENTITY_TYPES)
    assert "B-PARTY" in BIO_LABELS and "I-JURISDICTION" in BIO_LABELS


def test_span_validation():
    with pytest.raises(ValueError):
        Span(5, 5, "PARTY")
    with pytest.raises(ValueError):
        Span(0, 3, "NOT_A_LABEL")


def test_document_rejects_overlaps_and_fills_text():
    text = "บริษัท เอบีซี จำกัด ตกลง"
    d = Document("x", text, [Span(0, 19, "PARTY")])
    assert d.entities[0].text == "บริษัท เอบีซี จำกัด"
    with pytest.raises(ValueError):
        Document("y", text, [Span(0, 10, "PARTY"), Span(5, 12, "START_DATE")])
    with pytest.raises(ValueError):
        Document("z", text, [Span(0, 999, "PARTY")])


def test_spans_to_bio_and_back_roundtrip():
    text = "on 1 January 2026 ABC Ltd pays"
    words = [(0, 2), (3, 4), (5, 12), (13, 17), (18, 21), (22, 25), (26, 30)]
    spans = [Span(3, 17, "START_DATE"), Span(18, 25, "PARTY")]
    tags = spans_to_bio(words, spans)
    assert tags == ["O", "B-START_DATE", "I-START_DATE", "I-START_DATE", "B-PARTY", "I-PARTY", "O"]
    back = bio_to_spans(tags, words, text)
    assert [(s.start, s.end, s.label, s.text) for s in back] == [
        (3, 17, "START_DATE", "1 January 2026"), (18, 25, "PARTY", "ABC Ltd")]


def test_spans_to_bio_partial_word_overlap_marks_whole_word():
    # a span that starts in the middle of a word still tags that word
    words = [(0, 5), (6, 10)]
    tags = spans_to_bio(words, [Span(3, 10, "PARTY")])
    assert tags == ["B-PARTY", "I-PARTY"]


def test_bio_to_spans_tolerates_missing_B():
    words = [(0, 1), (2, 3), (4, 5), (6, 7)]
    spans = bio_to_spans(["I-PARTY", "I-PARTY", "O", "I-END_DATE"], words)
    assert [(s.start, s.end, s.label) for s in spans] == [(0, 3, "PARTY"), (6, 7, "END_DATE")]


def test_iter_entities():
    tags = ["B-PARTY", "I-PARTY", "O", "B-PARTY", "B-END_DATE", "I-END_DATE"]
    assert list(iter_entities(tags)) == [("PARTY", 0, 2), ("PARTY", 3, 4), ("END_DATE", 4, 6)]


def test_train_config_entity_types(tmp_path):
    from contract_ner.train import TrainConfig

    cfg = TrainConfig.load("configs/english_cuad.yaml")
    assert cfg.entity_types == ["PARTY", "AGREEMENT_DATE", "START_DATE", "END_DATE", "JURISDICTION"]
    assert len(cfg.types) == 5
    th = TrainConfig.load("configs/thai_wangchanberta.yaml")
    assert th.entity_types is None and len(th.types) == 8
    import pytest
    with pytest.raises(ValueError):
        TrainConfig(name="x", model_name="m", lang="en", train="t", val="v", output_dir="o",
                    entity_types=["PARTY", "NOPE"])
