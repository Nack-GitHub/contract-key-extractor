"""Conditional random field baseline: feature extraction, sequence splitting and decoding."""
import pytest

from contract_ner.baseline_crf import bio_to_spans, document_to_sequences, word_features
from contract_ner.lang import rules_for
from contract_ner.schema import O_LABEL, Span, spans_to_bio

TEXT = ('This Agreement is made as of June 21, 1999 between Acme Holdings, Inc. and Beta Ltd. '
        'It is governed by the laws of New York.')


def _offsets():
    return rules_for("en").segment_words(TEXT)


def test_word_features_describe_shape_case_and_position():
    words = ["Acme", "Holdings", "Inc.", "1999"]
    feats = word_features(words, 0, position=0.01)
    assert feats["word.shape"] == "Xxxx"
    assert feats["word.is_title"] is True
    assert feats["doc.in_opening"] is True
    assert feats["+2:word.is_legal_suffix"] is True      # "Inc." two words to the right
    assert word_features(words, 3, position=0.5)["word.is_year"] is True
    assert word_features(words, 0, 0.0)["-1:boundary"] is True


def test_documents_are_split_into_sentence_sized_sequences():
    offsets = _offsets()
    sequences = document_to_sequences(TEXT, offsets, [O_LABEL] * len(offsets))
    assert len(sequences) == 2                                  # the text has two sentences
    assert sum(len(s.offsets) for s in sequences) == len(offsets)  # every word is kept exactly once
    assert all(len(s.features) == len(s.tags) == len(s.offsets) for s in sequences)


def test_sequences_carry_the_tags_of_their_words():
    offsets = _offsets()
    party = Span(TEXT.index("Acme"), TEXT.index("Acme") + len("Acme Holdings, Inc."), "PARTY")
    tags = spans_to_bio(offsets, [party])
    sequences = document_to_sequences(TEXT, offsets, tags)
    assert "B-PARTY" in sequences[0].tags


def test_bio_decoding_round_trips_a_span():
    offsets = _offsets()
    party = Span(TEXT.index("Acme"), TEXT.index("Acme") + len("Acme Holdings, Inc."), "PARTY")
    date = Span(TEXT.index("June"), TEXT.index("June") + len("June 21, 1999"), "AGREEMENT_DATE")
    tags = spans_to_bio(offsets, [date, party])
    decoded = bio_to_spans(TEXT, offsets, tags)
    assert [(s.label, s.text) for s in decoded] == [
        ("AGREEMENT_DATE", "June 21, 1999"), ("PARTY", "Acme Holdings, Inc.")]


def test_bio_decoding_closes_a_span_when_the_label_changes_without_an_outside_tag():
    offsets = [(0, 4), (5, 9), (10, 14)]
    text = "Acme Corp 1999"
    decoded = bio_to_spans(text, offsets, ["B-PARTY", "I-PARTY", "B-AGREEMENT_DATE"])
    assert [(s.label, s.text) for s in decoded] == [("PARTY", "Acme Corp"), ("AGREEMENT_DATE", "1999")]


def test_a_fitted_model_predicts_spans_it_was_trained_on(tmp_path):
    """Fitting on one document and predicting it back checks the whole path end to end."""
    pytest.importorskip("sklearn_crfsuite")
    pytest.importorskip("joblib")
    from contract_ner.baseline_crf import CrfExtractor, train_crf
    from contract_ner.schema import Document

    party = Span(TEXT.index("Acme"), TEXT.index("Acme") + len("Acme Holdings, Inc."), "PARTY")
    doc = Document(id="d1", text=TEXT, lang="en", entities=[party])
    train_crf([doc] * 5, tmp_path, entity_types=("PARTY",), name="crf-test", max_iterations=30,
              keep_empty_sequence_share=1.0)
    extractor = CrfExtractor(tmp_path)
    assert extractor.entity_types == ("PARTY",)
    spans = extractor.predict_with_confidence(TEXT)
    assert [s.text for s, _ in spans] == ["Acme Holdings, Inc."]
    assert 0.0 < spans[0][1] <= 1.0
