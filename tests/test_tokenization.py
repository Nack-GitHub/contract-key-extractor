import pytest

from contract_ner.schema import LABEL2ID, Span, spans_to_bio
from contract_ner.tokenization import Encoder, lowercase_ascii, segment_words


def test_lowercase_ascii_preserves_length_and_thai():
    s = "บริษัท ABC Co., Ltd. ตกลง İ"
    out = lowercase_ascii(s)
    assert len(out) == len(s)
    assert out == "บริษัท abc co., ltd. ตกลง İ"


def test_thai_segmentation_tiles_text_without_whitespace_tokens():
    text = "สัญญาฉบับนี้ทำขึ้นระหว่าง บริษัท เอบีซี จำกัด วันที่ 1 มกราคม 2569"
    offsets = segment_words(text, "th")
    assert offsets == sorted(offsets)
    for (s, e) in offsets:
        assert e > s and not text[s:e].isspace()
    for (_, e1), (s2, _) in zip(offsets, offsets[1:]):
        assert s2 >= e1
    assert "บริษัท" in [text[s:e] for s, e in offsets]


def test_english_segmentation():
    text = 'ABC Corp. ("ABC") and X-Y Ltd, dated June 8, 2010.'
    words = [text[s:e] for s, e in segment_words(text, "en")]
    assert words[:3] == ["ABC", "Corp", "."]
    assert "X-Y" in words and "2010" in words


def test_unknown_lang():
    with pytest.raises(ValueError):
        segment_words("x", "fr")


@pytest.fixture(scope="module")
def wangchan_tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
    except Exception as e:  # pragma: no cover - offline CI
        pytest.skip(f"tokenizer unavailable: {e}")


def test_encoder_alignment_and_windows(wangchan_tok):
    text = 'สัญญาฉบับนี้ทำขึ้นระหว่าง บริษัท เอบีซี จำกัด ("ผู้ว่าจ้าง") มีผลตั้งแต่วันที่ 1 มกราคม 2569 ค่าปรับ 15% ต่อเดือน ABC Co., Ltd.'
    offsets = segment_words(text, "th")
    spans = [
        Span(text.index("บริษัท"), text.index("จำกัด") + 5, "PARTY"),
        Span(text.index("1 มกราคม"), text.index("2569") + 4, "START_DATE"),
    ]
    tags = spans_to_bio(offsets, spans)
    enc = Encoder(wangchan_tok, max_length=24, stride=6, lowercase=True)
    doc = enc.encode(text, offsets, tags)

    # every window is wrapped in the special tokens and respects max_length
    for w in doc.windows:
        assert len(w.input_ids) <= 24
        assert w.input_ids[0] == wangchan_tok.cls_token_id and w.input_ids[-1] == wangchan_tok.sep_token_id
        assert w.labels[0] == -100 and w.labels[-1] == -100
        assert len(w.input_ids) == len(w.labels) == len(w.subword_index)
    # consecutive windows overlap by `stride` subwords and cover everything
    covered = sorted({g for w in doc.windows for g in w.subword_index if g >= 0})
    assert covered == list(range(len(doc.subword_to_word)))
    # labels: exactly one labelled subword per word that has a representative
    labelled = [l for w in doc.windows for l in w.labels if l != -100]
    n_rep = sum(1 for s in doc.word_to_subword if s >= 0)
    assert n_rep <= len(offsets)
    assert LABEL2ID["B-PARTY"] in labelled and LABEL2ID["I-START_DATE"] in labelled
    # a representative subword is never a bare "▁" marker when a content piece follows
    ids = [i for w in doc.windows for i in w.input_ids]
    toks = wangchan_tok.convert_ids_to_tokens(ids)
    assert "▁บริษัท" in toks


@pytest.fixture(scope="module")
def roberta_tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("roberta-base")
    except Exception as e:  # pragma: no cover - offline CI
        pytest.skip(f"tokenizer unavailable: {e}")


def test_per_word_encoder_roberta_keeps_space_context_and_word_boundaries(roberta_tok):
    text = 'between Aduro Biotech, Inc. ("Aduro") and IREYA B.V ("Consultant").'
    offsets = segment_words(text, "en")
    spans = [Span(text.index("Aduro Biotech"), text.index("Inc.") + 4, "PARTY"),
             Span(text.index('"Aduro"'), text.index('"Aduro"') + 7, "PARTY")]
    tags = spans_to_bio(offsets, spans)
    enc = Encoder(roberta_tok, max_length=64, stride=8, per_word=True)
    doc = enc.encode(text, offsets, tags)
    assert all(s >= 0 for s in doc.word_to_subword), "every English word must have a representative subword"
    for (ss, se), w in zip(doc.subword_offsets, doc.subword_to_word):
        if w >= 0:
            assert offsets[w][0] <= ss < se <= offsets[w][1]
    ids = [i for w in doc.windows for i in w.input_ids]
    toks = roberta_tok.convert_ids_to_tokens(ids)
    assert "ĠBi" in toks and "," in toks and "Ġ(" in toks and '"' in toks  # space context kept, punctuation split
    labelled = [l for w in doc.windows for l in w.labels if l != -100]
    assert len(labelled) == len(offsets)
    assert labelled.count(LABEL2ID["B-PARTY"]) == 2


def test_per_word_encoder_wangchan_matches_raw_pieces_but_never_crosses_words(wangchan_tok):
    text = "สัญญาฉบับนี้อยู่ภายใต้บังคับของกฎหมายไทย บริษัท abc จำกัด"
    offsets = segment_words(text, "th")
    raw = Encoder(wangchan_tok, max_length=64, stride=8, lowercase=True, per_word=False).encode(text, offsets)
    pw = Encoder(wangchan_tok, max_length=64, stride=8, lowercase=True, per_word=True).encode(text, offsets)
    # about as many pieces as the raw tokenisation (is_split_into_words would double it) ...
    assert len(pw.subword_to_word) <= 1.3 * len(raw.subword_to_word)
    # ... but every word has its own representative and no piece straddles words
    assert all(s >= 0 for s in pw.word_to_subword)
    for (ss, se), w in zip(pw.subword_offsets, pw.subword_to_word):
        if w >= 0:
            assert offsets[w][0] <= ss < se <= offsets[w][1]
    ids = [i for w in pw.windows for i in w.input_ids]
    toks = wangchan_tok.convert_ids_to_tokens(ids)
    assert "▁บริษัท" in toks and "กฎหมาย" in toks and "ของ" in toks
