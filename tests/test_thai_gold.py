from pathlib import Path

import pytest

from contract_ner.data.thai_gold import load_gold_dir, parse_markup
from contract_ner.schema import ENTITY_TYPES

GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "thai_gold"


def test_parse_markup_offsets():
    text, spans = parse_markup("ระหว่าง [[บริษัท ก จำกัด|PARTY]] กับ [[นาย ข|PARTY]] วันที่ [[1 มกราคม 2569|START_DATE]]")
    assert text == "ระหว่าง บริษัท ก จำกัด กับ นาย ข วันที่ 1 มกราคม 2569"
    assert [(text[s.start:s.end], s.label) for s in spans] == [
        ("บริษัท ก จำกัด", "PARTY"), ("นาย ข", "PARTY"), ("1 มกราคม 2569", "START_DATE")]


def test_parse_markup_rejects_bad_label():
    with pytest.raises(ValueError):
        parse_markup("[[x|NOPE]]")


def test_gold_set_loads_and_is_well_formed():
    docs = load_gold_dir(GOLD_DIR)
    assert len(docs) >= 20
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))
    labels = set()
    for d in docs:
        assert "[[" not in d.text and "]]" not in d.text
        for e in d.entities:
            assert d.text[e.start:e.end] == e.text
            labels.add(e.label)
    assert labels == set(ENTITY_TYPES)
