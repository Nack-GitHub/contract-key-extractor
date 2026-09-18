"""Runs only when a trained Thai model is present (``contract-ner train``)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "thai-wangchanberta"

pytestmark = pytest.mark.skipif(not (MODEL / "model.safetensors").exists(), reason="trained Thai model not found")


@pytest.fixture(scope="module")
def extractor():
    from contract_ner.predict import Extractor
    return Extractor(MODEL)


def test_deck_example_end_to_end(extractor):
    from contract_ner.postprocess import assemble
    text = (ROOT / "examples" / "thai_service_contract.txt").read_text(encoding="utf-8")
    rec = assemble(text, extractor.predict_with_confidence(text), lang="th")
    s = rec["summary"]
    assert s["Party_A"] == "บริษัท เอบีซี จำกัด"
    assert s["Party_B"] == "บริษัท เอ็กซ์วายแซด จำกัด"
    assert s["Start_Date"] == "2026-01-01" and s["End_Date"] == "2026-12-31"
    assert s["Contract_Value"] == "2,500,000 THB"
    assert s["Penalty_Rate"] == "15% per month"
    assert s["Jurisdiction"] == "Bangkok Civil Court"


def test_long_document_is_windowed_and_stitched(extractor):
    from contract_ner.data.thai_synthetic import generate
    doc = max(generate(30, seed=5), key=lambda d: len(d.text))
    enc = extractor.encoder.encode(doc.text, extractor.predict_words(doc.text).offsets)
    assert len(enc.windows) > 1
    spans = extractor.predict(doc.text)
    assert any(s.label == "PARTY" for s in spans)
