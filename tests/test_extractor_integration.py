"""End-to-end checks that run only when the corresponding trained model is present (``contract-ner train``)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "thai-wangchanberta"

_thai_model = pytest.mark.skipif(not (MODEL / "model.safetensors").exists() or importlib.util.find_spec("pythainlp") is None,
                                reason="trained Thai model or pythainlp not found")


@pytest.fixture(scope="module")
def extractor():
    from contract_ner.predict import Extractor
    return Extractor(MODEL)


@_thai_model
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


@_thai_model
def test_long_document_is_windowed_and_stitched(extractor):
    from contract_ner.data.thai_synthetic import generate
    doc = max(generate(30, seed=5), key=lambda d: len(d.text))
    enc = extractor.encoder.encode(doc.text, extractor.predict_words(doc.text).offsets)
    assert len(enc.windows) > 1
    spans = extractor.predict(doc.text)
    assert any(s.label == "PARTY" for s in spans)


EN_MODEL = ROOT / "models" / "english-cuad-roberta"


@pytest.mark.skipif(not (EN_MODEL / "model.safetensors").exists(), reason="trained English model not found")
def test_english_model_has_its_own_label_space():
    """The CUAD model is trained on 5 types → 11 BIO labels, recorded next to the weights."""
    from contract_ner.postprocess import assemble
    from contract_ner.predict import Extractor

    ex = Extractor(EN_MODEL)
    assert ex.lang == "en"
    assert ex.entity_types == ("PARTY", "AGREEMENT_DATE", "START_DATE", "END_DATE", "JURISDICTION")
    assert len(ex.model.config.id2label) == 11
    text = (ROOT / "examples" / "english_service_agreement.txt").read_text(encoding="utf-8")
    rec = assemble(text, ex.predict_with_confidence(text), lang="en")
    s = rec["summary"]
    assert s["Party_A"] == "Northwind Analytics, Inc." and s["Party_B"] == "Contoso Manufacturing Ltd."
    assert s["Agreement_Date"] == "2021-03-15" and s["End_Date"] == "2024-03-14"
    assert s["Jurisdiction"] == "State of New York"
