"""The demo HTTP API, exercised with the pattern-only model so no trained weights are needed."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from contract_ner.api import create_app, gold_check
from contract_ner.baseline import RegexExtractor
from contract_ner.postprocess import assemble
from contract_ner.schema import Document, Span
from contract_ner.slots import SlotScorer, gold_slots, pred_slots

CONTRACT = (
    'This Services Agreement is dated as of March 15, 2021, by and between Northwind Analytics, Inc., '
    'a Delaware corporation ("Northwind"), and Contoso Manufacturing Ltd. ("Customer").\n\n'
    "1. Term. This Agreement shall continue in effect for a period of three (3) years.\n\n"
    "2. Governing Law. This Agreement shall be governed by and construed in accordance with the laws "
    "of the State of New York.\n"
)


def _span(text: str, needle: str, label: str) -> dict:
    start = text.index(needle)
    return {"start": start, "end": start + len(needle), "label": label, "text": needle}


GOLD = [
    _span(CONTRACT, "March 15, 2021", "AGREEMENT_DATE"),
    _span(CONTRACT, "Northwind Analytics, Inc.", "PARTY"),
    _span(CONTRACT, "Contoso Manufacturing Ltd.", "PARTY"),
    _span(CONTRACT, "This Agreement shall continue in effect for a period of three (3) years", "END_DATE"),
    _span(CONTRACT, "This Agreement shall be governed by and construed in accordance with the laws of the State "
                    "of New York", "JURISDICTION"),
]


@pytest.fixture()
def project(tmp_path):
    """A project root with one demo contract and one report, and no model weights."""
    samples = tmp_path / "demo" / "samples"
    samples.mkdir(parents=True)
    rec = assemble(CONTRACT, RegexExtractor().predict_with_confidence(CONTRACT), lang="en")
    sample = {"id": "short", "order": 0, "title": "Short", "lang": "en", "source": "test", "talking_point": "",
              "default_model": "roberta", "text": CONTRACT, "entities": GOLD,
              "cached": {"roberta": {"record": rec, "seconds": 0.5}}}
    (samples / "short.json").write_text(json.dumps(sample), encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    cell = {"precision": 1.0, "recall": 0.5, "f1": 0.667, "support": 2, "docs_with_gold": 2}
    (reports / "regex-baseline__test.json").write_text(json.dumps({
        "n_docs": 2, "strict": {"f1": 0.5}, "relaxed": {"micro": {"f1": 0.6}}, "fields": {"micro": {"f1": 0.55}},
        "slots": {"Parties": cell, "micro": cell},
    }), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def client(project):
    return TestClient(create_app(root=project))


def test_models_report_availability_and_numbers(client):
    models = {m["id"]: m for m in client.get("/api/models").json()}
    assert models["regex"]["available"] and models["regex"]["metrics"]["slot"] == pytest.approx(0.667)
    assert not models["roberta"]["available"] and "weights not found" in models["roberta"]["unavailable_reason"]


def test_extract_pasted_text(client):
    r = client.post("/api/extract", json={"model": "regex", "text": CONTRACT})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "live" and "gold_check" not in body
    s = body["record"]["summary"]
    assert s["Party_A"] == "Northwind Analytics, Inc." and s["Agreement_Date"] == "2021-03-15"
    ent = body["record"]["entities"][0]
    assert CONTRACT[ent["start"]:ent["end"]] == ent["text"]  # offsets point into the text that was sent


def test_extract_sample_carries_the_gold_check(client):
    body = client.post("/api/extract", json={"model": "regex", "sample_id": "short"}).json()
    verdicts = {c["slot"]: c["verdict"] for c in body["gold_check"]}
    assert verdicts["Parties"] == "correct" and verdicts["Agreement_Date"] == "correct"
    assert verdicts["Jurisdiction"] == "correct"
    assert set(verdicts) == {"Parties", "Agreement_Date", "Start_Date", "End_Date", "Jurisdiction"}


def test_unavailable_model_falls_back_to_the_stored_record(client):
    body = client.post("/api/extract", json={"model": "roberta", "sample_id": "short"}).json()
    assert body["source"] == "cache" and "weights not found" in body["note"]
    # no stored record for pasted text, and none for a non-default confidence floor
    assert client.post("/api/extract", json={"model": "roberta", "text": CONTRACT}).status_code == 503
    r = client.post("/api/extract", json={"model": "roberta", "sample_id": "short", "min_confidence": 0.3})
    assert r.status_code == 503


def test_cached_only_never_loads_a_model(project):
    client = TestClient(create_app(root=project, cached_only=True))
    assert client.post("/api/extract", json={"model": "roberta", "sample_id": "short"}).json()["source"] == "cache"
    assert client.get("/api/health").json()["loaded"] == []


def test_bad_requests(client):
    assert client.post("/api/extract", json={"model": "nope", "text": "x"}).status_code == 404
    assert client.post("/api/extract", json={"model": "regex", "sample_id": "nope"}).status_code == 404
    assert client.post("/api/extract", json={"model": "regex"}).status_code == 422
    assert client.get("/api/samples/..%2Fsecret").status_code == 404


def test_upload_accepts_utf8_text_only(client):
    r = client.post("/api/extract/file", files={"file": ("c.txt", CONTRACT.encode(), "text/plain")},
                    data={"model": "regex"})
    assert r.status_code == 200 and r.json()["text"] == CONTRACT
    r = client.post("/api/extract/file", files={"file": ("c.bin", b"\xff\xfe\x00\x81", "application/octet-stream")},
                    data={"model": "regex"})
    assert r.status_code == 415


def test_compare_lists_cells_the_models_disagree_on(client):
    body = client.post("/api/compare", json={"models": ["regex", "roberta"], "sample_id": "short"}).json()
    assert set(body["results"]) == {"regex", "roberta"}
    assert body["results"]["roberta"]["source"] == "cache"
    assert body["disagreements"] == []  # the stored roberta record is the regex record here


def test_samples_and_metrics(client):
    listing = client.get("/api/samples").json()
    assert listing[0]["id"] == "short" and listing[0]["has_gold"] and listing[0]["cached_models"] == ["roberta"]
    detail = client.get("/api/samples/short").json()
    assert "cached" not in detail and detail["text"] == CONTRACT
    metrics = client.get("/api/metrics").json()
    assert [m["id"] for m in metrics["models"]] == ["regex"]
    assert metrics["iterations"][-1]["slot"] == pytest.approx(0.855)


def test_gold_check_agrees_with_the_slot_scorer():
    """A cell the demo marks right is a cell the evaluation counts as right."""
    doc = Document("d", CONTRACT, [Span(**g) for g in GOLD], lang="en")
    rec = assemble(CONTRACT, RegexExtractor().predict_with_confidence(CONTRACT), lang="en")
    rec["End_Date"] = None  # make one cell a miss
    check = {c["slot"]: c["verdict"] for c in gold_check(rec, doc)}
    scorer = SlotScorer()
    parties = [Span(e["start"], e["end"], e["label"], e["text"]) for e in rec["entities"]
               if e["valid"] and e["label"] == "PARTY"]
    scorer.add("d", gold_slots(doc), pred_slots(rec, parties))
    for slot, verdict in check.items():
        assert (verdict == "correct") == (scorer.tp[slot] > 0 and scorer.fp[slot] == 0 and scorer.fn[slot] == 0)
    assert check["End_Date"] == "missed"
