"""HTTP interface for the demo: extraction, model comparison and the evaluation numbers.

``contract-ner serve`` starts it; the single-page front end in ``contract_ner/web/``
is served from the same process, so the demo is one command and needs no network.

* ``GET  /api/models``           – the demo models, whether their weights are present, their test-split slot F1
* ``GET  /api/samples``          – the curated demo contracts (``demo/samples/*.json``)
* ``GET  /api/samples/{id}``     – one contract: text, gold spans and the talking point that goes with it
* ``POST /api/extract``          – the record ``assemble()`` builds, for pasted text or a demo contract;
  for a demo contract with gold spans it also says which table cell is right (``gold_check``)
* ``POST /api/extract/file``     – the same for an uploaded UTF-8 text file
* ``POST /api/compare``          – one contract through several models, with the cells they disagree on
* ``GET  /api/metrics``          – the slot-level numbers from ``reports/`` and the iteration history

Every demo contract carries the records the models produced for it when it was
curated (``scripts/pick_demo_docs.py``). When a model cannot be loaded — weights
missing, Thai extra not installed — or the server runs with ``--cached-only``,
those records are returned instead and the response says ``"source": "cache"``,
so a presentation does not depend on the model loading on the day.
"""
# no ``from __future__ import annotations`` here: FastAPI reads the request models' annotations at
# run time, and the models are defined inside ``create_app`` where string annotations cannot resolve
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from .augment import map_labels
from .postprocess import assemble
from .schema import Document, Span
from .slots import SLOTS, gold_slots, names_match, pred_slots

MAX_TEXT_CHARS = 1_000_000
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
WEB_DIR = Path(__file__).parent / "web"


@dataclass(frozen=True)
class ModelEntry:
    """A model the demo can run, and the evaluation report its numbers come from."""

    id: str
    title: str
    path: str                   # model directory relative to the project root, or "regex"
    lang: str
    report: str                 # reports/<report>.json
    description: str
    confidence_floor: bool      # the 0.6 floor applies to the encoders only, as in the reports


DEMO_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry("roberta", "RoBERTa fine-tuned on CUAD", "models/english-cuad-roberta-label-aware-augmented", "en",
               "english-cuad-roberta-label-aware-augmented__test",
               "The shipped English model: roberta-base, label-aware recipe with date and party augmentation.", True),
    ModelEntry("crf", "Conditional random field", "models/english-cuad-crf", "en", "crf-baseline__test",
               "Sequence-labelling baseline over hand-designed features, trained on the same contracts.", False),
    ModelEntry("regex", "Pattern-only baseline", "regex", "en", "regex-baseline__test",
               "Preamble patterns and the first governing-law and term sentences; no training.", False),
    ModelEntry("thai", "WangchanBERTa (Thai)", "models/thai-wangchanberta", "th", "thai-wangchanberta__test_gold_blind",
               "Thai track: trained on synthetic contracts, scored on hand-written blind contracts.", True),
)

# How the slot F1 of the English model moved across iterations (README, "Results — English").
# v1–v5 were measured before the slot view existed, so they carry span-level numbers only.
ITERATIONS: tuple[dict, ...] = (
    {"name": "v1", "change": "raw-text subword encoding, no validation", "strict": 0.254, "relaxed": 0.607, "slot": None},
    {"name": "v3", "change": "per-word encoding", "strict": 0.514, "relaxed": 0.654, "slot": None},
    {"name": "v4", "change": "+ span validation", "strict": 0.584, "relaxed": 0.735, "slot": None},
    {"name": "v5", "change": "5-type label space", "strict": 0.619, "relaxed": 0.738, "slot": None},
    {"name": "v6", "change": "stricter English validation", "strict": 0.633, "relaxed": 0.754, "slot": 0.804},
    {"name": "v7", "change": "normalisation + selection rules, confidence ≥ 0.6 (same weights)",
     "strict": 0.652, "relaxed": 0.759, "slot": 0.840},
    {"name": "v8", "change": "recipe reshaped by the label distribution", "strict": 0.547, "relaxed": 0.715, "slot": 0.845},
    {"name": "final", "change": "+ date and party-capitalisation augmentation", "strict": 0.545, "relaxed": 0.720,
     "slot": 0.855},
)


class ModelUnavailable(RuntimeError):
    pass


class ModelRegistry:
    """Loads demo models on first use and keeps them; one load at a time."""

    def __init__(self, root: Path, entries: tuple[ModelEntry, ...] = DEMO_MODELS):
        self.root = root
        self.entries = {e.id: e for e in entries}
        self._loaded: dict[str, object] = {}
        self._lock = threading.Lock()

    def entry(self, model_id: str) -> ModelEntry:
        if model_id not in self.entries:
            raise KeyError(model_id)
        return self.entries[model_id]

    def missing_reason(self, model_id: str) -> str | None:
        """Why the model cannot run here, or ``None`` when it can."""
        e = self.entry(model_id)
        if e.path == "regex":
            return None
        if not (self.root / e.path).is_dir():
            return f"weights not found at {e.path}"
        if e.lang == "th":
            import importlib.util
            if importlib.util.find_spec("pythainlp") is None:
                return 'the Thai extra is not installed (pip install -e ".[thai]")'
        return None

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def label_map(self, model_id: str) -> dict:
        return getattr(self._loaded.get(model_id), "label_map", None) or {}

    def get(self, model_id: str):
        e = self.entry(model_id)
        if model_id in self._loaded:
            return self._loaded[model_id]
        reason = self.missing_reason(model_id)
        if reason:
            raise ModelUnavailable(reason)
        with self._lock:
            if model_id not in self._loaded:
                self._loaded[model_id] = _load(e, self.root)
        return self._loaded[model_id]


def _load(e: ModelEntry, root: Path):
    if e.path == "regex":
        from .baseline import RegexExtractor
        return RegexExtractor()
    path = root / e.path
    if (path / "crf.joblib").exists():
        from .baseline_crf import CrfExtractor
        return CrfExtractor(path)
    from .predict import Extractor
    # the floor is applied per request (``min_confidence``), so the model keeps every span
    return Extractor(path, min_confidence=0.0)


class SampleStore:
    """The curated demo contracts: one JSON file each, written by ``scripts/pick_demo_docs.py``."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._cache: dict[str, dict] = {}

    def _paths(self) -> list[Path]:
        return sorted(self.directory.glob("*.json")) if self.directory.is_dir() else []

    def get(self, sample_id: str) -> dict:
        if sample_id not in self._cache:
            path = self.directory / f"{sample_id}.json"
            if not path.is_file() or path.parent != self.directory:
                raise KeyError(sample_id)
            self._cache[sample_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[sample_id]

    def listing(self) -> list[dict]:
        out = []
        for path in self._paths():
            s = self.get(path.stem)
            out.append({
                "id": s["id"], "title": s["title"], "lang": s["lang"], "order": s.get("order", 99),
                "chars": len(s["text"]), "has_gold": bool(s.get("entities")), "source": s.get("source"),
                "talking_point": s.get("talking_point", ""), "default_model": s.get("default_model"),
                "cached_models": sorted(s.get("cached", {})),
            })
        return sorted(out, key=lambda s: (s["order"], s["id"]))


def run_model(extractor, text: str, min_confidence: float, floor: bool) -> dict:
    spans = extractor.predict_with_confidence(text)
    if floor:
        spans = [(sp, c) for sp, c in spans if c >= min_confidence]
    return assemble(text, spans, lang=extractor.lang)


def gold_check(rec: dict, doc: Document, label_map: dict | None = None) -> list[dict]:
    """Per table cell: the gold value(s), the predicted value and whether they agree.

    Uses the same keys as the slot-level evaluation (``contract_ner.slots``), so a
    cell marked right here is a cell the reports count as right.
    """
    doc = map_labels(doc, label_map)
    gold = gold_slots(doc)
    parties = [Span(e["start"], e["end"], e["label"], e["text"])
               for e in rec["entities"] if e["valid"] and e["label"] == "PARTY"]
    pred = pred_slots(rec, parties)
    out = []
    for slot in SLOTS:
        g, p = gold[slot], pred[slot]
        if slot == "Parties":
            unmatched = list(g)
            extra = []
            for name in p:
                hit = next((x for x in unmatched if names_match(name, x)), None)
                if hit is None:
                    extra.append(name)
                else:
                    unmatched.remove(hit)
            if not g and not p:
                verdict = "none"
            elif not unmatched and not extra:
                verdict = "correct"
            elif not g:
                verdict = "extra"
            else:
                verdict = "wrong" if p else "missed"
            out.append({"slot": slot, "gold": list(g), "pred": list(p), "verdict": verdict,
                        "missing": unmatched, "extra": extra})
            continue
        if not g and p is None:
            verdict = "none"
        elif g and p in g:
            verdict = "correct"
        elif not g:
            verdict = "extra"           # predicted where the gold has nothing: a model error or a gold gap
        elif p is None:
            verdict = "missed"
        else:
            verdict = "wrong"
        out.append({"slot": slot, "gold": sorted(g), "pred": p, "verdict": verdict})
    return out


def _report_numbers(reports_dir: Path, name: str) -> dict | None:
    path = reports_dir / f"{name}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    slots = data.get("slots", {})
    return {
        "report": f"reports/{name}.json",
        "n_docs": data.get("n_docs"),
        "strict": data["strict"]["f1"],
        "relaxed": data["relaxed"]["micro"]["f1"],
        "field": data["fields"]["micro"]["f1"],
        "slot": slots.get("micro", {}).get("f1"),
        "slot_precision": slots.get("micro", {}).get("precision"),
        "slot_recall": slots.get("micro", {}).get("recall"),
        "cells": {k: {"f1": v["f1"], "precision": v["precision"], "recall": v["recall"],
                      "docs_with_gold": v.get("docs_with_gold")}
                  for k, v in slots.items() if k != "micro"},
    }


def create_app(root: str | Path = ".", samples_dir: str | Path | None = None, reports_dir: str | Path | None = None,
               cached_only: bool = False, preload: tuple[str, ...] = ()):
    """Build the FastAPI application (imported lazily: FastAPI is the ``[demo]`` extra)."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    root = Path(root).resolve()
    registry = ModelRegistry(root)
    samples = SampleStore(Path(samples_dir).resolve() if samples_dir else root / "demo" / "samples")
    reports = Path(reports_dir).resolve() if reports_dir else root / "reports"

    class ExtractRequest(BaseModel):
        model: str = "roberta"
        text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
        sample_id: str | None = None
        min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    class CompareRequest(BaseModel):
        models: list[str] = Field(default_factory=lambda: ["regex", "crf", "roberta"], min_length=1, max_length=4)
        text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
        sample_id: str | None = None
        min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    app = FastAPI(title="Contract key-information extraction", version="0.1.0",
                  description="Demo API over the contract NER pipeline (see contract_ner.api).")

    def _sample(sample_id: str) -> dict:
        try:
            return samples.get(sample_id)
        except KeyError:
            raise HTTPException(404, f"unknown sample '{sample_id}'") from None

    def _entry(model_id: str) -> ModelEntry:
        try:
            return registry.entry(model_id)
        except KeyError:
            raise HTTPException(404, f"unknown model '{model_id}'") from None

    def _extract(model_id: str, text: str | None, sample_id: str | None, min_confidence: float) -> dict:
        e = _entry(model_id)
        sample = _sample(sample_id) if sample_id else None
        if sample is not None:
            text = sample["text"]
        if not text or not text.strip():
            raise HTTPException(422, "send 'text' or 'sample_id'")
        cached = (sample or {}).get("cached", {}).get(model_id)
        # the cached record was made at the default floor; any other floor needs the live model
        cache_fits = cached is not None and abs(min_confidence - 0.6) < 1e-9
        record, source, seconds, note = None, "live", None, None
        if cached_only and cache_fits:
            record, source, seconds = cached["record"], "cache", cached.get("seconds")
        else:
            try:
                extractor = registry.get(model_id)
                t0 = time.perf_counter()
                record = run_model(extractor, text, min_confidence, e.confidence_floor)
                seconds = round(time.perf_counter() - t0, 2)
            except ModelUnavailable as exc:
                if not cache_fits:
                    raise HTTPException(503, f"model '{model_id}' is unavailable: {exc}") from None
                record, source, seconds, note = cached["record"], "cache", cached.get("seconds"), str(exc)
        out = {"model": model_id, "lang": e.lang, "source": source, "seconds": seconds, "record": record}
        if note:
            out["note"] = note
        if sample is not None and sample.get("entities"):
            doc = Document.from_dict(sample)
            out["gold_check"] = gold_check(record, doc, registry.label_map(model_id))
        return out

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "cached_only": cached_only,
                "loaded": [m for m in registry.entries if registry.is_loaded(m)]}

    @app.get("/api/models")
    def models() -> list[dict]:
        out = []
        for e in registry.entries.values():
            reason = registry.missing_reason(e.id)
            out.append({"id": e.id, "title": e.title, "lang": e.lang, "description": e.description,
                        "available": reason is None, "unavailable_reason": reason,
                        "loaded": registry.is_loaded(e.id), "metrics": _report_numbers(reports, e.report)})
        return out

    @app.get("/api/samples")
    def sample_list() -> list[dict]:
        return samples.listing()

    @app.get("/api/samples/{sample_id}")
    def sample_detail(sample_id: str) -> dict:
        s = _sample(sample_id)
        return {k: v for k, v in s.items() if k != "cached"}

    @app.post("/api/extract")
    def extract(req: ExtractRequest) -> dict:
        return _extract(req.model, req.text, req.sample_id, req.min_confidence)

    @app.post("/api/extract/file")
    async def extract_file(file: Annotated[UploadFile, File()], model: Annotated[str, Form()] = "roberta",
                           min_confidence: Annotated[float, Form()] = 0.6) -> dict:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(415, "only UTF-8 plain-text contracts are accepted") from None
        out = _extract(model, text, None, min_confidence)
        out["text"] = text
        out["filename"] = file.filename
        return out

    @app.post("/api/compare")
    def compare(req: CompareRequest) -> dict:
        results = {}
        for model_id in req.models:
            try:
                r = _extract(model_id, req.text, req.sample_id, req.min_confidence)
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise
                results[model_id] = {"model": model_id, "error": exc.detail}
                continue
            results[model_id] = {"model": model_id, "source": r["source"], "seconds": r["seconds"],
                                 "summary": r["record"]["summary"], "gold_check": r.get("gold_check"),
                                 "slots": _slot_values(r["record"])}
        ran = [r for r in results.values() if "slots" in r]
        disagreements = [s for s in SLOTS if len({json.dumps(r["slots"][s], sort_keys=True) for r in ran}) > 1]
        return {"results": results, "disagreements": disagreements}

    @app.get("/api/metrics")
    def metrics() -> dict:
        rows = []
        for e in registry.entries.values():
            numbers = _report_numbers(reports, e.report)
            if numbers:
                rows.append({"id": e.id, "title": e.title, "lang": e.lang, **numbers})
        return {"models": rows, "iterations": list(ITERATIONS), "target": {"with_review": 0.80, "without_review": 0.90}}

    if WEB_DIR.is_dir():
        @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
        def index():
            return FileResponse(WEB_DIR / "index.html")

        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    for model_id in preload:
        try:
            registry.get(model_id)
        except (KeyError, ModelUnavailable) as exc:
            print(f"preload {model_id}: skipped ({exc})")
    return app


def _slot_values(rec: dict) -> dict:
    """The slot keys of a record, comparable across models (names and dates normalised)."""
    parties = [Span(e["start"], e["end"], e["label"], e["text"])
               for e in rec["entities"] if e["valid"] and e["label"] == "PARTY"]
    values = pred_slots(rec, parties)
    values["Parties"] = sorted(values["Parties"])
    return values
