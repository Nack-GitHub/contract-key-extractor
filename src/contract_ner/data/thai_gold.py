"""Hand-annotated Thai contracts in a light inline mark-up.

Format (one file, documents separated by ``=== <doc id> ===`` lines)::

    === th-gold-001 ===
    สัญญาฉบับนี้ทำขึ้นระหว่าง [[บริษัท เอบีซี จำกัด|PARTY]] ...

``[[surface text|LABEL]]`` marks an entity. The mark-up is stripped and
converted to character offsets, which is far less error-prone than typing
offsets by hand and keeps the gold set readable in code review.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schema import ENTITY_TYPES, Document, Span

_ENT_RE = re.compile(r"\[\[(.+?)\|([A-Z_]+)\]\]", re.DOTALL)
_HEADER_RE = re.compile(r"^=== *(\S+) *===\s*$", re.MULTILINE)


def parse_markup(marked: str) -> tuple[str, list[Span]]:
    """Strip ``[[text|LABEL]]`` mark-up, returning plain text and spans."""
    out: list[str] = []
    spans: list[Span] = []
    pos = 0
    plain_len = 0
    for m in _ENT_RE.finditer(marked):
        chunk = marked[pos : m.start()]
        out.append(chunk)
        plain_len += len(chunk)
        surface, label = m.group(1), m.group(2)
        if label not in ENTITY_TYPES:
            raise ValueError(f"unknown label {label!r} in mark-up near {surface!r}")
        spans.append(Span(plain_len, plain_len + len(surface), label, surface))
        out.append(surface)
        plain_len += len(surface)
        pos = m.end()
    tail = marked[pos:]
    out.append(tail)
    text = "".join(out)
    if "[[" in text or "]]" in text:
        raise ValueError("unbalanced mark-up left in text")
    return text, spans


def load_gold_file(path: str | Path, lang: str = "th", source: str = "gold") -> list[Document]:
    raw = Path(path).read_text(encoding="utf-8")
    parts = _HEADER_RE.split(raw)
    # parts = [preamble, id1, body1, id2, body2, ...]
    docs: list[Document] = []
    for i in range(1, len(parts), 2):
        doc_id, body = parts[i], parts[i + 1].strip("\n")
        if not body.strip():
            continue
        text, spans = parse_markup(body)
        docs.append(Document(id=doc_id, text=text, entities=spans, lang=lang, source=source))
    return docs


def load_gold_dir(path: str | Path, lang: str = "th") -> list[Document]:
    docs: list[Document] = []
    for f in sorted(Path(path).glob("*.txt")):
        docs.extend(load_gold_file(f, lang=lang, source=f"gold:{f.stem}"))
    return docs
