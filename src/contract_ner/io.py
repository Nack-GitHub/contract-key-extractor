"""JSONL persistence for annotated documents."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schema import Document


def write_jsonl(docs: Iterable[Document], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document.from_dict(json.loads(line)))
    return docs
