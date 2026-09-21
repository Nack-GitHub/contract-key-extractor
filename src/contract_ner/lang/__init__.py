"""Language-specific rules behind one seam.

Everything in the pipeline that differs between languages — how text is cut
into words, which surface forms can be a value of an entity type, how place
names are romanised — lives in one :class:`LangRules` per language. The shared
modules (``tokenization``, ``postprocess``, ``evaluate``) only ever call
``rules_for(lang)``; they never import a language toolkit themselves, so the
English/CUAD track installs and runs without PyThaiNLP.

Adding a language = a new module here exposing ``RULES`` + an encoder config.
"""
from __future__ import annotations

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass

from ..schema import Span

SUPPORTED: tuple[str, ...] = ("en", "th")

_THAI_CHAR_RE = re.compile(r"[฀-๿]")


@dataclass(frozen=True)
class LangRules:
    """Callables the shared pipeline needs from a language."""

    lang: str
    segment_words: Callable[[str], list[tuple[int, int]]]
    """``text → [(start, end), …]`` character offsets of the words BIO tags live on."""
    validate_span: Callable[[Span], bool]
    """Language-specific surface check (keyword patterns); the shared generic checks run first."""
    romanize: Callable[[str], str]
    """Place/court name → Latin script (identity for Latin-script languages)."""


@functools.cache
def rules_for(lang: str) -> LangRules:
    """Resolve a language's rules, importing its toolkit only on first use."""
    if lang == "en":
        from .en import RULES
    elif lang == "th":
        from .th import RULES
    else:
        raise ValueError(f"unsupported lang {lang!r} (known: {', '.join(SUPPORTED)})")
    return RULES


def guess_lang(text: str) -> str:
    """Cheap script-based guess used only where no explicit ``lang`` is available."""
    return "th" if _THAI_CHAR_RE.search(text) else "en"
