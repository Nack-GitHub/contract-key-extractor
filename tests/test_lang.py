import importlib
import subprocess
import sys

import pytest

from contract_ner.lang import guess_lang, rules_for
from contract_ner.postprocess import validate_span
from contract_ner.schema import Span


def test_core_modules_import_without_pythainlp():
    """The English/CUAD track must not need the Thai toolkit (checked in a fresh interpreter)."""
    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def block(name, *a, **k):\n"
        "    if name.split('.')[0] == 'pythainlp': raise ImportError('blocked')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = block\n"
        "import contract_ner.tokenization, contract_ner.postprocess, contract_ner.evaluate, contract_ner.predict\n"
        "from contract_ner.lang import rules_for\n"
        "assert rules_for('en').segment_words('Acme Corp.') == [(0, 4), (5, 9), (9, 10)]\n"
        "from contract_ner.postprocess import validate_span\n"
        "from contract_ner.schema import Span\n"
        "assert validate_span(Span(0, 5, 'JURISDICTION', 'laws of the State of New York'), 'en')\n"
        "assert 'pythainlp' not in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_rules_for_unknown_language():
    with pytest.raises(ValueError):
        rules_for("fr")


def test_guess_lang():
    assert guess_lang("บริษัท เอบีซี จำกัด") == "th"
    assert guess_lang("Acme Corp.") == "en"


def test_english_validation_is_keyword_based_not_alias_aware():
    """Validation only checks surface plausibility; defined-term aliases such as
    ``"Company"`` survive it and are demoted later, in ``group_parties``."""
    en = rules_for("en")
    assert en.validate_span(Span(0, 9, "PARTY", '"Company"'))
    assert en.validate_span(Span(0, 5, "JURISDICTION", "courts located in San Francisco, California"))
    assert not en.validate_span(Span(0, 5, "JURISDICTION", "San Francisco, California"))  # no court/law keyword
    assert not en.validate_span(Span(0, 5, "END_DATE", "December"))
    assert en.validate_span(Span(0, 5, "END_DATE", "five (5) years from the Effective Date"))
    assert validate_span(Span(0, 5, "PARTY", "the"), "en") is False  # shared stop-word check


def test_thai_rules_available_with_extra():
    pytest.importorskip("pythainlp")
    th = rules_for("th")
    assert th.romanize("กรุงเทพมหานคร")  # PLACE_EN lookup or transliteration, never empty
    words = [w for w in th.segment_words("ศาลแพ่งกรุงเทพใต้ ตามกฎหมายไทย")]
    assert (0, len("ศาลแพ่งกรุงเทพใต้")) in words  # domain dictionary keeps the court name whole
    assert importlib.import_module("contract_ner.lang.th").RULES is th
