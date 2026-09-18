from contract_ner.data.cuad import align_contract, span_regex, split_documents, split_spans
from contract_ner.schema import Document, Span

CONTRACT = (
    "TELKOM CONTRACT No. 076C/01\n\nLICENCE AGREEMENT entered into on the 8th day of May 2014 by and between "
    "SYSTEMS APPLICATIONS PRODUCTS (PTY) LIMITED (“SAP AFRICA”) and TELKOM SOUTH AFRICA LIMITED (\"TELKOM\").\n"
    "This Agreement shall commence on the Effective Date, being 8th day of May 2014, and expire on 31 December 2019.\n"
    "... lots of text ... TELKOM shall pay ...\n"
    "This Agreement shall be governed by the laws of the Republic of South Africa."
)


def test_split_spans_strips_page_markers():
    assert split_spans("May 8, 2014 (Page 1)\n\n8th day of May 2014 (Page 2)") == ["May 8, 2014", "8th day of May 2014"]


def test_span_regex_is_quote_space_case_insensitive():
    pat = span_regex('Mount   Kowledge ("Marketing Affiliate")')
    assert pat.search('mount kowledge\n(“marketing   affiliate”)')


def test_align_contract_anchors_aliases_and_resolves_overlaps():
    clauses = [
        ("Parties", "SYSTEMS APPLICATIONS PRODUCTS (PTY) LIMITED (Page 1)\n\n\"SAP AFRICA\" (Page 1)\n\n"
                    "TELKOM SOUTH AFRICA LIMITED (Page 1)\n\n\"TELKOM\" (Page 1)"),
        ("Agreement Date", "8th day of May 2014 (Page 1)"),
        ("Effective Date", "8th day of May 2014 (Page 1)"),
        ("Expiration Date", "31 December 2019 (Page 1)"),
        ("Governing Law", "This Agreement shall be governed by the laws of the Republic of South Africa. (Page 9)"),
        ("Cap on Liability", "ignored category (Page 3)"),
    ]
    spans, tried, found = align_contract(CONTRACT, clauses)
    assert tried == 8 and found == 8
    by_label = {}
    for s in spans:
        by_label.setdefault(s.label, []).append(s.text)
    # the quoted alias "TELKOM" is matched next to the legal name (quotes are part of the CUAD span)
    telkom = [s for s in spans if s.text.strip('"“”') == "TELKOM"]
    assert len(telkom) == 1 and telkom[0].start > CONTRACT.index("TELKOM SOUTH AFRICA")
    assert "SYSTEMS APPLICATIONS PRODUCTS (PTY) LIMITED" in by_label["PARTY"]
    assert by_label["AGREEMENT_DATE"] == ["8th day of May 2014"]
    assert "START_DATE" not in by_label  # identical span → AGREEMENT_DATE wins
    assert by_label["END_DATE"] == ["31 December 2019"]
    assert by_label["JURISDICTION"][0].startswith("This Agreement shall be governed")
    starts = [s.start for s in spans]
    assert starts == sorted(starts)
    for a, b in zip(spans, spans[1:]):
        assert not a.overlaps(b)


def test_split_is_deterministic_and_disjoint():
    docs = [Document(f"cuad-{i:04d}", "x y z", [Span(0, 1, "PARTY")], lang="en") for i in range(300)]
    tr, va, te = split_documents(docs)
    assert len(tr) + len(va) + len(te) == 300
    assert 15 <= len(te) <= 45 and 15 <= len(va) <= 45
    tr2, _, _ = split_documents(docs)
    assert [d.id for d in tr] == [d.id for d in tr2]
