"""The seeded dataset loads, and every answer-key span round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dataset import (
    RECALL_TYPES,
    DatasetError,
    DefectType,
    SeededDocument,
    load_dataset,
    load_document,
)
from reviewdesk.contracts import Profile, Severity, Span, Verdict
from tests.eval.helpers import TEXT, tiny_seeded


def test_ten_documents_load_and_spans_round_trip() -> None:
    docs = load_dataset()
    assert len(docs) >= 10
    assert len({d.id for d in docs}) == len(docs)
    for doc in docs:
        assert doc.profile in (Profile.OPINION, Profile.DESIGN_DOC)
        for defect in doc.defects:
            assert defect.located.text_of(doc.text) == defect.quote, (doc.id, defect.id)


def test_every_document_seeds_several_defect_types() -> None:
    for doc in load_dataset():
        types = {d.type for d in doc.defects if d.type in RECALL_TYPES}
        assert len(types) >= 4, doc.id
        assert doc.of_type(DefectType.WRONG_FACT), doc.id
        assert doc.of_type(DefectType.WEAK_ARGUMENT), doc.id


def test_dataset_covers_every_conflict_rule_and_genre() -> None:
    docs = load_dataset()
    rules = {d.rule for doc in docs for d in doc.defects if d.rule is not None}
    assert len(rules) == 4
    genres = {doc.genre.value for doc in docs}
    assert {"football_blog", "design_doc"} <= genres


def test_design_docs_do_not_seed_borrowed_sentences() -> None:
    """The design-doc profile runs no originality checker."""
    for doc in load_dataset():
        if doc.profile is Profile.DESIGN_DOC:
            assert not doc.of_type(DefectType.BORROWED_SENTENCE), doc.id


def test_type_specific_fields_and_defaults() -> None:
    for doc in load_dataset():
        for d in doc.of_type(DefectType.WEAK_ARGUMENT):
            assert d.known_rebuttal and len(d.known_rebuttal) > 40
        for d in doc.of_type(DefectType.BORROWED_SENTENCE):
            assert d.source_url and d.source_url.startswith("https://")
        for d in doc.of_type(DefectType.WRONG_FACT):
            assert d.expected_verdict is Verdict.WRONG
            assert d.expected_severity is Severity.FACTUAL_ERROR
        for d in doc.of_type(DefectType.UNSUPPORTED):
            assert d.expected_verdict is Verdict.UNSUPPORTED


def test_document_property_uses_seeded_id() -> None:
    seeded = tiny_seeded()
    doc = seeded.document
    assert doc.id == "tiny" and doc.text == TEXT and doc.word_count == len(TEXT.split())


def test_load_dataset_filters_by_id() -> None:
    docs = load_dataset(ids=["02-possession", "07-postgres-queue"])
    assert [d.id for d in docs] == ["02-possession", "07-postgres-queue"]
    with pytest.raises(DatasetError, match="unknown"):
        load_dataset(ids=["nope"])


def _base(**defect: object) -> dict[str, object]:
    return {
        "id": "x",
        "title": "t",
        "genre": "op_ed",
        "profile": "opinion",
        "text": "Water boils at 90 degrees. Water boils at 90 degrees.",
        "defects": [{"id": "d1", "type": "unsupported", **defect}],
    }


def test_missing_quote_is_rejected() -> None:
    with pytest.raises(ValueError, match="not found"):
        SeededDocument.model_validate(_base(quote="Ice is hot."))


def test_ambiguous_quote_needs_occurrence() -> None:
    with pytest.raises(ValueError, match="occurrence"):
        SeededDocument.model_validate(_base(quote="Water boils"))
    doc = SeededDocument.model_validate(_base(quote="Water boils", occurrence=2))
    assert doc.defects[0].located.start == doc.text.index("Water boils", 1)
    with pytest.raises(ValueError, match="occurrence 3"):
        SeededDocument.model_validate(_base(quote="Water boils", occurrence=3))


def test_explicit_span_must_cover_quote() -> None:
    ok = SeededDocument.model_validate(_base(quote="Water", span={"start": 0, "end": 5}))
    assert ok.defects[0].located == Span(start=0, end=5)
    with pytest.raises(ValueError, match="span does not cover"):
        SeededDocument.model_validate(_base(quote="Water", span={"start": 1, "end": 6}))


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ({"type": "weak_argument", "quote": "Water"}, "known_rebuttal"),
        ({"type": "borrowed_sentence", "quote": "Water"}, "source_url"),
        (
            {"type": "borrowed_sentence", "quote": "Water", "source_url": "ftp://x"},
            "http",
        ),
        ({"type": "conflict", "quote": "Water"}, "rule"),
        ({"type": "grammar", "quote": "Water", "rule": "same_span_merge"}, "rule"),
    ],
)
def test_type_specific_validation(defect: dict[str, object], message: str) -> None:
    raw = _base()
    raw["defects"] = [{"id": "d1", "occurrence": 1, **defect}]
    with pytest.raises(ValueError, match=message):
        SeededDocument.model_validate(raw)


def test_auto_profile_and_duplicate_ids_rejected() -> None:
    raw = _base(quote="Water", occurrence=1)
    raw["profile"] = "auto"
    with pytest.raises(ValueError, match="concrete profile"):
        SeededDocument.model_validate(raw)
    raw = _base(quote="Water", occurrence=1)
    raw["defects"] = [raw["defects"][0], raw["defects"][0]]  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate"):
        SeededDocument.model_validate(raw)


def test_load_document_reads_text_file_and_checks_id(tmp_path: Path) -> None:
    (tmp_path / "doc-a.md").write_text("Water boils at 90 degrees.\n", encoding="utf-8")
    key = {
        "title": "t",
        "genre": "op_ed",
        "profile": "opinion",
        "defects": [{"id": "d1", "type": "wrong_fact", "quote": "90 degrees"}],
    }
    (tmp_path / "doc-a.json").write_text(json.dumps(key), encoding="utf-8")
    doc = load_document(tmp_path / "doc-a.json")
    assert doc.id == "doc-a" and doc.defects[0].located.text_of(doc.text) == "90 degrees"

    (tmp_path / "doc-b.json").write_text(json.dumps({**key, "id": "other"}), encoding="utf-8")
    with pytest.raises(DatasetError, match="missing text file"):
        load_document(tmp_path / "doc-b.json")
    (tmp_path / "doc-b.md").write_text("Water boils at 90 degrees.\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="must match"):
        load_document(tmp_path / "doc-b.json")
    (tmp_path / "doc-c.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="cannot read"):
        load_document(tmp_path / "doc-c.json")
