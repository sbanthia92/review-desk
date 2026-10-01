"""The seeded dataset loads, and every answer-key span round-trips."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.dataset import (
    DATA_DIR,
    RECALL_TYPES,
    ConflictRule,
    DatasetError,
    DefectType,
    Genre,
    SeededDocument,
    load_dataset,
    load_document,
)
from reviewdesk.contracts import Profile, Severity, Span, Verdict
from tests.eval.helpers import TEXT, tiny_seeded

FACT_TYPES = (DefectType.WRONG_FACT, DefectType.UNSUPPORTED)
FIRST_LONG_DOC = 11
"""Documents from this number on are 300-900 words and carry research-loop tags."""


def _number(doc: SeededDocument) -> int:
    return int(doc.id.split("-", 1)[0])


def test_thirty_documents_load_and_spans_round_trip() -> None:
    docs = load_dataset()
    assert len(docs) >= 30
    assert len({d.id for d in docs}) == len(docs)
    assert sorted(_number(d) for d in docs) == list(range(1, len(docs) + 1))
    for doc in docs:
        assert doc.profile in (Profile.OPINION, Profile.DESIGN_DOC)
        assert (doc.profile is Profile.DESIGN_DOC) == (doc.genre is Genre.DESIGN_DOC), doc.id
        assert doc.text.startswith(f"# {doc.title}\n"), doc.id
        for defect in doc.defects:
            assert defect.located.text_of(doc.text) == defect.quote, (doc.id, defect.id)


def test_every_document_seeds_several_defect_types() -> None:
    for doc in load_dataset():
        types = {d.type for d in doc.defects if d.type in RECALL_TYPES}
        assert len(types) >= 4, doc.id
        assert doc.of_type(DefectType.WRONG_FACT), doc.id
        assert doc.of_type(DefectType.UNSUPPORTED), doc.id
        assert doc.of_type(DefectType.WEAK_ARGUMENT), doc.id
        assert doc.of_type(DefectType.GRAMMAR), doc.id
        assert doc.of_type(DefectType.CONFLICT), doc.id


def test_dataset_covers_every_conflict_rule_and_genre() -> None:
    docs = load_dataset()
    rules = Counter(d.rule for doc in docs for d in doc.defects if d.rule is not None)
    assert set(rules) == set(ConflictRule)
    assert min(rules.values()) >= 8
    genres = Counter(doc.genre for doc in docs)
    assert set(genres) == set(Genre)
    assert genres[Genre.FOOTBALL_BLOG] >= 10
    assert genres[Genre.OP_ED] >= 6
    assert genres[Genre.DESIGN_DOC] >= 9


def test_new_documents_mix_and_length() -> None:
    new = [d for d in load_dataset() if FIRST_LONG_DOC <= _number(d) <= 30]
    assert len(new) == 20
    genres = Counter(d.genre for d in new)
    assert genres[Genre.FOOTBALL_BLOG] + genres[Genre.TECH_BLOG] == 9
    assert genres[Genre.OP_ED] == 5
    assert genres[Genre.DESIGN_DOC] == 6
    for doc in new:
        assert 300 <= doc.document.word_count <= 900, doc.id
    lengths = [d.document.word_count for d in new]
    assert max(lengths) - min(lengths) >= 300  # varied lengths


def test_wrong_facts_record_the_true_value() -> None:
    for doc in load_dataset():
        for d in doc.of_type(DefectType.WRONG_FACT):
            assert d.correction and len(d.correction) > 10, (doc.id, d.id)


def test_recall_defects_do_not_overlap() -> None:
    """Overlapping defects would be merged or dropped by the conflict rules."""
    for doc in load_dataset():
        recall = doc.of_type(*RECALL_TYPES)
        for i, a in enumerate(recall):
            for b in recall[i + 1 :]:
                assert not a.located.overlaps(b.located), (doc.id, a.id, b.id)


def test_seeded_conflicts_are_well_formed() -> None:
    for doc in load_dataset():
        recall = doc.of_type(*RECALL_TYPES)
        for c in doc.of_type(DefectType.CONFLICT):
            inside = [d for d in recall if d.located.overlaps(c.located)]
            where = (doc.id, c.id)
            if c.rule is ConflictRule.COPYEDIT_YIELDS_TO_FACT:
                assert any(d.type in FACT_TYPES for d in inside), where
                assert not any(d.type is DefectType.GRAMMAR for d in inside), where
            elif c.rule is ConflictRule.EVIDENCE_FREE_REBUTTAL:
                assert [d.type for d in inside] == [DefectType.WEAK_ARGUMENT], where
                assert inside[0].located == c.located, where
            elif c.rule is ConflictRule.SAME_SPAN_MERGE:
                assert doc.profile is Profile.DESIGN_DOC, where
                assert [d.type for d in inside] == [DefectType.GRAMMAR], where
            else:
                assert c.rule is ConflictRule.REBUTTAL_ON_VERIFIED_FACT
                assert not inside, where


def test_multi_part_and_secondary_source_claims_are_seeded() -> None:
    """DESIGN.md: seed the claims where research loops should matter most."""
    new = [d for d in load_dataset() if _number(d) >= FIRST_LONG_DOC]
    for tag in ("[multi-part]", "[secondary-source]"):
        tagged = [doc.id for doc in new if any(tag in d.note for d in doc.defects)]
        assert len(tagged) >= 6, (tag, tagged)
    for doc in load_dataset():
        for d in doc.defects:
            if "[multi-part]" in d.note or "[secondary-source]" in d.note:
                fact = d.type in FACT_TYPES
                verified = d.rule is ConflictRule.REBUTTAL_ON_VERIFIED_FACT
                assert fact or verified, (doc.id, d.id)
                assert d.note.startswith("["), (doc.id, d.id)


def test_readme_lists_every_seeded_fact_for_hand_verification() -> None:
    readme = (DATA_DIR.parent / "README.md").read_text(encoding="utf-8")
    assert "## Facts to verify by hand" in readme
    facts = readme.split("## Facts to verify by hand", 1)[1]
    assert "from memory" in facts
    for doc in load_dataset():
        assert f"### {doc.id}\n" in facts, doc.id
        section = facts.split(f"### {doc.id}\n", 1)[1].split("\n### ", 1)[0]
        for d in doc.of_type(DefectType.WRONG_FACT):
            assert d.correction and d.correction.replace("|", "\\|") in section, (doc.id, d.id)
        for d in doc.of_type(DefectType.BORROWED_SENTENCE):
            assert d.source_url and d.source_url in section, (doc.id, d.id)
        for d in doc.defects:
            if d.rule is ConflictRule.REBUTTAL_ON_VERIFIED_FACT:
                assert d.quote in section, (doc.id, d.id)


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
