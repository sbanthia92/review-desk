"""Seeded-error evaluation set and harness (T9).

- ``eval.dataset``: seeded documents with answer keys, and the loader.
- ``eval.metrics``: scoring one ``Report`` against its answer key.
- ``eval.judge``: pluggable LLM judge (precision, citations, rebuttals).
- ``eval.baseline``: the single-prompt baseline reviewer.
- ``eval.fake_pipeline``: ``FakeAgent``-based pipelines for offline runs.
- ``eval.harness``: runs pipelines over the dataset and aggregates results.
- ``eval.results``: result models and the printed results table.

CLI: ``uv run python -m eval --help``. See ``eval/README.md`` for the dataset
format and the matching rules.
"""
