# Review Desk eval

Seeded-error evaluation set and harness (see "Evaluation" in `DESIGN.md`).

```
uv run python -m eval validate                     # check every answer-key span
uv run python -m eval list                         # documents and defect counts
uv run python -m eval run -p oracle -p noisy --repeats 3
uv run python -m eval run -p baseline -p full=reviewdesk.pipeline:run_review \
    --llm mypkg.llm:make_client --judge --repeats 3
```

`run` prints a markdown table (one row per metric, one column per pipeline,
mean ± stdev across repeats) and writes `eval/results/<stamp>.json` and `.md`
(gitignored). Pipelines are built-in names (`oracle`, `oracle-no-rules`,
`noisy`, `empty`, `baseline`) or `NAME=module:attr` for any callable of the
`run_review` shape. `--llm module:attr` is a zero-argument factory returning an
`LLMClient`; it powers the baseline and, with `--judge`, the LLM judge.

## Dataset format

Each seeded document is two files in `eval/data/`: `<id>.md` (the text,
original and invented) and `<id>.json` (the answer key):

```json
{
  "id": "01-leicester-money",
  "title": "...",
  "genre": "football_blog | op_ed | tech_blog | design_doc",
  "profile": "opinion | design_doc",
  "defects": [
    {"id": "d1", "type": "wrong_fact", "quote": "exact text", "correction": "the true fact"},
    {"id": "d2", "type": "unsupported", "quote": "exact text"},
    {"id": "d3", "type": "weak_argument", "quote": "exact text", "known_rebuttal": "..."},
    {"id": "d4", "type": "borrowed_sentence", "quote": "exact text", "source_url": "https://..."},
    {"id": "d5", "type": "grammar", "quote": "exact text", "correction": "fixed text"},
    {"id": "c1", "type": "conflict", "rule": "copyedit_yields_to_fact", "quote": "exact text"}
  ]
}
```

- `quote` must occur exactly in the text; add `"occurrence": n` (1-based) if
  it occurs more than once, or an explicit `"span": {"start", "end"}` that
  must slice to the quote. Spans are validated on load.
- `expected_severity` and `expected_verdict` default per type (wrong fact →
  `factual_error` / `wrong`, unsupported → `unsupported` / `unsupported`).
- Conflict rules: `copyedit_yields_to_fact`, `rebuttal_on_verified_fact`,
  `same_span_merge`, `evidence_free_rebuttal`.
- Design docs seed no borrowed sentences (that profile runs no originality
  checker).
- Borrowed sentences come from public-domain texts on Project Gutenberg.

## Matching rules

A report is normalised into items: findings from `must_fix`, `should_fix`,
`polish` and `originality`, and rebuttals from `counter_case` plus the ledger
(spans = the targeted claims' spans). Each item has roles from its severity and
from every contributing agent (`agent` + `merged_from`); `heuristic` adds
`originality`.

An item **matches** a defect when:

1. its roles include one the defect type accepts: `wrong_fact` → wrong;
   `unsupported` → unsupported or wrong; `weak_argument` → rebuttal;
   `borrowed_sentence` → originality; `grammar` → style; and
2. one of its spans overlaps the defect span by at least one character and is
   no longer than `max(4 × defect length, defect length + 80)` characters (no
   catch-all spans).

## Metrics

Pooled over documents per run, then mean ± stdev across repeats.

| Metric | Definition |
| --- | --- |
| Recall per type | caught defects / seeded defects of that type |
| Precision (matched + judge) | (matched items + unmatched items the judge calls real) / (matched + judged) |
| Precision (matched only) | matched items / all items (lower bound, no judge) |
| Citation validity | cited (url, excerpt) pairs the judge says support the statement / judged pairs; malformed ones count as unsupported |
| Citations well-formed | non-empty excerpt and absolute http(s) URL |
| Rebuttal strength | judge score 1–5 of the best matched rebuttal (up to 3 tried) vs the known strongest rebuttal |
| Conflict accuracy | resolved / (resolved + violated); `not_triggered` conflicts are excluded and reported as "Conflicts triggered" |
| Cost and latency | tokens, search and LLM calls from `Report.usage`; wall-clock seconds measured by the harness |

Conflict checks:

- `copyedit_yields_to_fact`: triggered if a fact finding overlaps the span or
  an overlapping claim is wrong/unsupported; violated by any standalone copy
  edit on the span (merged into the fact finding is fine).
- `rebuttal_on_verified_fact`: triggered if an overlapping claim is verified;
  violated by any rebuttal with `target == fact` against it.
- `same_span_merge`: triggered if findings on the span come from two or more
  agents; resolved only if one merged finding carries them all.
- `evidence_free_rebuttal`: triggered by a devil's-advocate finding (in a
  section or `ledger.findings`) on the span or its claims with no evidence;
  violated if its severity is above `consider`.
