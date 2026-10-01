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
- Every `wrong_fact` carries a `correction` with the true value.
- `note` is free text. Two tags at the start of a note mark the cases where
  research loops should matter most (see "Evaluation" in `DESIGN.md`):
  - `[multi-part]`: the sentence holds two or more facts and only one is
    wrong. Where the true half stands on its own it is also seeded as a
    `rebuttal_on_verified_fact` conflict, so a pipeline that splits the claim
    gets one `wrong` and one `verified` verdict.
  - `[secondary-source]`: the top search results for the claim are likely to
    be secondary (blog posts, explainers, news retellings), and the note names
    the primary source (an RFC, a paper, official documentation, the laws of
    the game) that a checker should follow through to.

## The seeded set

30 documents: 11 football blog posts, 4 tech blog posts, 6 op-eds and 9 design
docs. Documents 01–10 are short (130–210 words); documents 11–30 run from 338
to 745 words. All text is original and invented for this dataset.

| Defect type | Seeded |
| --- | --- |
| `wrong_fact` | 61 |
| `unsupported` | 34 |
| `weak_argument` | 30 |
| `borrowed_sentence` | 20 |
| `grammar` | 90 |
| `conflict` | 64 |

Conflicts by rule: `copyedit_yields_to_fact` 24, `rebuttal_on_verified_fact`
22, `same_span_merge` 8, `evidence_free_rebuttal` 10. Among documents 11–30,
all 20 seed at least one `[multi-part]` claim and 13 seed at least one
`[secondary-source]` claim. `python -m eval list` prints the per-document
counts.

Rules the answer keys follow, so that the oracle pipeline scores 1.0 and real
pipelines are not penalised for following the conflict rules (checked in
`tests/eval/test_dataset.py`):

- Recall defects never overlap each other, and a `grammar` defect never
  overlaps a `wrong_fact` or `unsupported` span (the copy edit would be
  dropped by the conflict rules).
- A `copyedit_yields_to_fact` conflict overlaps a `wrong_fact` or
  `unsupported` defect and contains a deliberate slip that is *not* listed as
  a `grammar` defect.
- An `evidence_free_rebuttal` conflict has exactly the span of a
  `weak_argument`.
- A `same_span_merge` conflict appears only in design docs (copy editor plus
  structure reviewer) and contains exactly one `grammar` defect.
- A `rebuttal_on_verified_fact` conflict overlaps no other defect.

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

## Facts to verify by hand

**Every fact below was written from memory, with no network access, and has
not been checked against a source.** Before any live run (real search, real
fact-checker), verify each row against a primary or authoritative source and
fix the answer key if a "true value" is wrong or out of date. A wrong key
silently punishes a correct pipeline.

What to check, per document:

- each seeded wrong fact: the seeded text is false and the "true value" is
  right;
- each fact seeded as true (the `rebuttal_on_verified_fact` conflicts): it
  must really be true, or the conflict tests the wrong thing;
- each borrowed sentence: the wording matches the public-domain text exactly
  and the Project Gutenberg ebook number is the right book;
- the `known_rebuttal` texts in the answer keys also cite facts from memory
  (for example recent tournament results, project overruns and standards
  requirements); skim them before using the rebuttal-strength judge;
- the unseeded background facts in each document (dates, names and scores
  mentioned in passing) were written to be true; if a live run flags one as
  wrong, check it and either fix the text or add it to the key.

Lower-confidence items worth checking first: the exact wording and Gutenberg
numbers of the borrowed sentences in documents 11–24 (especially 1404, 2944,
34901 and 12); the Fisher space pen correction in `24`; the "six minutes" for
Liverpool's three goals in `12`; the 2,509 figure for Carnegie libraries in
`23`; the Kafka default retention in `26`; and the date-sensitive claims, which
were true as far as known at the time of writing but can change (S3's
consistency and durability wording in `28`, Germany's reactor shutdown in
`20`).

Known wart in the original ten: in `04-var`, `d1` quotes only a short clause
of a long sentence, so a finding that spans the whole sentence is too long to
match it under the matching rules. It was left unchanged to keep earlier
results comparable.

### 01-leicester-money

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Claudio Ranieri's side begun the season as 500-1 outsiders with the bookmakers. | Leicester were widely quoted at 5000-1 to win the 2015-16 Premier League before the season. |

Seeded as true (must hold): "Leicester City won the 2015-16 Premier League title".

Borrowed: "If you know the enemy and know yourself, you need not fear the result of a hundred battles." from The Art of War by Sun Tzu, translated by Lionel Giles (<https://www.gutenberg.org/ebooks/132>).

### 02-possession

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Spain won Euro 2008, the 2010 World Cup and Euro 2016 | Spain won Euro 2008, the 2010 World Cup and Euro 2012; Portugal won Euro 2016. |

Borrowed: "Hence to fight and conquer in all your battles is not supreme excellence; supreme excellence consists in breaking the enemy's resistance without fighting." from The Art of War by Sun Tzu, translated by Lionel Giles (<https://www.gutenberg.org/ebooks/132>).

### 03-invincibles

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d2 | Wenger had arrived from Japanese side Nagoya Grampus in 1990 | Wenger joined Arsenal from Nagoya Grampus in 1996. |
| d1 | Arsenal went unbeaten for all 38 league games in 2004-05 | Arsenal's unbeaten Premier League season was 2003-04. |

Seeded as true (must hold): "They won 26 games and drew 12".

### 04-var

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | the first tournament in Brazil in 1930 | The first World Cup was held in Uruguay in 1930. |
| d2 | the Premier League introduced it in the 2016-17 season | The Premier League introduced VAR in the 2019-20 season. |

Seeded as true (must hold): "VAR was used at the 2018 World Cup in Russia".

Borrowed: "The mass of men lead lives of quiet desperation." from Walden, and On The Duty Of Civil Disobedience by Henry David Thoreau (<https://www.gutenberg.org/ebooks/205>).

### 05-four-day-week

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Henry Ford adopted the five-day week for his factory workers in 1946 | Ford Motor Company adopted the five-day, 40-hour week in 1926. |

Borrowed: "It is not from the benevolence of the butcher, the brewer, or the baker, that we expect our dinner, but from their regard to their own interest." from An Inquiry into the Nature and Causes of the Wealth of Nations by Adam Smith (<https://www.gutenberg.org/ebooks/3300>).

### 06-sqlite-production

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | it was first released in 2010 | SQLite was first released in 2000. |

Seeded as true (must hold): "SQLite was created by D. Richard Hipp".

Borrowed: "A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts." from The Elements of Style by William Strunk Jr. (<https://www.gutenberg.org/ebooks/37134>).

### 07-postgres-queue

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | PostgreSQL only added SKIP LOCKED in version 12 | SKIP LOCKED was added in PostgreSQL 9.5 (released January 2016). |

### 08-uuid-keys

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | UUIDv7 was standardized in 2019 by RFC 9562. | RFC 9562, which defines UUIDv7, was published in May 2024. |

Seeded as true (must hold): "UUIDs are 128 bits long".

### 09-cdn-cache

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Cache-Control was first introduced in HTTP/2 | Cache-Control was introduced with HTTP/1.1 (RFC 2068, 1997). |

### 10-best-league

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Alan Shearer is the Premier League's all-time top scorer with 206 goals | Alan Shearer's Premier League record is 260 goals. |
| d2 | Manchester City became the first team to reach 100 points in 2018-19 | Manchester City reached 100 points in 2017-18. |

Seeded as true (must hold): "The Premier League was founded in 1992".

Borrowed: "Our life is frittered away by detail." from Walden, and On The Duty Of Civil Disobedience by Henry David Thoreau (<https://www.gutenberg.org/ebooks/205>).

### 11-greece-2004

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | beat the hosts Portugal 2-0 in the final in Lisbon | Greece beat Portugal 1-0 in the Euro 2004 final in Lisbon. |
| d2 | Under their Italian coach Otto Rehhagel | Otto Rehhagel, Greece's coach at Euro 2004, is German. |

Seeded as true (must hold): "Denmark won the European Championship in 1992".

Borrowed: "He will win who knows when to fight and when not to fight." from The Art of War by Sun Tzu, translated by Lionel Giles (<https://www.gutenberg.org/ebooks/132>).

### 12-istanbul-momentum

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Andriy Shevchenko added two more before the break | Hernan Crespo scored Milan's second and third goals in the 2005 final; Shevchenko did not score. |
| d2 | Liverpool scored three times in sixteen minutes | Liverpool's three goals came in six minutes (Gerrard 54', Smicer 56', Alonso 60'). |

Seeded as true (must hold): "Paolo Maldini opened the scoring inside the first minute".

Borrowed: "Nothing is so painful to the human mind as a great and sudden change." from Frankenstein; Or, The Modern Prometheus by Mary Wollstonecraft Shelley (<https://www.gutenberg.org/ebooks/84>).

### 13-five-substitutes

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | The back-pass rule, introduced in 1998, stopped goalkeepers picking the ball up from a team-mate's pass | The back-pass rule was introduced in 1992. |
| d2 | English football switched to three points for a win in 1981, and FIFA followed at the 1990 World Cup | FIFA first used three points for a win at the 1994 World Cup (the 1981 date for English football is correct). |

Seeded as true (must hold): "Yellow and red cards were first used at the 1970 World Cup".

Borrowed: "Now, here, you see, it takes all the running you can do, to keep in the same place." from Through the Looking-Glass by Lewis Carroll (<https://www.gutenberg.org/ebooks/12>).

### 14-brazil-1970

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | beating Italy 4-2 in the final at the Azteca | Brazil beat Italy 4-1 in the 1970 World Cup final at the Estadio Azteca. |
| d2 | Brazil's fourth title | 1970 was Brazil's third World Cup title (after 1958 and 1962); their fourth came in 1994. |
| d3 | The team was coached by Telê Santana | Brazil's coach at the 1970 World Cup was Mário Zagallo; Telê Santana coached Brazil at the 1982 and 1986 World Cups. |

Seeded as true (must hold): "Jairzinho scored in every one of those matches".

Borrowed: "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness" from A Tale of Two Cities by Charles Dickens (<https://www.gutenberg.org/ebooks/98>).

### 15-penalty-lottery

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | The penalty spot is ten yards from goal | The penalty mark is 12 yards (11 metres) from the goal line. |
| d2 | The World Cup's first shoot-out came in the 1982 semi-final, when Italy beat France in Seville | The first World Cup shoot-out was West Germany's win over France in the 1982 semi-final in Seville. |

Seeded as true (must hold): "Roberto Baggio missed the decisive kick in the 1994 final".

Borrowed: "All warfare is based on deception." from The Art of War by Sun Tzu, translated by Lionel Giles (<https://www.gutenberg.org/ebooks/132>).

### 16-womens-game-stadiums

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | The first Women's World Cup was held in 1991 in the United States | The first FIFA Women's World Cup was held in 1991 in China (the United States won it). |
| d2 | Spain won the 2023 World Cup by beating England 2-0 in Sydney | Spain beat England 1-0 in the 2023 Women's World Cup final in Sydney. |

Seeded as true (must hold): "The Lionesses beat Germany 2-1 at Wembley in the Euro 2022 final in front of 87,192 people".

Borrowed: "If a man does not keep pace with his companions, perhaps it is because he hears a different drummer." from Walden, and On The Duty Of Civil Disobedience by Henry David Thoreau (<https://www.gutenberg.org/ebooks/205>).

### 17-no-kubernetes

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | handed it to the Apache Software Foundation a year later | Google donated Kubernetes to the Cloud Native Computing Foundation (CNCF, part of the Linux Foundation) in 2015. |
| d2 | The name comes from the Latin word for helmsman | 'Kubernetes' comes from the Greek word for helmsman or pilot. |

Borrowed: "It is a capital mistake to theorize before one has data." from The Adventures of Sherlock Holmes by Arthur Conan Doyle (<https://www.gutenberg.org/ebooks/1661>).

### 18-python-type-hints

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Python 3.0 followed in 2012 | Python 3.0 was released in December 2008. |
| d2 | Type hints arrived with PEP 484 in Python 2.7 | PEP 484 type hints were introduced in Python 3.5, released in September 2015. |

Seeded as true (must hold): "Guido van Rossum released the first version of Python in 1991".

Borrowed: "A foolish consistency is the hobgoblin of little minds, adored by little statesmen and philosophers and divines." from Essays, First Series by Ralph Waldo Emerson (<https://www.gutenberg.org/ebooks/2944>).

### 19-big-rewrite

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Joel Spolsky's famous essay "Things You Should Never Do" was about Microsoft's decision to rewrite Word from scratch | Spolsky's essay 'Things You Should Never Do, Part I' (April 2000) was about Netscape's decision to rewrite its browser from scratch. |
| d2 | Fred Brooks published The Mythical Man-Month in 1995 | The Mythical Man-Month was first published in 1975 (an anniversary edition followed in 1995). |
| d3 | it was published in the winter of 1999 | The Agile Manifesto was written and published in February 2001. |

Seeded as true (must hold): "The Agile Manifesto was written by seventeen people at a ski resort in Utah".

Borrowed: "There is no instance of a country having benefited from prolonged warfare." from The Art of War by Sun Tzu, translated by Lionel Giles (<https://www.gutenberg.org/ebooks/132>).

### 20-nuclear-climate

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | commits countries to holding warming well below three degrees Celsius | The Paris Agreement aims to hold warming well below 2 degrees Celsius above pre-industrial levels, and to pursue efforts to limit it to 1.5 degrees. |
| d2 | the meltdowns at Fukushima in 2001 | The Fukushima Daiichi accident happened in March 2011. |
| d3 | Germany switched off its last three reactors in April 2013 | Germany shut down its last three nuclear reactors in April 2023. |

Seeded as true (must hold): "The Paris Agreement, adopted in 2015".

Borrowed: "If men were angels, no government would be necessary." from The Federalist Papers by Alexander Hamilton, John Jay and James Madison (No. 51) (<https://www.gutenberg.org/ebooks/1404>).

### 21-phones-in-schools

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | The first iPhone went on sale in 2010 | The first iPhone was announced in January 2007 and went on sale in June 2007. |
| d2 | The PISA tests, run by UNESCO every three years | PISA is run by the OECD (Organisation for Economic Co-operation and Development). |

Borrowed: "He who knows only his own side of the case, knows little of that." from On Liberty by John Stuart Mill (<https://www.gutenberg.org/ebooks/34901>).

### 22-clock-changes

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | to save coal during the Second World War | Germany adopted daylight saving time in 1916, during the First World War. |
| d2 | The European Parliament voted in 2009 to scrap the twice-yearly change | The European Parliament voted in March 2019 to end seasonal clock changes. |

Seeded as true (must hold): "William Willett, a British builder, campaigned for it in a 1907 pamphlet".

Borrowed: "The real price of everything, what everything really costs to the man who wants to acquire it, is the toil and trouble of acquiring it." from An Inquiry into the Nature and Causes of the Wealth of Nations by Adam Smith (<https://www.gutenberg.org/ebooks/3300>).

### 23-close-the-libraries

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Parliament passed the Public Libraries Act in 1750 | The Public Libraries Act was passed in 1850. |
| d2 | Andrew Carnegie, the Scottish-born steel magnate, paid for more than 25,000 libraries | Andrew Carnegie funded about 2,500 libraries (2,509 is the usual figure). |
| d3 | Wikipedia, launched in 2011 | Wikipedia was launched in January 2001. |

Seeded as true (must hold): "the Gutenberg Bible was printed in Mainz in the 1450s".

Borrowed: "People of the same trade seldom meet together, even for merriment and diversion, but the conversation ends in a conspiracy against the public, or in some contrivance to raise prices." from An Inquiry into the Nature and Causes of the Wealth of Nations by Adam Smith (<https://www.gutenberg.org/ebooks/3300>).

### 24-robots-not-astronauts

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Neil Armstrong and Michael Collins walked on the surface on 20 July 1969 | Neil Armstrong and Buzz Aldrin walked on the Moon on 20 July 1969; Michael Collins stayed in lunar orbit. |
| d2 | The last of them left in December 1975 | The last crewed Moon landing, Apollo 17, left the Moon in December 1972. |
| d3 | Voyager 1, launched in 1987 | Voyager 1 was launched in September 1977. |
| d4 | NASA spent millions of dollars developing a pen that would write in zero gravity, while the Soviet cosmonauts simply used pencils. | The space pen was developed privately by the Fisher Pen Company at its own expense; NASA paid nothing for its development and later bought the pens at a few dollars each, as did the Soviet space programme. |

Seeded as true (must hold): "Twelve people have walked on the Moon.".

Borrowed: "It is not down in any map; true places never are." from Moby Dick; Or, The Whale by Herman Melville (<https://www.gutenberg.org/ebooks/2701>).

### 25-jwt-sessions

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | JSON Web Tokens are defined in RFC 6749 | JSON Web Tokens are defined in RFC 7519 (May 2015); RFC 6749 is the OAuth 2.0 Authorization Framework. |
| d2 | the payload is encrypted by default, so we can put the user's id, role and tenant in it without exposing them to the browser | A standard signed JWT (JWS) has a payload that is only base64url-encoded and readable by anyone who holds the token; encryption requires JWE (RFC 7516). |

### 26-kafka-stream

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Kafka was originally built at Twitter | Apache Kafka was originally developed at LinkedIn. |
| d2 | Kafka guarantees that messages are delivered in order across all partitions of a topic | Kafka guarantees ordering only within a single partition, not across the partitions of a topic. |
| d3 | Kafka keeps messages for 30 days by default | Kafka's default log retention is 7 days (log.retention.hours=168). |

Seeded as true (must hold): "open-sourced in 2011".

### 27-rate-limiter

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | HTTP 429 Too Many Requests, which is defined in RFC 2616 | Status code 429 Too Many Requests is defined in RFC 6585 (April 2012); RFC 2616 (HTTP/1.1, 1999) does not contain it. |
| d2 | Redis, first released in 2009 by Salvatore Sanfilippo, listens on port 5432 by default | Redis listens on port 6379 by default; 5432 is PostgreSQL's default port. |

### 28-audit-log-archive

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | is designed for 99.99% durability | Amazon S3 is designed for 99.999999999% (eleven nines) durability; 99.99% is the availability design target of the S3 Standard storage class. |
| d2 | S3 is only eventually consistent, so a read that follows a write may return stale data. | Since December 2020, Amazon S3 has provided strong read-after-write consistency for all requests. |

Seeded as true (must hold): "Amazon S3, launched in 2006".

### 29-raft-config-store

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | Raft was published in 2014 by Leslie Lamport | Raft was published in 2014 by Diego Ongaro and John Ousterhout ('In Search of an Understandable Consensus Algorithm'); Leslie Lamport is the author of Paxos. |
| d2 | ZooKeeper, which also uses Raft | ZooKeeper uses its own atomic broadcast protocol, Zab, not Raft. |
| d3 | A five-node cluster tolerates three failed nodes | A five-node Raft cluster needs a majority of three, so it tolerates two failed nodes. |

Seeded as true (must hold): "an entry is committed once a majority of nodes have stored it".

### 30-tls13-internal

| Defect | Seeded (wrong) text | True value |
| --- | --- | --- |
| d1 | TLS 1.3 was published as RFC 8446 in 2014 | TLS 1.3 was published as RFC 8446 in August 2018. |
| d2 | A full TLS 1.3 handshake takes two round trips, the same as TLS 1.2 | A full TLS 1.3 handshake takes one round trip; TLS 1.2 takes two. |

Seeded as true (must hold): "TLS 1.0 and 1.1 were formally deprecated by RFC 8996 in 2021".

