# Review Desk

Review Desk reviews a draft the way a careful editor would, and shows its
sources. You give it an article, a blog post or a design doc. It returns one
report:

- **Facts checked against the web.** Each claim is looked up, and every
  "this is wrong" comes with the page that says so.
- **The strongest counter-case.** A devil's advocate builds the best argument
  against your piece from sources it retrieves.
- **Copy edits.** Grammar and clarity fixes that keep your voice.
- **Originality notes.** Sentences that closely match text found online.

It runs as an [MCP](https://modelcontextprotocol.io) server with one tool,
`review_document`, so you can ask for a review from Claude Code or Claude
Desktop. It also has a command-line tool. It runs on your machine with **your
own API keys**; nothing is sent to the author of this project.

A finding looks like this:

> **Factual error** — "Mjällby clinched the title on 20 October 2025 by winning
> 2-0 at IFK Göteborg, with two games to spare"
>
> The date, the 2-0 score and the opponent are confirmed. ESPN and Flashscore
> both say three rounds were left (27 of 30 played), so "two games to spare" is
> wrong.

## What you need

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).
- A model API key: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.
- One search API key: `TAVILY_API_KEY`, `BRAVE_SEARCH_API_KEY` or
  `EXA_API_KEY`.

The recommended setup is Anthropic with Tavily, which is what the examples
below use.

### Getting your keys

You need one model key and one search key. Each takes a couple of minutes:
sign up, create a key, copy it.

| Key | Where to get it |
| --- | --- |
| `ANTHROPIC_API_KEY` | Sign in at <https://platform.claude.com>, add credit under Billing, then create a key under **API keys** (<https://platform.claude.com/settings/keys>) |
| `OPENAI_API_KEY` | Sign in at <https://platform.openai.com>, add credit under Billing, then create a key at <https://platform.openai.com/api-keys> |
| `TAVILY_API_KEY` | Sign up at <https://app.tavily.com>; the key is shown on the dashboard |
| `BRAVE_SEARCH_API_KEY` | Subscribe to a plan at <https://brave.com/search/api/>, then create a key in the API dashboard |
| `EXA_API_KEY` | Sign up at <https://dashboard.exa.ai> and create a key under **API keys** |

Model keys are pay-as-you-go: reviews fail with a clear message if the account
has no credit. Keep keys out of chats and out of version control; put them in
the MCP client's `env` block or in a `.env` file as shown below.

A review of a 700-word article takes about two minutes and uses roughly
150,000–180,000 tokens on your model key (about $0.60–$1.20 at Anthropic's
list prices in October 2026) plus about 30 searches.

## Install: Claude Code

```sh
claude mcp add review-desk \
  -e ANTHROPIC_API_KEY=your-key \
  -e TAVILY_API_KEY=your-key \
  -- uvx --from git+https://github.com/sbanthia92/review-desk reviewdesk-mcp
```

Start Claude Code with a longer tool timeout, because a review takes minutes:

```sh
MCP_TOOL_TIMEOUT=600000 claude
```

Then ask: "Use review-desk to review this draft", and paste the text or give a
public URL.

## Install: Claude Desktop

Add this to `claude_desktop_config.json` (on macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`), then restart
the app. If Claude Desktop cannot find `uvx`, use its full path (`which uvx`).

```json
{
  "mcpServers": {
    "review-desk": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/sbanthia92/review-desk", "reviewdesk-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key",
        "TAVILY_API_KEY": "your-key"
      }
    }
  }
}
```

To try it without keys, replace the `env` block with
`{"REVIEWDESK_FAKE": "1"}`. Demo mode returns placeholder findings and makes
no model calls.

## Command line

```sh
git clone https://github.com/sbanthia92/review-desk && cd review-desk
cp .env.example .env        # then put your keys in .env
uv run reviewdesk review path/to/draft.md --profile opinion > report.md
```

Progress prints to the terminal; the report goes to `report.md`. Use
`--profile design_doc` for technical design docs, or leave the profile out to
let the tool decide. A URL works in place of a file path.

## The tool

`review_document` takes:

| Argument | Required | Notes |
| --- | --- | --- |
| `content` | one of `content` / `url` | The draft as text or markdown (up to 20,000 words) |
| `url` | one of `content` / `url` | A public web page to fetch and review |
| `profile` | no | `auto` (default), `opinion` or `design_doc` |
| `focus` | no | A steer, e.g. "check the financial figures" |

It returns the report as markdown and sends progress updates while it works.

The report has these sections: a verdict line; **Must fix** (factual errors and
claims the sources do not back); **Could not verify** (claims it found no
adequate source for, which is not the same as wrong); **Strongest
counter-case**; **Should fix** (structure); **Polish** (copy edits);
**Originality notes**; and an appendix listing every claim with its verdict and
sources.

## Settings

Set these in the MCP client's `env` block, in the environment, or in a `.env`
file (the server reads `.env` from its working directory, or the file named by
`REVIEWDESK_ENV_FILE`).

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | Model provider key. Anthropic is used if both are set |
| `TAVILY_API_KEY`, `BRAVE_SEARCH_API_KEY` or `EXA_API_KEY` | Search provider key |
| `REVIEWDESK_LLM_PROVIDER` | `anthropic` or `openai`, to force one |
| `REVIEWDESK_SEARCH_PROVIDER` | `tavily`, `brave` or `exa`, to force one |
| `REVIEWDESK_MODEL_CHEAP`, `_MID`, `_STRONG` | Override the model used for each tier |
| `REVIEWDESK_MAX_WORDS` | Document size cap (default 20000) |
| `REVIEWDESK_FAKE=1` | Demo mode: no keys, no model calls |

## What leaves your machine

- **Your document** goes to your model provider (Anthropic or OpenAI), under
  your key and their terms.
- **Search queries** built from your document's claims, and up to eight
  sentences quoted exactly for the originality check, go to your search
  provider.
- **Web pages** named in search results are fetched directly from your machine.
  The fetcher refuses private and local network addresses.

Nothing is stored after the review finishes, and nothing is sent anywhere
else. Keys are read from your environment and are never logged.

## How well it works

Measured on a small test in October 2026: four articles with planted errors
that cannot be caught from memory (recent results, exact figures), one run
each, compared with asking the same model to "review this and argue the other
side" in a single prompt.

| | Single prompt | Review Desk |
| --- | --- | --- |
| Planted wrong facts caught (19) | 7 | 13 |
| ...with a cited source | 0 | 13 |
| Weak arguments rebutted (4) | 2 | 4 |
| Borrowed sentences caught (3) | 3 | 1 |
| Tokens per article | about 5,000 | about 136,000 |

Review Desk earns its cost on claims that need looking up: recent events,
exact figures, and details a model cannot know from memory. Every finding
names its source, so you can check it in one click.

Getting the most from it:

- Open the cited source before acting on a "factual error". The report links
  the page and quotes the passage it relied on.
- Search results change, so two runs on the same draft can differ slightly.
- The originality check is a heuristic: it flags close matches found online.
- Sources are web pages; PDFs are not read.

## How it works

An orchestrator runs specialist agents and settles their disagreements:

1. A **claim extractor** splits the document into its thesis, supporting
   claims and checkable facts.
2. In parallel: a **fact-checker** researches each fact in a bounded loop
   (search, fetch, read, assess; at most three rounds per claim); a **devil's
   advocate** gathers counter-evidence; a **copy editor**; an **originality
   checker**; and, for design docs, a **structure reviewer**.
3. A reaction round: a verified fact the devil's advocate disputes is
   re-checked, and the two argue once over the most important disagreements.
4. Fixed rules settle what is left (for example, a grammar fix on a sentence
   flagged as wrong is dropped), findings are ranked, and the report is written.

`DESIGN.md` has the full design. `ARCHITECTURE.md` describes the shared types.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Tests run offline with fakes. Tests that call real providers are marked `live`
and skipped by default (`uv run pytest -m live`). The evaluation harness and
its 40 seeded documents are in `eval/` (see `eval/README.md`); running it
against real providers spends money on your keys.

## Status

The local MCP server and command-line tool work. A hosted version (sign in,
add a key once, get the report by email) is designed in `DESIGN.md` but not
built.
