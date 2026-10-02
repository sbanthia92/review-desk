# Review Desk local MCP server (stdio)

One tool, `review_document`, served over stdio with the official MCP Python SDK.

| Argument | Required | Notes |
| --- | --- | --- |
| `content` | one of content/url | Document text or markdown (default cap 20,000 words) |
| `url` | one of content/url | Public http(s) page, fetched with SSRF protection |
| `profile` | no | `auto` (default), `opinion`, `design_doc` |
| `focus` | no | Free-text steer, e.g. "check the financial figures" |

Progress is streamed as `notifications/progress` (progress = percent, total =
100) when the client sends a progress token. The result is the markdown report.
Errors (missing setup, rejected key, document too long, URL unreachable, rate
limit) come back as tool errors with a readable message.

## Run it

```sh
uv run --directory /path/to/review-desk python -m reviewdesk.mcp_local
```

Environment variables. Keep them in the repo's `.env` file (the server loads `.env` from its
working directory, or the file named by `REVIEWDESK_ENV_FILE`), or put them in the client's
`env` block. Never pass them as tool arguments:

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | LLM provider key (read by `reviewdesk.registry`) |
| `BRAVE_SEARCH_API_KEY`, `TAVILY_API_KEY` or `EXA_API_KEY` | Search provider key |
| `REVIEWDESK_FAKE=1` | Offline demo: fake agents, no keys, no model calls (URL input still fetches the page) |
| `REVIEWDESK_MAX_WORDS` | Word cap (default 20000) |
| `REVIEWDESK_LOG_LEVEL` | stderr log level (default `INFO`) |

Logs go to stderr only; they never contain keys or document text.

## Claude Desktop

Edit `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`), then
restart Claude Desktop. `uv` must be on the PATH Claude Desktop sees; if not,
use the absolute path from `which uv` as `command`.

```json
{
  "mcpServers": {
    "review-desk": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/review-desk",
        "python", "-m", "reviewdesk.mcp_local"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "BRAVE_SEARCH_API_KEY": "..."
      }
    }
  }
}
```

To try it without keys, replace the `env` block with `{"REVIEWDESK_FAKE": "1"}`.

## Claude Code

```sh
claude mcp add review-desk \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e BRAVE_SEARCH_API_KEY=... \
  -- uv run --directory /path/to/review-desk python -m reviewdesk.mcp_local
```

Demo mode:

```sh
claude mcp add review-desk-demo -e REVIEWDESK_FAKE=1 \
  -- uv run --directory /path/to/review-desk python -m reviewdesk.mcp_local
```

Then ask, for example: "Review this draft with review_document: <paste>". A real
review takes a few minutes; raise the client's tool timeout if it gives up
early (Claude Code: `MCP_TOOL_TIMEOUT=600000 claude`).

## In code

```python
from reviewdesk.mcp_local import create_server

server = create_server(pipeline=my_runner, fetcher=my_fetcher, max_words=20_000)
```

`pipeline` is any `ReviewRunner`: `async (doc, profile, on_progress, *, focus=None) -> Report`.
Wrap a plain contract `ReviewPipeline` with `adapt_pipeline`. The default is
`EnvPipeline`, which builds the real registry lazily from
`reviewdesk.registry.build_registry_from_env()` (T11) and reports a clear
missing-setup error if that is absent or raises `MissingSetup`.
