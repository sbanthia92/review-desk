"""Real-network search tests. Run with ``uv run pytest -m live``; skipped without keys."""

import os

import pytest

from reviewdesk.providers.search import PROVIDERS, make_search_client

pytestmark = pytest.mark.live


@pytest.mark.parametrize("name", list(PROVIDERS))
async def test_live_search(name):
    env_var, _cls = PROVIDERS[name]
    key = os.environ.get(env_var, "").strip()
    if not key:
        pytest.skip(f"{env_var} not set")
    async with make_search_client(name, key) as client:
        results = await client.search("HTTP/3 runs over QUIC", k=3)
    assert 1 <= len(results) <= 3
    assert all(r.url.startswith(("http://", "https://")) for r in results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
