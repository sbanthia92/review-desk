"""Real-network fetch tests. Run with ``uv run pytest -m live``."""

import pytest

from reviewdesk.contracts import BlockedURLError
from reviewdesk.providers.fetch import HttpFetcher

pytestmark = pytest.mark.live


async def test_live_fetch_public_page():
    async with HttpFetcher() as fetcher:
        page = await fetcher.fetch("https://example.com/")
    assert page.status == 200
    assert "Example Domain" in page.text


async def test_live_real_dns_localhost_blocked():
    async with HttpFetcher() as fetcher:
        with pytest.raises(BlockedURLError):
            await fetcher.fetch("http://localtest.me/")  # public DNS name for 127.0.0.1
