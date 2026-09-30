"""Page fetching implementing ``Fetcher``.

``HttpFetcher`` fetches public http(s) pages with SSRF protection (see
``ssrf``), redirect/size/time caps and readable-text extraction (see
``extract``). Fetched text is untrusted data.
"""

from reviewdesk.providers.fetch.extract import extract_readable
from reviewdesk.providers.fetch.fetcher import HttpFetcher
from reviewdesk.providers.fetch.guarded import GuardedNetworkBackend, GuardedTransport
from reviewdesk.providers.fetch.ssrf import (
    Resolver,
    check_url,
    check_url_static,
    is_public_ip,
    system_resolver,
)

__all__ = [
    "GuardedNetworkBackend",
    "GuardedTransport",
    "HttpFetcher",
    "Resolver",
    "check_url",
    "check_url_static",
    "extract_readable",
    "is_public_ip",
    "system_resolver",
]
