"""SSRF protection for outbound fetches.

Rules (each raises ``BlockedURLError``):

- Only ``http`` and ``https`` schemes; URLs must have a host and must not
  carry credentials (``user:pass@host``).
- Only allowed ports (default 80 and 443).
- ``localhost`` names and ambiguous numeric hosts (``2130706433``,
  ``0x7f.1``) are refused without resolving.
- The host is resolved and **every** address must be public: private,
  loopback, link-local, multicast, reserved, unspecified, shared (CGNAT) and
  other non-global addresses are refused, for IPv4 and IPv6. IPv6 forms that
  embed an IPv4 address (IPv4-mapped, IPv4-compatible, 6to4, Teredo, NAT64)
  are checked against the embedded address too.

DNS resolution is injectable (``Resolver``) so tests run offline. The
guarded network backend (``GuardedNetworkBackend``) repeats the check at
connect time and connects to the vetted IP, closing the DNS-rebinding gap
between check and connect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from reviewdesk.contracts import BlockedURLError, FetchError

Resolver = Callable[[str, int], Awaitable[list[str]]]
"""Async ``(host, port) -> list of IP address strings``."""

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}
DEFAULT_ALLOWED_PORTS: frozenset[int] = frozenset({80, 443})

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_NUMERIC_LABEL = re.compile(r"^(0x[0-9a-f]*|[0-9]+)$", re.IGNORECASE)


async def system_resolver(host: str, port: int) -> list[str]:
    """Resolve ``host`` with the OS resolver (``getaddrinfo``)."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise FetchError("could not resolve host") from exc
    addresses: list[str] = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        address = str(sockaddr[0])
        if address not in addresses:
            addresses.append(address)
    return addresses


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    packed = ip.packed
    if packed[:12] == bytes(12) and int(ip) > 1:
        # Deprecated IPv4-compatible form ``::a.b.c.d``.
        return ipaddress.IPv4Address(packed[12:])
    return None


def is_public_ip(ip: IPAddress) -> bool:
    """True only for globally routable unicast addresses."""
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    ):
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.is_site_local:
            return False
        embedded = _embedded_ipv4(ip)
        if embedded is not None and not is_public_ip(embedded):
            return False
    return True


def parse_ip(address: str) -> IPAddress:
    """Parse an IP string, dropping any IPv6 zone id (``fe80::1%eth0``)."""
    return ipaddress.ip_address(address.split("%", 1)[0])


def check_addresses(addresses: Iterable[str]) -> list[str]:
    """Return ``addresses`` if all are public; raise ``BlockedURLError`` otherwise."""
    vetted: list[str] = []
    for address in addresses:
        try:
            ip = parse_ip(address)
        except ValueError:
            raise BlockedURLError("host resolved to an invalid address") from None
        if not is_public_ip(ip):
            raise BlockedURLError("host resolves to a non-public address")
        vetted.append(str(ip))
    if not vetted:
        raise FetchError("could not resolve host")
    return vetted


def _is_ambiguous_numeric(host: str) -> bool:
    labels = host.rstrip(".").split(".")
    return all(label and _NUMERIC_LABEL.match(label) for label in labels)


def _is_localhost_name(host: str) -> bool:
    host = host.rstrip(".").lower()
    return host == "localhost" or host.endswith(".localhost")


@dataclass(frozen=True)
class CheckedURL:
    """A URL that passed the static checks, with its parts."""

    url: str
    scheme: str
    host: str
    port: int


def check_url_static(
    url: str, *, allowed_ports: frozenset[int] | None = DEFAULT_ALLOWED_PORTS
) -> CheckedURL:
    """Apply every rule that does not need DNS. ``allowed_ports=None`` allows any."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        raise BlockedURLError("malformed URL") from None
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise BlockedURLError("only http and https URLs can be fetched")
    host = (parts.hostname or "").lower()
    if not host:
        raise BlockedURLError("URL has no host")
    if parts.username is not None or parts.password is not None:
        raise BlockedURLError("URLs with credentials are not fetched")
    effective_port = port if port is not None else DEFAULT_PORTS[scheme]
    if allowed_ports is not None and effective_port not in allowed_ports:
        raise BlockedURLError(f"port {effective_port} is not allowed")
    if _is_localhost_name(host):
        raise BlockedURLError("localhost is not allowed")
    try:
        ip = parse_ip(host)
    except ValueError:
        if _is_ambiguous_numeric(host):
            raise BlockedURLError("ambiguous numeric host") from None
    else:
        if not is_public_ip(ip):
            raise BlockedURLError("URL points at a non-public address")
    return CheckedURL(url=url.strip(), scheme=scheme, host=host, port=effective_port)


async def resolve_public(host: str, port: int, resolver: Resolver) -> list[str]:
    """Resolve ``host`` and return its addresses if every one is public."""
    try:
        literal = parse_ip(host)
    except ValueError:
        pass
    else:
        return check_addresses([str(literal)])
    if _is_localhost_name(host):
        raise BlockedURLError("localhost is not allowed")
    if _is_ambiguous_numeric(host):
        raise BlockedURLError("ambiguous numeric host")
    try:
        addresses = await resolver(host, port)
    except FetchError:
        raise
    except (OSError, UnicodeError) as exc:
        raise FetchError("could not resolve host") from exc
    return check_addresses(addresses)


async def check_url(
    url: str,
    resolver: Resolver = system_resolver,
    *,
    allowed_ports: frozenset[int] | None = DEFAULT_ALLOWED_PORTS,
) -> CheckedURL:
    """Apply every SSRF rule to ``url``, including DNS resolution."""
    checked = check_url_static(url, allowed_ports=allowed_ports)
    await resolve_public(checked.host, checked.port, resolver)
    return checked
