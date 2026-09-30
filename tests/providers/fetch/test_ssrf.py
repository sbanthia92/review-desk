import ipaddress

import pytest

from reviewdesk.contracts import BlockedURLError, FetchError
from reviewdesk.providers.fetch.ssrf import (
    check_addresses,
    check_url,
    check_url_static,
    is_public_ip,
    resolve_public,
)

PUBLIC = "93.184.216.34"


def resolver_for(mapping: dict[str, list[str]]):
    calls: list[tuple[str, int]] = []

    async def resolve(host: str, port: int) -> list[str]:
        calls.append((host, port))
        if host not in mapping:
            raise FetchError("could not resolve host")
        return mapping[host]

    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


# --- IP classification -----------------------------------------------------

BLOCKED_IPS = [
    ("10.0.0.1", "private 10/8"),
    ("172.16.5.4", "private 172.16/12"),
    ("192.168.1.1", "private 192.168/16"),
    ("127.0.0.1", "loopback"),
    ("127.8.9.10", "loopback range"),
    ("169.254.169.254", "link-local (cloud metadata)"),
    ("224.0.0.1", "multicast"),
    ("239.255.255.250", "multicast"),
    ("240.0.0.1", "reserved"),
    ("255.255.255.255", "broadcast"),
    ("0.0.0.0", "unspecified"),
    ("100.64.0.1", "shared CGNAT"),
    ("192.0.2.1", "documentation"),
    ("198.18.0.1", "benchmarking"),
    ("::1", "IPv6 loopback"),
    ("::", "IPv6 unspecified"),
    ("fe80::1", "IPv6 link-local"),
    ("fc00::1", "IPv6 unique local"),
    ("fd12:3456::1", "IPv6 unique local"),
    ("fec0::1", "IPv6 site-local"),
    ("ff02::1", "IPv6 multicast"),
    ("ff0e::1", "IPv6 global-scope multicast"),
    ("::ffff:127.0.0.1", "IPv4-mapped loopback"),
    ("::ffff:10.0.0.1", "IPv4-mapped private"),
    ("::ffff:169.254.169.254", "IPv4-mapped metadata"),
    ("::127.0.0.1", "IPv4-compatible loopback"),
    ("64:ff9b::a00:1", "NAT64 of 10.0.0.1"),
    ("2002:0a00:0001::1", "6to4 of 10.0.0.1"),
    ("2001:db8::1", "IPv6 documentation"),
]


@pytest.mark.parametrize(("address", "why"), BLOCKED_IPS)
def test_non_public_ips_blocked(address, why):
    assert not is_public_ip(ipaddress.ip_address(address)), why


@pytest.mark.parametrize("address", [PUBLIC, "8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_public_ips_allowed(address):
    assert is_public_ip(ipaddress.ip_address(address))


def test_ipv4_mapped_public_allowed():
    assert is_public_ip(ipaddress.ip_address("::ffff:8.8.8.8"))


def test_check_addresses_requires_every_address_public():
    assert check_addresses([PUBLIC, "8.8.8.8"]) == [PUBLIC, "8.8.8.8"]
    with pytest.raises(BlockedURLError):
        check_addresses([PUBLIC, "10.0.0.1"])
    with pytest.raises(BlockedURLError):
        check_addresses(["not-an-ip"])
    with pytest.raises(FetchError):
        check_addresses([])


def test_zone_id_is_stripped():
    with pytest.raises(BlockedURLError):
        check_addresses(["fe80::1%eth0"])


# --- static URL rules --------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "data:text/html,hi",
        "javascript:alert(1)",
        "//example.com/no-scheme",
        "example.com",
    ],
)
def test_non_http_schemes_blocked(url):
    with pytest.raises(BlockedURLError):
        check_url_static(url)


def test_http_and_https_allowed():
    assert check_url_static("http://example.com/a").port == 80
    checked = check_url_static("HTTPS://Example.COM/a?b=1")
    assert (checked.scheme, checked.host, checked.port) == ("https", "example.com", 443)


@pytest.mark.parametrize("url", ["http:///path", "https://"])
def test_missing_host_blocked(url):
    with pytest.raises(BlockedURLError):
        check_url_static(url)


@pytest.mark.parametrize("url", ["http://user:pw@example.com/", "http://user@example.com/"])
def test_credentials_blocked(url):
    with pytest.raises(BlockedURLError, match="credentials"):
        check_url_static(url)


def test_ports():
    with pytest.raises(BlockedURLError, match="port 22"):
        check_url_static("http://example.com:22/")
    with pytest.raises(BlockedURLError, match="port 6379"):
        check_url_static("https://example.com:6379/")
    assert check_url_static("http://example.com:8080/", allowed_ports=None).port == 8080
    assert check_url_static("http://example.com:443/").port == 443
    with pytest.raises(BlockedURLError):
        check_url_static("http://example.com:99999/")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://LOCALHOST./",
        "http://app.localhost/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/",
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0x7f.1/",
        "http://017700000001/",
        "http://127.1/",
    ],
)
def test_local_and_numeric_hosts_blocked_without_dns(url):
    with pytest.raises(BlockedURLError):
        check_url_static(url)


def test_hex_looking_hostname_is_not_numeric():
    assert check_url_static("https://cafe.be/").host == "cafe.be"


def test_public_ip_literal_allowed():
    assert check_url_static(f"http://{PUBLIC}/").host == PUBLIC


# --- DNS-based rules ---------------------------------------------------------


async def test_hostname_resolving_to_private_blocked():
    resolve = resolver_for({"evil.example": ["10.1.2.3"]})
    with pytest.raises(BlockedURLError):
        await check_url("https://evil.example/", resolve)
    assert resolve.calls == [("evil.example", 443)]


async def test_hostname_with_any_private_address_blocked():
    resolve = resolver_for({"mixed.example": [PUBLIC, "::1"]})
    with pytest.raises(BlockedURLError):
        await check_url("http://mixed.example/", resolve)


async def test_hostname_resolving_to_mapped_ipv6_blocked():
    resolve = resolver_for({"sneaky.example": ["::ffff:192.168.0.1"]})
    with pytest.raises(BlockedURLError):
        await check_url("http://sneaky.example/", resolve)


async def test_public_hostname_allowed():
    resolve = resolver_for({"example.com": [PUBLIC]})
    checked = await check_url("https://example.com/page", resolve)
    assert checked.host == "example.com"


async def test_unresolvable_host_is_fetch_error_not_blocked():
    resolve = resolver_for({})
    with pytest.raises(FetchError) as info:
        await check_url("https://nowhere.example/", resolve)
    assert not isinstance(info.value, BlockedURLError)


async def test_resolver_oserror_is_fetch_error():
    async def broken(host: str, port: int) -> list[str]:
        raise OSError("dns down")

    with pytest.raises(FetchError, match="could not resolve"):
        await check_url("https://example.com/", broken)


async def test_ip_literal_skips_resolver():
    resolve = resolver_for({})
    assert await resolve_public(PUBLIC, 80, resolve) == [PUBLIC]
    assert resolve.calls == []
