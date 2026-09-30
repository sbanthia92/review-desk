import pytest

from reviewdesk.contracts import MissingSetup
from reviewdesk.providers.search import (
    PROVIDERS,
    BraveSearch,
    ExaSearch,
    TavilySearch,
    make_search_client,
    search_clients_from_env,
)
from reviewdesk.providers.search.retry import parse_retry_after


def test_env_var_names():
    assert {name: env for name, (env, _cls) in PROVIDERS.items()} == {
        "brave": "BRAVE_SEARCH_API_KEY",
        "tavily": "TAVILY_API_KEY",
        "exa": "EXA_API_KEY",
    }


def test_from_env_skips_missing_and_blank_keys():
    clients = search_clients_from_env({"BRAVE_SEARCH_API_KEY": "k1", "EXA_API_KEY": "  "})
    assert list(clients) == ["brave"]
    assert isinstance(clients["brave"], BraveSearch)


def test_from_env_all_and_only():
    env = {"BRAVE_SEARCH_API_KEY": "a", "TAVILY_API_KEY": "b", "EXA_API_KEY": "c"}
    clients = search_clients_from_env(env)
    assert [type(c) for c in clients.values()] == [BraveSearch, TavilySearch, ExaSearch]
    assert list(search_clients_from_env(env, only=["exa"])) == ["exa"]
    with pytest.raises(MissingSetup):
        search_clients_from_env(env, only=["bing"])


def test_make_search_client():
    assert isinstance(make_search_client("tavily", "k"), TavilySearch)
    with pytest.raises(MissingSetup, match="unknown search provider"):
        make_search_client("nope", "k")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("5", 5.0), (" 1.5 ", 1.5), ("-1", None), ("Wed, 21 Oct", None), (None, None)],
)
def test_parse_retry_after(value, expected):
    assert parse_retry_after(value) == expected
