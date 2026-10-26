"""Offline tests for the Kalshi client: pagination, retries, 404s, caching, endpoint routing."""
import pytest

from data.kalshi_client import KalshiClient, KalshiError, NotFound


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def make_client(responses, **kw):
    session = FakeSession(responses)
    client = KalshiClient("https://example.test/v2", session=session, sleep=lambda s: None, **kw)
    return client, session


def test_paginate_follows_cursor_until_empty():
    client, session = make_client([
        FakeResponse(200, {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "c1"}),
        FakeResponse(200, {"markets": [{"ticker": "C"}], "cursor": ""}),
    ])
    tickers = [m["ticker"] for m in client.paginate("/markets", "markets", {"series_ticker": "S"})]
    assert tickers == ["A", "B", "C"]
    assert "cursor" not in session.calls[0][1]
    assert session.calls[1][1]["cursor"] == "c1"
    assert session.calls[0][1]["limit"] == 1000


def test_retries_on_429_then_succeeds():
    client, session = make_client([FakeResponse(429), FakeResponse(503), FakeResponse(200, {"ok": 1})])
    assert client.get("/x") == {"ok": 1}
    assert len(session.calls) == 3


def test_gives_up_after_max_retries():
    client, _ = make_client([FakeResponse(429)] * 3, max_retries=3)
    with pytest.raises(KalshiError, match="Gave up"):
        client.get("/x")


def test_404_raises_not_found_without_retrying():
    client, session = make_client([FakeResponse(404)])
    with pytest.raises(NotFound):
        client.get("/series/NOPE")
    assert len(session.calls) == 1


def test_other_errors_raise_immediately():
    client, _ = make_client([FakeResponse(400, text="bad request")])
    with pytest.raises(KalshiError, match="HTTP 400"):
        client.get("/x")


def test_none_params_are_dropped():
    client, session = make_client([FakeResponse(200, {})])
    client.get("/markets", {"series_ticker": "S", "min_settled_ts": None, "cursor": ""})
    assert session.calls[0][1] == {"series_ticker": "S"}


def test_cache_serves_repeat_requests(tmp_path):
    client, session = make_client([FakeResponse(200, {"v": 1})], cache_dir=tmp_path)
    assert client.get("/a", {"p": 1}) == {"v": 1}
    assert client.get("/a", {"p": 1}) == {"v": 1}      # from cache; no second response queued
    assert len(session.calls) == 1


def test_broken_cache_file_is_refetched(tmp_path):
    client, session = make_client([FakeResponse(200, {"v": 2})], cache_dir=tmp_path)
    client._cache_path("/a", {}).write_text('{"v": ')       # truncated by an interrupted run
    assert client.get("/a") == {"v": 2}
    assert len(session.calls) == 1
    assert client.get("/a") == {"v": 2}                       # now served from the repaired cache
    assert not list(tmp_path.glob("*.tmp"))


def test_historical_cutoff_is_never_cached(tmp_path):
    client, session = make_client([FakeResponse(200, {"market_settled_ts": "a"}),
                                   FakeResponse(200, {"market_settled_ts": "b"})], cache_dir=tmp_path)
    assert client.historical_cutoff()["market_settled_ts"] == "a"
    assert client.historical_cutoff()["market_settled_ts"] == "b"
    assert len(session.calls) == 2


def test_candlesticks_routes_live_and_historical():
    client, session = make_client([FakeResponse(200, {"candlesticks": [1]}),
                                   FakeResponse(200, {"candlesticks": [2]})])
    assert client.candlesticks("SER", "MKT", 0, 10, historical=False) == [1]
    assert client.candlesticks("SER", "MKT", 0, 10, historical=True) == [2]
    assert session.calls[0][0].endswith("/series/SER/markets/MKT/candlesticks")
    assert session.calls[1][0].endswith("/historical/markets/MKT/candlesticks")
    assert session.calls[0][1] == {"start_ts": 0, "end_ts": 10, "period_interval": 60}
