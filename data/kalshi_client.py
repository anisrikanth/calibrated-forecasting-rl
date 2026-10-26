"""Small read-only client for Kalshi's public market-data API.

Market data endpoints (series, events, markets, candlesticks, historical) need no API key.
Kalshi splits older data into a historical tier: markets settled before the cutoff returned by
GET /historical/cutoff are only available from the /historical/... endpoints.

Docs: https://docs.kalshi.com/getting_started/historical_data
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import requests

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class KalshiError(RuntimeError):
    """The API returned an error we can't recover from by retrying."""


class NotFound(KalshiError):
    """404: the series, market or endpoint doesn't exist."""


class KalshiClient:
    """GET-only client with throttling, retries with exponential backoff, cursor pagination,
    and an optional on-disk cache so interrupted pulls can be resumed cheaply."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path | None = None,
        min_interval: float = 0.2,   # ~5 requests/second; Kalshi's basic tier allows more
        max_retries: int = 6,
        timeout: float = 30.0,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self._sleep = sleep
        self._last_request = 0.0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- core -------------------------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        cache_file = self._cache_path(path, params) if use_cache else None
        if cache_file and cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except ValueError:            # half-written file from an interrupted run: refetch it
                cache_file.unlink()

        last_problem = ""
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_problem = repr(exc)
                self._backoff(attempt)
                continue
            if resp.status_code == 200:
                data = resp.json()
                if cache_file:
                    # Write to a temp file, then rename: a rename is all-or-nothing, so Ctrl+C
                    # can't leave a half-written cache file behind.
                    tmp = cache_file.with_suffix(".tmp")
                    tmp.write_text(json.dumps(data))
                    os.replace(tmp, cache_file)
                return data
            if resp.status_code == 404:
                raise NotFound(f"404 for {path} {params}")
            if resp.status_code in RETRY_STATUSES:
                last_problem = f"HTTP {resp.status_code}"
                self._backoff(attempt)
                continue
            raise KalshiError(f"HTTP {resp.status_code} for {path} {params}: {resp.text[:300]}")
        raise KalshiError(f"Gave up on {path} {params} after {self.max_retries} attempts ({last_problem})")

    def paginate(self, path: str, key: str, params: dict[str, Any] | None = None,
                 limit: int = 1000) -> Iterator[dict]:
        """Yields every item under `key`, following Kalshi's `cursor` until it comes back empty."""
        cursor = None
        while True:
            page_params = dict(params or {}, limit=limit)
            if cursor:
                page_params["cursor"] = cursor
            data = self.get(path, page_params)
            yield from data.get(key) or []
            cursor = data.get("cursor")
            if not cursor:
                return

    # ---- endpoints used by the pipeline -------------------------------------------------

    def historical_cutoff(self) -> dict:
        # Never cached: the cutoff advances over time.
        return self.get("/historical/cutoff", use_cache=False)

    def series(self, series_ticker: str) -> dict:
        return self.get(f"/series/{series_ticker}").get("series", {})

    def series_list(self, category: str | None = None) -> list[dict]:
        return self.get("/series", {"category": category}).get("series") or []

    def settled_events(self, series_ticker: str) -> Iterator[dict]:
        return self.paginate("/events", "events",
                             {"series_ticker": series_ticker, "status": "settled"}, limit=200)

    def settled_markets_live(self, series_ticker: str, min_settled_ts: int | None = None) -> Iterator[dict]:
        return self.paginate("/markets", "markets", {
            "series_ticker": series_ticker, "status": "settled", "min_settled_ts": min_settled_ts})

    def markets_historical(self, series_ticker: str) -> Iterator[dict]:
        return self.paginate("/historical/markets", "markets", {"series_ticker": series_ticker})

    def candlesticks(self, series_ticker: str, ticker: str, start_ts: int, end_ts: int,
                     period_interval: int = 60, historical: bool = False) -> list[dict]:
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        path = (f"/historical/markets/{ticker}/candlesticks" if historical
                else f"/series/{series_ticker}/markets/{ticker}/candlesticks")
        return self.get(path, params).get("candlesticks") or []

    # ---- internals ----------------------------------------------------------------------

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(60.0, 1.0 * 2 ** attempt))

    def _cache_path(self, path: str, params: dict) -> Path | None:
        if not self.cache_dir:
            return None
        key = path + "?" + json.dumps(params, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
