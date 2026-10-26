"""Kalshi data pipeline for the calibrated-forecasting RL project.

Each step reads the previous step's files, so any step can be rerun on its own. API responses are
cached under data/raw/cache/, so an interrupted run picks up where it stopped.

  discover  list series in the categories we care about        -> data/raw/series_catalog.csv
  markets   settled markets and their events, per series          -> data/raw/markets/<SERIES>.jsonl
                                                                      data/raw/events/<SERIES>.json
  prices    market price at each question time (hourly candles)  -> data/raw/prices.csv
  build     one row per market: question, outcome, benchmark
            price, date-based split                               -> data/processed/markets.csv
                                                                      data/processed/summary.txt
  all       markets, then prices, then build

Usage (from the repo root):
  python -m data.pull_kalshi discover
  python -m data.pull_kalshi all
  python -m data.pull_kalshi all --series KXPAYROLLS      # one series, for a quick first run
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from data.kalshi_client import DEFAULT_BASE_URL, KalshiClient, NotFound
from data.series import DISCOVER_CATEGORIES, HELDOUT_GROUPS, SERIES, SINCE_DEFAULT

Log = Callable[[str], None]
PRICE_FIELDS = ["yes_bid", "yes_ask", "spread", "last_trade", "market_prob", "market_prob_source", "price_ts"]


# ---------------------------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------------------------

def parse_time(value: Any) -> pd.Timestamp | None:
    """ISO-8601 string or Unix seconds -> UTC timestamp. None for missing or unparseable values."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ts = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def to_prob(value: Any, key: str = "") -> float | None:
    """Price -> probability in dollars (0 to 1). Current fields end in `_dollars` and are fixed-point
    strings like "0.5600"; older fields may be whole cents (56), which are divided by 100."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    is_cents = not key.endswith("_dollars") and (
        isinstance(value, int) or (isinstance(value, str) and "." not in value))
    return v / 100.0 if is_cents else v


def _price_field(obj: dict, name: str) -> float | None:
    """Reads e.g. `close_dollars`, falling back to `close` (historical responses may use either)."""
    for key in (f"{name}_dollars", name):
        if obj.get(key) not in (None, ""):
            return to_prob(obj[key], key)
    return None


def outcome_from_result(result: Any) -> int | None:
    """Market `result` -> 1 (yes) or 0 (no). Scalar, voided or empty results return None."""
    return {"yes": 1, "no": 0}.get(str(result).lower()) if result is not None else None


def question_time(open_time: pd.Timestamp | None, close_time: pd.Timestamp | None,
                  horizon_hours: float) -> tuple[pd.Timestamp | None, bool]:
    """Question time = close minus the horizon, but at least 1 hour after the market opened.
    Returns (time, clamped). Time is None if the market was open for an hour or less."""
    if close_time is None:
        return None, False
    qt = close_time - pd.Timedelta(hours=horizon_hours)
    clamped = False
    if open_time is not None and qt < open_time + pd.Timedelta(hours=1):
        qt, clamped = open_time + pd.Timedelta(hours=1), True
    if qt >= close_time:
        return None, clamped
    return qt, clamped


def price_at_question_time(candles: Iterable[dict], qt_unix: int, max_spread: float = 0.20) -> dict:
    """Benchmark price from the latest hourly candle that ended at or before the question time.

    market_prob is the bid/ask midpoint when the spread is at most `max_spread`, otherwise the last
    traded price, otherwise None (only a wide quote, which isn't a usable price)."""
    usable = sorted((c for c in candles if (c.get("end_period_ts") or 0) <= qt_unix),
                    key=lambda c: c["end_period_ts"], reverse=True)
    for c in usable:
        bid = _price_field(c.get("yes_bid") or {}, "close")
        ask = _price_field(c.get("yes_ask") or {}, "close")
        price = c.get("price") or {}
        last = _price_field(price, "close")
        if last is None:
            last = _price_field(price, "previous")
        has_quote = bid is not None and ask is not None and ask >= bid
        if not has_quote and last is None:
            continue
        spread = round(ask - bid, 4) if has_quote else None
        if has_quote and spread <= max_spread:
            prob, source = (bid + ask) / 2, "mid"
        elif last is not None:
            prob, source = last, "last_trade"
        else:
            prob, source = None, "wide_quote_only"
        return {"yes_bid": bid, "yes_ask": ask, "spread": spread, "last_trade": last,
                "market_prob": prob, "market_prob_source": source, "price_ts": c["end_period_ts"]}
    return {**{k: None for k in PRICE_FIELDS}, "market_prob_source": "no_candles"}


def build_question_text(market: dict, event: dict) -> str:
    """The market's resolution rule is the most precise wording; fall back to titles."""
    if market.get("rules_primary"):
        return market["rules_primary"].strip()
    parts = [event.get("title") or market.get("title"), market.get("yes_sub_title") or market.get("subtitle")]
    return " - ".join(p for p in parts if p)


def assign_splits(df: pd.DataFrame, heldout_groups: Iterable[str] = HELDOUT_GROUPS,
                  train_frac: float = 0.70, val_frac: float = 0.15) -> pd.DataFrame:
    """Date-based split by event. Every threshold of an event (a ladder) lands in the same split.
    Held-out groups (weather) all go to "heldout". Other events are ordered by their last close
    time: the oldest `train_frac` -> train, the next `val_frac` -> val, the newest -> test."""
    df = df.copy()
    held = df["group"].isin(list(heldout_groups))
    df["split"] = None
    df.loc[held, "split"] = "heldout"

    rest = df.loc[~held]
    if not rest.empty:
        event_close = rest.groupby("event_ticker")["close_time"].max().sort_values(kind="stable")
        n = len(event_close)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        split_of_event = dict(zip(event_close.index, labels))
        df.loc[~held, "split"] = df.loc[~held, "event_ticker"].map(split_of_event)
    return df


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _iso(ts: pd.Timestamp | None) -> str | None:
    return ts.isoformat() if ts is not None else None


# ---------------------------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------------------------

def discover(client: KalshiClient, categories: Iterable[str], raw_dir: Path, log: Log = print) -> pd.DataFrame:
    rows = []
    for category in categories:
        items = client.series_list(category or None)
        log(f"{category or 'ALL CATEGORIES'}: {len(items)} series")
        for s in items:
            rows.append({"ticker": s.get("ticker"), "frequency": s.get("frequency"),
                         "category": s.get("category"), "title": s.get("title")})
    df = pd.DataFrame(rows, columns=["ticker", "frequency", "category", "title"])
    df = df.drop_duplicates("ticker").sort_values(["category", "ticker"]).reset_index(drop=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "series_catalog.csv", index=False)
    if not df.empty:
        log(df.to_string(index=False, max_colwidth=70))
    log(f"\nSaved {len(df)} series to {raw_dir / 'series_catalog.csv'}")
    return df


def pull_markets(client: KalshiClient, series_cfg: dict, raw_dir: Path, since: pd.Timestamp,
                 log: Log = print) -> dict[str, int]:
    """Settled markets (live and historical tiers, merged by ticker) that closed on or after
    `since`, plus each series' settled events."""
    (raw_dir / "markets").mkdir(parents=True, exist_ok=True)
    (raw_dir / "events").mkdir(parents=True, exist_ok=True)
    since_ts = int(since.timestamp())
    counts: dict[str, int] = {}

    for series_ticker in series_cfg:
        try:
            client.series(series_ticker)
        except NotFound:
            log(f"[skip] {series_ticker}: series not found. Check `discover` and data/series.py.")
            continue

        events = {e["event_ticker"]: e for e in client.settled_events(series_ticker)}

        merged: dict[str, dict] = {}
        try:
            for m in client.markets_historical(series_ticker):
                merged[m["ticker"]] = {"source": "historical", "market": m}
        except NotFound:
            log(f"[warn] {series_ticker}: historical markets endpoint not found; using live data only")
        for m in client.settled_markets_live(series_ticker, min_settled_ts=since_ts):
            merged[m["ticker"]] = {"source": "live", "market": m}

        kept = []
        for record in merged.values():
            close = parse_time(record["market"].get("close_time"))
            if close is not None and close >= since:
                kept.append(record)
        kept.sort(key=lambda r: (r["market"].get("close_time") or "", r["market"]["ticker"]))

        with (raw_dir / "markets" / f"{series_ticker}.jsonl").open("w") as f:
            for record in kept:
                f.write(json.dumps(record) + "\n")
        (raw_dir / "events" / f"{series_ticker}.json").write_text(json.dumps(events))

        counts[series_ticker] = len(kept)
        n_hist = sum(r["source"] == "historical" for r in kept)
        log(f"{series_ticker}: {len(kept)} settled markets since {since.date()} "
            f"({n_hist} historical tier, {len(kept) - n_hist} live), {len(events)} events")
    return counts


def pull_prices(client: KalshiClient, series_cfg: dict, raw_dir: Path, lookback_hours: int = 168,
                log: Log = print) -> pd.DataFrame:
    """For each market, the benchmark price at its question time, from hourly candlesticks over the
    `lookback_hours` before it. One request per market (cached)."""
    rows = []
    for series_ticker, cfg in series_cfg.items():
        path = raw_dir / "markets" / f"{series_ticker}.jsonl"
        if not path.exists():
            log(f"[skip] {series_ticker}: no markets file; run the `markets` step first")
            continue
        records = _read_jsonl(path)
        log(f"{series_ticker}: fetching prices for {len(records)} markets")
        for i, record in enumerate(records, 1):
            m = record["market"]
            qt, clamped = question_time(parse_time(m.get("open_time")), parse_time(m.get("close_time")),
                                        cfg["horizon_hours"])
            row = {"market_ticker": m["ticker"], "question_time": _iso(qt), "qt_clamped": clamped}
            if qt is None:
                rows.append({**row, **{k: None for k in PRICE_FIELDS}, "market_prob_source": "open_under_1h"})
                continue
            qt_unix = int(qt.timestamp())
            start = qt_unix - lookback_hours * 3600
            historical = record["source"] == "historical"
            try:
                candles = client.candlesticks(series_ticker, m["ticker"], start, qt_unix, 60, historical)
            except NotFound:
                try:  # the market may have crossed the historical cutoff since `markets` ran
                    candles = client.candlesticks(series_ticker, m["ticker"], start, qt_unix, 60, not historical)
                except NotFound:
                    candles = []
            rows.append({**row, **price_at_question_time(candles, qt_unix)})
            if i % 100 == 0 or i == len(records):
                log(f"  {i}/{len(records)}")

    df = pd.DataFrame(rows, columns=["market_ticker", "question_time", "qt_clamped", *PRICE_FIELDS])
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "prices.csv", index=False)
    log(f"Saved prices for {len(df)} markets to {raw_dir / 'prices.csv'}")
    return df


def build_dataset(series_cfg: dict, raw_dir: Path, processed_dir: Path, since: pd.Timestamp,
                  heldout_groups: Iterable[str] = HELDOUT_GROUPS, log: Log = print) -> pd.DataFrame:
    """Joins markets, events and prices into one row per market and assigns splits."""
    skipped: Counter = Counter()
    rows = []
    for series_ticker, cfg in series_cfg.items():
        path = raw_dir / "markets" / f"{series_ticker}.jsonl"
        if not path.exists():
            continue
        events_path = raw_dir / "events" / f"{series_ticker}.json"
        events = json.loads(events_path.read_text()) if events_path.exists() else {}
        for record in _read_jsonl(path):
            m = record["market"]
            outcome = outcome_from_result(m.get("result"))
            if outcome is None:
                skipped["no yes/no result"] += 1
                continue
            event = events.get(m.get("event_ticker"), {})
            rows.append({
                "market_ticker": m["ticker"],
                "event_ticker": m.get("event_ticker"),
                "series_ticker": series_ticker,
                "group": cfg["group"],
                "question_text": build_question_text(m, event),
                "event_title": event.get("title") or m.get("title"),
                "yes_sub_title": m.get("yes_sub_title") or m.get("subtitle"),
                "strike_type": m.get("strike_type"),
                "floor_strike": m.get("floor_strike"),
                "cap_strike": m.get("cap_strike"),
                "open_time": _iso(parse_time(m.get("open_time"))),
                "close_time": _iso(parse_time(m.get("close_time"))),
                "settlement_time": _iso(parse_time(m.get("settlement_ts") or m.get("expiration_time"))),
                "outcome": outcome,
                "source": record["source"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        log("No markets to build. Run the `markets` step first (and check data/series.py).")
        return df

    prices_path = raw_dir / "prices.csv"
    if prices_path.exists():
        df = df.merge(pd.read_csv(prices_path), on="market_ticker", how="left")
    else:
        log("[warn] no prices.csv yet; benchmark columns will be empty. Run the `prices` step.")
        for col in ["question_time", "qt_clamped", *PRICE_FIELDS]:
            df[col] = None

    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df = df.loc[df["close_time"] >= since].copy()
    df = assign_splits(df, heldout_groups)
    df = df.sort_values(["close_time", "market_ticker"]).reset_index(drop=True)
    df["close_time"] = df["close_time"].map(lambda t: t.isoformat())

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "markets.csv", index=False)
    text = summarize(df, skipped)
    (processed_dir / "summary.txt").write_text(text)
    log(text)
    log(f"Saved {len(df)} markets to {processed_dir / 'markets.csv'}")
    return df


def summarize(df: pd.DataFrame, skipped: Counter | None = None) -> str:
    """Counts and sanity checks to read before Phase 2."""
    d = df.copy()
    d["market_prob"] = pd.to_numeric(d["market_prob"], errors="coerce")
    d["spread"] = pd.to_numeric(d["spread"], errors="coerce")
    d["has_price"] = d["market_prob"].notna()
    d["qt_clamped_b"] = d["qt_clamped"].map(lambda v: v is True or str(v).lower() == "true")
    d["sq_err"] = (d["market_prob"] - d["outcome"]) ** 2          # NaN where there's no price
    d["close_dt"] = pd.to_datetime(d["close_time"], utc=True)
    d["open_hours"] = (d["close_dt"] - pd.to_datetime(d["open_time"], utc=True)).dt.total_seconds() / 3600

    by_series = d.groupby("series_ticker").agg(
        markets=("market_ticker", "size"),
        events=("event_ticker", "nunique"),
        yes_rate=("outcome", "mean"),
        priced=("has_price", "mean"),
        median_spread=("spread", "median"),
        qt_clamped=("qt_clamped_b", "mean"),
        median_open_hours=("open_hours", "median"),
        market_brier=("sq_err", "mean"),
    ).round(3)
    by_split = d.groupby("split").agg(
        markets=("market_ticker", "size"),
        events=("event_ticker", "nunique"),
        yes_rate=("outcome", "mean"),
        first_close=("close_dt", "min"),
        last_close=("close_dt", "max"),
    )
    by_split["yes_rate"] = by_split["yes_rate"].round(3)
    for col in ("first_close", "last_close"):
        by_split[col] = by_split[col].dt.date

    overall_brier = round(float(d["sq_err"].mean()), 4) if d["has_price"].any() else None
    lines = [
        "=" * 80, "Dataset summary", "=" * 80,
        f"Markets: {len(d)}   Events (ladders): {d['event_ticker'].nunique()}   "
        f"With a benchmark price: {int(d['has_price'].sum())} ({d['has_price'].mean():.1%})",
        f"Market-price Brier score, all priced markets: {overall_brier}  "
        "(lower is better; always guessing 0.5 scores 0.25)",
        "", "By series:", by_series.to_string(), "", "By split:", by_split.to_string(),
    ]
    if skipped:
        lines += ["", "Skipped: " + ", ".join(f"{k}: {v}" for k, v in skipped.items())]
    lines += [
        "", "What to check:",
        "- priced well below 1.0 for a series: thin trading; consider a longer horizon or dropping it",
        "- qt_clamped near 1.0: markets open less than horizon_hours before close; lower horizon_hours",
        "- market_brier near 0.25: the market itself has little skill there (hard questions)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def _select_series(arg: str | None) -> dict:
    if not arg:
        return SERIES
    wanted = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SERIES]
    if unknown:
        raise SystemExit(f"Not in data/series.py: {', '.join(unknown)}. Add them there first.")
    return {s: SERIES[s] for s in wanted}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["discover", "markets", "prices", "build", "all"])
    parser.add_argument("--series", help="comma-separated subset of data/series.py, e.g. KXPAYROLLS,KXCPI")
    parser.add_argument("--since", default=SINCE_DEFAULT, help=f"keep markets closing on/after this date (default {SINCE_DEFAULT})")
    parser.add_argument("--category", action="append", help="category for `discover` (repeatable; \"\" lists all)")
    parser.add_argument("--lookback-hours", type=int, default=168, help="candle window before question time (default 168)")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--no-cache", action="store_true", help="ignore and don't write the response cache")
    args = parser.parse_args(argv)

    raw_dir, processed_dir = args.data_dir / "raw", args.data_dir / "processed"
    since = pd.Timestamp(args.since, tz="UTC")
    series_cfg = _select_series(args.series)
    client = KalshiClient(args.base_url, cache_dir=None if args.no_cache else raw_dir / "cache")

    if args.step == "discover":
        discover(client, args.category if args.category is not None else DISCOVER_CATEGORIES, raw_dir)
        return
    if args.step in ("markets", "all"):
        pull_markets(client, series_cfg, raw_dir, since)
    if args.step in ("prices", "all"):
        pull_prices(client, series_cfg, raw_dir, args.lookback_hours)
    if args.step in ("build", "all"):
        build_dataset(series_cfg, raw_dir, processed_dir, since)


if __name__ == "__main__":
    main()