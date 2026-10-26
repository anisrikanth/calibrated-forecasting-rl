"""Kalshi data pipeline for the calibrated-forecasting RL project.

Each step reads the previous step's files, so any step can be rerun on its own. API responses are
cached under data/raw/cache/, so an interrupted run picks up where it stopped.

  discover  list series in the categories we care about        -> data/raw/series_catalog.csv
  markets   settled markets and their events, per series          -> data/raw/markets/<SERIES>.jsonl
            (plus ~13 months before --since, used only as context)    data/raw/events/<SERIES>.json
  prices    market price at each question time (hourly candles)  -> data/raw/prices.csv
  build     one row per market: question, outcome, benchmark
            price, context (earlier results public before the
            question time), date-based split                      -> data/processed/markets.csv
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
DEFAULT_CONTEXT_N = 6   # earlier results shown per question, unless a series sets `context_n`
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


def to_utc(values: pd.Series) -> pd.Series:
    """Column of timestamps -> UTC datetimes. Kalshi mixes whole and fractional seconds
    ("...T06:24:28+00:00" and "...T06:24:28.932309+00:00"), so strings are parsed as general
    ISO-8601 instead of guessing one format from the first rows."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True)
    return pd.to_datetime(values, utc=True, format="ISO8601")


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
    time: the oldest `train_frac` -> train, the next `val_frac` -> val, the newest -> test.
    Events closing on the same UTC day always share a split, so one release day (e.g. CPI and
    core CPI) never straddles two splits."""
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
        # Same-day events take the label of that day's first event (the earlier split).
        day = to_utc(event_close).dt.floor("D")
        label_of_day: dict = {}
        for event, d in day.items():
            label_of_day.setdefault(d, split_of_event[event])
        split_of_event = {event: label_of_day[d] for event, d in day.items()}
        df.loc[~held, "split"] = df.loc[~held, "event_ticker"].map(split_of_event)
    return df


def drop_cross_series_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The same question can be listed under a KX ticker and its older ticker (same family).
    Keeps one copy, preferring the KX one. Two questions match if their tickers are the same
    after the series prefix (KXFED-24DEC-T4.50 vs FED-24DEC-T4.50), or if they close on the same
    day with the same strike and outcome label."""
    if "family" not in df.columns or df.empty:
        return df, 0
    d = df.copy()
    d["_kx_first"] = ~d["series_ticker"].str.startswith("KX")
    d["_suffix"] = [t[len(s):] if t.startswith(s) else t
                    for t, s in zip(d["market_ticker"], d["series_ticker"])]
    d["_close_day"] = to_utc(d["close_time"]).dt.floor("D")
    d = d.sort_values(["_kx_first", "market_ticker"], kind="stable")
    before = len(d)
    d = d.drop_duplicates(subset=["family", "_suffix"])
    d = d.drop_duplicates(subset=["family", "_close_day", "strike_type", "floor_strike",
                                  "cap_strike", "yes_sub_title"])
    return d.drop(columns=["_kx_first", "_suffix", "_close_day"]), before - len(d)


def parse_value(text: Any) -> float | None:
    """Kalshi's `expiration_value` -> number: "29,000" -> 29000.0, "64.00" -> 64.0, "0.3%" -> 0.3.
    Non-numeric values (e.g. "Hold") return None; the text is still shown in the context."""
    if text is None:
        return None
    s = str(text).strip().replace(",", "").rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def collect_context_events(records: list[dict], events: dict, series_ticker: str, family: str) -> list[dict]:
    """One entry per settled event: its actual result (`expiration_value`, the same on every
    market of the event) and when it settled. Settlement comes after the public release, so
    using it as the "known from" time is conservative."""
    by_event: dict[str, dict] = {}
    for record in records:
        m = record["market"]
        event_ticker = m.get("event_ticker")
        value = m.get("expiration_value")
        if not event_ticker or value in (None, "") or event_ticker in by_event:
            continue
        settled = parse_time(m.get("settlement_ts") or m.get("latest_expiration_time") or m.get("expiration_time"))
        if settled is None:
            continue
        event = events.get(event_ticker, {})
        by_event[event_ticker] = {
            "family": family,
            "series_ticker": series_ticker,
            "event_ticker": event_ticker,
            "suffix": event_ticker[len(series_ticker):] if event_ticker.startswith(series_ticker) else event_ticker,
            "label": event.get("title") or m.get("title") or event_ticker,
            "value": str(value).strip(),
            "value_num": parse_value(value),
            "settled": settled,
        }
    return list(by_event.values())


def attach_context(df: pd.DataFrame, ctx: pd.DataFrame, context_n: dict[str, int]) -> pd.DataFrame:
    """For each question, the most recent `context_n[series]` results from the same family that
    settled strictly before its question time. Adds:
      context_json   list of {"label", "value", "settled"}, oldest first
      context_text   the same, rendered for a prompt
      n_context      how many results were found
      context_latest_settled   settlement time of the newest one (for the leakage audit)"""
    df = df.copy()
    empty_text = "No earlier results from this series are available."
    df["context_json"], df["context_text"] = "[]", empty_text
    df["n_context"], df["context_latest_settled"] = 0, None
    if ctx.empty or "question_time" not in df.columns:
        return df

    # The same result can be listed under a KX ticker and its older ticker: keep one (KX first).
    ctx = ctx.copy()
    ctx["_kx_last"] = ~ctx["series_ticker"].str.startswith("KX")
    ctx["_day"] = to_utc(ctx["settled"]).dt.floor("D")
    ctx = (ctx.sort_values(["_kx_last", "event_ticker"], kind="stable")
              .drop_duplicates(["family", "suffix"])
              .drop_duplicates(["family", "_day", "value"]))
    ctx["settled"] = to_utc(ctx["settled"])

    # Per family: settlement times (sorted) and the matching context items, built once.
    by_family = {}
    for fam, g in ctx.groupby("family"):
        g = g.sort_values("settled")
        items = [{"label": r.label, "value": r.value, "settled": r.settled.isoformat(), "_event": r.event_ticker}
                 for r in g.itertuples()]
        by_family[fam] = (g["settled"].reset_index(drop=True), items)
    qt_all = to_utc(df["question_time"])

    json_col, text_col, n_col, latest_col = [], [], [], []
    for family, series, event, qt in zip(df["family"], df["series_ticker"], df["event_ticker"], qt_all):
        if family not in by_family or pd.isna(qt):
            json_col.append("[]"); text_col.append(empty_text); n_col.append(0); latest_col.append(None)
            continue
        settled, fam_items = by_family[family]
        n = context_n.get(series, DEFAULT_CONTEXT_N)
        cut = int(settled.searchsorted(qt, side="left"))                # strictly before qt
        items = [{k: v for k, v in it.items() if k != "_event"}
                 for it in fam_items[max(0, cut - n):cut] if it["_event"] != event]   # belt and braces
        json_col.append(json.dumps(items))
        if items:
            lines = ["Recent results in this series (oldest first; all were public before this question):"]
            lines += [f"- {it['label']}: {it['value']}" for it in items]
            text_col.append("\n".join(lines))
            latest_col.append(items[-1]["settled"])
        else:
            text_col.append(empty_text)
            latest_col.append(None)
        n_col.append(len(items))

    df["context_json"], df["context_text"] = json_col, text_col
    df["n_context"], df["context_latest_settled"] = n_col, latest_col
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
                 context_lookback_days: int = 400, log: Log = print) -> dict[str, int]:
    """Settled markets (live and historical tiers, merged by ticker), plus each series' settled
    events. Markets closing on/after `since` become questions; markets from the
    `context_lookback_days` before that are kept only to supply earlier results as context."""
    (raw_dir / "markets").mkdir(parents=True, exist_ok=True)
    (raw_dir / "events").mkdir(parents=True, exist_ok=True)
    keep_from = since - pd.Timedelta(days=context_lookback_days)
    since_ts = int(keep_from.timestamp())
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

        kept, n_questions = [], 0
        for record in merged.values():
            close = parse_time(record["market"].get("close_time"))
            if close is not None and close >= keep_from:
                kept.append(record)
                n_questions += close >= since
        kept.sort(key=lambda r: (r["market"].get("close_time") or "", r["market"]["ticker"]))

        with (raw_dir / "markets" / f"{series_ticker}.jsonl").open("w") as f:
            for record in kept:
                f.write(json.dumps(record) + "\n")
        (raw_dir / "events" / f"{series_ticker}.json").write_text(json.dumps(events))

        counts[series_ticker] = len(kept)
        log(f"{series_ticker}: {n_questions} settled markets since {since.date()}, "
            f"plus {len(kept) - n_questions} earlier ones kept for context; {len(events)} events")
    return counts


def pull_prices(client: KalshiClient, series_cfg: dict, raw_dir: Path, lookback_hours: int = 168,
                since: pd.Timestamp | None = None, log: Log = print) -> pd.DataFrame:
    """For each question market (closing on/after `since`), the benchmark price at its question
    time, from hourly candlesticks over the `lookback_hours` before it. One request per market
    (cached). Earlier, context-only markets are skipped."""
    rows = []
    for series_ticker, cfg in series_cfg.items():
        path = raw_dir / "markets" / f"{series_ticker}.jsonl"
        if not path.exists():
            log(f"[skip] {series_ticker}: no markets file; run the `markets` step first")
            continue
        records = _read_jsonl(path)
        if since is not None:
            records = [r for r in records
                       if (ct := parse_time(r["market"].get("close_time"))) is not None and ct >= since]
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
    """Joins markets, events and prices into one row per market, attaches each question's context
    (earlier results that were public before its question time), and assigns splits."""
    skipped: Counter = Counter()
    rows, context_events = [], []
    for series_ticker, cfg in series_cfg.items():
        path = raw_dir / "markets" / f"{series_ticker}.jsonl"
        if not path.exists():
            continue
        events_path = raw_dir / "events" / f"{series_ticker}.json"
        events = json.loads(events_path.read_text()) if events_path.exists() else {}
        records = _read_jsonl(path)
        context_events += collect_context_events(records, events, series_ticker,
                                                 cfg.get("family", series_ticker))
        for record in records:
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
                "family": cfg.get("family", series_ticker),
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

    df["close_time"] = to_utc(df["close_time"])
    df = df.loc[df["close_time"] >= since].copy()
    df, n_dupes = drop_cross_series_duplicates(df)
    if n_dupes:
        skipped["same question under KX and older ticker"] += n_dupes
    context_n = {s: cfg.get("context_n", DEFAULT_CONTEXT_N) for s, cfg in series_cfg.items()}
    df = attach_context(df, pd.DataFrame(context_events), context_n)
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
    d["close_dt"] = to_utc(d["close_time"])
    d["open_hours"] = (d["close_dt"] - to_utc(d["open_time"])).dt.total_seconds() / 3600
    if "n_context" not in d.columns:
        d["n_context"], d["context_latest_settled"] = 0, None
    d["n_context"] = pd.to_numeric(d["n_context"], errors="coerce").fillna(0)
    d["has_context"] = d["n_context"] > 0
    latest = to_utc(d["context_latest_settled"])
    n_leaks = int((latest.notna() & (latest >= to_utc(d["question_time"]))).sum())

    by_series = d.groupby("series_ticker").agg(
        markets=("market_ticker", "size"),
        events=("event_ticker", "nunique"),
        yes_rate=("outcome", "mean"),
        priced=("has_price", "mean"),
        median_spread=("spread", "median"),
        qt_clamped=("qt_clamped_b", "mean"),
        median_open_hours=("open_hours", "median"),
        market_brier=("sq_err", "mean"),
        has_context=("has_context", "mean"),
        median_context=("n_context", "median"),
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
        f"Context: {d['has_context'].mean():.1%} of questions have earlier results. "
        f"Leakage audit: {n_leaks} questions with a context entry settled at or after question time "
        f"({'OK' if n_leaks == 0 else 'PROBLEM: investigate before training'})",
        "", "By series:", by_series.to_string(), "", "By split:", by_split.to_string(),
    ]
    if skipped:
        lines += ["", "Skipped: " + ", ".join(f"{k}: {v}" for k, v in skipped.items())]
    lines += [
        "", "What to check:",
        "- priced well below 1.0 for a series: thin trading; consider a longer horizon or dropping it",
        "- qt_clamped near 1.0: markets open less than horizon_hours before close; lower horizon_hours",
        "- market_brier near 0.25: the market itself has little skill there (hard questions)",
        "- has_context well below 1.0: early questions with no earlier results; check context_lookback_days",
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
    parser.add_argument("--context-lookback-days", type=int, default=400,
                        help="also keep markets this many days before --since, as context only (default 400)")
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
        pull_markets(client, series_cfg, raw_dir, since, args.context_lookback_days)
    if args.step in ("prices", "all"):
        pull_prices(client, series_cfg, raw_dir, args.lookback_hours, since)
    if args.step in ("build", "all"):
        build_dataset(series_cfg, raw_dir, processed_dir, since)


if __name__ == "__main__":
    main()
