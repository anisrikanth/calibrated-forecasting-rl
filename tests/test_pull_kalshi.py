"""Offline tests for the Kalshi pipeline, using a fake API client."""
import json

import pandas as pd
import pytest

from data.kalshi_client import NotFound
from data.pull_kalshi import (
    assign_splits, build_dataset, build_question_text, discover, outcome_from_result,
    price_at_question_time, pull_markets, pull_prices, question_time, to_prob,
)

UTC = "UTC"
SINCE = pd.Timestamp("2024-01-01", tz=UTC)


# ---- parsing helpers --------------------------------------------------------------------------

@pytest.mark.parametrize("value,key,expected", [
    ("0.5600", "close_dollars", 0.56),
    ("1.0000", "close_dollars", 1.0),
    (56, "close", 0.56),          # old whole-cents field
    ("56", "close", 0.56),
    (1, "close", 0.01),           # 1 cent, not $1
    (0.56, "close", 0.56),        # already dollars
    (None, "close", None),
    ("", "close_dollars", None),
    ("n/a", "close", None),
])
def test_to_prob(value, key, expected):
    assert to_prob(value, key) == expected


def test_outcome_from_result():
    assert outcome_from_result("yes") == 1
    assert outcome_from_result("NO") == 0
    assert outcome_from_result("scalar") is None
    assert outcome_from_result("") is None
    assert outcome_from_result(None) is None


def test_question_time_uses_horizon():
    open_, close = pd.Timestamp("2024-03-01", tz=UTC), pd.Timestamp("2024-03-10 12:00", tz=UTC)
    qt, clamped = question_time(open_, close, 24)
    assert qt == pd.Timestamp("2024-03-09 12:00", tz=UTC) and not clamped


def test_question_time_clamps_to_open_plus_one_hour():
    open_, close = pd.Timestamp("2024-03-10 00:00", tz=UTC), pd.Timestamp("2024-03-10 12:00", tz=UTC)
    qt, clamped = question_time(open_, close, 24)
    assert qt == pd.Timestamp("2024-03-10 01:00", tz=UTC) and clamped


def test_question_time_none_when_open_under_an_hour():
    open_, close = pd.Timestamp("2024-03-10 11:30", tz=UTC), pd.Timestamp("2024-03-10 12:00", tz=UTC)
    assert question_time(open_, close, 24)[0] is None


def candle(ts, bid=None, ask=None, last=None, prev=None, suffix="_dollars"):
    c = {"end_period_ts": ts, "yes_bid": {}, "yes_ask": {}, "price": {}}
    if bid is not None:
        c["yes_bid"]["close" + suffix] = bid
    if ask is not None:
        c["yes_ask"]["close" + suffix] = ask
    if last is not None:
        c["price"]["close" + suffix] = last
    if prev is not None:
        c["price"]["previous" + suffix] = prev
    return c


def test_price_uses_mid_when_spread_is_tight():
    p = price_at_question_time([candle(100, "0.40", "0.46", "0.41")], qt_unix=100)
    assert p["market_prob"] == pytest.approx(0.43) and p["market_prob_source"] == "mid"
    assert p["spread"] == pytest.approx(0.06)


def test_price_falls_back_to_last_trade_when_spread_is_wide():
    p = price_at_question_time([candle(100, "0.05", "0.95", "0.30")], qt_unix=100)
    assert p["market_prob"] == pytest.approx(0.30) and p["market_prob_source"] == "last_trade"


def test_price_ignores_candles_after_question_time():
    candles = [candle(100, "0.40", "0.42"), candle(200, "0.90", "0.92")]
    assert price_at_question_time(candles, qt_unix=150)["market_prob"] == pytest.approx(0.41)


def test_price_walks_back_past_empty_candles_and_uses_previous():
    candles = [candle(100, last="0.20"), candle(200), candle(300, prev="0.25")]
    p = price_at_question_time(candles, qt_unix=300)
    assert p["market_prob"] == pytest.approx(0.25) and p["price_ts"] == 300


def test_price_reads_cent_fields_from_historical_candles():
    p = price_at_question_time([candle(100, 40, 46, suffix="")], qt_unix=100)
    assert p["market_prob"] == pytest.approx(0.43)


def test_price_with_no_usable_candles():
    p = price_at_question_time([candle(500, "0.4", "0.5")], qt_unix=100)
    assert p["market_prob"] is None and p["market_prob_source"] == "no_candles"


def test_to_utc_handles_mixed_fractional_seconds():
    from data.pull_kalshi import to_utc
    s = pd.Series(["2025-11-03T06:24:28+00:00", "2025-11-03T06:24:28.932309+00:00", "2024-01-01T00:00:00Z"])
    out = to_utc(s)
    assert out.notna().all() and str(out.dt.tz) == "UTC"
    assert out[1].microsecond == 932309


@pytest.mark.parametrize("text,expected", [
    ("29,000", 29000.0), ("64.00", 64.0), ("0.3%", 0.3), ("-12,000", -12000.0),
    ("Hold", None), ("", None), (None, None),
])
def test_parse_value(text, expected):
    from data.pull_kalshi import parse_value
    assert parse_value(text) == expected


def test_context_excludes_results_settled_at_or_after_question_time():
    from data.pull_kalshi import attach_context
    q = pd.DataFrame([{"market_ticker": "M", "event_ticker": "E9", "series_ticker": "S", "family": "F",
                       "question_time": "2024-05-01T12:00:00+00:00"}])
    ctx = pd.DataFrame([
        {"family": "F", "series_ticker": "S", "event_ticker": f"E{i}", "suffix": f"E{i}", "label": f"L{i}",
         "value": str(i), "value_num": float(i), "settled": ts(s)}
        for i, s in enumerate(["2024-04-01", "2024-05-01 11:59", "2024-05-01 12:00", "2024-05-02"])
    ])
    out = attach_context(q, ctx, {"S": 6}).iloc[0]
    assert [c["value"] for c in json.loads(out["context_json"])] == ["0", "1"]
    assert out["context_latest_settled"].startswith("2024-05-01T11:59")


def test_question_text_prefers_rules():
    m = {"rules_primary": "  If payrolls exceed 125,000, resolves Yes. ", "yes_sub_title": "Above 125k"}
    assert build_question_text(m, {"title": "Jobs"}) == "If payrolls exceed 125,000, resolves Yes."
    assert build_question_text({"yes_sub_title": "Above 125k"}, {"title": "Jobs in May"}) == "Jobs in May - Above 125k"


def test_splits_keep_ladders_together_and_are_chronological():
    rows = []
    for i in range(20):  # 20 events, 3 thresholds each
        close = pd.Timestamp("2024-01-15", tz=UTC) + pd.DateOffset(months=i)
        for t in range(3):
            rows.append({"event_ticker": f"E{i:02d}", "group": "econ", "close_time": close})
    rows += [{"event_ticker": "W1", "group": "weather", "close_time": pd.Timestamp("2024-02-01", tz=UTC)}]
    df = assign_splits(pd.DataFrame(rows))

    assert (df.groupby("event_ticker")["split"].nunique() == 1).all()
    assert df.loc[df.group == "weather", "split"].eq("heldout").all()
    order = {"train": 0, "val": 1, "test": 2}
    econ = df[df.group == "econ"].sort_values("close_time")
    assert econ["split"].map(order).is_monotonic_increasing
    counts = econ.drop_duplicates("event_ticker")["split"].value_counts()
    assert (counts["train"], counts["val"], counts["test"]) == (14, 3, 3)


def test_same_day_events_never_straddle_a_split_boundary():
    rows = []
    for i in range(10):  # 10 events on 10 different days -> 7 train, 2 val, 1 test
        rows.append({"event_ticker": f"E{i}", "group": "econ",
                     "close_time": pd.Timestamp("2024-01-10 13:30", tz=UTC) + pd.DateOffset(months=i)})
    # A second release on the day of the first val event (e.g. core CPI alongside CPI), later that day
    boundary_day = rows[7]["close_time"]
    rows.append({"event_ticker": "E7b", "group": "econ", "close_time": boundary_day - pd.Timedelta(hours=1)})
    rows.append({"event_ticker": "E6b", "group": "econ",
                 "close_time": rows[6]["close_time"] + pd.Timedelta(hours=2)})  # same day as last train event
    split = assign_splits(pd.DataFrame(rows)).set_index("event_ticker")["split"]
    assert split["E6"] == split["E6b"] == "train"
    assert split["E7"] == split["E7b"]


def test_cross_series_duplicates_keep_the_kx_copy():
    from data.pull_kalshi import drop_cross_series_duplicates
    base = {"family": "FED", "close_time": pd.Timestamp("2024-12-18 19:00", tz=UTC),
            "strike_type": "greater", "cap_strike": None}
    df = pd.DataFrame([
        dict(base, market_ticker="FED-24DEC-T4.50", series_ticker="FED", floor_strike=4.5, yes_sub_title="Above 4.50%"),
        dict(base, market_ticker="KXFED-24DEC-T4.50", series_ticker="KXFED", floor_strike=4.5, yes_sub_title="Above 4.50%"),
        dict(base, market_ticker="KXFED-24DEC-T4.25", series_ticker="KXFED", floor_strike=4.25, yes_sub_title="Above 4.25%"),
        # Different ticker naming, same question: caught by the strike/day/label match
        dict(base, market_ticker="FED-1218-T4.25", series_ticker="FED", floor_strike=4.25, yes_sub_title="Above 4.25%"),
        # Same day, different family: kept
        dict(base, family="CPI", market_ticker="CPI-24DEC-T0.3", series_ticker="CPI", floor_strike=4.5,
             yes_sub_title="Above 4.50%"),
    ])
    out, dropped = drop_cross_series_duplicates(df)
    assert dropped == 2
    assert set(out["market_ticker"]) == {"KXFED-24DEC-T4.50", "KXFED-24DEC-T4.25", "CPI-24DEC-T0.3"}


# ---- end to end with a fake API ------------------------------------------------------------------

def ts(s):
    return pd.Timestamp(s, tz=UTC)


def make_market(ticker, event, close, result="yes", open_days=10, strike=100, value=None):
    close_t = ts(close)
    return {
        "ticker": ticker, "event_ticker": event, "result": result, "expiration_value": value,
        # Some open times carry fractional seconds, as real Kalshi data does
        "open_time": (close_t - pd.Timedelta(days=open_days, microseconds=932309 if strike == 200 else 0)).isoformat(),
        "close_time": close_t.isoformat(),
        "settlement_ts": (close_t + pd.Timedelta(hours=1)).isoformat(),
        "rules_primary": f"If the value exceeds {strike}, the market resolves to Yes.",
        "yes_sub_title": f"Above {strike}", "strike_type": "greater", "floor_strike": strike,
    }


class FakeClient:
    def __init__(self):
        econ_hist, econ_live, events = [], [], {}
        for i in range(10):
            close = (ts("2024-01-12 13:30") + pd.DateOffset(months=i)).isoformat()
            ev = f"SER-24M{i:02d}"
            events[ev] = {"event_ticker": ev, "title": f"Value in month {i}"}
            value = f"{(i + 1) * 1000:,}"          # "1,000", "2,000", ... like Kalshi's strings
            low = make_market(f"{ev}-T100", ev, close, "yes", strike=100, value=value)
            high = make_market(f"{ev}-T200", ev, close, "no", strike=200, value=value)
            (econ_hist if i < 5 else econ_live).extend([low, high])
        # Same ticker in both tiers: live copy should win.
        dup = dict(econ_hist[0], rules_primary="LIVE COPY")
        econ_live.append(dup)
        # Closed before the cutoff: not a question, but its result is context for January 2024.
        econ_hist.append(make_market("SER-23DEC-T100", "SER-23DEC", "2023-12-12 13:30", value="500"))
        events["SER-23DEC"] = {"event_ticker": "SER-23DEC", "title": "Old"}
        # Scalar result: dropped at build.
        econ_live.append(make_market("SER-24M09-T300", "SER-24M09", econ_live[-2]["close_time"], "scalar"))

        wx = [make_market(f"WX-24JUL0{d}-B80", f"WX-24JUL0{d}", f"2024-07-0{d} 23:59", "yes", open_days=1,
                          value=f"8{d}.00")
              for d in range(1, 4)]
        # Older-name copy of one SER question (same family): should be dropped as a duplicate.
        legacy = dict(econ_hist[2], ticker="OLD" + econ_hist[2]["ticker"][3:],
                      event_ticker="OLD" + econ_hist[2]["event_ticker"][3:])
        self.data = {
            "OLD": {"events": {legacy["event_ticker"]: {"event_ticker": legacy["event_ticker"], "title": "Old name"}},
                    "hist": [legacy], "live": []},
            "SER": {"events": events, "hist": econ_hist, "live": econ_live},
            "WX": {"events": {m["event_ticker"]: {"event_ticker": m["event_ticker"], "title": "High temp"} for m in wx},
                   "hist": [], "live": wx},
        }
        self.candle_calls = []

    def series(self, t):
        if t not in self.data:
            raise NotFound(t)
        return {"ticker": t}

    def series_list(self, category=None):
        return [{"ticker": "SER", "title": "Series", "frequency": "monthly", "category": "Economics"}]

    def settled_events(self, t):
        return iter(self.data[t]["events"].values())

    def markets_historical(self, t):
        return iter(self.data[t]["hist"])

    def settled_markets_live(self, t, min_settled_ts=None):
        return iter(self.data[t]["live"])

    def candlesticks(self, series_ticker, ticker, start_ts, end_ts, period_interval=60, historical=False):
        self.candle_calls.append((ticker, historical))
        if ticker.endswith("-T200"):
            return [candle(end_ts - 3600, "0.10", "0.14")]   # priced at 0.12
        if ticker.startswith("WX"):
            return []                                         # no trading
        return [candle(end_ts, "0.80", "0.84")]               # priced at 0.82


CFG = {"SER": {"group": "econ", "family": "SER", "horizon_hours": 24},
       "OLD": {"group": "econ", "family": "SER", "horizon_hours": 24},
       "WX": {"group": "weather", "horizon_hours": 18},
       "MISSING": {"group": "econ", "horizon_hours": 24}}


def test_pipeline_end_to_end(tmp_path):
    client = FakeClient()
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    logs = []

    counts = pull_markets(client, CFG, raw, SINCE, log=logs.append)
    assert counts == {"SER": 22, "OLD": 1, "WX": 3}  # SER: 20 + 1 scalar + the Dec 2023 context-only market
    assert any("[skip] MISSING" in line for line in logs)
    records = [json.loads(l) for l in (raw / "markets" / "SER.jsonl").read_text().splitlines()]
    dup = next(r for r in records if r["market"]["ticker"] == "SER-24M00-T100")
    assert dup["source"] == "live" and dup["market"]["rules_primary"] == "LIVE COPY"

    prices = pull_prices(client, CFG, raw, since=SINCE, log=logs.append)
    assert len(prices) == 25                         # the context-only 2023 market isn't priced
    assert "SER-23DEC-T100" not in set(prices["market_ticker"])
    hist_calls = {t for t, h in client.candle_calls if h}
    assert "SER-24M01-T100" in hist_calls and "SER-24M00-T100" not in hist_calls

    df = build_dataset(CFG, raw, processed, SINCE, log=logs.append)
    assert len(df) == 23                             # scalar market and the older-name duplicate dropped
    assert df["market_ticker"].str.endswith("-24M01-T100").sum() == 1   # one copy of the duplicated question
    assert set(df["split"]) == {"train", "val", "test", "heldout"}
    assert (df.groupby("event_ticker")["split"].nunique() == 1).all()
    assert df.loc[df.series_ticker == "WX", "split"].eq("heldout").all()

    row = df.set_index("market_ticker").loc["SER-24M03-T200"]
    assert row["outcome"] == 0 and row["market_prob"] == pytest.approx(0.12)
    assert row["question_time"] == (ts("2024-04-12 13:30") - pd.Timedelta(hours=24)).isoformat()
    assert row["question_text"].startswith("If the value exceeds 200")

    wx = df[df.series_ticker == "WX"]
    assert wx["market_prob"].isna().all()
    assert wx["qt_clamped"].astype(str).eq("False").all()   # open 24h, horizon 18h: no clamp

    # Context: earlier results of the same family, settled before the question time
    ctx = df.set_index("market_ticker")
    first = json.loads(ctx.loc["SER-24M00-T100", "context_json"])
    assert [c["value"] for c in first] == ["500"]                          # December 2023 only
    sixth = json.loads(ctx.loc["SER-24M06-T100", "context_json"])
    assert [c["value"] for c in sixth] == ["1,000", "2,000", "3,000", "4,000", "5,000", "6,000"]  # capped at 6
    assert ctx.loc["SER-24M06-T100", "n_context"] == 6
    assert "- Value in month 5: 6,000" in ctx.loc["SER-24M06-T100", "context_text"]
    third = json.loads(ctx.loc["SER-24M02-T100", "context_json"])
    assert [c["value"] for c in third] == ["500", "1,000", "2,000"]        # legacy duplicate of month 1 dropped
    assert list(ctx.loc[["WX-24JUL01-B80", "WX-24JUL02-B80", "WX-24JUL03-B80"], "n_context"]) == [0, 1, 2]
    assert ctx.loc["WX-24JUL01-B80", "context_text"].startswith("No earlier results")
    for _, r in df.iterrows():                                             # nothing from after the question
        for c in json.loads(r["context_json"]):
            assert ts(c["settled"]) < ts(r["question_time"])

    summary = (processed / "summary.txt").read_text()
    assert "Leakage audit: 0 questions" in summary
    assert "Markets: 23" in summary and "no yes/no result: 1" in summary
    assert "same question under KX and older ticker: 1" in summary
    saved = pd.read_csv(processed / "markets.csv")
    assert len(saved) == 23 and list(saved["close_time"]) == sorted(saved["close_time"])


def test_discover_writes_catalog(tmp_path):
    df = discover(FakeClient(), ["Economics"], tmp_path, log=lambda s: None)
    assert list(df["ticker"]) == ["SER"]
    assert (tmp_path / "series_catalog.csv").exists()
