# Data pipeline

Pulls resolved Kalshi markets and turns them into a leak-controlled, date-split dataset. Kalshi's market-data endpoints are public, so **no API key is needed**. Run everything from the repo root.

## Run it

```bash
# 1. See which series tickers exist, then edit data/series.py to match
python -m data.pull_kalshi discover

# 2. Quick first run on the one confirmed series
python -m data.pull_kalshi all --series KXPAYROLLS

# 3. Full run
python -m data.pull_kalshi all
```

Each step can also be run alone: `markets`, `prices`, `build`. API responses are cached in `data/raw/cache/`, so if a run stops, rerunning it skips everything already fetched. Use `--no-cache` to force fresh requests.

Rough timing at the default ~5 requests/second: economic series take a few minutes each; a daily weather series takes 15–25 minutes (one price request per market, several thousand markets).

## What each step does

| Step | Output | Notes |
| --- | --- | --- |
| `discover` | `raw/series_catalog.csv` | Lists series in the Economics and Climate and Weather categories |
| `markets` | `raw/markets/<SERIES>.jsonl`, `raw/events/<SERIES>.json` | Settled markets closing on/after `--since` (default 2024-01-01). Merges Kalshi's live and historical tiers by ticker |
| `prices` | `raw/prices.csv` | Benchmark price at question time from hourly candlesticks |
| `build` | `processed/markets.csv`, `processed/summary.txt` | One row per market, with outcome and split |

## Key choices (and where to change them)

- **Leakage cutoff:** only markets that closed on or after 2024-01-01, because Llama 3.2's knowledge cutoff is December 2023 (`SINCE_DEFAULT` in `series.py`).
- **Question time:** close time minus `horizon_hours` (24 for economic series, 18 for weather), but never earlier than 1 hour after the market opened. Check `qt_clamped` and `median_open_hours` in the summary and adjust per series.
- **Benchmark price:** from the last hourly candle ending at or before the question time. The bid/ask midpoint if the spread is at most 0.20, otherwise the last traded price. Never shown to the model.
- **Splits:** weather series all go to `heldout`. Economic events (ladders) are sorted by their last close time: oldest 70% train, next 15% val, newest 15% test. Every threshold of the same event stays in the same split.
- **Outcomes:** `result` of "yes" → 1, "no" → 0. Scalar or voided markets are dropped (counted under "Skipped" in the summary).

## `processed/markets.csv` columns

| Column | Meaning |
| --- | --- |
| `market_ticker`, `event_ticker`, `series_ticker` | Kalshi identifiers; `event_ticker` groups a ladder of thresholds |
| `group` | `econ` or `weather` |
| `question_text` | The market's resolution rule (`rules_primary`), or titles if missing |
| `event_title`, `yes_sub_title` | Human-readable titles |
| `strike_type`, `floor_strike`, `cap_strike` | Threshold definition (e.g. `greater`, 125000) |
| `open_time`, `close_time`, `settlement_time` | UTC, ISO-8601 |
| `question_time`, `qt_clamped` | When the question is asked; whether it was pushed later to fit the market's open time |
| `outcome` | 1 = resolved yes, 0 = no |
| `yes_bid`, `yes_ask`, `spread`, `last_trade` | Prices (0 to 1) at question time |
| `market_prob`, `market_prob_source` | Benchmark probability and how it was chosen (`mid`, `last_trade`, `wide_quote_only`, `no_candles`) |
| `price_ts` | Unix end time of the candle the price came from |
| `source` | `live` or `historical` Kalshi tier |
| `split` | `train`, `val`, `test`, or `heldout` |

## Not built yet

Context snapshots (previous releases, consensus forecasts, archived weather forecasts) are the next piece of Phase 1. They'll be keyed on `question_time`, so nothing dated after it can reach the model.
