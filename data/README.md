# Data pipeline

Pulls resolved Kalshi markets and turns them into a leak-controlled, date-split dataset. Kalshi's market-data endpoints are public, so **no API key is needed**. Run everything from the repo root.

## Run it

```bash
# 1. (Optional) list every series in the Economics and Climate and Weather categories
python -m data.pull_kalshi discover

# 2. Quick first run on one series
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
| `markets` | `raw/markets/<SERIES>.jsonl`, `raw/events/<SERIES>.json` | Settled markets closing on/after `--since` (default 2024-01-01), plus the 400 days before it as context only (`--context-lookback-days`). Merges Kalshi's live and historical tiers by ticker |
| `prices` | `raw/prices.csv` | Benchmark price at question time from hourly candlesticks (question markets only) |
| `build` | `processed/markets.csv`, `processed/summary.txt` | One row per market, with outcome, context and split |

## Key choices (and where to change them)

- **Leakage cutoff:** only markets that closed on or after 2024-01-01, because Llama 3.2's knowledge cutoff is December 2023 (`SINCE_DEFAULT` in `series.py`).
- **Question time:** close time minus `horizon_hours` (24 for economic series, 18 for weather), but never earlier than 1 hour after the market opened. Check `qt_clamped` and `median_open_hours` in the summary and adjust per series.
- **Benchmark price:** from the last hourly candle ending at or before the question time. The bid/ask midpoint if the spread is at most 0.20, otherwise the last traded price. Never shown to the model.
- **Series and duplicates:** every series is pulled under both its KX ticker and its older ticker (e.g. `KXCPI` and `CPI`), which share a `family`. If the same question appears under both, `build` keeps the KX copy.
- **Splits:** weather series all go to `heldout`. Economic events (ladders) are sorted by their last close time: oldest 70% train, next 15% val, newest 15% test. Every threshold of the same event stays in the same split, and events closing on the same day (e.g. CPI and core CPI) always share a split.
- **Outcomes:** `result` of "yes" → 1, "no" → 0. Scalar or voided markets are dropped (counted under "Skipped" in the summary).
- **Context:** each question gets the most recent actual results from the same family (6 for economic series, 7 for daily temperatures), taken from Kalshi's `expiration_value` on earlier events. A result is only included if its event **settled strictly before the question time**; settlement comes after the public release, so this is conservative. For weather, that usually means yesterday's high isn't in yet at 7am, so the context runs through the day before. The summary's leakage audit counts any context entry dated at or after its question time and should always read 0.

## `processed/markets.csv` columns

| Column | Meaning |
| --- | --- |
| `market_ticker`, `event_ticker`, `series_ticker` | Kalshi identifiers; `event_ticker` groups a ladder of thresholds |
| `family` | The underlying series, shared by its KX and older tickers (e.g. `CPI`) |
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
| `context_json` | Earlier results as a list of `{"label", "value", "settled"}`, oldest first |
| `context_text` | The same, rendered as text for the prompt |
| `n_context`, `context_latest_settled` | How many results; when the newest one settled (for the leakage audit) |
| `split` | `train`, `val`, `test`, or `heldout` |

## Possible additions

Context is currently each series' own recent results. Richer context (consensus forecasts for economic releases, archived weather forecasts) would need outside data sources; it's an option if the model plateaus.
