"""Evaluation shared by the baseline and training notebooks.

Picks the evaluation questions, renders prompts, collects sampled answers, and builds the result
tables (scores per method, per series, consistency, leak probe). The model call itself stays in the
notebook, because it needs the GPU; everything here runs anywhere and is unit-tested.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from env import metrics as M
from env.parser import extract_probability
from env.prompt import build_messages, chat_template_kwargs

REQUIRED_COLUMNS = {"question_time", "question_text", "context_text", "outcome", "split", "market_prob",
                    "series_ticker", "event_ticker", "market_ticker", "close_time"}


def load_markets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"markets.csv is missing {sorted(missing)}: upload the latest build")
    df["close_dt"] = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")
    return df


def pick_eval_sets(df: pd.DataFrame, weather_events: int = 60, seed: int = 3407,
                   weather_from: str = "2026-01-01") -> pd.DataFrame:
    """val_econ = every validation question; heldout_weather = whole days sampled from `weather_from`
    on. Adds `base_rate`: each econ series' yes-rate in train, each weather series' yes-rate in
    held-out days before `weather_from`."""
    val = df[df["split"] == "val"]
    held = df[df["split"] == "heldout"]
    days = held.loc[held["close_dt"] >= weather_from, "event_ticker"].drop_duplicates()
    picked = days.sample(n=min(weather_events, len(days)), random_state=seed) if len(days) else days
    weather = held[held["event_ticker"].isin(picked)]

    train = df[df["split"] == "train"]
    base_rates = pd.concat([train.groupby("series_ticker")["outcome"].mean(),
                            held[held["close_dt"] < weather_from].groupby("series_ticker")["outcome"].mean()])
    out = pd.concat([val.assign(eval_set="val_econ"), weather.assign(eval_set="heldout_weather")],
                    ignore_index=True)
    fallback = train["outcome"].mean() if len(train) else 0.5
    out["base_rate"] = out["series_ticker"].map(base_rates).fillna(fallback)
    out["always_0.5"] = 0.5
    out["market"] = out["market_prob"]
    return out


def render_prompt(tokenizer: Any, row: dict, include_context: bool = True) -> str:
    """Chat-formatted prompt text, with the question's date passed to the chat template."""
    return tokenizer.apply_chat_template(build_messages(row, include_context), tokenize=False,
                                         add_generation_prompt=True, **chat_template_kwargs(row))


def collect_answers(rows: Sequence[dict], outputs: Sequence[Any], condition: str) -> pd.DataFrame:
    """Flattens vLLM outputs (one per prompt, each with `.outputs` samples) into one row per answer."""
    records = []
    for row, out in zip(rows, outputs):
        for k, c in enumerate(out.outputs):
            records.append({"market_ticker": row["market_ticker"], "condition": condition, "sample": k,
                            "text": c.text, "prob": extract_probability(c.text),
                            "n_tokens": len(getattr(c, "token_ids", []) or [])})
    return pd.DataFrame(records, columns=["market_ticker", "condition", "sample", "text", "prob", "n_tokens"])


def add_forecast_columns(eval_df: pd.DataFrame, answers: pd.DataFrame, condition: str, prefix: str) -> pd.DataFrame:
    """Adds `<prefix>_single` (first answer) and `<prefix>_mean4` (average of readable answers).
    Unreadable or missing answers count as 0.5."""
    a = answers[answers["condition"] == condition]
    first = a[a["sample"] == 0].set_index("market_ticker")["prob"]
    mean = a.groupby("market_ticker")["prob"].mean()
    eval_df = eval_df.copy()
    eval_df[f"{prefix}_single"] = eval_df["market_ticker"].map(first).astype(float).fillna(0.5)
    eval_df[f"{prefix}_mean4"] = eval_df["market_ticker"].map(mean).astype(float).fillna(0.5)
    return eval_df


def scored_subset(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Questions with a market price: every method is scored on the same set."""
    return eval_df[eval_df["market_prob"].notna()]


def score_table(eval_df: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    for eval_set, g in scored_subset(eval_df).groupby("eval_set"):
        for m in methods:
            s = M.summarize(g[m], g["outcome"])
            rows.append({"eval_set": eval_set, "method": m,
                         **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}})
    return pd.DataFrame(rows)


def by_series_table(eval_df: pd.DataFrame, methods: Iterable[str], eval_set: str = "val_econ") -> pd.DataFrame:
    g_all = scored_subset(eval_df)
    g_all = g_all[g_all["eval_set"] == eval_set]
    rows = []
    for series, g in g_all.groupby("series_ticker"):
        rows.append({"series_ticker": series, "n": len(g),
                     **{m: round(M.brier(g[m], g["outcome"]), 4) for m in methods}})
    return pd.DataFrame(rows).set_index("series_ticker") if rows else pd.DataFrame()


def consistency_table(eval_df: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    rows = []
    econ = scored_subset(eval_df)
    econ = econ[econ["eval_set"] == "val_econ"]
    weather = eval_df[eval_df["eval_set"] == "heldout_weather"]
    for m in methods:
        lad = M.ladder_violations(econ, m)
        part = M.partition_sum_error(weather, m)
        rows.append({"method": m, "ladder_pairs": lad["pairs"],
                     "ladder_violation_rate": round(lad["violation_rate"], 3),
                     "weather_days": part["events"],
                     "bracket_sum_abs_error": round(part["mean_abs_sum_error"], 3)})
    return pd.DataFrame(rows)


def leak_table(eval_df: pd.DataFrame, noctx_column: str) -> pd.DataFrame:
    """Share of the market's edge over the base rate that the no-context model captures:
    (base-rate Brier − no-context Brier) ÷ (base-rate Brier − market Brier). Above 0.5 is flagged;
    blank where the market beats the base rate by less than 0.02."""
    g_all = scored_subset(eval_df)
    g_all = g_all[g_all["eval_set"] == "val_econ"]
    rows = []
    for series, g in g_all.groupby("series_ticker"):
        rows.append({"series_ticker": series, "n": len(g),
                     "market_brier": round(M.brier(g["market"], g["outcome"]), 4),
                     "base_rate_brier": round(M.brier(g["base_rate"], g["outcome"]), 4),
                     "noctx_brier": round(M.brier(g[noctx_column], g["outcome"]), 4)})
    t = pd.DataFrame(rows).set_index("series_ticker")
    edge = t["base_rate_brier"] - t["market_brier"]
    t["share_of_market_edge"] = ((t["base_rate_brier"] - t["noctx_brier"]) / edge).where(edge >= 0.02).round(2)
    t["flag"] = np.where(t["share_of_market_edge"] > 0.5, "CHECK", "ok")
    return t
