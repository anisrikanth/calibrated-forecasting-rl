"""Evaluation shared by the baseline and training notebooks.

Picks the evaluation questions, renders prompts, collects sampled answers, and builds the result
tables (scores per method, per series, consistency, leak probe). The model call itself stays in the
notebook, because it needs the GPU; everything here runs anywhere and is unit-tested.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from env import metrics as M
from env.baselines import add_rule_baselines
from env.features import recent_hit_rate
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
                   weather_from: str = "2026-01-01", econ_split: str = "val") -> pd.DataFrame:
    """val_econ = every validation question (test_econ with econ_split="test": the untouched test
    split, scored once at the end); heldout_weather = whole days sampled from `weather_from` on
    (none with weather_events=0). Adds baselines: `base_rate` (each econ series' yes-rate in train, each weather series'
    yes-rate in held-out days before `weather_from`), `always_0.5`, `market`, `recent_hits`
    (share of the recent results in the context that would have resolved YES under this question's
    rule), `latest_rule` (1 if the latest result already would have, else 0) and `logistic` (a
    logistic regression on the key numbers, fitted on train; env/baselines.py). The last three use
    base_rate where there is no numeric threshold or no recent result."""
    val = df[df["split"] == econ_split]
    held = df[df["split"] == "heldout"]
    days = held.loc[held["close_dt"] >= weather_from, "event_ticker"].drop_duplicates()
    picked = days.sample(n=min(weather_events, len(days)), random_state=seed) if len(days) else days
    weather = held[held["event_ticker"].isin(picked)]

    train = df[df["split"] == "train"]
    base_rates = pd.concat([train.groupby("series_ticker")["outcome"].mean(),
                            held[held["close_dt"] < weather_from].groupby("series_ticker")["outcome"].mean()])
    out = pd.concat([val.assign(eval_set=f"{econ_split}_econ"), weather.assign(eval_set="heldout_weather")],
                    ignore_index=True)
    fallback = train["outcome"].mean() if len(train) else 0.5
    out["base_rate"] = out["series_ticker"].map(base_rates).fillna(fallback)
    out["always_0.5"] = 0.5
    out["market"] = out["market_prob"]
    hits = pd.Series([recent_hit_rate(r) for r in out.to_dict("records")], index=out.index, dtype=float)
    out["recent_hits"] = hits.fillna(out["base_rate"])
    return add_rule_baselines(out, train, out["base_rate"])


def render_prompt(tokenizer: Any, row: dict, include_context: bool = True, key_numbers: bool = False,
                  key_parts: Optional[Sequence[str]] = None, estimate_first: bool = False) -> str:
    """Chat-formatted prompt text, with the question's date passed to the chat template."""
    return tokenizer.apply_chat_template(build_messages(row, include_context, key_numbers, key_parts, estimate_first),
                                         tokenize=False, add_generation_prompt=True, **chat_template_kwargs(row))


def collect_answers(rows: Sequence[dict], outputs: Sequence[Any], condition: str) -> pd.DataFrame:
    """Flattens vLLM outputs (one per prompt, each with `.outputs` samples) into one row per answer."""
    records = []
    for row, out in zip(rows, outputs):
        for k, c in enumerate(out.outputs):
            records.append({"market_ticker": row["market_ticker"], "condition": condition, "sample": k,
                            "text": c.text, "prob": extract_probability(c.text),
                            "n_tokens": len(getattr(c, "token_ids", []) or [])})
    return pd.DataFrame(records, columns=["market_ticker", "condition", "sample", "text", "prob", "n_tokens"])


def text_key(text: Any) -> str:
    """Short fingerprint of an answer's text (line endings and outer whitespace ignored), so a
    re-run can be checked against stored answers without storing the text."""
    norm = str(text).replace("\r\n", "\n").strip() if isinstance(text, str) else ""
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def reproduction_check(new: pd.DataFrame, stored: pd.DataFrame) -> dict:
    """Compares re-generated answers with stored ones for the same questions and samples (same
    prompt and sampling seed). Both have market_ticker, sample, prob; `new` has text, `stored` has
    text_key. A correctly loaded model gives mostly identical answers (0.88 of answers in our
    reproduction of the untrained model); a different training seed of the same setup gives almost
    none, and its question averages differ by 0.07 to 0.11. status: "match" (at least half the
    answers identical), "close" (averages within 0.04 on average) or "MISMATCH"."""
    k = ["market_ticker", "sample"]
    n = new.assign(text_key=new["text"].map(text_key))
    j = n[k + ["prob", "text_key"]].merge(stored[k + ["prob", "text_key"]], on=k, suffixes=("_new", "_stored"))
    if j.empty:
        return {"questions": 0, "answers": 0, "identical_text": np.nan, "mean_abs_diff": np.nan,
                "corr": np.nan, "status": "no overlap"}
    m = j.groupby("market_ticker")[["prob_new", "prob_stored"]].mean().fillna(0.5)
    identical = float((j["text_key_new"] == j["text_key_stored"]).mean())
    mad = float((m["prob_new"] - m["prob_stored"]).abs().mean())
    corr = float(m["prob_new"].corr(m["prob_stored"])) if m["prob_new"].std() > 0 and m["prob_stored"].std() > 0 else np.nan
    status = "match" if identical >= 0.5 else ("close" if mad <= 0.04 else "MISMATCH")
    return {"questions": len(m), "answers": len(j), "identical_text": round(identical, 3),
            "mean_abs_diff": round(mad, 4), "corr": round(corr, 3) if corr == corr else np.nan, "status": status}


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


def consistency_table(eval_df: pd.DataFrame, methods: Iterable[str], econ_set: str = "val_econ") -> pd.DataFrame:
    rows = []
    econ = scored_subset(eval_df)
    econ = econ[econ["eval_set"] == econ_set]
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


def add_shift_matched(eval_df: pd.DataFrame, from_col: str, to_col: str) -> pd.DataFrame:
    """Adds `<from_col>_shifted`: `from_col` moved up or down (per eval set, on the scored
    questions) so its average matches `to_col`'s, clipped to [0, 1]. If a trained model only beats
    its untrained self by this much, training moved the average, not the ordering."""
    eval_df = eval_df.copy()
    name = f"{from_col}_shifted"
    eval_df[name] = eval_df[from_col]
    scored = eval_df["market_prob"].notna()
    for eval_set in eval_df["eval_set"].unique():
        m = eval_df["eval_set"] == eval_set
        s = m & scored
        shift = eval_df.loc[s, to_col].mean() - eval_df.loc[s, from_col].mean()
        eval_df.loc[m, name] = (eval_df.loc[m, from_col] + shift).clip(0, 1)
    return eval_df


BOOTSTRAP_STATS = {"brier": M.brier, "log_loss": M.log_loss, "ece": M.ece, "auc": M.auc, "gap": M.gap,
                   "mean_p": lambda p, y: float(np.mean(p))}
HIGHER_IS_BETTER = {"auc", "gap"}


def paired_bootstrap(g: pd.DataFrame, a: str, b: str, n_boot: int = 4000, seed: int = 0,
                     stats: dict = None) -> pd.DataFrame:
    """Change from method `a` to method `b` on the same questions, with a 95% interval from
    resampling whole events (a weather day, or one release's ladder of thresholds) with replacement.
    Questions in one event move together, so resampling single questions would make the interval
    too narrow. `share_better` = share of draws where b beats a (mean_p: share where b is lower)."""
    stats = stats or BOOTSTRAP_STATS
    rng = np.random.default_rng(seed)
    g = g.reset_index(drop=True)
    ev = g["event_ticker"].to_numpy()
    events = pd.unique(ev)
    idx = {e: np.flatnonzero(ev == e) for e in events}
    y, pa, pb = g["outcome"].to_numpy(), g[a].to_numpy(dtype=float), g[b].to_numpy(dtype=float)
    draws = {k: np.empty(n_boot) for k in stats}
    for i in range(n_boot):
        pick = np.concatenate([idx[e] for e in rng.choice(events, size=len(events), replace=True)])
        for k, f in stats.items():
            draws[k][i] = f(pb[pick], y[pick]) - f(pa[pick], y[pick])
    rows = []
    for k, f in stats.items():
        d = draws[k][~np.isnan(draws[k])]
        lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
        better = np.mean(d > 0) if k in HIGHER_IS_BETTER else np.mean(d < 0)
        rows.append({"metric": k, a: f(pa, y), b: f(pb, y), "change": f(pb, y) - f(pa, y),
                     "ci_low": lo, "ci_high": hi, "share_better": float(better) if len(d) else np.nan})
    return pd.DataFrame(rows)


def paired_bootstrap_groups(g: pd.DataFrame, a_cols: Sequence[str], b_cols: Sequence[str], n_boot: int = 4000,
                            seed: int = 0, stats: dict = None) -> pd.DataFrame:
    """Like paired_bootstrap, for groups of forecasts (e.g. three training seeds): change = average
    over b_cols of each metric minus the average over a_cols, with every draw resampling the same
    whole events for all columns. The interval covers question sampling, not seed variation;
    b_min / b_max give the spread across b_cols."""
    stats = stats or BOOTSTRAP_STATS
    rng = np.random.default_rng(seed)
    g = g.reset_index(drop=True)
    ev = g["event_ticker"].to_numpy()
    events = pd.unique(ev)
    idx = {e: np.flatnonzero(ev == e) for e in events}
    y = g["outcome"].to_numpy()
    A = [g[c].to_numpy(dtype=float) for c in a_cols]
    B = [g[c].to_numpy(dtype=float) for c in b_cols]

    def diff(f, pick):
        return np.mean([f(b[pick], y[pick]) for b in B]) - np.mean([f(a[pick], y[pick]) for a in A])

    draws = {k: np.empty(n_boot) for k in stats}
    for i in range(n_boot):
        pick = np.concatenate([idx[e] for e in rng.choice(events, size=len(events), replace=True)])
        for k, f in stats.items():
            draws[k][i] = diff(f, pick)
    rows, everything = [], np.arange(len(g))
    for k, f in stats.items():
        d = draws[k][~np.isnan(draws[k])]
        lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
        better = np.mean(d > 0) if k in HIGHER_IS_BETTER else np.mean(d < 0)
        per_b = [f(b, y) for b in B]
        rows.append({"metric": k, "a": float(np.mean([f(a, y) for a in A])), "b": float(np.mean(per_b)),
                     "change": diff(f, everything), "ci_low": lo, "ci_high": hi,
                     "share_better": float(better) if len(d) else np.nan,
                     "b_min": float(np.min(per_b)), "b_max": float(np.max(per_b))})
    return pd.DataFrame(rows)


def paired_bootstrap_table(eval_df: pd.DataFrame, a: str, b: str, n_boot: int = 4000, seed: int = 0) -> pd.DataFrame:
    """paired_bootstrap for each eval set, on the scored questions."""
    out = []
    for eval_set, g in scored_subset(eval_df).groupby("eval_set"):
        t = paired_bootstrap(g, a, b, n_boot=n_boot, seed=seed)
        t.insert(0, "eval_set", eval_set)
        t["events"] = g["event_ticker"].nunique()
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
