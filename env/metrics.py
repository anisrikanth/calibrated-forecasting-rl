"""Calibration and accuracy metrics for probability forecasts on yes/no outcomes.

Notation in words: p = forecast probability, y = outcome (1 yes, 0 no), N = number of questions.

- Brier score: the average of (p − y) squared. Lower is better; always saying 0.5 scores 0.25.
- Log loss: the average of −log(probability given to what happened), with p clipped to [0.01, 0.99].
- Expected calibration error (ECE): group forecasts into 10 equal-width probability bins; in each
  bin compare the average forecast with the share that actually resolved yes; ECE is the average
  size of that gap, weighted by how many forecasts are in each bin. 0 means perfectly calibrated.
- Murphy decomposition of the Brier score (using the same bins):
    reliability = weighted average of (average forecast in bin − yes-share in bin) squared.
                  Lower is better: how far forecasts are from the frequencies they claim.
    resolution  = weighted average of (yes-share in bin − overall yes-share) squared.
                  Higher is better: how much the forecasts separate yes-questions from no-questions.
                  Always forecasting the base rate is perfectly calibrated but has zero resolution.
    uncertainty = overall yes-share × (1 − overall yes-share). A property of the questions, not the forecaster.
  Brier score ≈ reliability − resolution + uncertainty.
- Ladder consistency: for thresholds of the same event ("above 100k", "above 150k", ...), the
  probability should fall as an "above" threshold rises. We count adjacent pairs that break this.
- Partition error: for events whose markets are non-overlapping brackets covering every outcome
  (e.g. daily temperature ranges), the probabilities should add up to 1. We report the average
  absolute gap from 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLIP = 0.01


def _arrays(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape:
        raise ValueError("forecasts and outcomes must have the same length")
    return p, y


def brier(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, clip: float = CLIP) -> float:
    p, y = _arrays(p, y)
    p = np.clip(p, clip, 1 - clip)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(p, y, n_bins: int = 10) -> pd.DataFrame:
    """One row per non-empty bin: bin range, count, average forecast, share that resolved yes."""
    p, y = _arrays(p, y)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": int(mask.sum()),
                         "mean_forecast": float(p[mask].mean()), "frac_yes": float(y[mask].mean())})
    return pd.DataFrame(rows, columns=["bin_low", "bin_high", "count", "mean_forecast", "frac_yes"])


def ece(p, y, n_bins: int = 10) -> float:
    t = reliability_table(p, y, n_bins)
    n = t["count"].sum()
    return float((t["count"] * (t["mean_forecast"] - t["frac_yes"]).abs()).sum() / n) if n else float("nan")


def murphy_decomposition(p, y, n_bins: int = 10) -> dict:
    p, y = _arrays(p, y)
    t = reliability_table(p, y, n_bins)
    n, base = len(p), float(y.mean())
    return {
        "reliability": float((t["count"] * (t["mean_forecast"] - t["frac_yes"]) ** 2).sum() / n),
        "resolution": float((t["count"] * (t["frac_yes"] - base) ** 2).sum() / n),
        "uncertainty": base * (1 - base),
    }


def summarize(p, y, n_bins: int = 10) -> dict:
    p, y = _arrays(p, y)
    out = {"n": int(len(p)), "brier": brier(p, y), "log_loss": log_loss(p, y), "ece": ece(p, y, n_bins)}
    out.update(murphy_decomposition(p, y, n_bins))
    return out


def ladder_violations(df: pd.DataFrame, prob_col: str, tol: float = 1e-9) -> dict:
    """Counts adjacent threshold pairs within an event whose probabilities go the wrong way.
    "greater"-type thresholds should have probabilities that fall as the threshold rises;
    "less"-type thresholds should rise. Uses floor_strike, falling back to cap_strike."""
    pairs = violations = 0
    d = df.copy()
    d["_threshold"] = d["floor_strike"].where(d["floor_strike"].notna(), d["cap_strike"])
    for direction, types in ((-1, ("greater", "greater_or_equal")), (1, ("less", "less_or_equal"))):
        sub = d[d["strike_type"].isin(types) & d["_threshold"].notna() & d[prob_col].notna()]
        for _, g in sub.groupby("event_ticker"):
            probs = g.sort_values("_threshold")[prob_col].to_numpy(dtype=float)
            if len(probs) < 2:
                continue
            steps = np.diff(probs) * direction        # should be >= 0
            pairs += len(steps)
            violations += int((steps < -tol).sum())
    return {"pairs": pairs, "violations": violations,
            "violation_rate": violations / pairs if pairs else float("nan")}


def partition_sum_error(df: pd.DataFrame, prob_col: str) -> dict:
    """For events with bracket markets ("between"), the average |sum of probabilities − 1|.
    Only events whose outcomes all resolved (exactly one yes) are used, so the set is complete."""
    errors = []
    for _, g in df[df[prob_col].notna()].groupby("event_ticker"):
        if not (g["strike_type"] == "between").any() or g["outcome"].sum() != 1:
            continue
        errors.append(abs(float(g[prob_col].sum()) - 1.0))
    return {"events": len(errors), "mean_abs_sum_error": float(np.mean(errors)) if errors else float("nan")}
