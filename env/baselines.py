"""Simple, non-LLM baselines built from the same evidence the model sees.

- `latest_rule`: 1 if the latest recent result already satisfies the question, else 0. RL run 3
  collapsed onto exactly this rule (writeup/hacks.md, entry 5).
- `LogisticBaseline`: a logistic regression on five numbers derived from the key numbers, fitted on
  the training questions only. It answers "how well does a tiny statistical model do with the
  evidence in the prompt?", the question Tan et al. (2024) ask of LLM time-series forecasters.

Features (all computed from the question's threshold and the recent results in the context):
  hit_rate    share of recent results that would have resolved YES
  latest_yes  1 if the latest result would have resolved YES
  z_latest    how far the latest result is on the YES side of the threshold (negative = NO side),
              in units of the series' typical change between consecutive results
  z_mean      the same for the average of the recent results
  z_trend     how much the latest change moved toward YES, same units
z values are clipped to [-5, 5]. For "between" questions, "YES side" means inside the range,
measured to the nearest edge.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from env.features import key_numbers, threshold_rule, unit_of

FEATURES = ("hit_rate", "latest_yes", "z_latest", "z_mean", "z_trend")
Z_CLIP = 5.0


def latest_rule(row: Mapping[str, Any]) -> Optional[float]:
    rule = threshold_rule(row, unit_of(row))
    k = key_numbers(row)
    if rule is None or not k or not k["values"]:
        return None
    return float(rule[1](k["values"][-1]))


def _margin(row: Mapping[str, Any], v: float) -> float:
    """Signed distance of value v from the question's YES region (positive = inside)."""
    st, lo, hi = row.get("strike_type"), row.get("floor_strike"), row.get("cap_strike")
    if st in ("greater", "greater_or_equal"):
        return v - lo
    if st in ("less", "less_or_equal"):
        return hi - v
    return min(v - lo, hi - v)                                   # between


def rule_features(row: Mapping[str, Any]) -> Optional[np.ndarray]:
    k = key_numbers(row)
    if k is None or not k["values"]:
        return None
    rule = threshold_rule(row, unit_of(row))
    vals = np.asarray(k["values"], dtype=float)
    diffs = np.abs(np.diff(vals))
    scale = float(np.median(diffs)) if len(diffs) else 0.0
    if scale <= 1e-9:
        scale = float(diffs.mean()) if len(diffs) and diffs.mean() > 1e-9 else 0.0
    if scale <= 1e-9:                                            # flat series (e.g. an unchanged Fed rate)
        ref = rule[2][1]
        center = ref if not isinstance(ref, tuple) else (ref[0] + ref[1]) / 2
        scale = max(0.05 * abs(center), 1e-3)
    z = lambda m: float(np.clip(m / scale, -Z_CLIP, Z_CLIP))      # noqa: E731
    latest = vals[-1]
    trend = _margin(row, latest) - _margin(row, vals[-2]) if len(vals) >= 2 else 0.0
    return np.array([k["hits"] / len(vals), float(rule[1](latest)), z(_margin(row, latest)),
                     z(_margin(row, vals.mean())), z(trend)])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Logistic regression by Newton's method with an L2 penalty (not on the intercept).
    Returns weights [intercept, w_1, ..., w_d]."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    penalty = np.full(Xb.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) + penalty * w
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(penalty) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


class LogisticBaseline:
    """Fit on training rows (each with `outcome`); predict a probability, or NaN where a row has
    no numeric threshold or no recent numeric result."""

    def __init__(self, l2: float = 1.0, min_rows: int = 20):
        self.l2, self.min_rows, self.w = l2, min_rows, None

    def fit(self, rows: pd.DataFrame) -> "LogisticBaseline":
        feats = [rule_features(r) for r in rows.to_dict("records")]
        keep = [i for i, f in enumerate(feats) if f is not None]
        if len(keep) >= self.min_rows:
            X = np.vstack([feats[i] for i in keep])
            y = rows["outcome"].to_numpy(dtype=float)[keep]
            if 0 < y.mean() < 1:
                self.w = fit_logistic(X, y, self.l2)
        return self

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        out = np.full(len(rows), np.nan)
        if self.w is not None:
            for i, r in enumerate(rows.to_dict("records")):
                f = rule_features(r)
                if f is not None:
                    out[i] = 1 / (1 + np.exp(-(self.w[0] + f @ self.w[1:])))
        return pd.Series(out, index=rows.index)

    def coefficients(self) -> dict:
        return {} if self.w is None else dict(zip(("intercept",) + FEATURES, np.round(self.w, 3)))


def latest_rule_column(rows: pd.DataFrame) -> pd.Series:
    return pd.Series([latest_rule(r) for r in rows.to_dict("records")], index=rows.index, dtype=float)


def add_rule_baselines(out: pd.DataFrame, train: pd.DataFrame, fallback: Sequence[float]) -> pd.DataFrame:
    """Adds `latest_rule` and `logistic` columns; rows without a numeric threshold get `fallback`."""
    fb = pd.Series(np.asarray(fallback, dtype=float), index=out.index)
    out["latest_rule"] = latest_rule_column(out).fillna(fb)
    out["logistic"] = LogisticBaseline().fit(train).predict(out).fillna(fb)
    return out
