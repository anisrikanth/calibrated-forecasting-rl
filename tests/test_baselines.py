"""Tests for env/baselines.py (latest-result rule, logistic baseline) and the shuffled-outcome control."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from env.baselines import (FEATURES, LogisticBaseline, Z_CLIP, _margin, fit_logistic, latest_rule,
                           rule_features)
from env.evaluation import pick_eval_sets
from env.training import MonitoredScoreReward, build_train_records


def ctx(*values):
    return json.dumps([{"label": f"r{i}", "value": v} for i, v in enumerate(values)])


def greater(th, *values, **kw):
    return dict(strike_type="greater", floor_strike=th, cap_strike=None, question_text=f"above {th}%",
                context_json=ctx(*values), **kw)


def test_latest_rule_all_strike_types():
    assert latest_rule(greater(2.5, "2.4", "2.6")) == 1.0
    assert latest_rule(greater(2.5, "2.6", "2.5")) == 0.0                     # equal is not "more than"
    less = dict(strike_type="less", cap_strike=59.0, question_text="below 59°", context_json=ctx("60", "58"))
    assert latest_rule(less) == 1.0
    between = dict(strike_type="between", floor_strike=59.0, cap_strike=60.0, question_text="59-60°",
                   context_json=ctx("61", "60"))
    assert latest_rule(between) == 1.0
    assert latest_rule(dict(strike_type="custom", context_json=ctx("1"))) is None
    assert latest_rule(greater(2.5)) is None                                   # no recent results


def test_margin_signs():
    assert _margin(greater(2.5), 2.7) == pytest.approx(0.2)
    assert _margin(dict(strike_type="less", cap_strike=59.0), 61) == -2
    b = dict(strike_type="between", floor_strike=59.0, cap_strike=62.0)
    assert _margin(b, 60) == 1 and _margin(b, 64) == -2 and _margin(b, 57) == -2


def test_rule_features_values():
    f = rule_features(greater(2.5, "2.4", "2.6", "2.8"))
    assert len(f) == len(FEATURES)
    hit_rate, latest_yes, z_latest, z_mean, z_trend = f
    assert hit_rate == pytest.approx(2 / 3) and latest_yes == 1.0
    # typical change = median(|0.2|, |0.2|) = 0.2; latest 2.8 is 0.3 above 2.5 -> 1.5 changes
    assert z_latest == pytest.approx(1.5) and z_mean == pytest.approx(0.5) and z_trend == pytest.approx(1.0)


def test_rule_features_flat_series_and_clipping():
    f = rule_features(greater(3.25, "4.5", "4.5", "4.5"))          # unchanged Fed rate: scale falls back
    assert np.isfinite(f).all() and f[2] == Z_CLIP                  # 1.25 / (0.05 * 3.25) > 5, clipped
    f1 = rule_features(greater(2.5, "2.6"))                         # one result: no trend
    assert f1[4] == 0.0
    assert rule_features(greater(2.5)) is None
    assert rule_features(dict(strike_type="custom", context_json=ctx("1"))) is None


def test_fit_logistic_solves_penalized_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (rng.random(400) < 1 / (1 + np.exp(-(0.5 + 2 * X[:, 0] - X[:, 1])))).astype(float)
    w = fit_logistic(X, y, l2=1.0)
    assert w[1] > 1 and w[2] < -0.5                                   # recovers the directions
    Xb = np.column_stack([np.ones(len(X)), X])
    p = 1 / (1 + np.exp(-Xb @ w))
    grad = Xb.T @ (p - y) + np.array([0, 1.0, 1.0]) * w
    assert np.abs(grad).max() < 1e-6                                  # optimum of the penalized loss


def rows_for_fit(n=60, seed=0):
    """Questions where the outcome mostly follows the latest result, with some noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        latest = 2.5 + rng.choice([-0.3, -0.1, 0.1, 0.3])
        y = int((latest > 2.5) if rng.random() < 0.85 else (latest <= 2.5))
        rows.append(greater(2.5, "2.4", "2.6", f"{latest:.1f}", outcome=y))
    return pd.DataFrame(rows)


def test_logistic_baseline_fit_predict():
    train = rows_for_fit()
    lb = LogisticBaseline().fit(train)
    assert set(lb.coefficients()) == {"intercept", *FEATURES}
    test = pd.DataFrame([greater(2.5, "2.4", "2.6", "2.8"), greater(2.5, "2.4", "2.6", "2.2"),
                         dict(strike_type="custom", context_json=ctx("Cut 25bps"))])
    p = lb.predict(test)
    assert 0.5 < p.iloc[0] < 1 and 0 < p.iloc[1] < 0.5 and math.isnan(p.iloc[2])
    assert LogisticBaseline().fit(train.head(5)).predict(test).isna().all()      # too few rows: no model


def make_markets():
    rows = []
    for m in range(8):
        split = "train" if m < 6 else "val"
        close = pd.Timestamp("2025-01-10 13:30", tz="UTC") + pd.DateOffset(months=m)
        for th, y in ((2.5, 1), (3.5, 0)):
            rows.append(dict(market_ticker=f"CPI-{m}-{th}", event_ticker=f"CPI-{m}", series_ticker="KXCPIYOY",
                             split=split, outcome=y, market_prob=0.5, strike_type="greater", floor_strike=th,
                             cap_strike=None, question_time=(close - pd.Timedelta(hours=24)).isoformat(),
                             close_time=close.isoformat(), question_text=f"CPI more than {th}%?",
                             context_text="Recent results:\n- a: 3\n- b: 2.8", context_json=ctx("3", "2.8")))
    rows.append(dict(rows[-1], market_ticker="FED-x", event_ticker="FED-7", series_ticker="KXFEDDECISION",
                     strike_type="custom", floor_strike=None, outcome=0))
    df = pd.DataFrame(rows)
    df["close_dt"] = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")
    return df


def test_pick_eval_sets_adds_rule_baselines():
    ev = pick_eval_sets(make_markets(), weather_events=0).set_index("market_ticker")
    assert ev.loc["CPI-6-2.5", "latest_rule"] == 1.0 and ev.loc["CPI-6-3.5", "latest_rule"] == 0.0
    assert ev.loc["FED-x", "latest_rule"] == ev.loc["FED-x", "base_rate"]
    # 12 training rows is below the logistic baseline's minimum, so it falls back to the base rate
    assert (ev["logistic"] == ev["base_rate"]).all()
    assert ev[["latest_rule", "logistic"]].notna().all().all()


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, date_string=None):
        return "<|begin_of_text|>" + " || ".join(m["content"] for m in messages)


def test_shuffled_outcomes_control():
    df = pd.concat([make_markets()] * 5, ignore_index=True)
    df["market_ticker"] = df["market_ticker"] + "-" + df.index.astype(str)
    df.loc[df.index % 3 == 0, "outcome"] = 1 - df.loc[df.index % 3 == 0, "outcome"]
    real = build_train_records(df, FakeTokenizer(), seed=1)
    shuf = build_train_records(df, FakeTokenizer(), seed=1, shuffle_outcomes=True)
    assert all(r["outcome"] == r["true_outcome"] for r in real)
    assert [r["market_ticker"] for r in real] == [r["market_ticker"] for r in shuf]
    assert [r["true_outcome"] for r in shuf] == [r["outcome"] for r in real]
    assert sorted(r["outcome"] for r in shuf) == sorted(r["outcome"] for r in real)    # same yes-rate
    assert any(r["outcome"] != r["true_outcome"] for r in shuf)
    assert shuf == build_train_records(df, FakeTokenizer(), seed=1, shuffle_outcomes=True)   # reproducible


def test_monitor_records_true_outcome():
    r = MonitoredScoreReward("brier", print_every=100, log=lambda s: None)
    r(["Probability: 0.9", "Probability: 0.2"], outcome=[1, 0], true_outcome=[0, 0], series_ticker=["A", "B"])
    h = r.history()
    assert h["outcome"].tolist() == [1, 0] and h["true_outcome"].tolist() == [0, 0]
    r(["Probability: 0.5"], outcome=[1])                                             # no true_outcome passed
    assert r.history()["true_outcome"].tolist()[-1] == 1
