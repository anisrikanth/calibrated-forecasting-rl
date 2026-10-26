"""Tests for the forecasting environment: parser, prompt, rewards, metrics."""
import math

import numpy as np
import pandas as pd
import pytest

from env.metrics import (brier, ece, ladder_violations, log_loss, murphy_decomposition,
                         partition_sum_error, reliability_table, summarize)
from env.parser import extract_probability
from env.prompt import build_messages, build_user_message, chat_template_kwargs, format_question_time
from env.rewards import (brier_reward, format_reward, log_score_reward, trl_brier_reward,
                         trl_format_reward, trl_log_score_reward)


# ---- parser ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Reasoning.\nProbability: 0.35", 0.35),
    ("Probability: 0.2 ... actually\nProbability: 0.4", 0.4),
    ("Probability: 8/17 = 0.47", 0.47),
    ("probability: .6", 0.6),
    ("Probability: 1", 1.0),
    ("Probability: 0", 0.0),
    ("Probability: 70%", None),
    ("Probability: 1.5", None),
    ("I think it's likely.", None),
    ("", None),
    (None, None),
])
def test_extract_probability(text, expected):
    assert extract_probability(text) == expected


# ---- prompt ----------------------------------------------------------------------------------------

ROW = {
    "question_time": "2026-10-01T12:29:00+00:00",
    "question_text": "If payrolls rise above 90,000 in September 2026, the market resolves to Yes.",
    "context_text": "Recent results in this series (oldest first):\n- Jobs numbers in Aug 2026: 162,000",
    "outcome": 0, "market_prob": 0.3,
}


def test_question_time_is_shown_in_eastern_time():
    assert format_question_time(ROW["question_time"]) == "Thursday, October 1, 2026, 8:29 AM Eastern"
    assert format_question_time("2024-07-02T05:59:00Z") == "Tuesday, July 2, 2024, 1:59 AM Eastern"


def test_user_message_with_and_without_context():
    with_ctx = build_user_message(ROW, include_context=True)
    without = build_user_message(ROW, include_context=False)
    assert "Jobs numbers in Aug 2026: 162,000" in with_ctx
    assert "162,000" not in without and "No recent results are provided" in without
    for msg in (with_ctx, without):
        assert "Question: If payrolls" in msg and "Current date: Thursday, October 1, 2026" in msg


def test_prompt_never_leaks_outcome_or_market_price():
    text = " ".join(m["content"] for m in build_messages(ROW))
    assert "0.3" not in text and "outcome" not in text.lower()


def test_chat_template_date_matches_question_date():
    assert chat_template_kwargs(ROW) == {"date_string": "01 Oct 2026"}


# ---- rewards ---------------------------------------------------------------------------------------

def test_brier_reward():
    assert brier_reward(1.0, 1) == 1.0
    assert brier_reward(0.5, 0) == 0.75
    assert brier_reward(1.0, 0) == 0.0
    assert brier_reward(None, 1) == 0.0


def test_log_score_reward_is_clipped():
    assert log_score_reward(0.8, 1) == pytest.approx(math.log(0.8))
    assert log_score_reward(0.0, 1) == pytest.approx(math.log(0.01))
    assert log_score_reward(1.0, 0) == pytest.approx(math.log(0.01))
    assert log_score_reward(None, 0) == pytest.approx(math.log(0.01))


def test_proper_scoring_honesty_pays():
    """If the true chance is 0.3, stating 0.3 has the best expected reward under both rules."""
    q = 0.3
    for reward in (brier_reward, log_score_reward):
        expected = {p: q * reward(p, 1) + (1 - q) * reward(p, 0) for p in np.linspace(0.05, 0.95, 19)}
        assert max(expected, key=expected.get) == pytest.approx(q)


def test_trl_wrappers_accept_chat_and_string_completions():
    comps = [[{"role": "assistant", "content": "Probability: 0.9"}], "Probability: 0.2", "no idea"]
    assert trl_format_reward(comps) == [0.5, 0.5, -1.0]
    assert [round(r, 4) for r in trl_brier_reward(comps, outcome=[1, 1, 0])] == [0.99, 0.36, 0.0]
    assert trl_log_score_reward(comps, outcome=[1, 0, 1])[0] == pytest.approx(math.log(0.9))
    assert format_reward("Probability: 0.5") == 0.5


# ---- metrics ---------------------------------------------------------------------------------------

def test_brier_and_log_loss():
    assert brier([0.5, 0.5], [1, 0]) == 0.25
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert log_loss([0.0], [1]) == pytest.approx(-math.log(0.01))


def test_reliability_and_ece_for_a_perfectly_calibrated_forecaster():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert ece(p, y) < 0.01
    t = reliability_table(p, y)
    assert len(t) == 10 and t["count"].sum() == p.size


def test_murphy_decomposition_adds_up_and_base_rate_has_no_resolution():
    rng = np.random.default_rng(1)
    p = np.round(rng.uniform(0, 1, 5000), 1)       # forecasts exactly on bin values
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    d = murphy_decomposition(p, y)
    assert brier(p, y) == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=0.01)
    base = np.full(p.size, y.mean())
    assert murphy_decomposition(base, y)["resolution"] == pytest.approx(0.0)
    assert set(summarize(p, y)) >= {"n", "brier", "log_loss", "ece", "reliability", "resolution", "uncertainty"}


def test_ladder_violations():
    df = pd.DataFrame([
        # "above" ladder: should fall as the threshold rises; one violation (0.5 -> 0.6)
        {"event_ticker": "E1", "strike_type": "greater", "floor_strike": 100, "cap_strike": None, "p": 0.8},
        {"event_ticker": "E1", "strike_type": "greater", "floor_strike": 150, "cap_strike": None, "p": 0.5},
        {"event_ticker": "E1", "strike_type": "greater", "floor_strike": 200, "cap_strike": None, "p": 0.6},
        # "below" ladder: should rise; consistent
        {"event_ticker": "E2", "strike_type": "less", "floor_strike": None, "cap_strike": 3.0, "p": 0.2},
        {"event_ticker": "E2", "strike_type": "less", "floor_strike": None, "cap_strike": 4.0, "p": 0.7},
    ])
    assert ladder_violations(df, "p") == {"pairs": 3, "violations": 1, "violation_rate": pytest.approx(1 / 3)}


def test_partition_sum_error():
    df = pd.DataFrame([
        {"event_ticker": "W1", "strike_type": "less", "outcome": 0, "p": 0.2},
        {"event_ticker": "W1", "strike_type": "between", "outcome": 1, "p": 0.5},
        {"event_ticker": "W1", "strike_type": "greater", "outcome": 0, "p": 0.1},     # sums to 0.8
        {"event_ticker": "C1", "strike_type": "greater", "outcome": 1, "p": 0.9},     # no brackets: ignored
    ])
    out = partition_sum_error(df, "p")
    assert out["events"] == 1 and out["mean_abs_sum_error"] == pytest.approx(0.2)
