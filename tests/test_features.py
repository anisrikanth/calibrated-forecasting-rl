"""Tests for the key-numbers block (env/features.py), its use in prompts, and the new metrics and
evaluation helpers that go with run 3 (AUC, gap, rule baseline, shift-matched baseline, bootstrap)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from env import metrics as M
from env.evaluation import add_shift_matched, paired_bootstrap, paired_bootstrap_table, pick_eval_sets
from env.features import HEADER, fmt, key_numbers, key_numbers_text, recent_hit_rate
from env.prompt import build_user_message
from env.training import build_train_records


def ctx(*pairs):
    return json.dumps([{"label": label, "value": value} for label, value in pairs])


def cpi_row(**kw):
    row = dict(question_time="2026-01-12T13:29:00Z", strike_type="greater", floor_strike=2.7, cap_strike=None,
               question_text="If CPI increases by more than 2.7% in the twelve months ending December 2025, Yes.",
               context_text="Recent results in this series:\n- Inflation in Oct 2025? (CPI YoY): 3.3\n- ...",
               context_json=ctx(("Inflation in Sep 2025? (CPI YoY)", "3"), ("Inflation in Oct 2025? (CPI YoY)", "3.3"),
                                ("Inflation in Nov 2025? (CPI YoY)", "2.6%")))
    row.update(kw)
    return row


@pytest.mark.parametrize("x, unit, expected", [
    (147000, "", "147,000"), (0.30000000004, "", "0.3"), (2.5666, "%", "2.57%"), (4.0, "", "4"),
    (-0.1, "%", "-0.1%"), (-0.001, "", "0"), (68, "°", "68°"), (3.25, "%", "3.25%"),
])
def test_fmt(x, unit, expected):
    assert fmt(x, unit) == expected


def test_greater_with_percent_values():
    text = key_numbers_text(cpi_row())
    lines = text.split("\n")
    assert lines[0] == HEADER
    assert lines[1] == "- This question resolves YES if the result is above 2.7%."
    # "?" is stripped from labels; "2.6%" parses; 2.6 - 2.7 = -0.1 without float noise
    assert lines[2] == "- Latest result (Inflation in Nov 2025 (CPI YoY)): 2.6%, 0.1 percentage points below the threshold."
    assert lines[3] == "- Change from the result before: -0.7 percentage points (from 3.3% to 2.6%)."
    assert lines[4] == "- Recent results that would have resolved YES: 2 of 3."
    assert lines[5] == "- Recent results: average 2.97%, lowest 2.6%, highest 3.3%."


def test_equal_to_threshold_is_not_a_hit_for_greater():
    row = cpi_row(context_json=ctx(("a", "2.7"), ("b", "2.7")))
    text = key_numbers_text(row)
    assert "2.7%, equal to the threshold" in text
    assert "resolved YES: 0 of 2" in text
    assert "Change from the result before: 0 percentage points" in text


def test_commas_and_no_unit_for_payrolls():
    row = dict(strike_type="greater", floor_strike=150000.0, cap_strike=float("nan"),
               question_text="If the increase in total non-farm payroll employment is above 150000 ...",
               context_json=ctx(("Jobs numbers in Jul 2025", "73,000"), ("Jobs numbers in Aug 2025", "222,000")))
    text = key_numbers_text(row)
    assert "above 150,000." in text
    assert "222,000, 72,000 above the threshold" in text
    assert "Change from the result before: +149,000 (from 73,000 to 222,000)" in text
    assert "resolved YES: 1 of 2" in text


def test_weather_less_between_greater():
    temps = ctx(("d1", "58"), ("d2", "59"), ("d3", "61"), ("d4", "66"))
    less = dict(strike_type="less", floor_strike=None, cap_strike=59.0, context_json=temps,
                question_text="highest temperature ... less than 59°")
    t = key_numbers_text(less)
    assert "below 59°." in t and "66°, 7° above the threshold" in t and "1 of 4" in t
    between = dict(strike_type="between", floor_strike=59.0, cap_strike=61.0, context_json=temps,
                   question_text="highest temperature ... between 59-61°")
    t = key_numbers_text(between)
    assert "from 59° to 61° (inclusive)." in t
    assert "66°, 5° above the range" in t
    assert "2 of 4" in t                                     # 59 and 61 are inside: inclusive
    below = dict(between, context_json=ctx(("d1", "70"), ("d2", "55")))
    assert "55°, 4° below the range" in key_numbers_text(below)
    inside = dict(between, context_json=ctx(("d1", "60")))
    assert "60°, inside the range" in key_numbers_text(inside)
    assert "Change from the result before" not in key_numbers_text(inside)    # one result only
    greater = dict(strike_type="greater", floor_strike=65.0, cap_strike=None, context_json=temps,
                   question_text="highest temperature ... greater than 65°")
    assert "1 of 4" in key_numbers_text(greater)


def test_or_equal_variants():
    ge = dict(strike_type="greater_or_equal", floor_strike=3.0, context_json=ctx(("a", "3"), ("b", "2.9")),
              question_text="at least 3.0%")
    assert "at or above 3%" in key_numbers_text(ge) and "1 of 2" in key_numbers_text(ge)
    le = dict(strike_type="less_or_equal", cap_strike=3.0, context_json=ctx(("a", "3"), ("b", "2.9")),
              question_text="at most 3.0%")
    assert "at or below 3%" in key_numbers_text(le) and "2 of 2" in key_numbers_text(le)


@pytest.mark.parametrize("row", [
    dict(strike_type="custom", floor_strike=None, cap_strike=None, context_json=ctx(("a", "Cut 25bps"))),
    dict(strike_type="greater", floor_strike=float("nan"), cap_strike=3.0, context_json=ctx(("a", "1"))),
    dict(strike_type="between", floor_strike=1.0, cap_strike=None, context_json=ctx(("a", "1"))),
    dict(context_json=ctx(("a", "1"))),
])
def test_no_numeric_threshold_gives_none(row):
    assert key_numbers_text(row) is None and key_numbers(row) is None and recent_hit_rate(row) is None


@pytest.mark.parametrize("context_json", [None, float("nan"), "", "not json", "[]",
                                          ctx(("a", "Cut 25bps"), ("b", "n/a"))])
def test_no_numeric_results(context_json):
    row = cpi_row(context_json=context_json)
    assert key_numbers_text(row).endswith("- No earlier numeric results are available.")
    assert recent_hit_rate(row) is None


def test_non_numeric_values_are_skipped():
    row = cpi_row(context_json=ctx(("a", "3.0"), ("b", "n/a"), ("c", "2.5")))
    k = key_numbers(row)
    assert k["values"] == [3.0, 2.5] and k["hits"] == 1
    assert recent_hit_rate(row) == 0.5


def test_prompt_adds_block_only_when_asked_and_context_shown():
    row = cpi_row()
    plain = build_user_message(row, include_context=True)
    with_block = build_user_message(row, include_context=True, key_numbers=True)
    assert HEADER not in plain
    assert with_block == plain.replace("\n\nWhat is the probability",
                                       "\n\n" + key_numbers_text(row) + "\n\nWhat is the probability")
    # The block is derived from the context, so no context means no block
    assert build_user_message(row, include_context=False, key_numbers=True) == build_user_message(row, include_context=False)
    fed = cpi_row(strike_type="custom", floor_strike=None)
    assert build_user_message(fed, True, True) == build_user_message(fed, True)


def test_auc_and_gap():
    y = [0, 0, 1, 1]
    assert M.auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert M.auc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert M.auc([0.5] * 4, y) == 0.5
    assert M.auc([0.1, 0.6, 0.4, 0.9], y) == 0.75
    # A uniform shift changes Brier but not AUC
    p = np.array([0.3, 0.5, 0.4, 0.7])
    assert M.auc(p - 0.2, y) == M.auc(p, y) and M.brier(p - 0.2, y) != M.brier(p, y)
    assert math.isnan(M.auc([0.1, 0.2], [1, 1])) and math.isnan(M.gap([0.1, 0.2], [0, 0]))
    assert M.gap([0.1, 0.3, 0.6, 0.8], y) == pytest.approx(0.5)
    s = M.summarize([0.1, 0.2, 0.8, 0.9], y)
    assert s["auc"] == 1.0 and s["gap"] == pytest.approx(0.7) and s["mean_p"] == pytest.approx(0.5)


def eval_frame():
    rows = []
    for e in range(6):
        for k, (y, base, trained) in enumerate([(1, 0.6, 0.5), (0, 0.5, 0.3), (0, 0.4, 0.3)]):
            rows.append(dict(eval_set="val_econ" if e < 3 else "heldout_weather", event_ticker=f"E{e}",
                             market_ticker=f"E{e}-{k}", outcome=y, market_prob=0.5,
                             base=base, trained=trained))
    return pd.DataFrame(rows)


def test_add_shift_matched():
    ev = eval_frame()
    ev.loc[0, "market_prob"] = np.nan                       # unscored row: shifted too, but not used for the mean
    out = add_shift_matched(ev, "base", "trained")
    for _, g in out[out["market_prob"].notna()].groupby("eval_set"):
        assert g["base_shifted"].mean() == pytest.approx(g["trained"].mean())
        assert M.auc(g["base_shifted"], g["outcome"]) == M.auc(g["base"], g["outcome"])
    assert "base_shifted" not in ev.columns
    clipped = add_shift_matched(ev.assign(trained=1.0), "base", "trained")
    assert clipped["base_shifted"].max() == 1.0


def test_paired_bootstrap():
    ev = eval_frame()
    same = paired_bootstrap(ev, "base", "base", n_boot=50).set_index("metric")
    assert (same["change"] == 0).all() and (same["ci_low"] == 0).all() and (same["ci_high"] == 0).all()
    t = paired_bootstrap(ev, "base", "trained", n_boot=200, seed=1).set_index("metric")
    assert t.loc["brier", "change"] == pytest.approx(M.brier(ev["trained"], ev["outcome"]) - M.brier(ev["base"], ev["outcome"]))
    assert t.loc["brier", "change"] < 0 and t.loc["brier", "share_better"] == 1.0  # better in every event
    assert t.loc["mean_p", "ci_high"] < 0
    assert t.loc["brier", "ci_low"] <= t.loc["brier", "change"] <= t.loc["brier", "ci_high"]
    table = paired_bootstrap_table(ev, "base", "trained", n_boot=20)
    assert set(table["eval_set"]) == {"val_econ", "heldout_weather"} and (table["events"] == 3).all()


def make_markets():
    rows = []
    for m in range(6):
        split = "train" if m < 4 else "val"
        close = pd.Timestamp("2025-10-10 13:30", tz="UTC") + pd.DateOffset(months=m)
        for th, y in ((2.5, 1), (3.5, 0)):
            rows.append(dict(market_ticker=f"CPI-{m}-{th}", event_ticker=f"CPI-{m}", series_ticker="KXCPIYOY",
                             split=split, outcome=y, market_prob=0.5, strike_type="greater", floor_strike=th,
                             cap_strike=None, question_time=(close - pd.Timedelta(hours=24)).isoformat(),
                             close_time=close.isoformat(), question_text=f"CPI more than {th}%?",
                             context_text="Recent results:\n- a: 3\n- b: 2.8",
                             context_json=ctx(("a", "3"), ("b", "2.8"))))
    rows.append(dict(rows[-1], market_ticker="FED-x", event_ticker="FED-5", series_ticker="KXFEDDECISION",
                     strike_type="custom", floor_strike=None, outcome=0))
    df = pd.DataFrame(rows)
    df["close_dt"] = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")
    return df


def test_recent_hits_baseline_in_eval_sets():
    ev = pick_eval_sets(make_markets(), weather_events=0).set_index("market_ticker")
    assert ev.loc["CPI-4-2.5", "recent_hits"] == 1.0          # 3 and 2.8 are both above 2.5
    assert ev.loc["CPI-4-3.5", "recent_hits"] == 0.0
    assert ev.loc["FED-x", "recent_hits"] == ev.loc["FED-x", "base_rate"]   # no threshold: falls back


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, date_string=None):
        return "<|begin_of_text|>" + " || ".join(m["content"] for m in messages)


def test_train_records_with_key_numbers():
    df = make_markets()
    with_block = build_train_records(df, FakeTokenizer(), include_context=True, key_numbers=True, seed=1)
    without = build_train_records(df, FakeTokenizer(), include_context=True, seed=1)
    assert all(HEADER in r["prompt"] for r in with_block)
    assert not any(HEADER in r["prompt"] for r in without)
    assert [r["market_ticker"] for r in with_block] == [r["market_ticker"] for r in without]


def test_key_number_parts_for_the_ablation():
    from env.features import PARTS
    row = cpi_row()
    assert key_numbers_text(row) == key_numbers_text(row, PARTS) == key_numbers_text(row, list(reversed(PARTS)))
    only_rule = key_numbers_text(row, ["rule"]).split("\n")
    assert only_rule == [HEADER, "- This question resolves YES if the result is above 2.7%."]
    rule_hits = key_numbers_text(row, ["rule", "hits"]).split("\n")
    assert rule_hits[1].startswith("- This question resolves YES") and rule_hits[2].startswith("- Recent results that would")
    assert len(rule_hits) == 3
    full = key_numbers_text(row).split("\n")
    assert key_numbers_text(row, ["rule", "latest", "hits"]).split("\n") == [full[0], full[1], full[2], full[4]]
    with pytest.raises(ValueError):
        key_numbers_text(row, ["rule", "median"])
    assert key_numbers_text(cpi_row(context_json="[]"), ["latest"]).endswith("No earlier numeric results are available.")


def test_estimate_first_prompt():
    from env.prompt import ESTIMATE_FIRST_QUESTION, FINAL_QUESTION
    row = cpi_row()
    plain = build_user_message(row, True)
    est = build_user_message(row, True, estimate_first=True)
    assert plain.endswith(FINAL_QUESTION) and est.endswith(ESTIMATE_FIRST_QUESTION)
    assert est.replace(ESTIMATE_FIRST_QUESTION, FINAL_QUESTION) == plain
    both = build_user_message(row, True, key_numbers=True, key_parts=["rule"], estimate_first=True)
    assert HEADER in both and both.endswith(ESTIMATE_FIRST_QUESTION) and "Latest result" not in both


def test_render_prompt_passes_ablation_options():
    from env.evaluation import render_prompt

    class Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, date_string=None):
            return messages[1]["content"]

    row = cpi_row()
    assert render_prompt(Tok(), row, True, True, ["rule"]) == build_user_message(row, True, True, ["rule"])
    assert render_prompt(Tok(), row, include_context=True, estimate_first=True) == build_user_message(row, True, estimate_first=True)


def test_paired_bootstrap_groups():
    from env.evaluation import paired_bootstrap_groups
    ev = eval_frame()
    ev["trained2"] = ev["trained"] - 0.05
    one = paired_bootstrap(ev, "base", "trained", n_boot=300, seed=3)
    grp = paired_bootstrap_groups(ev, ["base"], ["trained"], n_boot=300, seed=3)
    for c in ("change", "ci_low", "ci_high", "share_better"):
        assert np.allclose(one[c].to_numpy(float), grp[c].to_numpy(float), equal_nan=True)
    t = paired_bootstrap_groups(ev, ["base"], ["trained", "trained2"], n_boot=200, seed=0).set_index("metric")
    b1, b2 = M.brier(ev["trained"], ev["outcome"]), M.brier(ev["trained2"], ev["outcome"])
    assert t.loc["brier", "b"] == pytest.approx((b1 + b2) / 2)
    assert t.loc["brier", "change"] == pytest.approx((b1 + b2) / 2 - M.brier(ev["base"], ev["outcome"]))
    assert t.loc["brier", "b_min"] == pytest.approx(min(b1, b2)) and t.loc["brier", "b_max"] == pytest.approx(max(b1, b2))
    assert t.loc["brier", "ci_low"] <= t.loc["brier", "change"] <= t.loc["brier", "ci_high"]
    both = paired_bootstrap_groups(ev, ["trained", "trained2"], ["trained", "trained2"], n_boot=50).set_index("metric")
    assert (both["change"].abs() < 1e-12).all()


def test_text_key_and_reproduction_check():
    from env.evaluation import reproduction_check, text_key
    assert text_key("a\r\nb  ") == text_key("a\nb") != text_key("a b")
    assert text_key(None) == text_key(float("nan")) == text_key("")
    rows = [dict(market_ticker=f"Q{q}", sample=s, text=f"answer {q}-{s}", prob=(q % 5) / 5) for q in range(20) for s in range(4)]
    new = pd.DataFrame(rows)
    stored = new.assign(text_key=new["text"].map(text_key)).drop(columns="text")
    assert reproduction_check(new, stored)["status"] == "match"
    other = new.assign(text=new["text"] + " (other seed)")
    assert reproduction_check(other, stored)["status"] == "close"               # different text, same numbers
    moved = other.assign(prob=1 - other["prob"])
    r = reproduction_check(moved, stored)
    assert r["status"] == "MISMATCH" and r["corr"] == pytest.approx(-1.0) and r["questions"] == 20
    assert reproduction_check(new.assign(market_ticker="X"), stored)["status"] == "no overlap"
