"""Tests for env/evaluation.py and env/training.py with a fake tokenizer and fake model outputs."""
import math
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from env.evaluation import (add_forecast_columns, by_series_table, collect_answers, consistency_table,
                            leak_table, pick_eval_sets, render_prompt, score_table)
from env.training import MonitoredScoreReward, build_train_records, window_stats


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, date_string=None):
        assert not tokenize and add_generation_prompt
        assert re.fullmatch(r"\d\d \w{3} \d{4}", date_string)
        return f"<|begin_of_text|>[{date_string}]" + " || ".join(m["content"] for m in messages)


def make_df():
    rows = []

    def add(series, split, event, close, strike_type, floor, cap, outcome, mp):
        rows.append(dict(market_ticker=f"{event}-{strike_type}-{floor or cap}", event_ticker=event,
                         series_ticker=series, split=split, outcome=outcome, market_prob=mp,
                         question_time=(pd.Timestamp(close) - pd.Timedelta(hours=24)).isoformat(),
                         close_time=pd.Timestamp(close).isoformat(), strike_type=strike_type,
                         floor_strike=floor, cap_strike=cap, question_text=f"{series} above {floor}?",
                         context_text="Recent results:\n- x: 1"))

    for split, months in (("train", range(0, 4)), ("val", range(4, 6))):
        for m in months:
            ev = f"KXCPI-{m}"
            close = pd.Timestamp("2025-10-10 13:30", tz="UTC") + pd.DateOffset(months=m)
            for th, y, p in ((1, 1, 0.9), (2, 0, 0.2)):
                add("KXCPI", split, ev, close, "greater", th, None, y, p)
    for d, day in enumerate(["2025-12-01", "2025-12-02", "2026-02-01", "2026-02-02", "2026-02-03"]):
        ev = f"KXHIGHNY-{d}"
        for b, st in enumerate(["less", "between", "greater"]):
            add("KXHIGHNY", "heldout", ev, f"{day} 05:00", st, None if b == 0 else 60 + b, 60 if b == 0 else None,
                int(b == 1), 0.6 if b == 1 else 0.2)
    df = pd.DataFrame(rows)
    df["close_dt"] = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")
    return df


def test_pick_eval_sets():
    ev = pick_eval_sets(make_df(), weather_events=2, seed=0)
    econ, wx = ev[ev.eval_set == "val_econ"], ev[ev.eval_set == "heldout_weather"]
    assert len(econ) == 4 and (econ.split == "val").all()
    assert wx["event_ticker"].nunique() == 2 and (wx["close_dt"] >= "2026-01-01").all()
    assert (wx.groupby("event_ticker").size() == 3).all()                      # whole days only
    assert econ["base_rate"].eq(0.5).all()                                        # train KXCPI: 4 yes of 8
    assert wx["base_rate"].iloc[0] == pytest.approx(1 / 3)                        # 2025 weather days
    assert ev["always_0.5"].eq(0.5).all() and ev["market"].equals(ev["market_prob"])


def test_render_prompt_uses_question_date():
    row = make_df().iloc[0].to_dict()
    text = render_prompt(FakeTokenizer(), row, include_context=False)
    assert text.startswith("<|begin_of_text|>[09 Oct 2025]") and "No recent results" in text


def fake_outputs(texts_per_prompt):
    return [SimpleNamespace(outputs=[SimpleNamespace(text=t, token_ids=[0] * 3) for t in texts])
            for texts in texts_per_prompt]


def test_collect_answers_and_forecast_columns():
    ev = pick_eval_sets(make_df(), weather_events=2, seed=0)
    rows = ev.to_dict("records")[:2]
    out = fake_outputs([["Probability: 0.8", "Probability: 0.6", "junk", "Probability: 0.7"],
                        ["junk", "junk", "junk", "junk"]])
    ans = collect_answers(rows, out, "ctx")
    assert len(ans) == 8 and ans["n_tokens"].eq(3).all()
    ev2 = add_forecast_columns(ev, ans, "ctx", "lora_ctx")
    first, second = ev2.iloc[0], ev2.iloc[1]
    assert first["lora_ctx_single"] == 0.8 and first["lora_ctx_mean4"] == pytest.approx(0.7)
    assert second["lora_ctx_single"] == 0.5 and second["lora_ctx_mean4"] == 0.5   # unreadable -> 0.5
    assert ev2.iloc[2]["lora_ctx_mean4"] == 0.5                                      # never asked -> 0.5


def test_tables():
    ev = pick_eval_sets(make_df(), weather_events=2, seed=0)
    ev["model"] = 0.5
    t = score_table(ev, ["market", "always_0.5", "model"])
    assert set(t["eval_set"]) == {"val_econ", "heldout_weather"} and len(t) == 6
    m = t[(t.eval_set == "val_econ") & (t.method == "market")].iloc[0]
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.01, 0.04]))
    s = by_series_table(ev, ["market", "model"])
    assert list(s.index) == ["KXCPI"] and s.loc["KXCPI", "n"] == 4
    c = consistency_table(ev, ["market", "model"]).set_index("method")
    assert c.loc["market", "ladder_violation_rate"] == 0 and c.loc["market", "bracket_sum_abs_error"] == 0
    assert c.loc["model", "bracket_sum_abs_error"] == pytest.approx(0.5)            # 3 x 0.5 = 1.5
    leak = leak_table(ev, "model")
    expected = (0.25 - 0.25) / (0.25 - 0.025)
    assert leak.loc["KXCPI", "share_of_market_edge"] == pytest.approx(round(expected, 2))
    assert leak.loc["KXCPI", "flag"] == "ok"


def test_build_train_records():
    df = make_df()
    recs = build_train_records(df, FakeTokenizer(), include_context=True, seed=1)
    assert len(recs) == 8 and {r["outcome"] for r in recs} == {0, 1}
    assert all(r["prompt"].startswith("<|begin_of_text|>[") and "Recent results" in r["prompt"] for r in recs)
    assert [r["market_ticker"] for r in recs] == [r["market_ticker"] for r in build_train_records(df, FakeTokenizer(), seed=1)]
    assert set(r["market_ticker"] for r in recs) == set(df[df.split == "train"]["market_ticker"])


def test_window_stats():
    st = window_stats([0.9, 0.8, 0.2, None], [1, 1, 0, 0])
    assert st["format_ok"] == 0.75 and st["gap"] == pytest.approx(0.65)
    assert st["mean_p"] == pytest.approx(0.6333, abs=1e-3)
    hedge = window_stats([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    assert hedge["sd_p"] == 0 and hedge["near_half"] == 1 and hedge["gap"] == 0
    assert math.isnan(window_stats([None], [1])["mean_p"])


def test_monitored_reward_scores_logs_and_records():
    lines = []
    r = MonitoredScoreReward("brier", print_every=1, log=lines.append)
    assert r.__name__ == "brier_reward"
    comps = [[{"role": "assistant", "content": "Reasoning.\nProbability: 0.9"}], "Probability: 0.2", "nothing"]
    out = r(completions=comps, outcome=[1, 1, 0], series_ticker=["A", "B", "C"], prompts=["p"] * 3)
    assert [round(x, 4) for x in out] == [0.99, 0.36, 0.0]
    assert len(r.history()) == 3 and r.history()["series"].tolist() == ["A", "B", "C"]
    assert lines and "format_ok 0.67" in lines[0] and "sample (outcome 1" in lines[1]
    lg = MonitoredScoreReward("log", log=lambda s: None)
    assert lg(["Probability: 0.8"], [1])[0] == pytest.approx(math.log(0.8))
    fl = r.first_last(n=2)
    assert list(fl.index) == ["first", "last"] and fl.loc["first", "n"] == 2
    with pytest.raises(ValueError):
        MonitoredScoreReward("accuracy")


def test_pick_eval_sets_test_split_without_weather():
    df = make_df()
    df.loc[df["event_ticker"] == "KXCPI-5", "split"] = "test"
    ev = pick_eval_sets(df, weather_events=0, econ_split="test")
    assert set(ev["eval_set"]) == {"test_econ"} and len(ev) == 2 and (ev["split"] == "test").all()
    assert ev["base_rate"].eq(0.5).all() and ev["logistic"].notna().all()
    default = pick_eval_sets(df, weather_events=2, seed=0)                       # default unchanged
    assert set(default["eval_set"]) == {"val_econ", "heldout_weather"} and (default.loc[default.eval_set == "val_econ", "split"] == "val").all()


def test_consistency_table_on_another_econ_set():
    ev = pick_eval_sets(make_df(), weather_events=2, seed=0)
    ev["m"] = 0.5
    t = consistency_table(ev.assign(eval_set=ev["eval_set"].replace({"val_econ": "test_econ"})), ["m"], econ_set="test_econ")
    assert t.loc[0, "ladder_pairs"] == consistency_table(ev, ["m"]).loc[0, "ladder_pairs"] > 0
