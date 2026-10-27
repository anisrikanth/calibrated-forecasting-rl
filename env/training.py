"""GRPO training helpers: the training dataset and a reward function that also watches for hacks.

Prompts are rendered to text here (rather than passed as chat messages for TRL to format) so each
question's own date goes into Llama's chat template, exactly as in evaluation. TRL tokenizes text
prompts without adding special tokens, so the rendered <|begin_of_text|> isn't doubled.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from env.evaluation import render_prompt
from env.parser import extract_probability
from env.rewards import _text, brier_reward, log_score_reward


def build_train_records(df: pd.DataFrame, tokenizer: Any, include_context: bool = True,
                        split: str = "train", seed: int = 3407, key_numbers: bool = False,
                        shuffle_outcomes: bool = False) -> list[dict]:
    """One record per training question: rendered prompt text, outcome, and ids for logging.
    (TRL reshuffles the dataset itself each pass; `shuffle_dataset=True` by default.)

    `shuffle_outcomes=True` is the control run: outcomes are randomly reassigned across the training
    questions, keeping the overall yes-rate but breaking any link between a question and its outcome.
    The reward then uses the shuffled outcome; `true_outcome` keeps the real one for analysis."""
    rows = df[df["split"] == split].sample(frac=1.0, random_state=seed).to_dict("records")
    true = [int(r["outcome"]) for r in rows]
    used = list(np.random.default_rng(seed).permutation(true)) if shuffle_outcomes else true
    return [{"prompt": render_prompt(tokenizer, r, include_context, key_numbers), "outcome": int(u),
             "true_outcome": t, "series_ticker": r["series_ticker"], "market_ticker": r["market_ticker"]}
            for r, u, t in zip(rows, used, true)]


def window_stats(probs: Sequence[float | None], outcomes: Sequence[int]) -> dict:
    """What the model has been stating recently, in plain terms:
    format_ok   share of answers with a readable probability
    mean_p      average stated probability
    sd_p        spread of stated probabilities (near 0 means it says the same number every time)
    near_half   share of stated probabilities between 0.45 and 0.55
    gap         average probability on questions that resolved yes minus on those that resolved no.
                This is the discrimination signal: hedging can lower the Brier score while the gap stays at 0.
    brier       Brier score of readable answers"""
    p = np.array([np.nan if x is None else x for x in probs], dtype=float)
    y = np.asarray(outcomes, dtype=float)
    ok = ~np.isnan(p)
    out = {"n": int(len(p)), "format_ok": float(ok.mean()) if len(p) else float("nan")}
    if ok.any():
        pv, yv = p[ok], y[ok]
        out.update(mean_p=float(pv.mean()), sd_p=float(pv.std()),
                   near_half=float(((pv >= 0.45) & (pv <= 0.55)).mean()),
                   brier=float(((pv - yv) ** 2).mean()))
        out["gap"] = (float(pv[yv == 1].mean() - pv[yv == 0].mean())
                      if (yv == 1).any() and (yv == 0).any() else float("nan"))
    else:
        out.update(mean_p=float("nan"), sd_p=float("nan"), near_half=float("nan"),
                   brier=float("nan"), gap=float("nan"))
    return out


class MonitoredScoreReward:
    """A scoring reward (Brier or log score) for TRL's GRPOTrainer that also records every stated
    probability, and every `print_every` calls prints recent stats plus one sample answer.

    TRL calls it with the batch's completions and each dataset column as keyword arguments, so the
    dataset needs `outcome` (and `series_ticker` for the log)."""

    def __init__(self, kind: str = "brier", window: int = 256, print_every: int = 10,
                 log: Callable[[str], None] = print):
        if kind not in ("brier", "log"):
            raise ValueError("kind must be 'brier' or 'log'")
        self.kind = kind
        self.__name__ = f"{kind}_reward"            # TRL names logged reward columns after this
        self.score = brier_reward if kind == "brier" else log_score_reward
        self.window, self.print_every, self.log = window, print_every, log
        self.records: list[dict] = []
        self.calls = 0

    def __call__(self, completions: Sequence[Any], outcome: Sequence[int], **kwargs) -> list[float]:
        texts = [_text(c) for c in completions]
        probs = [extract_probability(t) for t in texts]
        rewards = [self.score(p, int(y)) for p, y in zip(probs, outcome)]
        series = kwargs.get("series_ticker") or [None] * len(texts)
        true = kwargs.get("true_outcome") or outcome
        for p, y, r, s, t in zip(probs, outcome, rewards, series, true):
            self.records.append({"call": self.calls, "prob": p, "outcome": int(y), "reward": r, "series": s,
                                 "true_outcome": int(t)})
        if self.calls % self.print_every == 0:
            recent = self.records[-self.window:]
            st = window_stats([r["prob"] for r in recent], [r["outcome"] for r in recent])
            self.log(f"[call {self.calls}] last {st['n']}: format_ok {st['format_ok']:.2f} | "
                     f"mean_p {st['mean_p']:.2f} | sd_p {st['sd_p']:.2f} | near_half {st['near_half']:.2f} | "
                     f"gap {st['gap']:+.2f} | brier {st['brier']:.3f}")
            self.log(f"    sample (outcome {int(outcome[0])}, reward {rewards[0]:.2f}): "
                     + texts[0][-300:].replace("\n", " "))
        self.calls += 1
        return rewards

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["call", "prob", "outcome", "reward", "series", "true_outcome"])

    def first_last(self, n: int | None = None) -> pd.DataFrame:
        """Stats on the first and last `n` answers of training, side by side. By default n is 512,
        or half the run if it's shorter, so the two windows never overlap."""
        h = self.records
        n = n if n is not None else max(1, min(512, len(h) // 2))
        rows = {"first": window_stats([r["prob"] for r in h[:n]], [r["outcome"] for r in h[:n]]),
                "last": window_stats([r["prob"] for r in h[-n:]], [r["outcome"] for r in h[-n:]])}
        return pd.DataFrame(rows).T.round(3)
