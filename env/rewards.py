"""Rewards for GRPO. In words:

- Brier reward = 1 minus the squared gap between the stated probability and what happened
  (1 if the question resolved yes, 0 if no). A confident correct answer scores 1, always saying 0.5
  scores 0.75, a confident wrong answer scores 0. The Brier score is a "proper" scoring rule: over
  many questions, the best strategy is to state your honest belief.
- Log-score reward = the natural log of the probability the model gave to what actually happened,
  with probabilities clipped to [0.01, 0.99] so one confident miss can't dominate. Also proper, and
  harsher on confident mistakes than Brier. Used as an ablation.
- Format reward = +0.5 for a readable "Probability: 0.XX" line, -1.0 otherwise.

An unreadable answer gets the worst value of each scoring reward (0 for Brier, log(0.01) for log score).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from env.parser import extract_probability

FORMAT_OK, FORMAT_BAD = 0.5, -1.0
LOG_CLIP = 0.01


def brier_reward(p: float | None, outcome: int) -> float:
    return 0.0 if p is None else 1.0 - (p - outcome) ** 2


def log_score_reward(p: float | None, outcome: int, clip: float = LOG_CLIP) -> float:
    if p is None:
        return math.log(clip)
    p = min(max(p, clip), 1.0 - clip)
    return math.log(p if outcome == 1 else 1.0 - p)


def format_reward(text: str) -> float:
    return FORMAT_OK if extract_probability(text) is not None else FORMAT_BAD


# ---- TRL GRPOTrainer reward functions -----------------------------------------------------------
# TRL calls each with the batch's completions plus every dataset column as a keyword argument, so the
# dataset needs an `outcome` column (0/1). Completions are chat messages or plain strings.

def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def trl_format_reward(completions: Sequence[Any], **kwargs) -> list[float]:
    return [format_reward(_text(c)) for c in completions]


def trl_brier_reward(completions: Sequence[Any], outcome: Sequence[int], **kwargs) -> list[float]:
    return [brier_reward(extract_probability(_text(c)), int(y)) for c, y in zip(completions, outcome)]


def trl_log_score_reward(completions: Sequence[Any], outcome: Sequence[int], **kwargs) -> list[float]:
    return [log_score_reward(extract_probability(_text(c)), int(y)) for c, y in zip(completions, outcome)]
