"""Turns one row of data/processed/markets.csv into a chat prompt.

The model sees: the date and time the question is asked (US Eastern), the market's resolution
rule as the question, and (optionally) the series' recent results that were public by then.
It never sees the market price, the outcome, or anything dated after the question time.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

SYSTEM_PROMPT = (
    "You are a careful forecaster. You will get a yes/no question about an upcoming data release "
    "or measurement, the current date, and sometimes recent results for the same data series.\n"
    "Think it through briefly (under 150 words), then end with one final line in exactly this form:\n"
    "Probability: 0.XX\n"
    "where 0.XX is your probability, between 0 and 1, that the question resolves YES."
)

NO_CONTEXT_NOTE = "No recent results are provided for this question."


def question_datetime_et(question_time: Any) -> pd.Timestamp:
    """Question time (UTC string or timestamp) in US Eastern time."""
    ts = pd.Timestamp(question_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York")


def format_question_time(question_time: Any) -> str:
    """e.g. 'Thursday, October 1, 2026, 8:29 AM Eastern'."""
    t = question_datetime_et(question_time)
    hour = t.hour % 12 or 12
    return f"{t:%A}, {t:%B} {t.day}, {t.year}, {hour}:{t:%M} {'AM' if t.hour < 12 else 'PM'} Eastern"


def build_user_message(row: Mapping[str, Any], include_context: bool = True) -> str:
    parts = [f"Current date: {format_question_time(row['question_time'])}",
             f"Question: {row['question_text']}"]
    context = row.get("context_text") if include_context else None
    parts.append(context if isinstance(context, str) and context.strip() else NO_CONTEXT_NOTE)
    parts.append("What is the probability that this question resolves YES?")
    return "\n\n".join(parts)


def build_messages(row: Mapping[str, Any], include_context: bool = True) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(row, include_context)}]


def chat_template_kwargs(row: Mapping[str, Any]) -> dict:
    """Llama 3.x chat templates write "Today Date: ..." into the system header, defaulting to a fixed
    2024 date. Passing the question's own date keeps the header consistent with the question."""
    t = question_datetime_et(row["question_time"])
    return {"date_string": f"{t.day:02d} {t:%b} {t.year}"}
