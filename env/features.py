"""Key numbers: the comparison a forecaster would make, spelled out in the prompt.

Run 1 suggested the model can't turn raw recent results into a forecast: to answer "above 2.9%?"
it has to compare 2.9 with the latest values, which a 3B model does poorly from text. This block
states that comparison directly. It uses only the question's threshold and the recent results that
are already in the context, so it adds no new information and no leakage:

    Key numbers (worked out from the question and the recent results above):
    - This question resolves YES if the result is above 2.5%.
    - Latest result (Inflation in Aug 2025): 2.9%, 0.4 percentage points above the threshold.
    - Change from the result before: +0.2 (from 2.7% to 2.9%).
    - Recent results that would have resolved YES: 3 of 6.
    - Recent results: average 2.57%, lowest 2.3%, highest 2.9%.

Thresholds: "greater" questions use floor_strike ("more than X"), "less" use cap_strike ("less than X"),
"between" use both (inclusive, e.g. "between 65-66°"). Questions without a numeric threshold
(FOMC hike/cut/hold) get no block.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Mapping, Optional, Tuple

HEADER = "Key numbers (worked out from the question and the recent results above):"
DIFF_UNITS = {"%": " percentage points", "°": "°", "": ""}


def _num(text: Any) -> Optional[float]:
    """'147,000' -> 147000.0; '2.8%' -> 2.8; 'Cut 25bps' -> None."""
    if text is None:
        return None
    s = str(text).strip().replace(",", "").rstrip("%").rstrip("°").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def fmt(x: float, unit: str = "") -> str:
    """At most 2 decimals, thousands separators, float noise removed: 147000 -> '147,000';
    0.30000000004 -> '0.3'; 2.5666 -> '2.57'; 4.0 -> '4'."""
    s = f"{x:,.2f}".rstrip("0").rstrip(".")
    return ("0" if s in ("-0", "") else s) + unit


def unit_of(row: Mapping[str, Any]) -> str:
    q = str(row.get("question_text") or "")
    if "%" in q:
        return "%"
    if "°" in q:
        return "°"
    return ""


def threshold_rule(row: Mapping[str, Any], unit: str = "") -> Optional[Tuple[str, Callable[[float], bool], tuple]]:
    """(description, yes-test, reference) for the question's yes-condition, or None if it has no
    numeric threshold."""
    st, lo, hi = row.get("strike_type"), row.get("floor_strike"), row.get("cap_strike")
    if st == "greater" and _present(lo):
        return f"above {fmt(lo, unit)}", (lambda v: v > lo), ("threshold", lo)
    if st == "greater_or_equal" and _present(lo):
        return f"at or above {fmt(lo, unit)}", (lambda v: v >= lo), ("threshold", lo)
    if st == "less" and _present(hi):
        return f"below {fmt(hi, unit)}", (lambda v: v < hi), ("threshold", hi)
    if st == "less_or_equal" and _present(hi):
        return f"at or below {fmt(hi, unit)}", (lambda v: v <= hi), ("threshold", hi)
    if st == "between" and _present(lo) and _present(hi):
        return (f"from {fmt(lo, unit)} to {fmt(hi, unit)} (inclusive)", (lambda v: lo <= v <= hi),
                ("range", (lo, hi)))
    return None


def _position(v: float, ref: tuple, unit: str) -> str:
    du = DIFF_UNITS.get(unit, "")
    kind, val = ref
    if kind == "threshold":
        d = v - val
        if abs(d) < 1e-9:
            return "equal to the threshold"
        return f"{fmt(abs(d))}{du} {'above' if d > 0 else 'below'} the threshold"
    lo, hi = val
    if v < lo - 1e-9:
        return f"{fmt(lo - v)}{du} below the range"
    if v > hi + 1e-9:
        return f"{fmt(v - hi)}{du} above the range"
    return "inside the range"


def _clean_label(label: Any) -> str:
    return re.sub(r"\s+", " ", str(label or "").replace("?", "")).strip()


def key_numbers(row: Mapping[str, Any]) -> Optional[dict]:
    """The numbers behind the block, or None if the question has no numeric threshold.
    `values` are the recent numeric results, oldest first; `hits` counts those that would have
    resolved YES under this question's rule."""
    unit = unit_of(row)
    rule = threshold_rule(row, unit)
    if rule is None:
        return None
    desc, test, ref = rule
    try:
        items = json.loads(row.get("context_json") or "[]")
    except (TypeError, ValueError):
        items = []
    pairs = [(_clean_label(it.get("label")), _num(it.get("value"))) for it in items if isinstance(it, dict)]
    pairs = [(label, v) for label, v in pairs if v is not None]
    return {"unit": unit, "rule": desc, "ref": ref, "labels": [label for label, _ in pairs],
            "values": [v for _, v in pairs], "hits": sum(bool(test(v)) for _, v in pairs)}


def recent_hit_rate(row: Mapping[str, Any]) -> Optional[float]:
    """Share of recent results that would have resolved YES (a simple rule-based baseline)."""
    k = key_numbers(row)
    return k["hits"] / len(k["values"]) if k and k["values"] else None


def key_numbers_text(row: Mapping[str, Any]) -> Optional[str]:
    k = key_numbers(row)
    if k is None:
        return None
    unit, nums = k["unit"], k["values"]
    lines = [HEADER, f"- This question resolves YES if the result is {k['rule']}."]
    if not nums:
        lines.append("- No earlier numeric results are available.")
        return "\n".join(lines)
    label, latest = k["labels"][-1], nums[-1]
    lines.append(f"- Latest result{f' ({label})' if label else ''}: {fmt(latest, unit)}, "
                 f"{_position(latest, k['ref'], unit)}.")
    if len(nums) >= 2:
        prev, d = nums[-2], nums[-1] - nums[-2]
        sign = "+" if d > 1e-9 else ("-" if d < -1e-9 else "")
        lines.append(f"- Change from the result before: {sign}{fmt(abs(d))}{DIFF_UNITS.get(unit, '')} "
                     f"(from {fmt(prev, unit)} to {fmt(latest, unit)}).")
    lines += [
        f"- Recent results that would have resolved YES: {k['hits']} of {len(nums)}.",
        f"- Recent results: average {fmt(sum(nums) / len(nums), unit)}, lowest {fmt(min(nums), unit)}, "
        f"highest {fmt(max(nums), unit)}.",
    ]
    return "\n".join(lines)
