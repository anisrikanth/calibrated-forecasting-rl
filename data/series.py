"""Which Kalshi series to pull, and how far before close each question is asked.

All tickers below were confirmed with `python -m data.pull_kalshi discover` (Oct 26, 2026).
Kalshi renamed many series to a "KX" prefix, and the older names still exist, so each series
is listed under both names. Both share a `family`; if the same question shows up under both
names, `build` keeps one copy (preferring the KX one).

group:          "econ" series are used for training/validation/test; groups listed in
                HELDOUT_GROUPS are kept entirely out of training (the generalization test).
family:         the underlying series, shared by its KX and older tickers.
horizon_hours:  question time = close time minus this many hours. The model only gets
                information from before this time, and the market price at this time is the
                benchmark. Clamped to at least 1 hour after the market opened.
"""

# Llama 3.2's knowledge cutoff is December 2023, so only markets that closed after this are used.
SINCE_DEFAULT = "2024-01-01"


def _both_names(family: str, group: str, horizon_hours: float, note: str) -> dict:
    """The KX ticker and the older ticker for the same series."""
    cfg = {"group": group, "family": family, "horizon_hours": horizon_hours, "note": note}
    return {f"KX{family}": dict(cfg), family: dict(cfg, note=note + " (older ticker)")}


SERIES = {
    # Training category: economic data releases
    **_both_names("CPI",         "econ", 24, "CPI, month over month"),
    **_both_names("CPIYOY",      "econ", 24, "CPI, year over year"),
    **_both_names("CPICORE",     "econ", 24, "Core CPI, month over month"),
    **_both_names("CPICOREYOY",  "econ", 24, "Core CPI, year over year"),
    **_both_names("PAYROLLS",    "econ", 24, "Nonfarm payrolls"),
    **_both_names("U3",          "econ", 24, "Unemployment rate"),
    **_both_names("FED",         "econ", 24, "Fed funds target rate after an FOMC meeting"),
    **_both_names("FEDDECISION", "econ", 24, "FOMC decision: hike, cut or hold"),
    # Held-out category: daily high temperature
    **_both_names("HIGHNY",      "weather", 18, "NYC daily high temperature"),
    **_both_names("HIGHCHI",     "weather", 18, "Chicago daily high temperature"),
}

HELDOUT_GROUPS = ("weather",)

# Categories listed by `discover`. Kalshi matches these exactly; run `discover --category ""`
# to list every series.
DISCOVER_CATEGORIES = ("Economics", "Climate and Weather")
