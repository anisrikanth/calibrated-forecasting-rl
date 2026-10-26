"""Which Kalshi series to pull, and how far before close each question is asked.

group:          "econ" series are used for training/validation/test; groups listed in
                HELDOUT_GROUPS are kept entirely out of training (the generalization test).
horizon_hours:  question time = close time minus this many hours. The model only gets
                information from before this time, and the market price at this time is the
                benchmark. Clamped to at least 1 hour after the market opened.

Only KXPAYROLLS has been confirmed so far. Run `python -m data.pull_kalshi discover` to see the
real tickers in each category, then edit this list. Tickers that don't exist are skipped with
a warning, so an unconfirmed entry costs nothing. Kalshi renamed many series to a "KX" prefix;
if `discover` shows an older name for the same series (e.g. without "KX"), add it too, since
early-2024 markets may live under it.
"""

# Llama 3.2's knowledge cutoff is December 2023, so only markets that closed after this are used.
SINCE_DEFAULT = "2024-01-01"

SERIES = {
    # Training category: economic data releases
    "KXCPI":      {"group": "econ", "horizon_hours": 24, "note": "CPI, month over month"},
    "KXCPIYOY":   {"group": "econ", "horizon_hours": 24, "note": "CPI, year over year"},
    "KXPAYROLLS": {"group": "econ", "horizon_hours": 24, "note": "Nonfarm payrolls (confirmed ticker)"},
    "KXFED":      {"group": "econ", "horizon_hours": 24, "note": "Fed funds target rate after an FOMC meeting"},
    # Held-out category: daily high temperature
    "KXHIGHNY":   {"group": "weather", "horizon_hours": 18, "note": "NYC daily high temperature"},
    "KXHIGHCHI":  {"group": "weather", "horizon_hours": 18, "note": "Chicago daily high temperature"},
}

HELDOUT_GROUPS = ("weather",)

# Categories listed by `discover`. Kalshi matches these exactly; if one returns nothing,
# run `discover --category ""` to list every series and find the right category name.
DISCOVER_CATEGORIES = ("Economics", "Climate and Weather")
