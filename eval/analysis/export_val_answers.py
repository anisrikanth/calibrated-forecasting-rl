"""Writes eval/answers/val_<label>.csv: each model's validation-econ answers (stated probability and a
fingerprint of the text, not the text itself) for every saved model. The test-set notebook re-runs a
few of these questions with each saved adapter and checks it gets the same answers, which shows the
adapter was loaded correctly before the test set is scored.

Usage: python -I eval/analysis/export_val_answers.py <repo_dir> <data_dir>
data_dir holds markets.csv and the eval answer CSVs downloaded from Kaggle, renamed as in FILES.
"""
import os
import sys

import pandas as pd

REPO, DATA = sys.argv[1:3]
sys.path.insert(0, REPO)
from env.evaluation import load_markets, text_key  # noqa: E402

# label -> (answer file, condition in that file)
FILES = {
    "untrained_plain": ("eval_answers_v1.csv", "base"),
    "untrained_keynum": ("eval_answers_run3.csv", "base"),
    "run1": ("eval_answers_lr1e-5.csv", "trained"),
    "run2": ("eval_answers_run2.csv", "trained"),
    **{f"run{i}": (f"eval_answers_run{i}.csv", "trained") for i in range(3, 11)},
}
df = load_markets(f"{DATA}/markets.csv")
val = set(df.loc[df["split"] == "val", "market_ticker"])
out_dir = os.path.join(REPO, "eval", "answers")
os.makedirs(out_dir, exist_ok=True)
for label, (path, cond) in FILES.items():
    if not os.path.exists(f"{DATA}/{path}"):
        print("missing", path, "- skipped")
        continue
    a = pd.read_csv(f"{DATA}/{path}")
    a = a[(a["condition"] == cond) & a["market_ticker"].isin(val)]
    assert a["market_ticker"].nunique() == len(val) and len(a) == 4 * len(val), label
    a = a.assign(text_key=a["text"].map(text_key))[["market_ticker", "sample", "prob", "text_key"]]
    a.sort_values(["market_ticker", "sample"]).to_csv(f"{out_dir}/val_{label}.csv", index=False)
    print(f"val_{label}.csv: {len(a)} answers")
