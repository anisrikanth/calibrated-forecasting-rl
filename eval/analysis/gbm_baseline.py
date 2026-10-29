"""Gradient-boosted trees on the key numbers: a stronger simple competitor than the logistic regression.

Two versions, both fitted on the training questions only and scored like every other method:
- `gbm`: the same 5 features as the logistic regression (env/baselines.py: share of recent results that
  would have resolved YES, whether the latest one would have, and the distances of the latest result, the
  recent average and the last change from the threshold, in units of the series' typical move).
- `gbm_series`: those 5 plus the series (CPI, payrolls, ...) as a categorical feature, so the trees can
  learn, e.g., that payrolls are noisy.
The procedure is fixed in advance: scikit-learn's HistGradientBoostingClassifier, hyperparameters chosen
from a small grid by 5-fold cross-validation on the training questions (folds keep each announcement whole:
all series released together, with all their thresholds),
lowest log loss wins, then refitted on all training questions. Questions without a numeric threshold
(Fed decisions) get the series base rate, as for the logistic regression.

Usage: python -I eval/analysis/gbm_baseline.py <repo_dir> <data_dir> <out_dir> [val|test]
data_dir holds markets.csv (and, for val, the eval answer CSVs of runs 4-6 for the paired comparison).
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

REPO, DATA, OUT = sys.argv[1:4]
SPLIT = sys.argv[4] if len(sys.argv) > 4 else "val"
sys.path.insert(0, REPO)
from env import metrics as M                                                         # noqa: E402
from env.baselines import rule_features                                              # noqa: E402
from env.evaluation import (add_forecast_columns, announcement_ids, load_markets,     # noqa: E402
                            paired_bootstrap, paired_bootstrap_groups, pick_eval_sets)

GRID = {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "max_iter": [100, 300]}
FIXED = dict(min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=0)


def features(rows, series_codes=None):
    f = [rule_features(r) for r in rows.to_dict("records")]
    ok = np.array([x is not None for x in f])
    X = np.vstack([x for x in f if x is not None]) if ok.any() else np.empty((0, 5))
    if series_codes is not None:
        codes = rows["series_ticker"].map(series_codes).fillna(-1).to_numpy()[ok]
        X = np.column_stack([X, codes])
    return X, ok


def make(params, with_series):
    cat = [False] * 5 + ([True] if with_series else [])
    return HistGradientBoostingClassifier(**params, **FIXED, categorical_features=cat)


def fit_gbm(train, with_series):
    codes = {s: i for i, s in enumerate(sorted(train["series_ticker"].unique()))} if with_series else None
    X, ok = features(train, codes)
    y = train["outcome"].to_numpy(int)[ok]
    groups = announcement_ids(train).to_numpy()[ok]           # folds keep each announcement whole
    best, results = None, []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        losses = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            p = make(params, with_series).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            losses.append(M.log_loss(p, y[te]))
        results.append({**params, "cv_log_loss": float(np.mean(losses))})
        if best is None or results[-1]["cv_log_loss"] < best[1]:
            best = (params, results[-1]["cv_log_loss"])
    model = make(best[0], with_series).fit(X, y)
    return model, codes, best[0], pd.DataFrame(results)


def predict(model, codes, rows, fallback):
    X, ok = features(rows, codes)
    out = np.asarray(fallback, dtype=float).copy()
    if ok.any():
        out[ok] = model.predict_proba(X)[:, 1]
    return out


df = load_markets(os.path.join(DATA, "markets.csv"))
train = df[df["split"] == "train"].reset_index(drop=True)
ev = pick_eval_sets(df, weather_events=60, seed=3407, econ_split=SPLIT)
cv_tables = []
for name, with_series in (("gbm", False), ("gbm_series", True)):
    model, codes, params, cv = fit_gbm(train, with_series)
    ev[name] = predict(model, codes, ev, ev["base_rate"])
    cv_tables.append(cv.assign(model=name))
    print(name, "chosen:", params)
cv = pd.concat(cv_tables, ignore_index=True)
ev = ev[ev["market_prob"].notna()].reset_index(drop=True)

methods = ["market", "gbm_series", "gbm", "logistic", "recent_hits", "latest_rule", "base_rate"]
answers = {}
if SPLIT == "val":
    for run in ("run4", "run5", "run6"):
        path = os.path.join(DATA, f"eval_answers_{run}.csv")
        if os.path.exists(path):
            ev = add_forecast_columns(ev, pd.read_csv(path), "trained", run)
            answers[run] = f"{run}_mean4"
rows = []
for es, g in ev.groupby("eval_set"):
    for m in methods + list(answers.values()):
        s = M.summarize(g[m], g["outcome"])
        rows.append({"eval_set": es, "method": m, **{k: s[k] for k in ("brier", "log_loss", "ece", "auc", "mean_p")}})
scores = pd.DataFrame(rows)

boots = []
for es, g in ev.groupby("eval_set"):
    g = g.reset_index(drop=True)
    for a, b in (("logistic", "gbm"), ("logistic", "gbm_series")):
        t = paired_bootstrap(g, a, b, n_boot=4000, seed=0, cluster="announcement").rename(columns={a: "a", b: "b"})
        boots.append(t.assign(eval_set=es, comparison=f"{b} vs {a}"))
    if len(answers) == 3:
        for ref in ("gbm", "gbm_series"):
            t = paired_bootstrap_groups(g, [ref], list(answers.values()), n_boot=4000, seed=0, cluster="announcement")
            boots.append(t.assign(eval_set=es, comparison=f"runs 4-6 vs {ref}"))
boot = pd.concat(boots, ignore_index=True)[["eval_set", "comparison", "metric", "a", "b", "change", "ci_low", "ci_high"]]

pd.set_option("display.width", 200)
print(scores.round(4).to_string(index=False)); print()
print(boot[boot.metric.isin(["brier", "log_loss", "ece", "auc"])].round(4).to_string(index=False))
scores.round(4).to_csv(os.path.join(OUT, f"gbm_{SPLIT}_scores.csv"), index=False)
boot.round(4).to_csv(os.path.join(OUT, f"gbm_{SPLIT}_paired_bootstrap.csv"), index=False)
cv.round(4).to_csv(os.path.join(OUT, "gbm_cv.csv"), index=False)
ev[["eval_set", "market_ticker", "outcome", "gbm", "gbm_series"]].round(4).to_csv(
    os.path.join(OUT, f"gbm_{SPLIT}_forecasts.csv"), index=False)
