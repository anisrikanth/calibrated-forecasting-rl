"""Builds notebooks/phase2_baselines.ipynb (run: python notebooks/build_notebooks.py).

Kept as a script so the notebook's code is reviewable in diffs. The install cell is the same pinned
recipe as Unsloth's Kaggle notebooks (LGPL-3.0), which worked in the Phase 0 smoke test.
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
HERE = Path(__file__).resolve().parent

INSTALL = '''os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
!pip install --upgrade -qqq uv
try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
except: _numpy = "numpy"; _pil = "pillow"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
_vllm, _triton = ('vllm==0.11.2', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')
!uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
!uv pip install -qqq {_triton} "huggingface_hub>=0.34.0" "datasets==4.3.0"
try:
    import importlib.metadata as _md; _torch_v = tuple(int(_p) for _p in _md.version("torch").split("+")[0].split(".")[:2])
except Exception:
    _torch_v = ()
_torchao = "torchao>=0.16.0" if _torch_v >= (2, 10) else "torchao>=0.16.0,<0.18.0"
!uv pip install -qqq --no-deps --upgrade "{_torchao}"
!uv pip install transformers==4.57.6
!uv pip install --no-deps trl==0.22.2'''

GPU_CHECK = '''import os, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU; must run before torch is imported
try:
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"]).decode())
except Exception as e:
    raise RuntimeError("No GPU found. Set Accelerator to 'GPU T4 x2' in the right sidebar "
                       "(needs a phone-verified Kaggle account).") from e'''


def baselines() -> nbf.NotebookNode:
    cells = [
        md("""# Phase 2 baselines: how good is Llama 3.2 before any RL?

Runs the base model on the **validation** economic questions (410) and a sample of **held-out weather** questions, two ways:

- **with context**: the series' recent results, as in training
- **without context**: the question alone. This doubles as the **leak probe**: if the model scored far better than a naive base rate *without* any context, it might have seen these outcomes in training.

and compares it with the **market price** (the benchmark), each series' **base rate**, and **always 0.5**.

**Before Run All**, in the right sidebar:
1. **Accelerator → GPU T4 x2**, **Internet → On**
2. **Add Input → Your Work → Datasets →** your `markets.csv` dataset
3. Private repo only: **Add-ons → Secrets**, add `GITHUB_TOKEN` (a GitHub token with repo access)

Use **Save Version → Save & Run All**. About 30–45 minutes for the 3B model. Then set `MODEL_NAME` to the 1B model and run again.

At the end, the notebook prints a block marked **COPY FROM HERE**: paste it back to Claude."""),
        md("## 1. Settings"),
        code('''MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"   # second run: "unsloth/Llama-3.2-1B-Instruct"
REPO_URL = "https://github.com/anisrikanth/calibrated-forecasting-rl.git"
N_SAMPLES = 4          # answers per question; we score the first one and the average of all four
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 320
WEATHER_EVENTS = 60    # held-out weather days sampled from 2026 (about 6 bracket markets each)
SEED = 3407
OUT_DIR = "/kaggle/working"
TAG = MODEL_NAME.split("/")[-1]'''),
        md("## 2. Check the GPU"),
        code(GPU_CHECK),
        md("## 3. Install Unsloth, vLLM and TRL (5–10 minutes)"),
        code(INSTALL),
        md("""## 4. Get the project code

Clones the repo into `/tmp` (not saved with the notebook's output, so a token never ends up in saved files) and imports the prompt, parser and metrics, so the notebook uses exactly the code that's tested in the repo."""),
        code('''import subprocess, sys

REPO_DIR = "/tmp/calibrated-forecasting-rl"

def _clone(url):
    return subprocess.run(["git", "clone", "--depth", "1", url, REPO_DIR], capture_output=True, text=True)

if not os.path.exists(REPO_DIR):
    result = _clone(REPO_URL)
    if result.returncode != 0:                      # private repo: try the Kaggle secret
        token = None
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except Exception:
            pass
        if not token:
            raise RuntimeError("Couldn't clone the repo. If it's private, add a Kaggle secret named GITHUB_TOKEN "
                               "(Add-ons > Secrets) or make the repo public.\\n" + result.stderr)
        result = _clone(REPO_URL.replace("https://", f"https://{token}@"))
        if result.returncode != 0:
            raise RuntimeError("Clone with token failed:\\n" + result.stderr.replace(token, "***"))
        subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL])

sys.path.insert(0, REPO_DIR)
from env.prompt import build_messages, chat_template_kwargs
from env.parser import extract_probability
from env import metrics as M
COMMIT = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Using repo commit", COMMIT)'''),
        md("""## 5. Load the data and pick the evaluation questions

- **val_econ:** every validation-split economic question (January–May 2026).
- **heldout_weather:** whole days (all of a day's temperature brackets) sampled from 2026, so the bracket probabilities can be checked for adding up to 1.
- **Base rate:** each economic series' yes-rate in the training split; each weather series' yes-rate in held-out days before 2026."""),
        code('''import glob
import numpy as np
import pandas as pd

paths = glob.glob("/kaggle/input/**/markets.csv", recursive=True)
assert paths, "No markets.csv found. Add your dataset: Add Input > Your Work > Datasets."
df = pd.read_csv(paths[0])
needed = {"question_time", "question_text", "context_text", "outcome", "split", "market_prob"}
assert needed <= set(df.columns), f"markets.csv is missing {needed - set(df.columns)}: upload the latest build."
df["close_dt"] = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")

val = df[df["split"] == "val"].copy()
held = df[df["split"] == "heldout"].copy()
recent_days = held.loc[held["close_dt"] >= "2026-01-01", "event_ticker"].drop_duplicates()
picked = recent_days.sample(n=min(WEATHER_EVENTS, len(recent_days)), random_state=SEED)
weather = held[held["event_ticker"].isin(picked)].copy()

base_econ = df[df["split"] == "train"].groupby("series_ticker")["outcome"].mean()
base_weather = held[held["close_dt"] < "2026-01-01"].groupby("series_ticker")["outcome"].mean()
base_rates = pd.concat([base_econ, base_weather])

eval_df = pd.concat([val.assign(eval_set="val_econ"), weather.assign(eval_set="heldout_weather")], ignore_index=True)
eval_df["base_rate"] = eval_df["series_ticker"].map(base_rates).fillna(df.loc[df["split"] == "train", "outcome"].mean())
print(eval_df.groupby("eval_set").agg(questions=("market_ticker", "size"), events=("event_ticker", "nunique"),
                                      yes_rate=("outcome", "mean"), priced=("market_prob", lambda s: s.notna().mean())))'''),
        md("## 6. Load the base model (no fine-tuning)"),
        code('''from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    load_in_4bit = False,
    fast_inference = True,          # vLLM for fast batched sampling
    gpu_memory_utilization = 0.85,  # lower to 0.7 if out of memory
)'''),
        md("""## 7. Build prompts and generate

Each question is shown with its date and time in US Eastern time. Llama's chat template is given the question's date, so its built-in "Today Date" line matches. Here is one full prompt:"""),
        code('''import time
from vllm import SamplingParams

def make_prompt(row, include_context):
    return tokenizer.apply_chat_template(build_messages(row, include_context), tokenize=False,
                                         add_generation_prompt=True, **chat_template_kwargs(row))

records = eval_df.to_dict("records")
print(make_prompt(records[0], include_context=True))'''),
        code('''params = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=0.95, max_tokens=MAX_NEW_TOKENS, seed=SEED)

def run_condition(include_context, name):
    start = time.time()
    prompts = [make_prompt(r, include_context) for r in records]
    outputs = model.fast_generate(prompts, sampling_params=params)
    rows = []
    for r, out in zip(records, outputs):
        for k, c in enumerate(out.outputs):
            rows.append({"market_ticker": r["market_ticker"], "condition": name, "sample": k,
                         "text": c.text, "prob": extract_probability(c.text), "n_tokens": len(c.token_ids)})
    print(f"{name}: {len(rows)} answers in {(time.time() - start) / 60:.1f} min")
    return pd.DataFrame(rows)

raw = pd.concat([run_condition(True, "ctx"), run_condition(False, "noctx")], ignore_index=True)
raw.to_csv(f"{OUT_DIR}/baseline_raw_{TAG}.csv", index=False)
print("Readable probability:", raw.groupby("condition")["prob"].apply(lambda s: round(s.notna().mean(), 3)).to_dict())
print("Median answer length (tokens):", raw.groupby("condition")["n_tokens"].median().to_dict())'''),
        md("Two example answers, to read how the model reasons:"),
        code('''for cond in ("ctx", "noctx"):
    ex = raw[(raw["condition"] == cond) & (raw["sample"] == 0)].sample(1, random_state=SEED).iloc[0]
    q = eval_df.set_index("market_ticker").loc[ex["market_ticker"]]
    print(f"===== {cond} | {ex['market_ticker']} | outcome {q['outcome']} | market {q['market_prob']}")
    print(ex["text"][:1500], "\\n")'''),
        md("""## 8. Score everything

Per question: **single** = the first answer; **mean4** = the average of the readable answers. An unreadable answer counts as 0.5. All methods are scored on the same questions: those with a market price."""),
        code('''first = raw[raw["sample"] == 0].set_index(["market_ticker", "condition"])["prob"]
mean4 = raw.groupby(["market_ticker", "condition"])["prob"].mean()
for cond in ("ctx", "noctx"):
    eval_df[f"model_{cond}_single"] = eval_df["market_ticker"].map(first.xs(cond, level="condition")).fillna(0.5)
    eval_df[f"model_{cond}_mean4"] = eval_df["market_ticker"].map(mean4.xs(cond, level="condition")).fillna(0.5)
eval_df["always_0.5"] = 0.5
eval_df["market"] = eval_df["market_prob"]

METHODS = ["market", "base_rate", "always_0.5", "model_ctx_single", "model_ctx_mean4", "model_noctx_single", "model_noctx_mean4"]
scored = eval_df[eval_df["market_prob"].notna()].copy()

rows = []
for (eval_set), g in scored.groupby("eval_set"):
    for m in METHODS:
        s = M.summarize(g[m], g["outcome"])
        rows.append({"eval_set": eval_set, "method": m, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}})
results = pd.DataFrame(rows)
results.to_csv(f"{OUT_DIR}/baseline_metrics_{TAG}.csv", index=False)
results[["eval_set", "method", "n", "brier", "log_loss", "ece", "reliability", "resolution"]]'''),
        md("By series (economic validation questions):"),
        code('''by_series = (scored[scored["eval_set"] == "val_econ"]
             .groupby("series_ticker")
             .apply(lambda g: pd.Series({"n": len(g),
                                         **{m: round(M.brier(g[m], g["outcome"]), 4)
                                            for m in ["market", "base_rate", "model_ctx_mean4", "model_noctx_mean4"]}})))
by_series["n"] = by_series["n"].astype(int)
by_series'''),
        md("""## 9. Consistency checks

- **Ladder violations:** within one economic release, "above X" probabilities should fall as X rises.
- **Bracket sums:** a weather day's bracket probabilities should add up to 1."""),
        code('''consistency = []
for m in ["market", "model_ctx_mean4", "model_noctx_mean4"]:
    lad = M.ladder_violations(scored[scored["eval_set"] == "val_econ"], m)
    part = M.partition_sum_error(eval_df[eval_df["eval_set"] == "heldout_weather"], m)
    consistency.append({"method": m, "ladder_pairs": lad["pairs"], "ladder_violation_rate": round(lad["violation_rate"], 3),
                        "weather_days": part["events"], "bracket_sum_abs_error": round(part["mean_abs_sum_error"], 3)})
consistency = pd.DataFrame(consistency)
consistency'''),
        md("""## 10. Leak probe

Without context, the model only has the question and the date. Beating the base rate that way is fine: the question text alone carries information ("payrolls above 0" is likely yes). What would be suspicious is scoring **about as well as the live market**, which trades on real-time information the model doesn't have. That would suggest the model memorized these outcomes. Llama 3.2's training data ends in December 2023 and every question here is from 2026, so we expect **no** flags. A series is flagged if its no-context Brier score is within 0.02 of the market's."""),
        code('''val_scored = scored[scored["eval_set"] == "val_econ"]
leak = val_scored.groupby("series_ticker").apply(lambda g: pd.Series({
    "n": len(g),
    "market_brier": round(M.brier(g["market"], g["outcome"]), 4),
    "base_rate_brier": round(M.brier(g["base_rate"], g["outcome"]), 4),
    "noctx_brier": round(M.brier(g["model_noctx_mean4"], g["outcome"]), 4),
}))
leak["n"] = leak["n"].astype(int)
leak["flag"] = np.where(leak["noctx_brier"] <= leak["market_brier"] + 0.02, "CHECK", "ok")
leak'''),
        md("## 11. Reliability diagrams"),
        code('''import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (eval_set, g) in zip(axes, scored.groupby("eval_set")):
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--", label="perfect calibration")
    for m in ["market", "model_ctx_mean4", "model_noctx_mean4"]:
        t = M.reliability_table(g[m], g["outcome"])
        ax.plot(t["mean_forecast"], t["frac_yes"], marker="o", label=m)
    ax.set_title(f"{eval_set} ({len(g)} questions)")
    ax.set_xlabel("Forecast probability (bin average)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0].set_ylabel("Share that resolved yes")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle(f"{TAG}: base model vs market")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/reliability_{TAG}.png", dpi=150)
plt.show()'''),
        md("## 12. Summary to paste back"),
        code('''fmt = raw.groupby("condition")["prob"].apply(lambda s: s.notna().mean())
print("=" * 25, "COPY FROM HERE", "=" * 25)
print(f"Model: {MODEL_NAME} | repo commit {COMMIT} | samples {N_SAMPLES} | temperature {TEMPERATURE}")
print("Readable probability rate:", {k: round(v, 3) for k, v in fmt.items()})
print("Median answer tokens:", raw.groupby("condition")["n_tokens"].median().to_dict())
print()
print(results[["eval_set", "method", "n", "brier", "log_loss", "ece", "resolution"]].to_string(index=False))
print()
print(by_series.to_string())
print()
print(consistency.to_string(index=False))
print()
print(leak.to_string())
print("=" * 25, "COPY TO HERE", "=" * 27)'''),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nbf.validate(nb)
    return nb


if __name__ == "__main__":
    out = HERE / "phase2_baselines.ipynb"
    nbf.write(baselines(), out)
    print("wrote", out)
