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

Without context, the model only has the question and the date. Beating the base rate that way is fine: the question text alone carries information ("payrolls above 0" is likely yes). What would be suspicious is capturing **most of the live market's advantage**, since the market trades on real-time information the model doesn't have.

**Share of market edge** = (base-rate Brier − no-context Brier) ÷ (base-rate Brier − market Brier). 0 means no better than the base rate; 1 means as good as the market. A series is flagged if the share is above 0.5. Series where the market barely beats the base rate (by less than 0.02) can't be judged and show as blank. Llama 3.2's training data ends in December 2023 and every question here is from 2026, so we expect **no** flags."""),
        code('''val_scored = scored[scored["eval_set"] == "val_econ"]
leak = val_scored.groupby("series_ticker").apply(lambda g: pd.Series({
    "n": len(g),
    "market_brier": round(M.brier(g["market"], g["outcome"]), 4),
    "base_rate_brier": round(M.brier(g["base_rate"], g["outcome"]), 4),
    "noctx_brier": round(M.brier(g["model_noctx_mean4"], g["outcome"]), 4),
}))
leak["n"] = leak["n"].astype(int)
market_edge = leak["base_rate_brier"] - leak["market_brier"]
leak["share_of_market_edge"] = ((leak["base_rate_brier"] - leak["noctx_brier"]) / market_edge).where(market_edge >= 0.02).round(2)
leak["flag"] = np.where(leak["share_of_market_edge"] > 0.5, "CHECK", "ok")
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


CLONE_REPO = '''import subprocess, sys

REPO_DIR = "/tmp/calibrated-forecasting-rl"
if not os.path.exists(REPO_DIR):
    result = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Couldn't clone the repo (is it public?):\\n" + result.stderr)
sys.path.insert(0, REPO_DIR)
COMMIT = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("Using repo commit", COMMIT)'''


def training() -> nbf.NotebookNode:
    cells = [
        md("""# Phase 3: GRPO training on Kalshi forecasting questions

Trains a LoRA adapter on Llama 3.2 with GRPO. For each training question the model writes several answers. Each answer is scored with the **Brier reward** (1 minus the squared gap between its stated probability and what happened) plus a small **format reward**, and the model is nudged toward its better answers. Then it **evaluates the trained model and the untrained model in the same session** on the validation economic questions and held-out weather days, scored the same way as the Phase 2 baselines.

**Before Run All**, in the right sidebar: **Accelerator → GPU T4 x2**, **Internet → On**, **Add Input → Your Work → Datasets →** your `markets.csv` dataset.

**Runs** (change only the Settings cell, then Save & Run All):
- Run 1: `RUN_NAME = "3b_brier_ctx_lr1e-5"`, `LEARNING_RATE = 1e-5`, `MAX_STEPS = -1`, `KEY_NUMBERS = False`. One pass over the 1,391 training questions, about 2.5 hours including evaluation.
- Run 2: as run 1 but 2 passes at `LEARNING_RATE = 2e-5`.
- **Run 3 (the settings below):** as run 1 but `KEY_NUMBERS = True`. Each prompt gets a short block that spells out the comparison between the question's threshold and the recent results (latest result vs threshold, change, how many recent results would have resolved YES, average and range). The untrained model is scored with the same prompt, so the run separates "the prompt helps" from "RL learned to use it".
- Run 4: run 2's settings (`LEARNING_RATE = 2e-5`, `MAX_STEPS = 694`) with `SCALE_REWARDS = "none"` and `KEY_NUMBERS = False`. Run 2 collapsed to near-0/near-1 forecasts; this tests whether GRPO's division by the group's standard deviation causes it.
- For a quick check of a new setup, use `MAX_STEPS = 30` first (about 40 minutes).

**While it runs**, open the version's **Logs** to see lines like `format_ok 0.97 | mean_p 0.48 | sd_p 0.21 | gap +0.05 | brier 0.24` every 10 reward calls. The number to watch is **gap**: the average probability on questions that resolved yes minus on those that resolved no. Brier falling while gap stays near 0 means the model is hedging, not learning.

At the end, copy the block between **COPY FROM HERE** and **COPY TO HERE** back to Claude."""),
        md("## 1. Settings"),
        code('''MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
RUN_NAME = "3b_brier_keynum_lr1e-5"   # names the saved adapter and output files
REWARD = "brier"                 # "brier" (main) or "log" (ablation)
INCLUDE_CONTEXT = True           # show the series' recent results in the prompt
KEY_NUMBERS = True               # add the threshold-vs-recent-results block (needs INCLUDE_CONTEXT)

MAX_STEPS = -1                   # -1 = one full pass over the training questions; 30 = quick check
LEARNING_RATE = 1e-5
LORA_RANK = 32
SCALE_REWARDS = "group"          # "group" = TRL default (divide by the group's std); "none" = only subtract the group mean
NUM_GENERATIONS = 4              # answers per question, compared against each other
PER_DEVICE_BATCH = 4             # answers per forward pass (lower to 2 and raise GRAD_ACCUM to 8 if out of memory)
GRAD_ACCUM = 4                   # 4 x 4 = 16 answers per step = 4 questions per step
MAX_COMPLETION = 256
SAVE_STEPS = 25                  # checkpoint every 25 steps (Kaggle sessions end after 12 hours)
RESUME_FROM = None               # "auto" = continue from a previous version added as an input

EVAL_BASE = True                 # also score the untrained model now, for a paired comparison
EVAL_SAMPLES = 4
EVAL_TEMPERATURE = 0.7
WEATHER_EVENTS = 60
SEED = 3407

REPO_URL = "https://github.com/anisrikanth/calibrated-forecasting-rl.git"
OUT_DIR = "/kaggle/working"
OUTPUT_DIR = f"{OUT_DIR}/outputs_{RUN_NAME}"
LORA_DIR = f"{OUT_DIR}/lora_{RUN_NAME}"'''),
        md("## 2. Check the GPU"),
        code(GPU_CHECK),
        md("## 3. Install Unsloth, vLLM and TRL (5–10 minutes)"),
        code(INSTALL),
        md("## 4. Get the project code (env/ is tested in the repo)"),
        code(CLONE_REPO + '''
from env.evaluation import (load_markets, pick_eval_sets, render_prompt, collect_answers, add_forecast_columns,
                            score_table, by_series_table, consistency_table, add_shift_matched,
                            paired_bootstrap_table)
from env.training import build_train_records, MonitoredScoreReward
from env.rewards import trl_format_reward
from env import metrics as M'''),
        md("## 5. Load the data"),
        code('''import glob
import numpy as np
import pandas as pd

paths = glob.glob("/kaggle/input/**/markets.csv", recursive=True)
assert paths, "No markets.csv found. Add your dataset: Add Input > Your Work > Datasets."
df = load_markets(paths[0])
eval_df = pick_eval_sets(df, weather_events=WEATHER_EVENTS, seed=SEED)
print(df["split"].value_counts().to_dict())
print(eval_df.groupby("eval_set").size().to_dict())'''),
        md("## 6. Load the model and add LoRA"),
        code('''from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 1536,
    load_in_4bit = False,            # 16-bit LoRA, as in the Phase 0 smoke test
    fast_inference = True,           # vLLM samples the answers during training
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = 0.85,   # lower to 0.7 if out of memory
)
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_RANK,
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)'''),
        md("""## 7. Training data

Training questions only (econ, 2024–2025). Each prompt is rendered exactly as in evaluation, including the question's own date in Llama's header."""),
        code('''from datasets import Dataset

records = build_train_records(df, tokenizer, include_context=INCLUDE_CONTEXT, seed=SEED, key_numbers=KEY_NUMBERS)
train_ds = Dataset.from_list(records)
prompt_lengths = [len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]) for r in records]
max_prompt_length = max(prompt_lengths) + 8
print(f"{len(records)} training questions | prompt tokens: median {int(np.median(prompt_lengths))}, max {max(prompt_lengths)}")
print(f"Yes-rate: {np.mean([r['outcome'] for r in records]):.3f}")
print(records[0]["prompt"][-900:])'''),
        md("""## 8. Train

Two rewards are added: **format** (+0.5 for a readable `Probability: 0.XX`, −1.0 otherwise) and **score** (Brier or log). The score reward also records every stated probability and prints recent stats every 10 calls."""),
        code('''import glob as _glob, os, shutil, time
from trl import GRPOConfig, GRPOTrainer

monitor = MonitoredScoreReward(REWARD, window=256, print_every=10)

args = GRPOConfig(
    learning_rate = LEARNING_RATE,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = PER_DEVICE_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_generations = NUM_GENERATIONS,
    scale_rewards = SCALE_REWARDS,
    max_prompt_length = max_prompt_length,
    max_completion_length = MAX_COMPLETION,
    num_train_epochs = 1,
    max_steps = MAX_STEPS,
    save_steps = SAVE_STEPS,
    max_grad_norm = 1.0,
    report_to = "none",
    output_dir = OUTPUT_DIR,
    seed = SEED,
)

resume = False
if RESUME_FROM == "auto":
    found = _glob.glob("/kaggle/input/**/checkpoint-*", recursive=True)
    assert found, "No checkpoints in /kaggle/input: add the previous version as an input."
    RESUME_FROM = os.path.dirname(max(found, key=lambda p: int(p.rsplit("-", 1)[-1])))
if RESUME_FROM:
    shutil.copytree(RESUME_FROM, OUTPUT_DIR, dirs_exist_ok=True)
    print("Resuming from", RESUME_FROM)
    resume = True

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [trl_format_reward, monitor],
    args = args,
    train_dataset = train_ds,
)
start = time.time()
train_output = trainer.train(resume_from_checkpoint = resume)
TRAIN_MINUTES = (time.time() - start) / 60
print(f"Training took {TRAIN_MINUTES:.1f} min for {trainer.state.global_step} steps")'''),
        md("Training log (the live table doesn't appear in saved-version logs, so here it is as text):"),
        code('''log = pd.DataFrame([h for h in trainer.state.log_history if "reward" in h])
cols = [c for c in log.columns if c in ("step", "reward", "reward_std", "kl", "completions/mean_length",
                                        "completion_length", "rewards/trl_format_reward/mean",
                                        f"rewards/{monitor.__name__}/mean")]
step_every = max(1, len(log) // 25)
print(log[cols].iloc[::step_every].round(4).to_string(index=False))
monitor.history().to_csv(f"{OUT_DIR}/train_answers_{RUN_NAME}.csv", index=False)
log.to_csv(f"{OUT_DIR}/train_log_{RUN_NAME}.csv", index=False)
print("\\nFirst vs last training answers (512 each, or half the run if shorter):")
print(monitor.first_last().to_string())'''),
        md("## 9. Save the adapter"),
        code('''from safetensors import safe_open

model.save_lora(LORA_DIR)
with safe_open(f"{LORA_DIR}/adapter_model.safetensors", framework="pt") as f:
    nonzero = sum(int((f.get_tensor(k) != 0).any()) for k in f.keys())
print(f"Saved LoRA to {LORA_DIR} ({nonzero} non-zero tensors)")'''),
        md("""## 10. Evaluate: untrained vs trained

Same questions, prompts, sampling and scoring as the Phase 2 baselines (4 answers at temperature 0.7; mean4 = their average; unreadable = 0.5)."""),
        code('''from vllm import SamplingParams

eval_rows = eval_df.to_dict("records")
eval_prompts = [render_prompt(tokenizer, r, INCLUDE_CONTEXT, KEY_NUMBERS) for r in eval_rows]
params = SamplingParams(n=EVAL_SAMPLES, temperature=EVAL_TEMPERATURE, top_p=0.95, max_tokens=320, seed=SEED)

def run_eval(lora_request, name):
    t0 = time.time()
    outs = model.fast_generate(eval_prompts, sampling_params=params, lora_request=lora_request)
    ans = collect_answers(eval_rows, outs, name)
    print(f"{name}: {len(ans)} answers in {(time.time() - t0) / 60:.1f} min | readable {ans['prob'].notna().mean():.3f} "
          f"| median tokens {ans['n_tokens'].median():.0f}")
    return ans

answers = [run_eval(model.load_lora(LORA_DIR), "trained")]
if EVAL_BASE:
    answers.append(run_eval(None, "base"))
answers = pd.concat(answers, ignore_index=True)
answers.to_csv(f"{OUT_DIR}/eval_answers_{RUN_NAME}.csv", index=False)

for name in answers["condition"].unique():
    eval_df = add_forecast_columns(eval_df, answers, name, name)
model_cols = [c for c in ("base_single", "base_mean4", "trained_single", "trained_mean4") if c in eval_df.columns]
if {"base_mean4", "trained_mean4"} <= set(eval_df.columns):
    eval_df = add_shift_matched(eval_df, "base_mean4", "trained_mean4")   # untrained, moved to the trained average
    model_cols.append("base_mean4_shifted")
methods = ["market", "base_rate", "recent_hits", "always_0.5"] + model_cols
results = score_table(eval_df, methods)
series = by_series_table(eval_df, ["market", "base_rate", "recent_hits"] + [c for c in model_cols if c.endswith("mean4")])
consistency = consistency_table(eval_df, ["market"] + [c for c in model_cols if c.endswith("mean4")])
results.to_csv(f"{OUT_DIR}/eval_metrics_{RUN_NAME}.csv", index=False)
results[["eval_set", "method", "n", "brier", "log_loss", "ece", "auc", "gap", "mean_p", "resolution"]]'''),
        code('''for name in answers["condition"].unique():
    ex = answers[(answers["condition"] == name) & (answers["sample"] == 0)].sample(1, random_state=SEED).iloc[0]
    q = eval_df.set_index("market_ticker").loc[ex["market_ticker"]]
    print(f"===== {name} | {ex['market_ticker']} | outcome {q['outcome']} | market {q['market_prob']}")
    print(ex["text"][:1200], "\\n")'''),
        md("""## 11. Untrained vs trained, with 95% intervals

Resamples whole events (a weather day, or one release's ladder of thresholds), 4,000 draws, so the interval reflects that questions in one event move together. **AUC and gap** measure telling yes from no and are unaffected by moving every forecast up or down; `base_mean4_shifted` above is the untrained model moved to the trained model's average, so any Brier gain beyond it is not just a shift."""),
        code('''if {"base_mean4", "trained_mean4"} <= set(eval_df.columns):
    t0 = time.time()
    boot = paired_bootstrap_table(eval_df, "base_mean4", "trained_mean4", n_boot=4000, seed=0)
    boot.to_csv(f"{OUT_DIR}/paired_bootstrap_{RUN_NAME}.csv", index=False)
    print(f"({(time.time() - t0) / 60:.1f} min)")
    print(boot.round(4).to_string(index=False))
else:
    boot = None
    print("Set EVAL_BASE = True for the paired comparison.")'''),
        md("## 12. Reliability diagrams"),
        code('''import matplotlib.pyplot as plt

scored = eval_df[eval_df["market_prob"].notna()]
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, (eval_set, g) in zip(axes, scored.groupby("eval_set")):
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--", label="perfect calibration")
    for m in ["market"] + [c for c in model_cols if c.endswith("mean4")]:
        t = M.reliability_table(g[m], g["outcome"])
        ax.plot(t["mean_forecast"], t["frac_yes"], marker="o", label=m)
    ax.set_title(f"{eval_set} ({len(g)} questions)")
    ax.set_xlabel("Forecast probability (bin average)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0].set_ylabel("Share that resolved yes")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle(f"{RUN_NAME}: untrained vs trained vs market")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/reliability_{RUN_NAME}.png", dpi=150)
plt.show()'''),
        md("## 13. Summary to paste back"),
        code('''print("=" * 25, "COPY FROM HERE", "=" * 25)
print(f"Run {RUN_NAME} | {MODEL_NAME} | repo {COMMIT} | reward {REWARD} | context {INCLUDE_CONTEXT} | key numbers {KEY_NUMBERS}")
print(f"Steps {trainer.state.global_step} | lr {LEARNING_RATE} | lora r {LORA_RANK} | scale_rewards {SCALE_REWARDS} | {NUM_GENERATIONS} answers/question | "
      f"{PER_DEVICE_BATCH * GRAD_ACCUM // NUM_GENERATIONS} questions/step | {TRAIN_MINUTES:.0f} min training")
print()
print("Training answers, first vs last (512 each, or half the run):")
print(monitor.first_last().to_string())
print()
print(log[cols].iloc[::max(1, len(log) // 10)].round(4).to_string(index=False))
print()
for name in answers["condition"].unique():
    a = answers[answers["condition"] == name]
    print(f"Eval {name}: readable {a['prob'].notna().mean():.3f} | median tokens {a['n_tokens'].median():.0f}")
print(results[["eval_set", "method", "n", "brier", "log_loss", "ece", "auc", "gap", "mean_p"]].to_string(index=False))
print()
if boot is not None:
    print(boot.round(4).to_string(index=False))
    print()
print(series.to_string())
print()
print(consistency.to_string(index=False))
print("=" * 25, "COPY TO HERE", "=" * 27)'''),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nbf.validate(nb)
    return nb


if __name__ == "__main__":
    for name, build in (("phase2_baselines.ipynb", baselines), ("phase3_grpo_train.ipynb", training)):
        out = HERE / name
        nbf.write(build(), out)
        print("wrote", out)
