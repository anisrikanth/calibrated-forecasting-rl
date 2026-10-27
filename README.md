# Calibrated Forecasting with RL

Training a small open language model with reinforcement learning (GRPO) to give **calibrated probabilities** on real prediction-market questions, using resolved [Kalshi](https://kalshi.com) markets as ground truth.

## Research questions

1. Does RL with a proper scoring rule (the Brier score) improve calibration on held-out markets, compared with the base model and the market price?
2. Does that calibration transfer to a market category the model never trained on (train on economic releases, test on weather)?
3. What reward hacks appear during training, and what fixes them?

## Setup at a glance

| Piece | Choice |
| --- | --- |
| Models | Llama 3.2 1B Instruct (iteration), Llama 3.2 3B Instruct (final runs) |
| Knowledge cutoff | December 2023, so only markets that closed in 2024 or later are used |
| Training | GRPO with Unsloth + TRL, 16-bit LoRA, on free Kaggle T4 GPUs |
| Reward | 1 − (stated probability − outcome)², plus a small format reward |
| Training markets | Kalshi economic releases (CPI, jobs report, Fed decisions) |
| Held-out markets | Kalshi daily high-temperature markets |
| Benchmark | The market's own price at question time (never shown to the model) |

## Status

- [x] Phase 0: setup and GRPO smoke test on Kaggle (20 steps, plus checkpoint resume)
- [x] Phase 1: Kalshi data pipeline (14,662 markets, 10 series), context snapshots, leakage audit, date-based splits
- [x] Phase 2: forecasting environment, baselines and leak probe (`eval/baselines.md`)
- [ ] Phase 3: GRPO training and reward-hacking log. Pilot runs 1-4 done (`eval/rl_run1.md`, `eval/rl_run2_run3.md`, `eval/rl_run4.md`); 5 entries in `writeup/hacks.md`; next: repeat runs with more seeds
- [ ] Phase 4: generalization tests and ablations
- [ ] Phase 5: write-up

## Repository layout

```
data/        Kalshi pipeline (see data/README.md); raw/ and processed/ are git-ignored
env/         forecasting environment: prompt, parser, reward (Phase 2)
train/       GRPO training configs and scripts (Phase 3)
eval/        metrics, reliability diagrams, results tables
notebooks/   Kaggle notebooks
tests/       unit tests (run with pytest)
writeup/     notes, hacks.md (reward hacks and parser issues found), drafts
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Pull resolved Kalshi markets and build the dataset (no API key needed)
python -m data.pull_kalshi discover   # check which series tickers exist
python -m data.pull_kalshi all        # markets -> prices -> build

pytest                                # offline unit tests
```

## License

Code is MIT-licensed (see `LICENSE`). Notebooks in `notebooks/` are adapted from [Unsloth's notebooks](https://github.com/unslothai/notebooks) and remain under LGPL-3.0. Kalshi market data is subject to Kalshi's terms and is not redistributed here; the pipeline rebuilds it.
