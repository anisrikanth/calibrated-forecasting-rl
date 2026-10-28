# Limitations and data quirks (for the paper's limitations section)

Collected as they're found, with how much each one matters.

## Data

- **Context order follows settlement, not the calendar month.** Recent results are listed in the order their Kalshi markets settled. During the US federal government shutdown in fall 2025, some data releases came out late, so on 50 of the 410 validation econ questions one pair of months appears out of order. On 10 of them (unemployment questions asked in January 2026), the key numbers' "latest result" is October 2025 (4.4%) rather than November 2025 (4.6%). Both values were public before the questions were asked, so this is not leakage. It slightly mislabels "latest" for those 10 questions. Left unchanged so all runs see identical prompts.
- **Kalshi prices are a benchmark, not ground truth.** Low-priced contracts are known to be overpriced (Bürgi, Deng and Whelan 2026), which matters most for the weather brackets.

## Contamination and leakage

- **Evaluation questions postdate the model.** Validation questions are from 2026, after both Llama 3.2's stated knowledge cutoff (December 2023) and its release (September 2024), so pretraining can't have seen them. Some 2024 training questions resolved before the release date, so the model may have seen those outcomes. That affects what it trains on, not how it is evaluated.
- **Our pipeline only shows results settled before each question's time**, enforced with settlement timestamps and checked in the leakage audit.

## Evaluation

- **Development used one validation split** (36 releases). Settings were chosen by looking at it, so final numbers come from the untouched test split, evaluated once.
- **Few releases per series** (3 to 5 in validation), so per-series results are descriptive only.
- **In-training curves are not forward tests.** TRL reshuffles the training questions every pass, so the model can be rewarded on a later release before seeing an earlier one.

## Scope

- One model family (Llama 3.2, 3B and 1B), LoRA rank 32, one learning rate, and free Kaggle T4 GPUs. Larger models and full fine-tuning are out of budget.
- Questions are scheduled numeric releases and daily temperatures, not general forecasting questions.
