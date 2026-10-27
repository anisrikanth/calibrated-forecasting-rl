# Phase 2 baselines

Base models with no fine-tuning, from `notebooks/phase2_baselines.ipynb`. 4 answers per question at temperature 0.7; "mean4" is the average of the readable answers; unreadable answers count as 0.5. All methods are scored on the same questions. Brier score: lower is better; always 0.5 scores 0.25.

## Llama 3.2 3B Instruct (Oct 26, 2026; repo commit c9f9bd9)

Readable probability: 97.6% with context, 98.4% without. Median answer length: about 115 tokens.

| Eval set | Method | n | Brier | Log loss | ECE | Resolution |
| --- | --- | --- | --- | --- | --- | --- |
| val_econ | market | 410 | **0.0615** | 0.1997 | 0.0232 | 0.1882 |
| val_econ | base rate | 410 | 0.2432 | 0.6791 | 0.0423 | 0.0072 |
| val_econ | always 0.5 | 410 | 0.2500 | 0.6931 | 0.0390 | 0.0000 |
| val_econ | model, context, mean4 | 410 | 0.2720 | 0.7481 | 0.1204 | 0.0110 |
| val_econ | model, no context, mean4 | 410 | 0.2531 | 0.6998 | 0.0963 | 0.0082 |
| heldout_weather | market | 360 | **0.1008** | 0.3178 | 0.0550 | 0.0457 |
| heldout_weather | base rate | 360 | 0.1389 | 0.4506 | 0.0000 | 0.0000 |
| heldout_weather | always 0.5 | 360 | 0.2500 | 0.6931 | 0.3333 | 0.0000 |
| heldout_weather | model, context, mean4 | 360 | 0.2861 | 0.7770 | 0.3553 | 0.0042 |
| heldout_weather | model, no context, mean4 | 360 | 0.2655 | 0.7311 | 0.3401 | 0.0035 |

Single-answer scores are worse than mean4 in every case (e.g. val_econ with context: 0.3064).

Consistency (val_econ ladders: 362 adjacent threshold pairs; weather: 60 days of brackets):

| Method | Ladder violation rate | Bracket sum, average distance from 1 |
| --- | --- | --- |
| market | 0.052 | 0.031 |
| model, context | 0.483 | 2.105 |
| model, no context | 0.461 | 2.023 |

### What it shows

- **The base model has no forecasting skill yet.** It is worse than always saying 0.5 on economic questions and twice as bad as the base rate on weather. Its resolution (how well it separates yes from no) is near zero, against 0.19 for the market. Large headroom for RL.
- **Context currently hurts** (0.272 with vs 0.253 without on econ). The model reads the recent values but doesn't turn them into a sensible probability.
- **Its reasoning and its number often disagree.** Example (KXHIGHNY-26AUG13-T85): the model argued the high would likely be above 85°F, then answered `Probability: 0.02`.
- **No internal consistency.** About half of adjacent ladder pairs go the wrong way (random would be 50%), and a weather day's brackets add up to about 3.1 instead of 1: it gives each bracket roughly 0.5.
- **No sign of leakage.** Without context the model captures at most 36% of the market's edge over the base rate in any series (payrolls; most series are negative), far from what memorized outcomes would give. Share of market edge = (base-rate Brier − no-context Brier) ÷ (base-rate Brier − market Brier).

### Implications for Phase 3

- The easiest early gain for RL is hedging toward each series' base rate (Brier about 0.24 with zero resolution). That counts as a reward hack in our plan: progress means **resolution rising**, not just Brier falling.
- Ladder and bracket consistency are never trained directly (each question is scored alone), so whether RL improves them is a free test of whether the model learns coherent beliefs.

## Llama 3.2 1B Instruct (Oct 26, 2026; repo commit 8a0739d)

Readable probability: **58.4% with context**, 97.4% without. Median answer length: 41 tokens with context, **7 without** (it mostly answers "Probability: 0.2" with no reasoning). About 5 times faster than 3B (3.5 vs 16.8 minutes for 3,080 answers with context).

| Eval set | Method | n | Brier | Log loss | ECE | Resolution |
| --- | --- | --- | --- | --- | --- | --- |
| val_econ | model, context, mean4 | 410 | 0.3118 | 0.8979 | 0.2108 | 0.0089 |
| val_econ | model, no context, mean4 | 410 | 0.2645 | 0.7355 | 0.1066 | 0.0030 |
| heldout_weather | model, context, mean4 | 360 | 0.2741 | 0.7810 | 0.3046 | 0.0014 |
| heldout_weather | model, no context, mean4 | 360 | 0.1971 | 0.5853 | 0.2244 | 0.0028 |

(Market, base rate and always-0.5 rows are identical to the 3B table.) Ladder violation rate: 0.326 with context, 0.262 without. Bracket sums: 1.773 and 1.313 from 1.

### What it shows

- **Worse than 3B on economic questions** (0.312 vs 0.272 with context) and **context breaks its output format**: 42% of answers with context have no readable probability, which then count as 0.5.
- **Its better weather score is luck, not skill.** Without context it often answers 0.2, close to the 1-in-6 base rate for temperature brackets; resolution is 0.003. A reminder that Brier score alone can reward a convenient constant; resolution exposes it.
- **No leakage:** without context it captures at most 14% of the market's edge over the base rate in any series.

## Choice of model for RL

**3B** for the main result: it reasons, keeps its format with context (97.6% readable), and has the reasoning-versus-number mismatch that RL should fix. **1B** only for a quick pipeline shakedown, since it is 5 times faster.
