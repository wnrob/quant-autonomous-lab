# A/B test of proposal sources on the xs_factor family

Run: `envs/core/bin/python -m engine.research.ab_proposers` (results: `artifacts/ab/20261111-112437/results.json`, scratch stores: `artifacts/qae_scratch_ab/20261111-112437/`).
Model: Ollama `qwen2.5:14b`, loopback only. Data: cached ETF24 (`load_prices(..., offline=True)`, sha 476483a64551).

## Method
- Arms (same family, same frozen `Criteria`, same evaluator `evaluate_xs`): **RULE** = first N of the complexity-ordered grid (deterministic);
  **LLM** = `engine/agents/llm_xs.py` (strict schema, lessons + tested specs in prompt, temperature 0.7, seed = round seed, 4 proposals/call, <= 8 calls/round);
  **RANDOM** = seeded uniform draw (no replacement) from the same validated space (kind x lookback 5..252 x skip 0..63 x vol_adj).
- N = 12 experiments per arm per round, R = 5 rounds. Each (arm, round) has its own scratch store and an explicit research `Budget` (production 20/day budget untouched).
- **Validation only** (`allow_test=False`). Asserted every round: `store.test_looks() == 0`, no `test` metrics in any experiment. Total test looks = 0.
- Common-trial-count DSR: `PSR(active series, SR*)` with SR* = expected max Sharpe of N_common = 12 trials, trial-Sharpe variance pooled over all 120 distinct specs any arm evaluated. (The in-store sequential DSR, which counts only that store's trials, decides the store pass/fail.)
- Stats: exact permutation test on arm labels over the 5 round values (two-sided) + bootstrap 95% CI of the difference in means. Plus a 300-round random-spec reference distribution of best-of-12 validation active return.
- Diversity: mean pairwise spec distance (kind mismatch, log-lookback, skip, vol_adj; range 0..1).

## Results (means over 5 rounds)
| metric | RULE | LLM | RANDOM |
|--|--|--|--|
| best validation active return /yr | +1.05% | +1.97% | +2.08% |
| best common-N DSR | 0.255 | 0.367 | 0.373 |
| mean active return of all 12 | -2.12% | -1.08% | -1.16% |
| fraction passing validation criteria (store DSR) | 0 | 0 | 0 |
| fraction passing with common-N DSR | 0 | 0 | 0 |
| invalid-or-duplicate proposal rate | 0 | 0 (60/60 valid, 0 dup) | 0 (by construction) |
| diversity (mean pairwise dist) | 0.27 | 0.43 | 0.44 |

Differences (permutation p, bootstrap 95% CI): best active LLM-RULE +0.92pp (p=0.127, CI +0.05..+1.7pp); LLM-RANDOM -0.11pp (p=0.905, CI -1.1..+0.8pp); mean active LLM-RANDOM +0.08pp (p=0.75, CI -0.37..+0.51pp); best DSR LLM-RANDOM -0.006 (p=0.96).
Against the 300-round random reference (best-of-12 mean +1.57%, 5-95% band +0.05%..+3.3%), the arm means sit at percentile RULE 34, LLM 67, RANDOM 70.

## Verdict
**LLM is not distinguishable from random-spec sampling** on any metric (all p > 0.7). LLM > RULE on mean active return (p=0.008) and diversity, but this is an artifact of what the 12 rule specs are (the first 12 of the grid are momentum/lowvol-heavy; the grid's best three, all reversal L63, sit at positions 5, 13, 18) and RULE has zero between-round variance, so its p-values bottom out at 1/126 by construction and are not independent evidence; on best-of-12 active return LLM-RULE is not significant (p=0.13). **Nobody passed validation: 0/180 evaluated proposals.** The LLM adds no demonstrated value over random sampling; it did produce valid, non-duplicate, diverse proposals, so it is a safe but not a better proposer here.

## Caveats
- R=5 per arm: power is low; only large effects could be detected. "Not distinguishable" is not "equal".
- RULE is deterministic so its 5 rounds are replicates. RULE gets the first 12 of 20 grid candidates (the grid has 20, not 24, candidates).
- The best-of-12 validation active return is selection-biased upward for every arm; none exceeds what random search yields (percentiles above).
- Random draws include lowvol specs with irrelevant skip/vol_adj knobs (identical return streams, distinct hashes); LLM does the same. Space is validated, not behaviour-deduplicated.
- LLM rounds differ via seed (temperature 0.7) and in-round lesson state; stores are not shared across rounds, so no cross-round learning is tested. One LLM call failed in round 1 (counted in `failed_calls`, not in the stats of proposals).
- LLM calls: 15 successful + 1 failed (<= 40 cap), ~1080s of model time. Whole run ~2.7h wall-clock, dominated by the 300-round random reference and the concurrent full test-suite run.
- Single dataset and single (frozen) validation window; no claim about the protected test window.

## Full 20-candidate grid (scratch store, validation only, DSR counts all 20 trials)
See `grid_table` in results.json. 0/20 pass; decision VALIDATION_FAIL for all.

| # | candidate | val active/yr | DSR seq | DSR all-20 | active @3x cost/yr | decision |
|--|--|--|--|--|--|--|
| 1 | xs-momentum-L21-S0 | -5.78% | 0.02 | 0.00 | -7.41% | VALIDATION_FAIL |
| 2 | xs-reversal-L21-S0 | +1.05% | 0.30 | 0.20 | -0.48% | VALIDATION_FAIL |
| 3 | xs-lowvol-L21-S0 | -3.14% | 0.02 | 0.01 | -3.79% | VALIDATION_FAIL |
| 4 | xs-momentum-L63-S0 | -3.81% | 0.01 | 0.01 | -4.79% | VALIDATION_FAIL |
| 5 | xs-reversal-L63-S0 | +0.84% | 0.17 | 0.17 | -0.04% | VALIDATION_FAIL |
| 6 | xs-lowvol-L63-S0 | -2.03% | 0.03 | 0.03 | -2.32% | VALIDATION_FAIL |
| 7 | xs-momentum-L126-S0 | -1.03% | 0.06 | 0.06 | -1.74% | VALIDATION_FAIL |
| 8 | xs-lowvol-L126-S0 | -1.25% | 0.06 | 0.05 | -1.40% | VALIDATION_FAIL |
| 9 | xs-momentum-L252-S0 | -4.04% | 0.01 | 0.01 | -4.52% | VALIDATION_FAIL |
| 10 | xs-lowvol-L252-S0 | -1.00% | 0.07 | 0.06 | -1.10% | VALIDATION_FAIL |
| 11 | xs-reversal-L21-S21 | -2.22% | 0.03 | 0.02 | -3.80% | VALIDATION_FAIL |
| 12 | xs-momentum-L63-S21 | -2.99% | 0.01 | 0.01 | -3.93% | VALIDATION_FAIL |
| 13 | xs-reversal-L63-S21 | +0.98% | 0.18 | 0.18 | +0.11% | VALIDATION_FAIL |
| 14 | xs-momentum-L126-S21 | -0.62% | 0.08 | 0.08 | -1.33% | VALIDATION_FAIL |
| 15 | xs-momentum-L252-S21 | -1.58% | 0.04 | 0.04 | -2.09% | VALIDATION_FAIL |
| 16 | xs-reversal-L21-S21-va | -1.61% | 0.04 | 0.03 | -3.16% | VALIDATION_FAIL |
| 17 | xs-momentum-L63-S21-va | -1.97% | 0.03 | 0.03 | -2.91% | VALIDATION_FAIL |
| 18 | xs-reversal-L63-S21-va | +1.80% | 0.26 | 0.27 | +0.98% | VALIDATION_FAIL |
| 19 | xs-momentum-L126-S21-va | -1.30% | 0.05 | 0.05 | -1.93% | VALIDATION_FAIL |
| 20 | xs-momentum-L252-S21-va | -1.02% | 0.06 | 0.06 | -1.47% | VALIDATION_FAIL |

Common failure reasons: negative active return vs the equal-weight universe (16/20); reversal L21/L63 also fail 3x-cost stress or turnover (L21: 15.3x > 12x); best three (reversal L63 variants, +0.8..+1.8%/yr) fail all-trials DSR 0.17-0.27 < 0.90. Only #13 and #18 survive 3x costs.
