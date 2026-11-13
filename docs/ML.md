# ML agent (Agent 5): cross-sectional ML ladder with leakage-proof validation

Code: `engine/models/{features,ml,registry,monitor}.py`, tests `tests/test_ml.py`. Run: `envs/core/bin/python -m engine.models.ml`
(writes `artifacts/ml/<timestamp>/{results.json,report.md}`; scratch Store `artifacts/qae_scratch_ml/`, model artifacts `artifacts/models/`; refuses to use `artifacts/qae`).

## Prior and stance
The project already found no out-of-sample edge in OHLCV-only directional models (incl. a pretrained foundation model). This agent is rigorous
infrastructure plus an honest answer. Each model has ONE pre-declared configuration (`MODEL_PARAMS`); nothing was tuned on results; every
model x seed run is a counted trial.

## Method
- **Features** (price only, strictly causal): returns over 5/21/63/126/252d, skip-month momentum (12-1, 6-1), realised vol 21/63d, vol-of-vol, drawdown from 252d high;
  models see the per-date cross-sectional z-score and pct-rank of each (22 inputs), not raw levels.
- **Target**: 21-trading-day forward return minus cross-sectional mean, sampled at month-ends. Month-end spacing is 20-23 trading days, so consecutive label horizons
  overlap by up to ~2 days: "no overlap" is NOT assumed, labels are purged.
- **Walk-forward**: expanding window, refit annually (pre-declared), min 48 months of training; training rows kept only if `label_end_pos < first_test_pos - 5` (purge + 5-day embargo).
  First out-of-sample prediction 2011-01-31. Judged on the validation window only (2014-12-31..2019-12-31 per `V.Criteria().splits`, embargo applied by `Splits.slice`).
- **Ladder**: EW universe (reference; active return 0 by construction, not a trial); plain 12-1 momentum rank; Ridge (alpha 100); HistGradientBoosting (depth 3, 80 iters, lr .05, min leaf 60, L2 1, max_features .5);
  MLP (16 hidden, L2 .1). HGB and MLP run with 5 seeds (headline = seed 0, declared in advance); all 12 runs are trials.
- **Decision metric**: top-tercile long-only portfolio (same weighting/1-day lag/5 bps cost logic as `alpha.run_xs`; a test checks it matches `run_xs` for momentum), judged on ACTIVE return
  vs the equal-weight universe, DSR (`V.deflated_sharpe`) over all trials in the Store ledger (`family='ml_xs'`) plus the batch, 3x cost stress, `V.risk_gate`. Rank-IC is reported as a diagnostic only.
- **Protected test**: `guard_dev` raises `TestWindowAccess` for any date after `val_end`; `dev_prices` is the only entry for prices; no function records or evaluates test results.
  A `VALIDATION_PASS` means "pending the lead's single protected look"; this module never produces CANDIDATE.
- **Registry** (`ml_models`, `ml_status_history` in the Store DB): same status machine as Store (nothing above CANDIDATE, terminal states frozen), idempotent registration, joblib artifacts.
- **Monitor**: PSI (train vs recent), rolling rank-IC/relative-error degradation, 5-seed instability, and `retrain_trigger` with fixed documented thresholds (see `monitor.py` docstring). Recommend-only.

## Leakage tests (synthetic fixtures, software tests only)
shuffled training labels kill a planted signal; perturbing prices after T leaves features and predictions at/before T bit-identical; no training label horizon overlaps the evaluation fold
(and a naive un-purged split DOES leak, proving the purge matters); test-window guard raises; pure-noise universe gives no VALIDATION_PASS/CANDIDATE and consumes no test look;
a planted momentum-like signal IS recovered by Ridge/HGB/MLP; registry status machine + dedup; PSI/trigger on a constructed shift.

## Results (REAL cached ETF data, validation window, 12 trials)
| rung | active %/yr | active Sharpe | DSR | 3x-cost active | rank-IC (diag) |
|---|---|---|---|---|---|
| 12-1 momentum rank | -1.49 | -0.21 | 0.09 | negative | +0.011 |
| Ridge | -0.46 | -0.07 | 0.15 | negative | +0.047 |
| HGB seed0 (seeds 0-4) | +2.40 (+0.9..+2.8) | +0.43 (+0.16..+0.51) | 0.53 (0.30..0.59) | positive in 4/5 seeds | +0.078 |
| MLP seed0 (seeds 0-4) | -0.35 (-0.4..+0.9) | -0.07 (-0.08..+0.17) | 0.15 (0.14..0.31) | negative | +0.012 |

Seed instability (std of active Sharpe over 5 seeds): HGB 0.128, MLP 0.101 (both under the 0.30 limit). Every run VALIDATION_FAIL (DSR < 0.90; most also fail the 12x turnover limit, 11.7-14.3x).
Shuffled-label controls on real data scatter around 0 (ridge [0.08,-1.14,1.0], hgb [-0.13,0.39,-0.64] active Sharpe), which also shows how wide the noise band is at 5 years of validation.
**Verdict: NULL.** HGB's +2%/yr active is the best run but its DSR is ~0.5 after counting 12 trials, its tracking error is ~5.6%, and it is within the noise band shown by the shuffled controls. Not a finding; no further tuning.

## Caveats
- Validation is 5 years (~1,250 trading days): an active Sharpe has a standard error near 0.45, so only large effects can pass DSR. A null here is "no detectable edge", not "proved zero".
- Survivorship (ETFs that exist today) and Yahoo adjusted prices; costs are a flat 5 bps on turnover.
- Drift monitor demonstration: train (<=2014) vs the last 12 validation months flags large PSI on 5 features and 91% of features above 0.10. With only 12 monthly dates (highly autocorrelated, cross-sectionally dependent) PSI is noisy and over-triggers; treat "recommend retrain" as a prompt to re-run the evaluation, not as evidence of decay. Rank-IC "drop" is similarly a diagnostic on a small window.
- Pickled joblib artifacts must only be loaded from this registry's own output.
- Not done: the protected-test look (lead), forward/shadow evaluation, hyperparameter search (deliberately), feature sets beyond OHLC-derived prices, scheduler/CLI wiring.
