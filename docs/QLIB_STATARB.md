# qlib models + arbitragelab pairs in the validation pipeline (Task 7)

Both families use the frozen `Criteria` (digest `fc813df6d27810db` stored on every experiment), the store's experiment dedup (spec_hash + criteria_hash + data_sha), DSR with up-front trial counting,
`V.risk_gate`, 3x cost stress, and the dev window only. Neither calls `Store.record_test_access` (tests: runtime monkeypatch + source scan; mutation-checked). Scratch store: `artifacts/qae_scratch_families`
(refuses `artifacts/qae`). Real data: cached 24-ETF Yahoo prices (offline), truncated at 2019-12-31 (`common.dev_prices`). No protected-test look was taken. A pass would only mean "pending the lead's single look".

## Family `qlib_model` (`engine/families/qlib_model.py`, `adapters/qlib_models_adapter.py`, worker in envs/ml)
- Features: 19 close-only Alpha158-like qlib expressions (ROC5/21/63/126/252, MOM12_1, MOM6_1, MA5/20/60/120, STD21/63, MAX60/MIN60/MAX252, RSV20, QTLU60, QTLD60), fixed in `FEATURE_EXPRS`
  (hash `f2005a7d421b079b`); label `Ref($close,-21)/$close-1`; month-end samples; per-date cross-sectional z-score (clip +-5) and demeaned label.
- Models: qlib `LinearModel` (ridge, alpha 100) and qlib `LGBModel` (lightgbm 4.7.0, in envs/ml) via a minimal DatasetH shim (qlib's mlflow recorder is stubbed out because it needs a tracking dir; metrics unused).
  LGB early stopping uses the last 15% of dates INSIDE the purged training rows, never the validation window.
- Walk-forward: annual refit, purge (label horizon ends > 5 positions before the first test date) + embargo; first OOS prediction 2011-01-31; judged on validation 2015-01-06..2019-12-31.
- Portfolio/judgement: same construction as `engine/models/ml.py` (`run_scores`: long-only equal-weight top tercile, 1-day lag, 5 bps costs on turnover, 3x stress) vs equal-weight universe; ACTIVE return and DSR decide, rank-IC is not computed.
- PRE-REGISTERED trial list (code constant `TRIALS`, hash pinned in `TRIAL_LIST_HASH = ffb83f441e3727b9` before any real run, asserted at import and in `run_family`): 7 trials = ridge x1 + LGB config A x 3 seeds + LGB config B x 3 seeds.
  DSR of every trial deflates by all 7 (two-pass: compute all, then record) plus any prior ledger rows of the family. Shuffled-label controls are not trials.
- Guards: `common.guard_dev` raises `TestWindowAccess` on any bar after 2019-12-31 (core, before any env launch), the adapter also rejects fold dates past val_end, and the worker re-checks independently.

Result (real data, scratch store, 7 experiments recorded; run reproduced bit-identically twice): **NULL, no trial passed validation.**

| trial | val active %/yr | active Sharpe | DSR (7 trials) | active @3x cost (daily bp) | turnover x/yr | max DD |
|---|---|---|---|---|---|---|
| qlib-linear-s0 | +2.35 | +0.42 | 0.62 | +0.41 | 13.2 | -12% |
| qlib-lgb_a-s0/s1/s2 | -0.73 / -0.20 / -0.72 | -0.13 / -0.04 / -0.12 | 0.17 / 0.23 / 0.18 | -0.76 / -0.55 / -0.81 | 11.8 / 11.7 / 13.3 | -16..-19% |
| qlib-lgb_b-s0/s1/s2 | -1.33 / -0.00 / -0.48 | -0.21 / -0.00 / -0.08 | 0.13 / 0.26 / 0.21 | -0.99 / -0.50 / -0.69 | 11.7 / 12.6 / 12.5 | -16..-18% |

The best trial (ridge) fails DSR (0.62 < 0.90) and the frozen turnover limit (13.2x > 12x); every LGB trial has negative active return. Shuffled-label controls (10 seeds each, labels permuted within date,
judged by the same rule): ridge mean |active Sharpe| 0.26, max DSR 0.46; LGB-A mean |active Sharpe| 0.33, max DSR 0.34; 0/20 VALIDATION_PASS. Control run is small (n=10 per model): it shows no gross pipeline false-positive, it does not measure a rate.

## Family `statarb_pairs` (`engine/families/statarb_pairs.py`, `adapters/statarb_adapter.py`, worker in envs/statarb, py3.11)
Pre-registered (module constants, `PREREG_HASH = 82fb9105bd2a4957` pinned before any real run): all 276 = C(24,2) pairs (ETF24 order, one direction per pair); formation <= 2014-12-31; trading window = validation;
spread = lnA - beta lnB with beta fixed from formation; rolling 60d z; enter |z|>=2, exit |z|<=0.5, stop |z|>=4 (no re-entry until |z|<2); 1-day lag; 5 bps/turnover + 50 bps/yr short borrow;
pair gross exactly 1 (`wA=1/(1+|beta|)`), decision unit = equal-weight (1/N) portfolio of survivors so gross <= `max_leverage`.
Test: arbitragelab `EngleGrangerPortfolio` (hedge vector, ADF statistic; arbitragelab exposes critical values, not p-values; it was importable, no fallback was used on the real run) + statsmodels MacKinnon p-value (`coint`, trend c, maxlag 10, autolag aic).
**Multiple testing:** Benjamini-Hochberg q=0.05 over all 276 p-values is the survivor rule; Bonferroni and raw p<0.05 are reported. **DSR** is deflated by N=276 (every pair tested): trial-Sharpe variance = validation Sharpes of all 276 pairs under identical rules (used only for the variance, never to select).

Result (real data): **NULL. 0 of 276 pairs survive BH-FDR** (min p = 0.0022, XLE/XLB; Bonferroni 0; raw p<0.05: 8 vs 13.8 expected by chance under the null, i.e. no more cointegration than noise). The validation backtest is therefore not run for any pair; the experiment is recorded as VALIDATION_FAIL "NULL".
Top raw p-values: XLE/XLB 0.0022, XLE/XLI 0.011, XLE/XLY 0.024, XLU/XLB 0.028, XLE/XLK 0.029.
**Noise control** (100 independent random-walk universes of 24 assets, real calendar, full pipeline, not recorded): mean raw p<0.05 pairs 14.3 (expected 13.8); sims with >=1 BH survivor 7/100 (nominal q=0.05; 7/100 is within binomial noise of 5%, but the control does not show the rate is below 5%);
sims with a Bonferroni survivor 7; sims reaching VALIDATION_PASS 0/100. Control p-values used the statsmodels engine (arbitragelab's EngleGranger fit adds ~0.15 s/pair with no change to the p-value); disclosed deviation.

## Honest caveats
- Pairs control uses statsmodels-only p-values; the real run used arbitragelab for hedge vectors. Power is shown only on SYNTHETIC planted pairs (tests).
- The qlib family's controls are few (10/model); ridge's turnover (13.2x) breaches the frozen 12x limit, an inherent trait of monthly top-tercile rebalancing at this universe size (same as ml_xs).
- The scratch store was deleted once by mistake mid-session (after the first qlib run) and the qlib family was re-run; results are identical (verified), the first artifact dir `artifacts/families/qlib_model/20261111-164150` is kept.
- The first pairs run (arbitragelab) was killed and relaunched after a BLAS thread-oversubscription slowdown (worker now pins 1 thread) and after adding an `engine` switch for noise controls; no decision rule changed after seeing any result.
- Test-fixture fixes found by review runs: `test_positions_state_machine_enter_exit_stop` (fixture spikes were so large the stop fired the day after entry; code was right, the test expectation was wrong) and `test_dsr_deflated_by_all_pairs_...` (the test sliced validation as 2015-01-06 instead of `Splits.slice`, which starts at train_end+embargo; test wrong, code right).

## Run
```
envs/core/bin/python -m engine.families.qlib_model            # ~2-4 min, writes artifacts/families/qlib_model/<ts>/results.json
envs/core/bin/python -m engine.families.statarb_pairs --sims 100   # formation tests ~1 min + noise controls (long)
envs/core/bin/python artifacts/families/mutation_check.py     # mutation checks
envs/core/bin/python -m pytest tests/test_qlib_models.py tests/test_statarb.py [-m slow]
```
