# Portfolio construction (Agent 9)

Code: `engine/agents/portfolio.py`, `adapters/skfolio_adapter.py`, `adapters/workers/skfolio_worker.py`, tests `tests/test_portfolio.py`.
Run: `envs/core/bin/python -m engine.agents.portfolio [--max-weight 0.4] [--cost-bps 5] [--window DAYS] [--no-skfolio]` -> `artifacts/portfolio/<ts>/{results.json,report.md}`.

**Everything here combines RESEARCH SLEEVES, not promoted strategies.** Most strategy sleeves are REJECTED in the registry. The agent never touches the registry and never sets any status.

## Method
- Sleeves: dict of daily net return Series (optionally their asset-weight histories). Demo sleeves: trend-SPY, six xs_factor specs from `alpha.candidate_specs()` (`run_xs`), buy&hold SPY/TLT/GLD/DBC, all from the cached ETF24 dataset (`offline=True`).
- Allocators: equal-weight (baseline), inverse-vol, risk-parity (own SLSQP, equal risk contributions), min-variance (own SLSQP, `shrunk_cov` delta 0.1), skfolio HRP / min-variance / mean-risk (max-Sharpe-ratio objective, empirical moments) through the isolated `envs/portfolio` venv. Constraints: long-only, sum 1, per-sleeve cap (default 0.40; infeasible caps raise; verified every rebalance in tests).
- Walk-forward: monthly rebalance on the first trading day of the month; weights fit on rows strictly before that day (lag >= 1 enforced), expanding window (or rolling via `window`), >= 252 fit days. Weights drift with returns between rebalances; cost = bps x |target - drifted weights| (default 5 bps; sleeve-internal costs are already in sleeve returns).
- Data window: train + validation of `V.Criteria().splits` only. `assert_no_test` / `dev_window` raise `ProtectedTestAccess` for any test-window date. No protected-test look is consumed; no store is used.
- Statistics: net return, Sharpe, Sortino, Calmar, maxDD (`V.window_metrics`), diversification ratio, effective number of bets (Meucci PCA entropy), average and weight-averaged pairwise sleeve correlation, one-way turnover/yr, pre-trade drift, cost drag, Sharpe at 3x costs.
- Drift monitor (`drift_monitor`): at every rebalance compares the last 63 days' correlation matrix to the fitting window (flag if mean |delta corr| > 0.20) and new target vs previous target (flag if half-L1 > 0.25). Counts are reported per method.
- Risk gate: `V.risk_gate` on every portfolio, using look-through asset weights when sleeve weight histories are supplied (so sleeve-internal turnover counts against the 12x limit). Violations are listed, and a violation disqualifies a challenger.
- Decision vs equal-weight (PRE-SET rule, restored after review): challenger is BETTER only if, after costs, the RAW active return vs EW is > 0 at base and 3x costs, the active-return DSR (`V.deflated_sharpe`, trials = all methods compared) >= 0.90, net Sharpe >= EW + 0.10, and no risk-gate violation. A vol-matched active series is reported as a DIAGNOSTIC only and cannot create a winner.

## Results (real cached data, 20261111T112542)
| method | net ret | Sharpe | Sortino | Calmar | maxDD | DivRatio | ENB | avg corr | turnover/yr | drift flags | DSR | risk gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| equal_weight | +128.8% | 0.73 | 1.02 | 0.25 | -26.4% | 1.50 | 1.40 | 0.41 | 0.32x | 63/153 | 0.97 | PASS |
| inverse_vol | +134.8% | 0.84 | 1.19 | 0.32 | -22.0% | 1.49 | 1.66 | 0.44 | 0.30x | 63/153 | 0.99 | PASS |
| risk_parity | +127.0% | 0.85 | 1.21 | 0.37 | -18.1% | 1.66 | 2.14 | 0.35 | 0.40x | 63/153 | 0.99 | PASS |
| min_variance | +131.5% | 1.08 | 1.56 | 0.49 | -14.0% | 1.67 | 5.37 | 0.33 | 0.40x | 64/153 | 1.00 | PASS |
| skfolio_hrp | +121.4% | 0.94 | 1.34 | 0.37 | -17.6% | 1.67 | 3.38 | 0.36 | 0.77x | 63/153 | 1.00 | PASS |
| skfolio_min_variance | +130.8% | 1.07 | 1.54 | 0.48 | -14.0% | 1.70 | 5.37 | 0.30 | 0.52x | 64/153 | 1.00 | PASS |
| skfolio_mean_risk | +129.3% | 0.98 | 1.40 | 0.37 | -18.0% | 1.82 | 5.03 | 0.17 | 0.74x | 64/153 | 1.00 | PASS |
## Decision vs equal-weight baseline

**inverse_vol beats equal-weight after costs and DSR**

**CORRECTION (2026-11-11, independent review):** an earlier version of this document reported `inverse_vol` as BETTER (DSR 0.97). That came from a decision rule (vol-matching) added about a minute after the pre-set raw-return rule produced "no winner", with the two variants not counted as separate trials. Counting both, DSR falls to ~0.81 (< 0.90). The retracted claim is withdrawn. Result under the pre-set rule (artifacts/portfolio/20261111T143018):
- inverse_vol: NOT BETTER (raw active +0.10%/yr, active DSR 0.48, Sharpe margin +0.12) -- DSR 0.48 < 0.9
- risk_parity, min_variance, skfolio_hrp, skfolio_min_variance, skfolio_mean_risk: NOT BETTER (raw active -0.17% to -0.48%/yr, not positive at 3x costs, DSR 0.28-0.37)
- Honest summary: lower-vol allocators (min-variance, skfolio) cut max drawdown from -26% to about -14% with similar total return; that is risk reduction, not alpha. Equal-weight is retained.
Window 2007-04-02..2019-12-31 (3210 OOS days), 11 sleeves, cap 40%, 5 bps, 7 methods (6 challengers) counted as trials.

## Caveats (read these)
- The only challenger passing the rule, inverse-vol, is a near-clone of equal-weight (unscaled active +0.10%/yr, Sharpe margin +0.12 vs the 0.10 bar); this is marginal and sensitive to the vol-matching choice. Min-variance/HRP have much larger Sharpe margins (+0.2 to +0.36, lower drawdown) but do NOT clear DSR 0.90 with 6 trials, and all of them have lower raw return than equal-weight (-0.2 to -0.5%/yr). Under a raw-return active test (no vol matching) none beat equal-weight. This is a risk-reduction result, not an alpha result.
- Sleeves overlap heavily (equity beta; the xs and trend sleeves are mostly long equities), so effective bets stay low for naive methods.
- The sleeves themselves are mostly rejected strategies; the xs specs were picked by name from a grid partly examined in earlier sessions, and sleeve histories are only 2007-2019 in this study. Nothing about the protected test window is known.
- Only one window convention (expanding, monthly, lag 1) and one cap were run for the headline; trying others adds trials that should be fed through `extra_trial_sharpes_daily`.
- skfolio min-variance uses the plain empirical covariance; own min-variance uses shrunk covariance. The cross-check agrees to 5e-3 with delta=0 and to 0.08 with delta=0.1 (documented in the test).
- mean_risk uses sample means: noisy; included only so the DSR trial count is honest.
- Diversification ratio / ENB / correlations are computed ex-post on the OOS period with time-averaged weights.
- Look-through risk gate is only as good as the supplied sleeve weight histories.