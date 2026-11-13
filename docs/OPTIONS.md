# Options: implied vol, Greeks, py_vollib vs QuantLib (task 3)

## What exists
- `envs/derivs` (Python 3.12.12, **wheels only**, no source builds, QuantLib C++ clone untouched):
  `uv venv --python 3.12 envs/derivs && uv pip install --python envs/derivs/bin/python --only-binary :all: py_vollib QuantLib numpy pandas pyarrow`
  -> py_vollib 1.0.12 (now a shim over vollib 1.0.11; emits a DeprecationWarning), lets-be-rational 1.1.2, QuantLib 1.44,
  numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scipy 1.18.1. py_vollib's pure-python deps installed as wheels, so
  `scripts/sandbox_build.sh` was not needed.
- `adapters/derivs_adapter.py` (core side, no heavy imports): `price_greeks`, `implied_vol`, `american`, `pickle_scan`,
  `qqq_chains_iv`, `cboe_snapshots_iv`. Batches go to the worker through temp files (argv size limit).
- `adapters/workers/derivs_worker.py` (runs in envs/derivs via `run_in_env`).
- `engine/derivs/checks.py` (stdlib-only validation + no-arbitrage bounds, used by BOTH sides),
  `engine/derivs/safe_pickle.py` (two-gate pickle loader), `engine/derivs/crosscheck.py`, `engine/derivs/realdata.py`.

## Conventions (one set everywhere)
| item | convention |
|---|---|
| time | T = calendar days / 365 (ACT/365F). QuantLib is day-granular: its path requires T*365 to be an integer (else it raises); flat `FlatForward` curves with `Actual365Fixed` + `Continuous` compounding, fixed evaluation date, maturity = eval + days, so QuantLib's year fraction is exactly T. |
| rates | r, q continuously compounded, annual. |
| models | `bsm` (spot S, dividend yield q); `black76` (forward F passed as S; implemented in QuantLib as BSM with q = r, which is algebraically Black-76). |
| Greeks units | delta dV/dS (dV/dF for Black-76), gamma, vega per 1.00 vol, theta per year (dV/dt), rho per 1.00 rate. py_vollib natively returns vega/100, rho/100, theta/365: rescaled. Black-76 rho holds F fixed (= -T V): QuantLib `rho()+dividendRho()`. |
| validation | flag exactly `'c'`/`'p'` (no `'C'`, `'call'`, `1`, `True`); S, K, T, sigma > 0, finite; abs(r), abs(q) <= 1; sigma <= 10. Raised in core before any subprocess AND again in the worker. |
| IV | NaN + reason if price is not finite / <= 0 / below the model-free lower bound / at the lower bound (vol not identifiable) / >= the upper bound; solver failures are NaN + the library's error. Never clamped. |
| American | QuantLib CRR binomial (801 steps), labelled extra; returned next to European analytic and European CRR. |

## Cross-check (synthetic fixed grid; tolerances set before the first run)
Grid: S=100; K in {50,80,90,95,100,105,110,120,150}; days {1,7,30,91,182,365,730}; sigma {5%..150%}; r {-1%,0,3%,8%};
q {0,2%,5%}; c/p; BSM + Black-76 -> 12,096 rows. `envs/core/bin/python -m engine.derivs.crosscheck` -> `artifacts/derivs/crosscheck.json`.

| check | tolerance | observed max abs diff |
|---|---|---|
| price | 1e-9 | 4.3e-14 |
| delta | 1e-9 | 3.4e-14 |
| gamma | 1e-9 | 7.1e-15 |
| vega | 1e-7 | 1.7e-13 |
| theta (per year) | 1e-7 | 6.0e-12 |
| rho | 1e-7 | 2.6e-13 |
| put-call parity (each lib) | 1e-10 | 3.2e-14 / 4.3e-14 |
| IV round trip (9,258 identifiable rows) | 1e-7 | 2.2e-10 (vollib) / 1.2e-10 (QuantLib); 0 NaN |
| bound rows (60: above upper / below lower) | NaN + reason | all NaN, all with reasons |

Non-identifiable rows (2,838: time value < 1e-6 or vega < 1e-4) are reported, not hidden: 1,801 `at_lower_bound`, 619
`price_nonpositive` (QuantLib price underflows to 0), 418 still solved. Also tested: an independent stdlib BSM reference
(1e-10), Hull's textbook example, finite-difference vega/theta, American put > European put, American call (q=0) ==
European CRR.

## Real data (descriptive only; protected era)
All local options data is dated 2024-08-30 or later, i.e. inside the protected era. `engine/derivs/realdata.py` refuses
any purpose other than `"descriptive"` (`ProtectedEraUse`), never imports the store, and takes no protected-test look.
Gap report: `artifacts/derivs/data_gap.md|json`. Results: `artifacts/derivs/realdata_iv.json`; per-day/smile CSVs and
parquets in `artifacts/qae_scratch_derivs/`.

**qqq_chains.pkl** (42 calls, 21 monthly expiries, 1,942 daily bars 2024-08-30..2026-08-21). Opcode scan: 23 globals, all
allowlisted; loaded in envs/derivs with `RestrictedUnpickler`. Provenance conventions (r=4.5%, q=0, T=DTE/365, option close
vs underlying close): 1,862 bars solved by both libraries, max |IV diff| 2.5e-13; 38 closes below the no-arbitrage lower
bound (stale / non-synchronous prints) and 42 expiry-day bars -> NaN with reason. ATM IV (DTE 20-45, closest strike):
370 days, median 0.1929 (p10 0.162, p90 0.244), reproducing the provenance file's reported median 0.1929.

**Cboe snapshots** (927 snapshots, 30 session dates 2026-09-21..2026-11-10; one per date selected, closest to 17:00 UTC; 29
dates). 94,641 OTM quotes across 630 expiry fits: both libraries solved every row, max |IV diff| 3.7e-13. Our Black-76 IV
minus Cboe's reported IV: median +0.0011, p05..p95 -0.0019..+0.0041 (Cboe's conventions undocumented). 30d ATM IV: median
0.209 (range 0.172..0.305); correlation with Cboe's `iv30` 0.99998, median +0.0025 above it. Term structure: upward
sloping (30d < ~1y) on 69% of dates; inverted on the high-vol dates (e.g. 2026-11-05: front 0.404, 30d 0.305, ~1y 0.250).
Smile (expiry nearest 30d, median over dates) by k = ln(K/F): -0.15: 0.336, -0.10: 0.292, -0.05: 0.250, 0: 0.208,
+0.05: 0.175, +0.10: 0.178 -> steep put skew (k-0.10 minus k+0.05 = 0.105), slight call-wing upturn.
r sensitivity: r = 3.5% instead of 4.5% moves median 30d ATM IV by -0.0002.

### Rule change disclosed
The pre-stated forward rule (v1: OLS of C_mid - P_mid on K, DF from the slope) failed: 501/630 expiry fits had DF > 1.0005
(median DF 1.005), consistent with early-exercise premium in ITM American puts. It was replaced, after seeing that
diagnostic and the v1 summary numbers, by v2 (forward from the strike minimising |C-P|, DF from an assumed r = 4.5%, the
provenance code's constant). v1 output is preserved in `artifacts/derivs/realdata_iv_v1_parity_regression.json`. This is
descriptive work with no decision rule and no trials, so nothing is re-counted; the change is recorded here instead.

## Tests and mutation checks
`tests/test_derivs.py`: 40 tests, ~17 s, none marked slow; env-dependent tests skip cleanly if `envs/derivs` is
missing, real-data tests skip if `/Users/newman/quantalpha/data` is absent. `artifacts/derivs/mutation_check.py`: 14
mutants (vol/time/flag validation, bound checks, IV clamping in each library, worker re-validation, vega units, pickle
find_class / static gate / unresolved STACK_GLOBAL, protected-era guard disabled or moved after a worker call): **14/14
caught**, restored code passes (`artifacts/derivs/mutation_check.json`).

## Not built
No options strategy, signal, or backtest (no dev-era data). No volatility surface fitting (SVI etc.). No dividend
or rate curve: flat r, q. No American IV inversion (American is pricing-only). The Cboe study ignores intraday time to
expiry (T in whole days) and uses Cboe's 15-minute-delayed quotes. The full Cboe run takes ~4 min (two r settings over
~95k quotes; QuantLib objects are built per row).
