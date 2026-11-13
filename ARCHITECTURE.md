# Architecture

## Principles
1. **External code is quarantined.** `external/**` holds shallow, read-only clones for inspection. Nothing in `engine/` or `adapters/` imports from or copies those paths.
2. **Stable internal interfaces, replaceable backends.** The engine depends on small contracts (below); third-party libraries are reached only through `adapters/`, imported lazily inside functions, so a missing/incompatible library raises `AdapterUnavailable` instead of breaking the engine.
3. **Smallest reliable library per function.** Core logic (returns, risk, vector backtest, scipy optimiser) is ~400 lines of numpy/pandas/scipy that we can test fully. Heavy libraries are cross-checks or optional backends.
4. **One env per heavy library.** `envs/core` is all the CLI needs. Heavy libs get their own venv (see DEPENDENCY_COMPATIBILITY.md).
5. **Honest status.** `adapters/registry.py` marks an adapter `operational` only if a test exercises it.
6. **Provenance everywhere.** Each dataset has `.meta.json` (provider, kind real/synthetic, date range, sha256). Synthetic data is labelled in provider name, metadata, and report banner.
7. **No execution path.** No broker credentials, no order routing. Repos with live-trading modules (Lean, nautilus_trader, FinRL-Trading) are policy-disabled.

## Layout
```
engine/                     # our code (package `engine`, CLI `qal`)
  data/ingest.py            # DataProvider protocol; Yahoo/CSV/Synthetic; parquet cache + provenance
  analytics/{returns,report}.py
  research/hypothesis.py    # pre-stated hypothesis test (Welch t + bootstrap)
  backtesting/vector.py     # long/flat vectorised engine, 1-day signal lag, bps costs
  optimization/portfolio.py # long-only min-var / max-Sharpe, shrinkage, chronological OOS split
  risk/metrics.py           # hist VaR/CVaR, drawdown, underwater
  orchestration/pipeline.py # composes all stages -> artifacts/runs/<ts>_<kind>/
  strategies/ models/ agents/ memory/   # reserved for next stage (empty packages)
adapters/                   # third-party bridges + registry (+ health checks)
external/{core,portfolio,backtesting,ai,directories}/   # 24 shallow clones (untouched)
scripts/                    # acquire.py, audit.py, probe_envs.py, build_docs.py, curated.py
datasets/ artifacts/ experiments/ configs/ docs/ tests/ envs/
```

## Data flow
```
DataProvider.fetch -> validate_prices -> parquet+meta (datasets/)
   -> simple_returns -> describe / hypothesis / backtest(trend_signal -> run_backtest) / optimise(OOS) / risk
   -> report.write_run -> results.json + report.md + equity.png (artifacts/runs/)
```

## Contracts
| contract | signature |
|---|---|
| DataProvider | `fetch(symbols, start, end) -> wide DataFrame (adj. close)`; attrs `name`, `kind` |
| signal | `Series[0/1]` on trading-day index (NaN = warmup); engine applies `.shift(1)` |
| optimiser adapter | `(returns: DataFrame) -> Series weights (long-only, sum 1)` |
| metrics adapter | `(returns: Series) -> dict` |

## Where the next stage plugs in
- `engine/agents` + `engine/memory`: experiment registry (spec -> result -> verdict) writing to `experiments/`; RD-Agent is the candidate loop driver, run in a sandboxed env.
- `engine/strategies`: strategy objects emitting signals; `engine/backtesting` gains adapters for vectorbt (sweeps) and nautilus (fidelity).
- Add a pre-registration + walk-forward gate (the repo's existing /backtest desk pattern) before any strategy is allowed a verdict.
