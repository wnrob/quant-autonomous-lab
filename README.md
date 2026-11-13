# Quant Autonomous Lab

A local quantitative-research workspace: 24 open-source quant repositories acquired, audited and catalogued, plus a small, tested integration layer (`engine/`, `adapters/`) and a CLI (`qal`) that runs six end-to-end demos on real market data.

> Research tooling only. No broker credentials, no order execution. Nothing here is investment advice.

## Setup (exact commands)
Requires `git`, `uv`, and network access. System Python can be anything; the lab uses an isolated 3.12 venv.
```bash
cd quant-autonomous-lab
uv venv --python 3.12 envs/core --seed
uv pip install --python envs/core/bin/python -e ".[data,portfolio,report,dev]"
```
Re-acquire the 24 repos (idempotent; shallow clones, hooks disabled, nothing executed):
```bash
python3 -I scripts/acquire.py        # -> artifacts/acquisition_report.json, retry_report.md
python3 -I scripts/audit.py          # -> artifacts/audit_raw.json   (static, read-only)
python3 -I scripts/probe_envs.py     # optional, slow: PyPI-wheel import probe per library
envs/core/bin/python scripts/build_docs.py   # regenerates manifest + matrices + audits
```

## Launch
```bash
envs/core/bin/qal demo                          # real data: SPY/TLT/GLD 2005-today via Yahoo -> artifacts/runs/<ts>_real/report.md
envs/core/bin/qal demo --provider synthetic --symbols A B C --start 2000-01-03 --end 2020-12-31   # software test data only, banner-labelled
envs/core/bin/qal ingest --symbols QQQ IEF       # fetch + cache + provenance
envs/core/bin/qal adapters --check               # adapter status (+ live health checks)
envs/core/bin/qal repos                          # the 24 repos with license/SHA/role
envs/core/bin/python -m pytest                   # tests
```
Add `--offline` to use cached data only.

## The six demos (all in `qal demo`)
1. **Data ingestion** - provider interface, validation, parquet cache, provenance + sha256.
2. **Descriptive analysis** - return/vol/skew/kurtosis/drawdown per asset.
3. **Hypothesis** - "positive trailing-12m return predicts higher next-month return": Welch t + bootstrap, rule fixed in advance.
4. **Backtest** - 12m trend long/flat, month-end signal, next-day execution, bps costs, fixed parameters (no tuning).
5. **Portfolio optimisation** - long-only min-var / max-Sharpe, weights frozen on first 70%, scored on last 30%.
6. **Risk & performance report** - VaR/CVaR/drawdown/underwater + Markdown, JSON and equity PNG.

## Docs
`REPOSITORY_MANIFEST.json` · `CAPABILITY_MATRIX.md` · `LICENSE_AUDIT.md` · `DEPENDENCY_COMPATIBILITY.md` · `INTEGRATION_PLAN.md` · `ARCHITECTURE.md` · `IMPLEMENTATION_STATUS.md` · `artifacts/retry_report.md`

## Where things are
`external/` (24 untouched clones, ~2.2 GB) · `engine/` (our code) · `adapters/` (third-party bridges) · `envs/` (venvs; `core` is required) · `datasets/` (cache) · `artifacts/` (runs, audits).
