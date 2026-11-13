# Integration Plan

## Roles (decision per repo)
| tier | repos | decision |
|---|---|---|
| **Core (own code)** | - | returns/risk/backtest/optimiser in `engine/` |
| **Operational adapters** | PyPortfolioOpt, quantstats | tested, in `envs/core` |
| **Next adapters (priority order)** | skfolio (CV'd optimisation), py_vollib (IV/Greeks), alphalens-reloaded (factor eval), vectorbt (sweeps), QuantLib wheel (curves) | one venv each; write adapter + cross-check test against engine before marking operational |
| **Heavy / later** | qlib, RD-Agent, nautilus_trader, zipline-reloaded, OpenBB, Riskfolio-Lib, bt, QuantInvestStrats | evaluate after the experiment registry exists |
| **Research-only, quarantined** | FinRL, FinRL-Trading, arbitragelab | isolated env; no credentials; expect dependency friction |
| **Unsupported / reference** | Lean (.NET), backtrader (GPL, stale), machine-learning-for-trading (notebooks), 3 awesome-lists | never imported |

## Method for each new adapter
1. `uv venv envs/<name> --python 3.12`; install the **released wheel** (`--only-binary :all:`), never run the clone's `setup.py`/build scripts.
2. Write adapter in `adapters/<lib>_adapter.py` honouring the contract; lazy import; raise `AdapterUnavailable`.
3. Add a test that compares against the engine's own implementation on fixed data.
4. Only then flip status to `operational` in `registry.py` and regenerate docs (`scripts/build_docs.py`).

## Risks and mitigations
- **Dependency conflicts** -> per-library envs; Python 3.12 pin; probe results tracked.
- **License contamination** -> GPL (backtrader) never linked; LGPL (nautilus) kept as external dependency; unlicensed lists read-only.
- **Supply chain / code execution** -> wheels only, no clone builds, shallow clones with hooks disabled, RD-Agent/FinRL only in sandboxed envs.
- **Overfitting by automation** (the real risk of the next stage) -> every automated experiment needs pre-registered spec, chronological OOS, and a fixed qualification bar; optimiser demo already shows frozen "optimal" weights losing to equal-weight OOS.
- **Data risk** -> Yahoo is convenient, not institutional; keep provenance hashes; add a second source before trusting results.
