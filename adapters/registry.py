"""Adapter registry. Status is 'operational' ONLY if an automated test exercises the adapter (see tests/test_adapters.py)."""
from __future__ import annotations
from adapters.base import AdapterSpec

R = AdapterSpec
REGISTRY: list[AdapterSpec] = [
    R("PyPortfolioOpt", "portfolio optimisation", "operational", "core", "tests/test_adapters.py cross-checks vs engine optimiser"),
    R("quantstats", "performance metrics / tearsheets", "operational", "core", "tests/test_adapters.py cross-checks vs engine metrics"),
    R("skfolio", "portfolio optimisation (sklearn API)", "operational", "envs/portfolio", "HRP/min-variance/mean-risk via isolated worker; cross-checked vs own scipy min-variance (tests/test_portfolio.py)"),
    R("Riskfolio-Lib", "portfolio / risk optimisation", "untested", "envs/portfolio", "import probe only; adapter not written"),
    R("vectorbt", "fast vectorised backtests", "untested", "envs/backtest", "import probe only; adapter not written"),
    R("bt", "tree-based portfolio backtests", "untested", "envs/backtest", "import probe only; adapter not written"),
    R("alphalens-reloaded", "factor evaluation", "untested", "envs/factor", "import probe only; adapter not written"),
    R("py_vollib", "options Black-Scholes/Greeks/IV", "operational", "envs/derivs", "BSM/Black-76 price/Greeks/IV via adapters/derivs_adapter.py worker; cross-checked vs QuantLib on a fixed grid (tests/test_derivs.py); European only; real options data is post-2024 (protected era) so descriptive use ONLY"),
    R("QuantLib", "derivatives/curve pricing", "operational", "envs/derivs", "PyPI wheel 1.44 only (cloned C++ source NOT built); AnalyticEuropeanEngine price/Greeks/IV + CRR American; flat r/q only, no curves (tests/test_derivs.py)"),
    R("QuantInvestStrats", "strategy analytics/reporting", "untested", "envs/portfolio", "import probe only"),
    R("arbitragelab", "pairs/stat-arb research", "operational", "envs/statarb", "py3.11 ONLY, source-built in sandbox, unmaintained since 2024-05; Engle-Granger hedge vector/ADF stat only (p-values from statsmodels MacKinnon); pairs family BH-FDR over 276 pairs, real result NULL (tests/test_statarb.py)"),
    R("zipline-reloaded", "event-driven backtests", "untested", "envs/backtest", "needs bundle ingestion; not attempted"),
    R("qlib", "ML factor/alpha platform", "operational", "envs/ml", "expression engine (features(), verified vs pandas) + LinearModel/LGBModel family (engine/families/qlib_model.py, dev window only, 7 pre-registered trials, real result NULL); real LinearModel fit/predict exercised by a fast test on a tiny synthetic panel, LGBModel only by an @slow test; qlib backtest/workflows/recorder NOT used (tests/test_qlib_models.py)"),
    R("RD-Agent", "LLM-driven R&D agents", "operational", "envs/agents", "LLM BACKEND ONLY via sandbox + local Ollama; scenarios/loops NOT enabled (execute generated code; no Docker); live call exercised only by an @slow test, health_check checks the env exists"),
    R("FinRL", "RL trading environments", "untested", "envs/rl", "torch stack; research only"),
    R("FinRL-Trading", "RL trading pipelines", "untested", "envs/rl", "has broker connectors - DISABLED by policy"),
    R("OpenBB", "data platform / provider hub", "untested", "envs/data", "large dependency tree; keep isolated"),
    R("nautilus_trader", "high-fidelity event-driven backtest/execution", "operational", "envs/nautilus", "LGPL-3 wheel 1.231.0, unmodified, worker process; BacktestEngine + simulated venue ONLY, dev window only; live/execution clients forbidden by test (tests/test_nautilus.py); trend baseline cross-check only"),
    R("Lean", "full algo-trading engine (C#/.NET)", "unsupported", "n/a", "needs .NET SDK + Docker/LEAN CLI; out of scope for Python layer"),
    R("backtrader", "legacy event-driven backtester", "reference_only", "n/a", "GPL-3.0 + unmaintained since 2023-04; do not link into the engine"),
    R("machine-learning-for-trading", "book code / notebooks", "reference_only", "n/a", "learning material, pinned to py3.14 env"),
    R("awesome-quant", "curated link directory", "reference_only", "n/a", "no license file - treat as read-only reference; not redistributed"),
    R("awesome-systematic-trading", "curated link directory", "reference_only", "n/a", "no license file - read-only reference"),
    R("awesome-open-source-trading", "curated link directory", "reference_only", "n/a", "CC0"),
]

def health_check(spec: AdapterSpec) -> str:
    """Run a tiny real call through an operational adapter. Returns 'ok' or the failure."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (400, 3)), columns=list("ABC"), index=pd.bdate_range("2020-01-01", periods=400))
    try:
        if spec.repo == "PyPortfolioOpt":
            from adapters.pypfopt_adapter import min_vol_weights; min_vol_weights(r)
        elif spec.repo == "quantstats":
            from adapters.quantstats_adapter import metrics; metrics(r["A"])
        elif spec.repo == "skfolio":
            import adapters.envrun as e; return "ok" if (e.ROOT / "envs/portfolio/bin/python").exists() else "FAIL: envs/portfolio missing"
        elif spec.repo == "qlib":
            import adapters.envrun as e; return "ok" if (e.ROOT / "envs/ml/bin/python").exists() else "FAIL: envs/ml missing"
        elif spec.repo in ("py_vollib", "QuantLib"):
            import adapters.envrun as e; return "ok" if (e.ROOT / "envs/derivs/bin/python").exists() else "FAIL: envs/derivs missing"
        elif spec.repo == "nautilus_trader":
            import adapters.envrun as e; return "ok" if (e.ROOT / "envs/nautilus/bin/python").exists() else "FAIL: envs/nautilus missing"
        elif spec.repo == "arbitragelab":
            import adapters.envrun as e; return "ok" if (e.ROOT / "envs/statarb/bin/python").exists() else "FAIL: envs/statarb missing"
        elif spec.repo == "RD-Agent":
            import adapters.envrun as e; return "ok" if (e.ROOT / "envs/agents/bin/python").exists() else "FAIL: envs/agents missing"
        else: return "n/a"
        return "ok"
    except Exception as e: return f"FAIL: {e}"
