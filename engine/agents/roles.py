"""Agent roster. Each agent is a deterministic, rule-based component with a narrow responsibility.
`implemented=False` agents are declared so the dashboard shows the true state. No agent here uses an LLM; no model weights are trained or updated."""
from __future__ import annotations
import json
from dataclasses import dataclass
import pandas as pd
from engine.data.quality import QualityReport, check_series
from engine.memory.store import Store, h
from engine.research.hypothesis import momentum_hypothesis
from engine.research.validation import Criteria
from engine.strategies.trend import baseline_spec, name_of

@dataclass(frozen=True)
class AgentInfo:
    id: int; name: str; implemented: bool; scope: str

ROSTER = [
 AgentInfo(1, "Chief Research Officer", True, "rule-based proposer + backlog from memory (engine/agents/roles.py:propose); no adaptive budget allocation across research directions, only fixed daily caps"),
 AgentInfo(2, "Data Intelligence", True, "quality checks (gaps/dups/outliers/stale), provenance hash; point-in-time availability check exists (engine/data/pit.py) but is NOT called by the pipelines yet; futures roll/stale-contract audit of the user's data (engine/futures, descriptive only); never fills data; no delisted-ETF or per-contract futures data"),
 AgentInfo(3, "Statistical Research", True, "momentum hypothesis test; Engle-Granger cointegration over all 276 ETF pairs with Benjamini-Hochberg FDR (engine/families/statarb_pairs.py, dev window only; real result: 0/276 survive); stationarity/regime tests NOT implemented"),
 AgentInfo(4, "Alpha Discovery", True, "cross-sectional ETF factors (momentum/reversal/lowvol grid, complexity-ordered), judged on tradeable active return not IC; 24 ETFs, survivorship-biased (engine/data/pit.py can flag this and check inception dates, but it is NOT yet wired into this agent; no delisted ETFs available); no nonlinear/ML search, Alphalens not wired"),
 AgentInfo(5, "Machine Learning", True, "model ladder (momentum/Ridge/HistGB/MLP) on 24 ETFs, purged+embargoed walk-forward, dev-window-only guard, ml_models registry, PSI/seed drift monitor; real run = NULL (docs/ML.md); qlib Linear/LGB family via envs/ml also NULL (docs/QLIB_STATARB.md); no deep RL, no protected-test look yet"),
 AgentInfo(6, "Strategy Evolution", True, "trend family only (lookback, vol-target); futures roll/back-adjust and options IV tooling exist (engine/futures, adapters/derivs_adapter.py) but NO futures/options strategy research (no per-contract futures or pre-2020 options data)"),
 AgentInfo(7, "Backtesting & Validation", True, "vector backtest, costs, chronological split+embargo, protected test, DSR, bootstrap; event-driven cross-check via nautilus_trader (simulated venue only, dev window only, docs/EVENT_DRIVEN.md); no futures rolls inside the backtester"),
 AgentInfo(8, "Risk Management", True, "mandatory frozen limits gate (drawdown/leverage/turnover); no stress/VaR scenarios beyond CVaR"),
 AgentInfo(9, "Portfolio Construction", True, "7 allocators (own scipy + skfolio via isolated env), 40% cap, walk-forward on dev window only, drift monitor, DSR-vs-equal-weight verdict; research sleeves only, nothing promoted (docs/PORTFOLIO.md)"),
 AgentInfo(10, "Research Evaluator & Memory", True, "SQLite+FTS memory, lessons incl. negative results, dedup; no semantic/embedding search, no automatic detection of unproductive research directions beyond dedup"),
 AgentInfo(11, "Forward Shadow Evaluation", True, "hash-chained pre-outcome signal log, next-bar PaperBroker fills, cost/tracking report, human-only approval gate for SHADOW/PAPER_APPROVED (docs/FORWARD.md); trend family daily single-asset only; NOT a broker, no real orders; nothing in the main store is in SHADOW and no forward P&L exists yet"),
 AgentInfo(12, "Coding Agent (Part 6)", True, "may only PROPOSE unified diffs: isolated git workspace copy, sandboxed tests (no secrets, no network except loopback), golden-result diff, auto-rollback, audit report (docs/CODING_AGENT.md); cannot apply to the lab or touch protected paths without a human approval bound to the patch sha256; LLM patch path tested with a fake transport only"),
]

def data_agent(px: pd.Series, symbol: str) -> QualityReport: return check_series(px, symbol)

def stat_agent(px: pd.Series, store: Store, symbol: str) -> int:
    hy = momentum_hypothesis(px)
    text = f"[{symbol}] {hy['hypothesis']}"
    return store.add_hypothesis(text, json.dumps({k: hy[k] for k in ("diff", "welch_t", "p_value_two_sided", "verdict", "n_pos", "n_neg")}), source="statistical_research_agent")

def propose(store: Store, base_spec: dict, base_result: dict, crit: Criteria, data_sha: str) -> tuple[dict, str] | None:
    """CRO proposer: derive the next falsifiable variant from the baseline's failure analysis; skip anything already tested."""
    m = base_result["metrics"]; v = m["validation"]; vb = m["validation_benchmark"]; cands: list[tuple[dict, str]] = []
    risk_hot = v["max_drawdown"] < 0.6 * crit.risk.max_drawdown or v["ann_vol"] > 0.12
    if risk_hot:
        for vt in (0.10, 0.12): cands.append(({**base_spec, "vol_target": vt}, f"Baseline validation vol/drawdown is high ({v['ann_vol']:.1%} / {v['max_drawdown']:.1%}); hypothesis: a {vt:.0%} vol-target overlay raises validation Sharpe by >= {crit.challenger_margin_sharpe} without breaching risk limits."))
    if m.get("turnover_annual", 0) > 1.5:
        cands.append(({**base_spec, "lookback_months": 15}, f"Turnover {m['turnover_annual']:.1f}x/yr is high; hypothesis: a 15m lookback cuts whipsaw costs and raises Sharpe."))
    if v["sharpe"] < vb["sharpe"]:
        for lb in (10, 9, 6): cands.append(({**base_spec, "lookback_months": lb}, f"Baseline Sharpe {v['sharpe']:.2f} trails B&H {vb['sharpe']:.2f}; hypothesis: a {lb}m lookback responds faster to regime change."))
    for lb in (10, 9, 6, 15): cands.append(({**base_spec, "lookback_months": lb}, f"Neighbourhood robustness: does a {lb}m lookback behave like the 12m baseline? (fragility check)"))
    seen = set(); 
    for spec, why in cands:
        sh = h(spec)
        if sh in seen: continue
        seen.add(sh)
        if store.find_experiment(sh, crit.digest(), data_sha): continue
        return spec, why
    return None
