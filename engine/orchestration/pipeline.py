"""End-to-end demo pipeline composing all stages."""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import pandas as pd
from engine.analytics import report, returns as rt
from engine.backtesting.vector import run_backtest, trend_signal
from engine.data.ingest import DataProvider, DatasetMeta, load_prices
from engine.optimization.portfolio import evaluate_oos
from engine.research.hypothesis import momentum_hypothesis
from engine.risk.metrics import risk_summary
log = logging.getLogger(__name__)
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

def analyse(prices: pd.DataFrame, meta: DatasetMeta, hyp_asset: str, cost_bps: float = 2.0) -> tuple[dict, dict[str, pd.Series]]:
    rets = rt.simple_returns(prices)
    px = prices[hyp_asset].dropna()
    bt = run_backtest(px, trend_signal(px, 12), cost_bps)
    bh = px.pct_change().reindex(bt.returns.index)
    multi = rets.dropna()   # common-history window for portfolio work
    res = {"describe": rt.describe_all(rets), "hypothesis_asset": hyp_asset,
           "hypothesis": momentum_hypothesis(px),
           "backtest": {"cost_bps": cost_bps, "turnover_annual": bt.turnover_annual,
                        "strategy": rt.describe(bt.returns), "buy_and_hold": rt.describe(bh)},
           "optimization": evaluate_oos(multi), "risk": risk_summary(bt.returns)}
    return res, {"strategy": bt.returns, "buy_and_hold": bh}

def run_demo(provider: DataProvider, symbols: Sequence[str], start: str, end: str | None = None, *,
             hyp_asset: str | None = None, cost_bps: float = 2.0, offline: bool = False, refresh: bool = False,
             outdir: Path | None = None) -> Path:
    prices, meta = load_prices(provider, symbols, start, end, offline=offline, refresh=refresh)
    log.info("dataset %s rows=%d %s..%s kind=%s", meta.provider, meta.rows, meta.first_date, meta.last_date, meta.kind)
    res, curves = analyse(prices, meta, hyp_asset or symbols[0], cost_bps)
    outdir = outdir or ARTIFACTS / "runs" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{meta.kind}"
    path = report.write_run(outdir, meta, res, curves)
    log.info("report written: %s", path)
    return path
