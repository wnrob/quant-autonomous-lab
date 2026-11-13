"""nautilus_trader adapter (core side - never imports nautilus_trader). The event-driven backtest runs in the isolated envs/nautilus
venv through adapters/workers/nautilus_worker.py (subprocess; data crosses as parquet/JSON files).

Scope (ARCHITECTURE principle 7, no execution path): offline BacktestEngine + one simulated venue only. No live node, no
execution/data clients, no credentials. LGPL-3.0: the library is an unmodified installed wheel; nothing is vendored or copied.

Protected window: any input dated after DEV_END (2019-12-31 = end of validation) raises ProtectedWindowAccess. There is NO override.
"""
from __future__ import annotations
import contextlib, json, tempfile
from pathlib import Path
import pandas as pd
from adapters.envrun import ROOT, run_in_env
from engine.errors import AdapterUnavailable

DEV_END = pd.Timestamp("2019-12-31")          # end of train+validation; protected test window (+embargo) starts after this
MODES = ("same_bar_close", "next_bar_close")


class ProtectedWindowAccess(RuntimeError):
    """Raised when data after DEV_END is passed to the event-driven backtest. Not configurable."""


def available() -> bool:
    return (ROOT / "envs" / "nautilus" / "bin" / "python").exists()


def assert_dev_only(index) -> None:
    idx = pd.DatetimeIndex(index)
    bad = idx[idx > DEV_END]
    if len(bad): raise ProtectedWindowAccess(f"{len(bad)} dates after {DEV_END.date()} ({bad[0].date()}..{bad[-1].date()}) passed to the event-driven backtest; refusing (no override)")


def month_end_dates(px: pd.Series) -> list[str]:
    """Last trading day of each calendar month in the series (same rule as engine.backtesting.vector.trend_signal).
    This plays the role of a published exchange calendar: the strategy is told in advance which sessions are month-ends."""
    return px.groupby(px.index.to_period("M")).tail(1).index.strftime("%Y-%m-%d").tolist()


def run_trend(prices: pd.Series, lookback_months: int = 12, cost_bps: float = 2.0, starting_cash: float = 1_000_000.0,
              mode: str = "same_bar_close", price_precision: int = 4, workdir: Path | None = None, timeout: int = 900) -> dict:
    """Event-driven long/flat trend baseline in nautilus. Returns dict with 'equity' (DataFrame indexed by date: cash, qty, close, equity),
    'returns' (daily equity returns), 'fills' (DataFrame), 'decisions', 'final', 'nautilus_version', 'config'."""
    if mode not in MODES: raise ValueError(f"mode must be one of {MODES}")
    px = prices.dropna().astype(float)
    if not isinstance(px.index, pd.DatetimeIndex) or px.empty: raise ValueError("prices must be a non-empty Series on a DatetimeIndex")
    assert_dev_only(px.index)                                                     # guard BEFORE anything is written or launched
    with (contextlib.nullcontext(str(workdir)) if workdir else tempfile.TemporaryDirectory(prefix="qal_nautilus_")) as d:
        wd = Path(d); wd.mkdir(parents=True, exist_ok=True)
        f, out = wd / "prices.parquet", wd / "result.json"
        px.rename("close").to_frame().to_parquet(f)
        run_in_env("nautilus", "nautilus_worker.py", {"data": str(f), "out": str(out), "lookback_months": int(lookback_months),
                   "cost_bps": float(cost_bps), "starting_cash": float(starting_cash), "mode": mode,
                   "month_ends": month_end_dates(px), "price_precision": int(price_precision)}, timeout=timeout)
        res = json.loads(out.read_text())
    eq = pd.DataFrame(res["equity"]); eq.index = pd.DatetimeIndex(eq.pop("date"));
    if len(eq) != len(px): raise AdapterUnavailable(f"nautilus worker returned {len(eq)} bars for {len(px)} inputs")
    fills = pd.DataFrame(res["fills"], columns=["date", "side", "qty", "price", "commission"])
    if len(fills): fills["date"] = pd.to_datetime(fills["date"])
    res.update(equity=eq, returns=eq["equity"].pct_change().rename("event_driven"), fills=fills)
    return res
