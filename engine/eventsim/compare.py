"""Fidelity comparison: engine/backtesting/vector.py vs the nautilus_trader event-driven run of the same trend baseline.
This is a simulator cross-check, NOT a strategy evaluation (no Criteria/DSR verdict is produced and nothing is stored in a QAE store).

Run (train+validation only; anything after 2019-12-31 raises inside the adapter):
    envs/core/bin/python -m engine.eventsim.compare            -> artifacts/nautilus/
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from adapters import nautilus_adapter as NA
from engine.backtesting.vector import run_backtest, trend_signal

OUT = Path(__file__).resolve().parents[2] / "artifacts" / "nautilus"


def _stats(r: pd.Series) -> dict:
    r = r.dropna(); eq = (1 + r).cumprod()
    sd = r.std(ddof=1)
    return {"total_return": float(eq.iloc[-1] - 1), "ann_return": float(eq.iloc[-1] ** (252 / len(r)) - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
            "ann_vol": float(sd * np.sqrt(252)), "max_drawdown": float((eq / eq.cummax() - 1).min()), "days": int(len(r))}


def vector_side(px: pd.Series, lookback: int, cost_bps: float, starting_cash: float) -> dict:
    bt = run_backtest(px, trend_signal(px, lookback), cost_bps)
    w = bt.weights; r = bt.returns
    eq = starting_cash * (1 + r).cumprod(); eq_prev = eq.shift(1).fillna(starting_cash)
    dw = w.diff().fillna(w)                                   # first valid day: entry from flat (vector convention: fillna(w.abs()))
    cost_frac = dw.abs() * cost_bps / 1e4
    idx = px.index
    trades = []
    for d in dw[dw != 0].index:
        dec = idx[idx.get_loc(d) - 1]                         # w_d = signal at close of previous session -> implied trade at that close
        trades.append({"decision_date": dec, "side": "BUY" if dw[d] > 0 else "SELL", "vec_px": float(px[dec]),
                       "vec_notional": float(abs(dw[d]) * eq_prev[d]), "vec_cost": float(cost_frac[d] * eq_prev[d])})
    return {"returns": r, "weights": w, "equity": eq, "trades": pd.DataFrame(trades), "cost_frac": cost_frac,
            "total_cost_dollars": float((cost_frac * eq_prev).sum()), "total_cost_return_units": float(cost_frac.sum())}


def compare(px: pd.Series, ev: dict, lookback: int = 12, cost_bps: float = 2.0, starting_cash: float = 1_000_000.0) -> dict:
    NA.assert_dev_only(px.index)
    v = vector_side(px, lookback, cost_bps, starting_cash)
    win = v["returns"].index
    re = ev["returns"].reindex(win); rv = v["returns"]
    diff = re - rv
    e = ev["equity"]; f = ev["fills"].copy()
    # per-trade table: match vector trade (decision date) to the nautilus fill of the same side in order
    vt = v["trades"]; rows = []
    fq = {s: f[f.side == s].reset_index(drop=True) for s in ("BUY", "SELL")}; k = {"BUY": 0, "SELL": 0}
    for _, t in vt.iterrows():
        s = t["side"]; i = k[s]; k[s] += 1
        x = fq[s].iloc[i] if i < len(fq[s]) else None
        row = {"decision_date": t["decision_date"].date().isoformat(), "side": s, "vec_px": t["vec_px"], "vec_notional": t["vec_notional"], "vec_cost": t["vec_cost"]}
        if x is not None:
            fd = pd.Timestamp(x["date"]); after = e.index[e.index > fd]
            cash_after = float(e.loc[after[0], "cash"]) if len(after) else float(ev["final"]["cash"])
            row.update(ev_fill_date=fd.date().isoformat(), ev_qty=x["qty"], ev_px=x["price"],
                       px_diff_bps=(x["price"] / t["vec_px"] - 1) * 1e4, ev_notional=x["qty"] * x["price"], ev_commission=x["commission"],
                       ev_cash_after=cash_after, fill_lag_sessions=int(e.index.get_loc(fd) - e.index.get_loc(t["decision_date"])))
        rows.append(row)
    unmatched = {s: len(fq[s]) - k[s] for s in fq}
    tbl = pd.DataFrame(rows)
    # diagnostics (analysis of the given runs, not new runs): residual-cash drag while long, fee-basis difference
    # row d is recorded at the START of on_bar(d), i.e. after the fill at close d-1: (cash_d, qty_d) is what was held over day d
    held_val = e["qty"] * e["close"].shift(1)
    cash_w = (e["cash"] / (e["cash"] + held_val)).reindex(win)
    long_days = (e["qty"] > 0).reindex(win).fillna(False)
    px_ret = px.pct_change().reindex(win)
    cash_drag = float((-(cash_w.where(long_days, 0.0)) * px_ret).sum())
    ev_comm = float(f["commission"].sum()) if len(f) else 0.0
    out = {"window": [win[0].date().isoformat(), win[-1].date().isoformat()], "days": int(len(win)),
           "vector": {**_stats(rv), "trades": int(len(vt)), "total_cost_dollars": v["total_cost_dollars"], "total_cost_return_units": v["total_cost_return_units"]},
           "event_driven": {**_stats(re), "trades": int(len(f)), "total_cost_dollars": ev_comm,
                            "final_equity": float(ev["final"]["equity"]), "final_cash": float(ev["final"]["cash"]), "final_qty": float(ev["final"]["qty"])},
           "daily_return_corr": float(re.corr(rv)), "tracking_error_ann": float(diff.std(ddof=1) * np.sqrt(252)),
           "max_abs_daily_diff": float(diff.abs().max()), "max_abs_daily_diff_date": diff.abs().idxmax().date().isoformat(),
           "days_abs_diff_gt_1bp": int((diff.abs() > 1e-4).sum()), "days_abs_diff_gt_10bp": int((diff.abs() > 1e-3).sum()),
           "sum_daily_diff": float(diff.sum()),
           "max_abs_fill_px_diff_bps": float(tbl["px_diff_bps"].abs().max()) if "px_diff_bps" in tbl and len(tbl) else None,
           "unmatched_event_fills": unmatched, "cash_drag_estimate_return_units": cash_drag,
           "mean_cash_weight_while_long": float(cash_w[long_days].mean()) if long_days.any() else 0.0}
    return {"summary": out, "trades": tbl, "diff": diff.rename("event_minus_vector")}


def load_spy_dev() -> tuple:
    from engine.data.ingest import YahooProvider, load_prices
    df, meta = load_prices(YahooProvider(), ["SPY"], "2005-01-01", offline=True)
    assert meta.kind == "real"
    px = df["SPY"].dropna()
    return px[px.index <= NA.DEV_END], meta                         # slice to train+validation immediately


def main(argv=None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    px, meta = load_spy_dev()
    prereg = json.loads((OUT / "prereg.json").read_text())
    configs = [("primary_same_bar_close_1m", "same_bar_close", 1_000_000.0),
               ("sens_next_bar_close_1m", "next_bar_close", 1_000_000.0),
               ("sens_same_bar_close_10k", "same_bar_close", 10_000.0)]
    allres = {"data": {"provider": meta.provider, "kind": meta.kind, "sha256": meta.sha256, "first": px.index[0].date().isoformat(), "last": px.index[-1].date().isoformat(), "rows": len(px)},
              "prereg_written_at": prereg["written_at_utc"], "runs": {}}
    for name, mode, cash in configs:                                  # one heavy job at a time
        ev = NA.run_trend(px, 12, 2.0, cash, mode)
        c = compare(px, ev, 12, 2.0, cash)
        c["summary"].update(mode=mode, starting_cash=cash, nautilus_version=ev["nautilus_version"], config=ev["config"])
        allres["runs"][name] = c["summary"]
        c["trades"].to_csv(OUT / f"trades_{name}.csv", index=False)
        c["diff"].to_csv(OUT / f"daily_diff_{name}.csv")
        print(name, json.dumps({k: c["summary"][k] for k in ("daily_return_corr", "tracking_error_ann", "max_abs_daily_diff")}))
    (OUT / "comparison.json").write_text(json.dumps(allres, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
