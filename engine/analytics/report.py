"""Markdown + JSON + PNG reporting. Reports always state data provenance and synthetic status."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from engine.data.ingest import DatasetMeta

def _pct(x): return f"{x*100:.2f}%"

def plot_equity(curves: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for n, r in curves.items():
        eq = (1 + r.fillna(0)).cumprod(); ax[0].plot(eq.index, eq, label=n); ax[1].plot(eq.index, eq / eq.cummax() - 1, lw=0.8)
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title(title); ax[1].set_ylabel("drawdown")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def render_markdown(meta: DatasetMeta, res: dict) -> str:
    L = ["# Quant Autonomous Lab - demo report", ""]
    if meta.kind == "synthetic":
        L += ["> **SYNTHETIC TEST DATA - NOT REAL MARKET DATA. Nothing below is an investment finding.**", ""]
    L += [f"- Data: `{meta.provider}` ({meta.kind}); symbols {', '.join(meta.symbols)}; {meta.first_date} -> {meta.last_date}; {meta.rows} rows; sha256 `{meta.sha256[:16]}`", ""]
    L += ["## 1. Descriptive statistics (daily returns)", "", "| asset | ann.ret | ann.vol | Sharpe(rf=0) | skew | ex.kurt | maxDD |", "|---|---|---|---|---|---|---|"]
    for a, d in res["describe"].items():
        L.append(f"| {a} | {_pct(d['ann_return_geometric'])} | {_pct(d['ann_vol'])} | {d['sharpe_rf0']:.2f} | {d['skew']:.2f} | {d['excess_kurtosis']:.1f} | {_pct(d['max_drawdown'])} |")
    h = res["hypothesis"]
    L += ["", f"## 2. Research hypothesis ({res['hypothesis_asset']})", "", f"*{h['hypothesis']}*", "",
          f"- n(after positive)={h['n_pos']}, n(after negative)={h['n_neg']}",
          f"- mean next-month return: {_pct(h['mean_next_after_pos'])} vs {_pct(h['mean_next_after_neg'])}; diff {_pct(h['diff'])}",
          f"- Welch t = {h['welch_t']:.2f} (p={h['p_value_two_sided']:.3f}); bootstrap 95% CI of diff: [{_pct(h['bootstrap_ci95_diff'][0])}, {_pct(h['bootstrap_ci95_diff'][1])}]",
          f"- **Verdict: {h['verdict']}**", "- Caveats: " + "; ".join(h["caveats"])]
    b = res["backtest"]
    L += ["", f"## 3. Strategy backtest - 12m trend long/flat on {res['hypothesis_asset']}", "",
          f"Signal at month-end, applied next trading day; {b['cost_bps']} bps per unit turnover; annual turnover {b['turnover_annual']:.2f}x. Parameters fixed a priori (no tuning).", "",
          "| series | ann.ret | ann.vol | Sharpe | maxDD |", "|---|---|---|---|---|"]
    for n in ("strategy", "buy_and_hold"):
        d = b[n]; L.append(f"| {n} | {_pct(d['ann_return_geometric'])} | {_pct(d['ann_vol'])} | {d['sharpe_rf0']:.2f} | {_pct(d['max_drawdown'])} |")
    o = res["optimization"]
    L += ["", "## 4. Portfolio optimisation (long-only, out-of-sample)", "", f"Train {o['train'][0]}..{o['train'][1]}; test {o['test'][0]}..{o['test'][1]} (weights frozen after train).", "",
          "| portfolio | weights | OOS ret | OOS vol | OOS Sharpe | OOS maxDD |", "|---|---|---|---|---|---|"]
    for n, p in o["portfolios"].items():
        L.append(f"| {n} | {p['weights']} | {_pct(p['oos_ann_return'])} | {_pct(p['oos_ann_vol'])} | {p['oos_sharpe']:.2f} | {_pct(p['oos_max_drawdown'])} |")
    r = res["risk"]
    L += ["", "## 5. Risk (strategy, historical simulation)", "", "| metric | value |", "|---|---|"] + [f"| {k} | {_pct(v) if 'days' not in k else v} |" for k, v in r.items()]
    L += ["", "![equity](equity.png)", "", "---", "Research tooling output; not investment advice. Backtests are hypothetical and subject to the stated caveats."]
    return "\n".join(L) + "\n"

def write_run(outdir: Path, meta: DatasetMeta, res: dict, curves: dict[str, pd.Series]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps({"dataset": meta.__dict__, **res}, indent=2, default=str))
    (outdir / "report.md").write_text(render_markdown(meta, res))
    plot_equity(curves, outdir / "equity.png", "Strategy vs buy & hold" + (" [SYNTHETIC]" if meta.kind == "synthetic" else ""))
    return outdir / "report.md"
