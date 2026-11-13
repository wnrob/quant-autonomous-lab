"""Portfolio Construction agent (Agent 9): combine return streams ("sleeves") into long-only, fully-invested, weight-capped portfolios and
compare allocation methods by WALK-FORWARD out-of-sample evaluation, judged against the equal-weight baseline with multiple-testing-aware
(DSR) and cost-aware criteria. Fancy allocators that do not beat equal-weight are REPORTED as not better.

Hard rules enforced in code:
  * Only the train + validation windows of V.Criteria().splits are ever used. Any test-window date raises ProtectedTestAccess.
  * Weights effective on day t are fit on rows strictly before t (lag >= 1), monthly rebalance, expanding or rolling window.
  * Every combined portfolio passes through V.risk_gate; violations are surfaced, never hidden.
  * Nothing here writes to the strategy registry or promotes anything. Demo sleeves are RESEARCH SLEEVES, NOT PROMOTED STRATEGIES.
"""
from __future__ import annotations
import argparse, json, logging, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np, pandas as pd
from scipy.optimize import minimize
from engine.analytics.returns import TRADING_DAYS
from engine.optimization.portfolio import shrunk_cov
from engine.research import validation as V
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
LABEL = "RESEARCH SLEEVES, NOT PROMOTED STRATEGIES (most are REJECTED in the registry; nothing here is CANDIDATE or higher)"
OWN_METHODS = ("equal_weight", "inverse_vol", "risk_parity", "min_variance")
SKFOLIO_METHODS = ("skfolio_hrp", "skfolio_min_variance", "skfolio_mean_risk")
BASELINE = "equal_weight"

class ProtectedTestAccess(RuntimeError):
    """Raised whenever protected test-window dates are requested."""

# ----------------------------------------------------------------------------- guards
def assert_no_test(index: pd.DatetimeIndex, crit: V.Criteria = V.Criteria()) -> None:
    """Raise if any date in `index` falls in the protected test window (crit.splits.slice(...)['test'])."""
    idx = pd.DatetimeIndex(index); test_idx = idx[idx > pd.Timestamp(crit.splits.val_end)]      # strict: includes the embargo days
    if len(test_idx): raise ProtectedTestAccess(f"{len(test_idx)} dates in the protected test window ({test_idx[0].date()}..{test_idx[-1].date()}) were requested; refusing")

def dev_window(obj: "pd.DataFrame | pd.Series", crit: V.Criteria = V.Criteria(), splits: tuple[str, ...] = ("train", "validation")):
    """Restrict to train+validation only. Asking for 'test' raises."""
    if "test" in splits: raise ProtectedTestAccess("the 'test' split is protected; this agent may only use train/validation")
    s = crit.splits.slice(pd.Series(0.0, index=obj.index)); keep = s[splits[0]].index
    for k in splits[1:]: keep = keep.union(s[k].index)
    return obj.loc[obj.index.isin(keep)]

# ----------------------------------------------------------------------------- allocators (long-only, sum 1, w <= cap)
def _check_cap(n: int, cap: float) -> None:
    if not 0 < cap <= 1 or cap * n < 1 - 1e-9: raise ValueError(f"max_weight {cap} infeasible for {n} sleeves (need cap*n >= 1)")

def apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Water-filling: clip weights above cap, redistribute the excess pro-rata over uncapped names, repeat. Keeps sum==1, w>=0."""
    w = np.clip(np.asarray(w, float), 0, None); w = w / w.sum(); _check_cap(len(w), cap)
    for _ in range(len(w) + 1):
        over = w > cap + 1e-12
        if not over.any(): break
        excess = (w[over] - cap).sum(); w[over] = cap; free = ~over & (w < cap - 1e-12)
        if not free.any(): break
        w[free] += excess * w[free] / w[free].sum() if w[free].sum() > 0 else excess / free.sum()
    return w

def _slsqp(obj: Callable, n: int, cap: float, x0: np.ndarray) -> np.ndarray:
    res = minimize(obj, x0, method="SLSQP", bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 800, "ftol": 1e-14})
    if not res.success and not np.isfinite(res.fun): raise RuntimeError(f"optimiser failed: {res.message}")
    return apply_cap(res.x, cap)

def w_equal(r: pd.DataFrame, cap: float) -> np.ndarray:
    n = r.shape[1]; _check_cap(n, cap); return apply_cap(np.full(n, 1 / n), cap)
def w_inverse_vol(r: pd.DataFrame, cap: float) -> np.ndarray:
    iv = 1 / r.std(ddof=1).to_numpy(); return apply_cap(iv, cap)
def w_risk_parity(r: pd.DataFrame, cap: float) -> np.ndarray:
    """Equal risk contribution (own scipy). Minimise sum_i (RC_i - 1/n)^2, RC_i = w_i (S w)_i / w'Sw, subject to long-only, sum 1, cap."""
    S = shrunk_cov(r); n = S.shape[0]; x0 = apply_cap(1 / np.sqrt(np.diag(S)), cap)
    def obj(w):
        v = w @ S @ w; rc = w * (S @ w) / v; return float(((rc - 1 / n) ** 2).sum()) * 1e3
    return _slsqp(obj, n, cap, x0)
def w_min_variance(r: pd.DataFrame, cap: float, delta: float = 0.1) -> np.ndarray:
    S = shrunk_cov(r, delta); n = S.shape[0]; _check_cap(n, cap); return _slsqp(lambda w: w @ S @ w, n, cap, np.full(n, 1 / n))
OWN: dict[str, Callable[[pd.DataFrame, float], np.ndarray]] = {"equal_weight": w_equal, "inverse_vol": w_inverse_vol, "risk_parity": w_risk_parity, "min_variance": w_min_variance}

# ----------------------------------------------------------------------------- drift monitor
def _offdiag(c: np.ndarray) -> np.ndarray: return c[~np.eye(len(c), dtype=bool)]

def drift_monitor(fit_returns: pd.DataFrame, recent_returns: pd.DataFrame, fit_weights: "pd.Series | None" = None, current_weights: "pd.Series | None" = None,
                  corr_thresh: float = 0.20, weight_thresh: float = 0.25) -> dict:
    """Flags when (a) mean absolute change of pairwise correlations (recent vs fitting window) > corr_thresh, or (b) half-L1 distance between
    current and fitted weights (one-way turnover to get back) > weight_thresh. Also reports the change in average pairwise correlation."""
    cf, cr = fit_returns.corr().to_numpy(), recent_returns.corr().to_numpy()
    d = np.abs(_offdiag(cr) - _offdiag(cf)); out = {"corr_mean_abs_change": float(np.nanmean(d)), "corr_max_abs_change": float(np.nanmax(d)),
                                                   "avg_corr_fit": float(np.nanmean(_offdiag(cf))), "avg_corr_recent": float(np.nanmean(_offdiag(cr))), "weight_shift": None}
    out["corr_flag"] = bool(out["corr_mean_abs_change"] > corr_thresh); out["weight_flag"] = False
    if fit_weights is not None and current_weights is not None:
        ws = 0.5 * float((current_weights.reindex(fit_weights.index).fillna(0) - fit_weights).abs().sum()); out["weight_shift"] = ws; out["weight_flag"] = bool(ws > weight_thresh)
    out["flag"] = out["corr_flag"] or out["weight_flag"]; return out

# ----------------------------------------------------------------------------- walk-forward
def rebalance_schedule(index: pd.DatetimeIndex, min_train: int = 252, lag: int = 1, window: int | None = None) -> list[tuple[int, int, int]]:
    """Monthly: first trading day i of each month. Fit rows = [fit_start, fit_end) with fit_end = i - lag + 1 <= i, i.e. STRICTLY before i
    (lag >= 1 enforced). Returns (i, fit_start, fit_end). Needs >= min_train fit rows."""
    if lag < 1: raise ValueError("lag must be >= 1 (fit data must be strictly before the rebalance day)")
    firsts = pd.Series(np.arange(len(index)), index=index).groupby(index.to_period("M")).first().to_numpy(); out = []
    for i in firsts:
        fe = int(i) - lag + 1; fs = 0 if window is None else max(0, fe - window)
        if fe - fs >= min_train: out.append((int(i), fs, fe))
    return out

@dataclass
class WFResult:
    name: str
    gross: pd.Series
    turnover: pd.Series                  # one-way |target - drifted weights|.sum() on rebalance days, else 0
    targets: pd.DataFrame                # step-function target weights (date x sleeve), effective on the date shown
    drift_events: list[dict] = field(default_factory=list)
    pre_trade_drift: list[float] = field(default_factory=list)
    def net(self, cost_bps: float) -> pd.Series: return (self.gross - self.turnover * cost_bps / 1e4).rename(self.name)

def simulate(returns: pd.DataFrame, schedule: list[tuple[int, int, int]], targets: list[np.ndarray], name: str) -> WFResult:
    """Hold target weights, let them drift with returns between rebalances; turnover at rebalance = |target - drifted|."""
    R = returns.to_numpy(); n = R.shape[1]; start = schedule[0][0]; nxt = {s[0]: k for k, s in enumerate(schedule)}
    w = np.zeros(n); gross, turn, T = [], [], []; tgt_rows = np.zeros((len(R) - start, n)); cur_t = np.zeros(n); drifts = []
    for i in range(start, len(R)):
        t = 0.0
        if i in nxt:
            cur_t = targets[nxt[i]]; t = float(np.abs(cur_t - w).sum()); w = cur_t.copy()
            if i != start: drifts.append(t)
        tgt_rows[i - start] = cur_t; rp = float(w @ R[i]); gross.append(rp); turn.append(t); w = w * (1 + R[i]) / (1 + rp)
    idx = returns.index[start:]
    return WFResult(name, pd.Series(gross, idx, name=name), pd.Series(turn, idx), pd.DataFrame(tgt_rows, idx, returns.columns), pre_trade_drift=drifts)

def walk_forward(returns: pd.DataFrame, methods: tuple[str, ...] = OWN_METHODS + SKFOLIO_METHODS, *, max_weight: float = 0.40, min_train: int = 252, lag: int = 1,
                 window: int | None = None, crit: V.Criteria = V.Criteria(), recent: int = 63, corr_thresh: float = 0.20, weight_thresh: float = 0.25,
                 skfolio_ok: bool | None = None) -> dict[str, WFResult]:
    """Monthly walk-forward. `returns` must be a complete (no NaN) sleeve x day frame lying entirely in train/validation (else ProtectedTestAccess)."""
    assert_no_test(returns.index, crit)
    if returns.isna().any().any(): raise ValueError("returns contain NaN; align sleeves to a common sample first")
    _check_cap(returns.shape[1], max_weight); sched = rebalance_schedule(returns.index, min_train, lag, window)
    if not sched: raise ValueError("not enough history for a single rebalance")
    if skfolio_ok is None:
        from adapters import skfolio_adapter as S; skfolio_ok = S.available()
    res: dict[str, WFResult] = {}
    for m in methods:
        if m in OWN: tg = [OWN[m](returns.iloc[fs:fe], max_weight) for _, fs, fe in sched]
        elif m in SKFOLIO_METHODS:
            if not skfolio_ok: log.warning("skfolio env not available; skipping %s", m); continue
            from adapters import skfolio_adapter as S
            tg = [w.to_numpy() for w in S.weights_batch(returns, m.removeprefix("skfolio_"), max_weight, [(fs, fe) for _, fs, fe in sched])]
        else: raise ValueError(f"unknown method {m}")
        r = simulate(returns, sched, tg, m); prev_t, prev_fit = None, None
        for k, (i, fs, fe) in enumerate(sched):
            rec = returns.iloc[max(fs, fe - recent):fe]; fit = returns.iloc[fs:fe]
            cw = pd.Series(tg[k], returns.columns); pw = pd.Series(tg[k - 1], returns.columns) if k else None
            d = drift_monitor(fit, rec, pw, cw, corr_thresh, weight_thresh); d["date"] = str(returns.index[i].date()); r.drift_events.append(d)
        res[m] = r
    return res

# ----------------------------------------------------------------------------- statistics
def enb_meucci(w: np.ndarray, cov: np.ndarray) -> float:
    """Effective number of (uncorrelated) bets: exp(entropy) of PCA risk contributions (Meucci 2009)."""
    lam, E = np.linalg.eigh(cov); lam = np.clip(lam, 1e-18, None); p = (E.T @ w) ** 2 * lam; p = p / p.sum(); p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))

def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float: return float(w @ np.sqrt(np.diag(cov)) / np.sqrt(w @ cov @ w))

def lookthrough(targets: pd.DataFrame, sleeve_weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Underlying-asset exposure = sum_s portfolio_weight_s * sleeve_asset_weights_s (sleeve weights are as-traded, already lagged)."""
    parts = []
    for s in targets.columns:
        sw = sleeve_weights[s]; sw = sw.to_frame(s) if isinstance(sw, pd.Series) else sw
        parts.append(sw.reindex(targets.index).ffill().fillna(0.0).mul(targets[s], axis=0))
    out = parts[0]
    for p in parts[1:]: out = out.add(p, fill_value=0.0)
    return out.fillna(0.0)

def summarize(r: WFResult, sleeves: pd.DataFrame, crit: V.Criteria, cost_bps: float, sleeve_weights: dict[str, pd.DataFrame] | None = None) -> dict:
    net = r.net(cost_bps); oos = sleeves.loc[net.index]; cov = oos.cov().to_numpy() * TRADING_DAYS; wbar = r.targets.mean().to_numpy(); wbar = wbar / wbar.sum()
    cor = oos.corr().to_numpy(); off = ~np.eye(len(cor), dtype=bool); ww = np.outer(wbar, wbar)
    gate_w = lookthrough(r.targets, sleeve_weights) if sleeve_weights else r.targets
    yrs = len(net) / TRADING_DAYS; m = V.window_metrics(net)
    m.update({"gross_sharpe": V.sharpe(r.gross), "net_sharpe_stress_cost": V.sharpe(r.net(cost_bps * crit.stress_cost_mult)),
              "diversification_ratio": diversification_ratio(wbar, cov), "effective_bets": enb_meucci(wbar, cov), "avg_pairwise_corr_sleeves": float(cor[off].mean()),
              "weighted_avg_pairwise_corr": float((ww * cor)[off].sum() / ww[off].sum()),
              "turnover_oneway_annual": float(r.turnover.sum() / yrs), "mean_pretrade_drift": float(np.mean(r.pre_trade_drift)) if r.pre_trade_drift else 0.0,
              "cost_drag_annual": float((r.gross - net).sum() / yrs), "max_weight_observed": float(r.targets.to_numpy().max()),
              "avg_weights": {c: round(float(v), 4) for c, v in zip(sleeves.columns, wbar)},
              "drift_flags": int(sum(e["flag"] for e in r.drift_events)), "n_rebalances": len(r.drift_events),
              "corr_drift_flags": int(sum(e["corr_flag"] for e in r.drift_events)), "weight_drift_flags": int(sum(e["weight_flag"] for e in r.drift_events)),
              "risk_violations": V.risk_gate(net, gate_w, crit.risk), "risk_gate_weights": "look-through sleeve asset weights" if sleeve_weights else "sleeve target weights"})
    return m

def compare(results: dict[str, WFResult], sleeves: pd.DataFrame, crit: V.Criteria = V.Criteria(), cost_bps: float = 5.0,
            sleeve_weights: dict[str, pd.DataFrame] | None = None, extra_trial_sharpes_daily: list[float] | None = None) -> dict:
    """Per-method statistics + DSR-aware decision against the equal-weight baseline.
    DECISION uses the RAW active return vs equal-weight (the rule fixed before any result was seen). A vol-matched active series is computed as a DIAGNOSTIC ONLY and can never produce a BETTER verdict:
    an earlier version let it decide after seeing a raw-criterion 'no winner' result, which is a post-hoc rule change (caught in independent review).
    Trials = every method compared here (+ extra_trial_sharpes_daily, e.g. earlier allocator searches). A challenger is BETTER only if ALL hold:
    raw active return (vs EW) mean > 0 net of costs AND at stress costs; active-return DSR >= crit.min_dsr over the challenger trials; net Sharpe >= EW + crit.challenger_margin_sharpe;
    no risk-gate violations."""
    if BASELINE not in results: raise ValueError("equal_weight baseline missing")
    nets = {k: r.net(cost_bps) for k, r in results.items()}; stress = {k: r.net(cost_bps * crit.stress_cost_mult) for k, r in results.items()}
    ew = nets[BASELINE]; ews = stress[BASELINE]; table = {k: summarize(r, sleeves, crit, cost_bps, sleeve_weights) for k, r in results.items()}
    all_trials = [V.sharpe(n) / np.sqrt(TRADING_DAYS) for n in nets.values()] + list(extra_trial_sharpes_daily or [])
    chal = [k for k in results if k != BASELINE]
    def vm(x: pd.Series, base: pd.Series) -> pd.Series:      # vol-matched to the baseline (hypothetical scaling, used ONLY for the test): isolates risk-adjusted edge
        return (x * (base.std(ddof=1) / x.std(ddof=1)) - base).dropna()
    raw = {k: (nets[k] - ew).dropna() for k in chal}; act = raw; diag_vm = {k: vm(nets[k], ew) for k in chal}
    act_trials = [V.sharpe(a) / np.sqrt(TRADING_DAYS) for a in act.values()] + list(extra_trial_sharpes_daily or [])
    for k in results: table[k]["dsr_own_sharpe"] = V.deflated_sharpe(nets[k], all_trials)
    verdict = {}
    for k in chal:
        a = act[k]; sa = (stress[k] - ews).dropna(); dsr_a = V.deflated_sharpe(a, act_trials); reasons = []
        if not a.mean() > 0: reasons.append(f"active return vs equal-weight not positive after {cost_bps:g}bps ({a.mean() * TRADING_DAYS:+.2%}/yr)")
        if not sa.mean() > 0: reasons.append(f"active return not positive at {crit.stress_cost_mult:g}x costs")
        if not dsr_a >= crit.min_dsr: reasons.append(f"active-return DSR {dsr_a:.2f} < {crit.min_dsr} (trials={len(act_trials)})")
        dsh = table[k]["sharpe"] - table[BASELINE]["sharpe"]
        if not dsh >= crit.challenger_margin_sharpe: reasons.append(f"net Sharpe margin {dsh:+.2f} < {crit.challenger_margin_sharpe} vs equal-weight")
        if table[k]["risk_violations"]: reasons += ["RISK: " + v for v in table[k]["risk_violations"]]
        verdict[k] = {"better_than_equal_weight": not reasons, "active_ann_return": float(a.mean() * TRADING_DAYS), "raw_active_ann_return": float(raw[k].mean() * TRADING_DAYS), "diagnostic_vol_matched_active_ann_return": float(diag_vm[k].mean() * TRADING_DAYS), "diagnostic_vol_matched_active_sharpe": V.sharpe(diag_vm[k]), "active_sharpe": V.sharpe(a), "active_dsr": dsr_a,
                      "sharpe_margin": float(dsh), "reasons": reasons or ["beats equal-weight on all predefined criteria (still research-only)"] }
        if not reasons and verdict[k]["raw_active_ann_return"] <= 0: verdict[k]["reasons"].append(f"CAVEAT: win is risk-adjusted only; unscaled return is {verdict[k]['raw_active_ann_return']:+.2%}/yr vs equal-weight (lower vol)")
    winners = [k for k, v in verdict.items() if v["better_than_equal_weight"]]; best = max(winners, key=lambda k: verdict[k]["active_dsr"]) if winners else None
    return {"label": LABEL, "cost_bps": cost_bps, "n_trials": len(all_trials), "table": table, "verdict_vs_equal_weight": verdict,
            "selected": best or BASELINE, "summary": (f"{best} beats equal-weight after costs and DSR" if best else "NO allocator beats the equal-weight baseline after costs/DSR; equal-weight retained")}

# ----------------------------------------------------------------------------- demo sleeves (cached data only)
XS_PICK = ("xs-momentum-L126-S0", "xs-momentum-L252-S21", "xs-momentum-L63-S0", "xs-reversal-L21-S0", "xs-lowvol-L63-S0", "xs-lowvol-L252-S0")
BH = ("SPY", "TLT", "GLD", "DBC")

def build_demo_sleeves(px: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Returns (sleeve returns, sleeve asset-weight histories). Pure functions of cached prices; sleeves are causal (1-day lag inside)."""
    from engine.agents import alpha as A
    from engine.backtesting.vector import run_backtest
    from engine.strategies import trend as T
    rets: dict[str, pd.Series] = {}; wts: dict[str, pd.DataFrame] = {}
    sp = T.baseline_spec("SPY"); bt = run_backtest(px["SPY"], T.build_weights(sp, px["SPY"]), sp["cost_bps"]); rets["trend-SPY"] = bt.returns; wts["trend-SPY"] = bt.weights.rename("SPY").to_frame()
    specs = {A.name_of(s): s for s, _ in A.candidate_specs()}
    for nm in XS_PICK:
        net, W, _ = A.run_xs(specs[nm], px); rets[nm] = net; wts[nm] = W
    for sym in BH:
        rets[f"bh-{sym}"] = px[sym].pct_change().dropna(); wts[f"bh-{sym}"] = pd.DataFrame({sym: 1.0}, index=px.index)
    return pd.DataFrame(rets), wts

# ----------------------------------------------------------------------------- reporting / entry point
def render_report(res: dict, meta: dict) -> str:
    t = res["table"]; L = [f"# Portfolio construction report", "", f"**{LABEL}**", "",
         f"- windows: train+validation only ({meta['first']}..{meta['last']}; test window never touched), {meta['n_obs']} OOS days, monthly walk-forward, lag {meta['lag']}d, "
         f"{'expanding' if not meta['window'] else str(meta['window']) + 'd rolling'} fit", f"- sleeves ({len(meta['sleeves'])}): {', '.join(meta['sleeves'])}",
         f"- max sleeve weight {meta['max_weight']:.0%}; costs {res['cost_bps']:g}bps on allocation turnover (sleeve-internal costs already in sleeve returns); trials counted for DSR: {res['n_trials']}", "",
         "| method | net ret | Sharpe | Sortino | Calmar | maxDD | DivRatio | ENB | avg corr | turnover/yr | drift flags | DSR | risk gate |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in t.items():
        L.append(f"| {k} | {m['net_return_total']:+.1%} | {m['sharpe']:.2f} | {m['sortino']:.2f} | {m['calmar']:.2f} | {m['max_drawdown']:.1%} | {m['diversification_ratio']:.2f} | {m['effective_bets']:.2f} | "
                 f"{m['weighted_avg_pairwise_corr']:.2f} | {m['turnover_oneway_annual']:.2f}x | {m['drift_flags']}/{m['n_rebalances']} | {m['dsr_own_sharpe']:.2f} | {'PASS' if not m['risk_violations'] else 'VIOLATION'} |")
    L += ["", "## Decision vs equal-weight baseline", "", f"**{res['summary']}**", ""]
    for k, v in res["verdict_vs_equal_weight"].items():
        L.append(f"- {k}: {'BETTER' if v['better_than_equal_weight'] else 'NOT BETTER'} (raw active {v['active_ann_return']:+.2%}/yr [decision basis], vol-matched diagnostic {v['diagnostic_vol_matched_active_ann_return']:+.2%}/yr, active DSR {v['active_dsr']:.2f}, Sharpe margin {v['sharpe_margin']:+.2f}) -- {'; '.join(v['reasons'])}")
    viol = {k: m["risk_violations"] for k, m in t.items() if m["risk_violations"]}
    L += ["", "## Risk gate", "", (json.dumps(viol, indent=1) if viol else "no violations (limits: V.Criteria().risk, applied to " + next(iter(t.values()))["risk_gate_weights"] + ")")]
    return "\n".join(L) + "\n"

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m engine.agents.portfolio", description="Compare allocation methods over research sleeves (train+validation only, cached data)")
    ap.add_argument("--max-weight", type=float, default=0.40); ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=None, help="rolling fit window in days (default expanding)"); ap.add_argument("--no-skfolio", action="store_true")
    ap.add_argument("--out", default=None); a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from engine.data.ingest import YahooProvider, load_prices; from engine.data.universe import ETF24
    px, m = load_prices(YahooProvider(), ETF24, "2005-01-01", offline=True)              # offline=True: cache only, never network
    crit = V.Criteria(); sleeves_all, sw = build_demo_sleeves(px)
    sleeves = dev_window(sleeves_all, crit).dropna()                                        # train+validation ONLY, common sample
    from adapters import skfolio_adapter as S; use_sk = S.available() and not a.no_skfolio
    methods = OWN_METHODS + (SKFOLIO_METHODS if use_sk else ()); t0 = time.time()
    wf = walk_forward(sleeves, methods, max_weight=a.max_weight, window=a.window, crit=crit, skfolio_ok=use_sk)
    res = compare(wf, sleeves, crit, a.cost_bps, sw); res["skfolio_used"] = use_sk
    meta = {"first": str(next(iter(wf.values())).gross.index[0].date()), "last": str(next(iter(wf.values())).gross.index[-1].date()), "n_obs": len(next(iter(wf.values())).gross),
            "lag": 1, "window": a.window, "sleeves": list(sleeves.columns), "max_weight": a.max_weight, "data_sha": m.sha256, "secs": round(time.time() - t0, 1)}
    res["meta"] = meta; out = Path(a.out) if a.out else ROOT / "artifacts" / "portfolio" / time.strftime("%Y%m%dT%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float)); (out / "report.md").write_text(render_report(res, meta))
    print(render_report(res, meta)); print(f"artifacts: {out}"); return res

if __name__ == "__main__": main()
