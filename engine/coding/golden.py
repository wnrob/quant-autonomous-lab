"""Golden behaviour probes: a FIXED set of engine function outputs on SYNTHETIC, seeded fixtures.

Run inside the sandbox by ws_runner.py against whatever `engine` the workspace holds (before and after the patch).
This file is loaded by path from the trusted LAB copy, so it must not import engine.* at module level.
All fixture data here is SYNTHETIC (seeded random walks); nothing touches real or protected-window market data.
"""
from __future__ import annotations
import dataclasses, math

REL_TOL, ABS_TOL = 1e-9, 1e-12
FIXTURE_LABEL = "SYNTHETIC seeded geometric random walks (numpy default_rng 20240611), business days 2005-01-03.., not market data"

def _fixtures():
    import numpy as np, pandas as pd
    rng = np.random.default_rng(20240611)
    idx = pd.bdate_range("2005-01-03", periods=2000)
    r = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx, name="synthetic_ret")
    px = (100 * (1 + r).cumprod()).rename("SYNTH")
    multi = pd.DataFrame(rng.normal(0.0003, 0.01, (len(idx), 4)), index=idx, columns=["S1", "S2", "S3", "S4"])
    multi["S2"] = 0.6 * multi["S1"] + 0.4 * multi["S2"]
    return r, px, multi

def _ser(x):
    import numpy as np, pandas as pd
    if dataclasses.is_dataclass(x) and not isinstance(x, type): return {f.name: _ser(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, pd.DataFrame): return {"__frame__": list(map(str, x.columns)), "cols": {str(c): _ser(x[c]) for c in x.columns}}
    if isinstance(x, pd.Series): x = x.to_numpy(dtype=float)
    if isinstance(x, np.ndarray):
        a = np.asarray(x, dtype=float).ravel(); v = a[~np.isnan(a)]
        if len(v) == 0: return {"__array__": int(a.size), "nan": int(a.size)}
        w = np.arange(1, len(v) + 1) / len(v)                         # position-weighted sum makes reorderings visible
        return {"__array__": int(a.size), "nan": int(a.size - v.size), "sum": float(v.sum()), "wsum": float((v * w).sum()),
                "min": float(v.min()), "max": float(v.max()), "first": float(v[0]), "last": float(v[-1])}
    if isinstance(x, dict): return {str(k): _ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_ser(v) for v in x]
    if isinstance(x, (bool, str)) or x is None: return x
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x)
    return repr(x)

def _probes():
    """name -> zero-arg callable. Imports happen inside each probe so a missing module is recorded, not fatal."""
    r, px, multi = _fixtures()
    def p_returns():
        from engine.analytics import returns as m
        return {"describe": m.describe(r), "max_drawdown": m.max_drawdown(r), "simple_returns": m.simple_returns(px), "rolling_vol": m.rolling_vol(r)}
    def p_risk():
        from engine.risk import metrics as m
        return {"risk_summary": m.risk_summary(r), "var_cvar_95": m.var_cvar(r, 0.95), "drawdown_series": m.drawdown_series(r)}
    def p_backtest():
        from engine.backtesting import vector as m
        out = {}
        for lb in (6, 12):
            sig = m.trend_signal(px, lb); res = m.run_backtest(px, sig, cost_bps=2.0)
            out[f"L{lb}"] = {"signal": sig, "result": res}
        return out
    def p_trend():
        from engine.strategies import trend as m
        s0 = m.baseline_spec("SYNTH"); s1 = {**s0, "lookback_months": 6, "vol_target": 0.10}
        return {"baseline_spec": s0, "w0": m.build_weights(s0, px), "w1": m.build_weights(s1, px), "names": [m.name_of(s0), m.name_of(s1)]}
    def p_validation():
        from engine.research import validation as m
        trials = [0.01, 0.03, -0.02, 0.05]
        return {"sharpe": m.sharpe(r), "sortino": m.sortino(r), "calmar": m.calmar(r), "psr": m.psr(r), "expected_max_sharpe": m.expected_max_sharpe(trials),
                "deflated_sharpe": m.deflated_sharpe(r, trials), "bootstrap_ci": m.bootstrap_sharpe_ci(r, n=200, seed=0), "window_metrics": m.window_metrics(r),
                "criteria": m.Criteria(), "criteria_digest": m.Criteria().digest(), "splits_lens": {k: len(v) for k, v in m.Splits().slice(r).items()}}
    def p_risk_gate():
        from engine.research import validation as m
        from engine.backtesting import vector as bt
        res = bt.run_backtest(px, bt.trend_signal(px, 12))
        return {"gate_default": m.risk_gate(res.returns, res.weights, m.RiskLimits()), "gate_tight": m.risk_gate(res.returns, res.weights, m.RiskLimits(max_drawdown=-0.01, max_annual_turnover=0.1))}
    def p_portfolio():
        from engine.optimization import portfolio as m
        return {"shrunk_cov": m.shrunk_cov(multi), "min_variance": m.min_variance(multi), "max_sharpe": m.max_sharpe(multi)}
    def p_paper():
        from engine.backtesting import paper as m
        b = m.PaperBroker(); b.order("SYNTH", 10, 100.0, "t0"); b.order("SYNTH", -4, 105.0, "t1")
        errs = []
        for q, p in ((0, 100.0), (10**9, 100.0), (-100, 100.0)):
            try: b.order("SYNTH", q, p, "t2"); errs.append("accepted")
            except ValueError as e: errs.append(str(e))
        return {"cash": b.cash, "positions": b.positions, "equity": b.equity({"SYNTH": 105.0}), "cost_total": b.execution_cost_total(), "n_fills": len(b.fills), "refusals": errs}
    def p_store_rules():
        from engine.memory import store as m
        return {"statuses": list(m.STATUSES), "allowed": {k: sorted(v) for k, v in m.ALLOWED.items()}, "h": m.h({"a": 1, "b": [1, 2]})}
    return {"analytics.returns": p_returns, "risk.metrics": p_risk, "backtesting.vector": p_backtest, "strategies.trend": p_trend,
            "research.validation": p_validation, "research.risk_gate": p_risk_gate, "optimization.portfolio": p_portfolio,
            "backtesting.paper": p_paper, "memory.store_rules": p_store_rules}

PROBE_NAMES = ("analytics.returns", "risk.metrics", "backtesting.vector", "strategies.trend", "research.validation", "research.risk_gate",
               "optimization.portfolio", "backtesting.paper", "memory.store_rules")

def compute() -> dict:
    out = {}
    for name, fn in _probes().items():
        try: out[name] = {"ok": True, "value": _ser(fn())}
        except BaseException as e:                                      # noqa: BLE001 - a crashing probe is a result, not an abort
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out

def _diff(a, b, path: str, out: list) -> None:
    if len(out) >= 20: return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b or type(a) is not type(b): out.append({"at": path, "before": a, "after": b})
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb): return
        if math.isnan(fa) or math.isnan(fb) or abs(fa - fb) > max(ABS_TOL, REL_TOL * max(abs(fa), abs(fb))): out.append({"at": path, "before": a, "after": b})
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: out.append({"at": f"{path}.{k}", "before": a.get(k, "<absent>"), "after": b.get(k, "<absent>")})
            else: _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out.append({"at": path, "before": f"len {len(a)}", "after": f"len {len(b)}"}); return
        for i, (x, y) in enumerate(zip(a, b)): _diff(x, y, f"{path}[{i}]", out)
    else:
        out.append({"at": path, "before": repr(a)[:80], "after": repr(b)[:80]})

def compare(before: dict, after: dict, declared=()) -> dict:
    """Any probe whose output moved beyond tolerance is a diff. Diffs on probes named in `declared` are reported but allowed."""
    declared = set(declared or ()); changed = {}
    for name in sorted(set(before) | set(after)):
        d: list = []
        _diff(before.get(name, {"ok": False, "error": "<absent>"}), after.get(name, {"ok": False, "error": "<absent>"}), name, d)
        if d: changed[name] = d
    undeclared = sorted(n for n in changed if n not in declared)
    return {"probes": len(set(before) | set(after)), "probes_ok_before": sum(1 for v in before.values() if v.get("ok")),
            "probes_ok_after": sum(1 for v in after.values() if v.get("ok")), "changed": changed, "declared": sorted(declared),
            "declared_and_changed": sorted(n for n in changed if n in declared), "undeclared_changed": undeclared,
            "rel_tol": REL_TOL, "abs_tol": ABS_TOL, "fixtures": FIXTURE_LABEL, "ok": not undeclared}
