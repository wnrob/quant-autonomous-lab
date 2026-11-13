"""derivs worker (runs inside envs/derivs via adapters.envrun.run_in_env; never imported by core).

Ops (payload["op"]):
  price_greeks : rows [{flag,S,K,T,r,q,sigma,model}] -> price + delta/gamma/vega/theta/rho from py_vollib AND QuantLib
  iv           : rows [{flag,S,K,T,r,q,price,model}] -> implied vol from both libs (NaN + reason outside no-arb bounds)
  american     : rows [{flag,S,K,T,r,q,sigma,steps}] -> QuantLib CRR binomial American price (labelled extra) + European ref
  pickle_scan  : {path} -> static opcode scan of a pickle (never loads it)
  qqq_chains   : {path, spot_path, out, r, q} -> restricted-unpickle the cached QQQ call histories, IV both libs -> parquet
  cboe         : {paths, out} -> raw Cboe delayed-quote chain snapshots -> forward/DF per expiry -> Black-76 IV both libs -> parquet

UNIT CONVENTION of every returned Greek (both libs normalised to this):
  delta dV/dS, gamma d2V/dS2, vega dV/dsigma (per 1.00 vol), theta dV/dt per YEAR (calendar time, = -dV/dT),
  rho dV/dr per 1.00 rate. For model='black76' the underlying is the forward F (delta = dV/dF) and rho is taken
  holding F fixed (= -T*V).
  py_vollib natively returns vega and rho per 1% and theta per day/365 -> multiplied by 100 / 100 / 365 here.
TIME: T = calendar days / 365 (ACT/365F). QuantLib is day-granular, so its path REQUIRES T*365 to be an integer
  (within 1e-9); evaluation date is fixed and maturity = eval + round(T*365) days, flat curves Actual365Fixed,
  continuous compounding, so QuantLib's year fraction equals T exactly.
"""
import importlib.util, json, math, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    s = importlib.util.spec_from_file_location(f"_derivs_{name}", ROOT / "engine" / "derivs" / f"{name}.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


C = _load("checks")          # same validation rules as the core adapter (stdlib-only module)
NAN = float("nan")

# ---------------------------------------------------------------- py_vollib
def _pv():
    from py_vollib.black_scholes_merton import black_scholes_merton as bsm
    from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_iv
    import py_vollib.black_scholes_merton.greeks.analytical as bsm_g
    from py_vollib.black import black
    from py_vollib.black.implied_volatility import implied_volatility as b_iv
    import py_vollib.black.greeks.analytical as b_g
    return dict(bsm=bsm, bsm_iv=bsm_iv, bsm_g=bsm_g, black=black, b_iv=b_iv, b_g=b_g)


def pv_price_greeks(P, flag, S, K, T, r, q, sig, model):
    if model == "bsm":
        g = P["bsm_g"]; a = (flag, S, K, T, r, sig, q)
        price = P["bsm"](*a)
    else:
        g = P["b_g"]; a = (flag, S, K, T, r, sig)
        price = P["black"](*a)
    return dict(price=price, delta=g.delta(*a), gamma=g.gamma(*a), vega=g.vega(*a) * 100.0,
                theta=g.theta(*a) * 365.0, rho=g.rho(*a) * 100.0)


def pv_iv(P, price, flag, S, K, T, r, q, model):
    try:
        v = P["bsm_iv"](price, S, K, T, r, q, flag) if model == "bsm" else P["b_iv"](price, S, K, r, T, flag)
    except Exception as e:                                  # py_vollib raises Below/AboveMaximum etc.
        return NAN, f"py_vollib:{type(e).__name__}"
    v = float(v)
    if not math.isfinite(v) or v <= 0: return NAN, "py_vollib:non_positive_or_nan"
    return v, None

# ---------------------------------------------------------------- QuantLib
_EVAL = None


def _ql():
    global _EVAL
    import QuantLib as ql
    if _EVAL is None:
        _EVAL = ql.Date(2, 1, 2030); ql.Settings.instance().evaluationDate = _EVAL
    return ql


def _days(T: float) -> int:
    n = round(T * 365.0)
    if n < 1 or abs(T * 365.0 - n) > 1e-9:
        raise C.InvalidOptionInput(f"QuantLib path needs T = integer days/365 (ACT/365F); got T*365={T*365.0!r}")
    return n


def _ql_option(ql, flag, S, K, T, r, q, sig):
    dc = ql.Actual365Fixed(); cal = ql.NullCalendar()
    spot = ql.SimpleQuote(S); vol = ql.SimpleQuote(sig)
    rts = ql.YieldTermStructureHandle(ql.FlatForward(_EVAL, ql.QuoteHandle(ql.SimpleQuote(r)), dc, ql.Continuous))
    qts = ql.YieldTermStructureHandle(ql.FlatForward(_EVAL, ql.QuoteHandle(ql.SimpleQuote(q)), dc, ql.Continuous))
    vts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(_EVAL, cal, ql.QuoteHandle(vol), dc))
    proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot), qts, rts, vts)
    pay = ql.PlainVanillaPayoff(ql.Option.Call if flag == "c" else ql.Option.Put, K)
    mat = _EVAL + _days(T)
    return pay, mat, proc


def ql_price_greeks(flag, S, K, T, r, q, sig, model):
    ql = _ql(); pay, mat, proc = _ql_option(ql, flag, S, K, T, r, q, sig)   # black76: q == r already (checks.validate)
    opt = ql.VanillaOption(pay, ql.EuropeanExercise(mat)); opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    rho = opt.rho() + (opt.dividendRho() if model == "black76" else 0.0)   # black76 rho holds F fixed: r moves both curves
    return dict(price=opt.NPV(), delta=opt.delta(), gamma=opt.gamma(), vega=opt.vega(), theta=opt.theta(), rho=rho)


def ql_iv(price, flag, S, K, T, r, q, model):
    ql = _ql(); pay, mat, proc = _ql_option(ql, flag, S, K, T, r, q, 0.2)
    opt = ql.VanillaOption(pay, ql.EuropeanExercise(mat)); opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    try:
        v = opt.impliedVolatility(price, proc, 1e-12, 10000, 1e-8, 10.0)
    except RuntimeError as e:
        return NAN, "quantlib:" + str(e).split("\n")[0][:80]
    return float(v), None


def ql_american(flag, S, K, T, r, q, sig, steps):
    ql = _ql(); pay, mat, proc = _ql_option(ql, flag, S, K, T, r, q, sig)
    am = ql.VanillaOption(pay, ql.AmericanExercise(_EVAL, mat)); am.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", int(steps)))
    eu = ql.VanillaOption(pay, ql.EuropeanExercise(mat)); eu.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    eub = ql.VanillaOption(pay, ql.EuropeanExercise(mat)); eub.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", int(steps)))
    return dict(american_crr=am.NPV(), european_analytic=eu.NPV(), european_crr=eub.NPV())

# ---------------------------------------------------------------- batch ops
def op_price_greeks(a):
    P = _pv(); out = []
    for row in a["rows"]:
        model = row.get("model", "bsm")
        flag, S, K, T, r, q, sig = C.validate(row["flag"], row["S"], row["K"], row["T"], row["r"], row.get("q", 0.0), row["sigma"], model)
        if sig is None: raise C.InvalidOptionInput("sigma required for pricing")
        out.append({"vollib": pv_price_greeks(P, flag, S, K, T, r, q, sig, model), "quantlib": ql_price_greeks(flag, S, K, T, r, q, sig, model)})
    return {"results": out}


def iv_both(P, price, flag, S, K, T, r, q, model):
    p, reason, (flag, S, K, T, r, q) = C.iv_precheck(price, flag, S, K, T, r, q, model)
    if reason: return {"vollib": NAN, "quantlib": NAN, "reason_vollib": reason, "reason_quantlib": reason}
    v1, e1 = pv_iv(P, p, flag, S, K, T, r, q, model); v2, e2 = ql_iv(p, flag, S, K, T, r, q, model)
    return {"vollib": v1, "quantlib": v2, "reason_vollib": e1, "reason_quantlib": e2}


def op_iv(a):
    P = _pv()
    return {"results": [iv_both(P, row["price"], row["flag"], row["S"], row["K"], row["T"], row["r"], row.get("q", 0.0), row.get("model", "bsm")) for row in a["rows"]]}


def op_american(a):
    out = []
    for row in a["rows"]:
        flag, S, K, T, r, q, sig = C.validate(row["flag"], row["S"], row["K"], row["T"], row["r"], row.get("q", 0.0), row["sigma"], "bsm")
        out.append(ql_american(flag, S, K, T, r, q, sig, row.get("steps", 801)))
    return {"results": out}


def op_pickle_scan(a):
    sp = _load("safe_pickle"); data = Path(a["path"]).read_bytes()
    found, problems = sp.scan_globals(data)
    bad = sorted(g for g in found if g not in sp.ALLOWED)
    return {"globals": sorted(map(list, found)), "problems": problems, "disallowed": [list(b) for b in bad], "bytes": len(data)}


def op_qqq_chains(a):
    import pandas as pd
    sp = _load("safe_pickle"); data = Path(a["path"]).read_bytes()
    chains = sp.safe_loads(data)                            # raises UnsafePickle before any object is built if not allowlisted
    if not isinstance(chains, dict): raise TypeError(f"expected dict, got {type(chains).__name__}")
    spot = pd.read_parquet(a["spot_path"])["close"]
    spot_by_date = {ts.date(): float(v) for ts, v in spot.items() if math.isfinite(v)}
    P = _pv(); r, q = float(a["r"]), float(a["q"]); rows = []
    for (expiry, strike), df in chains.items():
        for ts, bar in df.iterrows():
            d = ts.date(); dte = (expiry - d).days; S = spot_by_date.get(d)
            rec = dict(date=str(d), expiry=str(expiry), strike=float(strike), dte=dte, close=float(bar["close"]),
                       volume=float(bar["volume"]), spot=S if S is not None else NAN)
            if S is None: rec.update(reason="no_underlying_close")
            elif dte < 1: rec.update(reason="expiry_day")
            else:
                res = iv_both(P, float(bar["close"]), "c", S, float(strike), dte / 365.0, r, q, "bsm")
                rec.update(iv_vollib=res["vollib"], iv_quantlib=res["quantlib"], reason=res["reason_vollib"] or res["reason_quantlib"])
            rows.append(rec)
    pd.DataFrame(rows).to_parquet(a["out"])
    return {"rows": len(rows), "contracts": len(chains), "out": a["out"]}


def _parse_occ(sym: str):
    # e.g. QQQ261218C00455000 -> (root, date, flag, strike)
    import datetime as dt
    body = sym[-15:]; root = sym[:-15]
    return root, dt.date(2000 + int(body[0:2]), int(body[2:4]), int(body[4:6])), body[6].lower(), int(body[7:]) / 1000.0


def op_cboe(a):
    import datetime as dt
    import numpy as np, pandas as pd
    from zoneinfo import ZoneInfo
    P = _pv(); rows, fwd = [], []
    for path in a["paths"]:
        raw = json.loads(Path(path).read_text()); meta = json.loads(Path(path.replace(".raw.json", ".meta.json")).read_text())
        ts = pd.Timestamp(meta["source_event_ts"]); d0 = ts.tz_convert(ZoneInfo("America/New_York")).date()
        S = float(raw["data"]["current_price"]); iv30 = raw["data"].get("iv30")
        q_ = pd.DataFrame(raw["data"]["options"])
        parsed = [_parse_occ(s) for s in q_["option"]]
        q_["root"], q_["expiry"], q_["flag"], q_["K"] = zip(*parsed)
        q_["dte"] = [(e - d0).days for e in q_["expiry"]]
        q_ = q_[(q_["dte"] >= a.get("min_dte", 7)) & (q_["dte"] <= a.get("max_dte", 400))]
        q_ = q_[(q_["bid"] > 0) & (q_["ask"] > 0) & (q_["ask"] >= q_["bid"])].copy()
        q_["mid"] = (q_["bid"] + q_["ask"]) / 2; q_ = q_[(q_["ask"] - q_["bid"]) / q_["mid"] <= a.get("max_rel_spread", 0.5)]
        for (root, exp), g in q_.groupby(["root", "expiry"]):
            T = int(g["dte"].iloc[0]) / 365.0
            c = g[g["flag"] == "c"].set_index("K")["mid"]; p = g[g["flag"] == "p"].set_index("K")["mid"]
            ks = sorted(set(c.index) & set(p.index)); ks = [k for k in ks if abs(k / S - 1) <= 0.05]
            info = dict(snapshot=meta["snapshot_id"], date=str(d0), root=root, expiry=str(exp), dte=int(g["dte"].iloc[0]), spot=S, n_parity=len(ks))
            if len(ks) < 3: fwd.append({**info, "status": "too_few_parity_strikes"}); continue
            y = np.array([c[k] - p[k] for k in ks]); X = np.column_stack([np.ones(len(ks)), ks])
            (b0, b1), *_ = np.linalg.lstsq(X, y, rcond=None); DF_reg = -b1    # v1 rule; DIAGNOSTIC only (biased by American puts)
            method = a.get("forward_method", "atm_pair")
            if method == "parity_regression":
                DF = DF_reg
                if not (0.8 < DF <= 1.0005): fwd.append({**info, "status": "df_out_of_range", "DF": DF}); continue
                F = b0 / DF; r = -math.log(DF) / T
            elif method == "atm_pair":                                     # v2: F from the strike with min |C-P|, DF from assumed r
                r = float(a["r_fixed"]); DF = math.exp(-r * T)
                ks0 = min(ks, key=lambda k: abs(c[k] - p[k])); F = ks0 + (c[ks0] - p[ks0]) / DF
            else: raise ValueError(f"unknown forward_method {method!r}")
            if abs(F / S - 1) > 0.10: fwd.append({**info, "status": "forward_out_of_range", "F": F, "DF": DF}); continue
            fwd.append({**info, "status": "ok", "method": method, "F": F, "DF": DF, "r_used": r, "DF_regression_diag": DF_reg,
                        "r_implied_regression_diag": -math.log(DF_reg) / T if DF_reg > 0 else float("nan")})
            otm = g[((g["flag"] == "c") & (g["K"] >= F)) | ((g["flag"] == "p") & (g["K"] < F))]
            for o in otm.itertuples():
                res = iv_both(P, float(o.mid), o.flag, F, float(o.K), T, r, 0.0, "black76")
                rows.append(dict(snapshot=meta["snapshot_id"], ts=meta["source_event_ts"], date=str(d0), root=root, expiry=str(exp), dte=int(o.dte),
                                 flag=o.flag, K=float(o.K), bid=float(o.bid), ask=float(o.ask), mid=float(o.mid), F=F, r=r, spot=S, iv30=iv30,
                                 cboe_iv=float(o.iv), cboe_delta=float(o.delta), volume=float(o.volume), oi=float(o.open_interest),
                                 iv_vollib=res["vollib"], iv_quantlib=res["quantlib"], reason=res["reason_vollib"] or res["reason_quantlib"]))
    pd.DataFrame(rows).to_parquet(a["out"]); pd.DataFrame(fwd).to_parquet(a["out"].replace(".parquet", "_forwards.parquet"))
    return {"rows": len(rows), "expiry_fits": len(fwd), "out": a["out"]}


OPS = {"price_greeks": op_price_greeks, "iv": op_iv, "american": op_american, "pickle_scan": op_pickle_scan, "qqq_chains": op_qqq_chains, "cboe": op_cboe}


def _clean(o):
    """NaN/inf -> null so the output is strict JSON; the core adapter maps null back to NaN."""
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list): return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    if "payload_file" in a:                                 # large batches go via a file (argv size limit)
        a = {**json.loads(Path(a["payload_file"]).read_text()), **{k: a[k] for k in ("op", "result_file") if k in a}}
    out = OPS[a["op"]](a)
    if "result_file" in a:
        Path(a["result_file"]).write_text(json.dumps(_clean(out), allow_nan=False)); out = {"result_file": a["result_file"]}
    print(json.dumps(_clean(out), allow_nan=False))
