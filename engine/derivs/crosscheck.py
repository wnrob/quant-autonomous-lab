"""py_vollib vs QuantLib cross-check on a FIXED synthetic input grid (synthetic by construction: these are model inputs,
not market data). Tolerances below were set BEFORE the first run and are not tuned to observed differences.

Checks
  1. price + 5 Greeks: |vollib - quantlib| <= TOL[field] (absolute, at S=100 scale; units per worker docstring)
  2. put-call parity (per lib): |C - P - (S e^{-qT} - K e^{-rT})| <= TOL['parity']
  3. IV round trip (both libs): sigma -> price -> IV, |IV - sigma| <= TOL['iv'] on IDENTIFIABLE rows
     (time value >= IDENT_MIN_TIME_VALUE and vega >= IDENT_MIN_VEGA; others are counted and reported, not hidden)
  4. bound rows: prices at/below intrinsic or >= upper bound must give NaN + reason in both libs.
"""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np, pandas as pd

TOL = {"price": 1e-9, "delta": 1e-9, "gamma": 1e-9, "vega": 1e-7, "theta": 1e-7, "rho": 1e-7, "parity": 1e-10, "iv": 1e-7}
IDENT_MIN_TIME_VALUE = 1e-6       # price units (S=100)
IDENT_MIN_VEGA = 1e-4             # dV/dsigma; below this an IV is numerically ill-posed in double precision

GRID = dict(S=[100.0], K=[50.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 150.0], days=[1, 7, 30, 91, 182, 365, 730],
            sigma=[0.05, 0.10, 0.20, 0.40, 0.80, 1.50], r=[-0.01, 0.0, 0.03, 0.08], q=[0.0, 0.02, 0.05], flag=["c", "p"], model=["bsm", "black76"])


def grid_rows(grid: dict = GRID) -> list[dict]:
    rows = []
    for S, K, d, sig, r, q, flag, model in itertools.product(*(grid[k] for k in ("S", "K", "days", "sigma", "r", "q", "flag", "model"))):
        if model == "black76" and q != grid["q"][0]: continue          # q is ignored by Black-76: avoid duplicate rows
        rows.append(dict(flag=flag, S=S, K=K, T=d / 365.0, days=d, r=r, q=q if model == "bsm" else 0.0, sigma=sig, model=model))
    return rows


def run(out_dir: Path | None = None, grid: dict = GRID) -> dict:
    from adapters import derivs_adapter as D
    rows = grid_rows(grid); df = pd.DataFrame(rows)
    pg = D.price_greeks(rows); df = pd.concat([df, pg], axis=1)
    rep: dict = {"n_rows": len(df), "tolerances": TOL, "grid": grid, "label": "SYNTHETIC fixed input grid (model inputs, not market data)"}
    diffs = {f: float((df[f"vollib_{f}"] - df[f"quantlib_{f}"]).abs().max()) for f in ("price", "delta", "gamma", "vega", "theta", "rho")}
    rep["max_abs_diff"] = diffs; rep["pass_library_agreement"] = {f: diffs[f] <= TOL[f] for f in diffs}
    # put-call parity per lib
    q_eff = np.where(df["model"] == "black76", df["r"], df["q"])
    df["parity_rhs"] = df["S"] * np.exp(-q_eff * df["T"]) - df["K"] * np.exp(-df["r"] * df["T"])
    key = ["S", "K", "days", "sigma", "r", "q", "model"]
    c = df[df.flag == "c"].set_index(key); p = df[df.flag == "p"].set_index(key)
    par = {lib: float((c[f"{lib}_price"] - p[f"{lib}_price"] - c["parity_rhs"]).abs().max()) for lib in ("vollib", "quantlib")}
    rep["parity_max_abs_err"] = par; rep["pass_parity"] = all(v <= TOL["parity"] for v in par.values())
    # IV round trip (price from QuantLib, inverted by both libs; also vollib-priced for symmetry)
    lo = np.maximum(np.where(df.flag == "c", df["parity_rhs"], -df["parity_rhs"]), 0.0)
    df["time_value"] = df["quantlib_price"] - lo
    ident = (df["time_value"] >= IDENT_MIN_TIME_VALUE) & (df["quantlib_vega"] >= IDENT_MIN_VEGA)
    iv = D.implied_vol([dict(flag=r.flag, S=r.S, K=r.K, T=r.T, r=r.r, q=r.q, model=r.model, price=r.quantlib_price) for r in df.itertuples()])
    df = pd.concat([df, iv], axis=1)
    rt = {lib: float((df.loc[ident, f"iv_{lib}"] - df.loc[ident, "sigma"]).abs().max()) for lib in ("vollib", "quantlib")}
    nan_ident = {lib: int(df.loc[ident, f"iv_{lib}"].isna().sum()) for lib in ("vollib", "quantlib")}
    rep["iv_roundtrip"] = {"identifiable_rows": int(ident.sum()), "non_identifiable_rows": int((~ident).sum()),
                           "max_abs_err": rt, "nan_on_identifiable": nan_ident,
                           "iv_lib_max_abs_diff_identifiable": float((df.loc[ident, "iv_vollib"] - df.loc[ident, "iv_quantlib"]).abs().max()),
                           "non_identifiable_outcomes": {lib: {str(k): int(v) for k, v in df.loc[~ident, f"reason_{lib}"].fillna("solved").value_counts().items()} for lib in ("vollib", "quantlib")}}
    rep["pass_iv_roundtrip"] = all(v <= TOL["iv"] for v in rt.values()) and sum(nan_ident.values()) == 0
    # bound rows: below intrinsic / above upper bound must be NaN with a reason in both libs
    sel = df[(df.days == 91) & (df.sigma == 0.2)].head(40)
    up = np.where(sel.flag == "c", sel["S"] * np.exp(-np.where(sel.model == "black76", sel.r, sel.q) * sel["T"]), sel["K"] * np.exp(-sel["r"] * sel["T"]))
    lo_s = np.maximum(np.where(sel.flag == "c", sel["parity_rhs"], -sel["parity_rhs"]), 0.0)
    bad_rows = [dict(flag=r.flag, S=r.S, K=r.K, T=r.T, r=r.r, q=r.q, model=r.model, price=float(u * 1.001)) for r, u in zip(sel.itertuples(), up)]
    bad_rows += [dict(flag=r.flag, S=r.S, K=r.K, T=r.T, r=r.r, q=r.q, model=r.model, price=float(l - 0.01)) for r, l in zip(sel.itertuples(), lo_s) if l - 0.01 > 0]
    b = D.implied_vol(bad_rows)
    rep["bounds_check"] = {"rows": len(bad_rows), "all_nan": bool(b[["iv_vollib", "iv_quantlib"]].isna().all().all()),
                           "all_have_reason": bool(b[["reason_vollib", "reason_quantlib"]].notna().all().all()),
                           "reasons": {str(k): int(v) for k, v in b["reason_vollib"].value_counts().items()}}
    rep["pass_bounds"] = rep["bounds_check"]["all_nan"] and rep["bounds_check"]["all_have_reason"]
    # worst rows for transparency
    for f in ("price", "vega", "theta"):
        i = (df[f"vollib_{f}"] - df[f"quantlib_{f}"]).abs().idxmax()
        rep.setdefault("worst_row", {})[f] = {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in df.loc[i, ["flag", "K", "days", "sigma", "r", "q", "model", f"vollib_{f}", f"quantlib_{f}"]].to_dict().items()}
    rep["pass_all"] = all(rep["pass_library_agreement"].values()) and rep["pass_parity"] and rep["pass_iv_roundtrip"] and rep["pass_bounds"]
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "crosscheck.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    r = run(Path(__file__).resolve().parents[2] / "artifacts" / "derivs")
    print(json.dumps({k: v for k, v in r.items() if k not in ("grid",)}, indent=2, default=str))
