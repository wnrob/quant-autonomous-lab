"""Options adapter (core side; NO heavy imports). Black-Scholes-Merton / Black-76 prices, Greeks and implied vol from
py_vollib AND QuantLib, computed in the isolated envs/derivs venv by adapters/workers/derivs_worker.py.

Contract (all functions take a list of row dicts and return a DataFrame aligned to the input order):
  price_greeks(rows)  rows: flag ('c'|'p', strict), S (spot, or forward F for model='black76'), K, T (days/365, >0),
                      r, q (cont. comp.), sigma (>0), model ('bsm'|'black76').
                      -> columns vollib_{price,delta,gamma,vega,theta,rho}, quantlib_{...}
  implied_vol(rows)   same but `price` instead of sigma -> iv_vollib, iv_quantlib, reason_vollib, reason_quantlib.
                      Price outside model-free no-arbitrage bounds -> NaN + reason (never a clamped vol).
  american(rows)      QuantLib CRR binomial American price (labelled extra) next to European analytic + European CRR.
Units: vega per 1.00 vol, theta per year, rho per 1.00 rate (see worker docstring). Invalid inputs raise
engine.derivs.checks.InvalidOptionInput here, BEFORE any subprocess is started; the worker re-validates.
"""
from __future__ import annotations
import json, math, tempfile
from pathlib import Path
import pandas as pd
from adapters.envrun import ROOT, run_in_env
from engine.derivs import checks as C

ENV = "derivs"; WORKER = "derivs_worker.py"
FIELDS = ("price", "delta", "gamma", "vega", "theta", "rho")
LIBS = ("vollib", "quantlib")


def available() -> bool:
    return (ROOT / "envs" / ENV / "bin" / "python").exists()


def _call(op: str, payload: dict, timeout: int = 900) -> dict:
    with tempfile.TemporaryDirectory(prefix="qal_derivs_") as d:
        pf = Path(d) / "payload.json"; rf = Path(d) / "result.json"
        pf.write_text(json.dumps(payload, allow_nan=False))
        run_in_env(ENV, WORKER, {"op": op, "payload_file": str(pf), "result_file": str(rf)}, timeout=timeout)
        return json.loads(rf.read_text())


def _nan(x):
    return float("nan") if x is None else x


def _rows_for_pricing(rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        model = row.get("model", "bsm")
        try:
            flag, S, K, T, r, q, sig = C.validate(row.get("flag"), row.get("S"), row.get("K"), row.get("T"), row.get("r", 0.0),
                                                  row.get("q", 0.0), row.get("sigma"), model)
        except C.InvalidOptionInput as e:
            raise C.InvalidOptionInput(f"row {i}: {e}") from None
        if sig is None: raise C.InvalidOptionInput(f"row {i}: sigma is required for pricing")
        out.append(dict(flag=flag, S=S, K=K, T=T, r=r, q=row.get("q", 0.0) if model == "bsm" else 0.0, sigma=sig, model=model))
    return out


def price_greeks(rows: list[dict]) -> pd.DataFrame:
    res = _call("price_greeks", {"rows": _rows_for_pricing(rows)})["results"]
    if len(res) != len(rows): raise RuntimeError("derivs worker returned wrong number of rows")
    return pd.DataFrame([{f"{lib}_{f}": _nan(r[lib][f]) for lib in LIBS for f in FIELDS} for r in res])


def implied_vol(rows: list[dict]) -> pd.DataFrame:
    clean = []
    for i, row in enumerate(rows):
        model = row.get("model", "bsm")
        try:
            C.iv_precheck(row.get("price"), row.get("flag"), row.get("S"), row.get("K"), row.get("T"), row.get("r", 0.0), row.get("q", 0.0), model)
        except C.InvalidOptionInput as e:
            raise C.InvalidOptionInput(f"row {i}: {e}") from None
        p = row.get("price")
        clean.append(dict(flag=row["flag"], S=float(row["S"]), K=float(row["K"]), T=float(row["T"]), r=float(row.get("r", 0.0)),
                          q=float(row.get("q", 0.0)), model=model, price=float(p) if isinstance(p, (int, float)) and math.isfinite(p) else None))
    res = _call("iv", {"rows": clean})["results"]
    if len(res) != len(rows): raise RuntimeError("derivs worker returned wrong number of rows")
    return pd.DataFrame([{"iv_vollib": _nan(r["vollib"]), "iv_quantlib": _nan(r["quantlib"]),
                          "reason_vollib": r["reason_vollib"], "reason_quantlib": r["reason_quantlib"]} for r in res])


def american(rows: list[dict]) -> pd.DataFrame:
    clean = [{**r, "model": "bsm"} for r in _rows_for_pricing(rows)]
    for c, r in zip(clean, rows): c["steps"] = int(r.get("steps", 801))
    return pd.DataFrame(_call("american", {"rows": clean})["results"])


def pickle_scan(path: str | Path) -> dict:
    """Static opcode scan in envs/derivs (the file is never unpickled)."""
    return _call("pickle_scan", {"path": str(path)})


def qqq_chains_iv(path: str | Path, spot_path: str | Path, out: str | Path, r: float, q: float) -> pd.DataFrame:
    """Restricted-unpickle the cached QQQ call histories IN envs/derivs and compute IV with both libs (descriptive)."""
    _call("qqq_chains", {"path": str(path), "spot_path": str(spot_path), "out": str(out), "r": r, "q": q})
    return pd.read_parquet(out)


def cboe_snapshots_iv(paths: list[str], out: str | Path, **filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    _call("cboe", {"paths": [str(p) for p in paths], "out": str(out), **filters}, timeout=1800)
    return pd.read_parquet(out), pd.read_parquet(str(out).replace(".parquet", "_forwards.parquet"))
