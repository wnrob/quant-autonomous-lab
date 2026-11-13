"""Descriptive implied-vol statistics on the REAL options data found under /Users/newman/quantalpha/data (read-only).

Every dataset here is dated after 2019-12-31, i.e. inside the lab's PROTECTED era. Allowed use: descriptive statistics
and library cross-checks only. `require_descriptive` raises for any other purpose (strategy selection, backtest,
signal research); it is called by every entry point. Nothing here imports or writes the experiment store, so the
protected-test ledger is never touched.

Pre-stated sampling/filter rules (fixed before any IV was computed; changes must be disclosed):
  Cboe snapshots : one snapshot per session date = the status=='usable' snapshot whose source_event_ts is closest to
                   17:00 UTC; quotes with bid>0, ask>=bid, (ask-bid)/mid <= 0.5, 7 <= DTE <= 400; OTM options only
                   (calls K>=F, puts K<F); Black-76 on the mid; T = calendar DTE/365 (intraday hours ignored).
                   Forward, v1 (pre-stated, then ABANDONED - disclosed): F and DF per expiry from OLS of (C_mid - P_mid) on
                   K over strikes within +-5% of spot. It rejected 501/630 expiries (median fitted DF 1.005 > 1: the
                   slope is steeper than -1, as expected when ITM American puts carry early-exercise premium), so
                   European parity regression is invalid on these American options. v1 output is kept in
                   artifacts/derivs/realdata_iv_v1_parity_regression.json.
                   Forward, v2 (used): F = K* + (C-P)(K*) / DF at the strike K* minimising |C_mid - P_mid| (>= 3 two-sided
                   strikes within +-5%), DF = exp(-r T) with r ASSUMED = 4.5% (the provenance code's RISK_FREE constant,
                   not chosen from these results); sensitivity re-run at r = 3.5% is reported. |F/S-1| <= 10%.
  qqq_chains.pkl : every daily bar; BSM call on the bar close vs the QQQ daily close of the same date, with the
                   provenance script's own conventions (r = 4.5%, q = 0, T = DTE/365).
  ATM IV         : linear interpolation of IV in k = ln(K/F) to k = 0 between the nearest OTM strikes on either side.
  30d ATM IV     : linear interpolation of total variance (iv^2 * T) in T between the expiries bracketing 30 DTE.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from engine.research.validation import Splits

DATA = Path("/Users/newman/quantalpha/data")
SNAP_DIR = DATA / "external_flow" / "iv_snapshots"
CHAINS_PKL = DATA / "cache" / "qqq_chains.pkl"
QQQ_DAILY = DATA / "cache" / "QQQ_daily.parquet"
PROTECTED_AFTER = pd.Timestamp(Splits().val_end)          # 2019-12-31; test window = after this + embargo
ALLOWED_PURPOSES = ("descriptive",)
TARGET_UTC_HOUR = 17


class ProtectedEraUse(PermissionError):
    """Protected-era (post-2019-12-31) options data requested for something other than descriptive statistics."""


def require_descriptive(purpose: str, dates) -> None:
    d = pd.to_datetime(pd.Index(list(dates)), utc=True).tz_localize(None) if len(dates) else pd.DatetimeIndex([])
    n_prot = int((d > PROTECTED_AFTER).sum())
    if purpose not in ALLOWED_PURPOSES and n_prot:
        raise ProtectedEraUse(f"{n_prot} protected-era dates (> {PROTECTED_AFTER.date()}) requested for purpose={purpose!r}; "
                              f"only {ALLOWED_PURPOSES} is allowed on this data")


def select_snapshots(snap_dir: Path = SNAP_DIR) -> pd.DataFrame:
    rows = []
    for m in sorted(snap_dir.glob("*/*.meta.json")):
        meta = json.loads(m.read_text())
        raw = m.with_name(m.name.replace(".meta.json", ".raw.json"))
        rows.append(dict(day=m.parent.name, ts=pd.Timestamp(meta["source_event_ts"]), status=meta["status"], raw=str(raw), exists=raw.exists()))
    df = pd.DataFrame(rows)
    ok = df[(df.status == "usable") & df.exists].copy()
    ok["dist"] = (ok.ts - ok.ts.dt.normalize() - pd.Timedelta(hours=TARGET_UTC_HOUR)).abs()
    pick = ok.sort_values(["day", "dist", "ts"]).groupby("day").head(1).sort_values("ts")
    return df, pick


def atm_iv(g: pd.DataFrame, col: str) -> float:
    """IV at k=0 from OTM rows of one expiry (put side k<0, call side k>=0)."""
    g = g.dropna(subset=[col]); k = np.log(g.K / g.F)
    below, above = g[k < 0], g[k >= 0]
    if below.empty or above.empty: return float("nan")
    kb, ka = np.log(below.K.max() / below.F.iloc[0]), np.log(above.K.min() / above.F.iloc[0])
    vb, va = below.loc[below.K.idxmax(), col], above.loc[above.K.idxmin(), col]
    return float(vb + (va - vb) * (0 - kb) / (ka - kb))


def iv_at_k(g: pd.DataFrame, col: str, k0: float) -> float:
    g = g.dropna(subset=[col]).sort_values("K"); k = np.log(g.K / g.F).to_numpy()
    if len(k) < 2 or not (k.min() <= k0 <= k.max()): return float("nan")
    return float(np.interp(k0, k, g[col].to_numpy()))


def term_interp(atm: pd.DataFrame, days: int = 30) -> float:
    a = atm.dropna(subset=["atm_iv"]).sort_values("dte")
    lo, hi = a[a.dte <= days], a[a.dte >= days]
    if lo.empty or hi.empty: return float("nan")
    l, h = lo.iloc[-1], hi.iloc[0]
    if l.dte == h.dte: return float(l.atm_iv)
    wl, wh = l.atm_iv ** 2 * l.dte, h.atm_iv ** 2 * h.dte
    return float(math.sqrt((wl + (wh - wl) * (days - l.dte) / (h.dte - l.dte)) / days))


def _q(s: pd.Series) -> dict:
    s = s.dropna()
    return {"n": int(len(s)), "median": float(s.median()) if len(s) else None, "p95": float(s.quantile(0.95)) if len(s) else None,
            "p99": float(s.quantile(0.99)) if len(s) else None, "max": float(s.max()) if len(s) else None}


def cboe_study(scratch: Path, purpose: str = "descriptive", r_fixed: float = 0.045, tag: str = "") -> dict:
    from adapters import derivs_adapter as D
    allsnap, pick = select_snapshots()
    require_descriptive(purpose, pick.ts)
    rows, fwd = D.cboe_snapshots_iv(list(pick.raw), scratch / f"cboe_iv{tag}.parquet", forward_method="atm_pair", r_fixed=r_fixed)
    rows = rows[rows.root == "QQQ"].copy()
    both = rows.dropna(subset=["iv_vollib", "iv_quantlib"])
    d_lib = (both.iv_vollib - both.iv_quantlib).abs()
    cb = rows.dropna(subset=["iv_vollib"]); cb = cb[cb.cboe_iv > 0]
    d_cboe = cb.iv_vollib - cb.cboe_iv
    per_day, smiles = [], []
    for (snap, date), g in rows.groupby(["snapshot", "date"]):
        atm = pd.DataFrame([dict(expiry=e, dte=int(h.dte.iloc[0]), atm_iv=atm_iv(h, "iv_vollib")) for e, h in g.groupby("expiry")])
        iv30 = term_interp(atm, 30)
        near = atm.dropna().iloc[(atm.dropna().dte - 30).abs().argsort()[:1]] if atm.dropna().size else pd.DataFrame()
        sm = {}
        if len(near):
            h = g[g.expiry == near.expiry.iloc[0]]
            sm = {f"k{k:+.2f}": iv_at_k(h, "iv_vollib", k) for k in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10)}
            smiles.append(dict(date=date, expiry=near.expiry.iloc[0], dte=int(near.dte.iloc[0]), **sm))
        a = atm.dropna().sort_values("dte")
        per_day.append(dict(date=date, ts=g.ts.iloc[0], spot=float(g.spot.iloc[0]), cboe_iv30=float(g.iv30.iloc[0]) / 100 if g.iv30.iloc[0] is not None else None,
                            atm_iv_30d=iv30, atm_iv_front=float(a.atm_iv.iloc[0]) if len(a) else None, front_dte=int(a.dte.iloc[0]) if len(a) else None,
                            atm_iv_longest=float(a.atm_iv.iloc[-1]) if len(a) else None, longest_dte=int(a.dte.iloc[-1]) if len(a) else None,
                            n_expiries=int(len(a)), n_quotes=int(len(g))))
    pdf, sdf = pd.DataFrame(per_day), pd.DataFrame(smiles)
    fwd_ok = fwd[fwd.status == "ok"]
    rep = {
        "label": "REAL data, protected era (2026). DESCRIPTIVE ONLY: no strategy selection or backtest.",
        "source": "Cboe delayed-quote chain snapshots (cdn.cboe.com delayed_quotes/options/QQQ.json) collected by /Users/newman/quantalpha/collect_iv_walls.py",
        "coverage": {"snapshots_on_disk": int(len(allsnap)), "usable": int((allsnap.status == "usable").sum()), "session_dates": int(allsnap.day.nunique()),
                     "first": str(allsnap.ts.min()), "last": str(allsnap.ts.max()), "selected_snapshots": int(len(pick)),
                     "selected_dates": sorted(pick.day.unique().tolist()),
                     "expiries_per_snapshot_median": float(pdf.n_expiries.median()), "quotes_used": int(len(rows)),
                     "history": "intraday snapshots over ~7 weeks of one underlying; NOT a multi-year history"},
        "forward_fits": {"method": "atm_pair (v2)", "r_assumed": r_fixed, "expiry_fits": int(len(fwd)), "ok": int(len(fwd_ok)),
                         "status_counts": {str(k): int(v) for k, v in fwd.status.value_counts().items()},
                         "median_F_over_S_minus_1_dte_ge_120": float((fwd_ok[fwd_ok.dte >= 120].F / fwd_ok[fwd_ok.dte >= 120].spot - 1).median()),
                         "diag_parity_regression_DF_median": float(fwd.DF_regression_diag.median()),
                         "diag_parity_regression_share_DF_gt_1": float((fwd.DF_regression_diag > 1).mean())},
        "library_agreement": {"rows_solved_by_both": int(len(both)), "abs_iv_diff": _q(d_lib),
                              "nan_reasons_vollib": {str(k): int(v) for k, v in rows.reason.fillna("solved").value_counts().items()},
                              "only_one_lib_solved": int((rows.iv_vollib.isna() ^ rows.iv_quantlib.isna()).sum())},
        "vs_cboe_reported_iv": {"note": "Cboe's own model/rate/dividend conventions are undocumented in the payload (American-style options); descriptive comparison only",
                                "rows": int(len(d_cboe)), "ours_minus_cboe": {"median": float(d_cboe.median()), "mad": float((d_cboe - d_cboe.median()).abs().median()),
                                                                              "p05": float(d_cboe.quantile(.05)), "p95": float(d_cboe.quantile(.95))}},
        "atm_iv_30d": {"median": float(pdf.atm_iv_30d.median()), "min": float(pdf.atm_iv_30d.min()), "max": float(pdf.atm_iv_30d.max()),
                       "corr_with_cboe_iv30": float(pdf[["atm_iv_30d", "cboe_iv30"]].dropna().corr().iloc[0, 1]),
                       "median_ours_minus_cboe_iv30": float((pdf.atm_iv_30d - pdf.cboe_iv30).median())},
        "term_structure": {"median_front_minus_longest_atm_iv": float((pdf.atm_iv_front - pdf.atm_iv_longest).median()),
                           "share_days_upward_sloping_30d_to_longest": float(((pdf.atm_iv_longest > pdf.atm_iv_30d)).mean()),
                           "median_front_dte": float(pdf.front_dte.median()), "median_longest_dte": float(pdf.longest_dte.median())},
        "smile_nearest_30d": {c: float(sdf[c].median()) for c in sdf.columns if c.startswith("k")} if len(sdf) else {},
        "smile_skew_k-0.10_minus_k+0.05_median": float((sdf["k-0.10"] - sdf["k+0.05"]).median()) if len(sdf) else None,
    }
    pdf.to_csv(scratch / f"cboe_per_day{tag}.csv", index=False); sdf.to_csv(scratch / f"cboe_smiles{tag}.csv", index=False)
    return rep


def chains_study(scratch: Path, purpose: str = "descriptive") -> dict:
    from adapters import derivs_adapter as D
    spot = pd.read_parquet(QQQ_DAILY)
    require_descriptive(purpose, spot.index)                 # guard BEFORE any worker call
    scan = D.pickle_scan(CHAINS_PKL)
    if scan["disallowed"] or scan["problems"]:
        return {"loaded": False, "scan": scan, "reason": "pickle references non-allowlisted globals; NOT loaded"}
    df = D.qqq_chains_iv(CHAINS_PKL, QQQ_DAILY, scratch / "qqq_chains_iv.parquet", r=0.045, q=0.0)
    require_descriptive(purpose, df.date)
    both = df.dropna(subset=["iv_vollib", "iv_quantlib"])
    w = both[(both.dte >= 20) & (both.dte <= 45)].copy(); w["m"] = (w.strike / w.spot - 1).abs()
    atm = w.sort_values(["date", "m"]).groupby("date").first()
    return {
        "label": "REAL data, protected era (2024-08..2026-08). DESCRIPTIVE ONLY.",
        "source": "Massive/Polygon v2 daily aggregates of OCC call contracts, fetched by /Users/newman/quantalpha/fetch_options.py (2 near-ATM strikes per monthly expiry)",
        "pickle_safety": {"globals": scan["globals"], "disallowed": scan["disallowed"], "problems": scan["problems"], "loaded_in": "envs/derivs via RestrictedUnpickler"},
        "coverage": {"contracts": int(df.groupby(["expiry", "strike"]).ngroups), "expiries": int(df.expiry.nunique()),
                     "first_expiry": df.expiry.min(), "last_expiry": df.expiry.max(), "bar_dates": [df.date.min(), df.date.max()],
                     "bars": int(len(df)), "puts": 0, "bid_ask": False,
                     "history": "daily trade OHLCV of calls only (no puts, no quotes, no OI); 2 strikes per monthly expiry"},
        "conventions": "BSM call, r=0.045, q=0 (provenance script's choice; ignores QQQ's ~0.5-0.6% dividend), T=DTE/365, option CLOSE vs underlying CLOSE (non-synchronous)",
        "library_agreement": {"rows_solved_by_both": int(len(both)), "abs_iv_diff": _q((both.iv_vollib - both.iv_quantlib).abs()),
                              "nan_reasons": {str(k): int(v) for k, v in df.reason.fillna("solved").value_counts().items()}},
        "atm_iv_dte20_45": {"days": int(len(atm)), "median": float(atm.iv_vollib.median()), "p10": float(atm.iv_vollib.quantile(.1)), "p90": float(atm.iv_vollib.quantile(.9)),
                            "provenance_reported_median": 0.1929, "note": "provenance (qa/options.atm_iv_series) also drops IV outside (0.01, 3); we do not filter solved IVs"},
    }


def main(purpose: str = "descriptive") -> dict:
    root = Path(__file__).resolve().parents[2]
    scratch = root / "artifacts" / "qae_scratch_derivs"; scratch.mkdir(parents=True, exist_ok=True)
    out = root / "artifacts" / "derivs"; out.mkdir(parents=True, exist_ok=True)
    rep = {"qqq_chains_pkl": chains_study(scratch, purpose), "cboe_snapshots": cboe_study(scratch, purpose)}
    alt = cboe_study(scratch, purpose, r_fixed=0.035, tag="_r035")
    rep["cboe_snapshots"]["sensitivity_r_3.5pct"] = {"atm_iv_30d_median": alt["atm_iv_30d"]["median"],
        "delta_vs_r_4.5pct": alt["atm_iv_30d"]["median"] - rep["cboe_snapshots"]["atm_iv_30d"]["median"],
        "smile_nearest_30d": alt["smile_nearest_30d"]}
    (out / "realdata_iv.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
