"""Read-only audit of the user's local futures data (/Users/newman/quantalpha/data). DESCRIPTIVE ONLY.

The data (2024-08..2026-08) lies entirely in the lab's PROTECTED TEST ERA (after 2019-12-31 + embargo). This module
runs no strategy, no backtest, no selection, and never touches the experiment store or its test_access ledger.
It writes only to artifacts/futures/.

    envs/core/bin/python -m engine.futures.real_data [--data-root /Users/newman/quantalpha/data] [--out artifacts/futures]
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np, pandas as pd
from engine.futures import calendar as cal
from engine.futures.rules import CalendarRoll
from engine.futures.specs import SPECS, specs_table
from engine.futures.stale import detect_stale, resolve_contracts

LAB = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/Users/newman/quantalpha/data")
RULE = CalendarRoll(n_bdays=5, ref="spec")          # reference rule for the 'past roll date' check; fixed before looking


def _robust_sigma(x: pd.Series) -> float:
    x = x.dropna(); return float(1.4826 * np.median(np.abs(x - x.median()))) if len(x) else float("nan")


def session_label_to_trade_date(labels: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """HEURISTIC. The user's daily bars are stamped with the vendor's session window_start date (UTC midnight), which for
    CME evening-open sessions is the calendar day BEFORE the trade date (only Sun-Thu labels occur; MES label 2024-09-12
    closes at 5626.50 = the 2024-09-13 session). Map Sun-Thu labels to the next business day; Fri/Sat labels (grain
    daytime-only sessions after a holiday, a partial final day) are kept as-is. Collisions are reported, never merged."""
    out = []
    for t in pd.DatetimeIndex(labels):
        d = t.date()
        out.append(d if d.weekday() in (4, 5) else cal.add_business_days(d, 1))
    return pd.DatetimeIndex(pd.to_datetime(out))


def audit_daily(path: str, panel_end: pd.Timestamp) -> tuple[dict, list[dict]]:
    root = Path(path).name.split("_")[0]
    raw = pd.read_parquet(path)
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    labels = pd.DatetimeIndex(raw.index)
    wd = pd.Series(labels.day_of_week).value_counts().sort_index()
    td = session_label_to_trade_date(labels)
    collisions = int(td.duplicated().sum()); nonbiz = int(sum(not cal.is_business_day(d.date()) for d in td))
    df = raw.copy(); df.index = td
    idx = pd.DatetimeIndex(df.index)
    cid = resolve_contracts(df["contract"], root)
    span = [d.date() for d in pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D")]
    bdays = [d for d in span if cal.is_business_day(d)]
    have = set(idx.date)
    missing_bdays = [d for d in bdays if d not in have]
    lr = np.log(df["close"].astype(float)).diff()
    switch = df["contract"].ne(df["contract"].shift(1)) & df["contract"].shift(1).notna()
    sig = _robust_sigma(lr[~switch]); sd = float(lr[~switch].std())
    rolls = []
    for i in np.flatnonzero(switch.to_numpy()):
        a, b = cid.iloc[i - 1], cid.iloc[i]
        d = idx[i].date(); ref_a = a.roll_reference(); lt_a = a.last_trade()
        g = float(lr.iloc[i])
        dt_days = (b.last_trade() - lt_a).days
        rolls.append({"root": root, "label": str(labels[i].date()), "trade_date": str(d), "from": str(a), "to": str(b), "gap_logret": g, "gap_z": g / sig if sig > 0 else np.nan,
                      "bdays_before_last_trade": cal.business_days_between(d, lt_a),
                      "bdays_before_roll_ref": cal.business_days_between(d, ref_a), "roll_ref": str(ref_a),
                      "days_between_expiries": dt_days})
    rr = pd.DataFrame(rolls)
    rep = detect_stale(df, root, rule=RULE, panel_end=panel_end)
    out = {"root": root, "file": Path(path).name, "rows": len(df), "columns": list(df.columns), "index_tz": str(idx.tz),
           "first_label": str(labels[0].date()), "last_label": str(labels[-1].date()),
           "first": str(idx[0].date()), "last": str(idx[-1].date()), "date_basis": "trade date via session_label_to_trade_date (heuristic)",
           "trade_date_collisions": collisions, "non_business_trade_dates": nonbiz, "contracts_in_meta": meta.get("contracts"),
           "contracts_in_series": list(dict.fromkeys(df["contract"])), "one_contract_per_row": True,
           "label_weekday_counts_mon0": {int(k): int(v) for k, v in wd.items()},
           "sunday_rows": int(wd.get(6, 0)), "saturday_rows": int(wd.get(5, 0)),
           "missing_business_days": len(missing_bdays), "missing_examples": [str(d) for d in missing_bdays[:8]],
           "n_rolls": len(rr), "robust_sigma_daily": sig, "sd_nonroll_daily": sd,
           "rolls_abs_z_gt_3": int((rr["gap_z"].abs() > 3).sum()) if len(rr) else 0,
           "rolls_abs_z_gt_2": int((rr["gap_z"].abs() > 2).sum()) if len(rr) else 0,
           "median_abs_gap_logret": float(rr["gap_logret"].abs().median()) if len(rr) else None,
           "roll_bdays_before_last_trade": rr["bdays_before_last_trade"].tolist() if len(rr) else [],
           "roll_bdays_before_roll_ref": rr["bdays_before_roll_ref"].tolist() if len(rr) else [],
           "stale": rep.summary()}
    # Roll-gap PROXY for the calendar spread. gap = ln new(t) - ln old(t-1) = spread(t-1) + market move of new over (t-1,t].
    # The market move has mean ~0 and sd ~ sd_nonroll, so mean(gap) is an unbiased but NOISY spread estimate. Not carry.
    if len(rr) >= 2:
        ann = -rr["gap_logret"] * 365.0 / rr["days_between_expiries"]     # sign: + = backwardation (like roll_yield_ann)
        n = len(rr); se = sd * 365.0 / rr["days_between_expiries"].mean() / np.sqrt(n)
        out["roll_gap_proxy"] = {"n_rolls": n, "mean_gap_logret": float(rr["gap_logret"].mean()),
                                 "implied_ann_roll_yield_mean": float(ann.mean()), "noise_se_ann": float(se),
                                 "ci95": [float(ann.mean() - 1.96 * se), float(ann.mean() + 1.96 * se)],
                                 "distinguishable_from_zero": bool(abs(ann.mean()) > 1.96 * se),
                                 "caveat": "PROXY ONLY: each gap mixes the calendar spread with one day of market noise; not a carry measurement"}
    # Effect on a naive pct_change of the unadjusted splice (what qa/trend.py:returns does)
    r = df["close"].astype(float).pct_change()
    out["naive_pctchange_roll_days"] = {"n": int(switch.sum()), "sum_abs_roll_day_ret": float(r[switch].abs().sum()),
                                        "max_abs_roll_day_ret": float(r[switch].abs().max()) if switch.any() else None,
                                        "share_of_total_abs_ret": float(r[switch].abs().sum() / r.abs().sum())}
    return out, rolls


def audit_intraday(path: str) -> dict:
    root = Path(path).name.split("_")[0]
    df = pd.read_parquet(path)
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    rep = detect_stale(df, root, rule=CalendarRoll(n_bdays=5, ref="spec"), thin_segment_min_rows=390)   # 1m: one full RTH session
    return {"file": Path(path).name, "root": root, "rows": len(df), "first": str(df.index[0]), "last": str(df.index[-1]),
            "contracts": list(dict.fromkeys(df["contract"])), "rolls_in_meta": meta.get("rolls"), "stale": rep.summary()}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", default=str(DEFAULT_DATA)); ap.add_argument("--out", default=str(LAB / "artifacts" / "futures"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    daily = sorted(glob.glob(os.path.join(a.data_root, "daily", "*_1d_*.parquet")))
    intraday = sorted(glob.glob(os.path.join(a.data_root, "cache", "*_1m_*.parquet")))
    lasts = [session_label_to_trade_date(pd.read_parquet(p, columns=["close"]).index)[-1] for p in daily]
    panel_end = max(lasts)
    d_res, all_rolls = [], []
    for p in daily:
        r, rolls = audit_daily(p, panel_end); d_res.append(r); all_rolls += rolls
    i_res = [audit_intraday(p) for p in intraday]
    pd.DataFrame(all_rolls).to_csv(out / "roll_events_daily.csv", index=False)
    (out / "specs_table.json").write_text(json.dumps(specs_table(), indent=2))
    res = {"generated_by": "engine.futures.real_data", "data_root": a.data_root, "panel_end": str(panel_end),
           "protected_era_note": "all data is after 2019-12-31 (lab protected-test era): descriptive statistics only; no strategy evaluated; store untouched",
           "reference_roll_rule": {"rule": str(RULE), "digest": RULE.digest()}, "daily": d_res, "intraday": i_res}
    (out / "real_data_audit.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    r = main()
    for d in r["daily"]:
        s = d["stale"]["counts"]
        print(f"{d['root']:4} rows={d['rows']:4} {d['first']}..{d['last']} rolls={d['n_rolls']:2} |z|>3={d['rolls_abs_z_gt_3']:2} "
              f"back={s.get('back_switch',0)} pastLT={s.get('past_last_trade',0)} pastRoll={s.get('past_roll_date',0)} "
              f"staleEnd={s.get('days_before_panel_end')}d sun={d['sunday_rows']} missBD={d['missing_business_days']} ok={d['stale']['ok']}")
    for d in r["intraday"]:
        s = d["stale"]["counts"]
        print(f"{d['file']:42} rows={d['rows']:7} back={s.get('back_switch',0)} multi={s.get('multi_contract_timestamp',0)} "
              f"pastLT={s.get('past_last_trade',0)} pastRoll={s.get('past_roll_date',0)} fwd={s.get('forward_switches')} thin={s.get('thin_volume_frac')} ok={d['stale']['ok']}")
