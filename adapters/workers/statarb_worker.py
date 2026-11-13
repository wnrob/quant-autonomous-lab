"""Runs INSIDE envs/statarb (py3.11). Engle-Granger cointegration for every pair of columns on the FORMATION window only.
Hedge vector + ADF statistic: arbitragelab.cointegration_approach.EngleGrangerPortfolio (arbitragelab exposes critical values, not p-values).
p-value: statsmodels.tsa.stattools.coint (MacKinnon), pre-registered trend='c', maxlag=10, autolag='aic'. If arbitragelab fails to import, an OLS hedge from
statsmodels replaces it and engine="statsmodels" is reported. Dev-window guard: raises if any date is after VAL_END or after formation_end is used.
Input: argv[1] JSON {"npz": path, "formation_end": "YYYY-MM-DD", "min_obs": int, "out": path}. npz: dates (str), symbols, px (T x N float64, NaN allowed)."""
import json, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
VAL_END = pd.Timestamp("2019-12-31")
class TestWindowAccess(RuntimeError): pass

def check_dates(dates, fe):
    """Dev-window guard (importable for boundary tests): prices must end <= 2019-12-31, formation_end <= 2014-12-31."""
    dates = pd.DatetimeIndex(dates)
    if len(dates) and dates.max() > VAL_END: raise TestWindowAccess(f"prices reach {dates.max().date()} > {VAL_END.date()}: protected test window")
    if pd.Timestamp(fe) > pd.Timestamp("2014-12-31"): raise TestWindowAccess(f"formation_end {pd.Timestamp(fe).date()} is after the train window end 2014-12-31")

def main():
    a = json.loads(sys.argv[1]); z = np.load(a["npz"], allow_pickle=False); dates = pd.DatetimeIndex(z["dates"]); syms = [str(s) for s in z["symbols"]]; px = z["px"]
    fe = pd.Timestamp(a["formation_end"]); check_dates(dates, fe)
    from statsmodels.tsa.stattools import coint
    engine = "arbitragelab"
    try:
        if a.get("engine") == "statsmodels": raise ImportError("statsmodels engine requested explicitly (fast path for noise controls)")
        from arbitragelab.cointegration_approach import EngleGrangerPortfolio
    except Exception as e:                                                                       # fallback is stated, not silent
        engine = "statsmodels"; EngleGrangerPortfolio = None; err = repr(e)
    lp = pd.DataFrame(np.log(px), index=dates, columns=syms).loc[:fe]                           # formation window ONLY
    rows = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            d = lp[[syms[i], syms[j]]].dropna()
            if len(d) < a["min_obs"]: rows.append({"a": syms[i], "b": syms[j], "n_obs": int(len(d)), "beta": None, "adf_stat": None, "pvalue": None}); continue
            if EngleGrangerPortfolio is not None:
                e = EngleGrangerPortfolio(); e.fit(d, add_constant=True); beta = -float(e.cointegration_vectors.iloc[0, 1]); adf = float(e.adf_statistics.loc["statistic_value"].iloc[0])
            else:
                x = np.column_stack([np.ones(len(d)), d.iloc[:, 1].to_numpy()]); c = np.linalg.lstsq(x, d.iloc[:, 0].to_numpy(), rcond=None)[0]; beta = float(c[1]); adf = None
            p = float(coint(d.iloc[:, 0], d.iloc[:, 1], trend="c", maxlag=10, autolag="aic")[1])
            rows.append({"a": syms[i], "b": syms[j], "n_obs": int(len(d)), "beta": beta, "adf_stat": adf, "pvalue": p})
    json.dump({"engine": engine, "formation_end": str(fe.date()), "pairs": rows}, open(a["out"], "w")); print(json.dumps({"engine": engine, "n_pairs": len(rows)}))
if __name__ == "__main__": main()
