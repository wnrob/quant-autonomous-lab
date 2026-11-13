"""Runs INSIDE envs/ml. qlib expression features + qlib's own LGBModel / LinearModel, purged walk-forward fit/predict.
Modes: "features" -> feature/label panel parquet;  "fit_predict" -> month-end score parquet.
DEV-WINDOW GUARD (defense in depth, independent of the core): raises if the data end, any calendar date, or any fold date is after VAL_END."""
import json, sys
import numpy as np, pandas as pd
VAL_END = pd.Timestamp("2019-12-31")

class TestWindowAccess(RuntimeError): pass

def _guard(*dates, what="dates"):
    for d in dates:
        if len(d) and pd.Timestamp(max(d)) > VAL_END: raise TestWindowAccess(f"{what} reach {pd.Timestamp(max(d)).date()} > {VAL_END.date()}: protected test window")

class Shim:
    """Minimal DatasetH stand-in exposing prepare(segment, col_set, data_key) as qlib models expect."""
    def __init__(self, segs): self.segs = segs; self.segments = {k: None for k in segs}
    def prepare(self, key, col_set="feature", data_key=None):
        df = self.segs[key]
        if isinstance(col_set, str): return df[col_set]
        return df

def _panel(a):
    import qlib
    from qlib.config import REG_US
    from qlib.data import D
    qlib.init(provider_uri=a["provider_uri"], region=REG_US, kernels=1, expression_cache=None, dataset_cache=None)
    cal = [pd.Timestamp(x) for x in D.calendar(start_time=a["start"], end_time=a["end"], freq="day")]
    _guard(cal, what="qlib calendar"); _guard([pd.Timestamp(a["end"])], what="end")
    names = list(a["exprs"]); df = D.features(D.instruments("all"), [a["exprs"][n] for n in names] + [a["label_expr"]], start_time=a["start"], end_time=a["end"], freq="day")
    df.columns = names + ["__label__"]; df = df.swaplevel().sort_index(); df.index.names = ["date", "symbol"]
    return df, names

def _cs_panel(df, names, dates, min_assets):
    d = df[df.index.get_level_values("date").isin(dates)]; d = d.dropna(subset=names)
    d = d[d.groupby(level="date")[names[0]].transform("size") >= min_assets]
    g = d.groupby(level="date"); X = (d[names] - g[names].transform("mean")) / g[names].transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
    y = d["__label__"] - d.groupby(level="date")["__label__"].transform("mean")
    X = X.fillna(0.0).clip(-5, 5); return X, y

def _fit(model_kind, params, seed, Xtr, ytr):
    import qlib.contrib.model.gbdt as _g
    class _NoRecorder:                                                                          # qlib's R.log_metrics needs an mlflow tracking dir; metrics are not used here
        def log_metrics(self, *a, **k): pass
    _g.R = _NoRecorder(); LGBModel = _g.LGBModel
    from qlib.contrib.model.linear import LinearModel
    def seg(X, y): 
        out = X.copy(); out.columns = pd.MultiIndex.from_product([["feature"], X.columns]); out[("label", "LABEL0")] = y.to_numpy(); return out
    if model_kind == "qlib_linear":
        m = LinearModel(estimator="ridge", alpha=params["alpha"], fit_intercept=True); m.fit(Shim({"train": seg(Xtr, ytr).dropna()})); return m
    dts = sorted(set(Xtr.index.get_level_values("date"))); cut = dts[int(len(dts) * 0.85)]          # early-stopping split INSIDE the (already purged) training rows
    is_v = Xtr.index.get_level_values("date") >= cut; ysc = float(ytr.std(ddof=0)) or 1.0
    p = {k: v for k, v in params.items() if k not in ("num_boost_round", "early_stopping_rounds")}
    m = LGBModel(loss="mse", early_stopping_rounds=params["early_stopping_rounds"], num_boost_round=params["num_boost_round"], seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                 num_threads=1, deterministic=True, **p)
    m.fit(Shim({"train": seg(Xtr[~is_v], ytr[~is_v] / ysc).dropna(), "valid": seg(Xtr[is_v], ytr[is_v] / ysc).dropna()}), verbose_eval=0)
    return m

def main():
    a = json.loads(sys.argv[1]); df, names = _panel(a)
    sample = [pd.Timestamp(x) for x in a["sample_dates"]]; _guard(sample, what="sample dates")
    if a["mode"] == "features":
        d = df[df.index.get_level_values("date").isin(sample)].reset_index(); d.to_parquet(a["out"]); print(json.dumps({"rows": len(d), "cols": names})); return
    folds, rows = a["folds"], []; info = []
    for i, f in enumerate(folds):
        tr_d, te_d = [pd.Timestamp(x) for x in f["train_dates"]], [pd.Timestamp(x) for x in f["test_dates"]]; _guard(tr_d, te_d, what=f"fold {i}")
        Xtr, ytr = _cs_panel(df, names, tr_d, a["min_assets"]); ytr = ytr.dropna(); Xtr = Xtr.loc[ytr.index]
        if a.get("shuffle_seed") is not None:                                                  # CONTROL: permute training labels within date
            rng = np.random.default_rng(a["shuffle_seed"] + i); ytr = ytr.groupby(level="date").transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        m = _fit(a["model"], a["params"], a["seed"], Xtr, ytr)
        Xte, _ = _cs_panel(df, names, te_d, a["min_assets"])
        if len(Xte):
            if a["model"] == "qlib_linear": pr = pd.Series(Xte.to_numpy() @ m.coef_ + m.intercept_, index=Xte.index)
            else: pr = pd.Series(m.model.predict(Xte.to_numpy()), index=Xte.index)
            rows.append(pr)
        info.append({"fold": i, "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "best_iter": int(getattr(getattr(m, "model", None), "best_iteration", 0) or 0)})
    sc = pd.concat(rows).rename("score").reset_index(); sc.to_parquet(a["out"]); print(json.dumps({"rows": len(sc), "folds": info}))

if __name__ == "__main__": main()
