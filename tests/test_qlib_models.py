"""qlib_model family tests. ALL price fixtures are SYNTHETIC (software tests only; never evidence)."""
import re, subprocess
from pathlib import Path
import numpy as np, pandas as pd, pytest
from adapters import qlib_models_adapter as QA
from adapters.envrun import ROOT as LAB
from engine.errors import AdapterUnavailable
from engine.families import common, qlib_model as QM
from engine.families.common import TestWindowAccess
from engine.memory.store import Store, h
from engine.research import validation as V

CRIT = V.Criteria()
needs_ml = pytest.mark.skipif(not (LAB / "envs/ml/bin/python").exists(), reason="envs/ml not built")

def synth(n=4000, k=12, seed=3, plant=0.0, rho=0.7, full=False):
    """SYNTHETIC universe (labelled). plant>0: persistent monthly drift so past returns predict next month."""
    idx = pd.bdate_range("2005-01-03", periods=n); rng = np.random.default_rng(seed); per = idx.to_period("M"); months = pd.PeriodIndex(sorted(set(per))); mu = np.zeros((len(months), k))
    for m in range(1, len(months)): mu[m] = rho * mu[m - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, plant, k)
    px = pd.DataFrame(100 * np.cumprod(1 + mu[np.searchsorted(months, per)] + rng.normal(0.0003, 0.01, (n, k)), axis=0), index=idx, columns=[f"E{i}" for i in range(k)])
    return px if full else common.dev_prices(px, CRIT)

# ------------------------------------------------------------ env-exercising, fast
@needs_ml
def test_qlib_worker_features_match_pandas():
    px = synth(n=1800, k=9); p = QA.feature_panel(px)
    d = pd.Timestamp("2006-06-30"); d = px.index[px.index.get_indexer([d], method="pad")[0]]; me = p.index.get_level_values(0).unique(); d = me[me.get_indexer([d], method="pad")[0]]
    pos = px.index.get_loc(d); roc5 = (px.iloc[pos] / px.iloc[pos - 5] - 1); ma20 = px.iloc[pos - 19: pos + 1].mean() / px.iloc[pos]
    got = p.xs(d, level=0)
    np.testing.assert_allclose(got["ROC5"].reindex(px.columns).to_numpy(), roc5.to_numpy(), rtol=1e-4, atol=1e-6); np.testing.assert_allclose(got["MA20"].reindex(px.columns).to_numpy(), ma20.to_numpy(), rtol=1e-4)

# ------------------------------------------------------------ protected-window guards
def test_adapter_guard_raises_on_post_val_end_bar_without_launching_env():
    full = synth(full=True); assert full.index.max() > pd.Timestamp(CRIT.splits.val_end)
    for fn in (lambda: QA.feature_panel(full), lambda: QA.fit_predict(full, "qlib_linear", {"alpha": 1.0}, 0, [{"train_dates": ["2008-01-31"], "test_dates": ["2009-01-30"], "first_test_pos": 1}])):
        with pytest.raises(TestWindowAccess): fn()
    ok = synth(); late = [{"train_dates": ["2008-01-31"], "test_dates": ["2020-03-31"], "first_test_pos": 1}]
    with pytest.raises(TestWindowAccess): QA.fit_predict(ok, "qlib_linear", {"alpha": 1.0}, 0, late)        # a fold date past val_end is also rejected

def test_family_guard_dev_and_compute_run_raise(tmp_path):
    full = synth(full=True)
    with pytest.raises(TestWindowAccess): common.guard_dev(full)
    with pytest.raises(TestWindowAccess): QM.compute_run(QM.TRIALS[0], full, CRIT)
    with pytest.raises(TestWindowAccess): QM.run_family(Store(tmp_path / "g.db"), full, "d", CRIT)

@needs_ml
def test_worker_independent_guard_raises_even_if_core_is_bypassed(tmp_path):
    full = synth(full=True); qdir = QA.write_qlib_dir(full.iloc[:, :9], tmp_path / "q")
    with pytest.raises(AdapterUnavailable, match="TestWindowAccess"):
        from adapters.envrun import run_in_env
        run_in_env("ml", "qlib_model_worker.py", {"mode": "features", "provider_uri": str(qdir), "exprs": {"A": "$close"}, "label_expr": "$close", "start": str(full.index[0].date()), "end": str(full.index[-1].date()),
                   "out": str(tmp_path / "o.parquet"), "sample_dates": []})

# ------------------------------------------------------------ pre-registration / accounting (core only, stubbed predictions)
def stub_fit_predict(monkeypatch, plant=False):
    def fp(px, model, params, seed, plan, crit=CRIT, shuffle_seed=None, workdir=None):
        common.guard_dev(px, crit)
        rng = np.random.default_rng(seed + (7 if shuffle_seed else 0) + len(model)); te = pd.DatetimeIndex(sorted({d for f in plan for d in f["test_dates"]}))
        return pd.DataFrame(rng.normal(size=(len(te), px.shape[1])), index=te, columns=px.columns)
    monkeypatch.setattr(QM.QA, "fit_predict", fp)

def test_trial_list_is_pinned_and_changes_with_any_edit():
    assert len(QM.TRIALS) == 7 and QM.preregister(QM.TRIALS) == QM.TRIAL_LIST_HASH
    t2 = [dict(s) for s in QM.TRIALS]; t2[1] = {**t2[1], "params": {**t2[1]["params"], "learning_rate": 0.051}}; assert QM.preregister(t2) != QM.TRIAL_LIST_HASH

def test_run_family_counts_all_declared_trials_dedups_uses_frozen_criteria_and_never_touches_test(monkeypatch, tmp_path):
    stub_fit_predict(monkeypatch); st = Store(tmp_path / "s.db"); px = synth()
    def boom(*a, **k): raise AssertionError("record_test_access must never be called by engine.families")
    monkeypatch.setattr(Store, "record_test_access", boom)
    F = QM.run_family(st, px, "dsha", CRIT, trials=QM.TRIALS); assert F["n_trials"] == 7 and len(F["results"]) == 7
    ex = st.q("SELECT * FROM experiments"); assert len(ex) == 7 and {e["criteria_hash"] for e in ex} == {CRIT.digest()}
    import json
    assert all(json.loads(e["metrics"])["n_trials_family"] == 7 and json.loads(e["metrics"])["trial_list_hash"] == QM.TRIAL_LIST_HASH for e in ex)
    assert st.test_looks() == 0
    F2 = QM.run_family(st, px, "dsha", CRIT, trials=QM.TRIALS); assert all(r["duplicate"] for r in F2["results"].values()) and len(st.q("SELECT * FROM experiments")) == 7      # store dedup
    assert not re.search(r"record_test_access\s*\(", "".join(p.read_text() for p in Path(QM.__file__).parent.glob("*.py")) + Path(QA.__file__).read_text())

def test_dsr_deflates_with_more_trials(monkeypatch, tmp_path):
    stub_fit_predict(monkeypatch); px = synth(); st = Store(tmp_path / "a.db"); r = QM.evaluate(QM.TRIALS[0], px, CRIT, st, "d", record=False, plan=QA.fold_plan(px))
    sh = r["metrics"]["validation_active"]["sharpe"] / np.sqrt(252); rng = np.random.default_rng(0)
    d1 = V.deflated_sharpe(r["active"], [sh] + list(rng.normal(0, 0.03, 6))); d2 = V.deflated_sharpe(r["active"], [sh] + list(rng.normal(0, 0.03, 60))); assert d2 <= d1

# ------------------------------------------------------------ purge / no-lookahead (core)
def test_fold_plan_purges_label_horizon_and_embargo():
    px = synth(); plan = QA.fold_plan(px); assert len(plan) >= 5; naive = 0
    for f in plan:
        pos = px.index.get_indexer(pd.DatetimeIndex(f["train_dates"])); p0 = f["first_test_pos"]; assert (pos + QA.HORIZON).max() < p0 - 5
        naive += int(px.index.get_indexer(pd.DatetimeIndex([d for d in QA.sample_dates(px) if px.index.get_loc(d) < p0]))[-1] + QA.HORIZON >= p0)
    assert naive > 0                                                                     # without purging, horizons WOULD leak: purge does work

# ------------------------------------------------------------ slow: real qlib models, controls, look-ahead through the worker
@pytest.mark.slow
@needs_ml
def test_slow_shuffled_label_control_kills_planted_signal_and_future_perturbation_is_invisible():
    px = synth(plant=0.0012); plan = QA.fold_plan(px); params = QM.MODEL_PARAMS["qlib_linear"]
    def active_sharpe(shuf):
        sc = QA.fit_predict(px, "qlib_linear", params, 0, plan, shuffle_seed=shuf); from engine.models.ml import run_scores
        net, W, b = run_scores(sc, px, top_frac=0.33, min_assets=8, cost_bps=5.0, crit=None); v = CRIT.splits.slice(net)["validation"]; vb = CRIT.splits.slice(b)["validation"]; return V.sharpe((v - vb.reindex(v.index)).dropna())
    real = active_sharpe(None); shuf = [abs(active_sharpe(1000 + i)) for i in range(3)]
    assert real > 1.0 and np.mean(shuf) < 0.6
    me = QA.sample_dates(px); T = me[100]; px2 = px.copy(); px2[px2.index > T] *= 5.0; pl = [f for f in plan if pd.Timestamp(f["test_dates"][-1]) <= T]; assert pl
    pd.testing.assert_frame_equal(QA.fit_predict(px, "qlib_linear", params, 0, pl), QA.fit_predict(px2, "qlib_linear", params, 0, pl), rtol=1e-4, atol=1e-6)

@pytest.mark.slow
@needs_ml
def test_slow_lgb_runs_through_qlib_and_is_deterministic():
    px = synth(plant=0.0012, n=3200); plan = QA.fold_plan(px)[:2]; a = QA.fit_predict(px, "qlib_lgb_a", QM.LGB_A, 1, plan); b = QA.fit_predict(px, "qlib_lgb_a", QM.LGB_A, 1, plan); pd.testing.assert_frame_equal(a, b); assert a.shape[1] == px.shape[1]

# ------------------------------------------------------------ exact-boundary guards (reviewer finding) + real qlib model fit in the env
def _idx(*extra):
    return pd.DatetimeIndex(["2019-12-30", "2019-12-31", *extra])

def test_guard_dev_exact_boundary():
    common.guard_dev(_idx()); common.guard_dev(pd.DataFrame(1.0, index=_idx(), columns=["a"]))                         # ends exactly 2019-12-31: allowed
    for bad in ("2020-01-01", "2020-01-06"):
        with pytest.raises(TestWindowAccess): common.guard_dev(_idx(bad))
        with pytest.raises(TestWindowAccess): common.guard_dev(pd.DataFrame(1.0, index=_idx(bad), columns=["a"]))

def _env_py(snippet):
    r = subprocess.run([str(LAB / "envs/ml/bin/python"), "-I", "-W", "ignore", "-c", "import importlib.util as u;s=u.spec_from_file_location('w',r'%s');w=u.module_from_spec(s);s.loader.exec_module(w)\n%s" % (LAB / "adapters/workers/qlib_model_worker.py", snippet)], capture_output=True, text=True); return r

@needs_ml
def test_qlib_worker_guard_exact_boundary():
    ok = _env_py("import pandas as pd; w._guard([pd.Timestamp('2019-12-31')]); print('OK')"); assert ok.returncode == 0 and "OK" in ok.stdout, ok.stderr[-300:]
    for bad in ("2020-01-01", "2020-01-06"):
        r = _env_py(f"import pandas as pd; w._guard([pd.Timestamp('2019-12-31'), pd.Timestamp('{bad}')])"); assert r.returncode != 0 and "TestWindowAccess" in r.stderr

@needs_ml
def test_real_qlib_linearmodel_fit_predict_in_env_on_tiny_synthetic_panel():
    """SYNTHETIC tiny panel (labelled). Runs qlib's real LinearModel through the worker's shim: planted linear signal must be recovered (coef sign) - no stub."""
    r = _env_py("""
import numpy as np, pandas as pd
rng = np.random.default_rng(0); n = 600; X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list('abc')); y = pd.Series(2.0 * X['a'].to_numpy() + rng.normal(0, 0.1, n))
m = w._fit('qlib_linear', {'alpha': 1.0}, 0, X, y); print('COEF', [round(float(c), 2) for c in m.coef_])""")
    assert r.returncode == 0, r.stderr[-400:]
    coef = eval(r.stdout.split("COEF")[1]); assert abs(coef[0] - 2.0) < 0.1 and abs(coef[1]) < 0.1
