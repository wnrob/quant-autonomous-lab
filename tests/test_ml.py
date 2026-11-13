"""Leakage / sanity tests for the ML agent. ALL fixtures here are SYNTHETIC (software tests only; never evidence)."""
import numpy as np, pandas as pd, pytest
from engine.agents import alpha as A
from engine.memory.store import Store
from engine.models import features as F, ml, monitor as M
from engine.models.registry import Registry
from engine.research import validation as V

CRIT = V.Criteria()

def synth(n=4000, k=12, seed=3, plant=0.0, rho=0.7):
    """SYNTHETIC universe. plant>0: each asset's daily drift is a monthly AR(1) (persistent), so past returns genuinely predict next month."""
    idx = pd.bdate_range("2005-01-03", periods=n); rng = np.random.default_rng(seed); per = idx.to_period("M"); months = pd.PeriodIndex(sorted(set(per)))
    mu = np.zeros((len(months), k))
    for m in range(1, len(months)): mu[m] = rho * mu[m - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, plant, k)
    drift = mu[np.searchsorted(months, per)]
    px = pd.DataFrame(100 * np.cumprod(1 + drift + rng.normal(0.0003, 0.01, (n, k)), axis=0), index=idx, columns=[f"E{i}" for i in range(k)])
    px.iloc[:1500, 9:] = np.nan
    return ml.dev_prices(px, CRIT)

def test_a_shuffled_labels_control_kills_the_signal(tmp_path):
    px = synth(plant=0.0012); st = Store(tmp_path / "a.db"); panel = F.build_panel(px); plan = ml.fold_plan(panel)
    real = ml.evaluate_ml(ml.spec_for("ridge"), px, CRIT, st, "d", record=False, panel=panel, plan=plan)["metrics"]
    specs = [ml.spec_for("ridge", s, "shuffle_labels") for s in range(4)]; runs = [ml.evaluate_ml(sp, px, CRIT, st, "d", record=False, panel=panel, plan=plan) for sp in specs]
    batch = {r["spec_hash"]: r["metrics"]["validation_active"]["sharpe"] / np.sqrt(252) for r in runs}      # all 4 shuffles are trials -> DSR deflates for picking the best
    shuf = [ml.evaluate_ml(sp, px, CRIT, st, "d", record=False, panel=panel, plan=plan, batch=batch)["metrics"] for sp in specs]
    assert real["validation_active"]["sharpe"] > 1.0                                 # the planted signal is there ...
    assert np.mean([abs(m["validation_active"]["sharpe"]) for m in shuf]) < 0.6       # ... and vanishes with shuffled training labels
    assert max(m["validation"]["dsr"] for m in shuf) < CRIT.min_dsr

@pytest.mark.parametrize("k_me", [85, 110, 135, 160])
def test_b_future_price_perturbation_cannot_change_features_or_predictions(k_me):
    px = synth(plant=0.0012); me = px.groupby(px.index.to_period('M')).tail(1).index; T = me[k_me]   # T IS a month-end DECISION date, so 1-3 day look-ahead peeks are visible (review finding)
    px2 = px.copy(); px2[px2.index > T] *= 5.0
    p1, p2 = F.build_panel(px), F.build_panel(px2); a, b = p1[p1.index.get_level_values("date") <= T], p2[p2.index.get_level_values("date") <= T]
    cols = F.BASE_FEATURES + F.CS_FEATURES
    pd.testing.assert_frame_equal(a[cols], b[cols])
    s = []
    for p in (p1, p2):
        plan = ml.fold_plan(p); plan = [f for f in plan if f["test_dates"][-1] <= T]          # folds fully at/before T
        sc, _, _ = ml.walk_forward(p, lambda: ml.make_model("hgb", 0), plan); s.append(sc)
    pd.testing.assert_frame_equal(s[0], s[1])
    # labels of dates whose horizon crosses T legitimately change; labels fully before T do not
    ok = a["label_end_pos"] <= px.index.get_loc(T); pd.testing.assert_series_equal(a.loc[ok, "y"], b.loc[ok, "y"])

def test_c_purge_no_training_label_overlaps_evaluation_fold():
    panel = F.build_panel(synth()); plan = ml.fold_plan(panel, embargo_td=5); assert len(plan) >= 5; naive_violations = 0
    for f in plan:
        tr = panel[f["train_mask"]]; te_pos = panel[panel.index.get_level_values("date").isin(f["test_dates"])]["pos"]
        assert tr["label_end_pos"].max() < te_pos.min() - 5                           # no training horizon reaches the eval period (+embargo)
        assert tr["pos"].max() < te_pos.min() and tr["y"].notna().all()
        naive_violations += int(panel[panel["pos"] < f["first_test_pos"]]["label_end_pos"].max() >= f["first_test_pos"])
    assert naive_violations > 0                                                       # without purging, horizons WOULD leak: purge is doing work

def test_d_test_window_guard_raises():
    px_all = synth(); idx = pd.bdate_range("2005-01-03", periods=5600); full = pd.DataFrame(100.0, index=idx, columns=px_all.columns[:10]) * (1 + 0.0001 * np.arange(len(idx)))[:, None]
    assert full.index.max() > pd.Timestamp(CRIT.splits.val_end)
    with pytest.raises(ml.TestWindowAccess): ml.guard_dev(full)
    with pytest.raises(ml.TestWindowAccess): ml.guard_dev(full.index)
    with pytest.raises(ml.TestWindowAccess): ml.compute_run(ml.spec_for("ridge"), full, CRIT)
    with pytest.raises(ml.TestWindowAccess): ml.run_scores(pd.DataFrame(), full)
    assert ml.dev_prices(full).index.max() <= pd.Timestamp(CRIT.splits.val_end)

def test_e_pure_noise_universe_yields_no_candidate_and_no_validation_pass(tmp_path):
    st = Store(tmp_path / "n.db"); reg = Registry(st, tmp_path / "m"); px = synth(plant=0.0, seed=11)
    L = ml.run_ladder(st, px, "noise", CRIT, reg, n_seeds=3)
    assert all(r["decision"] == "VALIDATION_FAIL" for r in L["results"].values())
    assert st.q("SELECT COUNT(*) n FROM strategies WHERE status='CANDIDATE'")[0]["n"] == 0 and not reg.list("CANDIDATE")
    assert st.test_looks() == 0                                                       # protected test never consumed by this module
    assert L["n_trials"] == 2 + 2 * 3 and st.q("SELECT COUNT(*) n FROM lessons")[0]["n"] == L["n_trials"]

def test_f_planted_signal_is_recovered_by_the_ladder(tmp_path):
    st = Store(tmp_path / "p.db"); px = synth(plant=0.0012, seed=5); L = ml.run_ladder(st, px, "planted", CRIT, None, n_seeds=2); r = L["results"]
    assert r["ml-bench_mom_12_1-s0"]["metrics"]["validation_active"]["ann_active_return"] > 0    # sanity: the plant is momentum-like
    for k in ("ml-ridge-s0", "ml-hgb-s0", "ml-mlp-s0"):
        m = r[k]["metrics"]; assert m["validation_active"]["ann_active_return"] > 0.02 and m["validation_active"]["sharpe"] > 0.8, (k, m["validation_active"])
        assert m["diagnostic_rank_ic"]["mean"] > 0.05
    assert r["ml-ridge-s0"]["metrics"]["validation"]["dsr"] > 0.9

def test_run_scores_matches_alpha_run_xs_for_momentum():
    px = synth(); panel = F.build_panel(px); spec = {"family": "xs_factor", "universe": "t", "kind": "momentum", "lookback_days": 231, "skip_days": 21, "vol_adj": False, "top_frac": 0.33, "min_assets": 8, "cost_bps": 5.0}
    net_a, W_a, _ = A.run_xs(spec, px); sc = panel["z_mom_12_1"].unstack("asset"); net_m, W_m, _ = ml.run_scores(sc, px)
    common = net_a.index.intersection(net_m.index)[300:]; assert len(common) > 1000
    # selected sets agree except for assets whose complete-feature set differs on a date; require near-identical returns
    assert (net_a[common] - net_m[common]).abs().mean() < 5e-4

def test_g_registry_status_machine_and_dedup(tmp_path):
    st = Store(tmp_path / "r.db"); reg = Registry(st, tmp_path / "models")
    a, created = reg.register("m", "ml_xs", {"x": 1}, "fh", "ds", "cv", 0, metrics={"s": 1.0}, model={"w": [1, 2]}); b, created2 = reg.register("m", "ml_xs", {"x": 1}, "fh", "ds", "cv", 0)
    assert created and not created2 and a == b and reg.get(a)["status"] == "IDEA" and reg.load_artifact(a) == {"w": [1, 2]} and (tmp_path / "models").exists()
    assert reg.register("m", "ml_xs", {"x": 1}, "fh", "ds", "cv", 1)[1]                    # different seed -> new row
    with pytest.raises(ValueError): reg.set_status(a, "CANDIDATE", "skip")                  # illegal jump
    for s in ("RESEARCH", "VALIDATION", "CANDIDATE"): reg.set_status(a, s, "ok")
    with pytest.raises(PermissionError): reg.set_status(a, "SHADOW", "no")                  # nothing above CANDIDATE
    reg.set_status(a, "REJECTED", "bad"); assert not reg.advance(a, "RESEARCH", "resurrect") and reg.get(a)["status"] == "REJECTED"
    with pytest.raises(ValueError): reg.set_status(a, "IDEA", "resurrect")
    with pytest.raises(ValueError): reg.set_metrics(a, {"x": 1})                            # terminal: frozen
    assert len(st.q("SELECT * FROM ml_status_history WHERE model_id=?", a)) == 5

def test_h_psi_and_trigger_on_constructed_shift():
    rng = np.random.default_rng(0); base = rng.normal(0, 1, 5000)
    assert M.psi(base, rng.normal(0, 1, 2000)) < 0.05 and M.psi(base, rng.normal(1.5, 1, 2000)) > M.PSI_ALERT
    tr = pd.DataFrame({"a": base, "b": rng.normal(size=5000)}); same = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    shifted = same.assign(a=same["a"] + 2.0)
    assert not M.retrain_trigger(M.feature_psi(tr, same))["recommend_retrain"]
    t = M.retrain_trigger(M.feature_psi(tr, shifted)); assert t["recommend_retrain"] and "PSI" in t["reasons"][0]
    ic = pd.Series(np.r_[np.full(40, 0.10), np.full(12, -0.05)]); d = M.rolling_degradation(ic)
    assert M.retrain_trigger(None, d)["recommend_retrain"] and not M.retrain_trigger(None, M.rolling_degradation(pd.Series(np.full(52, 0.1))))["recommend_retrain"]
    assert M.retrain_trigger(model_age_months=16)["recommend_retrain"] and not M.retrain_trigger(model_age_months=12)["recommend_retrain"]
    s = M.seed_instability([1.0, 0.0, 0.9, -0.2, 0.5]); assert s["unstable"] and M.retrain_trigger(seed_stats=s)["blocks_promotion"] and not M.retrain_trigger(seed_stats=s)["recommend_retrain"]
    assert not M.seed_instability([0.5, 0.52, 0.49, 0.51, 0.5])["unstable"]

def test_labels_are_forward_and_demeaned():
    px = synth(); p = F.build_panel(px); d = p.index.get_level_values("date")[1200]; x = p.xs(d, level="date"); i = px.index.get_loc(d)
    fwd = pd.Series({a: px[a].iloc[i + 21] / px[a].iloc[i] - 1 for a in x.index}); np.testing.assert_allclose(x["y"].to_numpy(), (fwd - fwd.mean()).to_numpy(), atol=1e-12)
    assert abs(x["z_ret_21"].mean()) < 1e-9 and p["label_end_pos"].sub(p["pos"]).eq(21).all()


def test_one_day_lag_is_applied_in_run_scores():
    """Review finding: removing .shift(1) in run_scores survived. A huge return on the DECISION day itself must not be earned."""
    idx = pd.bdate_range("2010-01-04", periods=300); rng = np.random.default_rng(0)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.001, (300, 10)), axis=0), index=idx, columns=[f"E{i}" for i in range(10)])
    d = idx[100]; px.loc[d:, "E0"] *= 1.5                                         # +50% jump ON the decision day d, level held afterwards
    sc = pd.DataFrame(np.nan, index=[d], columns=px.columns); sc.loc[d, "E0"] = 1.0; sc.loc[d, [c for c in px.columns if c != "E0"]] = 0.0
    net, W, _ = ml.run_scores(sc, px, top_frac=0.1, min_assets=8, cost_bps=0.0)
    assert d not in W.index or W.loc[d].abs().sum() == 0                          # no position on the decision day itself
    assert W.index[0] == idx[101] and W.loc[idx[101], "E0"] > 0                  # position starts the NEXT day
    assert d not in net.index or abs(net.loc[d]) < 1e-12                          # jump on d is NOT earned
    assert net.abs().max() < 0.05                                                 # nothing like +50% anywhere
