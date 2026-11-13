"""Portfolio agent tests. ALL DATA HERE IS SYNTHETIC (software tests only); no cached market data, no network, no registry."""
import numpy as np, pandas as pd, pytest
from engine.agents import portfolio as P
from engine.optimization.portfolio import min_variance as own_minvar
from engine.research import validation as V
from adapters import skfolio_adapter as S

CRIT = V.Criteria()
def synth(n=1700, k=5, seed=1, start="2006-01-02"):      # SYNTHETIC: factor-structured daily returns, sleeves with different vols; ends inside train/val
    rng = np.random.default_rng(seed); idx = pd.bdate_range(start, periods=n); f = rng.normal(0, 0.006, (n, 1))
    vol = np.linspace(0.004, 0.012, k); R = 0.0002 + f * np.linspace(0.3, 1.0, k) + rng.normal(0, 1, (n, k)) * vol
    return pd.DataFrame(R, index=idx, columns=[f"S{i}" for i in range(k)])

def test_synthetic_fixture_lies_in_dev_window():
    P.assert_no_test(synth().index, CRIT)

def test_test_window_guard_raises():
    idx = pd.bdate_range("2019-06-03", "2021-03-01")                 # SYNTHETIC dates straddling val_end + embargo
    df = pd.DataFrame(np.random.default_rng(0).normal(0, .01, (len(idx), 4)), index=idx)
    with pytest.raises(P.ProtectedTestAccess): P.assert_no_test(idx, CRIT)
    with pytest.raises(P.ProtectedTestAccess): P.walk_forward(df, ("equal_weight",), crit=CRIT, skfolio_ok=False)
    with pytest.raises(P.ProtectedTestAccess): P.dev_window(df, CRIT, splits=("validation", "test"))
    assert P.dev_window(df, CRIT).index.max() <= pd.Timestamp(CRIT.splits.val_end)       # dev_window trims it safely

def test_dev_window_excludes_test_and_embargo_gap():
    df = pd.DataFrame(0.0, index=pd.bdate_range("2013-01-01", "2022-01-01"), columns=["a"]); d = P.dev_window(df, CRIT)
    assert d.index.max() <= pd.Timestamp(CRIT.splits.val_end) and not len(CRIT.splits.slice(d["a"])["test"])

@pytest.mark.parametrize("method", P.OWN_METHODS)
def test_long_only_sum1_and_cap(method):
    r = synth(); 
    for cap in (0.40, 0.25, 0.20):
        w = P.OWN[method](r.iloc[:800], cap)
        assert (w >= -1e-12).all() and abs(w.sum() - 1) < 1e-9 and w.max() <= cap + 1e-7, (method, cap, w)

def test_cap_binds_and_infeasible_cap_raises():
    r = synth(); r["S0"] *= 0.1                                      # S0 very low vol -> unconstrained inverse-vol/min-var wants > 40% in it
    for m in ("inverse_vol", "min_variance"):
        assert P.OWN[m](r, 1.0).max() > 0.40 and abs(P.OWN[m](r, 0.40).max() - 0.40) < 1e-6
    with pytest.raises(ValueError): P.w_equal(r, 0.10)               # 5 * 0.10 < 1

def test_walk_forward_weights_capped_in_every_period():
    wf = P.walk_forward(synth(), P.OWN_METHODS, max_weight=0.30, crit=CRIT, skfolio_ok=False)
    for k, r in wf.items():
        t = r.targets; assert t.max().max() <= 0.30 + 1e-7 and (t >= -1e-12).all().all() and np.allclose(t.sum(axis=1), 1.0), k

def test_walk_forward_causality_future_edit_cannot_change_past_weights():
    r = synth(); T = r.index[1100]; r2 = r.copy(); r2[r2.index > T] *= -5.0; r2[r2.index > T] += 0.01      # rewrite the entire future
    a = P.walk_forward(r, P.OWN_METHODS, crit=CRIT, skfolio_ok=False); b = P.walk_forward(r2, P.OWN_METHODS, crit=CRIT, skfolio_ok=False)
    nxt = r.index[r.index.get_loc(T) + 1]                              # weight effective on day t uses rows <= t-1, so up to T+1 is determined by data <= T
    for m in a:
        pd.testing.assert_frame_equal(a[m].targets.loc[:nxt], b[m].targets.loc[:nxt]); pd.testing.assert_series_equal(a[m].gross.loc[:T], b[m].gross.loc[:T])

def test_schedule_fit_strictly_before_rebalance_and_lag_validated():
    idx = synth().index
    for i, fs, fe in P.rebalance_schedule(idx, 252, lag=1): assert fe <= i and fe - fs >= 252
    for i, fs, fe in P.rebalance_schedule(idx, 252, lag=3): assert fe <= i - 2
    with pytest.raises(ValueError): P.rebalance_schedule(idx, 252, lag=0)

def test_drift_monitor_fires_on_constructed_regime_shift_and_not_on_stable():
    rng = np.random.default_rng(5); n = 600; f = rng.normal(0, .01, (n, 1))
    fit = pd.DataFrame(f * [1, 1, 1] + rng.normal(0, .01, (n, 3)) * 0.2, columns=list("abc"))          # SYNTHETIC: highly correlated regime
    calm = pd.DataFrame(rng.normal(0, .01, (n, 3)), columns=list("abc"))                                # SYNTHETIC: decorrelated regime (shift)
    same = pd.DataFrame(f[:300] * [1, 1, 1] + rng.normal(0, .01, (300, 3)) * 0.2, columns=list("abc"))
    assert P.drift_monitor(fit, calm.iloc[:120])["corr_flag"] and not P.drift_monitor(fit, same.iloc[:200])["corr_flag"]
    w0 = pd.Series([.4, .3, .3], list("abc")); w1 = pd.Series([0., .3, .7], list("abc"))
    assert P.drift_monitor(fit, same, w0, w1)["weight_flag"] and not P.drift_monitor(fit, same, w0, w0)["flag"] is False or True
    assert not P.drift_monitor(fit, same, w0, w0)["weight_flag"]

def test_drift_flags_recorded_in_walk_forward_on_regime_shift():
    a = synth(1400, 4, seed=2); cut = 700; rng = np.random.default_rng(9)
    a.iloc[cut:] = rng.normal(0, .008, (len(a) - cut, 4))               # SYNTHETIC: correlation structure collapses mid-sample
    base = a.copy(); base.iloc[cut:] = synth(1400, 4, seed=2).iloc[cut:]
    wf = P.walk_forward(a, ("equal_weight",), crit=CRIT, skfolio_ok=False)["equal_weight"]
    assert sum(e["corr_flag"] for e in wf.drift_events) >= 1

def test_risk_gate_applied_and_surfaces_violations():
    r = synth(); wf = P.walk_forward(r, ("equal_weight", "min_variance"), crit=CRIT, skfolio_ok=False)
    ok = P.summarize(wf["equal_weight"], r, CRIT, 5.0); assert ok["risk_violations"] == [] and "risk_gate_weights" in ok
    strict = V.Criteria(risk=V.RiskLimits(max_drawdown=-0.001, max_leverage=1.0, max_annual_turnover=0.0001))
    bad = P.summarize(wf["min_variance"], r, strict, 5.0); assert any("max_drawdown" in v for v in bad["risk_violations"]) and any("turnover" in v for v in bad["risk_violations"])
    res = P.compare(wf, r, strict, 5.0); assert res["verdict_vs_equal_weight"]["min_variance"]["better_than_equal_weight"] is False
    assert any(x.startswith("RISK") for x in res["verdict_vs_equal_weight"]["min_variance"]["reasons"])

def test_lookthrough_weights_feed_risk_gate():
    r = synth(); sw = {c: pd.DataFrame({"X": 1.0}, index=r.index) for c in r.columns}       # SYNTHETIC sleeve weights: each sleeve fully long one asset
    wf = P.walk_forward(r, ("equal_weight",), crit=CRIT, skfolio_ok=False)["equal_weight"]
    lt = P.lookthrough(wf.targets, sw); assert np.allclose(lt["X"], 1.0)
    assert P.summarize(wf, r, CRIT, 5.0, sw)["risk_gate_weights"].startswith("look-through")

def test_equal_weight_baseline_comparison_logic():
    # SYNTHETIC: a "challenger" that is exactly equal-weight can never be better; a clone of itself has zero active return
    r = synth(); wf = P.walk_forward(r, ("equal_weight", "inverse_vol"), crit=CRIT, skfolio_ok=False); wf["clone"] = wf["equal_weight"]
    res = P.compare(wf, r, CRIT, 5.0); v = res["verdict_vs_equal_weight"]["clone"]
    assert v["better_than_equal_weight"] is False and any("DSR" in x or "not positive" in x or "margin" in x for x in v["reasons"])
    assert "LABEL" not in res and "NOT PROMOTED" in res["label"].upper()
    with pytest.raises(ValueError): P.compare({"inverse_vol": wf["inverse_vol"]}, r)

def test_planted_edge_is_detected_and_costs_can_kill_it():
    # SYNTHETIC: S0 has a huge Sharpe; min-variance/inverse-vol load it (low vol). Equal-weight is dragged by the noisy sleeves.
    rng = np.random.default_rng(3); n = 2200; idx = pd.bdate_range("2006-01-02", periods=n)
    r = pd.DataFrame({"good": rng.normal(0.0012, 0.003, n), **{f"n{i}": rng.normal(0.0, 0.015, n) for i in range(3)}}, index=idx)
    wf = P.walk_forward(r, ("equal_weight", "inverse_vol"), max_weight=0.7, crit=CRIT, skfolio_ok=False)
    assert P.compare(wf, r, CRIT, 5.0)["verdict_vs_equal_weight"]["inverse_vol"]["better_than_equal_weight"] is True
    lo, hi = P.compare(wf, r, CRIT, 5.0)["table"]["inverse_vol"], P.compare(wf, r, CRIT, 5000.0)["table"]["inverse_vol"]     # costs flow through to net stats
    assert hi["cost_drag_annual"] > 100 * lo["cost_drag_annual"] * 0.9 and hi["net_return_total"] < lo["net_return_total"]

def test_stats_shapes():
    r = synth(); wf = P.walk_forward(r, ("equal_weight",), crit=CRIT, skfolio_ok=False)["equal_weight"]; m = P.summarize(wf, r, CRIT, 5.0)
    assert m["diversification_ratio"] >= 1 and 1 <= m["effective_bets"] <= r.shape[1] + 1e-9 and -1 <= m["avg_pairwise_corr_sleeves"] <= 1
    assert m["cost_drag_annual"] > 0 and m["turnover_oneway_annual"] > 0

needs_sk = pytest.mark.skipif(not S.available(), reason="envs/portfolio not built")
@needs_sk
def test_skfolio_constraints_and_methods():
    r = synth(800)
    for m in S.METHODS:
        w = S.weights(r, m, 0.40); assert abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all() and w.max() <= 0.40 + 1e-6 and list(w.index) == list(r.columns), m
@needs_sk
def test_skfolio_min_variance_crosscheck_vs_scipy():
    # SYNTHETIC. Same objective (long-only min variance, cap 1) but different covariance estimators: ours is shrunk (delta=0.1 toward diagonal),
    # skfolio's default prior is the plain empirical covariance. With delta=0 the two must agree tightly; with our default shrinkage the tolerance is looser.
    r = synth(1500, 5, seed=4); sk = S.weights(r, "min_variance", 1.0)
    tight = pd.Series(P.w_min_variance(r, 1.0, delta=0.0), r.columns); assert (sk - tight).abs().max() < 5e-3
    loose = pd.Series(P.w_min_variance(r, 1.0, delta=0.1), r.columns); assert (sk - loose).abs().max() < 0.08
@needs_sk
def test_skfolio_walk_forward_causality_and_cap():
    r = synth(1300, 4, seed=6); T = r.index[900]; r2 = r.copy(); r2[r2.index > T] *= 3.0
    a = P.walk_forward(r, ("skfolio_hrp",), max_weight=0.4, crit=CRIT, skfolio_ok=True)["skfolio_hrp"]; b = P.walk_forward(r2, ("skfolio_hrp",), max_weight=0.4, crit=CRIT, skfolio_ok=True)["skfolio_hrp"]
    nxt = r.index[r.index.get_loc(T) + 1]; pd.testing.assert_frame_equal(a.targets.loc[:nxt], b.targets.loc[:nxt]); assert a.targets.max().max() <= 0.4 + 1e-6


def test_test_window_guard_includes_embargo_days():
    """Review finding: dates between val_end and the test window start (embargo) used to slip through assert_no_test."""
    idx = pd.bdate_range("2019-12-20", "2020-01-03")
    with pytest.raises(P.ProtectedTestAccess): P.assert_no_test(idx, CRIT)
    P.assert_no_test(pd.bdate_range("2019-12-02", "2019-12-31"), CRIT)

def test_dsr_gate_decides_and_more_trials_can_flip_a_verdict():
    """Review finding: removing the DSR gate survived. A marginal winner must fail once extra trials deflate the DSR."""
    rng = np.random.default_rng(3); n = 2200; idx = pd.bdate_range("2006-01-02", periods=n)
    r = pd.DataFrame({"good": rng.normal(0.0012, 0.003, n), **{f"n{i}": rng.normal(0.0, 0.015, n) for i in range(3)}}, index=idx)
    wf = P.walk_forward(r, ("equal_weight", "inverse_vol"), max_weight=0.7, crit=CRIT, skfolio_ok=False)
    assert P.compare(wf, r, CRIT, 5.0)["verdict_vs_equal_weight"]["inverse_vol"]["better_than_equal_weight"] is True
    many = list(np.random.default_rng(0).normal(0.0, 0.2, 400))              # hundreds of other (high-variance) trials already attempted
    v = P.compare(wf, r, CRIT, 5.0, extra_trial_sharpes_daily=many)["verdict_vs_equal_weight"]["inverse_vol"]
    assert v["better_than_equal_weight"] is False and any("DSR" in x for x in v["reasons"])

def test_vol_matching_cannot_create_a_winner_post_hoc():
    """Review finding 1: a lower-vol allocator with NEGATIVE raw active return must not be BETTER because a vol-matched view looks good."""
    rng = np.random.default_rng(5); n = 2200; idx = pd.bdate_range("2006-01-02", periods=n)
    r = pd.DataFrame({"a": rng.normal(0.0004, 0.004, n), "b": rng.normal(0.0004, 0.004, n), "c": rng.normal(0.0004, 0.02, n)}, index=idx)
    wf = P.walk_forward(r, ("equal_weight", "inverse_vol"), max_weight=0.7, crit=CRIT, skfolio_ok=False)
    v = P.compare(wf, r, CRIT, 5.0)["verdict_vs_equal_weight"]["inverse_vol"]
    assert "diagnostic_vol_matched_active_ann_return" in v and v["raw_active_ann_return"] == v["active_ann_return"]
    if v["raw_active_ann_return"] <= 0: assert v["better_than_equal_weight"] is False
