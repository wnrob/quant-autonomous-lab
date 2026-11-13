"""statarb_pairs family tests. ALL price fixtures are SYNTHETIC (software tests only; never evidence)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, pytest
from adapters import statarb_adapter as SA
from adapters.envrun import ROOT as LAB
from engine.families import common, statarb_pairs as SP
from engine.families.common import TestWindowAccess
from engine.memory.store import Store
from engine.research import validation as V

CRIT = V.Criteria()
needs_sa = pytest.mark.skipif(not (LAB / "envs/statarb/bin/python").exists(), reason="envs/statarb not built")
DATES = pd.bdate_range("2005-01-03", "2019-12-31")

def rw(cols, seed=0, n=None, vol=0.01, dates=DATES):
    rng = np.random.default_rng(seed); n = n or len(dates); return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, vol, (n, len(cols))), axis=0)), index=dates[:n], columns=cols)

def with_coint_pair(seed=1, noise=0.01, cols=("SPY", "QQQ", "IWM", "EFA")):
    """SYNTHETIC: QQQ = 0.8*SPY (log) + stationary AR(1) noise -> cointegrated by construction."""
    px = rw(list(cols), seed); rng = np.random.default_rng(seed + 50); e = np.zeros(len(px))
    for t in range(1, len(e)): e[t] = 0.9 * e[t - 1] + rng.normal(0, noise)
    px["QQQ"] = np.exp(0.8 * np.log(px["SPY"]) + e + 0.5); return px

# ------------------------------------------------------------ multiple-testing correction (pure)
def test_bh_known_vector_and_bonferroni_and_nan_counts_in_m():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert SP.bh_fdr(p, 0.05).tolist() == [True, True] + [False] * 8                  # thresholds i/10*0.05: .005,.01,.015 ... only k=2
    assert SP.bonferroni(p, 0.05).sum() == 1                                          # 0.001 <= 0.005 only
    assert SP.bh_fdr(np.array([0.009] + [np.nan] * 9), 0.05).sum() == 0              # m stays 10 with NaNs (0.009 > 1/10*0.05); with m=1 it would be rejected
    assert SP.bh_fdr(np.array([0.009]), 0.05).sum() == 1

def test_bh_false_positive_rate_under_the_null_is_at_most_q():
    rng = np.random.default_rng(0); hits = sum(SP.bh_fdr(rng.uniform(size=276), 0.05).any() for _ in range(4000)) / 4000     # all-null p-values: P(any BH rejection) = q exactly
    raw = np.mean([(rng.uniform(size=276) < 0.05).sum() for _ in range(200)]); assert hits < 0.075 and 12 < raw < 16          # BH ~5% of sims; naive p<0.05 yields ~13.8 false pairs per sim

# ------------------------------------------------------------ guards
def test_adapter_guard_raises_on_prices_after_val_end_and_formation_after_train_end():
    full = rw(["A", "B", "C"], dates=pd.bdate_range("2005-01-03", "2021-06-30"))
    with pytest.raises(TestWindowAccess): SA.coint_pairs(full)
    with pytest.raises(TestWindowAccess): SP.evaluate_pairs(full, CRIT, None, "d", record=False)
    with pytest.raises(TestWindowAccess): SA.coint_pairs(rw(["A", "B", "C"]), formation_end="2016-12-31")

@needs_sa
def test_worker_independent_guard_raises_even_if_core_is_bypassed(tmp_path):
    from adapters.envrun import run_in_env
    from engine.errors import AdapterUnavailable
    full = rw(["A", "B", "C"], dates=pd.bdate_range("2005-01-03", "2021-06-30")); npz = tmp_path / "x.npz"
    np.savez(npz, dates=np.array([str(d.date()) for d in full.index]), symbols=np.array(list(full.columns)), px=full.to_numpy(float))
    with pytest.raises(AdapterUnavailable, match="TestWindowAccess"): run_in_env("statarb", "statarb_worker.py", {"npz": str(npz), "formation_end": "2014-12-31", "min_obs": 750, "out": str(tmp_path / "o.json")})

# ------------------------------------------------------------ env-exercising (arbitragelab)
@needs_sa
def test_worker_detects_planted_pair_and_not_random_walks_using_arbitragelab():
    px = common.dev_prices(with_coint_pair(), CRIT); r = SA.coint_pairs(px.iloc[:, :3]); t = r["pairs"].set_index(["a", "b"])
    assert r["engine"] == "arbitragelab" and t.loc[("SPY", "QQQ"), "pvalue"] < 1e-3                  # planted cointegrated pair found
    assert abs(t.loc[("SPY", "QQQ"), "beta"] - 1 / 0.8) < 0.35 or abs(t.loc[("SPY", "QQQ"), "beta"] - 0.8) < 0.35   # hedge ratio recovered (either regression orientation)
    assert t.loc[("SPY", "IWM"), "pvalue"] > 0.01 and t.loc[("QQQ", "IWM"), "pvalue"] > 0.01
    assert (t["n_obs"] == len(px.loc[:"2014-12-31"])).all()                           # formation window only (nothing after 2014-12-31 was used)

# ------------------------------------------------------------ trade rules / no look-ahead / leverage
def test_positions_state_machine_enter_exit_stop():
    """SYNTHETIC spread. (The first version of this test used spikes so large that the stop fired on the day after entry; the code was right and the fixture's expectation was wrong - fixed here.)"""
    idx = pd.bdate_range("2015-01-01", periods=400); rng = np.random.default_rng(0); base = rng.normal(0, 0.1, 400); base[200:203] = [0.28, 0.12, 0.0]; base[300:303] = [0.28, 1.5, 6.0]
    la = pd.Series(base, index=idx); lb = pd.Series(0.0, index=idx); R = SP.RULES; pos = SP.pair_positions(la, lb, 1.0, idx[100]); assert (pos.iloc[:100] == 0).all() and set(pos.unique()) <= {-1.0, 0.0, 1.0}
    z = ((la - la.rolling(60).mean()) / la.rolling(60).std(ddof=1)); assert R["entry"] <= z.iloc[200] < R["stop"] and pos.iloc[200] == -1.0      # high spread -> short spread
    assert R["exit"] < z.iloc[201] < R["entry"] and pos.iloc[201] == -1.0                                                                          # inside the band: hold
    assert abs(z.iloc[202]) <= R["exit"] and pos.iloc[202] == 0.0                                                                                   # exit rule
    assert R["entry"] <= z.iloc[300] < R["stop"] and pos.iloc[300] == -1.0 and z.iloc[301] >= R["stop"] and pos.iloc[301] == 0.0                     # stop -> flat
    assert z.iloc[302] >= R["entry"] and pos.iloc[302] == 0.0                                                                                       # blocked: no re-entry until |z| < entry
    for t in range(101, 400):                                                                                                                       # invariants everywhere
        if pos.iloc[t - 1] == 0 and pos.iloc[t] != 0: assert abs(z.iloc[t]) >= R["entry"]
        if pos.iloc[t - 1] != 0 and pos.iloc[t] != 0: assert R["exit"] < abs(z.iloc[t]) < R["stop"]

def test_future_perturbation_cannot_change_positions_or_weights_up_to_T():
    px = with_coint_pair(); T = pd.Timestamp("2017-06-30"); p2 = px.copy(); p2[p2.index > T] *= 3.0; start = pd.Timestamp("2015-01-02")
    a = SP.pair_positions(np.log(px["SPY"]), np.log(px["QQQ"]), 1.2, start); b = SP.pair_positions(np.log(p2["SPY"]), np.log(p2["QQQ"]), 1.2, start); pd.testing.assert_series_equal(a[:T], b[:T])
    n1, W1 = SP.pair_run(px, "SPY", "QQQ", 1.2, start); n2, W2 = SP.pair_run(p2, "SPY", "QQQ", 1.2, start); pd.testing.assert_series_equal(n1[:T], n2[:T])

def test_one_day_execution_lag():
    px = with_coint_pair(); start = pd.Timestamp("2015-01-02"); s = SP.pair_positions(np.log(px["SPY"]), np.log(px["QQQ"]), 1.2, start); _, W = SP.pair_run(px, "SPY", "QQQ", 1.2, start)
    g = 1 / 2.2; assert np.allclose(W["SPY"].to_numpy()[1:], (s * g).shift(1).loc[start:].to_numpy()[1:])

def test_pair_gross_is_exactly_one_and_portfolio_stays_within_max_leverage_and_gate_catches_overlever():
    px = with_coint_pair(); start = pd.Timestamp("2015-01-02"); _, W = SP.pair_run(px, "SPY", "QQQ", 1.2, start); assert abs(W.abs().sum(axis=1).max() - 1.0) < 1e-9
    pr = pd.DataFrame({"a": ["SPY", "SPY"], "b": ["QQQ", "IWM"], "beta": [1.2, 0.9]}); net, Wp = SP.portfolio_run(px, pr, start); assert Wp.abs().sum(axis=1).max() <= 1.0 + 1e-9
    v = V.risk_gate(net.loc["2015-01-06":], Wp.loc["2015-01-06":] * 2.5, CRIT.risk); assert any("leverage" in x for x in v)       # gate is live for a 2.5x-gross book

# ------------------------------------------------------------ DSR counts ALL pairs; store; protected test never touched
def fake_tests(px, surv=(("SPY", "QQQ", 1.0),)):
    cols = list(px.columns); rows = []
    for a, b in SP.PAIRS:
        if a in cols and b in cols: rows.append({"a": a, "b": b, "n_obs": 2500, "beta": 1.0, "adf_stat": -3.0, "pvalue": 0.3})
    tab = pd.DataFrame(rows)
    for a, b, be in surv: tab.loc[(tab.a == a) & (tab.b == b), ["pvalue", "beta"]] = [1e-6, be]
    return {"engine": "fake(test)", "table": tab}

def test_dsr_deflated_by_all_pairs_tested_not_survivors_and_no_test_access(monkeypatch, tmp_path):
    px = common.dev_prices(with_coint_pair(), CRIT); st = Store(tmp_path / "s.db"); T = fake_tests(px)
    def boom(*a, **k): raise AssertionError("record_test_access must never be called")
    monkeypatch.setattr(Store, "record_test_access", boom)
    r = SP.evaluate_pairs(px, CRIT, st, "d", tests=T); m = r["metrics"]; npairs = 6                       # 4 assets -> C(4,2)=6
    assert m["n_pairs_tested"] == npairs and m["n_bh_fdr"] == 1 and m["n_trials_family"] == npairs and len(r["trial_sharpes_daily"]) == npairs
    assert V.deflated_sharpe(CRIT.splits.slice(r["net"])["validation"], r["trial_sharpes_daily"]) == pytest.approx(m["validation"]["dsr"])
    e = st.q("SELECT * FROM experiments"); assert len(e) == 1 and e[0]["criteria_hash"] == CRIT.digest() and st.test_looks() == 0
    assert SP.evaluate_pairs(px, CRIT, st, "d", tests=T)["duplicate"] is True and len(st.q("SELECT * FROM experiments")) == 1       # dedup

def test_null_when_no_survivor_is_recorded_plainly(tmp_path):
    px = common.dev_prices(rw(["SPY", "QQQ", "IWM", "EFA"], 5), CRIT); T = fake_tests(px, surv=()); st = Store(tmp_path / "s.db"); r = SP.evaluate_pairs(px, CRIT, st, "d", tests=T)
    assert r["decision"] == "VALIDATION_FAIL" and r["reasons"][0].startswith("NULL: 0 of 6 pairs") and r["metrics"]["n_bh_fdr"] == 0 and st.test_looks() == 0

def test_prereg_pinned_and_universe_is_276_pairs_and_family_never_calls_record_test_access():
    import re
    assert len(SP.PAIRS) == 276 and SP.PREREG_HASH == SP.PREREG_HASH_PINNED
    src = "".join(p.read_text() for p in Path(SP.__file__).parent.glob("*.py")) + Path(SA.__file__).read_text()
    assert not re.search(r"\.record_test_access\s*\(", src)

# ------------------------------------------------------------ slow: noise control through the real worker
@pytest.mark.slow
@needs_sa
def test_slow_noise_universe_pipeline_has_no_false_positives_and_planted_pair_is_found(tmp_path):
    px = common.dev_prices(with_coint_pair(seed=4, cols=tuple(SP.ETF24[:5])), CRIT); r = SP.evaluate_pairs(px, CRIT, None, "d", record=False)
    assert r["metrics"]["n_bh_fdr"] >= 1 and r["metrics"]["engine"] == "arbitragelab"           # power: the planted SPY/QQQ pair survives BH among 10 pairs
    nz = common.dev_prices(rw(SP.ETF24[:6], 8), CRIT); rn = SP.evaluate_pairs(nz, CRIT, None, "d", record=False); assert rn["decision"] != "VALIDATION_PASS"

# ------------------------------------------------------------ exact-boundary guards (reviewer finding)
def test_guard_dev_exact_boundary_via_adapter():
    ok = rw(["A", "B", "C"], dates=pd.bdate_range("2019-12-02", "2019-12-31")); common.guard_dev(ok)                 # ends exactly 2019-12-31: allowed
    for bad_end in ("2020-01-01", "2020-01-06"):
        bad = rw(["A", "B", "C"], dates=pd.DatetimeIndex(list(ok.index) + [pd.Timestamp(bad_end)]))
        with pytest.raises(TestWindowAccess): SA.coint_pairs(bad)

@needs_sa
def test_statarb_worker_guard_exact_boundary():
    import subprocess
    def run(dates, fe="2014-12-31"):
        code = "import importlib.util as u,pandas as pd;s=u.spec_from_file_location('w',r'%s');w=u.module_from_spec(s);s.loader.exec_module(w);w.check_dates(pd.DatetimeIndex(%r), pd.Timestamp(%r));print('OK')" % (LAB / "adapters/workers/statarb_worker.py", dates, fe)
        return subprocess.run([str(LAB / "envs/statarb/bin/python"), "-I", "-W", "ignore", "-c", code], capture_output=True, text=True)
    r = run(["2019-12-30", "2019-12-31"]); assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-300:]
    for bad in ("2020-01-01", "2020-01-06"):
        r = run(["2019-12-31", bad]); assert r.returncode != 0 and "TestWindowAccess" in r.stderr
    assert run(["2019-12-31"], "2015-01-01").returncode != 0                                                         # formation_end boundary too
    assert run(["2019-12-31"], "2014-12-31").returncode == 0
