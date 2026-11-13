import numpy as np, pandas as pd, pytest
from engine.analytics.returns import describe, max_drawdown, simple_returns
from engine.backtesting.vector import run_backtest, trend_signal
from engine.data.ingest import SyntheticProvider, load_prices, validate_prices
from engine.errors import DataError
from engine.optimization.portfolio import evaluate_oos, max_sharpe, min_variance
from engine.risk.metrics import var_cvar, longest_underwater_days
from engine.research.hypothesis import momentum_hypothesis

@pytest.fixture(scope="module")
def px():  # SYNTHETIC fixture: software tests only
    return SyntheticProvider(1).fetch(["A", "B", "C"], "2005-01-03", "2020-12-31")

def test_describe_known_values():
    r = pd.Series([0.01, -0.01] * 50, index=pd.bdate_range("2020-01-01", periods=100))
    d = describe(r); assert abs(d["ann_vol"] - 0.01005 * np.sqrt(252)) < 1e-3 and d["n_obs"] == 100

def test_max_drawdown():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.0])) == pytest.approx(-0.5)

def test_describe_rejects_short():
    with pytest.raises(ValueError): describe(pd.Series([0.01] * 5))

def test_validate_rejects_bad():
    idx = pd.bdate_range("2020-01-01", periods=5)
    with pytest.raises(DataError): validate_prices(pd.DataFrame({"A": [1, 2, -1, 2, 3]}, index=idx), ["A"])
    with pytest.raises(DataError): validate_prices(pd.DataFrame({"A": [1.0] * 5}, index=idx), ["B"])

def test_cache_roundtrip_and_flags(tmp_path):
    df, m = load_prices(SyntheticProvider(), ["A"], "2010-01-01", "2012-01-01", cache_dir=tmp_path)
    assert m.kind == "synthetic" and "TEST_ONLY" in m.provider
    df2, m2 = load_prices(SyntheticProvider(), ["A"], "2010-01-01", "2012-01-01", cache_dir=tmp_path, offline=True)
    pd.testing.assert_frame_equal(df, df2, check_freq=False); assert m2.sha256 == m.sha256

def test_offline_without_cache_errors(tmp_path):
    with pytest.raises(DataError): load_prices(SyntheticProvider(), ["A"], "2010-01-01", cache_dir=tmp_path, offline=True)

def test_no_same_day_lookahead_in_backtest():
    """Classic bug: signal computed from day-t return trades day t. The engine's 1-day lag must remove that edge."""
    idx = pd.bdate_range("2020-01-01", periods=3000); rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, .01, 3000), index=idx)
    px = 100 * (1 + r).cumprod()
    same_day = (px.pct_change() > 0).astype(float)              # sees day-t return
    res = run_backtest(px, same_day, cost_bps=0)
    naive = (same_day * px.pct_change()).dropna().mean()         # what a lookahead bug would 'earn' (~0.004/day)
    assert naive > 0.003
    assert abs(res.returns.mean()) < 0.0006                      # lagged: ~zero on iid noise

def test_signal_is_applied_next_day():
    idx = pd.bdate_range("2020-01-01", periods=5); px = pd.Series([100, 110, 121, 121, 133.1], index=idx)
    sig = pd.Series([0, 1, 1, 1, 1.0], index=idx)
    w = run_backtest(px, sig, 0).weights
    assert w.loc[idx[1]] == 0 and w.loc[idx[2]] == 1             # signal set on day1 first trades day2

def test_signal_updates_only_month_end_and_warmup(px):
    s = trend_signal(px["A"], 12)
    assert s.iloc[:200].isna().all()                                 # 12-month warmup
    changes = s.dropna().diff().abs() > 0
    me = px["A"].groupby(px.index.to_period("M")).tail(1).index
    assert set(s.dropna().index[changes]) <= set(me)

def test_costs_reduce_return(px):
    sig = trend_signal(px["A"]); a = run_backtest(px["A"], sig, 0).returns.sum(); b = run_backtest(px["A"], sig, 50).returns.sum()
    assert b < a

def test_weights_valid(px):
    r = simple_returns(px)
    for f in (min_variance, max_sharpe):
        w = f(r); assert w.sum() == pytest.approx(1) and (w >= -1e-9).all()

def test_min_var_lowest_in_sample_vol(px):
    r = simple_returns(px); w = min_variance(r)
    assert (r @ w).std() <= (r @ pd.Series(1/3, index=r.columns)).std() + 1e-6

def test_oos_split_is_chronological(px):
    o = evaluate_oos(simple_returns(px).dropna())
    assert o["train"][1] < o["test"][0]

def test_var_cvar_ordering(px):
    v = var_cvar(simple_returns(px["A"])); assert v["cvar_95"] >= v["var_95"] > 0

def test_hypothesis_null_on_iid_noise():
    idx = pd.bdate_range("1990-01-01", "2020-12-31"); rng = np.random.default_rng(3)
    p = pd.Series(100 * np.cumprod(1 + rng.normal(0, .01, len(idx))), index=idx)
    h = momentum_hypothesis(p, n_boot=500); assert h["verdict"] == "fails to reject null" or abs(h["welch_t"]) < 4
    assert h["n_pos"] + h["n_neg"] > 300

def test_underwater():
    assert longest_underwater_days(pd.Series([0.1, -0.1, 0.0, 0.0, 0.2])) >= 2
