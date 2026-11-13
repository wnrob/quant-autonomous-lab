import numpy as np, pandas as pd, pytest
pytestmark = pytest.mark.adapter

@pytest.fixture(scope="module")
def r():  # SYNTHETIC fixture: software tests only
    rng = np.random.default_rng(5)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (1500, 3)) * [1, 1.5, 2], columns=list("ABC"), index=pd.bdate_range("2015-01-01", periods=1500))

def test_pypfopt_min_vol_matches_engine(r):
    pytest.importorskip("pypfopt")
    from adapters.pypfopt_adapter import min_vol_weights
    from engine.optimization.portfolio import min_variance
    a, b = min_vol_weights(r), min_variance(r)
    assert a.sum() == pytest.approx(1, abs=1e-3)
    assert (a - b).abs().max() < 0.12      # different shrinkage estimators -> close, not identical
    assert a.idxmax() == b.idxmax() == "A"

def test_quantstats_agrees_with_engine(r):
    pytest.importorskip("quantstats")
    from adapters.quantstats_adapter import metrics
    from engine.analytics.returns import describe
    s = r["A"]; q, e = metrics(s), describe(s)
    assert q["max_drawdown"] == pytest.approx(e["max_drawdown"], abs=1e-6)
    assert q["volatility"] == pytest.approx(e["ann_vol"], rel=1e-3)

def test_registry_operational_entries_are_tested():
    from adapters.registry import REGISTRY, health_check
    for s in REGISTRY:
        if s.status == "operational": assert health_check(s) == "ok"

@pytest.mark.slow
def test_qlib_expression_engine_matches_pandas(tmp_path):
    """qlib adapter (envs/ml, source-built) vs plain pandas on the same data, incl. a late-starting asset with gaps NOT filled."""
    from adapters.envrun import ROOT
    if not (ROOT / "envs/ml/bin/python").exists(): pytest.skip("envs/ml not built")
    from adapters.qlib_adapter import features
    rng = np.random.default_rng(2); idx = pd.bdate_range("2015-01-01", periods=600)
    px = pd.DataFrame({"AAA": 100 * np.cumprod(1 + rng.normal(0, .01, 600)), "BBB": 50 * np.cumprod(1 + rng.normal(0, .012, 600))}, index=idx)
    px.loc[idx[:100], "BBB"] = np.nan                                       # BBB starts late
    f = features(px, {"mom": "Ref($close,21)/Ref($close,252)-1", "ret1": "$close/Ref($close,1)-1"}, tmp_path)
    for sym in ("AAA", "BBB"):
        exp = px[sym].shift(21) / px[sym].shift(252) - 1
        got = f[(sym, "mom")].reindex(exp.index)
        both = exp.notna() & got.notna(); assert both.sum() > 100
        np.testing.assert_allclose(got[both], exp[both], rtol=2e-4, atol=1e-5)
        assert got[idx[:100]].isna().all() if sym == "BBB" else True        # nothing invented before inception

@pytest.mark.slow
def test_rdagent_llm_backend_reaches_local_ollama():
    import urllib.request
    from adapters.envrun import ROOT
    if not (ROOT / "envs/agents/bin/python").exists(): pytest.skip("envs/agents not built")
    try: urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
    except Exception: pytest.skip("ollama not running")
    from adapters.rdagent_adapter import chat
    from engine.errors import AdapterUnavailable
    try: assert len(chat("Reply with the word: ok", timeout=240).strip()) > 0
    except AdapterUnavailable as e: pytest.skip(f"rdagent/ollama unavailable: {e}")      # never hang the suite

def test_rdagent_adapter_timeout_kills_process_tree(tmp_path, monkeypatch):
    """Regression: a hung worker must raise within the timeout, not block forever."""
    import time, adapters.rdagent_adapter as ra
    from engine.errors import AdapterUnavailable
    fake = tmp_path / "sbx.sh"; fake.write_text("#!/bin/bash\nsleep 60 & wait\n"); fake.chmod(0o755)
    monkeypatch.setattr(ra, "ROOT", ra.ROOT); real = ra.subprocess.Popen
    monkeypatch.setattr(ra.subprocess, "Popen", lambda cmd, **k: real([str(fake)], **k))
    t = time.time()
    with pytest.raises(AdapterUnavailable, match="timed out"): ra.chat("x", timeout=2)
    assert time.time() - t < 15
