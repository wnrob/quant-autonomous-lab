import numpy as np, pytest
from engine.memory.store import Store
from engine.research.pooled import pooled_report
from engine.research import validation as V

def test_expected_max_grows_with_trials_and_is_zero_for_one():
    rng = np.random.default_rng(0); a = list(rng.normal(0, 0.05, 5)); b = list(rng.normal(0, 0.05, 500))
    assert V.expected_max_sharpe([0.1]) == 0.0 and V.expected_max_sharpe(b) > V.expected_max_sharpe(a) > 0

def test_pooling_external_trials_raises_the_ceiling_and_never_changes_decisions(tmp_path):
    st = Store(tmp_path / "p.db"); sid, _ = st.get_or_create_strategy({"family": "trend", "asset": "X", "lookback_months": 12, "vol_target": None, "cost_bps": 2.0}, "x")
    for i, sh in enumerate([0.9, 0.2, -0.3, 0.1]):
        st.add_experiment(strategy_id=sid, spec_hash=f"s{i}", criteria_hash="c", data_sha="d", code_version="v", seed=0, secs=0, metrics={"validation": {"sharpe": sh}}, decision="VALIDATION_PASS" if i == 0 else "VALIDATION_FAIL", reasons=["r"])
    before = st.q("SELECT decision FROM experiments ORDER BY id")
    r0 = pooled_report(st); r1 = pooled_report(st, list(np.random.default_rng(1).normal(0.1, 0.5, 300)))
    assert r1["noise_ceiling_sharpe_ann"] > r0["noise_ceiling_sharpe_ann"] and r1["n_pooled"] == 304
    assert st.q("SELECT decision FROM experiments ORDER BY id") == before                     # report-only
    assert any(x["id"] == 1 for x in r1["passing_decisions_that_do_not_clear"])               # the 0.9 'pass' does not clear a 300-trial pool
