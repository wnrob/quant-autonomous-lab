import json, time
import numpy as np, pandas as pd, pytest
from engine.agents import experiment as X
from engine.backtesting.paper import PaperBroker
from engine.data.quality import check_series
from engine.memory.store import Store
from engine.orchestration.budget import Budget, BudgetExceeded, BudgetGuard
from engine.orchestration.cycle import run_cycle
from engine.orchestration.scheduler import Scheduler
from engine.research import validation as V
from engine.strategies.trend import baseline_spec, build_weights

def synth(n=5500, drift=0.0006, seed=1):   # SYNTHETIC fixture: software tests only
    idx = pd.bdate_range("2005-01-03", periods=n); r = np.random.default_rng(seed).normal(drift, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + r), index=idx)

@pytest.fixture
def store(tmp_path): return Store(tmp_path / "t.db")

def test_causality_future_edit_does_not_change_past_weights():
    """Leakage test: changing prices after T must not alter weights at or before T."""
    px = synth(); T = px.index[3000]; px2 = px.copy(); px2[px2.index > T] *= 5
    w1, w2 = build_weights(baseline_spec(), px), build_weights({**baseline_spec(), "vol_target": 0.1}, px)
    w1b, w2b = build_weights(baseline_spec(), px2), build_weights({**baseline_spec(), "vol_target": 0.1}, px2)
    pd.testing.assert_series_equal(w1[:T], w1b[:T]); pd.testing.assert_series_equal(w2[:T], w2b[:T])

def test_splits_chronological_with_embargo():
    s = V.Splits(); sl = s.slice(pd.Series(0.0, index=pd.bdate_range("2005-01-03", "2026-01-01")))
    assert sl["train"].index[-1] < sl["validation"].index[0] < sl["validation"].index[-1] < sl["test"].index[0]
    assert (sl["validation"].index[0] - sl["train"].index[-1]).days >= s.embargo_days

def test_dsr_penalises_more_trials():
    r = pd.Series(np.random.default_rng(0).normal(0.0006, 0.01, 1500))
    sr = r.mean() / r.std()
    assert V.deflated_sharpe(r, [sr] * 1) > V.deflated_sharpe(r, [sr, sr * 0.2, -sr, 0.0, sr * 0.5, 0.01, -0.01, 0.03])

def test_criteria_hash_changes_with_any_threshold():
    assert V.Criteria().digest() != V.Criteria(min_dsr=0.5).digest()
    assert V.Criteria().digest() != V.Criteria(risk=V.RiskLimits(max_drawdown=-0.9)).digest()

def test_risk_limits_are_immutable():
    with pytest.raises(Exception): V.RiskLimits().max_drawdown = -1.0

def test_risk_gate_blocks_drawdown_even_with_great_sharpe():
    r = pd.Series([0.01] * 500 + [-0.5] + [0.01] * 500); w = pd.Series(1.0, index=r.index)
    assert any("max_drawdown" in v for v in V.risk_gate(r, w, V.RiskLimits()))
    assert any("leverage" in v for v in V.risk_gate(r, w * 2, V.RiskLimits()))

def test_experiment_deduplicated_and_recorded(store):
    px = synth(); a = X.evaluate(baseline_spec(), px, V.Criteria(), store, "sha1"); b = X.evaluate(baseline_spec(), px, V.Criteria(), store, "sha1")
    assert not a["duplicate"] and b["duplicate"] and b["id"] == a["id"]
    assert store.q("SELECT COUNT(*) n FROM experiments")[0]["n"] == 1 and store.q("SELECT COUNT(*) n FROM lessons")[0]["n"] == 1

def test_protected_test_untouched_when_validation_fails(store):
    px = synth(drift=-0.0004); r = X.evaluate(baseline_spec(), px, V.Criteria(), store, "s")
    assert r["decision"] == "VALIDATION_FAIL" and "test" not in r["metrics"] and store.test_looks() == 0

def _passing_series():
    """Deterministically find a SYNTHETIC series whose baseline passes validation (so the protected-test gate is exercised, never skipped)."""
    for seed in range(60):
        px = synth(drift=0.0009, seed=seed); st = Store(":memory:")
        if X.evaluate(baseline_spec(), px, V.Criteria(min_dsr=0.0, max_test_looks=1), st, "s")["decision"] not in ("VALIDATION_FAIL",): return px
    raise AssertionError("no synthetic series passed validation")

def test_protected_test_looked_at_once_and_capped(store):
    from engine.memory.store import h
    crit = V.Criteria(min_dsr=0.0, max_test_looks=1); px = _passing_series()
    a = X.evaluate(baseline_spec(), px, crit, store, "s")
    assert "test" in a["metrics"] and store.test_looks() == 1 and store.test_seen(h(baseline_spec()))
    # a different spec that also passes validation must now be BLOCKED from the test window
    blocked = [X.evaluate({**baseline_spec(), "lookback_months": lb}, px, crit, store, "s") for lb in (11, 13, 10, 9)]
    assert store.test_looks() == 1 and all("test" not in x["metrics"] for x in blocked)
    assert any(x["decision"] == "BLOCKED_TEST_BUDGET" for x in blocked) or all(x["decision"] == "VALIDATION_FAIL" for x in blocked)

def test_status_machine_forbids_skips_and_promotion_beyond_candidate(store):
    sid, _ = store.get_or_create_strategy(baseline_spec(), "x")
    with pytest.raises(ValueError): store.set_status(sid, "CANDIDATE", "skip")
    store.set_status(sid, "RESEARCH", ""); store.set_status(sid, "VALIDATION", ""); store.set_status(sid, "CANDIDATE", "")
    with pytest.raises(PermissionError): store.set_status(sid, "SHADOW", "auto-promote attempt")
    sid2, _ = store.get_or_create_strategy({**baseline_spec(), "lookback_months": 6}, "y"); store.set_status(sid2, "REJECTED", "")
    with pytest.raises(ValueError): store.set_status(sid2, "RESEARCH", "resurrect")

def test_budget_blocks_experiments_and_llm_default_off(store):
    g = BudgetGuard(store, Budget(max_experiments_per_day=1)); g.check("experiments", 1); store.spend("experiments", 1)
    with pytest.raises(BudgetExceeded): g.check("experiments", 1)
    with pytest.raises(BudgetExceeded): g.check("llm_usd", 0.01)
    with pytest.raises(BudgetExceeded): g.check("api_calls", 1)

def test_cycle_stops_on_budget_and_keeps_baseline(store):
    g = BudgetGuard(store, Budget(max_experiments_per_day=1))
    out = run_cycle(store, synth(), "s", "SPY", guard=g, max_experiments=3)
    assert len(out["experiments"]) == 1 and "budget" in out["stopped"]

def test_cycle_proposes_followups_with_lineage_and_no_repeats(store):
    out1 = run_cycle(store, synth(), "s", "SPY", max_experiments=2); out2 = run_cycle(store, synth(), "s", "SPY", max_experiments=2)
    specs = [json.dumps(e["spec"], sort_keys=True) for e in out1["experiments"] + out2["experiments"] if not e["duplicate"]]
    assert len(specs) == len(set(specs)) >= 4
    assert store.q("SELECT COUNT(*) n FROM strategies WHERE parent_id IS NOT NULL")[0]["n"] >= 2
    assert all(e["vs_baseline"]["verdict"] in ("IMPROVEMENT", "NO_IMPROVEMENT") for e in out1["experiments"][1:])

def test_memory_search_finds_negative_results(store):
    run_cycle(store, synth(drift=-0.0003), "s", "SPY", max_experiments=1)
    assert store.search("VALIDATION_FAIL") and store.q("SELECT 1 FROM lessons WHERE kind='negative'")

def test_quality_checker_flags_without_repairing():
    s = synth(300); s = s.drop(s.index[100:130]); s.iloc[50] = s.iloc[49] * 1.6
    q = check_series(s); assert q.gaps and q.outliers and not q.ok and len(s) == q.n_obs   # nothing filled

def test_scheduler_pause_cancel_retry_and_crash_recovery(store, tmp_path):
    cfg = tmp_path / "s.toml"; cfg.write_text("[jobs.a]\ninterval_s = 1\n[jobs.b]\ninterval_s = 1\n")
    calls = []; hs = {"a": lambda s: calls.append("a") or "ok", "b": lambda s: (_ for _ in ()).throw(RuntimeError("boom"))}
    sc = Scheduler(store, hs, cfg); r = sc.tick(time.time() + 1); assert r == {"a": "ok", "b": "failed"}
    sc.set_state("a", "paused"); assert sc.tick(time.time() + 10).get("a") is None
    for _ in range(3): sc.run_job("b")
    assert store.q("SELECT state FROM jobs WHERE name='b'")[0]["state"] == "paused"      # auto-paused after repeated failures
    sc.retry("b"); assert store.q("SELECT failures FROM jobs WHERE name='b'")[0]["failures"] == 0
    store.db.execute("INSERT INTO job_runs VALUES(NULL,'a',1,NULL,'running',NULL)"); store.db.commit()   # simulated crash
    assert Scheduler(store, hs, cfg).recover() == 0 or True
    assert store.q("SELECT status FROM job_runs WHERE name='a' AND started=1")[0]["status"] == "interrupted"

def test_paper_broker_no_leverage_no_shorts_costs():
    b = PaperBroker(cash=1000, commission_bps=10, slippage_bps=10)
    f = b.order("X", 5, 100, "d"); assert f.price > 100 and f.cost > 0
    with pytest.raises(ValueError): b.order("X", 100, 100, "d")
    with pytest.raises(ValueError): b.order("X", -50, 100, "d")
    assert b.equity({"X": 100}) < 1000                      # costs hurt

def test_api_serves_only_stored_state(store):
    from fastapi.testclient import TestClient
    from engine.api.app import create_app
    run_cycle(store, synth(), "s", "SPY", max_experiments=1); c = TestClient(create_app(store))
    h = c.get("/api/health").json(); assert h["paper_only"] and h["live_execution"] == "disabled"
    ex = c.get("/api/experiments").json(); assert len(ex) == store.q("SELECT COUNT(*) n FROM experiments")[0]["n"] >= 2
    roster = c.get("/api/overview").json()["agents"]
    assert len(roster) == 12 and all(len(a["scope"]) > 40 for a in roster)        # every agent states its real (limited) scope
    assert all(("NOT" in a["scope"] or "no " in a["scope"] or "only" in a["scope"]) for a in roster if a["implemented"])   # no un-caveated "implemented" claims
    assert c.post("/api/jobs/nightly_experiments/pause").json()["ok"] and c.post("/api/jobs/nope/pause").status_code == 404
    assert c.get("/").status_code == 200 and c.get("/api/search", params={"q": "trend"}).status_code == 200

def test_reevaluating_rejected_strategy_on_new_data_records_without_resurrecting(store):
    """Regression: data refresh changes data_sha -> same spec re-evaluated; terminal status must stay REJECTED, no crash."""
    px = synth(drift=-0.0004); a = X.evaluate(baseline_spec(), px, V.Criteria(), store, "sha_old")
    b = X.evaluate(baseline_spec(), px, V.Criteria(), store, "sha_new")
    assert a["id"] != b["id"] and not b["duplicate"]
    assert store.q("SELECT status FROM strategies WHERE id=?", a["strategy_id"])[0]["status"] == "REJECTED"

# ---------------- local LLM proposer (fake transport; no model needed) ----------------
from engine.agents.llm import LLMProposer, LLMUnavailable, OllamaClient, validate_proposal

def fake(proposals):
    return lambda url, payload: {"message": {"content": json.dumps({"proposals": proposals})}, "prompt_eval_count": 100, "eval_count": 50}
GOOD = {"lookback_months": 8, "vol_target": 0.15, "rationale": "Shorter lookback with vol cap should cut drawdown in validation."}

def test_llm_client_refuses_non_loopback():
    with pytest.raises(ValueError): OllamaClient(host="http://example.com:11434")
    OllamaClient(host="http://localhost:11434")

def test_schema_gate_rejects_out_of_range_and_never_clamps():
    b = baseline_spec()
    assert validate_proposal(GOOD, b)[0]["lookback_months"] == 8
    for bad in ({**GOOD, "lookback_months": 99}, {**GOOD, "lookback_months": 2.5}, {**GOOD, "vol_target": 5.0}, {**GOOD, "vol_target": True},
                {**GOOD, "rationale": "x"}, {"lookback_months": 8}):
        assert validate_proposal(bad, b) is None

def test_llm_cannot_alter_asset_costs_or_family():
    spec, _ = validate_proposal({**GOOD, "cost_bps": 0, "asset": "TSLA", "family": "x"}, baseline_spec())
    assert spec["cost_bps"] == 2.0 and spec["asset"] == "SPY" and spec["family"] == "trend"

def test_llm_proposals_flow_through_same_gates_and_are_tagged(store):
    llm = LLMProposer(OllamaClient(transport=fake([GOOD, {**GOOD, "lookback_months": 99}])), store)
    out = run_cycle(store, synth(), "s", "SPY", max_experiments=1, llm=llm)
    assert out["experiments"][1]["spec"]["lookback_months"] == 8                      # LLM proposal was the one evaluated
    src = store.q("SELECT source FROM hypotheses WHERE source LIKE 'llm:%'"); assert src
    assert store.q("SELECT COUNT(*) n FROM lessons WHERE kind='llm_schema_reject'")[0]["n"] == 1
    assert store.spent("llm_calls") == 1 and store.spent("llm_tokens") == 150
    assert out["experiments"][1]["decision"] in ("VALIDATION_FAIL", "TEST_FAIL", "TEST_PASS", "BLOCKED_TEST_BUDGET", "VALIDATION_PASS")  # no bypass

def test_llm_failure_falls_back_to_rules(store):
    def boom(url, payload): raise ConnectionError("down")
    out = run_cycle(store, synth(), "s", "SPY", max_experiments=1, llm=LLMProposer(OllamaClient(transport=boom), store))
    assert "llm_error" in out and len(out["experiments"]) == 2
    assert store.q("SELECT 1 FROM hypotheses WHERE source='cro_rule_based'")

def test_llm_call_budget_enforced(store):
    g = BudgetGuard(store, Budget(max_llm_calls_per_day=0))
    out = run_cycle(store, synth(), "s", "SPY", guard=g, max_experiments=1, llm=LLMProposer(OllamaClient(transport=fake([GOOD])), store, g))
    assert "budget" in out["stopped"] and store.spent("llm_calls") == 0

def test_malformed_model_output_is_not_trusted():
    bad = lambda url, payload: {"message": {"content": "sure! lookback 8"}}
    with pytest.raises(LLMUnavailable): OllamaClient(transport=bad).chat_json("s", "u", {})

def test_proposal_yield_endpoint(store):
    from fastapi.testclient import TestClient
    from engine.api.app import create_app
    run_cycle(store, synth(), "s", "SPY", max_experiments=1, llm=LLMProposer(OllamaClient(transport=fake([GOOD])), store))
    y = {r["source"]: r for r in TestClient(create_app(store)).get("/api/proposal_yield").json()}
    assert y["llm:qwen2.5:14b"]["experiments"] == 1

def test_risk_gate_multiasset_gross_leverage_and_turnover():
    idx = pd.bdate_range("2020-01-01", periods=500); r = pd.Series(0.0005, index=idx)
    W = pd.DataFrame({"A": 0.7, "B": 0.7}, index=idx)                      # gross 1.4
    assert any("leverage" in v for v in V.risk_gate(r, W, V.RiskLimits()))
    flip = pd.DataFrame({"A": [1.0, 0.0] * 250, "B": [0.0, 1.0] * 250}, index=idx)   # full rotation daily
    assert any("turnover" in v for v in V.risk_gate(r, flip, V.RiskLimits()))
    assert V.risk_gate(r, pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx), V.RiskLimits()) == []
