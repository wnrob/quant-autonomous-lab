"""A/B proposer harness tests. Fake transport only (no model needed); SYNTHETIC data. One optional slow live test."""
import json, urllib.request
import numpy as np, pandas as pd, pytest
from engine.agents import alpha as A
from engine.agents.llm import OllamaClient
from engine.agents.llm_xs import FIXED, LLMXSProposer, validate_proposal
from engine.memory.store import Store, h
from engine.orchestration.budget import Budget
from engine.research import ab_proposers as B, validation as V
from tests.test_alpha import universe

GOOD = {"kind": "momentum", "lookback_days": 100, "skip_days": 10, "vol_adj": True, "rationale": "Medium momentum with vol adjustment may be steadier."}

def fake_factory(proposals_by_call):
    """Returns a client_factory(seed) whose transport replays proposals; call index increments across calls (varies with seed)."""
    def make(seed):
        n = {"i": 0}
        def transport(url, payload):
            props = proposals_by_call(seed, n["i"]); n["i"] += 1
            return {"message": {"content": json.dumps({"proposals": props})}, "prompt_eval_count": 10, "eval_count": 5}
        return OllamaClient(model="fake:1b", transport=transport, seed=seed)
    return make
def gen(seed, i):
    rng = np.random.default_rng(seed * 100 + i)
    return [{"kind": str(rng.choice(["momentum", "reversal", "lowvol"])), "lookback_days": int(rng.integers(5, 253)), "skip_days": int(rng.integers(0, 64)),
             "vol_adj": bool(rng.integers(0, 2)), "rationale": "synthetic fake proposal for tests"} for _ in range(4)]

# ---- schema validator
@pytest.mark.parametrize("bad", [
    {**GOOD, "kind": "carry"}, {**GOOD, "lookback_days": 4}, {**GOOD, "lookback_days": 253}, {**GOOD, "skip_days": -1}, {**GOOD, "skip_days": 64},
    {**GOOD, "lookback_days": 100.0}, {**GOOD, "lookback_days": True}, {**GOOD, "vol_adj": 1}, {**GOOD, "vol_adj": "true"}, {**GOOD, "rationale": "short"},
    {**GOOD, "cost_bps": 0.0}, {**GOOD, "top_frac": 0.5}, {**GOOD, "universe": "sp500"}, {**GOOD, "family": "trend"}, {k: v for k, v in GOOD.items() if k != "kind"}, "nope", None])
def test_validator_rejects_never_clamps(bad): assert validate_proposal(bad) is None

def test_validator_accepts_edges_and_fixed_fields_cannot_change():
    for lb, sk in ((5, 0), (252, 63)):
        spec, _ = validate_proposal({**GOOD, "lookback_days": lb, "skip_days": sk})
        assert spec["lookback_days"] == lb and spec["skip_days"] == sk
        assert {k: spec[k] for k in FIXED} == FIXED and set(spec) == set(FIXED) | {"kind", "lookback_days", "skip_days", "vol_adj"}

def test_proposer_tags_source_dedups_and_counts_invalid(tmp_path):
    st = Store(tmp_path / "s.db")
    props = [GOOD, GOOD, {**GOOD, "lookback_days": 999}, {**GOOD, "kind": "reversal", "lookback_days": 20}]
    p = LLMXSProposer(fake_factory(lambda s, i: props)(1), st, None, 2); out = p.propose(V.Criteria(), "d")
    assert p.source == "llm:fake:1b" and len(out) == 2 and p.stats["invalid"] == 1 and p.stats["duplicate"] == 1
    st.get_or_create_strategy(out[0][0], "x"); out2 = p.propose(V.Criteria(), "d")          # already-tested spec is now deduped against the store
    assert len(out2) == 1 and p.calls_left == 0 and p.propose(V.Criteria(), "d") == []

def test_prompt_contains_lessons_and_tested_specs(tmp_path):
    st = Store(tmp_path / "s.db"); spec, _ = validate_proposal(GOOD); st.get_or_create_strategy(spec, "n"); st.add_lesson(None, "xs_factor", "negative", "LESSON-MARKER xyz")
    pr = LLMXSProposer(fake_factory(lambda s, i: [])(1), st).build_prompt(); assert "LESSON-MARKER" in pr and "momentum" in pr and "100" in pr

# ---- allow_test
def test_allow_test_false_never_touches_test_and_default_unchanged(tmp_path):
    px = universe(); crit = V.Criteria(); st = Store(tmp_path / "a.db")
    spec = {**A.candidate_specs()[0][0]}
    r = A.evaluate_xs(spec, px, crit, st, "d", allow_test=False)
    assert st.test_looks() == 0 and "test" not in r["metrics"] and not r["decision"].startswith("TEST_")
    st2 = Store(tmp_path / "b.db"); A.run_alpha_search(st2, px, "d", max_experiments=3, allow_test=False); assert st2.test_looks() == 0
    import inspect; assert inspect.signature(A.evaluate_xs).parameters["allow_test"].default is True
    assert inspect.signature(A.run_alpha_search).parameters["allow_test"].default is True

# ---- harness
@pytest.fixture(scope="module")
def px(): return universe()
def run(px, tmp_path, name="o", **kw):
    return B.run_ab(px, "d", tmp_path / name / "out", tmp_path / name / "work", n_exp=4, rounds=2, client_factory=fake_factory(gen), max_llm_calls=6, progress=lambda *_: None, n_ref_rounds=5, **kw)

def test_isolation_and_equal_budgets(px, tmp_path):
    res = run(px, tmp_path); assert res["test_looks_total"] == 0
    for arm in B.ARMS:
        assert [m["n_evaluated"] for m in res["per_round"][arm]] == [4, 4]            # equal experiment budget per arm/round
        for r in range(2): assert Store(tmp_path / "o" / "work" / arm / f"r{r}" / "qae.db").test_looks() == 0
    assert (tmp_path / "o" / "out" / "results.json").exists()
    main = Store(tmp_path / "main.db"); assert main.test_looks() == 0 and Budget().max_experiments_per_day == 20   # production budget untouched

def test_research_budget_is_explicit_and_separate():
    b = B.research_budget(12, 8); assert b.max_experiments_per_day == 12 and b.max_llm_calls_per_day == 8 and b.max_llm_usd_per_day == 0 and Budget().max_experiments_per_day == 20

def test_reproducible_with_fixed_seeds(px, tmp_path):
    a, b = run(px, tmp_path, "a"), run(px, tmp_path, "b")
    assert a["specs"] == b["specs"]
    for arm in B.ARMS: assert [m["best_active_return"] for m in a["per_round"][arm]] == [m["best_active_return"] for m in b["per_round"][arm]]
    assert B.random_specs(5, 6) == B.random_specs(5, 6) and B.random_specs(5, 6) != B.random_specs(6, 6)
    assert a["specs"]["LLM"][0] != a["specs"]["LLM"][1]                                 # rounds differ by seed

def test_random_specs_are_in_validated_space_and_distinct():
    ss = B.random_specs(1, 50); assert len({h(s) for s in ss}) == 50
    for s in ss: assert validate_proposal({**{k: s[k] for k in ("kind", "lookback_days", "skip_days", "vol_adj")}, "rationale": "x" * 12}) is not None

def test_llm_arm_stops_at_call_cap(px, tmp_path):
    rr = B.run_arm_round("LLM", 0, px, "d", tmp_path, 12, client_factory=fake_factory(lambda s, i: gen(s, i)[:1]), max_llm_calls=3, progress=lambda *_: None)
    assert rr["proposal_stats"]["calls"] == 3 and rr["n_evaluated"] <= 3

# ---- statistics
def test_permutation_identical_arms_not_significant():
    x = [0.1, 0.3, 0.2, 0.5, 0.4]; r = B.perm_test(x, x); assert r["p_value"] == 1.0 and not r["significant_at_0.05"] and r["diff"] == 0
    rng = np.random.default_rng(0); ps = [B.perm_test(rng.normal(size=5), rng.normal(size=5))["p_value"] for _ in range(200)]
    assert np.mean(np.array(ps) < 0.05) < 0.12                                           # ~5% false positives under the null

def test_permutation_detects_clear_difference():
    r = B.perm_test([10, 11, 12, 13, 14], [0, 1, 2, 3, 4]); assert r["significant_at_0.05"] and r["ci95"][0] > 0

@pytest.mark.slow
def test_live_ollama_one_call(tmp_path):
    try: urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
    except Exception: pytest.skip("ollama not running")
    p = LLMXSProposer(B.ollama_factory()(1), Store(tmp_path / "l.db"), None, 1); out = p.propose(V.Criteria(), "d")
    assert p.stats["calls"] == 1 and all(validate_proposal({**{k: s[k] for k in ("kind", "lookback_days", "skip_days", "vol_adj")}, "rationale": "x" * 12}) for s, _ in out)
