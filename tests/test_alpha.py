import numpy as np, pandas as pd, pytest
from engine.agents import alpha as A
from engine.memory.store import Store
from engine.orchestration.budget import Budget, BudgetGuard
from engine.research import validation as V

def universe(n=5500, k=12, seed=3):      # SYNTHETIC fixture: software tests only
    idx = pd.bdate_range("2005-01-03", periods=n); rng = np.random.default_rng(seed)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n, k)), axis=0), index=idx, columns=[f"E{i}" for i in range(k)])
    px.iloc[:1500, 8:] = np.nan                                        # four ETFs incept late
    return px
SPEC = {"family": "xs_factor", "universe": "t", "kind": "momentum", "lookback_days": 126, "skip_days": 21, "vol_adj": False, "top_frac": 0.33, "min_assets": 8, "cost_bps": 5.0}

def test_xs_causality_future_edit_cannot_change_past_weights():
    px = universe(); T = px.index[3000]; px2 = px.copy(); px2[px2.index > T] *= 7
    for kind in ("momentum", "reversal", "lowvol"):
        s = {**SPEC, "kind": kind}
        pd.testing.assert_frame_equal(A.target_weights(s, px)[:T], A.target_weights(s, px2)[:T])

def test_no_nan_filling_and_late_assets_not_held_before_inception():
    px = universe(); net, W, bench = A.run_xs(SPEC, px)
    assert (W.loc[W.index[:1400], ["E8", "E9", "E10", "E11"]] == 0).all().all()
    assert np.allclose(W.sum(axis=1)[W.sum(axis=1) > 0], 1.0)             # fully invested long-only, never levered

def test_too_few_assets_means_flat_not_guess():
    W = A.target_weights({**SPEC, "min_assets": 8}, universe().iloc[:, :6])
    assert W.abs().sum().sum() == 0

def test_costs_hurt_and_active_series_defined():
    px = universe(); a = A.run_xs(SPEC, px)[0]; b = A.run_xs({**SPEC, "cost_bps": 100.0}, px)[0]
    assert b.sum() < a.sum()

def test_complexity_penalty_orders_simple_first():
    c = A.candidate_specs(); cx = [A.complexity(s) for s, _ in c]
    assert cx == sorted(cx) and cx[0] == 1 and len({A.name_of(s) for s, _ in c}) == len(c)

def test_evaluate_records_dedups_and_counts_trials(tmp_path):
    st = Store(tmp_path / "a.db"); px = universe(); crit = V.Criteria()
    r1 = A.evaluate_xs(SPEC, px, crit, st, "d"); r2 = A.evaluate_xs(SPEC, px, crit, st, "d")
    assert r2["duplicate"] and r1["metrics"]["n_trials_family"] == 1
    r3 = A.evaluate_xs({**SPEC, "lookback_days": 63}, px, crit, st, "d"); assert r3["metrics"]["n_trials_family"] == 2
    assert st.q("SELECT COUNT(*) n FROM lessons")[0]["n"] == 2

def test_search_respects_budget_and_cap(tmp_path):
    st = Store(tmp_path / "b.db"); px = universe()
    out = A.run_alpha_search(st, px, "d", guard=BudgetGuard(st, Budget(max_experiments_per_day=2)), max_experiments=10)
    assert len(out["ran"]) == 2 and "budget" in out["stopped"]
    out2 = A.run_alpha_search(Store(tmp_path / "c.db"), px, "d", max_experiments=3); assert len(out2["ran"]) == 3 and out2["stopped"]

def test_random_noise_universe_yields_no_candidates(tmp_path):
    """False-positive control: on iid noise the full pipeline must not crown a CANDIDATE."""
    st = Store(tmp_path / "n.db"); px = universe(seed=11); A.run_alpha_search(st, px, "d", max_experiments=12)
    assert st.q("SELECT COUNT(*) n FROM strategies WHERE status='CANDIDATE'")[0]["n"] == 0
