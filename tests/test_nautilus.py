"""nautilus_trader event-driven adapter tests. ALL PRICE DATA HERE IS SYNTHETIC (seeded GBM, software tests only) except the one
`slow` test, which reads the cached real SPY file (train+validation only) and is skipped if the cache is absent."""
import ast, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd, pytest
from adapters import nautilus_adapter as NA
from adapters.envrun import ROOT
from engine.backtesting.vector import run_backtest, trend_signal
from engine.research.validation import Criteria

WORKER = ROOT / "adapters" / "workers" / "nautilus_worker.py"
ADAPTER = ROOT / "adapters" / "nautilus_adapter.py"
needs_env = pytest.mark.skipif(not NA.available(), reason="envs/nautilus not built")


def synth(start="2010-01-01", end="2013-12-31", seed=3):          # SYNTHETIC test prices (labelled; never evidence)
    idx = pd.bdate_range(start, end); r = np.random.default_rng(seed).normal(0.0003, 0.012, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx, name="SYNTHETIC_GBM")


# ------------------------------------------------------------------ no execution path (static)
FORBIDDEN_MODULE_PREFIXES = ("nautilus_trader.adapters", "nautilus_trader.live", "nautilus_trader.system.kernel")
FORBIDDEN_NAMES = ("LiveNode", "TradingNode", "LiveExecutionClient", "LiveDataClient", "LiveExecClientFactory", "LiveDataClientFactory",
                   "api_key", "api_secret", "API_KEY", "SECRET")

def _imports(src: str) -> list[str]:
    mods = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import): mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom): mods.append(n.module or "")
    return mods

def test_worker_imports_no_live_or_adapter_clients():
    src = WORKER.read_text(); mods = _imports(src)
    assert any(m.startswith("nautilus_trader.backtest") for m in mods)          # sanity: the parser sees the real imports
    bad = [m for m in mods if m.startswith(FORBIDDEN_MODULE_PREFIXES)]
    assert not bad, bad
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    for f in FORBIDDEN_NAMES:
        assert f not in src and f not in names, f
    assert "__import__" not in src and "importlib" not in src                  # no dynamic-import loophole

def test_core_adapter_never_imports_nautilus():
    assert not [m for m in _imports(ADAPTER.read_text()) if m.startswith("nautilus_trader")]


# ------------------------------------------------------------------ protected window guards
def test_dev_end_is_validation_end():
    assert NA.DEV_END == pd.Timestamp(Criteria().splits.val_end) == pd.Timestamp("2019-12-31")

def test_adapter_raises_on_protected_dates_before_launching(monkeypatch):
    launched = []
    monkeypatch.setattr(NA, "run_in_env", lambda *a, **k: launched.append(a) or {})
    px = synth("2019-06-03", "2020-03-31")                                        # SYNTHETIC, straddles DEV_END
    for mode in NA.MODES:
        with pytest.raises(NA.ProtectedWindowAccess): NA.run_trend(px, 3, mode=mode)
    with pytest.raises(NA.ProtectedWindowAccess): NA.run_trend(synth("2020-01-02", "2020-01-10"), 1)   # a single protected day
    with pytest.raises(NA.ProtectedWindowAccess): NA.assert_dev_only(pd.DatetimeIndex(["2020-01-01"]))
    assert not launched
    NA.assert_dev_only(pd.DatetimeIndex(["2019-12-31"]))                          # boundary day itself is validation, allowed

def test_no_override_parameter_exists():
    import inspect
    for fn in (NA.run_trend, NA.assert_dev_only):
        assert not [p for p in inspect.signature(fn).parameters if "protect" in p or "allow" in p or "override" in p]

def test_compare_raises_on_protected_dates():
    from engine.eventsim.compare import compare
    with pytest.raises(NA.ProtectedWindowAccess): compare(synth("2019-01-01", "2020-02-28"), {}, 3)

@needs_env
def test_worker_guard_refuses_protected_bars_directly(tmp_path):
    px = synth("2019-10-01", "2020-01-31"); f = tmp_path / "p.parquet"; px.rename("close").to_frame().to_parquet(f)
    a = {"data": str(f), "out": str(tmp_path / "o.json"), "lookback_months": 1, "cost_bps": 2.0, "starting_cash": 1e6,
         "mode": "same_bar_close", "month_ends": NA.month_end_dates(px)}
    r = subprocess.run([str(ROOT / "envs/nautilus/bin/python"), "-I", str(WORKER), json.dumps(a)], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "protected" in r.stderr and not (tmp_path / "o.json").exists()


# ------------------------------------------------------------------ calendar rule == vector engine
def test_month_end_dates_match_vector_rule():
    px = synth(); sig = trend_signal(px, 3)
    changes = set(sig.dropna().index[sig.dropna().diff().fillna(0) != 0].strftime("%Y-%m-%d"))
    assert changes <= set(NA.month_end_dates(px))                 # vector signal only changes on our month-end list


# ------------------------------------------------------------------ fast end-to-end (unmarked: registry evidence)
@needs_env
def test_end_to_end_synthetic_matches_vector_and_fills_at_decision_close():
    from engine.eventsim.compare import compare
    px = synth(); ev = NA.run_trend(px, lookback_months=3, cost_bps=2.0, starting_cash=1_000_000.0, mode="same_bar_close")
    f = ev["fills"]; assert len(f) >= 4 and ev["nautilus_version"]
    # execution semantics: market order submitted in on_bar(t) fills at close t (4dp price precision)
    for _, x in f.iterrows():
        assert abs(x["price"] - px[x["date"]]) < 1e-4 + 1e-12, x
        assert float(x["qty"]).is_integer() and x["qty"] > 0
        assert abs(x["commission"] - x["qty"] * x["price"] * 2e-4) <= 0.011       # fee = 2bp of notional, rounded to cents
    assert (ev["equity"]["cash"] >= -1e-9).all()                                 # cash account never borrows
    sides = f["side"].tolist(); assert all(a != b for a, b in zip(sides, sides[1:])) and sides[0] == "BUY"
    c = compare(px, ev, 3, 2.0, 1_000_000.0); s = c["summary"]
    vec_trades = (run_backtest(px, trend_signal(px, 3), 2.0).weights.diff().fillna(1.0) != 0).sum()
    assert s["event_driven"]["trades"] == s["vector"]["trades"] == vec_trades
    assert s["daily_return_corr"] > 0.99999 and s["max_abs_daily_diff"] < 5e-5
    assert (c["trades"]["fill_lag_sessions"] == 0).all()
    assert abs(s["event_driven"]["total_return"] - s["vector"]["total_return"]) < 2e-3

@needs_env
def test_next_bar_mode_fills_one_session_later():
    px = synth(); ev = NA.run_trend(px, lookback_months=3, mode="next_bar_close"); idx = px.index
    me = set(pd.to_datetime(NA.month_end_dates(px)))
    for _, x in ev["fills"].iterrows():
        prev = idx[idx.get_loc(x["date"]) - 1]
        assert prev in me and abs(x["price"] - px[x["date"]]) < 1e-4 + 1e-12


# ------------------------------------------------------------------ real cached data, train+validation only (slow)
@needs_env
@pytest.mark.slow
def test_real_spy_dev_window_primary_matches_vector():
    from engine.eventsim.compare import compare, load_spy_dev
    try: px, meta = load_spy_dev()
    except Exception as e: pytest.skip(f"cached SPY not available: {e}")
    assert px.index.max() <= NA.DEV_END and meta.kind == "real"
    c = compare(px, NA.run_trend(px, 12, 2.0, 1_000_000.0, "same_bar_close"), 12, 2.0, 1_000_000.0)["summary"]
    assert c["daily_return_corr"] > 0.99999 and c["event_driven"]["trades"] == c["vector"]["trades"]
