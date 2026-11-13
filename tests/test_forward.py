"""Forward SHADOW evaluation tests. ALL price data here is SYNTHETIC (seeded random walks dated 2029-2031) and exists only
to exercise software; nothing here is market evidence. Time is simulated by monkeypatching the logger's only clock
(engine.forward.signals._utcnow) and revealing bars one at a time. Mutation checks done on these tests are listed in docs/FORWARD.md."""
import inspect, json, os, pty, re, select, sqlite3, sys, types
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, pytest
from engine.forward import approval as A, scoring as SC, signals as S
from engine.forward.chain import GENESIS, ChainError, HashChain, canonical, record_hash
from engine.memory.store import Store, h

LAB = Path(__file__).resolve().parents[1]
SPEC = {"family": "trend", "asset": "SYN", "lookback_months": 3, "vol_target": None, "cost_bps": 2.0, "label": "SYNTHETIC_TEST_FIXTURE"}
SH = h(SPEC)

def synth(n=420, seed=3, drift=0.0005, vol=0.005):   # SYNTHETIC fixture: software tests only
    idx = pd.bdate_range("2029-06-01", periods=n); r = np.random.default_rng(seed).normal(drift, vol, n)
    return pd.DataFrame({"SYN": 100 * np.cumprod(1 + r)}, index=idx)

def after_close(day) -> datetime:                    # 22:00 UTC is after the 16:00 New York close in both DST regimes
    return datetime(day.year, day.month, day.day, 22, 0, tzinfo=timezone.utc)

def set_clock(mp, t): mp.setattr(S, "_utcnow", lambda: t)

def seed_candidate(store, spec=SPEC, name="SYNTHETIC-TEST-FIXTURE"):
    sid, _ = store.get_or_create_strategy(spec, name)
    for s in ("RESEARCH", "VALIDATION", "CANDIDATE"): store.set_status(sid, s, "synthetic test fixture")
    return sid

def const_w(w): return lambda spec, df: pd.Series(w, index=df.index, name="SYN")

def log_day(mp, log, store, px, k, w=1.0, kind="synthetic", spec=SPEC):
    """Simulate 'evening of bar k': clock just after that close, frame revealed only up to bar k."""
    t = after_close(px.index[k]); set_clock(mp, t)
    return log.log(store, spec, px.iloc[:k + 1], data_kind=kind, data_fetched_at=t.isoformat(), weights_fn=const_w(w) if w is not None else None)

@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in A.AUTOMATION_ENV: monkeypatch.delenv(k, raising=False)
    st = Store(tmp_path / "qae.db"); seed_candidate(st); home = S.forward_home(st)
    return types.SimpleNamespace(store=st, home=home, log=S.SignalLog(home), apr=A.ApprovalLog(home), px=synth(), mp=monkeypatch, tmp=tmp_path)

# ---------------------------------------------------------------- hash chain
def _chain(tmp_path, n=5):
    c = HashChain(tmp_path / "c.jsonl")
    for i in range(n): c.append({"i": i, "x": 0.1 * i})
    return c

def _rewrite(c, lines): c.path.write_text("".join(l + "\n" for l in lines))

def test_chain_roundtrip_and_each_record_commits_to_previous(tmp_path):
    c = _chain(tmp_path); recs = c.verify()
    assert [r["body"]["i"] for r in recs] == list(range(5)) and recs[0]["prev"] == GENESIS
    assert all(recs[i]["prev"] == recs[i - 1]["hash"] for i in range(1, 5)) and c.head() == (5, recs[-1]["hash"])

def test_chain_detects_edit(tmp_path):
    c = _chain(tmp_path); L = c._lines(); r = json.loads(L[2]); r["body"]["x"] = 99.0; L[2] = canonical(r); _rewrite(c, L)
    with pytest.raises(ChainError, match="content hash"): c.verify()

def test_chain_detects_delete_reorder_insert(tmp_path):
    c = _chain(tmp_path); L = c._lines()
    _rewrite(c, L[:2] + L[3:]);
    with pytest.raises(ChainError): c.verify()
    _rewrite(c, [L[0], L[2], L[1], L[3], L[4]])
    with pytest.raises(ChainError): c.verify()
    forged = {"seq": 2, "prev": json.loads(L[1])["hash"], "body": {"i": "inserted"}}; forged["hash"] = record_hash(2, forged["prev"], forged["body"])
    _rewrite(c, L[:2] + [canonical(forged)] + L[2:])
    with pytest.raises(ChainError): c.verify()

def test_chain_detects_delete_with_renumbered_and_rehashed_records(tmp_path):
    """A careful forger deletes record 2, renumbers and re-hashes each survivor against its own stored prev: only the prev-link check catches it."""
    c = _chain(tmp_path); recs = c.verify(); out = []
    for i, r in enumerate(recs[:2] + recs[3:]):
        out.append(canonical({"seq": i, "prev": r["prev"], "body": r["body"], "hash": record_hash(i, r["prev"], r["body"])}))
    _rewrite(c, out); c.head_path.write_text(json.dumps({"n": 4, "head": json.loads(out[-1])["hash"]}))
    with pytest.raises(ChainError, match="prev hash"): c.verify()

def test_chain_detects_tail_truncation_and_missing_anchor(tmp_path):
    c = _chain(tmp_path); L = c._lines(); _rewrite(c, L[:4])
    with pytest.raises(ChainError, match="head anchor"): c.verify()
    _rewrite(c, L); c.verify(); c.head_path.unlink()
    with pytest.raises(ChainError, match="head anchor"): c.verify()

def test_chain_external_anchor_detects_full_consistent_rewrite(tmp_path):
    c = _chain(tmp_path); n, head = c.head()
    c.path.unlink(); c.head_path.unlink(); c2 = HashChain(c.path)
    for i in range(5): c2.append({"i": i, "x": 0.1 * i + (1 if i == 3 else 0)})      # internally valid forgery
    c2.verify()
    with pytest.raises(ChainError): c2.verify(expected_head=head)
    with pytest.raises(ChainError): c2.verify(anchors=[(n - 1, head)])

def test_chain_append_refuses_on_broken_chain_and_rejects_nan(tmp_path):
    c = _chain(tmp_path); L = c._lines(); r = json.loads(L[0]); r["body"]["i"] = 7; L[0] = canonical(r); _rewrite(c, L)
    with pytest.raises(ChainError): c.append({"i": 5})
    with pytest.raises(ValueError): HashChain(tmp_path / "d.jsonl").append({"x": float("nan")})

# ---------------------------------------------------------------- signal logging guards
def test_log_record_contents(env):
    k = 300; r = log_day(env.mp, env.log, env.store, env.px, k, w=None)       # real trend weights fn
    b = r["body"]; from engine.strategies.trend import build_weights
    assert b["as_of"] == str(env.px.index[k].date()) and b["spec_hash"] == SH and b["forward_status"] == "CANDIDATE" and b["data_kind"] == "synthetic"
    assert b["weights"] == {"SYN": float(build_weights(SPEC, env.px["SYN"].iloc[:k + 1]).iloc[-1])}
    assert b["data_sha"] == S.frame_sha(env.px.iloc[:k + 1]) != S.frame_sha(env.px.iloc[:k + 2]) and b["n_bars"] == k + 1
    assert b["ref_prices"] == {"SYN": float(env.px["SYN"].iloc[k])} and b["as_of_bar_final"] is True and b["code_version"] and b["logged_at"].startswith(b["as_of"])
    assert pd.Timestamp(b["target_earliest"]) > pd.Timestamp(b["as_of"]) and env.log.records()[0]["hash"] == r["hash"]

def test_G1_refuses_when_bar_after_as_of_is_in_hand(env):
    k = 300; set_clock(env.mp, after_close(env.px.index[k]))                   # clock alone would allow as_of=k
    with pytest.raises(S.BackfillError, match="after as_of"):
        env.log.log(env.store, SPEC, env.px.iloc[:k + 2], data_kind="synthetic", as_of=env.px.index[k], weights_fn=const_w(1.0))
    assert env.log.records() == []

def test_G2_refuses_backfill_by_truncated_frame_once_outcome_can_exist(env):
    k = 300; d = env.px.index[k]; trunc = env.px.iloc[:k + 1]; dl = S.outcome_deadline_utc(d, S.LogPolicy())
    for late in (dl, dl + timedelta(seconds=1), dl + timedelta(days=900)):
        set_clock(env.mp, late)
        with pytest.raises(S.BackfillError, match="known outcome"): env.log.log(env.store, SPEC, trunc, data_kind="synthetic", weights_fn=const_w(1.0))
    assert env.log.records() == []
    set_clock(env.mp, dl - timedelta(seconds=1)); env.log.log(env.store, SPEC, trunc, data_kind="synthetic", weights_fn=const_w(1.0))   # last allowed second

def test_G2_weekend_deadline_is_monday_close_new_york():
    fri = pd.Timestamp("2030-01-11"); assert fri.weekday() == 4
    dl = S.outcome_deadline_utc(fri, S.LogPolicy()); assert dl == datetime(2030, 1, 14, 21, 0, tzinfo=timezone.utc)      # Mon 16:00 EST
    assert S.outcome_deadline_utc(pd.Timestamp("2030-07-10"), S.LogPolicy()) == datetime(2030, 7, 11, 20, 0, tzinfo=timezone.utc)   # EDT

def test_G3_refuses_future_dated_bar(env):
    k = 300; set_clock(env.mp, after_close(env.px.index[k - 1]))
    with pytest.raises(S.FutureBarError): env.log.log(env.store, SPEC, env.px.iloc[:k + 1], data_kind="synthetic", weights_fn=const_w(1.0))

def test_G4_refuses_duplicate_and_out_of_order(env):
    log_day(env.mp, env.log, env.store, env.px, 300)
    with pytest.raises(S.DuplicateSignal): log_day(env.mp, env.log, env.store, env.px, 300, w=0.0)
    set_clock(env.mp, after_close(env.px.index[299]))
    with pytest.raises(S.DuplicateSignal): env.log.log(env.store, SPEC, env.px.iloc[:300], data_kind="synthetic", weights_fn=const_w(1.0))
    assert len(env.log.records()) == 1

def test_G5_only_candidates_can_be_logged(env):
    spec2 = {**SPEC, "lookback_months": 4}; sid, _ = env.store.get_or_create_strategy(spec2, "idea")
    set_clock(env.mp, after_close(env.px.index[300]))
    with pytest.raises(PermissionError, match="CANDIDATE"): env.log.log(env.store, spec2, env.px.iloc[:301], data_kind="synthetic", weights_fn=const_w(1.0))
    with pytest.raises(PermissionError): env.log.log(env.store, {**SPEC, "lookback_months": 5}, env.px.iloc[:301], data_kind="synthetic", weights_fn=const_w(1.0))
    env.store.set_status(sid, "REJECTED", "x")
    with pytest.raises(PermissionError): env.log.log(env.store, spec2, env.px.iloc[:301], data_kind="synthetic", weights_fn=const_w(1.0))

def test_log_has_no_clock_or_timestamp_parameter_and_rejects_bad_weights(env):
    assert not {"now", "clock", "logged_at", "ts", "time"} & set(inspect.signature(S.SignalLog.log).parameters)
    set_clock(env.mp, after_close(env.px.index[300]))
    for bad in (const_w(1.5), const_w(-0.2), const_w(float("nan")), lambda s, df: pd.Series(1.0, index=df.index[:-1], name="SYN")):
        with pytest.raises(ValueError): env.log.log(env.store, SPEC, env.px.iloc[:301], data_kind="synthetic", weights_fn=bad)

def test_strategy_code_never_sees_bars_after_as_of(env):
    seen = []
    def spy(spec, df): seen.append(df.index.max()); return pd.Series(1.0, index=df.index, name="SYN")
    set_clock(env.mp, after_close(env.px.index[300])); env.log.log(env.store, SPEC, env.px.iloc[:301], data_kind="synthetic", weights_fn=spy)
    assert seen == [env.px.index[300]]

def test_signal_log_tamper_detected(env):
    for k in (300, 301, 302): log_day(env.mp, env.log, env.store, env.px, k)
    L = env.log.chain._lines(); r = json.loads(L[1]); r["body"]["weights"]["SYN"] = 0.0; L[1] = canonical(r); _rewrite(env.log.chain, L)
    with pytest.raises(ChainError): env.log.records()
    with pytest.raises(ChainError): SC.score(env.log, SH, env.px.iloc[301:304])
    with pytest.raises(ChainError): log_day(env.mp, env.log, env.store, env.px, 303)

# ---------------------------------------------------------------- scoring
def _three_day(env, prices=(100.0, 110.0, 121.0, 108.9), weights=(1.0, 1.0, 0.0)):
    px = pd.DataFrame({"SYN": prices}, index=pd.bdate_range("2030-03-04", periods=len(prices)))     # SYNTHETIC, hand-checkable
    for k, w in enumerate(weights): log_day(env.mp, env.log, env.store, px, k, w=w)
    return px

def test_scoring_uses_only_bars_strictly_after_as_of_exact_numbers(env):
    px = _three_day(env); rep = SC.score(env.log, SH, px.iloc[1:]); d = rep["daily"]
    assert [x["date"] for x in d] == ["2030-03-05", "2030-03-06", "2030-03-07"]
    # paper: filled at the NEXT bar's close (110), so the +10% of bar 1 is NOT earned; only costs (1bp commission + 2bp slippage)
    k = 1 / (1.0002 * 1.0001)
    assert d[0]["paper_ret"] == pytest.approx(k * (1 - 1e-9) + 1e-9 - 1 + 0, abs=2e-8) and d[0]["paper_ret"] == pytest.approx(-3e-4, abs=1e-6)
    assert d[1]["paper_ret"] == pytest.approx(0.10, abs=1e-6)
    assert d[2]["paper_ret"] == pytest.approx(0.9 * 0.9998 * 0.9999 - 1, abs=1e-6)
    # vector-engine equivalent: weight at as_of earns the very next bar, 2bp * |dw|
    assert [x["vector_ret"] for x in d] == pytest.approx([0.10 - 2e-4, 0.10, -2e-4], abs=1e-12)
    f = rep["fills"]; assert [(x["date"], x["signal_as_of"]) for x in f] == [("2030-03-05", "2030-03-04"), ("2030-03-07", "2030-03-06")]
    assert all(pd.Timestamp(x["date"]) > pd.Timestamp(x["signal_as_of"]) for x in f) and f[0]["qty"] > 0 > f[1]["qty"]
    c = rep["costs"]; assert c["paper_bps_of_notional"] == pytest.approx(3.0, abs=0.01) and c["vector_assumed_bps"] == 2.0
    assert c["paper_total"] == pytest.approx(c["commission"] + c["slippage"]) and c["paper_minus_vector_cost"] == pytest.approx(c["traded_notional"] * 1e-4, rel=0.01)
    assert rep["paper"]["total_return"] == pytest.approx((1 + d[0]["paper_ret"]) * (1 + d[1]["paper_ret"]) * (1 + d[2]["paper_ret"]) - 1)
    assert rep["tracking"]["paper_minus_vector_total_return"] == pytest.approx(rep["paper"]["total_return"] - ((1.0998) * 1.1 * 0.9998 - 1))
    assert rep["n_signals_pending"] == 0 and rep["n_eligible_days"] == 0 and not rep["evidence"]["sufficient"]

def test_S1_scorer_refuses_bars_at_or_before_first_as_of(env):
    px = _three_day(env)
    with pytest.raises(SC.ForwardWindowError): SC.score(env.log, SH, px)                       # includes the as_of bar itself
    hist = pd.DataFrame({"SYN": 1.0}, index=pd.bdate_range("2020-01-06", "2030-03-08"))        # the whole protected window
    with pytest.raises(SC.ForwardWindowError, match="protected"): SC.score(env.log, SH, hist)

def test_S2_no_same_bar_fill(env):
    px = _three_day(env); sig = env.log.for_spec(SH)[0]; from engine.backtesting.paper import PaperBroker
    with pytest.raises(SC.SameBarFillError): SC._rebalance(PaperBroker(), sig, {"SYN": 1.0}, {"SYN": 100.0}, pd.Timestamp(sig["as_of"]))
    rep = SC.score(env.log, SH, px.iloc[1:])
    by = {x["signal_as_of"]: x["date"] for x in rep["fills"]}
    assert by["2030-03-04"] == "2030-03-05"                                                    # exactly the first bar after as_of

def test_incremental_reveal_never_revises_past_days_and_latest_signal_is_pending(env):
    px = env.px; k0 = 300; W = [1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.25, 0.0, 0.0, 1.0, 1.0]; prev = None
    for j, w in enumerate(W):
        log_day(env.mp, env.log, env.store, px, k0 + j, w=w)
        rep = SC.score(env.log, SH, px.iloc[k0 + 1:k0 + j + 1])                                # only bars revealed so far
        assert rep["n_scored_days"] == j and rep["n_signals_pending"] == 1 and rep["n_signals"] == j + 1
        if prev: assert rep["daily"][:len(prev["daily"])] == prev["daily"] and rep["fills"][:len(prev["fills"])] == prev["fills"]
        prev = rep
    assert prev["paper"]["n_fills"] >= 6 and prev["n_stale_days"] == 0
    # outcome bars alone drive realised P&L: changing a FUTURE bar changes nothing already scored
    px2 = px.copy(); px2.iloc[k0 + len(W):] *= 3
    assert SC.score(env.log, SH, px2.iloc[k0 + 1:k0 + len(W)])["daily"] == prev["daily"]

def test_first_scoring_is_empty_and_says_so(env):
    log_day(env.mp, env.log, env.store, env.px, 300)
    rep = SC.score(env.log, SH, env.px.iloc[0:0])
    assert rep["n_scored_days"] == 0 and rep["n_signals_pending"] == 1 and rep["paper"]["total_return"] is None and rep["paper"]["n_fills"] == 0
    assert not rep["evidence"]["sufficient"] and rep["report_hash"] == SC.report_hash(rep)

def test_missed_log_days_are_counted_stale_and_missing_bars_stay_missing(env):
    px = env.px.copy(); k0 = 300
    for k in (k0, k0 + 1, k0 + 4): log_day(env.mp, env.log, env.store, px, k)                  # no signal logged for bars k0+2, k0+3
    px.iloc[k0 + 5] = np.nan
    rep = SC.score(env.log, SH, px.iloc[k0 + 1:k0 + 7])
    assert rep["n_skipped_missing_bars"] == 1 and rep["n_scored_days"] == 5 and rep["n_stale_days"] == 2   # bars k0+3, k0+4 have no fresh signal; k0+5 is missing (not scored); k0+6 uses the k0+4 signal

def test_scoring_never_touches_protected_test_ledger(env):
    def boom(*a, **k): raise AssertionError("forward code touched the protected-test ledger")
    for m in ("record_test_access", "test_seen", "test_looks"): env.mp.setattr(Store, m, boom)
    for k in range(300, 306): log_day(env.mp, env.log, env.store, env.px, k)
    SC.score(env.log, SH, env.px.iloc[301:306], expectation=SC.backtest_expectation(env.store, SH))
    assert env.store.q("SELECT COUNT(*) n FROM test_access")[0]["n"] == 0
    src = "".join(p.read_text() for p in (LAB / "engine" / "forward").glob("*.py") if p.name != "demo.py")
    assert "record_test_access" not in src and '["test"]' not in src and "allow_test" not in src

def test_backtest_expectation_reads_validation_only(env):
    env.store.add_experiment(strategy_id=1, spec_hash=SH, criteria_hash="c", data_sha="d", code_version="v", seed=0, secs=0.0, decision="TEST_PASS", reasons=["synthetic"],
                             metrics={"validation": {"sharpe": 1.0, "ann_vol": 0.1}, "test": {"sharpe": 9.0, "ann_vol": 9.0}})
    e = SC.backtest_expectation(env.store, SH)
    assert e["sharpe_ann"] == 1.0 and e["daily_mean"] == pytest.approx(0.1 / 252) and "validation" in e["source"]
    assert SC.backtest_expectation(env.store, "nope") is None

# ---------------------------------------------------------------- end-to-end job path on a synthetic parquet cache
def _cache(tmp, px):
    from engine.data.ingest import YahooProvider, _key
    from engine.forward.jobs import START
    k = _key(YahooProvider.name, ["SPY"], START, None); px.to_parquet(tmp / f"{k}.parquet")
    (tmp / f"{k}.meta.json").write_text(json.dumps({"provider": "synthetic_gbm(TEST_ONLY)", "kind": "synthetic", "symbols": ["SPY"], "start": START, "end": None, "rows": len(px),
        "first_date": str(px.index[0].date()), "last_date": str(px.index[-1].date()), "fetched_at": after_close(px.index[-1]).isoformat(), "sha256": "synthetic"}))

def test_job_path_reads_only_forward_bars_poisoning_history_changes_nothing(env):
    """SYNTHETIC cache under tmp. After logging, every bar <= first as_of (which in production includes the whole protected
    window) is overwritten with garbage: the forward report must be identical, i.e. scoring reads none of it. No sockets."""
    import socket
    from engine.forward import jobs as J
    env.mp.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    spec = {"family": "trend", "asset": "SPY", "lookback_months": 3, "vol_target": None, "cost_bps": 2.0, "label": "SYNTHETIC_TEST_FIXTURE"}
    seed_candidate(env.store, spec, "SYNTHETIC-job"); full = synth().rename(columns={"SYN": "SPY"}); full.index.name = "Date"; k0 = 400; cache = env.tmp / "cache"; cache.mkdir()
    for k in range(k0, k0 + 6):
        _cache(cache, full.iloc[:k + 1]); set_clock(env.mp, after_close(full.index[k])); r = J.log_one(env.store, spec, cache_dir=cache)
        assert r["body"]["as_of"] == str(full.index[k].date()) and r["body"]["data_kind"] == "synthetic" and r["body"]["as_of_bar_final"]
    with pytest.raises(S.DuplicateSignal): J.log_one(env.store, spec, cache_dir=cache)
    a = J.score_one(env.store, spec, cache_dir=cache); assert a["n_scored_days"] == 5 and a["n_signals_pending"] == 1 and a["first_scored"] == str(full.index[k0 + 1].date())
    poisoned = full.iloc[:k0 + 6].copy(); poisoned.iloc[:k0 + 1] = 1e9; _cache(cache, poisoned)
    b = J.score_one(env.store, spec, cache_dir=cache, write=False); assert b == a
    got = SC.load_forward_bars(next(cache.glob("*.parquet")), ["SPY"], full.index[k0]); assert got.index.min() > full.index[k0] and len(got) == 5
    env.mp.setattr(J, "DATASETS", cache)
    assert json.loads(J.forward_log(env.store))["SYNTHETIC-job"] == "already logged for the latest bar" and "SYNTHETIC-job" in json.loads(J.forward_score(env.store))

def test_forward_code_has_no_network_broker_or_refresh_paths():
    for p in (LAB / "engine" / "forward").glob("*.py"):
        src = p.read_text()
        for tok in ("import requests", "urllib", "import socket", "yfinance", "refresh=True", ".fetch(", "api_key", "ccxt", "alpaca", "ib_insync", "http://", "https://"):
            assert tok not in src, f"{p.name} contains {tok!r}"
        assert src.count("load_prices(") == src.count("offline=True"), f"{p.name}: every load_prices call must be offline"

# ---------------------------------------------------------------- approval gate
def test_store_set_status_still_refuses_and_forward_code_never_calls_it_for_shadow(env):
    sid = env.store.q("SELECT id FROM strategies WHERE spec_hash=?", SH)[0]["id"]
    for s in ("SHADOW", "PAPER_APPROVED"):
        with pytest.raises((PermissionError, ValueError)): env.store.set_status(sid, s, "x")
    assert env.apr.status(SH) is None

def test_A2_approval_refused_without_terminal_and_no_env_or_flag_enables_it(env):
    for k, v in {"QAE_FORWARD_AUTO_APPROVE": "1", "QAE_APPROVE": "yes", "QAE_FORCE": "1", "FORCE_TTY": "1", "QAE_HUMAN": "1"}.items(): env.mp.setenv(k, v)
    with pytest.raises(A.ApprovalRefused, match="terminal"): env.apr.approve("ENTER_SHADOW", env.store, SH, "tester", "trying to approve from a test run")
    assert env.apr.chain.verify() == [] and env.apr.status(SH) is None and not env.apr.db_path.exists()
    assert set(inspect.signature(A.ApprovalLog.approve).parameters) == {"self", "kind", "store", "spec_hash", "approver", "reason", "outcome_px", "expectation"}
    from engine.forward import cli
    assert cli.main(["approve", "enter-shadow", "--spec-hash", SH, "--approver", "t", "--reason", "non interactive attempt"]) == 2

def test_A1_env_markers_refuse(env):
    for k in A.AUTOMATION_ENV:
        env.mp.setenv(k, "1")
        with pytest.raises(A.ApprovalRefused, match="environment marker"): env.apr.approve("ENTER_SHADOW", env.store, SH, "tester", "scheduler context attempt")
        env.mp.delenv(k)

@pytest.mark.parametrize("mod", ["engine.orchestration.scheduler", "engine.orchestration.jobs", "engine.agents.roles", "engine.api.app", "engine.forward.jobs", "engine.coding.x"])
def test_A1_automated_call_stack_refuses(env, mod):
    fake = types.ModuleType(mod); exec("def handler(apr, store, sh):\n    return apr.approve('ENTER_SHADOW', store, sh, 'job', 'automatic promotion attempt')\n", fake.__dict__)
    with pytest.raises(A.ApprovalRefused, match="automated code path"): fake.handler(env.apr, env.store, SH)
    assert env.apr.chain.verify() == []

def test_real_scheduler_job_cannot_approve(env):
    from engine.orchestration.scheduler import Scheduler
    cfg = env.tmp / "s.toml"; cfg.write_text("[jobs.promote]\ninterval_s = 1\n"); seen = {}
    def promote(store):
        try: env.apr.approve("ENTER_SHADOW", store, SH, "scheduler", "automatic promotion attempt")
        except A.ApprovalRefused as e: seen["why"] = str(e); raise
    assert Scheduler(env.store, {"promote": promote}, cfg).run_job("promote") == "failed" and "QAE_JOB_CONTEXT" in seen["why"]   # scheduler env marker fires first
    assert env.apr.chain.verify() == []

def test_real_scheduler_job_cannot_approve_even_if_markers_stripped(env):
    """CTO integration: a job that strips the scheduler's env markers is still refused by the call-stack check."""
    import os
    from engine.orchestration.scheduler import Scheduler
    cfg = env.tmp / "s.toml"; cfg.write_text("[jobs.promote]\ninterval_s = 1\n"); seen = {}
    def promote(store):
        for k in A.AUTOMATION_ENV: os.environ.pop(k, None)
        try: env.apr.approve("ENTER_SHADOW", store, SH, "scheduler", "automatic promotion attempt")
        except A.ApprovalRefused as e: seen["why"] = str(e); raise
    assert Scheduler(env.store, {"promote": promote}, cfg).run_job("promote") == "failed" and "automated code path engine.orchestration.scheduler" in seen["why"]
    assert env.apr.chain.verify() == []

def test_status_mirror_is_not_trusted(env):
    with env.apr._db() as d: d.execute("INSERT INTO forward_status VALUES(?,?,?,?,?,?)", (SH, 1, "PAPER_APPROVED", 0, "x", "t"))
    with pytest.raises(ChainError, match="disagrees"): env.apr.status(SH)
    with pytest.raises(ChainError): log_day(env.mp, env.log, env.store, env.px, 300)             # logging refuses too

def test_forged_chain_record_with_illegal_transition_is_rejected(env):
    """No secret key exists, so code that writes the chain file directly CAN forge a well-formed record (documented limit).
    What is still enforced: transition order - PAPER_APPROVED without a prior ENTER_SHADOW never folds to a status."""
    env.apr.chain.append({"type": "approval", "kind": "PAPER_APPROVED", "strategy_id": 1, "spec_hash": SH, "ts": "t", "signal_chain": {"n": 0, "head": GENESIS}})
    with pytest.raises(ChainError, match="illegal forward transition"): env.apr.status(SH)

def test_evidence_rule_is_frozen_and_hashed():
    ev = SC.ForwardEvidence()
    with pytest.raises(Exception): ev.min_forward_days = 1
    assert ev.digest() != SC.ForwardEvidence(min_forward_days=1).digest() and S.LogPolicy().digest() != S.LogPolicy(session_close="23:59").digest()
    assert ev.min_forward_days == 60 and ev.min_calendar_days == 84

# --- the real interactive path, driven through a pseudo-terminal (this is what a human does; see docs/FORWARD.md on its limits)
def run_pty(argv, extra_env=None, respond="correct", timeout=300):
    envv = {k: v for k, v in os.environ.items() if k not in A.AUTOMATION_ENV}; envv.update(extra_env or {})
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(LAB); os.execve(sys.executable, [sys.executable, *argv], envv)
    buf, answered, prompts = b"", 0, 0
    while True:
        r, _, _ = select.select([fd], [], [], timeout)
        if not r: os.kill(pid, 9); raise TimeoutError(buf.decode(errors="replace")[-2000:])
        try: chunk = os.read(fd, 65536)
        except OSError: chunk = b""
        if not chunk: break
        buf += chunk
        ms = re.findall(rb"Type exactly: (APPROVE [A-Z_]+ [0-9a-f]+ [0-9a-f]+)\r?\n> ", buf)
        while answered < len(ms):
            os.write(fd, (ms[answered] if respond == "correct" else b"APPROVE yes") + b"\n"); answered += 1
    _, st = os.waitpid(pid, 0); os.close(fd)
    return os.waitstatus_to_exitcode(st), buf.decode(errors="replace"), answered

def test_pty_cli_enter_shadow_records_audited_approval_and_wrong_phrase_does_not(tmp_path):
    st = Store(tmp_path / "qae.db"); seed_candidate(st); home = S.forward_home(st); apr = A.ApprovalLog(home)
    argv = ["-m", "engine.forward.cli", "approve", "enter-shadow", "--spec-hash", SH, "--approver", "TEST-PTY (not a human)", "--reason", "pty-driven test of the interactive path"]
    code, out, n = run_pty(argv, {"QAE_HOME": str(tmp_path)}, respond="wrong")
    assert code == 2 and n == 1 and "did not match" in out and apr.chain.verify() == [] and apr.status(SH) is None
    code, out, n = run_pty(argv, {"QAE_HOME": str(tmp_path)})
    assert code == 0 and n == 1 and "recorded ENTER_SHADOW" in out, out
    (rec,) = apr.chain.verify(); b = rec["body"]
    assert apr.status(SH) == "SHADOW" and b["approver"] == "TEST-PTY (not a human)" and b["spec_hash"] == SH and b["kind"] == "ENTER_SHADOW" and b["reason"] and b["ts"]
    assert b["evidence_rule_hash"] == SC.ForwardEvidence().digest() and b["proof"]["tty"].startswith("/dev/") and len(b["proof"]["challenge_sha"]) == 64
    assert st.q("SELECT status FROM strategies WHERE spec_hash=?", SH)[0]["status"] == "CANDIDATE"       # research store untouched
    with sqlite3.connect(apr.db_path) as d: assert d.execute("SELECT status FROM forward_status").fetchall() == [("SHADOW",)]
    code, out, n = run_pty(argv, {"QAE_HOME": str(tmp_path)})                                            # SHADOW -> SHADOW is illegal
    assert code == 2 and n == 0 and "illegal forward transition" in out and len(apr.chain.verify()) == 1
    code, out, n = run_pty(argv, {"QAE_HOME": str(tmp_path / "other"), "QAE_JOB_CONTEXT": "1"})          # a TTY does not override A1
    assert code == 2 and n == 0 and "environment marker" in out

def child_lifecycle(tmp):   # runs INSIDE the pty child; SYNTHETIC data declared 'real' ONLY so the evidence-positive path can be exercised in a tmp dir
    st = Store(Path(tmp) / "qae.db"); seed_candidate(st); home = S.forward_home(st); log = S.SignalLog(home); apr = A.ApprovalLog(home); px = synth(); k0 = 300; res = {}
    class MP:
        def setattr(self, o, n, v): setattr(o, n, v)
    mp = MP(); exp = {"daily_mean": 0.0005, "daily_sd": 0.005}
    for k in range(k0, k0 + 3): log_day(mp, log, st, px, k, kind="real")                                 # logged BEFORE the shadow approval: not eligible
    apr.approve("ENTER_SHADOW", st, SH, "TEST-PTY (not a human)", "synthetic lifecycle test, enter shadow")   # prompt 1 (typed by the pty parent)
    for k in range(k0 + 3, k0 + 40): log_day(mp, log, st, px, k, kind="real")
    try: apr.approve("PAPER_APPROVED", st, SH, "TEST-PTY (not a human)", "too early, must be refused", outcome_px=px.iloc[k0 + 1:k0 + 40], expectation=exp)
    except A.InsufficientEvidence as e: res["early"] = str(e)
    try: apr.approve("PAPER_APPROVED", st, SH, "TEST-PTY (not a human)", "no outcome bars, must be refused")
    except A.InsufficientEvidence as e: res["no_bars"] = str(e)
    for k in range(k0 + 40, k0 + 70): log_day(mp, log, st, px, k, kind="real")
    out = px.iloc[k0 + 1:k0 + 70]; rep = SC.score(log, SH, out, expectation=exp); res["rep"] = {k: rep[k] for k in ("n_scored_days", "n_eligible_days", "evidence", "report_hash")}
    try: apr.approve("PAPER_APPROVED", st, SH, "TEST-PTY (not a human)", "no expectation supplied, must be refused", outcome_px=out)
    except A.InsufficientEvidence as e: res["no_expectation"] = str(e)
    rec = apr.approve("PAPER_APPROVED", st, SH, "TEST-PTY (not a human)", "synthetic lifecycle test, paper approval", outcome_px=out, expectation=exp)   # prompt 2
    res["paper"] = {"evidence_hash": rec["body"]["evidence_hash"], "signal_chain": rec["body"]["signal_chain"], "status": apr.status(SH)}
    res["bodies"] = [r["body"]["forward_status"] for r in log.records()]
    try: log_day(mp, log, st, px, k0 + 70, kind="real"); res["log_after_paper"] = log.for_spec(SH)[-1]["forward_status"]
    except Exception as e: res["log_after_paper"] = repr(e)
    # consistent full rewrite of the signal log (valid chain, fewer records): caught only by the anchors stored in approvals
    bodies = [r["body"] for r in log.records()][:20]; log.chain.path.unlink(); log.chain.head_path.unlink()
    for b in bodies: log.chain.append(b)
    log.chain.verify()
    try: log.records(); res["rewrite_detected"] = False
    except ChainError as e: res["rewrite_detected"] = str(e)
    print("RESULT:" + json.dumps(res))

def test_pty_full_lifecycle_evidence_gate_and_anchors(tmp_path):
    code, out, n = run_pty(["-c", f"import sys; sys.path.insert(0, 'tests'); import test_forward as T; T.child_lifecycle({str(tmp_path)!r})"])
    assert code == 0, out[-3000:]
    res = json.loads(re.search(r"RESULT:(\{.*\})", out).group(1))
    assert n == 2                                                                # the three refused requests never reached the human prompt
    assert "eligible forward days 36 < 60" in res["early"] and "calendar span" in res["early"] and "outcome bars" in res["no_bars"] and "expectation" in res["no_expectation"]
    assert res["rep"]["n_scored_days"] == 69 and res["rep"]["n_eligible_days"] == 66 and res["rep"]["evidence"]["sufficient"]
    assert res["paper"]["status"] == "PAPER_APPROVED" and res["paper"]["evidence_hash"] == res["rep"]["report_hash"] and res["paper"]["signal_chain"]["n"] == 70
    assert res["bodies"] == ["CANDIDATE"] * 3 + ["SHADOW"] * 67 and res["log_after_paper"] == "PAPER_APPROVED"
    assert res["rewrite_detected"] and "externally anchored" in res["rewrite_detected"]

def test_pty_does_not_override_automated_call_stack(tmp_path):
    code = ("import sys, types; sys.path.insert(0, 'tests'); import test_forward as T\n"
            f"st = T.Store({str(tmp_path / 'qae.db')!r}); T.seed_candidate(st); apr = T.A.ApprovalLog(T.S.forward_home(st))\n"
            "m = types.ModuleType('engine.orchestration.jobs'); exec('def f(apr, st, sh):\\n    return apr.approve(\"ENTER_SHADOW\", st, sh, \"job\", \"automatic promotion attempt\")', m.__dict__)\n"
            "try: m.f(apr, st, T.SH)\nexcept T.A.ApprovalRefused as e: print('REFUSED-OK', e)\n")
    rc, out, n = run_pty(["-c", code])
    assert rc == 0 and n == 0 and "REFUSED-OK" in out and "automated code path" in out
    assert A.ApprovalLog(tmp_path / "forward").chain.verify() == []

def test_S3_scorer_rechecks_wallclock_rule_on_stored_records(env):
    """A record written straight into the chain file (bypassing log()) with a logged_at after its outcome is refused at scoring."""
    r = log_day(env.mp, env.log, env.store, env.px, 300); late = {**r["body"], "as_of": str(env.px.index[301].date()), "logged_at": "2031-06-01T22:00:00+00:00"}
    env.log.chain.append(late)
    with pytest.raises(ChainError, match="wall-clock"): SC.score(env.log, SH, env.px.iloc[301:304])

def test_only_research_candidates_can_enter_forward_evaluation(env):
    spec2 = {**SPEC, "lookback_months": 4}; env.store.get_or_create_strategy(spec2, "idea")
    with pytest.raises(A.ApprovalRefused, match="research-store status is IDEA"): env.apr.approve("ENTER_SHADOW", env.store, h(spec2), "tester", "not a candidate, must be refused")
    with pytest.raises(A.ApprovalRefused, match="not in the research store"): env.apr.approve("ENTER_SHADOW", env.store, "feedfacefeedface", "tester", "unknown spec, must be refused")

def test_pty_with_piped_stdin_is_refused(tmp_path):
    """Controlling terminal present but stdin redirected (scripted invocation): refused before any prompt."""
    st = Store(tmp_path / "qae.db"); seed_candidate(st)
    envv = {k: v for k, v in os.environ.items() if k not in A.AUTOMATION_ENV}; envv["QAE_HOME"] = str(tmp_path)
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(LAB); os.execve("/bin/sh", ["/bin/sh", "-c", f"'{sys.executable}' -m engine.forward.cli approve enter-shadow --spec-hash {SH} --approver x --reason 'stdin is not a terminal here' < /dev/null"], envv)
    buf = b""
    while True:
        if not select.select([fd], [], [], 60)[0]: os.kill(pid, 9); os.waitpid(pid, 0); pytest.fail("approval waited for input instead of refusing: " + buf.decode(errors="replace")[-300:])
        try: ch = os.read(fd, 65536)
        except OSError: ch = b""
        if not ch: break
        buf += ch
    _, s = os.waitpid(pid, 0); os.close(fd)
    assert os.waitstatus_to_exitcode(s) == 2 and b"not a TTY" in buf and b"Type exactly" not in buf
    assert A.ApprovalLog(tmp_path / "forward").chain.verify() == []
