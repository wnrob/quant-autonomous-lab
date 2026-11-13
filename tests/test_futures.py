"""Tests for engine/futures. All per-contract price data here is SYNTHETIC (labelled), built for software tests only."""
import datetime as dt
import numpy as np, pandas as pd, pytest
from engine.errors import DataError
from engine.futures import calendar as cal
from engine.futures.adjust import build_continuous, unadjusted_splice
from engine.futures.carry import front_next, roll_yield
from engine.futures.rules import CalendarRoll, VolumeRoll, held_contracts, roll_events
from engine.futures.specs import SPECS, ContractId, parse_ticker, specs_table
from engine.futures.stale import detect_stale

D = dt.date


# ---------------- calendar / expiry ----------------
@pytest.mark.parametrize("root,y,m,expected", [
    ("MES", 2024, 12, D(2024, 12, 20)), ("MES", 2025, 3, D(2025, 3, 21)), ("MNQ", 2026, 3, D(2026, 3, 20)),
    ("CL", 2024, 12, D(2024, 11, 20)),   # 25 Nov 2024 is a Monday -> 3 bdays before
    ("CL", 2025, 1, D(2024, 12, 19)),    # 25 Dec holiday -> 24 Dec -> 3 bdays before
    ("NG", 2025, 1, D(2024, 12, 27)),    # 3 bdays before 1 Jan (31 Dec, 30 Dec, 27 Dec)
    ("GC", 2024, 12, D(2024, 12, 27)),   # third-last business day of Dec 2024
    ("ZN", 2024, 12, D(2024, 12, 19)),   # 7 bdays before 31 Dec 2024 (skips 25 Dec)
    ("6E", 2024, 12, D(2024, 12, 16)),   # 3rd Wed 18 Dec - 2 bdays
    ("ZC", 2024, 12, D(2024, 12, 13)),   # bday before 15 Dec (a Sunday)
])
def test_last_trade_known_dates(root, y, m, expected):
    assert SPECS[root].last_trade(y, m) == expected


def test_holidays_and_bdays():
    assert D(2025, 4, 18) in cal.holidays(2025)      # Good Friday 2025
    assert D(2024, 3, 29) in cal.holidays(2024)      # Good Friday 2024
    assert D(2025, 6, 19) in cal.holidays(2025) and D(2021, 6, 18) not in cal.holidays(2021)
    assert D(2026, 7, 3) in cal.holidays(2026)       # 4 Jul 2026 is Saturday -> observed Fri
    assert D(2021, 12, 31) not in cal.holidays(2021)  # Sat New Year not observed on prior Dec 31
    assert cal.add_business_days(D(2024, 12, 24), -3) == D(2024, 12, 19)
    assert cal.business_days_between(D(2024, 12, 19), D(2024, 12, 27)) == 5   # 20,23,24,26,27


def test_specs_table_and_tick_values():
    t = {r["root"]: r for r in specs_table()}
    assert set(t) >= {"6A", "6B", "6E", "6J", "CL", "GC", "HG", "M2K", "MES", "NG", "SI", "ZB", "ZC", "ZN", "MNQ", "MGC"}
    assert t["MES"]["tick_value"] == 1.25 and t["MNQ"]["tick_value"] == 0.5 and t["CL"]["tick_value"] == 10.0
    assert t["ZN"]["tick_value"] == 15.625 and t["ZB"]["tick_value"] == 31.25 and t["6E"]["tick_value"] == 6.25
    assert all(r["source"] == "CME contract specs as known to the agent; verify before trading" for r in t.values())
    with pytest.raises(ValueError): SPECS["MES"].last_trade(2025, 1)


def test_parse_ticker_resolves_decade():
    assert parse_ticker("MESZ4", "MES", D(2024, 9, 1)) == ContractId(2024, 12, "MES")
    assert parse_ticker("MESH5", "MES", D(2024, 12, 20)) == ContractId(2025, 3, "MES")
    assert parse_ticker("CLF0", "CL", D(2029, 12, 1)) == ContractId(2030, 1, "CL")
    # chronological, not alphabetical
    assert ContractId(2025, 12, "MNQ") < ContractId(2026, 3, "MNQ") and "MNQZ5" > "MNQH6"
    with pytest.raises(ValueError): parse_ticker("MESQ", "MES", D(2024, 1, 1))


# ---------------- synthetic per-contract fixtures (TEST ONLY) ----------------
def synthetic_chain(root="MES", start="2024-06-03", end="2025-06-30", seed=0, spread=0.01, holes=(), idio=0.0):
    """SYNTHETIC per-contract closes: a common random walk plus a per-contract premium that decays to 0 at expiry
    (contango of `spread` per quarter). Every contract trades every business day until its last trade."""
    idx = pd.DatetimeIndex([d for d in pd.bdate_range(start, end) if cal.is_business_day(d.date())])
    rng = np.random.default_rng(seed); spot = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    chain = [ContractId(y, m, root) for y in range(s.year, e.year + 2) for m in SPECS[root].listed_months]
    chain = [c for c in chain if c.last_trade() > s and c.last_trade() <= e + dt.timedelta(days=200)]
    px = {}
    for c in chain:
        tau = np.array([(c.last_trade() - t.date()).days for t in idx]) / 91.0
        p = spot * np.exp(spread * np.clip(tau, 0, None) + np.cumsum(rng.normal(0, idio, len(idx))) * (idio > 0))
        p[np.array([t.date() > c.last_trade() for t in idx])] = np.nan
        px[c] = p
    df = pd.DataFrame(px, index=idx)
    for t, c in holes: df.at[pd.Timestamp(t), c] = np.nan
    return df


def test_rule_frozen_and_hashed():
    r = CalendarRoll(5)
    with pytest.raises(Exception): r.n_bdays = 3
    assert r.digest() == CalendarRoll(5).digest() != CalendarRoll(4).digest()
    assert VolumeRoll(1).digest() != VolumeRoll(2).digest() != r.digest()


def test_calendar_roll_dates():
    px = synthetic_chain()
    held = held_contracts(px.index, list(px.columns), CalendarRoll(5))
    ev = roll_events(held)
    z4 = ContractId(2024, 12, "MES")
    e = [x for x in ev if x[2] == z4][0]
    assert e[1] == ContractId(2024, 9, "MES")
    assert e[0].date() == cal.add_business_days(D(2024, 9, 20), -5) == D(2024, 9, 13)
    for ts, a, b in ev: assert b > a and ts.date() < a.last_trade()            # never held into expiry


def test_backadjusted_returns_equal_held_contract_returns():
    px = synthetic_chain(idio=0.003)                       # contracts' own returns differ -> test can tell them apart
    held = held_contracts(px.index, list(px.columns), CalendarRoll(5))
    for method in ("ratio", "difference"):
        cs = build_continuous(px, held, method)
        adj_ret = cs["adjusted"].pct_change(fill_method=None); adj_pts = cs["adjusted"].diff()
        for i in range(1, len(px)):
            t, tp, c = px.index[i], px.index[i - 1], held.iloc[i]
            own_ret = px.at[t, c] / px.at[tp, c] - 1; own_pts = px.at[t, c] - px.at[tp, c]
            if method == "ratio": assert adj_ret.iloc[i] == pytest.approx(own_ret, abs=1e-12)
            else: assert adj_pts.iloc[i] == pytest.approx(own_pts, abs=1e-9)
        # latest segment is real traded prices
        last = cs.index[-1]; assert cs.at[last, "adjusted"] == pytest.approx(px.at[last, held.iloc[-1]])


def test_roll_day_has_no_spurious_jump_but_naive_splice_does():
    px = synthetic_chain(spread=0.03, idio=0.003)
    held = held_contracts(px.index, list(px.columns), CalendarRoll(5))
    cs = build_continuous(px, held, "ratio")
    naive = unadjusted_splice(px, held).pct_change(fill_method=None)
    for ts, a, b in roll_events(held):
        i = px.index.get_loc(ts); tp = px.index[i - 1]
        new_own = px.at[ts, b] / px.at[tp, b] - 1
        assert cs["held_return"].iloc[i] == pytest.approx(new_own)
        assert cs["adjusted"].pct_change(fill_method=None).iloc[i] == pytest.approx(new_own)
        old_own = px.at[ts, a] / px.at[tp, a] - 1
        assert abs(new_own - old_own) > 1e-4                   # fixture sanity: old and new contract returns differ
        jump = naive.iloc[i] - new_own                       # naive splice adds the calendar spread
        assert abs(jump) > 0.02


def test_missing_bars_stay_nan_never_filled():
    px = synthetic_chain(holes=[("2024-10-15", ContractId(2024, 12, "MES"))])
    held = held_contracts(px.index, list(px.columns), CalendarRoll(5))
    cs = build_continuous(px, held, "ratio")
    t = pd.Timestamp("2024-10-15"); nxt = px.index[px.index.get_loc(t) + 1]
    assert np.isnan(cs.at[t, "raw_close"]) and np.isnan(cs.at[t, "adjusted"])
    assert np.isnan(cs.at[t, "held_return"]) and np.isnan(cs.at[nxt, "held_return"])
    assert int(cs["adjusted"].isna().sum()) == 1 and len(cs) == len(px)     # no row dropped, no row invented
    yr = roll_yield(px); assert np.isnan(yr.at[t, "roll_yield_ann"]) and yr["roll_yield_ann"].isna().sum() >= 1


def test_roll_without_overlap_raises():
    px = synthetic_chain()
    held = held_contracts(px.index, list(px.columns), CalendarRoll(5))
    ts, a, b = roll_events(held)[0]
    px.at[px.index[px.index.get_loc(ts) - 1], b] = np.nan    # new contract not observed the day before the roll
    with pytest.raises(DataError): build_continuous(px, held, "ratio")


def test_volume_roll_has_no_lookahead():
    px = synthetic_chain()
    chain = list(px.columns); rng = np.random.default_rng(1)
    vol = pd.DataFrame(rng.integers(100, 1000, px.shape).astype(float), index=px.index, columns=chain)
    rule = VolumeRoll(confirm=2)
    base = held_contracts(px.index, chain, rule, vol)
    for k in (40, 90, 150, 200):
        cut = px.index[k]
        v2 = vol.copy(); v2.loc[cut:] = rng.integers(0, 10**6, v2.loc[cut:].shape)   # scramble info dated >= cut
        h2 = held_contracts(px.index, chain, rule, v2)
        assert (h2.loc[:cut] == base.loc[:cut]).all()        # decision at `cut` uses only data < cut
    # and it is monotone (never rolls back)
    keys = [(c.year, c.month) for c in base]; assert keys == sorted(keys)


def test_volume_roll_crossover_timing():
    px = synthetic_chain(start="2024-08-01", end="2024-10-31")
    u4, z4 = ContractId(2024, 9, "MES"), ContractId(2024, 12, "MES")
    vol = pd.DataFrame(np.nan, index=px.index, columns=list(px.columns)); vol[u4] = 1000.0; vol[z4] = 10.0
    cross = pd.Timestamp("2024-09-10"); vol.loc[cross:, z4] = 5000.0
    held = held_contracts(px.index, list(px.columns), VolumeRoll(confirm=1), vol)
    first_z = held[held == z4].index[0]
    assert first_z == px.index[px.index.get_loc(cross) + 1]  # crossover seen at close of 9/10 -> held from 9/11


def test_roll_yield_sign_and_value():
    px = synthetic_chain(spread=0.02)                      # contango -> negative roll yield
    yr = roll_yield(px).dropna(subset=["roll_yield_ann"])
    assert (yr["roll_yield_ann"] < 0).all()
    exp = -0.02 / 91 * 365                                   # per construction, ln(F1/F2) = -0.02*dtau
    assert yr["roll_yield_ann"].median() == pytest.approx(exp, rel=1e-6)
    f, n = front_next(D(2024, 12, 20), list(px.columns))     # on expiry day the expiring contract is no longer 'front'
    assert f == ContractId(2025, 3, "MES") and n == ContractId(2025, 6, "MES")


# ---------------- stale-contract detector ----------------
def fetch_full_reconstruction(seed=3):
    """SYNTHETIC reconstruction of the user's fetch_full bug (README.md:10-15 of the user's project; original source
    convexity/delta/data.py:95 no longer on disk): every quarterly contract fetched over the FULL window, concatenated in
    ticker order, deduplicated with keep='first'. The older contract keeps printing thin bars at a different price
    after the market has rolled, and wins every minute it prints."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-02 09:30", "2026-03-27 16:00", freq="1min", tz="America/New_York")
    idx = idx[(idx.time >= dt.time(9, 30)) & (idx.time <= dt.time(16, 0)) & (idx.day_of_week < 5)]
    spot = 20000 + np.cumsum(rng.normal(0, 5, len(idx)))
    frames = []
    for tk, prem, alive_to in [("MNQH6", 0.0, "2026-03-20"), ("MNQM6", 200.0, "2026-06-19"), ("MNQU6", 400.0, "2026-09-18")]:
        f = pd.DataFrame({"close": spot + prem, "volume": 2000.0, "contract": tk}, index=idx)
        f = f[f.index.date <= pd.Timestamp(alive_to).date()]
        if tk == "MNQH6":   # dead month after the roll: prints sporadic 1-lot bars
            after = f.index.date >= D(2026, 3, 12); keep = ~after | (rng.random(len(f)) < 0.3)
            f = f[keep]; f.loc[f.index.date >= D(2026, 3, 12), "volume"] = 1.0
        if tk == "MNQU6": f = f[rng.random(len(f)) < 0.05].assign(volume=1.0)
        frames.append(f)
    allc = pd.concat(frames)
    return allc[~allc.index.duplicated(keep="first")].sort_index()


def clean_splice():
    """SYNTHETIC correct splice of the same walk: H6 before the 12 Mar roll, M6 after (one contract per bar)."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-03-02 09:30", "2026-03-27 16:00", freq="1min", tz="America/New_York")
    idx = idx[(idx.time >= dt.time(9, 30)) & (idx.time <= dt.time(16, 0)) & (idx.day_of_week < 5)]
    spot = 20000 + np.cumsum(rng.normal(0, 5, len(idx)))
    pre = idx.date < D(2026, 3, 12)
    return pd.DataFrame({"close": np.where(pre, spot, spot + 200.0), "volume": 2000.0,
                         "contract": np.where(pre, "MNQH6", "MNQM6")}, index=idx)


def test_detector_catches_fetch_full_reconstruction():
    bad = fetch_full_reconstruction()
    rep = detect_stale(bad, "MNQ", rule=CalendarRoll(5, ref="last_trade"))
    checks = {f["check"] for f in rep.errors}
    assert not rep.ok and "back_switch" in checks
    assert rep.counts["back_switch"] > 100                     # interleaving, not a single roll
    assert rep.counts["past_roll_date"] > 0
    assert rep.counts.get("thin_volume", 0) > 0


def test_detector_passes_clean_splice_and_flags_raw_roll_gap_as_info():
    good = clean_splice()
    rep = detect_stale(good, "MNQ", rule=CalendarRoll(5, ref="last_trade"))
    assert rep.ok, rep.flags
    assert rep.counts["back_switch"] == 0 and rep.counts["forward_switches"] == 1 and rep.counts["past_last_trade"] == 0
    assert rep.counts["unadjusted_roll_gap"] == 1                 # raw splice: 200pt gap visible, info only
    rep2 = detect_stale(good, "MNQ", adjusted=True)               # same series CLAIMED adjusted -> error
    assert not rep2.ok and rep2.errors[0]["check"] == "unadjusted_roll_gap"


def test_detector_past_expiry_and_stale_end_and_multi():
    idx = pd.bdate_range("2024-12-16", "2024-12-31")
    df = pd.DataFrame({"close": np.linspace(100, 101, len(idx)), "contract": "MESZ4"}, index=idx)  # Z4 expired 20 Dec
    rep = detect_stale(df, "MES", panel_end="2025-01-31")
    assert rep.counts["past_last_trade"] == len(idx[idx.date > D(2024, 12, 20)]) and rep.counts["stale_end"] == 1
    dup = pd.concat([df.iloc[:3], df.iloc[:3].assign(contract="MESH5")])
    assert detect_stale(dup, "MES").counts["multi_contract_timestamp"] == 3
    frozen = df.iloc[:6].assign(close=100.0, contract="MESH5")
    assert detect_stale(frozen, "MES").counts["frozen_price"] == 1
    assert not detect_stale(df.drop(columns="contract"), "MES").ok


def test_detector_never_mutates_input():
    bad = fetch_full_reconstruction(); before = bad.copy()
    detect_stale(bad, "MNQ", rule=CalendarRoll(5))
    pd.testing.assert_frame_equal(bad, before)


# ---------------- the qa/roll.py ratchet hazard found on the real 6E/GC data ----------------
def _qa_style_ratchet(volume: pd.DataFrame) -> pd.Series:
    """Faithful re-statement (for tests) of the user's qa/roll.py daily_volume(.fillna(0)) + active_by_date: same-day
    argmax over contracts at/after a floor that only moves forward. NOT used by engine/futures."""
    v = volume.fillna(0.0); cols = list(v.columns); floor, out = 0, []
    for _, row in v.iterrows():
        el = row.iloc[floor:]
        if el.sum() == 0: out.append(out[-1] if out else cols[floor]); continue
        pick = el.idxmax(); floor = cols.index(pick); out.append(pick)
    return pd.Series(out, index=v.index)


def test_missing_front_bar_ratchet_is_flagged_and_volume_roll_resists_it():
    """SYNTHETIC: 6E-like chain; on one day the front month (M6) has no bar. The qa-style ratchet jumps to the thin
    next contract (U6, ~100 lots) and never returns; detect_stale flags the thin run; VolumeRoll does not jump."""
    idx = pd.DatetimeIndex([d for d in pd.bdate_range("2025-09-01", "2026-06-12") if cal.is_business_day(d.date())])
    m6, u6 = ContractId(2026, 6, "6E"), ContractId(2026, 9, "6E")
    vol = pd.DataFrame({m6: 150_000.0, u6: 100.0}, index=idx)
    gap = pd.Timestamp("2026-03-27"); vol.at[gap, m6] = np.nan
    vol.loc[pd.Timestamp("2026-06-08"):, u6] = 200_000.0                  # real migration a week before IMM expiry
    qa_held = _qa_style_ratchet(vol)
    assert (qa_held.loc[gap:] == u6).all()                                # ratchet: stuck on the thin contract
    series = pd.DataFrame({"close": 1.15, "volume": [vol.at[t, c] for t, c in qa_held.items()],
                           "contract": [str(c) for c in qa_held]}, index=idx)
    series["close"] = 1.15 + np.arange(len(idx)) * 1e-4
    rep = detect_stale(series, "6E")
    assert not rep.ok and rep.counts["thin_segment"] >= 40
    good = held_contracts(idx, [m6, u6], VolumeRoll(confirm=1), vol)
    assert good.loc[gap] == m6 and good.loc[:pd.Timestamp("2026-06-08")].eq(m6).all()
    assert good.iloc[-1] == u6                                            # it does roll once volume really moves


def test_thin_segment_ignores_short_quiet_runs():
    idx = pd.bdate_range("2025-01-01", periods=60)
    v = np.full(60, 1000.0); v[10:13] = 1.0                              # 3 quiet sessions: below the 5-row bar
    df = pd.DataFrame({"close": np.linspace(100, 110, 60), "volume": v, "contract": "MESH5"}, index=idx)
    assert detect_stale(df, "MES").counts.get("thin_segment", 0) == 0
