import numpy as np, pandas as pd, pytest
from engine.data import pit as P
from engine.data.ingest import check_availability
from engine.data.universe import ETF24, UNIVERSES

def table():   # SYNTHETIC fixture: software tests only
    mk = lambda av, dl=None: {"availability_date": av, "delisting": dl}
    return {"version": 0, "universes": {"t": {"survivorship_free": False, "members": {
        "OLD": mk("2005-01-03"), "NEW": mk("2007-04-11"), "GONE": mk("2005-01-03", {"last_trade_date": "2009-06-30", "successor": "OLD"})}}}}

def frame(start_new="2007-04-11"):
    idx = pd.bdate_range("2005-01-03", "2010-12-31"); df = pd.DataFrame(100.0, index=idx, columns=["OLD", "NEW", "GONE"])
    df.loc[df.index < start_new, "NEW"] = np.nan; df.loc[df.index > "2009-06-30", "GONE"] = np.nan; return df

def test_universe_at_excludes_unlisted_and_delisted():
    t = table()
    assert P.universe_at("2006-01-02", "t", t) == ["OLD", "GONE"]
    assert P.universe_at("2007-04-10", "t", t) == ["OLD", "GONE"] and "NEW" in P.universe_at("2007-04-11", "t", t)
    assert "GONE" in P.universe_at("2009-06-30", "t", t) and "GONE" not in P.universe_at("2009-07-01", "t", t)

def test_pit_prices_slices_future_and_nonmembers():
    t = table(); out = P.pit_prices(frame(), "2006-06-01", universe="t", pit=t)
    assert list(out.columns) == ["OLD", "GONE"] and out.index.max() <= pd.Timestamp("2006-06-01")

def test_pit_prices_raises_on_nonmember_request():
    with pytest.raises(P.PITViolation): P.pit_prices(frame(), "2006-06-01", ["OLD", "NEW"], universe="t", pit=table())
    with pytest.raises(P.PITViolation): P.pit_prices(frame(), "2010-01-04", ["GONE"], universe="t", pit=table())

def test_pit_prices_raises_on_pre_inception_value():
    bad = frame(start_new="2006-01-02")                       # NEW has data 15 months before availability
    with pytest.raises(P.PITViolation, match="pre-inception"): P.pit_prices(bad, "2008-01-02", universe="t", pit=table())

def test_validate_frame_flags_never_fills():
    bad = frame(start_new="2006-01-02"); before = bad.copy(); r = P.validate_frame_pit(bad, "t", table())
    assert not r["ok"] and r["violations"][0]["symbol"] == "NEW" and r["violations"][0]["kind"] == "pre_inception_values"
    pd.testing.assert_frame_equal(bad, before)                # untouched
    assert P.validate_frame_pit(frame(), "t", table())["ok"]
    late = frame(); late["OLD"] = late["OLD"].where(late.index > "2005-03-01")   # value after delisting also flagged
    late.loc["2010-01-04", "GONE"] = 1.0
    assert any(v["kind"] == "post_delisting_values" for v in P.validate_frame_pit(late, "t", table())["violations"])

def test_unknown_symbol_and_universe():
    with pytest.raises(P.PITViolation): P.availability_date("ZZZ", "t", table())
    with pytest.raises(P.PITViolation): P.universe_at("2010-01-01", "nope", table())

def test_survivorship_flag_etf24():
    f = P.survivorship_flag("etf24")
    assert f["survivorship_biased"] is True and f["n_members"] == 24 and f["n_delisted_members_in_table"] == 0 and "delisted" in f["reason"]
    assert P.survivorship_flag("t", table())["survivorship_biased"] is True

# ---- real table
def test_real_table_matches_etf24_and_later_date_rule():
    pit = P.load_pit(); m = pit["universes"]["etf24"]["members"]
    assert list(m) == ETF24 and UNIVERSES["etf24"] == ETF24
    for s, r in m.items():
        assert r["availability_date"] == max(r["official_inception"]["date"], r["first_valid_bar"]["date"])
        assert r["official_inception"]["source"] == "agent knowledge, verify" and r["official_inception"]["confidence"]
    assert m["HYG"]["availability_date"] == "2007-04-11"       # first Yahoo bar is later than official 2007-04-04
    assert P.universe_at("2005-06-01") == [s for s in ETF24 if s not in ("HYG", "SLV", "DBC", "USO")]
    assert P.universe_at("2026-01-02") == ETF24

def test_real_cache_passes_pit_check_if_present():
    from engine.data.ingest import DATASETS
    f = DATASETS / "dcfd243d77e6.parquet"
    if not f.exists(): pytest.skip("cache absent")
    df = pd.read_parquet(f); assert check_availability(df)["ok"]
    assert P.availability_by_year(range(2005, 2008), prices=df)[2005]["n_with_data"] == 20
