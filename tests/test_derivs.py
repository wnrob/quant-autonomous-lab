"""Options adapter tests. Rows are SYNTHETIC model inputs (labelled) except the two `real_` tests, which read the local
real data read-only and only check library agreement / loadability (descriptive, protected-era rules respected)."""
import io, math, pickle, pickletools
from pathlib import Path
import pytest
from engine.derivs import checks as C, safe_pickle as SP, realdata as RD
from adapters import derivs_adapter as D
from engine.errors import AdapterUnavailable

needs_env = pytest.mark.skipif(not D.available(), reason="envs/derivs not built")
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ref_bsm(flag, S, K, T, r, q, s):
    """Independent third implementation (stdlib) used as a reference."""
    d1 = (math.log(S / K) + (r - q + s * s / 2) * T) / (s * math.sqrt(T)); d2 = d1 - s * math.sqrt(T)
    if flag == "c": return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)


# ------------------------------------------------------------------ validation (core only, no env needed)
BAD = [dict(sigma=-0.2), dict(sigma=0.0), dict(T=0.0), dict(T=-0.1), dict(flag="C"), dict(flag="call"), dict(flag=1), dict(flag=True),
       dict(S=float("nan")), dict(S=-1.0), dict(K=0.0), dict(model="black"), dict(r=float("inf")), dict(sigma=True)]


@pytest.mark.parametrize("bad", BAD)
def test_invalid_inputs_raise_before_any_subprocess(bad, monkeypatch):
    monkeypatch.setattr(D, "run_in_env", lambda *a, **k: pytest.fail("worker must not be started for invalid input"))
    row = {**dict(flag="c", S=100.0, K=100.0, T=30 / 365, r=0.03, q=0.0, sigma=0.2), **bad}
    with pytest.raises(C.InvalidOptionInput):
        D.price_greeks([row])


@pytest.mark.parametrize("price,reason", [(0.5, "below_lower_bound"), (10.0, "below_lower_bound"), (0.0, "price_nonpositive"), (-1.0, "price_nonpositive"),
                                          (float("nan"), "price_not_finite"), (True, "price_not_finite"), (110.0, "at_or_above_upper_bound"), (150.0, "at_or_above_upper_bound"),
                                          (11.5, None)])
def test_iv_precheck_reasons_never_clamp(price, reason):
    # ITM call S=110, K=100, T=0.25, r=3%: lower bound = 110 - 100 e^{-rT} ~ 10.75, upper bound = S = 110
    p, why, _ = C.iv_precheck(price, "c", 110.0, 100.0, 0.25, 0.03, 0.0)
    assert why == reason
    assert (p is None) == (reason is not None)


def test_iv_precheck_at_lower_bound_is_not_identifiable():
    lo, _ = C.bounds("c", 110.0, 100.0, 0.25, 0.03, 0.0)
    assert C.iv_precheck(lo, "c", 110.0, 100.0, 0.25, 0.03, 0.0)[1] == "at_lower_bound"
    assert C.iv_precheck(lo + 0.5, "c", 110.0, 100.0, 0.25, 0.03, 0.0)[1] is None


def test_black76_sets_q_equal_r():
    assert C.validate("p", 100, 100, 0.5, 0.04, 0.02, 0.2, "black76")[5] == 0.04


# ------------------------------------------------------------------ cross-check through the env (fast, unmarked)
@needs_env
def test_crosscheck_grid_vollib_vs_quantlib(tmp_path):
    from engine.derivs import crosscheck as X
    rep = X.run(tmp_path)
    assert rep["n_rows"] > 10000
    for f, d in rep["max_abs_diff"].items(): assert d <= X.TOL[f], (f, d)
    assert rep["pass_parity"] and rep["pass_iv_roundtrip"] and rep["pass_bounds"] and rep["pass_all"]
    assert rep["iv_roundtrip"]["identifiable_rows"] > 9000 and sum(rep["iv_roundtrip"]["nan_on_identifiable"].values()) == 0


@needs_env
def test_against_independent_reference_and_textbook():
    rows = [dict(flag=f, S=42.0, K=40.0, T=183 / 365, r=0.10, q=0.0, sigma=0.20) for f in "cp"] + \
           [dict(flag=f, S=100.0, K=K, T=d / 365, r=0.02, q=0.01, sigma=s) for f in "cp" for K in (70.0, 100.0, 130.0) for d in (14, 400) for s in (0.15, 0.6)]
    df = D.price_greeks(rows)
    for row, (_, o) in zip(rows, df.iterrows()):
        ref = ref_bsm(row["flag"], row["S"], row["K"], row["T"], row["r"], row["q"], row["sigma"])
        assert abs(o.vollib_price - ref) < 1e-10 and abs(o.quantlib_price - ref) < 1e-10
    # Hull (Options, Futures & Other Derivatives) example: c ~ 4.76, p ~ 0.81 at T=0.5 (here T=183/365, so a 0.02 band)
    assert abs(df.vollib_price[0] - 4.76) < 0.02 and abs(df.vollib_price[1] - 0.81) < 0.02
    # finite-difference Greeks (vega, theta per year) on one row
    base = dict(flag="c", S=100.0, K=105.0, T=91 / 365, r=0.03, q=0.01, sigma=0.25)
    bump = D.price_greeks([base, {**base, "sigma": 0.25 + 1e-5}, {**base, "sigma": 0.25 - 1e-5}, {**base, "T": 90 / 365}])
    assert abs((bump.quantlib_price[1] - bump.quantlib_price[2]) / 2e-5 - bump.quantlib_vega[0]) < 1e-4
    assert abs((bump.quantlib_price[3] - bump.quantlib_price[0]) * 365 - bump.quantlib_theta[0]) < 0.05 * abs(bump.quantlib_theta[0])


@needs_env
def test_iv_nan_with_reason_outside_bounds_both_libs():
    S, K, T, r = 110.0, 100.0, 30 / 365, 0.03
    rows = [dict(flag="c", S=S, K=K, T=T, r=r, price=p) for p in (5.0, 0.0, 200.0, 12.0)] + [dict(flag="p", S=S, K=K, T=T, r=r, price=K * 1.01)]
    out = D.implied_vol(rows)
    assert out.iv_vollib[:3].isna().all() and out.iv_quantlib[:3].isna().all() and out.iv_vollib[4:].isna().all()
    assert list(out.reason_vollib[:3]) == ["below_lower_bound", "price_nonpositive", "at_or_above_upper_bound"]
    assert out.reason_quantlib[4] == "at_or_above_upper_bound"
    assert 0 < out.iv_vollib[3] < 10 and abs(out.iv_vollib[3] - out.iv_quantlib[3]) < 1e-8


@needs_env
def test_worker_revalidates_when_core_is_bypassed():
    from adapters.envrun import run_in_env
    with pytest.raises(AdapterUnavailable, match="InvalidOptionInput"):
        run_in_env("derivs", "derivs_worker.py", {"op": "price_greeks", "rows": [dict(flag="c", S=100, K=100, T=0.1, r=0.0, sigma=-0.2)]})
    with pytest.raises(AdapterUnavailable, match="InvalidOptionInput"):
        run_in_env("derivs", "derivs_worker.py", {"op": "price_greeks", "rows": [dict(flag="C", S=100, K=100, T=0.1, r=0.0, sigma=0.2)]})


@needs_env
def test_solver_failure_paths_return_nan_not_clamped():
    """Bypass the bounds precheck and hand both solvers impossible prices: each must return NaN + its own reason."""
    import subprocess, json as _j
    from adapters.envrun import ROOT
    code = ("import importlib.util,json,math,sys;"
            "s=importlib.util.spec_from_file_location('w',sys.argv[1]);w=importlib.util.module_from_spec(s);s.loader.exec_module(w);P=w._pv();out=[]\n"
            "for price in (0.5, 200.0):\n"
            "  a=w.pv_iv(P,price,'c',110.0,100.0,30/365,0.03,0.0,'bsm'); b=w.ql_iv(price,'c',110.0,100.0,30/365,0.03,0.0,'bsm')\n"
            "  out.append([None if math.isnan(a[0]) else a[0],a[1],None if math.isnan(b[0]) else b[0],b[1]])\n"
            "print(json.dumps(out))")
    r = subprocess.run([str(ROOT / "envs/derivs/bin/python"), "-I", "-W", "ignore", "-c", code, str(ROOT / "adapters/workers/derivs_worker.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    for v1, e1, v2, e2 in _j.loads(r.stdout.strip().splitlines()[-1]):
        assert v1 is None and e1.startswith("py_vollib:") and v2 is None and e2.startswith("quantlib:")


@needs_env
def test_quantlib_path_requires_integer_days():
    with pytest.raises(AdapterUnavailable, match="integer days"):
        D.price_greeks([dict(flag="c", S=100.0, K=100.0, T=0.5, r=0.0, sigma=0.2)])


@needs_env
def test_american_binomial_extra():
    out = D.american([dict(flag="p", S=100.0, K=100.0, T=365 / 365, r=0.05, q=0.0, sigma=0.2), dict(flag="c", S=100.0, K=100.0, T=365 / 365, r=0.05, q=0.0, sigma=0.2)])
    assert out.american_crr[0] > out.european_analytic[0] + 0.3                      # early-exercise premium on the put
    assert abs(out.american_crr[1] - out.european_crr[1]) < 1e-9                     # no early exercise for a call with q=0
    assert (out.european_crr - out.european_analytic).abs().max() < 0.01             # CRR(801) converges to BSM


# ------------------------------------------------------------------ safe pickle (core only; malicious payloads never executed)
class _Evil:
    def __init__(self, path): self.path = path
    def __reduce__(self): return (eval, (f"open({self.path!r}, 'w').write('pwned')",))


def test_malicious_pickle_refused_by_static_scan_and_never_executed(tmp_path):
    marker = tmp_path / "pwned"; data = pickle.dumps(_Evil(str(marker)), protocol=4)
    with pytest.raises(SP.UnsafePickle, match="builtins"):
        SP.safe_loads(data)
    assert not marker.exists()


def test_restricted_unpickler_refuses_even_without_static_scan(tmp_path):
    marker = tmp_path / "pwned"; data = pickle.dumps(_Evil(str(marker)), protocol=2)
    with pytest.raises(SP.UnsafePickle, match="find_class refused"):
        SP.RestrictedUnpickler(io.BytesIO(data)).load()
    assert not marker.exists()


def test_scan_resolves_memoized_stack_global_and_refuses_unresolvable():
    import collections, datetime as dt
    data = pickle.dumps([dt.date(2024, 1, 2), dt.date(2024, 1, 3), collections.OrderedDict(a=1)], protocol=4)
    found, problems = SP.scan_globals(data)
    assert ("datetime", "date") in found and ("collections", "OrderedDict") in found and not problems
    with pytest.raises(SP.UnsafePickle): SP.check(data)
    assert SP.safe_loads(pickle.dumps([dt.date(2024, 1, 2)], protocol=4)) == [dt.date(2024, 1, 2)]
    # STACK_GLOBAL whose operands are not literal strings (built via a REDUCE) must be refused statically
    crafted = b"\x80\x04" + b"\x8c\x08builtins\x94" + b"\x8c\x03str\x94" + b"\x93" + b"\x8c\x01x\x85R" + b"\x8c\x01y\x93."
    _, problems = SP.scan_globals(crafted)
    assert problems
    with pytest.raises(SP.UnsafePickle): SP.check(crafted)


def test_allowlist_is_exact_pairs_not_modules():
    assert ("builtins", "slice") in SP.ALLOWED
    for bad in [("builtins", "eval"), ("builtins", "getattr"), ("builtins", "__import__"), ("os", "system"), ("numpy", "load"), ("pandas", "read_pickle")]:
        assert bad not in SP.ALLOWED


# ------------------------------------------------------------------ protected-era guard
def test_protected_era_guard():
    RD.require_descriptive("descriptive", ["2026-10-01"])
    RD.require_descriptive("backtest", ["2015-06-01"])                       # dev-era dates are not this guard's concern
    for purpose in ("backtest", "selection", "signal", "Descriptive"):
        with pytest.raises(RD.ProtectedEraUse):
            RD.require_descriptive(purpose, ["2015-06-01", "2020-01-02"])


def test_studies_refuse_non_descriptive_before_computing(monkeypatch):
    monkeypatch.setattr(D, "_call", lambda *a, **k: pytest.fail("no computation may start for a non-descriptive purpose"))
    if not RD.QQQ_DAILY.exists() or not RD.SNAP_DIR.exists(): pytest.skip("real data not present")
    with pytest.raises(RD.ProtectedEraUse): RD.chains_study(Path("/nonexistent"), purpose="backtest")
    with pytest.raises(RD.ProtectedEraUse): RD.cboe_study(Path("/nonexistent"), purpose="backtest")


# ------------------------------------------------------------------ real data (read-only, descriptive; skip if absent)
@needs_env
def test_real_qqq_chains_pickle_loads_safely_and_libs_agree(tmp_path):
    if not RD.CHAINS_PKL.exists(): pytest.skip("qqq_chains.pkl not present")
    scan = D.pickle_scan(RD.CHAINS_PKL)
    assert scan["disallowed"] == [] and scan["problems"] == []
    df = D.qqq_chains_iv(RD.CHAINS_PKL, RD.QQQ_DAILY, tmp_path / "c.parquet", r=0.045, q=0.0)
    both = df.dropna(subset=["iv_vollib", "iv_quantlib"])
    assert len(both) > 1000 and (both.iv_vollib - both.iv_quantlib).abs().max() < 1e-9
    assert set(df.reason.dropna()) <= {"expiry_day", "below_lower_bound", "no_underlying_close", "at_lower_bound", "price_nonpositive", "at_or_above_upper_bound"}


@needs_env
def test_real_cboe_snapshot_libs_agree(tmp_path):
    if not RD.SNAP_DIR.exists(): pytest.skip("Cboe snapshots not present")
    _, pick = RD.select_snapshots()
    rows, fwd = D.cboe_snapshots_iv([pick.raw.iloc[-1]], tmp_path / "s.parquet", forward_method="atm_pair", r_fixed=0.045)
    both = rows.dropna(subset=["iv_vollib", "iv_quantlib"])
    assert len(both) > 100 and (both.iv_vollib - both.iv_quantlib).abs().max() < 1e-9
    assert ((rows.flag == "c") & (rows.K < rows.F)).sum() == 0 and ((rows.flag == "p") & (rows.K >= rows.F)).sum() == 0   # OTM only
