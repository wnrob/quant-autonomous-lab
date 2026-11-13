"""Input validation + no-arbitrage bounds for European options. STDLIB ONLY (math).

Loaded by BOTH the core adapter (normal import) and the envs/derivs worker (by file path), so the same rules
guard both sides of the process boundary. Nothing here clamps: invalid input raises, out-of-bounds prices give
NaN plus a reason string.

Conventions (one set, used everywhere in engine/derivs + the worker):
  T     year fraction, ACT/365F (calendar days / 365). Must be > 0.
  r, q  continuously compounded annual rates (r may be negative; q may be negative = borrow cost).
  sigma annualised Black vol as a decimal (0.20 = 20%). Must be > 0 for pricing.
  model 'bsm'  : underlying = spot S with continuous dividend yield q.
        'black76': underlying = forward F (pass it as S); q is ignored, discounting at r.
"""
from __future__ import annotations
import math

FLAGS = ("c", "p")
MODELS = ("bsm", "black76")


class InvalidOptionInput(ValueError):
    """Raised for inputs that are not a well-posed European option (never silently fixed)."""


def _finite(name: str, x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise InvalidOptionInput(f"{name} must be a real number, got {type(x).__name__}")
    x = float(x)
    if not math.isfinite(x):
        raise InvalidOptionInput(f"{name} must be finite, got {x}")
    return x


def validate(flag, S, K, T, r, q=0.0, sigma=None, model: str = "bsm") -> tuple:
    """Strict validation. flag must be exactly 'c' or 'p' (no 'C', 'call', 1, True). Returns floats."""
    if not isinstance(flag, str) or flag not in FLAGS:
        raise InvalidOptionInput(f"flag must be exactly 'c' or 'p', got {flag!r}")
    if model not in MODELS:
        raise InvalidOptionInput(f"model must be one of {MODELS}, got {model!r}")
    S, K, T, r, q = (_finite(n, v) for n, v in (("S", S), ("K", K), ("T", T), ("r", r), ("q", q)))
    if S <= 0: raise InvalidOptionInput(f"S (spot/forward) must be > 0, got {S}")
    if K <= 0: raise InvalidOptionInput(f"K must be > 0, got {K}")
    if T <= 0: raise InvalidOptionInput(f"T must be > 0 (expired/negative time is not priced), got {T}")
    if abs(r) > 1 or abs(q) > 1: raise InvalidOptionInput(f"|r|,|q| must be <= 1 (100%/yr), got r={r} q={q}")
    if sigma is not None:
        sigma = _finite("sigma", sigma)
        if sigma <= 0: raise InvalidOptionInput(f"sigma must be > 0, got {sigma}")
        if sigma > 10: raise InvalidOptionInput(f"sigma must be <= 10 (1000%), got {sigma}")
    if model == "black76": q = r      # Black-76 == BSM on the forward with q = r (S e^{-qT} = F e^{-rT})
    return flag, S, K, T, r, q, sigma


def bounds(flag: str, S: float, K: float, T: float, r: float, q: float) -> tuple[float, float]:
    """Model-free European no-arbitrage bounds (lower, upper). Inputs already validated (q==r for black76)."""
    fs, fk = S * math.exp(-q * T), K * math.exp(-r * T)
    return (max(fs - fk, 0.0), fs) if flag == "c" else (max(fk - fs, 0.0), fk)


def iv_precheck(price, flag, S, K, T, r, q=0.0, model: str = "bsm", rel_tol: float = 1e-12) -> tuple[float | None, str | None, tuple]:
    """Validate inputs and price for an implied-vol solve.
    Returns (price, None, validated_args) if solvable, else (None, reason, validated_args).
    Reasons: 'price_not_finite', 'price_nonpositive', 'below_lower_bound', 'at_lower_bound' (vol not identifiable: 0 or
    numerically indistinguishable), 'at_or_above_upper_bound'. Never clamps."""
    flag, S, K, T, r, q, _ = validate(flag, S, K, T, r, q, None, model)
    args = (flag, S, K, T, r, q)
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(float(price)):
        return None, "price_not_finite", args
    price = float(price)
    if price <= 0: return None, "price_nonpositive", args
    lo, hi = bounds(flag, S, K, T, r, q)
    eps = rel_tol * max(S, K)
    if price < lo - eps: return None, "below_lower_bound", args
    if price <= lo + eps: return None, "at_lower_bound", args
    if price >= hi - eps: return None, "at_or_above_upper_bound", args
    return price, None, args
