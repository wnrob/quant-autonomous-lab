"""Validation framework: frozen criteria, chronological splits w/ embargo, PSR/Deflated Sharpe, cost stress, bootstrap, risk gate."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np, pandas as pd
from scipy import stats
from engine.memory.store import h

@dataclass(frozen=True)
class Splits:
    """train: fit/explore. validation: selection. test: PROTECTED, looked at once per spec, counted in the store."""
    train_end: str = "2014-12-31"; val_end: str = "2019-12-31"; embargo_days: int = 5
    def slice(self, r: pd.Series) -> dict[str, pd.Series]:
        te, ve = pd.Timestamp(self.train_end), pd.Timestamp(self.val_end); e = pd.Timedelta(days=self.embargo_days)
        return {"train": r[:te], "validation": r[te + e: ve], "test": r[ve + e:]}

@dataclass(frozen=True)
class RiskLimits:
    """Mandatory. No agent may modify at runtime (frozen); the gate is applied by code, not by choice."""
    max_drawdown: float = -0.30; max_leverage: float = 1.0; max_annual_turnover: float = 12.0

@dataclass(frozen=True)
class Criteria:
    """Predefined promotion requirements. Changing any value changes criteria_hash, so old results are never silently re-judged."""
    min_obs: int = 750; min_dsr: float = 0.90; stress_cost_mult: float = 3.0
    max_test_sharpe_degradation: float = 0.5      # test Sharpe must be >= 50% of validation Sharpe
    challenger_margin_sharpe: float = 0.10; max_test_looks: int = 6
    risk: RiskLimits = RiskLimits(); splits: Splits = Splits()
    def digest(self) -> str: return h(asdict(self))

def sharpe(r: pd.Series, periods: int = 252) -> float:
    r = r.dropna(); sd = r.std(ddof=1); return float(r.mean() / sd * np.sqrt(periods)) if len(r) > 2 and sd > 0 else float("nan")
def sortino(r: pd.Series, periods: int = 252) -> float:
    d = r[r < 0]; dd = np.sqrt((d ** 2).sum() / len(r)); return float(r.mean() / dd * np.sqrt(periods)) if dd > 0 else float("nan")
def calmar(r: pd.Series, periods: int = 252) -> float:
    eq = (1 + r).cumprod(); mdd = float((eq / eq.cummax() - 1).min()); ann = float(eq.iloc[-1] ** (periods / len(r)) - 1)
    return ann / abs(mdd) if mdd < 0 else float("nan")

def psr(r: pd.Series, sr_benchmark_daily: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012), daily units."""
    r = r.dropna(); T = len(r); sd = r.std(ddof=1)
    if T < 30 or sd == 0: return float("nan")
    sr = r.mean() / sd; g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    den = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr_benchmark_daily) * np.sqrt(T - 1) / den))

def expected_max_sharpe(trial_sharpes_daily: list[float]) -> float:
    """Expected maximum DAILY Sharpe among N trials of zero-skill strategies (Bailey & Lopez de Prado). 0 if N < 2."""
    n = len(trial_sharpes_daily)
    if n < 2: return 0.0
    v = float(np.var(trial_sharpes_daily, ddof=1)); g = 0.5772156649
    return float(np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e))))

def deflated_sharpe(r: pd.Series, trial_sharpes_daily: list[float]) -> float:
    """DSR: PSR against the expected maximum Sharpe of N trials (multiple-testing correction). With N=1 it equals PSR(0)."""
    n = len(trial_sharpes_daily)
    if n <= 1: return psr(r, 0.0)
    v = float(np.var(trial_sharpes_daily, ddof=1)); g = 0.5772156649
    emax = np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e)))
    return psr(r, float(emax))

def bootstrap_sharpe_ci(r: pd.Series, n: int = 1000, block: int = 21, seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap 95% CI for annualised Sharpe."""
    x = r.dropna().to_numpy(); rng = np.random.default_rng(seed); T = len(x); k = int(np.ceil(T / block)); out = []
    for _ in range(n):
        st = rng.integers(0, T - block + 1, k); s = np.concatenate([x[i:i + block] for i in st])[:T]
        out.append(s.mean() / s.std(ddof=1) * np.sqrt(252))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def risk_gate(r: pd.Series, weights: "pd.Series | pd.DataFrame", limits: RiskLimits) -> list[str]:
    """Returns list of violated limits (empty = pass). Mandatory; called by the validator, not skippable via config.
    weights: Series (single asset) or DataFrame (date x asset). Leverage = max gross exposure; turnover = total |dw| per year."""
    w = weights.to_frame() if isinstance(weights, pd.Series) else weights
    eq = (1 + r).cumprod(); mdd = float((eq / eq.cummax() - 1).min())
    turn = float(w.diff().abs().sum().sum() / (len(w) / 252)); gross = float(w.abs().sum(axis=1).max())
    v = []
    if mdd < limits.max_drawdown: v.append(f"max_drawdown {mdd:.1%} < limit {limits.max_drawdown:.0%}")
    if gross > limits.max_leverage + 1e-9: v.append(f"leverage {gross:.2f} > {limits.max_leverage}")
    if turn > limits.max_annual_turnover: v.append(f"turnover {turn:.1f}x > {limits.max_annual_turnover}x")
    return v

def window_metrics(r: pd.Series) -> dict:
    return {"n_obs": int(len(r)), "net_return_total": float((1 + r).prod() - 1), "sharpe": sharpe(r), "sortino": sortino(r), "calmar": calmar(r),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(252)), "max_drawdown": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()),
            "cvar_95": float(-r[r <= np.quantile(r, 0.05)].mean())}
