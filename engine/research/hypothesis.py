"""H1 (pre-stated): after a positive trailing-12-month return, the next month's return is higher than after a negative one.

Method: month-end sampling, non-overlapping next-month returns, Welch t-test + percentile bootstrap of the mean difference.
Rule fixed in advance: report 'rejects null' only if |t| > 2 AND the 95% bootstrap CI excludes 0. No parameter search.
Scope caveat: single asset, in-sample, ~20y; a rejection is NOT evidence of a tradable edge after costs/regime change.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

def momentum_hypothesis(prices: pd.Series, lookback_months: int = 12, n_boot: int = 5000, seed: int = 0) -> dict:
    m = prices.dropna().resample("ME").last()
    trailing = m.pct_change(lookback_months)
    nxt = m.pct_change().shift(-1)
    df = pd.DataFrame({"trail": trailing, "next": nxt}).dropna()
    pos, neg = df.loc[df.trail > 0, "next"], df.loc[df.trail <= 0, "next"]
    if len(pos) < 12 or len(neg) < 12:
        raise ValueError(f"insufficient observations per group (pos={len(pos)}, neg={len(neg)})")
    t = stats.ttest_ind(pos, neg, equal_var=False)
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(pos, len(pos)).mean() - rng.choice(neg, len(neg)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rejects = abs(t.statistic) > 2 and not (lo <= 0 <= hi)
    return {"hypothesis": f"mean next-month return | trailing {lookback_months}m > 0  >  | trailing <= 0",
            "n_pos": len(pos), "n_neg": len(neg), "mean_next_after_pos": float(pos.mean()), "mean_next_after_neg": float(neg.mean()),
            "diff": float(pos.mean() - neg.mean()), "welch_t": float(t.statistic), "p_value_two_sided": float(t.pvalue),
            "bootstrap_ci95_diff": [float(lo), float(hi)],
            "verdict": "rejects null (in-sample, single asset)" if rejects else "fails to reject null",
            "caveats": ["in-sample, single hypothesis fixed a priori, but sample overlaps famous published results",
                        "monthly observations: low statistical power", "no transaction costs in this test",
                        "statistical significance != tradable edge"]}
