"""POOLED multiple-testing diagnostic (report-only; never changes a recorded decision).
Each family deflates only against its own trials, but every family is judged on the SAME validation window, so the honest number of looks is the
pool. This computes the noise ceiling (expected best annualised Sharpe of N zero-skill trials) over ALL stored experiments plus declared external trials
(e.g. ML ledger, portfolio, A/B harness runs that live in scratch stores) and flags experiments whose validation Sharpe does not clear it.
CRUDE by design: mixes total-return Sharpe (trend) and active-return Sharpe (xs_factor); treat as an upper bound on how much pooled deflation could matter."""
from __future__ import annotations
import json
import numpy as np
from engine.memory.store import Store
from engine.research.validation import expected_max_sharpe

def stored_trials(store: Store) -> list[dict]:
    out = []
    for r in store.q("SELECT e.id, s.name, s.family, e.decision, e.metrics FROM experiments e JOIN strategies s ON s.id=e.strategy_id"):
        m = json.loads(r["metrics"]); v = m.get("validation_active") or m.get("validation")
        if v and v.get("sharpe") == v.get("sharpe"): out.append({"id": r["id"], "name": r["name"], "family": r["family"], "decision": r["decision"], "sharpe_ann": float(v["sharpe"])})
    return out

def pooled_report(store: Store, external_trial_sharpes_ann: list[float] | None = None) -> dict:
    rows = stored_trials(store); ext = list(external_trial_sharpes_ann or [])
    daily = [x["sharpe_ann"] / np.sqrt(252) for x in rows] + [x / np.sqrt(252) for x in ext]
    ceiling = expected_max_sharpe(daily) * np.sqrt(252)
    for x in rows: x["clears_pooled_ceiling"] = x["sharpe_ann"] > ceiling
    return {"n_stored": len(rows), "n_external": len(ext), "n_pooled": len(daily), "noise_ceiling_sharpe_ann": float(ceiling),
            "experiments": rows, "passing_decisions_that_do_not_clear": [x for x in rows if x["decision"] in ("TEST_PASS", "VALIDATION_PASS") and not x["clears_pooled_ceiling"]]}
