"""OBSERVE -> HYPOTHESIZE -> DESIGN -> EXPERIMENT -> EVALUATE -> CRITIQUE -> IMPROVE -> VALIDATE -> PROMOTE/REJECT -> REMEMBER.
This is heuristic search + statistics. It does NOT train or modify any LLM; no LLM is called."""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from engine.agents import experiment as X, roles
from engine.memory.store import Store, h
from engine.agents.llm import LLMUnavailable
from engine.orchestration.budget import BudgetExceeded, BudgetGuard
from engine.research.validation import Criteria
from engine.strategies.trend import baseline_spec, name_of
log = logging.getLogger(__name__)

def run_cycle(store: Store, px: pd.Series, data_sha: str, symbol: str, crit: Criteria = Criteria(), guard: BudgetGuard | None = None,
              max_experiments: int = 2, report_dir: Path | None = None, llm=None) -> dict:
    out = {"symbol": symbol, "data_sha": data_sha, "experiments": [], "stopped": None}
    # OBSERVE
    q = roles.data_agent(px, symbol); out["quality"] = q.summary()
    if not q.ok and (q.duplicates or q.non_monotonic or q.nonpositive): out["stopped"] = "data quality blocking"; return out
    # HYPOTHESIZE
    hid = roles.stat_agent(px, store, symbol)
    base = baseline_spec(symbol)
    queue: list = []
    try:
        # DESIGN+EXPERIMENT+EVALUATE baseline
        b = X.evaluate(base, px, crit, store, data_sha, hypothesis_id=hid, guard=guard, report_dir=report_dir); out["experiments"].append(_brief(b))
        n = 0 if b["duplicate"] is False else 0
        for _ in range(max_experiments):
            # CRITIQUE + IMPROVE: proposer reads baseline failure analysis & memory
            base_full = b if "metrics" in b and isinstance(b["metrics"], dict) else _load(store, b)
            source = "cro_rule_based"
            if llm is not None and not queue:
                try: queue = llm.propose(base, base_full, crit, data_sha)
                except LLMUnavailable as e: log.warning("LLM unavailable, falling back to rules: %s", e); out["llm_error"] = str(e); llm = None
            prop = None
            while queue and prop is None:
                cand = queue.pop(0)
                if not store.find_experiment(h(cand[0]), crit.digest(), data_sha): prop, source = cand, llm.source
            prop = prop or roles.propose(store, base, base_full, crit, data_sha)
            if prop is None: out["stopped"] = "no un-tested proposals left (dedup)"; break
            spec, why = prop
            ch_h = store.add_hypothesis(why, f"derived from experiment #{b['id']}", source=source, parent_experiment=b["id"])
            sid, _ = store.get_or_create_strategy(spec, name_of(spec), parent_id=b["strategy_id"])
            c = X.evaluate(spec, px, crit, store, data_sha, hypothesis_id=ch_h, guard=guard, report_dir=report_dir)   # VALIDATE (protected test inside, once)
            cmp = compare(base_full, c if isinstance(c["metrics"], dict) else _load(store, c), crit)
            c_brief = _brief(c); c_brief["vs_baseline"] = cmp; out["experiments"].append(c_brief)
            store.add_lesson(c["id"], spec["family"], "comparison", f"{name_of(spec)} vs baseline {name_of(base)}: {cmp['verdict']} ({cmp['detail']}). Baseline preserved.")
    except BudgetExceeded as e:
        out["stopped"] = f"budget: {e}"
    return out

def _load(store: Store, row: dict) -> dict:
    import json
    r = store.q("SELECT * FROM experiments WHERE id=?", row["id"])[0]; return {**row, "metrics": json.loads(r["metrics"]), "decision": r["decision"]}

def compare(base: dict, chal: dict, crit: Criteria) -> dict:
    """Improvement = challenger beats baseline validation Sharpe by the predefined margin AND passes its own full validation
    (incl. protected test, risk gate, DSR). Otherwise the baseline stands. Margin comes from frozen criteria."""
    bs, cs = base["metrics"]["validation"]["sharpe"], chal["metrics"]["validation"]["sharpe"]
    margin_ok = cs - bs >= crit.challenger_margin_sharpe; passed = chal["decision"] == "TEST_PASS"
    if margin_ok and passed: return {"verdict": "IMPROVEMENT", "detail": f"val Sharpe {cs:.2f} vs {bs:.2f}; passed all criteria"}
    why = [] 
    if not margin_ok: why.append(f"val Sharpe gain {cs-bs:+.2f} < margin {crit.challenger_margin_sharpe}")
    if not passed: why.append(f"challenger decision {chal['decision']}")
    return {"verdict": "NO_IMPROVEMENT", "detail": "; ".join(why)}

def _brief(r: dict) -> dict:
    m = r["metrics"] if isinstance(r.get("metrics"), dict) else {}
    return {"id": r["id"], "decision": r["decision"], "duplicate": r.get("duplicate", False), "reasons": r["reasons"] if isinstance(r["reasons"], list) else r["reasons"],
            "spec": m.get("spec"), "val_sharpe": (m.get("validation") or {}).get("sharpe"), "report": r.get("report_path")}
