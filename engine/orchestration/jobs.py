"""Concrete job handlers wired to the engine (data refresh, daily review, nightly experiments, weekly report)."""
from __future__ import annotations
import json, time
from pathlib import Path
from engine.data.ingest import YahooProvider, load_prices
from engine.memory.store import HOME, Store
from engine.orchestration.budget import BudgetGuard
from engine.orchestration.cycle import run_cycle
SYMBOL, START = "SPY", "2005-01-01"

def _prices(refresh=False, offline=False):
    df, meta = load_prices(YahooProvider(), [SYMBOL], START, refresh=refresh, offline=offline); return df[SYMBOL].dropna(), meta

def data_refresh(s: Store) -> str:
    px, m = _prices(refresh=True); return f"{m.rows} rows to {m.last_date} sha {m.sha256[:12]}"
def nightly_experiments(s: Store) -> str:
    px, m = _prices(offline=True); r = run_cycle(s, px, m.sha256, SYMBOL, guard=BudgetGuard(s), max_experiments=3, report_dir=HOME / "reports")
    return json.dumps({"new": [(e["id"], e["decision"]) for e in r["experiments"] if not e["duplicate"]], "stopped": r["stopped"]})
def daily_review(s: Store) -> str:
    c = s.q("SELECT status, COUNT(*) n FROM strategies GROUP BY status"); return json.dumps({r["status"]: r["n"] for r in c})
def weekly_report(s: Store) -> str:
    L = ["# Weekly research report", f"generated {time.strftime('%Y-%m-%d %H:%M')}", ""]
    for e in s.q("SELECT e.id, s.name, e.decision, e.reasons FROM experiments e JOIN strategies s ON s.id=e.strategy_id ORDER BY e.id"): L.append(f"- #{e['id']} {e['name']}: **{e['decision']}** - {'; '.join(json.loads(e['reasons']))}")
    L += ["", "## Lessons"] + [f"- ({l['kind']}) {l['text']}" for l in s.q("SELECT * FROM lessons ORDER BY id")]
    p = HOME / "weekly_report.md"; p.write_text("\n".join(L)); return str(p)
def alpha_search(s: Store) -> str:
    from engine.agents.alpha import run_alpha_search; from engine.data.universe import ETF24
    px, m = load_prices(YahooProvider(), ETF24, START, offline=True); r = run_alpha_search(s, px, m.sha256, guard=BudgetGuard(s), max_experiments=6)
    return json.dumps({"new": [(e["name"], e["decision"]) for e in r["ran"] if not e["duplicate"]], "stopped": r["stopped"]})
from engine.forward.jobs import forward_log, forward_score
HANDLERS = {"forward_log": forward_log, "forward_score": forward_score, "alpha_search": alpha_search, "data_refresh": data_refresh, "daily_review": daily_review, "nightly_experiments": nightly_experiments, "weekly_report": weekly_report}
