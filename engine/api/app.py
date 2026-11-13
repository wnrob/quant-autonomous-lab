"""FastAPI backend. Every number served comes from the SQLite store; nothing is computed for display or invented."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from engine.agents.roles import ROSTER
from engine.memory.store import Store
from engine.orchestration.budget import BudgetGuard
from engine.orchestration.jobs import HANDLERS
from engine.orchestration.scheduler import Scheduler
from engine.api import views
STATIC = Path(__file__).parent / "static"
LAB_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

def create_app(store: Store | None = None, artifacts_root: Path | None = None) -> FastAPI:
    s = store or Store(); app = FastAPI(title="QAE", version="0.1")
    sched = Scheduler(s, HANDLERS)
    art = Path(artifacts_root) if artifacts_root else LAB_ARTIFACTS
    def exp_rows():
        out = []
        for r in s.q("SELECT e.*, s.name, s.family, s.status FROM experiments e JOIN strategies s ON s.id=e.strategy_id ORDER BY e.id"):
            r["metrics"] = json.loads(r["metrics"]); r["reasons"] = json.loads(r["reasons"]); out.append(r)
        return out
    @app.get("/api/health")
    def health(): return {"ok": True, "db": str(s.path), "paper_only": True, "live_execution": "disabled"}
    @app.get("/api/overview")
    def overview():
        return {"strategies_by_status": s.q("SELECT status, COUNT(*) n FROM strategies GROUP BY status"), "experiments": s.q("SELECT COUNT(*) n FROM experiments")[0]["n"],
                "test_looks_used": s.test_looks(), "budget": BudgetGuard(s).status(), "agents": [a.__dict__ for a in ROSTER]}
    @app.get("/api/experiments")
    def experiments(): return exp_rows()
    @app.get("/api/experiments/{eid}")
    def experiment(eid: int):
        r = [x for x in exp_rows() if x["id"] == eid]
        if not r: raise HTTPException(404)
        return r[0]
    @app.get("/api/strategies")
    def strategies():
        rows = s.q("SELECT * FROM strategies ORDER BY id")
        for r in rows: r["spec"] = json.loads(r["spec"]); r["history"] = s.q("SELECT old,new,reason,ts FROM status_history WHERE strategy_id=? ORDER BY id", r["id"])
        return rows
    @app.get("/api/hypotheses")
    def hypotheses(): return s.q("SELECT * FROM hypotheses ORDER BY id")
    @app.get("/api/proposal_yield")
    def proposal_yield():
        """Per proposal source: how many experiments, how many passed validation / test. Measures whether the LLM adds value over rules."""
        return s.q("""SELECT h.source, COUNT(e.id) experiments, SUM(e.decision IN ('VALIDATION_PASS','TEST_PASS','TEST_FAIL','BLOCKED_TEST_CONSUMED','BLOCKED_TEST_BUDGET')) passed_validation,
                      SUM(e.decision='TEST_PASS') passed_test FROM hypotheses h JOIN experiments e ON e.hypothesis_id=h.id GROUP BY h.source""")
    @app.get("/api/lessons")
    def lessons(): return s.q("SELECT * FROM lessons ORDER BY id")
    @app.get("/api/search")
    def search(q: str): return s.search(q)
    @app.get("/api/jobs")
    def jobs(): return {"jobs": s.q("SELECT * FROM jobs"), "runs": s.q("SELECT * FROM job_runs ORDER BY id DESC LIMIT 30")}
    @app.post("/api/jobs/{name}/{action}")
    def job_action(name: str, action: str):
        if not s.q("SELECT 1 FROM jobs WHERE name=?", name): raise HTTPException(404)
        if action == "pause": sched.set_state(name, "paused")
        elif action == "resume": sched.set_state(name, "enabled")
        elif action == "cancel": sched.set_state(name, "cancelled")
        elif action == "retry": sched.retry(name)
        else: raise HTTPException(400, "pause|resume|cancel|retry")
        return {"ok": True}
    @app.get("/api/reports/{eid}")
    def report(eid: int):
        r = s.q("SELECT report_path FROM experiments WHERE id=?", eid)
        if not r or not r[0]["report_path"] or not Path(r[0]["report_path"]).exists(): raise HTTPException(404)
        return FileResponse(r[0]["report_path"])
    # --- read-only dashboard views (GET only; stored state; empty state = {"status": "no_data", ...})
    def _run_or_404(fn, *a):
        try: return fn(*a)
        except views.UnknownRun: raise HTTPException(404, "unknown run: not a run directory under the artifact root")
    @app.get("/api/risk")
    def risk(): return views.risk_view(s)
    @app.get("/api/models")
    def models(run: str | None = None): return _run_or_404(views.models_view, s, art, run)
    @app.get("/api/portfolio")
    def portfolio(run: str | None = None): return _run_or_404(views.portfolio_view, art, run)
    @app.get("/api/research")
    def research(): return views.research_view(s)
    @app.get("/")
    def index(): return FileResponse(STATIC / "index.html")
    return app
