"""`qae` CLI: cycle, status, serve, scheduler control, search."""
from __future__ import annotations
import argparse, json, sys
from engine.logging_setup import setup

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("qae"); ap.add_argument("--log", default="INFO"); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cycle", help="run improvement cycle: baseline + proposed follow-ups"); c.add_argument("--max-experiments", type=int, default=2); c.add_argument("--offline", action="store_true"); c.add_argument("--llm", action="store_true", help="use local open-source LLM (Ollama, loopback) as an additional proposal source"); c.add_argument("--llm-model", default=None)
    al = sp.add_parser("alpha", help="cross-sectional ETF factor search (Alpha Discovery agent)"); al.add_argument("--max-experiments", type=int, default=6); al.add_argument("--offline", action="store_true")
    pf = sp.add_parser("portfolio", help="allocator comparison on research sleeves (dev window only)"); pf.add_argument("rest", nargs=argparse.REMAINDER)
    ml = sp.add_parser("ml", help="ML model ladder on ETF universe (dev window only)"); ml.add_argument("rest", nargs=argparse.REMAINDER)
    po = sp.add_parser("pooled", help="pooled multiple-testing noise ceiling across all stored experiments (report-only)"); po.add_argument("--external-trials", type=int, default=0, help="count of extra trials run in scratch stores (ML ledger, portfolio, A/B); modelled with the stored Sharpe variance")
    sp.add_parser("status"); s = sp.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    t = sp.add_parser("scheduler"); t.add_argument("action", choices=["tick", "loop", "pause", "resume", "cancel", "retry", "list"]); t.add_argument("job", nargs="?")
    q = sp.add_parser("search"); q.add_argument("query")
    fw = sp.add_parser("forward", help="forward SHADOW evaluation: log|score|status|verify|approve|demo (approve is human-only, interactive TTY)"); fw.add_argument("rest", nargs=argparse.REMAINDER)
    cd = sp.add_parser("code", help="coding-agent workflow: propose|status|approve|apply|restore|verify (approve/apply are human-only)"); cd.add_argument("rest", nargs=argparse.REMAINDER)
    sp.add_parser("futures-audit", help="read-only audit of the user's futures data + data-gap report (descriptive only, no backtest)")
    fm_ = sp.add_parser("families", help="qlib model / arbitragelab pairs families through the validation pipeline (dev window only, SCRATCH store only)"); fm_.add_argument("which", choices=["qlib", "pairs"]); fm_.add_argument("rest", nargs=argparse.REMAINDER)
    sp.add_parser("eventsim", help="vector vs nautilus event-driven trend baseline comparison (dev window only)")
    a = ap.parse_args(argv); setup(a.log)
    if a.cmd == "forward":
        from engine.forward.cli import main as fm; return fm(a.rest)
    if a.cmd == "code":
        from engine.coding.__main__ import main as cm; return cm(a.rest)
    if a.cmd == "futures-audit":
        from engine.futures import gap_report, real_data; real_data.main([]); gap_report.main(); return 0
    if a.cmd == "families":
        if a.which == "qlib": from engine.families.qlib_model import main as fam
        else: from engine.families.statarb_pairs import main as fam
        print(fam(a.rest)); return 0
    if a.cmd == "eventsim":
        from engine.eventsim.compare import main as em; return em([])
    from engine.memory.store import HOME, Store
    st = Store()
    if a.cmd == "cycle":
        from engine.orchestration.budget import BudgetGuard; from engine.orchestration.cycle import run_cycle; from engine.orchestration.jobs import _prices, SYMBOL
        px, m = _prices(offline=a.offline); g = BudgetGuard(st); llm = None
        if a.llm:
            from engine.agents.llm import DEFAULT_MODEL, LLMProposer, OllamaClient; llm = LLMProposer(OllamaClient(a.llm_model or DEFAULT_MODEL), st, g)
        print(json.dumps(run_cycle(st, px, m.sha256, SYMBOL, guard=g, max_experiments=a.max_experiments, report_dir=HOME / "reports", llm=llm), indent=2, default=str))
    elif a.cmd == "alpha":
        from engine.data.ingest import YahooProvider, load_prices; from engine.data.quality import check_series; from engine.data.universe import ETF24
        from engine.agents.alpha import run_alpha_search; from engine.orchestration.budget import BudgetGuard
        px, m = load_prices(YahooProvider(), ETF24, "2005-01-01", offline=a.offline)
        bad = {c: q.summary() for c in px if not (q := check_series(px[c], c)).ok}
        print(f"universe {len(px.columns)} assets, {m.first_date}..{m.last_date}; quality issues: {{ {', '.join(f'{k}: gaps={len(v['gaps'])}' for k, v in bad.items())} }}")
        print(json.dumps(run_alpha_search(st, px, m.sha256, guard=BudgetGuard(st), max_experiments=a.max_experiments), indent=1, default=str))
    elif a.cmd == "portfolio":
        from engine.agents.portfolio import main as pm; pm(a.rest)
    elif a.cmd == "ml":
        from engine.models.ml import main as mm; print(mm(a.rest))
    elif a.cmd == "pooled":
        import numpy as np; from engine.research.pooled import pooled_report, stored_trials
        base = [x["sharpe_ann"] for x in stored_trials(st)]; ext = list(np.random.default_rng(0).normal(np.mean(base) if base else 0, np.std(base, ddof=1) if len(base) > 1 else 0.3, a.external_trials)) if a.external_trials else []
        r = pooled_report(st, ext); print(f"pooled trials {r['n_pooled']} (stored {r['n_stored']} + external {r['n_external']}); noise ceiling = best expected Sharpe of that many ZERO-skill trials: {r['noise_ceiling_sharpe_ann']:.2f} (annualised)")
        for x in sorted(r["experiments"], key=lambda x: -x["sharpe_ann"])[:8]: print(f"  {x['name']:28} {x['decision']:18} val Sharpe {x['sharpe_ann']:+.2f}  clears ceiling: {x['clears_pooled_ceiling']}")
        print("passing decisions that do NOT clear the pooled ceiling:", [x["name"] for x in r["passing_decisions_that_do_not_clear"]] or "none")
    elif a.cmd == "status":
        for r in st.q("SELECT e.id,s.name,e.decision,e.reasons FROM experiments e JOIN strategies s ON s.id=e.strategy_id"): print(r["id"], r["name"], r["decision"], r["reasons"])
        print(st.q("SELECT status, COUNT(*) n FROM strategies GROUP BY status"))
    elif a.cmd == "serve":
        import uvicorn; from engine.api.app import create_app; uvicorn.run(create_app(st), host="127.0.0.1", port=a.port)
    elif a.cmd == "search": print(json.dumps(st.search(a.query), indent=1))
    else:
        from engine.orchestration.jobs import HANDLERS; from engine.orchestration.scheduler import Scheduler
        sc = Scheduler(st, HANDLERS)
        if a.action == "tick": print(sc.tick())
        elif a.action == "loop": sc.loop()
        elif a.action == "list": print(st.q("SELECT * FROM jobs"))
        elif a.action == "retry": sc.retry(a.job)
        else: sc.set_state(a.job, {"pause": "paused", "resume": "enabled", "cancel": "cancelled"}[a.action])
    return 0
if __name__ == "__main__": sys.exit(main())
