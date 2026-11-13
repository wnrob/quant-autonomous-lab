"""`python -m engine.forward.cli log|score|status|verify|approve|demo` (proposed as `qae forward ...`).

log/score/status/verify are safe to automate. `approve` is human-only: it has no --yes flag and reads a typed phrase from /dev/tty."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]

def _pick(store, spec_hash):
    from engine.forward.jobs import candidates
    cs = [c for c in candidates(store) if spec_hash in (None, c["spec_hash"])]
    if spec_hash and not cs: raise SystemExit(f"{spec_hash} is not a CANDIDATE in {store.path}")
    return cs

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("qae forward", description=__doc__); sp = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("log", "log today's signal for CANDIDATE strategies (before the outcome exists)"), ("score", "score logged signals against bars strictly after them")):
        p = sp.add_parser(name, help=hlp); p.add_argument("--spec-hash")
    sp.add_parser("status", help="forward status (from the approval chain) + log counts"); sp.add_parser("verify", help="verify both hash chains")
    a = sp.add_parser("approve", help="HUMAN ONLY: record an approval (interactive terminal required)")
    a.add_argument("kind", choices=["enter-shadow", "paper-approved", "retire"]); a.add_argument("--spec-hash", required=True); a.add_argument("--approver", required=True); a.add_argument("--reason", required=True)
    sp.add_parser("demo", help="mechanics demo in a scratch store (never artifacts/qae)")
    args = ap.parse_args(argv)
    from engine.memory.store import Store
    from engine.forward.approval import ApprovalLog
    from engine.forward.signals import SignalLog, forward_home
    st = Store(); home = forward_home(st); log = SignalLog(home); apr = ApprovalLog(home)
    if args.cmd == "demo":
        from engine.forward.demo import run_demo; print(json.dumps(run_demo(st), indent=1)); return 0
    if args.cmd == "log":
        from engine.forward.jobs import log_one
        cs = _pick(st, args.spec_hash)
        if not cs: print("no CANDIDATE strategies; nothing to log")
        for c in cs:
            try: r = log_one(st, c["spec"]); print(f"{c['name']}: logged as_of {r['body']['as_of']} weights {r['body']['weights']} seq {r['seq']} hash {r['hash'][:16]} bar_final={r['body']['as_of_bar_final']}")
            except Exception as e: print(f"{c['name']}: REFUSED {type(e).__name__}: {e}")
    elif args.cmd == "score":
        from engine.forward.jobs import score_one
        cs = _pick(st, args.spec_hash)
        if not cs: print("no CANDIDATE strategies; nothing to score")
        for c in cs:
            try: r = score_one(st, c["spec"]); print(json.dumps({k: v for k, v in r.items() if k not in ("daily", "fills", "evidence_rule")}, indent=1))
            except Exception as e: print(f"{c['name']}: not scored: {type(e).__name__}: {e}")
    elif args.cmd == "status":
        recs = log.records(); by = {}
        for r in recs: by.setdefault(r["body"]["spec_hash"], []).append(r["body"]["as_of"])
        print(json.dumps({"store": str(st.path), "forward_status": apr.all_status(), "signals": {k: {"n": len(v), "first_as_of": v[0], "last_as_of": v[-1]} for k, v in by.items()},
                          "research_store_candidates": [c["name"] for c in _pick(st, None)]}, indent=1))
    elif args.cmd == "verify":
        n, hd = log.chain.head(); log.records(); m, ha = apr.chain.head(); apr.all_status()
        print(json.dumps({"signals": {"n": n, "head": hd}, "approvals": {"n": m, "head": ha}, "ok": True}, indent=1))
    elif args.cmd == "approve":
        from engine.forward.scoring import backtest_expectation, load_forward_bars
        import pandas as pd
        kind = args.kind.upper().replace("-", "_"); px = exp = None
        if kind == "PAPER_APPROVED":
            from engine.data.ingest import DATASETS, YahooProvider, _key
            sigs = log.for_spec(args.spec_hash)
            if not sigs: raise SystemExit("no logged signals for this spec")
            syms = sorted(sigs[0]["weights"]); px = load_forward_bars(DATASETS / f"{_key(YahooProvider.name, syms, '2005-01-01', None)}.parquet", syms, pd.Timestamp(sigs[0]["as_of"]))
            exp = backtest_expectation(st, args.spec_hash)
        try: r = apr.approve(kind, st, args.spec_hash, args.approver, args.reason, outcome_px=px, expectation=exp)
        except PermissionError as e: print(f"REFUSED: {e}", file=sys.stderr); return 2
        print(f"recorded {kind} seq {r['seq']} hash {r['hash']}")
    return 0

if __name__ == "__main__": sys.exit(main())
