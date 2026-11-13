"""CLI: python -m engine.coding propose|status|approve|apply|restore|verify   (proposed to be mounted as `qae code ...`)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from engine.coding.workflow import CodingWorkflow
from engine.errors import QALError

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qae code"); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("propose", help="test a unified diff in an isolated sandboxed workspace; never applies it to the lab")
    p.add_argument("--patch", help="unified diff file ('-' = stdin)"); p.add_argument("--summary", default=""); p.add_argument("--declare-golden", nargs="*", default=[])
    p.add_argument("--llm", action="store_true", help="ask the local loopback LLM to write the patch"); p.add_argument("--path"); p.add_argument("--instruction")
    s = sp.add_parser("status"); s.add_argument("run_id", nargs="?")
    a = sp.add_parser("approve", help="HUMAN ONLY, interactive terminal: approve one exact patch sha256")
    a.add_argument("patch_sha256"); a.add_argument("--approver", required=True); a.add_argument("--reason", required=True)
    for name in ("apply", "restore"): sp.add_parser(name).add_argument("run_id")
    sp.add_parser("verify", help="verify the hash-chained run index and report hashes")
    args = ap.parse_args(argv); wf = CodingWorkflow()
    try:
        if args.cmd == "propose":
            intent = {"summary": args.summary, "golden_changes": args.declare_golden}; source = "human"
            if args.llm:
                from engine.agents.llm import OllamaClient
                from engine.coding.llm_patch import PatchProposer
                if not (args.path and args.instruction): ap.error("--llm needs --path and --instruction")
                text, intent, _ = PatchProposer(OllamaClient(), wf.lab, wf.out / ".ollama.lock").generate(args.path, args.instruction); source = "llm"
            else:
                if not args.patch: ap.error("--patch is required")
                text = sys.stdin.read() if args.patch == "-" else Path(args.patch).read_text()
            rep = wf.propose(text, intent, source=source)
            print(json.dumps({k: rep.get(k) for k in ("run_id", "status", "reasons", "patch_sha256", "report_paths")}, indent=1)); return 0 if rep["status"] == "READY_FOR_HUMAN_REVIEW" else 1
        if args.cmd == "status":
            out = wf.status(args.run_id)
            print(json.dumps(out if args.run_id else [{k: e[k] for k in ("seq", "run_id", "status", "report")} for e in out], indent=1)); return 0
        if args.cmd == "approve":                                      # calls the gate directly: approve() refuses any call that passes through workflow code
            from engine.coding import gate
            from engine.coding.patch import parse_patch, sha256_text
            pf = wf.out / "patches" / f"{args.patch_sha256}.diff"
            if not pf.is_file() or sha256_text(pf.read_text()) != args.patch_sha256: print("REFUSED: no stored patch with that sha256 (run `propose` first)", file=sys.stderr); return 2
            files = ", ".join(parse_patch(pf.read_text()).paths)
            print(json.dumps(gate.approve(wf.approvals_dir, args.patch_sha256, args.approver, args.reason, summary="Files: " + files), indent=1)); return 0
        if args.cmd in ("apply", "restore"):
            rep = getattr(wf, args.cmd)(args.run_id); print(json.dumps({k: rep.get(k) for k in ("run_id", "status", "reasons", "rollback")}, indent=1)); return 0 if rep["status"] in ("APPLIED", "RESTORED") else 1
        if args.cmd == "verify":
            v = wf.verify_index(); print(json.dumps(v, indent=1)); return 0 if v["ok"] else 1
    except QALError as e:
        print(f"REFUSED: {e}", file=sys.stderr); return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
