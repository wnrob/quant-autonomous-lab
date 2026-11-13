"""Shallow-clone all repos safely. No hooks, no LFS smudge, no submodules, no prompts.
Writes artifacts/acquisition_report.json. Idempotent: reuses existing clones."""
from __future__ import annotations
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from repos import REPOS

ROOT = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
G = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]

def run(args, cwd=None, timeout=900):
    return subprocess.run(args, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)

def acquire(item):
    rid, group, url, fallbacks = item
    dest = ROOT / "external" / group / rid
    rec = {"id": rid, "group": group, "requested_url": url, "path": str(dest.relative_to(ROOT)), "attempts": []}
    if (dest / ".git").exists():
        rec["status"] = "reused"; rec["used_url"] = run(G+["remote","get-url","origin"], cwd=dest).stdout.strip()
    else:
        rec["status"] = "failed"
        for u in [url] + fallbacks:
            t = time.time()
            try:
                r = run(G + ["clone", "--depth", "1", "--no-tags", "--single-branch", u, str(dest)])
                ok, err = r.returncode == 0, r.stderr.strip()[-300:]
            except subprocess.TimeoutExpired:
                ok, err = False, "timeout"
            rec["attempts"].append({"url": u, "ok": ok, "error": None if ok else err, "secs": round(time.time()-t,1)})
            if ok:
                rec["status"] = "cloned" if u == url else "cloned_via_fallback"; rec["used_url"] = u; break
    if rec["status"] != "failed":
        rec["head_sha"] = run(G+["rev-parse","HEAD"], cwd=dest).stdout.strip()
        rec["branch"] = run(G+["rev-parse","--abbrev-ref","HEAD"], cwd=dest).stdout.strip()
        du = subprocess.run(["du","-sk",str(dest)],capture_output=True,text=True).stdout.split()
        rec["size_mb"] = round(int(du[0])/1024,1) if du else None
    return rec

if __name__ == "__main__":
    out = ROOT / "artifacts"; out.mkdir(exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(acquire, REPOS))
    (out/"acquisition_report.json").write_text(json.dumps(recs, indent=2))
    for r in recs: print(f'{r["status"]:20} {r["id"]:30} {r.get("head_sha","")[:10]} {r.get("size_mb","")}')
    failed = [r["id"] for r in recs if r["status"]=="failed"]
    (out/"retry_report.md").write_text("# Retry report\n\n" + ("All repositories acquired.\n" if not failed else
        "Failed:\n" + "\n".join(f"- {r['id']}: {r['attempts']}" for r in recs if r['status']=='failed') + "\n\nRetry: `python3 -I scripts/acquire.py` (idempotent)\n"))
    print("FAILED:", failed)
