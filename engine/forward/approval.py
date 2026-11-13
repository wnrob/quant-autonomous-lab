"""Human-approval gate for forward status: CANDIDATE -> SHADOW -> PAPER_APPROVED (-> RETIRED).

`Store.set_status` keeps raising PermissionError for SHADOW/PAPER_APPROVED; this module does not touch it. Forward status
lives here: it is the fold of an append-only hash-chained approval log (approvals.jsonl), mirrored in forward.db for
queries. The mirror is never trusted: status() re-derives from the verified chain and raises if the table disagrees.

An approval record can only be created by approve(), which
  A1 refuses when any automation marker is present (job/agent env vars, or a caller frame from the scheduler, jobs, agents,
     API, coding-agent or model packages) - there is no flag, config key or env var that turns the gate OFF;
  A2 requires stdin+stdout to be a terminal AND a controlling terminal (/dev/tty), prints what is being approved and a
     one-time random code, and reads the exact phrase `APPROVE <KIND> <spec_hash> <code>` typed on /dev/tty;
  A3 checks the transition and, for PAPER_APPROVED, recomputes the forward report itself and requires the frozen
     ForwardEvidence minimum to be met BEFORE the human is even asked.
Honest limit: a process running as the same OS user can allocate a pseudo-terminal and type the phrase (the tests do
exactly that to exercise the real path). The gate stops accidental and config/flag/scheduler-driven promotion and leaves
an audit trail (tty, pid, parent process chain, user); it is not a cryptographic proof that a human was present.
No real orders can follow from any status here: PAPER_APPROVED means 'approved for paper', and only PaperBroker exists."""
from __future__ import annotations
import getpass, os, secrets, sqlite3, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from engine.forward.chain import ChainError, HashChain, sha
from engine.forward.scoring import ForwardEvidence, score
from engine.memory.store import Store

KINDS = {"ENTER_SHADOW": ({None}, "SHADOW"), "PAPER_APPROVED": ({"SHADOW"}, "PAPER_APPROVED"), "RETIRE": ({"SHADOW", "PAPER_APPROVED"}, "RETIRED")}
AUTOMATION_MODULES = ("engine.orchestration", "engine.agents", "engine.api", "engine.coding", "engine.models", "engine.research", "engine.forward.jobs", "adapters")
AUTOMATION_ENV = ("QAE_JOB_CONTEXT", "QAE_AGENT_CONTEXT", "QAE_SCHEDULER", "CI", "GITHUB_ACTIONS", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

class ApprovalRefused(PermissionError): """No approval record was written."""
class InsufficientEvidence(ApprovalRefused): """Frozen ForwardEvidence minimum not met; the human was not asked."""

def automation_context() -> str | None:
    """Why this call looks automated, or None. Env markers can only ADD a refusal; nothing can remove one."""
    for k in AUTOMATION_ENV:
        if os.environ.get(k): return f"environment marker {k} is set"
    f = sys._getframe(1)
    while f is not None:
        mod = f.f_globals.get("__name__", "")
        if any(mod == p or mod.startswith(p + ".") for p in AUTOMATION_MODULES): return f"called from automated code path {mod}"
        f = f.f_back
    return None

def _human_confirm(kind: str, spec_hash: str, summary: str) -> dict:
    if not (sys.stdin.isatty() and sys.stdout.isatty()): raise ApprovalRefused("approval needs an interactive terminal (stdin/stdout are not a TTY); it cannot be scripted, scheduled or piped")
    try: fd = os.open("/dev/tty", os.O_RDWR)
    except OSError as e: raise ApprovalRefused(f"no controlling terminal ({e}); approvals are human-only") from e
    try:
        code = secrets.token_hex(4); phrase = f"APPROVE {kind} {spec_hash} {code}"
        os.write(fd, f"\n=== HUMAN APPROVAL REQUIRED ===\n{summary}\nType exactly: {phrase}\n> ".encode())
        buf = b""
        while not buf.endswith(b"\n") and len(buf) < 4096:
            ch = os.read(fd, 1024)
            if not ch: break
            buf += ch
        if not secrets.compare_digest(buf.decode(errors="replace").strip(), phrase): raise ApprovalRefused("confirmation phrase did not match; nothing recorded")
        return {"tty": os.ttyname(fd), "challenge_sha": sha(code)}
    finally: os.close(fd)

def _ancestry() -> list[str]:
    """Best-effort parent process chain for the audit record (local `ps`; no network)."""
    out, pid = [], os.getpid()
    try:
        for _ in range(6):
            r = subprocess.run(["ps", "-o", "ppid=,comm=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.split(None, 1)
            if len(r) < 2: break
            out.append(f"{pid}:{os.path.basename(r[1].strip())}"); pid = int(r[0])
            if pid <= 1: break
    except Exception: pass
    return out

def signal_anchors(home: Path) -> list[tuple[int, str]]:
    """(seq, hash) of the signal-log head at each approval: lets SignalLog.verify detect a rewritten/truncated signal log."""
    c = HashChain(Path(home) / "approvals.jsonl")
    return [(r["body"]["signal_chain"]["n"] - 1, r["body"]["signal_chain"]["head"]) for r in c.verify() if r["body"]["signal_chain"]["n"] > 0]

class ApprovalLog:
    def __init__(self, home: Path):
        self.home = Path(home); self.chain = HashChain(self.home / "approvals.jsonl"); self.db_path = self.home / "forward.db"

    def _db(self) -> sqlite3.Connection:
        self.home.mkdir(parents=True, exist_ok=True); d = sqlite3.connect(self.db_path, timeout=30)
        d.execute("CREATE TABLE IF NOT EXISTS forward_status(spec_hash TEXT PRIMARY KEY, strategy_id INTEGER, status TEXT, approval_seq INTEGER, approval_hash TEXT, updated TEXT)")
        return d

    def _fold(self) -> dict[str, dict]:
        st: dict[str, dict] = {}
        for r in self.chain.verify():
            b = r["body"]; frm, to = KINDS[b["kind"]]; cur = st.get(b["spec_hash"], {}).get("status")
            if cur not in frm: raise ChainError(f"approval seq {r['seq']}: illegal forward transition {cur}->{to}")
            st[b["spec_hash"]] = {"status": to, "strategy_id": b["strategy_id"], "approval_seq": r["seq"], "approval_hash": r["hash"], "updated": b["ts"]}
        return st

    def all_status(self) -> dict[str, dict]:
        """Forward status per spec_hash, derived from the verified chain; raises if the forward.db mirror says anything else."""
        st = self._fold()
        if self.db_path.exists():
            with self._db() as d: mirror = {r[0]: (r[1], r[2], r[3]) for r in d.execute("SELECT spec_hash, status, approval_seq, approval_hash FROM forward_status")}
            want = {k: (v["status"], v["approval_seq"], v["approval_hash"]) for k, v in st.items()}
            if mirror != want: raise ChainError(f"forward.db status table disagrees with the approval chain (table edited outside approve()): {sorted(set(mirror.items()) ^ set(want.items()))}")
        return st

    def status(self, spec_hash: str) -> str | None:
        return self.all_status().get(spec_hash, {}).get("status")

    def approve(self, kind: str, store: Store, spec_hash: str, approver: str, reason: str, *, outcome_px: pd.DataFrame | None = None,
                expectation: dict | None = None) -> dict:
        """Create ONE approval record. Human-only (A1, A2). For PAPER_APPROVED the forward report is recomputed here from the
        verified signal log + the outcome bars given, and its hash becomes the evidence hash."""
        why = automation_context()                                                               # A1
        if why: raise ApprovalRefused(f"approval refused: {why}. Approvals are created only by a human at a terminal (python -m engine.forward.cli approve ...)")
        if kind not in KINDS: raise ValueError(f"kind must be one of {list(KINDS)}")
        if not approver or not approver.strip() or len((reason or "").strip()) < 10: raise ApprovalRefused("approver name and a reason (>= 10 chars) are required")
        from engine.forward.signals import SignalLog
        row = store.q("SELECT id, name, status FROM strategies WHERE spec_hash=?", spec_hash)
        if not row: raise ApprovalRefused(f"spec {spec_hash} is not in the research store")
        cur = self.status(spec_hash); frm, to = KINDS[kind]
        if cur not in frm: raise ApprovalRefused(f"illegal forward transition {cur}->{to}")
        if kind != "RETIRE" and row[0]["status"] != "CANDIDATE": raise ApprovalRefused(f"research-store status is {row[0]['status']}; only a CANDIDATE can be in forward evaluation")
        log = SignalLog(self.home); n, head = log.chain.head()
        log.records()                                                                            # also checks earlier anchors
        ev = ForwardEvidence(); evidence_hash, detail = None, ""
        if kind == "PAPER_APPROVED":                                                             # A3
            if outcome_px is None: raise InsufficientEvidence("PAPER_APPROVED needs forward outcome bars to recompute the report")
            rep = score(log, spec_hash, outcome_px, expectation=expectation, evidence=ev)
            if not rep["evidence"]["sufficient"]: raise InsufficientEvidence("forward evidence below the frozen minimum: " + "; ".join(rep["evidence"]["reasons"]))
            evidence_hash = rep["report_hash"]
            detail = f"forward report {evidence_hash[:16]}: {rep['n_eligible_days']} eligible days, paper return {rep['paper']['total_return']:+.2%}, max DD {rep['paper']['max_drawdown']:.2%}, z vs backtest {rep['expectation']['z_vs_backtest']:+.2f}\n"
        elif kind == "ENTER_SHADOW":
            e = store.q("SELECT id, decision, criteria_hash, data_sha, metrics FROM experiments WHERE spec_hash=? ORDER BY id DESC LIMIT 1", spec_hash)
            evidence_hash = sha(repr(sorted(e[0].items()))) if e else None
            detail = f"research evidence: {'experiment #%s decision %s' % (e[0]['id'], e[0]['decision']) if e else 'NO stored experiment for this spec'}\n"
        proof = _human_confirm(kind, spec_hash, f"{kind}: {row[0]['name']} (spec {spec_hash}), forward status {cur} -> {to}\n{detail}approver: {approver}\nreason: {reason}")   # A2
        body = {"type": "approval", "kind": kind, "strategy_id": row[0]["id"], "strategy_name": row[0]["name"], "spec_hash": spec_hash, "from": cur, "to": to,
                "approver": approver.strip(), "reason": reason.strip(), "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "evidence_hash": evidence_hash, "evidence_rule_hash": ev.digest(), "signal_chain": {"n": n, "head": head},
                "proof": {**proof, "pid": os.getpid(), "os_user": getpass.getuser(), "process_chain": _ancestry()}}
        rec = self.chain.append(body)
        with self._db() as d:
            d.execute("INSERT OR REPLACE INTO forward_status VALUES(?,?,?,?,?,?)", (spec_hash, row[0]["id"], to, rec["seq"], rec["hash"], body["ts"]))
        return rec
