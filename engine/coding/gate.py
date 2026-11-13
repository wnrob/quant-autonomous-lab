"""Hard human-approval gate for the coding workflow.

The protected list is CODE, not config: a frozen tuple whose sha256 is pinned below and re-checked at import time.
Nothing here reads a config file, an environment variable or a CLI flag to decide whether a patch is protected.
(The only environment influence is in the *stricter* direction: scheduler/CI markers make `approve` refuse.)

Honest limit: an approval record is a file and there is NO signing key. Any process with write access to
artifacts/coding/approvals can forge one (the self-hash is a consistency check, not a signature);
what this module guarantees is that (a) no code path in the workflow creates one without an interactive terminal and a typed
confirmation, (b) a record only ever authorises the exact patch bytes it names, and (c) approval never widens the sandbox.
"""
from __future__ import annotations
import hashlib, json, os, re, sys, time, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from engine.errors import QALError

class GateError(QALError): pass
class ApprovalRefused(GateError): pass

# Patterns: "dir/**" = everything under dir; "*name" / "name*" = basename glob; otherwise exact path. Matched case-insensitively
# (macOS filesystems are case-insensitive: "Engine/Coding/x.py" lands in engine/coding/).
PROTECTED_PATHS: tuple[str, ...] = (
    "engine/coding/**",                 # the workflow cannot modify itself
    "tests/test_coding*",               # ... nor the tests that pin its guards
    "engine/memory/**",                 # registry / status machine / protected-test accounting
    "engine/research/**",               # frozen criteria, splits, risk gate, hypothesis accounting
    "engine/agents/**",                 # agent permissions, protected-test look caps, local-LLM integration
    "engine/backtesting/**",            # backtester (no-lookahead lag) and the paper order path
    "engine/orchestration/**",          # scheduler, budgets, cycle
    "engine/forward/**",                # forward/shadow evaluation + its approval gate
    "engine/api/**",                    # served surface
    "engine/data/**",                   # acquisition, quality checks, universes / point-in-time membership
    "engine/models/**",                 # ML ladder, registry pickle-load integrity
    "engine/families/**", "engine/derivs/**", "engine/futures/**", "engine/eventsim/**", "engine/strategies/**",
    "engine/qae_cli.py", "engine/cli.py", "engine/errors.py", "engine/logging_setup.py", "engine/__init__.py",
    "adapters/**",                      # external integrations
    "scripts/**",                       # sandbox scripts, builders
    "configs/**",
    "pyproject.toml",
    "conftest.py", "*/conftest.py",     # changes how every test runs
    "sitecustomize.py", "*/sitecustomize.py", "usercustomize.py", "*/usercustomize.py", "*.pth", "*.sh",
)
# NOT protected (deliberately small): engine/analytics/**, engine/risk/**, engine/optimization/**, brand-new top-level engine
# packages, and tests/ files other than conftest.py and tests/test_coding*. Everything else needs a human.

# Added-line content scan: any hit forces the gate even on an unprotected path. Deliberately broad (fails closed; "order" in
# a docstring will ask for a human, which is the intended bias).
CONTENT_KEYWORDS: tuple[str, ...] = (
    r"broker", r"order", r"credential", r"exec", r"secret", r"password", r"passwd", r"token", r"api[_-]?key", r"private[_-]?key",
    r"account", r"live[_-]?trad", r"subprocess", r"socket", r"urllib", r"requests", r"httpx", r"http[s]?://", r"os\.system",
    r"popen", r"eval\s*\(", r"__import__", r"importlib", r"pickle", r"ctypes", r"os\.environ", r"getenv", r"sandbox",
    r"chmod", r"symlink", r"\.ssh", r"keychain",
)

def _digest(paths: tuple[str, ...], kws: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps([list(paths), list(kws)]).encode()).hexdigest()

PROTECTED_SHA256 = "e4b2faa0789703dd9f70e167c99e43a6b62c5fe3496a07a7f0f8e807fe15b875"
if _digest(PROTECTED_PATHS, CONTENT_KEYWORDS) != PROTECTED_SHA256:      # pragma: no cover - tripwire
    raise GateError("engine.coding.gate: protected list does not match its pinned hash (edited without re-pinning?)")
_KW_RE = re.compile("|".join(f"(?:{k})" for k in CONTENT_KEYWORDS), re.IGNORECASE)

def norm(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()

def _match(pattern: str, path: str) -> bool:
    p = norm(path); base = p.rsplit("/", 1)[-1]
    if pattern.endswith("/**"):
        pre = pattern[:-3]; return p == pre or p.startswith(pre + "/")
    if pattern.startswith("*/"): return p.endswith(pattern[1:])
    if pattern.startswith("*"): return base.endswith(pattern[1:])
    if pattern.endswith("*"): return p.startswith(pattern[:-1])
    return p == pattern

def protected_hits(paths) -> list[dict]:
    return [{"path": p, "rule": pat} for p in sorted(paths) for pat in PROTECTED_PATHS if _match(pat, p)]

def content_hits(added_lines: dict[str, list[str]]) -> list[dict]:
    out = []
    for path in sorted(added_lines):
        for ln in added_lines[path]:
            m = _KW_RE.search(ln)
            if m: out.append({"path": path, "keyword": m.group(0).lower(), "line": ln.strip()[:160]})
    return out

@dataclass(frozen=True)
class GateDecision:
    decision: str                      # NOT_PROTECTED | BLOCKED_NEEDS_APPROVAL | APPROVED_BY_HUMAN
    protected: list = field(default_factory=list)
    content: list = field(default_factory=list)
    approval: dict | None = None
    protected_list_sha256: str = PROTECTED_SHA256
    @property
    def requires_approval(self) -> bool: return bool(self.protected or self.content)
    @property
    def may_proceed(self) -> bool: return self.decision in ("NOT_PROTECTED", "APPROVED_BY_HUMAN")

def evaluate(paths, added_lines: dict[str, list[str]], patch_sha256: str, approvals_dir: Path) -> GateDecision:
    """No keyword arguments, no config, no env: the decision is a pure function of the patch and the approval records on disk."""
    prot, cont = protected_hits(paths), content_hits(added_lines)
    if not prot and not cont: return GateDecision("NOT_PROTECTED")
    rec = load_approval(approvals_dir, patch_sha256)
    return GateDecision("APPROVED_BY_HUMAN" if rec else "BLOCKED_NEEDS_APPROVAL", prot, cont, rec)

# ---------------------------------------------------------------- approval records
_AGENT_NAMES = ("agent", "llm", "auto", "bot", "scheduler", "cron", "claude", "qwen", "ollama", "system", "root", "none", "null", "ci")
_SCHEDULER_ENV = ("QAE_SCHEDULER", "QAE_JOB", "QAE_JOB_CONTEXT", "QAE_AGENT_CONTEXT", "QAE_NONINTERACTIVE", "CI", "GITHUB_ACTIONS", "CRON", "LAUNCHD_JOB",
                  "XPC_SERVICE_NAME_QAE", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
# approve() refuses when ANY frame on its call stack belongs to one of these (automation must not be able to reach it in-process)
_AUTOMATION_MODULES = ("engine.orchestration", "engine.agents", "engine.api", "engine.forward", "engine.models", "engine.research",
                       "engine.coding.workflow", "engine.coding.llm_patch", "adapters")

def automation_caller() -> str | None:
    f = sys._getframe(1)
    while f is not None:
        mod = f.f_globals.get("__name__", "")
        if any(mod == p or mod.startswith(p + ".") for p in _AUTOMATION_MODULES): return mod
        f = f.f_back
    return None

def _record_hash(rec: dict) -> str:
    body = {k: rec[k] for k in ("approver", "reason", "patch_sha256", "created", "scope")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

def _valid_fields(approver: str, reason: str, patch_sha256: str) -> str | None:
    a = (approver or "").strip()
    if len(a) < 2 or a.casefold() in _AGENT_NAMES: return "approver must name a human (not an agent/automation identity)"
    if len((reason or "").strip()) < 10: return "reason must be at least 10 characters"
    if not re.fullmatch(r"[0-9a-f]{64}", patch_sha256 or ""): return "patch_sha256 must be a full sha256 hex digest"
    return None

def load_approval(approvals_dir: Path, patch_sha256: str) -> dict | None:
    """Return the record only if it is well-formed AND names exactly this patch hash. Anything else -> None (= blocked)."""
    p = Path(approvals_dir) / f"{patch_sha256}.json"
    try:
        if p.is_symlink() or not p.is_file(): return None
        rec = json.loads(p.read_text())
        if not isinstance(rec, dict): return None
        if rec.get("patch_sha256") != patch_sha256: return None
        if _valid_fields(rec.get("approver"), rec.get("reason"), rec.get("patch_sha256")): return None
        if rec.get("interactive_tty") is not True: return None
        if rec.get("record_sha256") != _record_hash(rec): return None
        return rec
    except (OSError, ValueError, KeyError, TypeError):
        return None

def interactive_context() -> tuple[bool, str]:
    """True only on a real terminal with no scheduler/CI marker. Takes no arguments on purpose: callers cannot assert interactivity."""
    for k in _SCHEDULER_ENV:
        if os.environ.get(k): return False, f"scheduler/CI/agent context ({k} is set)"
    try:
        if not (os.isatty(0) and os.isatty(1)): return False, "stdin/stdout is not a terminal"
    except OSError:
        return False, "stdin/stdout is not a terminal"
    return True, "tty"

def approve(approvals_dir: Path, patch_sha256: str, approver: str, reason: str, summary: str = "") -> dict:
    """Create an approval record. Refuses unless run by a person at a terminal who re-types the patch hash prefix."""
    auto = automation_caller()
    if auto: raise ApprovalRefused(f"approval refused: called from automated code path {auto}")
    ok, why = interactive_context()
    if not ok: raise ApprovalRefused(f"approval refused: {why}. A human must run this at a terminal.")
    bad = _valid_fields(approver, reason, patch_sha256)
    if bad: raise ApprovalRefused(f"approval refused: {bad}")
    want = patch_sha256[:12]
    sys.stdout.write(f"{summary}\nApproving patch sha256 {patch_sha256}\nType the first 12 hex characters of the hash to confirm: "); sys.stdout.flush()
    with open(0, "r", closefd=False) as tty: typed = tty.readline().strip()
    if typed != want: raise ApprovalRefused("approval refused: confirmation did not match the patch hash")
    rec = {"approver": approver.strip(), "reason": reason.strip(), "patch_sha256": patch_sha256, "created": time.time(),
           "scope": "exact patch bytes only; sandbox, secrets and network policy unchanged", "interactive_tty": True}
    rec["record_sha256"] = _record_hash(rec)
    d = Path(approvals_dir); d.mkdir(parents=True, exist_ok=True)
    (d / f"{patch_sha256}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec
