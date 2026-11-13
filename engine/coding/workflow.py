"""Coding-agent improvement workflow: isolated workspace -> gate -> sandboxed before/after tests + golden diff -> report.

Never modifies the lab source in `propose`. The only code that writes to the lab is `apply` (separate, approval-gated, backed
up, hash-verified, reversible via `restore`).
"""
from __future__ import annotations
import fcntl, hashlib, json, os, shutil, signal, stat, subprocess, time, uuid
from pathlib import Path
from engine.errors import QALError
from engine.coding import gate, golden
from engine.coding.patch import ALLOWED_FILES, ALLOWED_ROOTS, ParsedPatch, PatchRejected, check_against_tree, parse_patch, sha256_text

class CodingError(QALError): pass
class ApplyRefused(CodingError): pass

TRUSTED_ROOT = Path(__file__).resolve().parents[2]                     # the real lab: sandbox script, runner, interpreter come from here only
SANDBOX_SCRIPT = TRUSTED_ROOT / "scripts" / "coding_sandbox_run.sh"
RUNNER, GOLDEN = Path(__file__).with_name("ws_runner.py"), Path(__file__).with_name("golden.py")
PYENV = (TRUSTED_ROOT / "envs" / "core").resolve()                      # resolved: sandbox rules match real paths
PYTHON = PYENV / "bin" / "python"
MAX_RESULT_BYTES = 20_000_000

def _exec_allowlist() -> list[str]:
    """The ONLY binaries the sandbox may exec: the env's interpreter and the base interpreter install it points at."""
    real = Path(os.path.realpath(PYTHON)); out = [str(PYTHON), str(real)]
    for parent in real.parents:                                         # .../Python.framework/Versions/3.x (framework builds re-exec Python.app)
        if parent.parent.name == "Versions" and parent.parent.parent.name.endswith(".framework"): out.append(str(parent)); break
    return out

def _read_untrusted_json(path: Path) -> dict:
    """Read a file the SANDBOXED process wrote: regular file only, never through a symlink, size-capped."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1: raise ValueError("result is not a plain single-link regular file")
        if st.st_size > MAX_RESULT_BYTES: raise ValueError("result file too large")
        data = b""
        while len(data) <= MAX_RESULT_BYTES:
            chunk = os.read(fd, 1 << 20)
            if not chunk: break
            data += chunk
    finally: os.close(fd)
    out = json.loads(data)
    if not isinstance(out, dict): raise ValueError("result is not a JSON object")
    return out

def _reap_survivors(ws: Path, tag: str) -> dict:
    """Best effort: kill processes left behind by sandboxed code (fork+setsid escapes killpg). Matches the run tag in the
    process environment or the workspace path in its command line. A survivor that scrubbed both is NOT found."""
    found = []
    try:
        ps = subprocess.run(["/bin/ps", "-axEww", "-o", "pid=,command="], capture_output=True, text=True, timeout=20, env={"PATH": "/usr/bin:/bin"}).stdout
        for line in ps.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != os.getpid() and (tag in parts[1] or str(ws) in parts[1]):
                found.append(int(parts[0]))
        for pid in found:
            try: os.kill(pid, signal.SIGKILL)
            except OSError: pass
    except Exception as e:                                               # noqa: BLE001
        return {"scanned": False, "error": str(e)[:120], "killed": found}
    return {"scanned": True, "killed": found}
DEFAULT_PYTEST_ARGS = ("tests", "-m", "not slow and not network", "--ignore-glob=tests/test_coding*")
STATUSES = ("READY_FOR_HUMAN_REVIEW", "ROLLED_BACK", "BLOCKED_NEEDS_APPROVAL", "REJECTED_INVALID_PATCH", "APPLIED", "APPLY_ROLLED_BACK", "RESTORED")
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_SKIP_SUFFIX = (".pyc", ".pyo", ".bak", ".DS_Store", ".sb", ".tmp")
MAX_WS_BYTES = 50_000_000

def _sha_file(p: Path) -> str | None:
    try:
        if p.is_symlink() or not p.is_file(): return None
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError: return None

def _write_file(dst: Path, data: bytes) -> None:
    """Atomic single-file write (module-level so tests can inject a mid-apply failure)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.coding_tmp"); tmp.write_bytes(data); os.replace(tmp, dst)

def _git(ws: Path, *args: str, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Host-side git with every config source neutralised (hooks off, no global/system config). Never run inside the sandbox."""
    cmd = ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
           "-c", "core.fileMode=true", "-c", "user.name=qae-coding", "-c", "user.email=qae-coding@localhost", "-c", "gc.auto=0", *args]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(ws.parent), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    r = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, input=stdin, timeout=120)
    if check and r.returncode != 0: raise CodingError(f"git {' '.join(args[:2])} failed: {r.stderr.strip()[-400:]}")
    return r

class CodingWorkflow:
    def __init__(self, lab_root: Path | None = None, out_root: Path | None = None, test_timeout: int = 1800):
        self.lab = Path(lab_root or TRUSTED_ROOT).resolve()
        self.out = Path(out_root or self.lab / "artifacts" / "coding").resolve()
        self.timeout = test_timeout
        for d in ("ws", "reports", "patches", "approvals", "staged", "backups"): (self.out / d).mkdir(parents=True, exist_ok=True)
        if self.lab == TRUSTED_ROOT and not str(self.out).startswith(str(TRUSTED_ROOT / "artifacts" / "coding")) and TRUSTED_ROOT in self.out.parents:
            raise CodingError("out_root inside the lab must be artifacts/coding")
    @property
    def approvals_dir(self) -> Path: return self.out / "approvals"
    @property
    def index_path(self) -> Path: return self.out / "index.jsonl"

    # ------------------------------------------------------------ workspace
    def _copy_source(self, ws: Path) -> dict:
        n = size = skipped_links = 0
        def copy_file(src: Path, rel: str):
            nonlocal n, size
            dst = ws / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); n += 1; size += src.stat().st_size
            if size > MAX_WS_BYTES: raise CodingError("workspace copy exceeds size cap")
        for root_name in ALLOWED_ROOTS:
            src_root = self.lab / root_name
            if src_root.is_symlink() or not src_root.is_dir(): continue
            for dirpath, dirnames, filenames in os.walk(src_root, followlinks=False):
                keep = []
                for d in dirnames:
                    if d in _SKIP_DIRS: continue
                    if os.path.islink(os.path.join(dirpath, d)): skipped_links += 1; continue
                    keep.append(d)
                dirnames[:] = keep
                for f in filenames:
                    p = Path(dirpath) / f
                    if f.endswith(_SKIP_SUFFIX): continue
                    if p.is_symlink() or not p.is_file(): skipped_links += 1; continue        # symlinks are never copied
                    try: copy_file(p, str(p.relative_to(self.lab)))
                    except FileNotFoundError: pass                                           # concurrent writers in the lab
        for f in ALLOWED_FILES:
            p = self.lab / f
            if p.is_file() and not p.is_symlink(): copy_file(p, f)
        return {"files": n, "bytes": size, "symlinks_skipped": skipped_links}

    def _make_workspace(self, run_id: str) -> tuple[Path, dict, str]:
        ws = self.out / "ws" / run_id
        if ws.exists(): raise CodingError("workspace already exists")
        ws.mkdir(parents=True)
        info = self._copy_source(ws)
        _git(ws, "init", "-q", "-b", "main")
        (ws / ".git" / "info" / "exclude").write_text(".coding_*\n__pycache__/\n*.pyc\n.pytest_cache/\n")
        _git(ws, "add", "-A"); _git(ws, "commit", "-q", "-m", "baseline: snapshot of lab source", "--allow-empty")
        base = _git(ws, "rev-parse", "HEAD").stdout.strip()
        info["baseline_commit"], info["baseline_tree"] = base, _git(ws, "rev-parse", "HEAD^{tree}").stdout.strip()
        return ws, info, base

    def _sandbox_run(self, ws: Path, tag_name: str, profile_out: Path, pytest_args) -> dict:
        """One sandboxed pytest + golden pass over the workspace. Result JSON is read back and removed from the workspace at once."""
        out_rel = f".coding_out/{tag_name}.json"; out_file = ws / out_rel
        if out_file.exists(): out_file.unlink()
        if out_file.is_symlink() or (ws / ".coding_out").is_symlink(): raise CodingError("result path is a symlink")
        ro = ":".join([str(PYENV), str(RUNNER), str(GOLDEN)]); tag = "codingrun-" + uuid.uuid4().hex
        cmd = [str(SANDBOX_SCRIPT), str(ws), str(profile_out), ro, ":".join(_exec_allowlist()), f"QAE_HOME={ws}/.coding_out/qae_{tag_name}", f"CODING_RUN_TAG={tag}", "--",
               str(PYTHON), "-I", "-W", "ignore", str(RUNNER), str(ws), str(out_file), str(GOLDEN), *pytest_args]
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=ws, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                             env={"PATH": "/usr/bin:/bin"})
        timed_out = False
        try: so, se = p.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try: os.killpg(p.pid, signal.SIGKILL)
            except (PermissionError, ProcessLookupError): p.kill()
        survivors = _reap_survivors(ws, tag)                              # before draining pipes: a setsid'd survivor may hold them open
        if timed_out:
            try: so, se = p.communicate(timeout=30)
            except subprocess.TimeoutExpired: so, se = "", "process did not exit after SIGKILL"
        res: dict = {"tests": {}, "collect_errors": [], "golden": None, "runner_error": None}
        try: res.update(_read_untrusted_json(out_file))
        except (OSError, ValueError) as e: res["runner_error"] = f"no readable result file ({type(e).__name__}: {str(e)[:80]}); stderr tail: {se[-300:]}"
        if timed_out: res["runner_error"] = f"timed out after {self.timeout}s (process group killed)"
        if survivors.get("killed"): res["runner_error"] = f"sandboxed code left {len(survivors['killed'])} process(es) running after the run (killed)"
        res["survivors"] = survivors
        try:
            if not out_file.is_symlink(): out_file.unlink()
        except OSError: pass
        if not isinstance(res.get("tests"), dict): res["tests"], res["runner_error"] = {}, "malformed result file"
        res.update(returncode=p.returncode, secs=round(time.time() - t0, 1), stdout_tail=so[-600:], stderr_tail=se[-600:])
        return res

    @staticmethod
    def _summ(res: dict) -> dict:
        c: dict = {}
        for v in res["tests"].values(): c[v] = c.get(v, 0) + 1
        return {"counts": c, "total": len(res["tests"]), "collect_errors": res.get("collect_errors", []), "runner_error": res.get("runner_error"),
                "pytest_exit": res.get("pytest_exit"), "secs": res.get("secs"), "engine_file": res.get("engine_file"), "survivors": res.get("survivors")}

    def _rollback(self, ws: Path, base: str, tree: str) -> dict:
        try:
            _git(ws, "reset", "-q", "--hard", base); _git(ws, "clean", "-q", "-fdx")
            clean = _git(ws, "status", "--porcelain").stdout.strip() == ""
            same = _git(ws, "rev-parse", "HEAD^{tree}").stdout.strip() == tree
            return {"performed": True, "workspace_reset_verified": bool(clean and same)}
        except Exception as e:                                           # noqa: BLE001
            return {"performed": True, "workspace_reset_verified": False, "error": str(e)[:200]}

    # ------------------------------------------------------------ propose
    def propose(self, patch_text: str, intent: dict | None = None, source: str = "human", pytest_args=None, keep_workspace: bool = False) -> dict:
        intent = dict(intent or {}); declared = [str(x) for x in intent.get("golden_changes", [])]
        run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        rep: dict = {"run_id": run_id, "started": time.time(), "source": source, "intent": {"summary": str(intent.get("summary", ""))[:500], "golden_changes": declared},
                     "patch_sha256": sha256_text(patch_text) if isinstance(patch_text, str) else None, "lab_root": str(self.lab), "status": None, "reasons": [],
                     "applied_to_lab": False, "sandbox": None, "gate": None, "tests": None, "golden": None,
                     "rollback": {"performed": False}, "workspace": None, "timeline": []}
        def mark(step): rep["timeline"].append({"step": step, "ts": time.time()})
        ws = None; touched_before: dict = {}
        try:
            mark("validate_patch")
            try: parsed = parse_patch(patch_text)
            except PatchRejected as e:
                rep["status"] = "REJECTED_INVALID_PATCH"; rep["reasons"].append(str(e)); return self._finish(rep)
            rep["files_touched"] = [{"path": f.path, "kind": f.kind, "added": len(f.added), "removed": f.removed} for f in parsed.files]
            (self.out / "patches" / f"{parsed.sha256}.diff").write_text(patch_text)
            unknown = [d for d in declared if d not in golden.PROBE_NAMES]
            if unknown:
                rep["status"] = "REJECTED_INVALID_PATCH"; rep["reasons"].append(f"intent declares unknown golden probes: {unknown}"); return self._finish(rep)
            mark("gate")
            dec = gate.evaluate(parsed.paths, parsed.added_lines, parsed.sha256, self.approvals_dir)
            rep["gate"] = {"decision": dec.decision, "requires_approval": dec.requires_approval, "protected_hits": dec.protected, "content_hits": dec.content[:40],
                           "approval": dec.approval, "protected_list_sha256": dec.protected_list_sha256}
            if not dec.may_proceed:
                rep["status"] = "BLOCKED_NEEDS_APPROVAL"
                rep["reasons"].append("patch touches protected paths/keywords; a human must create an approval record for this exact patch sha256")
                return self._finish(rep)
            touched_before = {p: _sha_file(self.lab / p) for p in parsed.paths}
            mark("workspace")
            ws, info, base = self._make_workspace(run_id); rep["workspace"] = {**info, "path": str(ws), "kept": keep_workspace}
            try: check_against_tree(parsed, ws)
            except PatchRejected as e:
                rep["status"] = "REJECTED_INVALID_PATCH"; rep["reasons"].append(str(e)); return self._finish(rep, ws, keep_workspace)
            args = list(pytest_args if pytest_args is not None else DEFAULT_PYTEST_ARGS)
            profile = self.out / "reports" / f"{run_id}.sandbox.sb"
            mark("baseline_tests")
            before = self._sandbox_run(ws, "before", profile, args)
            rep["sandbox"] = {"script": str(SANDBOX_SCRIPT), "script_sha256": _sha_file(SANDBOX_SCRIPT), "profile_path": str(profile), "profile_sha256": _sha_file(profile),
                              "runner_sha256": _sha_file(RUNNER), "golden_sha256": _sha_file(GOLDEN), "interpreter": str(PYTHON), "pytest_args": args,
                              "network": "denied except loopback", "env": "env -i (scrubbed)", "secrets": "none passed; /Users unreadable except workspace + interpreter env",
                              "profile": "deny-default", "exec_allowlist": _exec_allowlist()}
            if before.get("runner_error"):
                rep["tests"] = {"before": self._summ(before)}
                raise _Fail(f"baseline run could not be established: {before['runner_error']}")
            # the baseline run executed test code with write access to the tree: restore it before applying the patch
            _git(ws, "reset", "-q", "--hard", base); _git(ws, "clean", "-q", "-fdx")
            mark("apply_patch_in_workspace")
            chk = _git(ws, "apply", "--check", "--whitespace=nowarn", "-", check=False, stdin=patch_text)
            if chk.returncode != 0: raise _Fail(f"patch does not apply to the workspace: {chk.stderr.strip()[-300:]}")
            _git(ws, "apply", "--whitespace=nowarn", "-", stdin=patch_text)
            changed = sorted(l[3:] for l in _git(ws, "status", "--porcelain", "--untracked-files=all").stdout.splitlines() if l.strip())
            if changed != sorted(parsed.paths): raise _Fail(f"applied change set {changed} differs from validated paths {sorted(parsed.paths)}")
            for p in parsed.paths:
                fp = ws / p
                if fp.is_symlink() or (fp.exists() and not fp.is_file()): raise _Fail(f"patched path is not a plain file: {p}")
            _git(ws, "add", "-A")
            for raw in _git(ws, "diff", "--cached", "--raw", "--no-renames").stdout.splitlines():
                om, nm = raw.split()[0].lstrip(":"), raw.split()[1]
                if nm not in ("100644", "000000") and nm != om: raise _Fail(f"patch introduces a non-100644 mode ({om}->{nm}): {raw.split()[-1]}")
                if om not in ("000000", nm) and nm != "000000": raise _Fail(f"patch changes a file mode ({om}->{nm}): {raw.split()[-1]}")
            _git(ws, "commit", "-q", "-m", f"patch {parsed.sha256[:12]}")
            patched = _git(ws, "rev-parse", "HEAD").stdout.strip()
            rep["workspace"]["patched_commit"] = patched
            rep["workspace"]["diffstat"] = _git(ws, "diff", "--stat", base, patched).stdout.strip().splitlines()[-1:]
            staged = self._stage(ws, run_id, parsed, patched, touched_before)       # from the commit, before untrusted code runs
            mark("patched_tests")
            after = self._sandbox_run(ws, "after", profile, args)
            if _sha_file(profile) != rep["sandbox"]["profile_sha256"]: raise _Fail("sandbox profile differed between baseline and patched runs")
            mark("compare")
            bt, at = before["tests"], after["tests"]
            regress = sorted(t for t, o in bt.items() if o == "passed" and at.get(t) != "passed")
            rep["tests"] = {"before": self._summ(before), "after": self._summ(after),
                            "regressions": [{"test": t, "before": "passed", "after": at.get(t, "<missing>")} for t in regress],
                            "newly_passing": sorted(t for t, o in at.items() if o == "passed" and bt.get(t) != "passed"),
                            "new_tests": sorted(t for t in at if t not in bt), "removed_tests": sorted(t for t in bt if t not in at),
                            "pre_existing_failures": sorted(t for t, o in bt.items() if o in ("failed", "error"))}
            g = golden.compare(before.get("golden") or {}, after.get("golden") or {}, declared) if not after.get("runner_error") else None
            rep["golden"] = g
            fails = []
            if after.get("runner_error"): fails.append(f"patched run failed: {after['runner_error']}")
            if regress: fails.append(f"{len(regress)} test(s) passed before and do not pass after")
            new_bad = sorted(t for t, o in at.items() if o in ("failed", "error") and t not in bt)
            if new_bad: fails.append(f"{len(new_bad)} new test(s) added by the patch fail: {new_bad[:5]}")
            ce = [c for c in after.get("collect_errors", []) if c["nodeid"] not in {b["nodeid"] for b in before.get("collect_errors", [])}]
            if ce: fails.append(f"new collection errors: {[c['nodeid'] for c in ce]}")
            if g is not None and not g["ok"]: fails.append(f"golden outputs changed without being declared in intent: {g['undeclared_changed']}")
            if fails: raise _Fail(*fails)
            rep["status"] = "READY_FOR_HUMAN_REVIEW"; rep["staged"] = staged
            rep["reasons"].append("tests: no regressions; golden: no undeclared diffs. NOT applied to the lab; `apply` needs a human approval record.")
        except _Fail as f:
            rep["status"] = "ROLLED_BACK"; rep["reasons"].extend(f.args)
            if ws is not None: rep["rollback"] = self._rollback(ws, rep["workspace"]["baseline_commit"], rep["workspace"]["baseline_tree"])
            shutil.rmtree(self.out / "staged" / run_id, ignore_errors=True); rep["rollback"]["staged_discarded"] = True
        except Exception as e:                                           # noqa: BLE001 - any unexpected error is a failed run, never a pass
            rep["status"] = "ROLLED_BACK"; rep["reasons"].append(f"internal error: {type(e).__name__}: {str(e)[:300]}")
            if ws is not None and rep.get("workspace"): rep["rollback"] = self._rollback(ws, rep["workspace"]["baseline_commit"], rep["workspace"]["baseline_tree"])
            shutil.rmtree(self.out / "staged" / run_id, ignore_errors=True); rep["rollback"]["staged_discarded"] = True
        if touched_before:
            now = {p: _sha_file(self.lab / p) for p in touched_before}
            rep["lab_untouched"] = {"checked_paths": sorted(touched_before), "unchanged": now == touched_before}
            if now != touched_before: rep["reasons"].append("WARNING: lab copies of the touched files changed during the run (concurrent writer?)")
        return self._finish(rep, ws, keep_workspace)

    def _stage(self, ws: Path, run_id: str, parsed: ParsedPatch, commit: str, pre: dict) -> dict:
        d = self.out / "staged" / run_id; man = {"run_id": run_id, "patch_sha256": parsed.sha256, "files": []}
        for f in parsed.files:
            if f.kind == "delete": man["files"].append({"path": f.path, "kind": f.kind, "pre_sha256": pre[f.path], "post_sha256": None}); continue
            data = subprocess.run(["/usr/bin/git", "-c", "core.hooksPath=/dev/null", "show", f"{commit}:{f.path}"], cwd=ws, capture_output=True, timeout=60,
                                  env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(ws.parent)})
            if data.returncode != 0: raise CodingError(f"cannot stage {f.path}")
            dst = d / "files" / f.path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data.stdout)
            man["files"].append({"path": f.path, "kind": f.kind, "pre_sha256": pre[f.path], "post_sha256": hashlib.sha256(data.stdout).hexdigest()})
        (d / "manifest.json").parent.mkdir(parents=True, exist_ok=True); (d / "manifest.json").write_text(json.dumps(man, indent=1))
        man["manifest_sha256"] = _sha_file(d / "manifest.json"); return man

    # ------------------------------------------------------------ reports + hash-chained index
    def _finish(self, rep: dict, ws: Path | None = None, keep_workspace: bool = False) -> dict:
        if ws is not None and not keep_workspace:
            shutil.rmtree(ws, ignore_errors=True)
            if rep.get("workspace"): rep["workspace"]["removed"] = not ws.exists()
        rep["finished"] = time.time(); rep["secs"] = round(rep["finished"] - rep["started"], 1)
        return self._write_report(rep, rep["run_id"])

    def _write_report(self, rep: dict, name: str) -> dict:
        jp, mp = self.out / "reports" / f"{name}.json", self.out / "reports" / f"{name}.md"
        if jp.exists(): raise CodingError("report already exists (reports are write-once)")
        body = json.dumps(rep, indent=1, sort_keys=True, default=str); jp.write_text(body); mp.write_text(render_markdown(rep))
        entry = self._append_index({"run_id": rep["run_id"], "report": jp.name, "status": rep["status"], "patch_sha256": rep.get("patch_sha256"),
                                    "report_sha256": hashlib.sha256(body.encode()).hexdigest(), "ts": rep["finished"]})
        rep["index_entry"] = entry; rep["report_paths"] = {"json": str(jp), "md": str(mp)}
        return rep

    def _append_index(self, entry: dict) -> dict:
        with open(self.index_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.seek(0)
            lines = [l for l in f.read().splitlines() if l.strip()]
            prev = json.loads(lines[-1])["entry_hash"] if lines else "0" * 64
            e = {"seq": len(lines), **entry, "prev_hash": prev}
            e["entry_hash"] = hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()
            f.write(json.dumps(e, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())
        return e

    def verify_index(self) -> dict:
        """Recompute the chain and every report hash. Edit/delete/reorder of a line, or an edited report, is reported."""
        problems = []; prev = "0" * 64; n = 0
        if self.index_path.exists():
            for i, line in enumerate(l for l in self.index_path.read_text().splitlines() if l.strip()):
                n += 1
                try: e = json.loads(line)
                except ValueError: problems.append(f"line {i}: not JSON"); break
                body = {k: v for k, v in e.items() if k != "entry_hash"}
                if e.get("seq") != i: problems.append(f"line {i}: seq {e.get('seq')} out of order")
                if e.get("prev_hash") != prev: problems.append(f"line {i}: prev_hash does not chain")
                if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != e.get("entry_hash"): problems.append(f"line {i}: entry_hash mismatch (edited)")
                rp = self.out / "reports" / str(e.get("report"))
                if _sha_file(rp) != e.get("report_sha256"): problems.append(f"line {i}: report {e.get('report')} missing or modified")
                prev = e.get("entry_hash")
        return {"ok": not problems, "entries": n, "problems": problems, "head": prev}

    def status(self, run_id: str | None = None):
        if run_id: return json.loads((self.out / "reports" / f"{run_id}.json").read_text())
        return [json.loads(l) for l in self.index_path.read_text().splitlines() if l.strip()] if self.index_path.exists() else []

    # ------------------------------------------------------------ approval / apply / restore
    def _load_for_apply(self, run_id: str) -> tuple[dict, dict, Path]:
        v = self.verify_index()
        if not v["ok"]: raise ApplyRefused(f"report index failed verification: {v['problems'][:3]}")
        rep = self.status(run_id)
        if rep.get("status") != "READY_FOR_HUMAN_REVIEW": raise ApplyRefused(f"run status is {rep.get('status')}, not READY_FOR_HUMAN_REVIEW")
        if not any(e["run_id"] == run_id and e["report"] == f"{run_id}.json" for e in self.status()): raise ApplyRefused("run is not in the verified index")
        d = self.out / "staged" / run_id
        if _sha_file(d / "manifest.json") != rep["staged"]["manifest_sha256"]: raise ApplyRefused("staged manifest missing or modified since the run")
        man = json.loads((d / "manifest.json").read_text())
        if man["patch_sha256"] != rep["patch_sha256"]: raise ApplyRefused("staged manifest is for a different patch")
        for f in man["files"]:
            if f["kind"] != "delete" and _sha_file(d / "files" / f["path"]) != f["post_sha256"]: raise ApplyRefused(f"staged file modified since the run: {f['path']}")
        return rep, man, d

    def apply(self, run_id: str) -> dict:
        """The ONLY code path that writes to the lab. Needs a human approval record for the exact patch; restores from backup on any failure."""
        for k in gate._SCHEDULER_ENV:
            if os.environ.get(k): raise ApplyRefused(f"apply refused in scheduler/CI/agent context ({k} is set)")
        rep, man, d = self._load_for_apply(run_id)
        approval = gate.load_approval(self.approvals_dir, rep["patch_sha256"])
        if approval is None: raise ApplyRefused("apply requires a human approval record matching this exact patch sha256 (`approve` at a terminal)")
        for f in man["files"]:                                           # drift check BEFORE touching anything
            if _sha_file(self.lab / f["path"]) != f["pre_sha256"]: raise ApplyRefused(f"lab file changed since the run was tested (drift): {f['path']}")
            if (self.lab / f["path"]).is_symlink(): raise ApplyRefused(f"lab path is a symlink: {f['path']}")
        bdir = self.out / "backups" / run_id
        if bdir.exists(): raise ApplyRefused("a backup for this run already exists (already applied?)")
        for f in man["files"]:
            if f["pre_sha256"] is not None:
                b = bdir / "files" / f["path"]; b.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(self.lab / f["path"], b)
                if _sha_file(b) != f["pre_sha256"]: raise ApplyRefused("backup verification failed; nothing was changed")
        (bdir / "manifest.json").parent.mkdir(parents=True, exist_ok=True); (bdir / "manifest.json").write_text(json.dumps(man, indent=1))
        out = {"run_id": run_id, "kind": "apply", "patch_sha256": rep["patch_sha256"], "started": time.time(), "approval": approval, "backup_dir": str(bdir),
               "files": [f["path"] for f in man["files"]], "status": None, "reasons": []}
        try:
            for f in man["files"]:
                dst = self.lab / f["path"]
                if f["kind"] == "delete": dst.unlink()
                else: _write_file(dst, (d / "files" / f["path"]).read_bytes())
            bad = [f["path"] for f in man["files"] if _sha_file(self.lab / f["path"]) != f["post_sha256"]]
            if bad: raise CodingError(f"post-apply hash mismatch: {bad}")
            out["status"] = "APPLIED"; out["reasons"].append("all files written and verified by sha256; reversible with `restore`")
        except Exception as e:                                           # noqa: BLE001
            out["status"] = "APPLY_ROLLED_BACK"; out["reasons"].append(f"apply failed mid-way: {type(e).__name__}: {str(e)[:200]}")
            out["rollback"] = self._restore_files(man, bdir)
        out["finished"] = time.time()
        return self._write_report(out, f"{run_id}.apply-{uuid.uuid4().hex[:6]}")

    def _restore_files(self, man: dict, bdir: Path) -> dict:
        errs = []
        for f in man["files"]:
            dst = self.lab / f["path"]
            try:
                tmp = dst.with_name(f".{dst.name}.coding_tmp")
                if tmp.exists(): tmp.unlink()
                if f["pre_sha256"] is None:
                    if dst.exists(): dst.unlink()
                elif _sha_file(dst) != f["pre_sha256"]: shutil.copy2(bdir / "files" / f["path"], dst)
            except OSError as e: errs.append(f"{f['path']}: {e}")
        verified = all(_sha_file(self.lab / f["path"]) == f["pre_sha256"] for f in man["files"])
        return {"performed": True, "restored_verified_by_sha256": verified, "errors": errs}

    def restore(self, run_id: str) -> dict:
        """Undo an applied patch from its backup. Refuses if the lab files are no longer exactly what `apply` wrote."""
        bdir = self.out / "backups" / run_id
        if not (bdir / "manifest.json").is_file(): raise ApplyRefused("no backup for this run")
        man = json.loads((bdir / "manifest.json").read_text())
        for f in man["files"]:
            if _sha_file(self.lab / f["path"]) != f["post_sha256"]: raise ApplyRefused(f"lab file is not what apply wrote (edited since): {f['path']}")
            if f["pre_sha256"] is not None and _sha_file(bdir / "files" / f["path"]) != f["pre_sha256"]: raise ApplyRefused(f"backup damaged: {f['path']}")
        out = {"run_id": run_id, "kind": "restore", "patch_sha256": man["patch_sha256"], "started": time.time(), "files": [f["path"] for f in man["files"]],
               "rollback": self._restore_files(man, bdir), "reasons": []}
        out["status"] = "RESTORED" if out["rollback"]["restored_verified_by_sha256"] else "APPLY_ROLLED_BACK"
        out["finished"] = time.time()
        shutil.move(str(bdir), str(bdir.with_name(f"{run_id}.restored-{uuid.uuid4().hex[:6]}")))
        return self._write_report(out, f"{run_id}.restore-{uuid.uuid4().hex[:6]}")

class _Fail(Exception): pass

def render_markdown(rep: dict) -> str:
    L = [f"# Coding workflow report `{rep['run_id']}`", "", f"**Status: {rep['status']}**", ""]
    ts = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "-"
    L += [f"- kind: {rep.get('kind', 'propose')}", f"- patch sha256: `{rep.get('patch_sha256')}`", f"- started: {ts(rep.get('started'))}  finished: {ts(rep.get('finished'))}"]
    if "source" in rep: L += [f"- source: {rep['source']}", f"- intent: {rep['intent']['summary'] or '(none given)'}", f"- declared golden changes: {rep['intent']['golden_changes'] or 'none'}",
                              f"- applied to lab: {rep['applied_to_lab']} (propose never applies)"]
    L += ["", "## Reasons"] + [f"- {r}" for r in rep.get("reasons", [])]
    if rep.get("files_touched"):
        L += ["", "## Files touched", "| path | kind | +lines | -lines |", "|---|---|---|---|"] + [f"| `{f['path']}` | {f['kind']} | {f['added']} | {f['removed']} |" for f in rep["files_touched"]]
    g = rep.get("gate")
    if g:
        L += ["", "## Approval gate", f"- decision: **{g['decision']}** (protected list sha256 `{g['protected_list_sha256'][:16]}..`, encoded in code)"]
        L += [f"- protected path: `{h['path']}` (rule `{h['rule']}`)" for h in g["protected_hits"]]
        L += [f"- keyword `{h['keyword']}` in added line of `{h['path']}`: `{h['line'][:80]}`" for h in g["content_hits"][:10]]
        if g.get("approval"): L.append(f"- approved by {g['approval']['approver']}: {g['approval']['reason']}")
    s = rep.get("sandbox")
    if s: L += ["", "## Sandbox", f"- profile sha256: `{s['profile_sha256']}`", f"- script sha256: `{s['script_sha256']}`", f"- network: {s['network']}; env: {s['env']}", f"- pytest args: `{' '.join(s['pytest_args'])}`"]
    t = rep.get("tests")
    if t:
        L += ["", "## Tests (before -> after)"]
        for k in ("before", "after"):
            if k in t: L.append(f"- {k}: {t[k]['counts']} total {t[k]['total']} in {t[k]['secs']}s; runner_error={t[k]['runner_error']}")
        if "regressions" in t:
            L += [f"- regressions: {len(t['regressions'])}; new tests: {len(t['new_tests'])}; removed tests: {len(t['removed_tests'])}; pre-existing failures (baseline, not counted): {len(t['pre_existing_failures'])}"]
            L += [f"  - `{r['test']}`: passed -> {r['after']}" for r in t["regressions"][:20]]
    g2 = rep.get("golden")
    if g2:
        L += ["", "## Golden diff", f"- probes: {g2['probes']} (ok before {g2['probes_ok_before']}, after {g2['probes_ok_after']}); tolerance rel {g2['rel_tol']} abs {g2['abs_tol']}",
              f"- fixtures: {g2['fixtures']}", f"- changed: {sorted(g2['changed']) or 'none'}; undeclared: {g2['undeclared_changed'] or 'none'}"]
        for n, d in g2["changed"].items(): L += [f"  - `{x['at']}`: {x['before']} -> {x['after']}" for x in d[:5]]
    rb = rep.get("rollback")
    if rb: L += ["", "## Rollback", f"- {rb}"]
    if rep.get("lab_untouched"): L += ["", "## Lab integrity", f"- touched paths unchanged in lab: {rep['lab_untouched']['unchanged']}"]
    if rep.get("workspace"): L += ["", "## Workspace", f"- {rep['workspace'].get('files')} files, {rep['workspace'].get('bytes')} bytes, baseline `{rep['workspace'].get('baseline_commit', '')[:12]}`, removed after run: {rep['workspace'].get('removed')}"]
    return "\n".join(L) + "\n"
