"""Coding-agent workflow (engine/coding). All fixtures are SYNTHETIC: a tiny throw-away "lab" in tmp_path, never the real one.
Sandbox tests use the real macOS sandbox-exec via scripts/coding_sandbox_run.sh; approvals use a real pty (the only way in)."""
import hashlib, inspect, json, os, pty, shutil, subprocess, sys
from pathlib import Path
import pytest
from engine.coding import gate, golden, workflow as wfmod
from engine.coding.patch import PatchRejected, check_against_tree, parse_patch, sha256_text
from engine.coding.workflow import ApplyRefused, CodingWorkflow

LAB = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists(), reason="needs macOS sandbox-exec")

# ---------------------------------------------------------------- synthetic lab + patches
MATHX = "def add(a, b):\n    return a + b\n\ndef double(a):\n    return add(a, a)\n"
TEST_MATHX = "from engine.mathx import add, double\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_double():\n    assert double(4) == 8\n"

@pytest.fixture
def lab(tmp_path, monkeypatch):
    for k in gate._SCHEDULER_ENV: monkeypatch.delenv(k, raising=False)
    root = tmp_path / "lab"
    (root / "engine" / "risk").mkdir(parents=True); (root / "engine" / "orchestration").mkdir(); (root / "tests").mkdir()
    (root / "engine" / "__init__.py").write_text(""); (root / "engine" / "risk" / "__init__.py").write_text(""); (root / "engine" / "orchestration" / "__init__.py").write_text("")
    (root / "engine" / "mathx.py").write_text(MATHX); (root / "tests" / "test_mathx.py").write_text(TEST_MATHX)
    (root / "engine" / "orchestration" / "sched.py").write_text("INTERVAL = 60\n")
    shutil.copy(LAB / "engine" / "risk" / "metrics.py", root / "engine" / "risk" / "metrics.py")     # one real golden-probe target
    return root

@pytest.fixture
def wf(lab, tmp_path): return CodingWorkflow(lab_root=lab, out_root=tmp_path / "out", test_timeout=300)

def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file(): h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def modify(path: str, old_lines: list[str], new_lines: list[str], start: int = 1) -> str:
    body = "".join(f"-{l}\n" for l in old_lines) + "".join(f"+{l}\n" for l in new_lines)
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -{start},{len(old_lines)} +{start},{len(new_lines)} @@\n{body}"

def edit(path: str, old: str, new: str) -> str:
    import difflib
    assert old != new
    return f"diff --git a/{path} b/{path}\n" + "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))

def newfile(path: str, lines: list[str], mode: str = "100644") -> str:
    return f"diff --git a/{path} b/{path}\nnew file mode {mode}\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{l}\n" for l in lines)

HARMLESS = newfile("engine/util_fmt.py", ['"""Formatting helper."""', "def pct(x):", '    return f"{x:.1%}"']) + \
           newfile("tests/test_util_fmt.py", ["from engine.util_fmt import pct", "", "def test_pct():", '    assert pct(0.125) == "12.5%"'])
BREAKING = edit("engine/mathx.py", MATHX, MATHX.replace("a + b", "a - b"))
GOLDEN_SHIFT = None   # built per-test from the copied metrics.py

def _pty_approve(approvals: Path, sha: str, approver="Test Human", reason="read the diff line by line", typed=None, extra_env=None):
    """Drive gate.approve through a real pseudo-terminal: the same path a person uses. Returns (returncode, stderr)."""
    code = ("import sys\nfrom pathlib import Path\nfrom engine.coding import gate\n"
            "try: gate.approve(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])\n"
            "except gate.ApprovalRefused as e: sys.stderr.write(str(e)); sys.exit(7)\n")
    m, s = pty.openpty()
    env = {"PATH": "/usr/bin:/bin", **(extra_env or {})}
    p = subprocess.Popen([sys.executable, "-c", code, str(approvals), sha, approver, reason], stdin=s, stdout=s, stderr=subprocess.PIPE, cwd=LAB, env=env)
    os.close(s); os.write(m, ((typed if typed is not None else sha[:12]) + "\n").encode())
    try: _, err = p.communicate(timeout=60)
    finally: os.close(m)
    return p.returncode, err.decode()

# ---------------------------------------------------------------- gate: protected list is code, pinned, not configurable
REQUIRED_PROTECTED = ["engine/memory/store.py", "engine/research/validation.py", "engine/agents/roles.py", "adapters/registry.py", "adapters/workers/x.py",
                      "scripts/sandbox_run.sh", "scripts/coding_sandbox_run.sh", "engine/backtesting/paper.py", "engine/orchestration/jobs.py",
                      "engine/forward/log.py", "engine/coding/gate.py", "engine/coding/sub/x.py", "configs/schedule.toml", "pyproject.toml",
                      "engine/api/app.py", "tests/test_coding_workflow.py", "tests/conftest.py", "engine/agents/experiment.py", "engine/agents/alpha.py",
                      "engine/models/ml.py", "engine/families/x.py", "engine/derivs/iv.py", "engine/data/pit.py", "engine/backtesting/vector.py",
                      "engine/research/pooled.py", "engine/futures/roll.py", "engine/eventsim/x.py", "engine/strategies/trend.py", "engine/memory/x.py"]

def test_protected_list_is_frozen_and_hash_pinned():
    assert isinstance(gate.PROTECTED_PATHS, tuple) and isinstance(gate.CONTENT_KEYWORDS, tuple)
    assert gate._digest(gate.PROTECTED_PATHS, gate.CONTENT_KEYWORDS) == gate.PROTECTED_SHA256
    assert gate.PROTECTED_SHA256 == "e4b2faa0789703dd9f70e167c99e43a6b62c5fe3496a07a7f0f8e807fe15b875"   # second pin: changing the list must break a test

@pytest.mark.parametrize("path", REQUIRED_PROTECTED)
def test_required_paths_are_protected(path, tmp_path):
    assert gate.protected_hits([path]), path
    assert gate.evaluate([path], {path: ["x = 1"]}, "0" * 64, tmp_path).decision == "BLOCKED_NEEDS_APPROVAL"

def test_unprotected_path_passes_gate_and_case_variants_do_not_slip_through(tmp_path):
    assert gate.evaluate(["engine/risk/metrics.py"], {"engine/risk/metrics.py": ["x = 1"]}, "0" * 64, tmp_path).decision == "NOT_PROTECTED"
    assert gate.protected_hits(["Engine/Coding/x.py"]) and gate.protected_hits(["ADAPTERS/registry.py"])
    with pytest.raises(PatchRejected): parse_patch(newfile("Engine/Coding/x.py", ["x = 1"]))

def test_gate_has_no_config_env_or_flag_bypass(tmp_path, monkeypatch):
    assert list(inspect.signature(gate.evaluate).parameters) == ["paths", "added_lines", "patch_sha256", "approvals_dir"]
    assert list(inspect.signature(gate.interactive_context).parameters) == []
    for k in ("QAE_CODING_GATE", "QAE_SKIP_GATE", "QAE_APPROVE", "CODING_AUTO_APPROVE", "QAE_HOME"): monkeypatch.setenv(k, "off")
    assert gate.evaluate(["engine/memory/store.py"], {}, "0" * 64, tmp_path).decision == "BLOCKED_NEEDS_APPROVAL"
    src = "".join(Path(p).read_text() for p in (gate.__file__, wfmod.__file__))
    assert "tomllib" not in src and "configs/" not in src.replace('"configs/**"', "")        # the gate reads no config file
    from engine.coding.__main__ import main
    for flag in ("--force", "--no-gate", "--yes", "--approve"):
        with pytest.raises(SystemExit): main(["propose", "--patch", "x.diff", flag])

@pytest.mark.parametrize("line", ["from x import Broker", "def send_order(q): pass", "CREDENTIAL = 1", "os.system('ls')", "import subprocess", "exec(code)",
                                  "api_key = None", "import socket", "live_trading = True", "account_id = 3"])
def test_content_scan_forces_gate_on_added_lines(line, tmp_path):
    d = gate.evaluate(["engine/risk/new.py"], {"engine/risk/new.py": [line]}, "0" * 64, tmp_path)
    assert d.decision == "BLOCKED_NEEDS_APPROVAL" and d.content and not d.protected

def test_content_scan_ignores_removed_lines_and_clean_code():
    p = parse_patch(modify("engine/risk/x.py", ["import subprocess"], ["import math"]))
    assert gate.content_hits(p.added_lines) == []

# ---------------------------------------------------------------- approval records
def test_approve_refuses_without_a_terminal(tmp_path, monkeypatch):
    assert not os.isatty(0) or not os.isatty(1) or pytest.skip("pytest is attached to a tty with -s")
    for k in gate._SCHEDULER_ENV: monkeypatch.delenv(k, raising=False)       # isolate the tty check from the env-marker check
    with pytest.raises(gate.ApprovalRefused, match="not a terminal"): gate.approve(tmp_path, "a" * 64, "Test Human", "looked at the diff")
    assert list(tmp_path.iterdir()) == []

def test_approve_via_real_pty_creates_record_bound_to_exact_hash(tmp_path):
    sha, other = "a" * 64, "b" * 64
    rc, err = _pty_approve(tmp_path, sha); assert rc == 0, err
    rec = gate.load_approval(tmp_path, sha)
    assert rec and rec["approver"] == "Test Human" and rec["patch_sha256"] == sha
    assert gate.load_approval(tmp_path, other) is None
    shutil.copy(tmp_path / f"{sha}.json", tmp_path / f"{other}.json")                      # replaying a record under another patch's name
    assert gate.load_approval(tmp_path, other) is None
    rec["reason"] = "edited after the fact"; (tmp_path / f"{sha}.json").write_text(json.dumps(rec))
    assert gate.load_approval(tmp_path, sha) is None                                      # edited record no longer matches its own hash

def test_approve_refused_on_wrong_confirmation_scheduler_env_or_agent_identity(tmp_path):
    sha = "c" * 64
    rc, err = _pty_approve(tmp_path, sha, typed="yes"); assert rc == 7 and "did not match" in err
    for k in ("QAE_SCHEDULER", "CI"):
        rc, err = _pty_approve(tmp_path, sha, extra_env={k: "1"}); assert rc == 7 and "scheduler" in err
    rc, err = _pty_approve(tmp_path, sha, approver="agent"); assert rc == 7 and "human" in err
    rc, err = _pty_approve(tmp_path, sha, reason="ok"); assert rc == 7
    assert list(tmp_path.iterdir()) == [] and gate.load_approval(tmp_path, sha) is None

def test_malformed_or_symlinked_approval_is_not_an_approval(tmp_path):
    sha = "d" * 64
    (tmp_path / f"{sha}.json").write_text(json.dumps({"approver": "Test Human", "reason": "long enough reason", "patch_sha256": sha}))
    assert gate.load_approval(tmp_path, sha) is None
    (tmp_path / f"{sha}.json").write_text("not json"); assert gate.load_approval(tmp_path, sha) is None
    rc, _ = _pty_approve(tmp_path / "real", "e" * 64); assert rc == 0
    os.symlink(tmp_path / "real" / f"{'e' * 64}.json", tmp_path / f"{'e' * 64}.json")
    assert gate.load_approval(tmp_path, "e" * 64) is None

# ---------------------------------------------------------------- patch validation
BAD_PATCHES = {
    "absolute": newfile("engine/x.py", ["x=1"]).replace("b/engine/x.py", "/etc/x.py"),
    "abs_in_git_header": "diff --git a//tmp/x.py b//tmp/x.py\n--- /dev/null\n+++ b//tmp/x.py\n@@ -0,0 +1,1 @@\n+x\n",
    "dotdot": newfile("engine/../../x.py", ["x=1"]),
    "git_dir": newfile(".git/hooks/pre-commit", ["x=1"]),
    "git_dir_nested": newfile("engine/.git/config", ["x=1"]),
    "gitattributes": newfile("engine/.gitattributes", ["* filter=x"]),
    "exec_bit": newfile("engine/x.py", ["x=1"], mode="100755"),
    "symlink_mode": newfile("engine/x.py", ["/etc/passwd"], mode="120000"),
    "mode_change": "diff --git a/engine/mathx.py b/engine/mathx.py\nold mode 100644\nnew mode 100755\n",
    "rename": "diff --git a/engine/mathx.py b/engine/m2.py\nsimilarity index 100%\nrename from engine/mathx.py\nrename to engine/m2.py\n",
    "binary": "diff --git a/engine/x.bin b/engine/x.bin\nnew file mode 100644\nGIT binary patch\nliteral 3\nKcmZQzU|?VY0RR910RR91\n\n",
    "binary_files": "diff --git a/engine/x.bin b/engine/x.bin\nBinary files /dev/null and b/engine/x.bin differ\n",
    "outside_roots": newfile("docs/x.md", ["x"]),
    "top_level_file": newfile("sitecustomize.py", ["x=1"]),
    "envs": newfile("envs/core/lib/x.py", ["x=1"]),
    "backslash": newfile("engine\\x.py", ["x=1"]),
    "quoted": 'diff --git "a/engine/x y.py" "b/engine/x y.py"\n--- /dev/null\n+++ "b/engine/x y.py"\n@@ -0,0 +1,1 @@\n+x\n',
    "junk_between": newfile("engine/x.py", ["x=1"]) + "rm -rf /\n",
    "hunk_overrun": "--- a/engine/mathx.py\n+++ b/engine/mathx.py\n@@ -1,1 +1,1 @@\n-a\n+b\n+c\n+++ b/engine/coding/gate.py\n",
    "scratch_dir": newfile("engine/.coding_out/x.py", ["x=1"]),
    "empty": "   \n",
    "nul": newfile("engine/x.py", ["x=\x001"]),
}
@pytest.mark.parametrize("name", sorted(BAD_PATCHES))
def test_invalid_patches_are_rejected(name):
    with pytest.raises(PatchRejected): parse_patch(BAD_PATCHES[name])

def test_valid_patch_parses_and_hidden_header_inside_hunk_is_content_not_a_file():
    p = parse_patch(HARMLESS); assert p.paths == ["engine/util_fmt.py", "tests/test_util_fmt.py"] and p.sha256 == sha256_text(HARMLESS)
    tricky = "--- a/engine/mathx.py\n+++ b/engine/mathx.py\n@@ -1,2 +1,2 @@\n--- a/engine/coding/gate.py\n+++ b/engine/coding/gate.py\n context\n"
    q = parse_patch(tricky); assert q.paths == ["engine/mathx.py"] and q.files[0].added == ["++ b/engine/coding/gate.py"]

def test_patch_through_symlink_or_onto_missing_file_is_rejected(lab, tmp_path):
    outside = tmp_path / "outside"; outside.mkdir(); os.symlink(outside, lab / "engine" / "link")
    with pytest.raises(PatchRejected, match="symlink"): check_against_tree(parse_patch(newfile("engine/link/x.py", ["x=1"])), lab)
    with pytest.raises(PatchRejected, match="does not exist"): check_against_tree(parse_patch(modify("engine/nope.py", ["a"], ["b"])), lab)
    with pytest.raises(PatchRejected, match="already exists"): check_against_tree(parse_patch(newfile("engine/mathx.py", ["x=1"])), lab)
    with pytest.raises(PatchRejected, match="case"): check_against_tree(parse_patch(newfile("engine/MATHX.py", ["x=1"])), lab)

def test_workspace_copy_skips_symlinks(lab, wf, tmp_path):
    secret = tmp_path / "secret.txt"; secret.write_text("s3cr3t"); os.symlink(secret, lab / "engine" / "leak.txt"); os.symlink(tmp_path, lab / "engine" / "updir")
    ws, info, _ = wf._make_workspace("t1")
    assert not (ws / "engine" / "leak.txt").exists() and not (ws / "engine" / "updir").exists() and info["symlinks_skipped"] == 2
    assert (ws / "engine" / "mathx.py").read_text() == MATHX and (ws / ".git").is_dir()

# ---------------------------------------------------------------- the sandbox itself
PROBE = r'''
import json, os, socket, sys
out = {}
def t(name, f):
    try: f(); out[name] = "ALLOWED"
    except Exception as e: out[name] = "DENIED:" + type(e).__name__
t("tcp_out", lambda: socket.create_connection(("1.1.1.1", 53), timeout=5).close())
t("udp_out", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53)))
t("dns", lambda: socket.getaddrinfo("example.com", 80))
def loop():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1); socket.create_connection(s.getsockname(), timeout=5).close(); s.close()
t("loopback", loop)
home = sys.argv[2]
t("read_ssh", lambda: os.listdir(home + "/.ssh"))
t("read_home", lambda: os.listdir(home))
t("read_claude", lambda: os.listdir(home + "/.claude"))
t("read_market_data", lambda: os.listdir("/Users/newman/quantalpha/data"))
t("read_lab_source", lambda: open(sys.argv[3] + "/engine/memory/store.py").read())
t("read_prod_store", lambda: os.listdir(sys.argv[3] + "/artifacts"))
t("write_outside", lambda: open(sys.argv[4], "w").write("x"))
t("write_git", lambda: open(sys.argv[1] + "/.git/hooks/pre-commit", "w").write("x"))
t("write_ws", lambda: open(sys.argv[1] + "/ok.txt", "w").write("x"))
def no_exec(cmd):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, timeout=20)
    if r.returncode != 0: raise PermissionError("nonzero")
t("exec_sh", lambda: no_exec(["/bin/sh", "-c", "true"]))
t("exec_open", lambda: no_exec(["/usr/bin/open", "-g", "-j", sys.argv[1] + "/bundle"]))
t("exec_osascript", lambda: no_exec(["/usr/bin/osascript", "-e", "1"]))
t("exec_curl", lambda: no_exec(["/usr/bin/curl", "--version"]))
t("exec_python", lambda: no_exec([sys.executable, "-I", "-c", "pass"]))
def app():
    os.makedirs(sys.argv[1] + "/esc.app/Contents/MacOS")
t("write_app_bundle", app)
def mach():
    import ctypes
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib"); port = ctypes.c_uint()
    rc = lib.bootstrap_look_up(ctypes.c_uint.in_dll(lib, "bootstrap_port"), sys.argv[6].encode(), ctypes.byref(port))
    if rc != 0: raise PermissionError(str(rc))
t("mach_launchservices", mach)
t("signal_host", lambda: os.kill(int(sys.argv[5]), 0))
t("read_private_tmp", lambda: os.listdir("/private/tmp"))
t("read_var_folders", lambda: os.listdir(os.path.dirname(os.path.dirname(sys.argv[4]))))
out["env"] = sorted(os.environ)
print(json.dumps(out))
'''
MACH_SERVICE = "com.apple.coreservices.launchservicesd"
def _exec_arg(): return ":".join(wfmod._exec_allowlist())
def test_sandbox_blocks_outbound_network_secrets_and_writes_outside_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BROKER_TOKEN", "synthetic-not-a-secret"); monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic")
    ws = tmp_path / "ws"; (ws / ".git" / "hooks").mkdir(parents=True); (ws / "bundle").mkdir(); (ws / "probe.py").write_text(PROBE); outside = tmp_path / "outside.txt"
    home = os.path.expanduser("~")
    r = subprocess.run([str(wfmod.SANDBOX_SCRIPT), str(ws), str(tmp_path / "p.sb"), str(wfmod.PYENV), _exec_arg(), "--",
                        str(wfmod.PYTHON), "-I", str(ws / "probe.py"), str(ws.resolve()), home, str(LAB), str(outside), str(os.getpid()), MACH_SERVICE], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for k in ("tcp_out", "udp_out", "dns", "read_ssh", "read_home", "read_claude", "read_market_data", "read_lab_source", "read_prod_store", "write_outside", "write_git",
              "exec_sh", "exec_open", "exec_osascript", "exec_curl", "write_app_bundle", "mach_launchservices", "signal_host", "read_private_tmp", "read_var_folders"):
        assert out[k].startswith("DENIED"), (k, out[k])
    assert out["loopback"] == "ALLOWED" and out["write_ws"] == "ALLOWED" and out["exec_python"] == "ALLOWED"
    assert not outside.exists() and not (ws / ".git" / "hooks" / "pre-commit").exists() and not (ws / "esc.app").exists()
    assert "FAKE_BROKER_TOKEN" not in out["env"] and "AWS_SECRET_ACCESS_KEY" not in out["env"]
    assert set(out["env"]) <= {"PATH", "HOME", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "MPLBACKEND", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
    prof = (tmp_path / "p.sb").read_text()
    assert "(deny default)" in prof and "(allow default)" not in prof and "/Users/newman/quantalpha/data" in prof

def test_mach_probe_is_a_real_control_outside_the_sandbox():
    """The in-sandbox DENIED for LaunchServices means something only if the same lookup succeeds unsandboxed."""
    import ctypes
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib"); port = ctypes.c_uint()
    assert lib.bootstrap_look_up(ctypes.c_uint.in_dll(lib, "bootstrap_port"), MACH_SERVICE.encode(), ctypes.byref(port)) == 0

ESCAPE = r'''
import os, subprocess, sys, time
w, marker = sys.argv[1], sys.argv[2]
script = "#!/bin/sh\necho escaped > " + marker + "\n"
plist = '<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleExecutable</key><string>esc</string><key>CFBundleIdentifier</key><string>x.esc</string></dict></plist>'
for name in ("esc.app", "esc.bundle", "esc"):
    app = os.path.join(w, name)
    try:
        os.makedirs(app + "/Contents/MacOS", exist_ok=True); open(app + "/Contents/Info.plist", "w").write(plist)
        open(app + "/Contents/MacOS/esc", "w").write(script); os.chmod(app + "/Contents/MacOS/esc", 0o755)
    except OSError as e: print("write", name, "blocked", e)
    for cmd in (["/usr/bin/open", "-g", "-j", app], ["/usr/bin/open", "-a", app], ["/bin/sh", app + "/Contents/MacOS/esc"], [app + "/Contents/MacOS/esc"],
                ["/usr/bin/open", "http://127.0.0.1:9/"], ["/usr/bin/osascript", "-e", 'do shell script "echo escaped > ' + marker + '"']):
        try: r = subprocess.run(cmd, capture_output=True, timeout=20); print(cmd[0], r.returncode)
        except OSError as e: print(cmd[0], "blocked", e)
time.sleep(4)
'''
def test_launchservices_app_bundle_escape_is_blocked(tmp_path):
    """Regression for the reviewer's escape: write an .app in the workspace and `open` it so LaunchServices runs it unsandboxed."""
    ws = tmp_path / "ws"; ws.mkdir(); (ws / "escape.py").write_text(ESCAPE); marker = tmp_path / "ESCAPED"
    r = subprocess.run([str(wfmod.SANDBOX_SCRIPT), str(ws), str(tmp_path / "p.sb"), str(wfmod.PYENV), _exec_arg(), "--",
                        str(wfmod.PYTHON), "-I", str(ws / "escape.py"), str(ws.resolve()), str(marker)], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert not marker.exists(), "sandboxed code got something to run outside the sandbox"
    assert not (ws / "esc.app").exists() and r.stdout.count("blocked") >= 15, r.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.sb", "ws"]

def test_sandbox_script_refuses_profile_inside_writable_dir_non_env_args_and_bad_exec_lists(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir(); ex = _exec_arg()
    r = subprocess.run([str(wfmod.SANDBOX_SCRIPT), str(ws), str(ws.resolve() / "p.sb"), "", ex, "--", "/usr/bin/true"], capture_output=True, text=True)
    assert r.returncode == 2 and "outside" in r.stderr
    r = subprocess.run([str(wfmod.SANDBOX_SCRIPT), str(ws), str(tmp_path / "p.sb"), "", ex, "/usr/bin/true"], capture_output=True, text=True)
    assert r.returncode == 2
    r = subprocess.run([str(wfmod.SANDBOX_SCRIPT), str(ws), str(tmp_path / "p.sb"), "", "", "--", "/usr/bin/true"], capture_output=True, text=True)
    assert r.returncode == 2 and "allowlist is empty" in r.stderr
    r = subprocess.run([str(wfmod.SANDBOX_SCRIPT), str(ws), str(tmp_path / "p.sb"), "", str(ws.resolve() / "evil"), "--", "/usr/bin/true"], capture_output=True, text=True)
    assert r.returncode == 2 and "inside the writable dir" in r.stderr                       # sandboxed code must never be able to write its own executable

# ---------------------------------------------------------------- end-to-end on the synthetic lab
def test_harmless_patch_is_ready_for_review_and_lab_is_untouched(lab, wf):
    before = tree_hash(lab)
    rep = wf.propose(HARMLESS, {"summary": "add pct helper"})
    assert rep["status"] == "READY_FOR_HUMAN_REVIEW", rep["reasons"]
    assert rep["applied_to_lab"] is False and tree_hash(lab) == before and not (lab / "engine" / "util_fmt.py").exists()
    assert rep["gate"]["decision"] == "NOT_PROTECTED" and rep["tests"]["regressions"] == [] and rep["golden"]["ok"]
    assert rep["tests"]["before"]["counts"] == {"passed": 2} and rep["tests"]["after"]["counts"] == {"passed": 3}
    assert rep["tests"]["after"]["engine_file"].startswith(rep["workspace"]["path"])       # workspace code ran, not the installed lab
    assert rep["golden"]["probes_ok_before"] >= 1                                          # the copied risk.metrics probe really executed
    assert rep["workspace"]["removed"] and not Path(rep["workspace"]["path"]).exists()
    assert len(rep["sandbox"]["profile_sha256"]) == 64 and "(deny default)" in Path(rep["sandbox"]["profile_path"]).read_text()
    j = json.loads(Path(rep["report_paths"]["json"]).read_text()); md = Path(rep["report_paths"]["md"]).read_text()
    assert j["patch_sha256"] == sha256_text(HARMLESS) and "READY_FOR_HUMAN_REVIEW" in md and j["patch_sha256"] in md
    assert wf.verify_index() == {"ok": True, "entries": 1, "problems": [], "head": rep["index_entry"]["entry_hash"]}

def test_breaking_patch_is_rolled_back(lab, wf):
    before = tree_hash(lab)
    rep = wf.propose(BREAKING, {"summary": "break add"})
    assert rep["status"] == "ROLLED_BACK"
    assert {r["test"] for r in rep["tests"]["regressions"]} == {"tests/test_mathx.py::test_add", "tests/test_mathx.py::test_double"}
    assert rep["rollback"]["performed"] and rep["rollback"]["workspace_reset_verified"] and rep["rollback"]["staged_discarded"]
    assert tree_hash(lab) == before and rep["lab_untouched"]["unchanged"] and not (wf.out / "staged" / rep["run_id"]).exists()
    assert not Path(rep["workspace"]["path"]).exists()
    with pytest.raises(ApplyRefused, match="ROLLED_BACK"): wf.apply(rep["run_id"])

def test_deleting_a_passing_test_counts_as_regression(lab, wf):
    lines = TEST_MATHX.splitlines()
    p = f"diff --git a/tests/test_mathx.py b/tests/test_mathx.py\ndeleted file mode 100644\n--- a/tests/test_mathx.py\n+++ /dev/null\n@@ -1,{len(lines)} +0,0 @@\n" + "".join(f"-{l}\n" for l in lines)
    rep = wf.propose(p, {"summary": "remove tests"})
    assert rep["status"] == "ROLLED_BACK" and len(rep["tests"]["regressions"]) == 2 and rep["tests"]["regressions"][0]["after"] == "<missing>"

def test_new_failing_test_in_patch_fails_the_run(lab, wf):
    rep = wf.propose(newfile("tests/test_new_bad.py", ["def test_bad():", "    assert 1 == 2"]), {})
    assert rep["status"] == "ROLLED_BACK" and "new test(s) added by the patch fail" in " ".join(rep["reasons"])

def _golden_shift(lab):
    src = (lab / "engine" / "risk" / "metrics.py").read_text(); key = '"max_drawdown": float(drawdown_series(returns).min()),'
    assert key in src
    return edit("engine/risk/metrics.py", src, src.replace(key, key.replace(".min()),", ".min()) * 1.0001,")))

def test_undeclared_golden_change_is_rolled_back_even_though_all_tests_pass(lab, wf):
    rep = wf.propose(_golden_shift(lab), {"summary": "tiny silent numeric change"})
    assert rep["status"] == "ROLLED_BACK" and rep["tests"]["regressions"] == []
    assert rep["golden"]["undeclared_changed"] == ["risk.metrics"] and "golden outputs changed" in " ".join(rep["reasons"])

def test_declared_golden_change_is_allowed_but_reported_and_unknown_probe_names_rejected(lab, wf):
    rep = wf.propose(_golden_shift(lab), {"summary": "intended change", "golden_changes": ["risk.metrics"]})
    assert rep["status"] == "READY_FOR_HUMAN_REVIEW" and rep["golden"]["declared_and_changed"] == ["risk.metrics"] and rep["golden"]["changed"]
    bad = wf.propose(_golden_shift(lab), {"golden_changes": ["everything"]})
    assert bad["status"] == "REJECTED_INVALID_PATCH"

def test_golden_compare_tolerance_and_structure():
    a = {"p": {"ok": True, "value": {"x": 1.0, "y": [1, 2]}}}
    assert golden.compare(a, {"p": {"ok": True, "value": {"x": 1.0 + 1e-13, "y": [1, 2]}}})["ok"]
    assert not golden.compare(a, {"p": {"ok": True, "value": {"x": 1.0 + 1e-6, "y": [1, 2]}}})["ok"]
    assert not golden.compare(a, {"p": {"ok": True, "value": {"x": 1.0, "y": [1, 2, 3]}}})["ok"]
    assert not golden.compare(a, {"p": {"ok": False, "error": "boom"}})["ok"] and not golden.compare(a, {})["ok"]
    assert golden.compare(a, {"p": {"ok": False, "error": "boom"}}, declared=["p"])["ok"]

def test_protected_patch_is_blocked_before_any_workspace_or_sandbox(lab, wf):
    p = modify("engine/orchestration/sched.py", ["INTERVAL = 60"], ["INTERVAL = 1"])
    rep = wf.propose(p, {"summary": "speed up scheduler"})
    assert rep["status"] == "BLOCKED_NEEDS_APPROVAL" and rep["gate"]["decision"] == "BLOCKED_NEEDS_APPROVAL"
    assert rep["gate"]["protected_hits"] == [{"path": "engine/orchestration/sched.py", "rule": "engine/orchestration/**"}]
    assert rep["workspace"] is None and rep["sandbox"] is None and rep["tests"] is None and list((wf.out / "ws").iterdir()) == []
    kw = wf.propose(newfile("engine/newmod.py", ["def place_order(q):", "    return q"]), {})
    assert kw["status"] == "BLOCKED_NEEDS_APPROVAL" and kw["gate"]["content_hits"][0]["keyword"] == "order"

def test_invalid_patch_is_rejected_with_report(lab, wf):
    rep = wf.propose(BAD_PATCHES["git_dir"], {})
    assert rep["status"] == "REJECTED_INVALID_PATCH" and rep["workspace"] is None and Path(rep["report_paths"]["json"]).exists()

def test_approved_protected_patch_still_runs_in_the_same_sandbox(lab, wf):
    p = modify("engine/orchestration/sched.py", ["INTERVAL = 60"], ["INTERVAL = 61"])
    assert wf.propose(p, {})["status"] == "BLOCKED_NEEDS_APPROVAL"
    rc, err = _pty_approve(wf.approvals_dir, sha256_text(p)); assert rc == 0, err
    rep = wf.propose(p, {})
    assert rep["status"] == "READY_FOR_HUMAN_REVIEW" and rep["gate"]["decision"] == "APPROVED_BY_HUMAN" and rep["gate"]["approval"]["approver"] == "Test Human"
    prof = Path(rep["sandbox"]["profile_path"]).read_text()
    assert "(deny default)" in prof and "(allow default)" not in prof and '(deny file-read* (subpath "/Users")' in prof and rep["applied_to_lab"] is False
    other = modify("engine/orchestration/sched.py", ["INTERVAL = 60"], ["INTERVAL = 62"])          # approval does not carry over to a different patch
    assert wf.propose(other, {})["status"] == "BLOCKED_NEEDS_APPROVAL"

def test_patched_code_cannot_write_to_the_lab_or_read_the_home_dir(lab, wf, tmp_path):
    target = lab / "PWNED.txt"; home = os.path.expanduser("~")
    t = newfile("tests/test_escape.py", ["import os, pytest", "def test_cannot_write_lab():",
            f"    with pytest.raises(PermissionError): open({str(target)!r}, 'w').write('x')", "def test_cannot_list_home():",
            f"    with pytest.raises(PermissionError): os.listdir({home!r})"])
    (lab / "tests" / "test_env.py").write_text("import os\ndef test_qae_home_is_inside_workspace():\n"
        "    assert '.coding_out' in os.environ['QAE_HOME'] and os.environ['QAE_HOME'].startswith(os.path.realpath(os.getcwd()))\n"
        "    assert not [k for k in os.environ if 'TOKEN' in k or 'SECRET' in k or 'KEY' in k]\n")
    assert gate.content_hits(parse_patch(t).added_lines) == []
    os.environ["FAKE_API_TOKEN"] = "synthetic-not-a-secret"
    try: rep = wf.propose(t, {})
    finally: del os.environ["FAKE_API_TOKEN"]
    assert rep["status"] == "READY_FOR_HUMAN_REVIEW", rep["reasons"]
    assert rep["tests"]["after"]["counts"] == {"passed": 5} and not target.exists()

def test_result_forgery_surface_is_limited_baseline_results_are_read_before_patch_runs(lab, wf):
    """Patched test code may write anything inside the workspace; the baseline pass set must already be out of its reach."""
    t = newfile("tests/test_forge.py", ["import json, pathlib", "def test_forge():",
            "    d = pathlib.Path('.coding_out'); d.mkdir(exist_ok=True)", "    (d / 'before.json').write_text(json.dumps({'tests': {}, 'golden': {}}))"])
    rep = wf.propose(BREAKING + t, {})
    assert rep["status"] == "ROLLED_BACK" and len(rep["tests"]["regressions"]) == 2

# ---------------------------------------------------------------- audit index
def test_index_is_hash_chained_and_detects_edit_delete_reorder_and_report_tamper(lab, wf):
    r1 = wf.propose(BAD_PATCHES["dotdot"], {}); r2 = wf.propose(BAD_PATCHES["binary"], {}); r3 = wf.propose(BAD_PATCHES["rename"], {})
    assert wf.verify_index()["ok"] and wf.verify_index()["entries"] == 3
    orig = wf.index_path.read_text(); lines = orig.splitlines()
    wf.index_path.write_text(orig.replace("REJECTED_INVALID_PATCH", "READY_FOR_HUMAN_REVIEW", 1)); assert not wf.verify_index()["ok"]
    wf.index_path.write_text("\n".join([lines[0], lines[2]]) + "\n"); assert not wf.verify_index()["ok"]
    wf.index_path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n"); assert not wf.verify_index()["ok"]
    wf.index_path.write_text("\n".join(lines[:2]) + "\n"); assert wf.verify_index()["ok"]                # truncation of the tail is NOT detectable by a chain alone (documented)
    forged = {k: v for k, v in json.loads(lines[1]).items() if k != "entry_hash"}; forged["status"] = "READY_FOR_HUMAN_REVIEW"   # self-consistent rewrite of one entry
    forged["entry_hash"] = hashlib.sha256(json.dumps(forged, sort_keys=True).encode()).hexdigest()
    wf.index_path.write_text("\n".join([lines[0], json.dumps(forged, sort_keys=True), lines[2]]) + "\n")
    v = wf.verify_index(); assert not v["ok"] and any("prev_hash" in x for x in v["problems"])        # only the chain to the NEXT entry exposes it
    wf.index_path.write_text(orig); assert wf.verify_index()["ok"]
    rp = Path(r2["report_paths"]["json"]); rp.write_text(rp.read_text().replace("REJECTED_INVALID_PATCH", "READY_FOR_HUMAN_REVIEW"))
    v = wf.verify_index(); assert not v["ok"] and "modified" in v["problems"][0]
    assert r1["index_entry"]["seq"] == 0 and r3["index_entry"]["prev_hash"] == r2["index_entry"]["entry_hash"]

# ---------------------------------------------------------------- apply / restore (the only path that writes to the lab)
TWO_FILE = edit("engine/mathx.py", MATHX, MATHX.replace("a + b", "b + a")) + \
           newfile("engine/util_fmt.py", ['"""Formatting helper."""', "def pct(x):", '    return f"{x:.1%}"'])

def test_apply_requires_matching_human_approval_then_applies_and_restore_reverses(lab, wf):
    before = tree_hash(lab); rep = wf.propose(TWO_FILE, {"summary": "commute + helper"})
    assert rep["status"] == "READY_FOR_HUMAN_REVIEW"
    with pytest.raises(ApplyRefused, match="approval"): wf.apply(rep["run_id"])
    assert tree_hash(lab) == before
    rc, err = _pty_approve(wf.approvals_dir, "f" * 64); assert rc == 0                              # an approval for some OTHER patch
    with pytest.raises(ApplyRefused, match="approval"): wf.apply(rep["run_id"])
    rc, err = _pty_approve(wf.approvals_dir, rep["patch_sha256"]); assert rc == 0, err
    ap = wf.apply(rep["run_id"])
    assert ap["status"] == "APPLIED" and "return b + a" in (lab / "engine" / "mathx.py").read_text() and (lab / "engine" / "util_fmt.py").exists()
    assert tree_hash(lab) != before
    with pytest.raises(ApplyRefused): wf.apply(rep["run_id"])                                       # not twice
    rs = wf.restore(rep["run_id"])
    assert rs["status"] == "RESTORED" and rs["rollback"]["restored_verified_by_sha256"] and tree_hash(lab) == before
    assert wf.verify_index()["ok"] and [e["status"] for e in wf.status()] == ["READY_FOR_HUMAN_REVIEW", "APPLIED", "RESTORED"]

def test_apply_failure_midway_restores_lab_from_backup_verified_by_hash(lab, wf, monkeypatch):
    before = tree_hash(lab); rep = wf.propose(TWO_FILE, {})
    rc, err = _pty_approve(wf.approvals_dir, rep["patch_sha256"]); assert rc == 0, err
    real = wfmod._write_file; calls = []
    def flaky(dst, data):
        calls.append(dst)
        if len(calls) == 2: raise OSError("synthetic disk failure on second file")
        real(dst, data)
    monkeypatch.setattr(wfmod, "_write_file", flaky)
    ap = wf.apply(rep["run_id"])
    assert len(calls) == 2 and ap["status"] == "APPLY_ROLLED_BACK" and ap["rollback"]["restored_verified_by_sha256"] is True
    assert tree_hash(lab) == before and (lab / "engine" / "mathx.py").read_text() == MATHX and not (lab / "engine" / "util_fmt.py").exists()

def test_apply_refuses_on_lab_drift_tampered_staging_scheduler_context_and_bad_index(lab, wf, monkeypatch):
    rep = wf.propose(TWO_FILE, {}); rc, err = _pty_approve(wf.approvals_dir, rep["patch_sha256"]); assert rc == 0, err
    monkeypatch.setenv("QAE_SCHEDULER", "1")
    with pytest.raises(ApplyRefused, match="scheduler"): wf.apply(rep["run_id"])
    monkeypatch.delenv("QAE_SCHEDULER")
    (lab / "engine" / "mathx.py").write_text(MATHX + "# someone edited the lab meanwhile\n"); snap = tree_hash(lab)
    with pytest.raises(ApplyRefused, match="drift"): wf.apply(rep["run_id"])
    assert tree_hash(lab) == snap; (lab / "engine" / "mathx.py").write_text(MATHX)
    staged = wf.out / "staged" / rep["run_id"] / "files" / "engine" / "util_fmt.py"; good = staged.read_text()
    staged.write_text("import os\n")
    with pytest.raises(ApplyRefused, match="staged file modified"): wf.apply(rep["run_id"])
    staged.write_text(good); orig = wf.index_path.read_text(); wf.index_path.write_text(orig.replace(rep["patch_sha256"], "0" * 64))
    with pytest.raises(ApplyRefused, match="index"): wf.apply(rep["run_id"])
    wf.index_path.write_text(orig); assert not (lab / "engine" / "util_fmt.py").exists()

# ---------------------------------------------------------------- optional local-LLM patch source (fake transport; no model call)
def test_llm_patch_source_fake_transport_loopback_only_and_single_flight(lab, wf, tmp_path):
    from engine.agents.llm import LLMUnavailable, OllamaClient
    from engine.coding.llm_patch import LLMBusy, PatchProposer, single_flight
    with pytest.raises(ValueError, match="loopback"): OllamaClient(host="http://api.example.com:11434")
    calls = []; lock = tmp_path / "ollama.lock"
    new = MATHX.replace("def double(a):", 'def double(a):\n    """Twice a."""')
    def fake(url, payload):
        calls.append(url)
        assert url.startswith("http://127.0.0.1:")
        with pytest.raises(LLMBusy):                                    # a second request while one is in flight is refused, not queued
            with single_flight(lock): pass
        return {"message": {"content": json.dumps({"new_content": new, "summary": "add docstring", "golden_changes": []})}, "eval_count": 5, "prompt_eval_count": 7}
    pp = PatchProposer(OllamaClient(transport=fake), lab, lock)
    patch, intent, usage = pp.generate("engine/mathx.py", "add a docstring to double")
    assert len(calls) == 1 and usage == {"tokens": 12} and intent["summary"].startswith("[llm:")
    assert parse_patch(patch).paths == ["engine/mathx.py"] and (lab / "engine" / "mathx.py").read_text() == MATHX   # generation never touches the lab
    rep = wf.propose(patch, intent, source="llm")
    assert rep["status"] == "READY_FOR_HUMAN_REVIEW" and rep["source"] == "llm"
    with single_flight(lock): pass                                      # lock released afterwards
    def down(url, payload): raise ConnectionError("synthetic: ollama not running")
    with pytest.raises(LLMUnavailable): PatchProposer(OllamaClient(transport=down), lab, lock).generate("engine/mathx.py", "x")
    evil = lambda url, payload: {"message": {"content": json.dumps({"new_content": MATHX + "import subprocess\n", "summary": "s"})}}
    p2, i2, _ = PatchProposer(OllamaClient(transport=evil), lab, lock).generate("engine/mathx.py", "x")
    assert wf.propose(p2, i2, source="llm")["status"] == "BLOCKED_NEEDS_APPROVAL"                     # LLM output gets no special treatment
    with pytest.raises(PatchRejected): pp.generate("../outside.py", "x")

# ---------------------------------------------------------------- review follow-ups: automation context, untrusted result file, survivors
def test_approve_refuses_agent_env_markers_and_automation_call_stacks(tmp_path, monkeypatch):
    sha = "9" * 64
    for k in ("QAE_JOB_CONTEXT", "QAE_AGENT_CONTEXT", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        assert k in gate._SCHEDULER_ENV
        rc, err = _pty_approve(tmp_path, sha, extra_env={k: "1"}); assert rc == 7 and k in err, (k, err)
    for k in gate._SCHEDULER_ENV: monkeypatch.delenv(k, raising=False)
    for mod in ("engine.orchestration.jobs", "engine.agents.alpha", "engine.api.app", "engine.coding.workflow", "engine.coding.llm_patch", "adapters.registry"):
        ns = {"__name__": mod, "gate": gate, "d": tmp_path, "sha": sha}
        exec("def call(): return gate.approve(d, sha, 'Test Human', 'looked at the diff carefully')", ns)       # a function whose module is `mod`
        with pytest.raises(gate.ApprovalRefused, match="automated code path"): ns["call"]()
    assert not hasattr(CodingWorkflow, "approve")                       # the workflow object offers no in-process approval entry point
    assert list(tmp_path.iterdir()) == []

def test_untrusted_result_file_is_never_read_through_a_symlink_fifo_or_hardlink(tmp_path):
    good = tmp_path / "good.json"; good.write_text(json.dumps({"tests": {}})); assert wfmod._read_untrusted_json(good) == {"tests": {}}
    secret = tmp_path / "secret.json"; secret.write_text(json.dumps({"tests": {"leak": "passed"}}))
    link = tmp_path / "link.json"; os.symlink(secret, link)
    with pytest.raises(OSError): wfmod._read_untrusted_json(link)
    hard = tmp_path / "hard.json"; os.link(secret, hard)
    with pytest.raises(ValueError, match="single-link"): wfmod._read_untrusted_json(hard)
    fifo = tmp_path / "fifo.json"; os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"): wfmod._read_untrusted_json(fifo)
    (tmp_path / "list.json").write_text("[1]")
    with pytest.raises(ValueError): wfmod._read_untrusted_json(tmp_path / "list.json")

def test_result_file_replaced_by_symlink_fails_the_run(lab, wf):
    (lab / "tests" / "test_plant.py").write_text("import os\ndef test_plant():\n    os.makedirs('.coding_out', exist_ok=True)\n"
        "    open('decoy.json', 'w').write('{\"tests\": {}}')\n    os.symlink(os.path.abspath('decoy.json'), '.coding_out/before.json')\n")
    rep = wf.propose(HARMLESS, {})
    assert rep["status"] == "ROLLED_BACK" and "no readable result file" in " ".join(rep["reasons"])

def test_detached_survivor_process_is_killed_and_fails_the_run_even_if_killpg_is_denied(lab, tmp_path, monkeypatch):
    (lab / "tests" / "test_daemon.py").write_text("import os, time\ndef test_daemon():\n    if os.fork() == 0:\n        os.setsid(); time.sleep(600); os._exit(0)\n")
    def denied(pid, sig): raise PermissionError("synthetic: killpg not permitted")
    monkeypatch.setattr(wfmod.os, "killpg", denied)
    rep = CodingWorkflow(lab_root=lab, out_root=tmp_path / "out", test_timeout=20).propose(HARMLESS, {})
    assert rep["status"] == "ROLLED_BACK" and "process(es) running after the run (killed)" in " ".join(rep["reasons"]), rep["reasons"]
    killed = rep["tests"]["before"]["survivors"]["killed"]; assert killed
    import time; time.sleep(1)
    for pid in killed:
        with pytest.raises(ProcessLookupError): os.kill(pid, 0)
