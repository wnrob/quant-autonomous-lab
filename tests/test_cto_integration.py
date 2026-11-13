"""CTO integration tests: scheduler marks job context for the human-approval gates; CLI wiring; roster honesty."""
from __future__ import annotations
import os
from pathlib import Path
from engine.memory.store import Store
from engine.orchestration.scheduler import Scheduler


def _sched(tmp_path, handler):
    cfg = tmp_path / "s.toml"; cfg.write_text("[jobs.j]\ninterval_s = 60\n")
    return Scheduler(Store(tmp_path / "q.db"), {"j": handler}, config=cfg)


def test_scheduler_sets_and_clears_automation_markers(tmp_path, monkeypatch):
    monkeypatch.delenv("QAE_SCHEDULER", raising=False); monkeypatch.delenv("QAE_JOB_CONTEXT", raising=False)
    seen = {}
    def h(s):
        seen.update(sched=os.environ.get("QAE_SCHEDULER"), ctx=os.environ.get("QAE_JOB_CONTEXT")); return "ok"
    assert _sched(tmp_path, h).run_job("j") == "ok"
    assert seen == {"sched": "1", "ctx": "j"}
    assert "QAE_SCHEDULER" not in os.environ and "QAE_JOB_CONTEXT" not in os.environ   # restored after the job


def test_coding_gate_refuses_inside_scheduler_job_even_with_tty_faked(tmp_path, monkeypatch):
    """Coding approve() relies on env markers (no stack check): the scheduler marker must make it refuse."""
    import engine.coding.gate as gate
    monkeypatch.setattr(os, "isatty", lambda fd: True)          # pretend a terminal; the marker alone must refuse
    for k in gate._SCHEDULER_ENV: monkeypatch.delenv(k, raising=False)
    out = {}
    def h(s):
        out["r"] = gate.interactive_context(); return "ok"
    _sched(tmp_path, h).run_job("j")
    assert out["r"][0] is False and "QAE_SCHEDULER" in out["r"][1]


def test_forward_gate_refuses_inside_scheduler_job(tmp_path, monkeypatch):
    from engine.forward.approval import AUTOMATION_ENV, automation_context
    for k in AUTOMATION_ENV: monkeypatch.delenv(k, raising=False)
    out = {}
    def h(s):
        out["r"] = automation_context(); return "ok"
    _sched(tmp_path, h).run_job("j")
    assert out["r"] and "QAE_JOB_CONTEXT" in out["r"]


def test_cli_exposes_new_commands_without_preapproval_flags():
    import argparse, engine.qae_cli as cli
    src = Path(cli.__file__).read_text()
    for cmd in ('"forward"', '"code"', '"futures-audit"', '"eventsim"'): assert f"add_parser({cmd}" in src
    for flag in ("--yes", "--force", "--approve", "--no-gate"): assert flag not in src


def test_roster_new_agents_state_limits():
    from engine.agents.roles import ROSTER
    new = [a for a in ROSTER if a.id in (11, 12)]
    assert len(new) == 2 and all(("NOT" in a.scope or "no " in a.scope or "only" in a.scope) for a in new)
