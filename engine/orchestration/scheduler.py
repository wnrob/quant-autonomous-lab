"""Durable scheduler: job state lives in SQLite, so pause/resume/cancel/retry and crash-recovery survive restarts.
Intentionally simple (poll loop) - single-process local operation; APScheduler/queue workers are a later-stage option."""
from __future__ import annotations
import logging, os, time, tomllib, traceback
from pathlib import Path
from typing import Callable
from engine.memory.store import Store
from engine.orchestration.budget import BudgetExceeded
log = logging.getLogger(__name__)
MAX_FAILURES = 3
Handler = Callable[[Store], str]

class Scheduler:
    def __init__(self, store: Store, handlers: dict[str, Handler], config: Path | None = None):
        self.s, self.handlers = store, handlers
        cfg = tomllib.loads((config or Path(__file__).resolve().parents[2] / "configs/schedule.toml").read_text())["jobs"]
        now = time.time()
        for name, c in cfg.items():
            if name not in handlers: raise ValueError(f"no handler for job {name}")
            if not store.q("SELECT 1 FROM jobs WHERE name=?", name):
                with store.tx() as d: d.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,0)", (name, c["interval_s"], "enabled", now, None, None))
            else:
                with store.tx() as d: d.execute("UPDATE jobs SET interval_s=? WHERE name=?", (c["interval_s"], name))   # config-driven intervals
        self.recover()
    def recover(self) -> int:
        """Runs left 'running' by a crash become 'interrupted' and their job becomes due immediately."""
        n = self.s.q("SELECT COUNT(*) n FROM job_runs WHERE status='running'")[0]["n"]
        with self.s.tx() as d:
            d.execute("UPDATE job_runs SET status='interrupted', finished=? WHERE status='running'", (time.time(),))
            d.execute("UPDATE jobs SET next_run=? WHERE name IN (SELECT name FROM job_runs WHERE status='interrupted')", (time.time(),))
        return n
    def set_state(self, name: str, state: str):
        if state not in ("enabled", "paused", "cancelled"): raise ValueError(state)
        with self.s.tx() as d: d.execute("UPDATE jobs SET state=?, failures=CASE WHEN ?='enabled' THEN 0 ELSE failures END WHERE name=?", (state, state, name))
    def retry(self, name: str):
        with self.s.tx() as d: d.execute("UPDATE jobs SET state='enabled', failures=0, next_run=? WHERE name=?", (time.time(), name))
    def run_job(self, name: str) -> str:
        t = time.time()
        with self.s.tx() as d: rid = d.execute("INSERT INTO job_runs VALUES(NULL,?,?,NULL,'running',NULL)", (name, t)).lastrowid
        prev = {k: os.environ.get(k) for k in ("QAE_SCHEDULER", "QAE_JOB_CONTEXT")}
        os.environ["QAE_SCHEDULER"] = "1"; os.environ["QAE_JOB_CONTEXT"] = name    # human-approval gates (forward, coding) refuse under these markers
        try: detail, status = self.handlers[name](self.s), "ok"
        except BudgetExceeded as e: detail, status = f"budget: {e}", "budget_blocked"
        except Exception as e: detail, status = traceback.format_exc()[-800:], "failed"; log.exception("job %s failed", name)
        finally:
            for k, v in prev.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
        j = self.s.q("SELECT interval_s, failures FROM jobs WHERE name=?", name)[0]; fails = j["failures"] + 1 if status == "failed" else 0
        with self.s.tx() as d:
            d.execute("UPDATE job_runs SET finished=?, status=?, detail=? WHERE id=?", (time.time(), status, detail, rid))
            d.execute("UPDATE jobs SET last_run=?, last_status=?, failures=?, next_run=?, state=CASE WHEN ?>=? THEN 'paused' ELSE state END WHERE name=?",
                      (t, status, fails, time.time() + j["interval_s"], fails, MAX_FAILURES, name))
        return status
    def tick(self, now: float | None = None) -> dict[str, str]:
        now = now or time.time(); ran = {}
        for j in self.s.q("SELECT name FROM jobs WHERE state='enabled' AND next_run<=? ORDER BY next_run", now): ran[j["name"]] = self.run_job(j["name"])
        return ran
    def loop(self, poll_s: float = 30):
        while True: self.tick(); time.sleep(poll_s)
