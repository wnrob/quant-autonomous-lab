"""Durable research memory + experiment/strategy registry + job state (SQLite, WAL). Negative results are first-class."""
from __future__ import annotations
import hashlib, json, os, sqlite3, threading, time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

HOME = Path(os.environ.get("QAE_HOME", Path(__file__).resolve().parents[2] / "artifacts" / "qae"))
STATUSES = ("IDEA", "RESEARCH", "VALIDATION", "REJECTED", "CANDIDATE", "SHADOW", "PAPER_APPROVED", "RETIRED")
ALLOWED = {"IDEA": {"RESEARCH", "REJECTED"}, "RESEARCH": {"VALIDATION", "REJECTED"}, "VALIDATION": {"CANDIDATE", "REJECTED"},
           "CANDIDATE": {"SHADOW", "RETIRED", "REJECTED"}, "SHADOW": {"PAPER_APPROVED", "RETIRED"}, "PAPER_APPROVED": {"RETIRED"},
           "REJECTED": set(), "RETIRED": set()}
SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses(id INTEGER PRIMARY KEY, created REAL, text TEXT, rationale TEXT, source TEXT, parent_experiment INTEGER);
CREATE TABLE IF NOT EXISTS strategies(id INTEGER PRIMARY KEY, name TEXT, family TEXT, spec_hash TEXT UNIQUE, spec TEXT, status TEXT, parent_id INTEGER, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS status_history(id INTEGER PRIMARY KEY, strategy_id INTEGER, old TEXT, new TEXT, reason TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY, strategy_id INTEGER, hypothesis_id INTEGER, spec_hash TEXT, criteria_hash TEXT, data_sha TEXT,
  code_version TEXT, seed INTEGER, created REAL, secs REAL, metrics TEXT, decision TEXT, reasons TEXT, report_path TEXT, UNIQUE(spec_hash, criteria_hash, data_sha));
CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY, ts REAL, experiment_id INTEGER, family TEXT, kind TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS test_access(id INTEGER PRIMARY KEY, ts REAL, spec_hash TEXT UNIQUE, family TEXT, data_sha TEXT);
CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY, interval_s REAL, state TEXT, next_run REAL, last_run REAL, last_status TEXT, failures INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS job_runs(id INTEGER PRIMARY KEY, name TEXT, started REAL, finished REAL, status TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS ledger(id INTEGER PRIMARY KEY, ts REAL, resource TEXT, amount REAL, note TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(kind, ref, text);
"""
def h(obj: Any) -> str: return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

class Store:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else HOME / "qae.db"; self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, check_same_thread=False); self.lock = threading.RLock(); self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.executescript(SCHEMA)
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            try: yield self.db; self.db.commit()
            except Exception: self.db.rollback(); raise
    def q(self, sql: str, *a) -> list[dict]:
        with self.lock: return [dict(r) for r in self.db.execute(sql, a).fetchall()]
    def _index(self, kind, ref, text):
        self.db.execute("INSERT INTO memory_fts(kind,ref,text) VALUES(?,?,?)", (kind, str(ref), text))
    # --- hypotheses / strategies
    def add_hypothesis(self, text, rationale, source="rule_based", parent_experiment=None) -> int:
        with self.tx() as d:
            cur = d.execute("INSERT INTO hypotheses VALUES(NULL,?,?,?,?,?)", (time.time(), text, rationale, source, parent_experiment))
            self._index("hypothesis", cur.lastrowid, f"{text} {rationale}"); return cur.lastrowid
    def get_or_create_strategy(self, spec: dict, name: str, parent_id: int | None = None) -> tuple[int, bool]:
        sh = h(spec); r = self.q("SELECT id FROM strategies WHERE spec_hash=?", sh)
        if r: return r[0]["id"], False
        with self.tx() as d:
            cur = d.execute("INSERT INTO strategies VALUES(NULL,?,?,?,?,?,?,?,?)", (name, spec["family"], sh, json.dumps(spec, sort_keys=True), "IDEA", parent_id, time.time(), time.time()))
            d.execute("INSERT INTO status_history VALUES(NULL,?,?,?,?,?)", (cur.lastrowid, None, "IDEA", "created", time.time()))
            self._index("strategy", cur.lastrowid, f"{name} {json.dumps(spec)}"); return cur.lastrowid, True
    def set_status(self, sid: int, new: str, reason: str) -> None:
        old = self.q("SELECT status FROM strategies WHERE id=?", sid)[0]["status"]
        if new not in ALLOWED[old]: raise ValueError(f"illegal transition {old}->{new}")
        if new in ("SHADOW", "PAPER_APPROVED"): raise PermissionError("promotion beyond CANDIDATE requires a separate forward-evaluation process (not implemented) ")
        with self.tx() as d:
            d.execute("UPDATE strategies SET status=?, updated=? WHERE id=?", (new, time.time(), sid))
            d.execute("INSERT INTO status_history VALUES(NULL,?,?,?,?,?)", (sid, old, new, reason, time.time()))
    # --- experiments
    def find_experiment(self, spec_hash, criteria_hash, data_sha) -> dict | None:
        r = self.q("SELECT * FROM experiments WHERE spec_hash=? AND criteria_hash=? AND data_sha=?", spec_hash, criteria_hash, data_sha); return r[0] if r else None
    def add_experiment(self, **k) -> int:
        with self.tx() as d:
            cur = d.execute("INSERT INTO experiments VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)", (k["strategy_id"], k.get("hypothesis_id"), k["spec_hash"], k["criteria_hash"], k["data_sha"],
                k["code_version"], k["seed"], time.time(), k["secs"], json.dumps(k["metrics"], default=float), k["decision"], json.dumps(k["reasons"]), k.get("report_path")))
            self._index("experiment", cur.lastrowid, f"{k['decision']} {' '.join(k['reasons'])}"); return cur.lastrowid
    def add_lesson(self, experiment_id, family, kind, text):
        with self.tx() as d:
            cur = d.execute("INSERT INTO lessons VALUES(NULL,?,?,?,?,?)", (time.time(), experiment_id, family, kind, text)); self._index("lesson", cur.lastrowid, text)
    def search(self, query: str, limit: int = 20) -> list[dict]:
        safe = " ".join(f'"{t}"' for t in query.replace('"', " ").split()) or '""'
        return self.q("SELECT kind, ref, snippet(memory_fts, 2, '[', ']', '…', 24) AS snippet FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?", safe, limit)
    # --- protected test set
    def test_looks(self) -> int: return self.q("SELECT COUNT(*) n FROM test_access")[0]["n"]
    def record_test_access(self, spec_hash, family, data_sha): 
        with self.tx() as d: d.execute("INSERT INTO test_access VALUES(NULL,?,?,?,?)", (time.time(), spec_hash, family, data_sha))
    def test_seen(self, spec_hash) -> bool: return bool(self.q("SELECT 1 FROM test_access WHERE spec_hash=?", spec_hash))
    # --- budget ledger
    def spend(self, resource, amount, note=""):
        with self.tx() as d: d.execute("INSERT INTO ledger VALUES(NULL,?,?,?,?)", (time.time(), resource, amount, note))
    def spent(self, resource, since: float = 0) -> float:
        return self.q("SELECT COALESCE(SUM(amount),0) s FROM ledger WHERE resource=? AND ts>=?", resource, since)[0]["s"]
