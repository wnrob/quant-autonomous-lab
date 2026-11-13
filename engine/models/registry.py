"""Model registry: SQLite table `ml_models` living in the Store's database (created here, store.py untouched).

Status machine is the Store's (IDEA -> RESEARCH -> VALIDATION -> CANDIDATE|REJECTED ...): same ALLOWED table, terminal states
(REJECTED, RETIRED) are never resurrected, and nothing above CANDIDATE is reachable (PermissionError), exactly like Store.set_status.
Registration is idempotent: the same (name, family, params, feature_hash, data_sha, seed) returns the existing row.
Artifacts are joblib files under artifacts/models/. Loading is gated: confined path + sha256 seal (ml_artifact_seals) verified BEFORE joblib.load (pickle = code execution).
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Any
from engine.memory.store import ALLOWED, Store, h

from engine.errors import ArtifactIntegrityError
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "models"
SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_models(id INTEGER PRIMARY KEY, name TEXT, family TEXT, params TEXT, feature_hash TEXT, data_sha TEXT, code_version TEXT,
  seed INTEGER, created REAL, metrics TEXT, status TEXT, parent_id INTEGER, artifact_path TEXT, dedup_key TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS ml_artifact_seals(model_id INTEGER PRIMARY KEY, sha256 TEXT NOT NULL, sealed_at REAL, how TEXT);
CREATE TABLE IF NOT EXISTS ml_status_history(id INTEGER PRIMARY KEY, model_id INTEGER, old TEXT, new TEXT, reason TEXT, ts REAL);
"""

class Registry:
    def __init__(self, store: Store, artifact_dir: Path | None = None):
        self.store = store; self.dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACTS
        with store.lock: store.db.executescript(SCHEMA); store.db.commit()

    def register(self, name: str, family: str, params: dict, feature_hash: str, data_sha: str, code_version: str, seed: int,
                 metrics: dict | None = None, parent_id: int | None = None, model: Any = None) -> tuple[int, bool]:
        """Returns (id, created). A duplicate returns the existing id and does not overwrite metrics, status or artifact."""
        key = h([name, family, params, feature_hash, data_sha, seed]); old = self.store.q("SELECT id FROM ml_models WHERE dedup_key=?", key)
        if old: return old[0]["id"], False
        with self.store.tx() as d:
            cur = d.execute("INSERT INTO ml_models VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)", (name, family, json.dumps(params, sort_keys=True, default=str), feature_hash,
                            data_sha, code_version, seed, time.time(), json.dumps(metrics or {}, default=float), "IDEA", parent_id, None, key))
            mid = cur.lastrowid; d.execute("INSERT INTO ml_status_history VALUES(NULL,?,?,?,?,?)", (mid, None, "IDEA", "registered", time.time()))
        if model is not None: self.save_artifact(mid, model)
        return mid, True

    @staticmethod
    def _sha(p: Path) -> str:
        import hashlib; return hashlib.sha256(p.read_bytes()).hexdigest()

    def _confined(self, p: str | Path) -> Path:
        """Resolve symlinks; the real path must sit inside this registry's artifact dir and not be a symlink itself."""
        raw = Path(p); real = raw.resolve(); base = self.dir.resolve()
        if raw.is_symlink() or not real.is_relative_to(base): raise ArtifactIntegrityError(f"artifact path {raw} is outside the registry directory {base} (or a symlink); refusing")
        if not real.is_file(): raise FileNotFoundError(real)
        return real

    def save_artifact(self, mid: int, model: Any) -> Path:
        import joblib
        row = self.get(mid); self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{mid:05d}_{row['name']}_s{row['seed']}.joblib"; joblib.dump(model, p)
        with self.store.tx() as d:
            d.execute("UPDATE ml_models SET artifact_path=? WHERE id=?", (str(p), mid))
            d.execute("INSERT OR REPLACE INTO ml_artifact_seals VALUES(?,?,?,?)", (mid, self._sha(p), time.time(), "written_by_registry"))
        return p

    def seal_existing(self, mid: int) -> str:
        """EXPLICIT human/lead action for legacy rows with no seal: trust-on-first-use of a file that already sits inside the registry dir."""
        p = self._confined(self.get(mid)["artifact_path"] or ""); sha = self._sha(p)
        with self.store.tx() as d: d.execute("INSERT OR REPLACE INTO ml_artifact_seals VALUES(?,?,?,?)", (mid, sha, time.time(), "manual_seal"))
        return sha

    def load_artifact(self, mid: int) -> Any:
        """joblib.load has pickle semantics, so it runs ONLY after: path confined to the registry dir (no symlink escape), a seal exists, and the sha256 matches."""
        import joblib
        path = self.get(mid)["artifact_path"]
        if not path: raise FileNotFoundError(f"model {mid} has no artifact")
        p = self._confined(path); seal = self.store.q("SELECT sha256 FROM ml_artifact_seals WHERE model_id=?", mid)
        if not seal: raise ArtifactIntegrityError(f"model {mid} artifact has no integrity seal (legacy row); verify it yourself then call seal_existing({mid})")
        if self._sha(p) != seal[0]["sha256"]: raise ArtifactIntegrityError(f"model {mid} artifact hash mismatch: file modified after it was written; refusing to load")
        return joblib.load(p)

    def get(self, mid: int) -> dict:
        r = self.store.q("SELECT * FROM ml_models WHERE id=?", mid)
        if not r: raise KeyError(f"no ml model {mid}")
        r = r[0]; r["params"] = json.loads(r["params"]); r["metrics"] = json.loads(r["metrics"]); return r

    def set_metrics(self, mid: int, metrics: dict) -> None:
        if self.get(mid)["status"] in ("REJECTED", "RETIRED"): raise ValueError("terminal model: metrics are frozen")
        with self.store.tx() as d: d.execute("UPDATE ml_models SET metrics=? WHERE id=?", (json.dumps(metrics, default=float), mid))

    def set_status(self, mid: int, new: str, reason: str) -> None:
        old = self.get(mid)["status"]
        if new not in ALLOWED[old]: raise ValueError(f"illegal transition {old}->{new}")
        if new in ("SHADOW", "PAPER_APPROVED"): raise PermissionError("promotion beyond CANDIDATE requires a separate forward-evaluation process (not implemented)")
        with self.store.tx() as d:
            d.execute("UPDATE ml_models SET status=? WHERE id=?", (new, mid))
            d.execute("INSERT INTO ml_status_history VALUES(NULL,?,?,?,?,?)", (mid, old, new, reason, time.time()))

    def advance(self, mid: int, new: str, reason: str) -> bool:
        """Move only if legal (never resurrects terminal states); returns whether it moved."""
        if new in ALLOWED[self.get(mid)["status"]]: self.set_status(mid, new, reason); return True
        return False

    def list(self, status: str | None = None) -> list[dict]:
        return self.store.q("SELECT id,name,family,seed,status,artifact_path FROM ml_models" + (" WHERE status=?" if status else "") + " ORDER BY id", *([status] if status else []))
