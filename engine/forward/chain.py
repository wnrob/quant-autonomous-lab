"""Append-only, tamper-evident JSONL log (hash chain).

Each line is {"seq", "prev", "body", "hash"} with hash = sha256(prev + canonical_json({"seq","prev","body"})).
A sidecar `<file>.head` holds {"n", "head"} so that truncating the tail is detected too.

Limits (stated, not hidden): there is no secret key. Someone with write access who rewrites the WHOLE file and the
sidecar consistently produces a valid chain. Protection against that needs an anchor held elsewhere: pass
`expected_head`/`anchors` to verify() (approval records anchor the signal log this way), or keep the head hash off-box."""
from __future__ import annotations
import fcntl, hashlib, json, os
from pathlib import Path
from typing import Any, Iterable

GENESIS = "0" * 64

class ChainError(Exception):
    """The log failed verification (edit / delete / reorder / insert / truncation) or an append was refused."""

def canonical(obj: Any) -> str:
    """Deterministic JSON. Non-JSON types are rejected (no default=str), so what is hashed is exactly what is stored."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def sha(text: str) -> str: return hashlib.sha256(text.encode()).hexdigest()

def record_hash(seq: int, prev: str, body: dict) -> str:
    return sha(prev + canonical({"seq": seq, "prev": prev, "body": body}))

class HashChain:
    def __init__(self, path: Path):
        self.path = Path(path); self.head_path = self.path.with_name(self.path.name + ".head")

    # --- read
    def _lines(self) -> list[str]:
        if not self.path.exists(): return []
        return [ln for ln in self.path.read_text().split("\n") if ln != ""]

    def verify(self, expected_head: str | None = None, anchors: Iterable[tuple[int, str]] = ()) -> list[dict]:
        """Recompute the whole chain. Returns the records; raises ChainError on any inconsistency.
        anchors: (seq, hash) pairs recorded elsewhere that must still be present at those positions."""
        recs: list[dict] = []; prev = GENESIS
        for i, ln in enumerate(self._lines()):
            try: r = json.loads(ln)
            except json.JSONDecodeError as e: raise ChainError(f"line {i}: not JSON ({e})") from e
            if not isinstance(r, dict) or set(r) != {"seq", "prev", "body", "hash"}: raise ChainError(f"line {i}: wrong record shape")
            if r["seq"] != i: raise ChainError(f"line {i}: seq {r['seq']} != position {i} (delete/reorder/insert)")
            if r["prev"] != prev: raise ChainError(f"line {i}: prev hash does not match preceding record (delete/reorder/insert)")
            if record_hash(r["seq"], r["prev"], r["body"]) != r["hash"]: raise ChainError(f"line {i}: content hash mismatch (record edited)")
            if ln != canonical(r): raise ChainError(f"line {i}: non-canonical encoding")
            recs.append(r); prev = r["hash"]
        if self.head_path.exists():
            hd = json.loads(self.head_path.read_text())
            if hd != {"n": len(recs), "head": prev}: raise ChainError(f"head anchor {hd} != chain (n={len(recs)}, head={prev[:12]}): tail truncated or anchor edited")
        elif recs: raise ChainError("head anchor file missing for a non-empty log")
        if expected_head is not None and prev != expected_head: raise ChainError(f"head {prev[:12]} != expected {expected_head[:12]}")
        for seq, hh in anchors:
            if seq >= len(recs) or recs[seq]["hash"] != hh: raise ChainError(f"externally anchored record seq={seq} hash={hh[:12]} is missing or changed")
        return recs

    def head(self) -> tuple[int, str]:
        recs = self.verify(); return len(recs), (recs[-1]["hash"] if recs else GENESIS)

    # --- write (append only; the whole chain is re-verified under an exclusive lock before every append)
    def append(self, body: dict) -> dict:
        canonical(body)                                            # reject non-serialisable / NaN before touching the file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        with open(lock, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            recs = self.verify()
            seq = len(recs); prev = recs[-1]["hash"] if recs else GENESIS
            rec = {"seq": seq, "prev": prev, "body": body, "hash": record_hash(seq, prev, body)}
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try: os.write(fd, (canonical(rec) + "\n").encode()); os.fsync(fd)
            finally: os.close(fd)
            tmp = self.head_path.with_name(self.head_path.name + ".tmp")
            tmp.write_text(json.dumps({"n": seq + 1, "head": rec["hash"]})); os.replace(tmp, self.head_path)
        return rec
