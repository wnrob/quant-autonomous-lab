"""Optional patch source: local open-source LLM over loopback (reuses engine.agents.llm.OllamaClient, which refuses non-loopback hosts).

The model never emits a diff directly (models are unreliable at hunk arithmetic). It returns the full new text of ONE file;
the diff is computed here with difflib, then goes through exactly the same validation, gate, sandbox and review as a human patch.
At most one request at a time across processes (flock, non-blocking: a second caller gets LLMBusy, it does not queue).
"""
from __future__ import annotations
import difflib, fcntl
from contextlib import contextmanager
from pathlib import Path
from engine.agents.llm import LLMUnavailable, OllamaClient
from engine.coding.patch import PatchRejected, check_path, parse_patch
from engine.coding import golden

class LLMBusy(LLMUnavailable): pass
MAX_FILE_BYTES = 12_000
SCHEMA = {"type": "object", "properties": {"new_content": {"type": "string"}, "summary": {"type": "string"},
          "golden_changes": {"type": "array", "items": {"type": "string"}}}, "required": ["new_content", "summary"]}
SYSTEM = ("You are a careful Python maintainer. You are given ONE file and ONE instruction. Return JSON with `new_content` (the COMPLETE new "
          "text of that file), `summary` (one sentence) and `golden_changes` (names of behaviour probes whose numeric output you intend to change, "
          f"from {list(golden.PROBE_NAMES)}; normally empty). Change as little as possible. Do not add imports of networking or process modules.")

@contextmanager
def single_flight(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    try:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e: raise LLMBusy("another local LLM request is in flight (max 1 concurrent)") from e
        yield
    finally:
        f.close()                                                    # closing releases the flock

def diff_for(path: str, old: str, new: str) -> str:
    if old and not old.endswith("\n"): raise PatchRejected("original file has no trailing newline; not supported by the LLM path")
    if not new.endswith("\n"): new += "\n"
    if old == new: raise PatchRejected("model returned the file unchanged")
    body = "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{path}", f"b/{path}", n=3))
    return f"diff --git a/{path} b/{path}\n{body}"

class PatchProposer:
    def __init__(self, client: OllamaClient, lab_root: Path, lock_path: Path):
        self.c, self.lab, self.lock = client, Path(lab_root), Path(lock_path)
    def generate(self, path: str, instruction: str) -> tuple[str, dict, dict]:
        """-> (unified diff text, intent, usage). Raises LLMUnavailable/LLMBusy/PatchRejected; never touches the lab."""
        check_path(path); src = self.lab / path
        if src.is_symlink() or not src.is_file(): raise PatchRejected(f"no such regular file: {path}")
        old = src.read_text()
        if len(old.encode()) > MAX_FILE_BYTES: raise PatchRejected(f"file too large for the local model context ({len(old.encode())} > {MAX_FILE_BYTES} bytes)")
        user = f"File: {path}\nInstruction: {instruction}\n----- CURRENT CONTENT -----\n{old}"
        with single_flight(self.lock):
            raw, usage = self.c.chat_json(SYSTEM, user, SCHEMA)
        new = raw.get("new_content")
        if not isinstance(new, str) or not new.strip(): raise PatchRejected("model output has no usable new_content")
        patch = diff_for(path, old, new); parse_patch(patch)           # must already satisfy the strict grammar
        gc = [g for g in raw.get("golden_changes") or [] if isinstance(g, str)]
        return patch, {"summary": f"[llm:{self.c.model}] {str(raw.get('summary', ''))[:300]} (instruction: {instruction[:200]})", "golden_changes": gc}, usage
