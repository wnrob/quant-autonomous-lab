"""Strict unified-diff parser/validator. Whitelist grammar: anything it does not recognise is rejected (fails closed).

Accepted: text hunks for regular files, new files and deletions with mode 100644, paths under the copied source roots.
Rejected: absolute paths, `..`, backslashes, quoted/non-ASCII paths, any `.git*` component, binary patches, renames/copies,
mode changes / exec bits / symlinks (120000) / submodules, paths outside the allowed roots, paths that traverse a symlink.
"""
from __future__ import annotations
import hashlib, os, re
from dataclasses import dataclass, field
from pathlib import Path
from engine.errors import QALError

class PatchRejected(QALError): pass

ALLOWED_ROOTS: tuple[str, ...] = ("engine", "adapters", "tests", "configs", "scripts")
ALLOWED_FILES: tuple[str, ...] = ("pyproject.toml",)
MAX_PATCH_BYTES = 400_000
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_PATH_OK = re.compile(r"^[A-Za-z0-9_.\-/]+$")

@dataclass
class FilePatch:
    path: str
    kind: str = "modify"                 # modify | add | delete
    added: list[str] = field(default_factory=list)
    removed: int = 0

@dataclass
class ParsedPatch:
    sha256: str
    files: list[FilePatch]
    @property
    def paths(self) -> list[str]: return [f.path for f in self.files]
    @property
    def added_lines(self) -> dict[str, list[str]]: return {f.path: f.added for f in self.files}

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def check_path(raw: str) -> str:
    """Validate one repo-relative path (no a/ b/ prefix). Returns it unchanged or raises."""
    if not raw or raw != raw.strip(): raise PatchRejected(f"empty or padded path: {raw!r}")
    if raw.startswith('"') or "\\" in raw: raise PatchRejected(f"quoted/backslash path not allowed: {raw!r}")
    if not _PATH_OK.match(raw): raise PatchRejected(f"path has characters outside [A-Za-z0-9_.-/]: {raw!r}")
    if raw.startswith("/") or raw.startswith("~"): raise PatchRejected(f"absolute path not allowed: {raw!r}")
    parts = raw.split("/")
    if any(p in ("", ".", "..") for p in parts): raise PatchRejected(f"path traversal / empty component: {raw!r}")
    if any(p.casefold().startswith(".git") for p in parts): raise PatchRejected(f"path touches git metadata: {raw!r}")
    if any(p.startswith(".coding_") for p in parts): raise PatchRejected(f"path touches workflow scratch dirs: {raw!r}")
    top = parts[0]                                  # exact case: "Engine/..." is rejected, not folded
    if len(parts) == 1:
        if top not in ALLOWED_FILES: raise PatchRejected(f"path outside the workspace source roots: {raw!r}")
    elif top not in ALLOWED_ROOTS: raise PatchRejected(f"path outside the workspace source roots: {raw!r}")
    return raw

def _strip(side: str, prefix: str) -> str | None:
    side = side.split("\t")[0].rstrip("\r")
    if side == "/dev/null": return None
    if not side.startswith(prefix): raise PatchRejected(f"expected '{prefix}' prefix in header path: {side!r}")
    return check_path(side[len(prefix):])

def parse_patch(text: str) -> ParsedPatch:
    if not isinstance(text, str) or not text.strip(): raise PatchRejected("empty patch")
    if len(text.encode("utf-8", "replace")) > MAX_PATCH_BYTES: raise PatchRejected("patch too large")
    if "\x00" in text: raise PatchRejected("NUL byte in patch (binary?)")
    if "GIT binary patch" in text or re.search(r"^Binary files ", text, re.M): raise PatchRejected("binary patches are not allowed")
    lines = text.split("\n")
    if lines and lines[-1] == "": lines.pop()
    files: list[FilePatch] = []; cur: FilePatch | None = None; i = 0; pending_git: tuple[str, str] | None = None
    old_path = None; have_old = False; declared_kind = None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("diff --git "):
            m = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", ln.rstrip("\r"))
            if not m: raise PatchRejected(f"malformed diff --git header: {ln!r}")
            a, b = check_path(m.group(1)), check_path(m.group(2))
            if a != b: raise PatchRejected("renames/copies are not allowed (a/ and b/ paths differ)")
            pending_git, cur, have_old, declared_kind = (a, b), None, False, None; i += 1; continue
        if ln.startswith("index "):
            if not re.fullmatch(r"index [0-9a-f]+\.\.[0-9a-f]+( 100644)?", ln.rstrip("\r")): raise PatchRejected(f"index line with non-100644 mode or bad format: {ln!r}")
            i += 1; continue
        if ln.startswith("new file mode ") or ln.startswith("deleted file mode "):
            if ln.rstrip("\r").rsplit(" ", 1)[-1] != "100644": raise PatchRejected(f"only mode 100644 is allowed (no exec bits, symlinks, submodules): {ln!r}")
            declared_kind = "add" if ln.startswith("new") else "delete"; i += 1; continue
        if ln.startswith("--- "):
            old_path = _strip(ln[4:], "a/"); have_old = True; i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "): raise PatchRejected("'---' header not followed by '+++'")
            new_path = _strip(lines[i][4:], "b/"); i += 1
            if old_path is None and new_path is None: raise PatchRejected("both sides are /dev/null")
            if old_path and new_path and old_path != new_path: raise PatchRejected("renames are not allowed (--- and +++ paths differ)")
            path = new_path or old_path; kind = "add" if old_path is None else "delete" if new_path is None else "modify"
            if pending_git and pending_git[0] != path: raise PatchRejected("diff --git header and ---/+++ paths disagree")
            if declared_kind and declared_kind != kind: raise PatchRejected("file mode line disagrees with ---/+++ headers")
            if any(f.path.casefold() == path.casefold() for f in files): raise PatchRejected(f"path appears twice in patch: {path}")
            cur = FilePatch(path, kind); files.append(cur); pending_git = None; declared_kind = None
            if i >= len(lines) or not _HUNK.match(lines[i]): raise PatchRejected(f"no hunk after file header for {path}")
            continue
        m = _HUNK.match(ln)
        if m:
            if cur is None: raise PatchRejected("hunk before any file header")
            n_old = int(m.group(2)) if m.group(2) is not None else 1; n_new = int(m.group(4)) if m.group(4) is not None else 1
            i += 1
            while n_old > 0 or n_new > 0:
                if i >= len(lines): raise PatchRejected("truncated hunk")
                h = lines[i]; tag = h[:1]
                if tag == "+": cur.added.append(h[1:]); n_new -= 1
                elif tag == "-": cur.removed += 1; n_old -= 1
                elif tag == " " or h == "": n_old -= 1; n_new -= 1
                elif tag == "\\": pass
                else: raise PatchRejected(f"unexpected line inside hunk: {h[:60]!r}")
                if n_old < 0 or n_new < 0: raise PatchRejected("hunk line counts do not match its header")
                i += 1
            if i < len(lines) and lines[i].startswith("\\"): i += 1          # "\ No newline at end of file"
            continue
        raise PatchRejected(f"unrecognised patch line (mode change, rename, binary or junk): {ln[:80]!r}")
    if pending_git is not None: raise PatchRejected("diff --git header without content (mode-only or binary change)")
    if not files: raise PatchRejected("patch contains no file changes")
    return ParsedPatch(sha256_text(text), files)

def check_against_tree(parsed: ParsedPatch, root: Path) -> None:
    """Paths must resolve inside `root` without crossing a symlink; add must not exist, modify/delete must be a regular file."""
    root = Path(root).resolve()
    for f in parsed.files:
        p = root
        for part in f.path.split("/"):
            p = p / part
            if p.is_symlink(): raise PatchRejected(f"path crosses a symlink: {f.path}")
            if p.parent.is_dir() and part not in os.listdir(p.parent) and any(x.casefold() == part.casefold() for x in os.listdir(p.parent)):
                raise PatchRejected(f"path differs only by case from an existing entry: {f.path}")
        if root not in p.resolve().parents: raise PatchRejected(f"path escapes the workspace: {f.path}")
        if f.kind == "add" and p.exists(): raise PatchRejected(f"'new file' already exists: {f.path}")
        if f.kind != "add" and not p.is_file(): raise PatchRejected(f"file to {f.kind} does not exist: {f.path}")
