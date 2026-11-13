"""Two-gate safe loading for untrusted pickles. STDLIB ONLY.

Gate 1 (static, never executes the file): `scan_globals` walks the opcode stream with `pickletools.genops`, resolves
every GLOBAL / INST / STACK_GLOBAL to a (module, name) pair (tracking the memo for STACK_GLOBAL operands), and
`check` refuses the file if ANY reference is unresolvable or not in the exact-pair allowlist.
Gate 2 (at load): `RestrictedUnpickler.find_class` re-checks every lookup against the same allowlist.

The allowlist is EXACT (module, name) pairs, not whole modules: e.g. `builtins.slice` is allowed but `builtins.eval`,
`builtins.getattr`, `os.system` are not. Pairs are the pandas/numpy/datetime reconstruction helpers needed for a
DataFrame with a DatetimeIndex + string columns, plus three pyarrow reconstruction helpers (pandas' ArrowStringArray
pickles through pyarrow). The pyarrow entries are a DISCLOSED extension of the task's pandas/numpy/builtins/datetime list.
Loading must happen in envs/derivs (the worker), never in the core env.
"""
from __future__ import annotations
import io, pickle, pickletools

ALLOWED: frozenset[tuple[str, str]] = frozenset({
    ("builtins", "slice"),
    ("datetime", "date"), ("datetime", "datetime"), ("datetime", "timedelta"), ("datetime", "timezone"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.numeric", "_frombuffer"),
    ("pandas", "DataFrame"), ("pandas", "Series"), ("pandas", "Index"), ("pandas", "DatetimeIndex"),
    ("pandas", "StringDtype"), ("pandas", "DatetimeTZDtype"), ("pandas", "Timestamp"),
    ("pandas.arrays", "ArrowStringArray"), ("pandas.arrays", "DatetimeArray"), ("pandas.arrays", "StringArray"),
    ("pandas.core.internals.managers", "BlockManager"), ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas._libs.internals", "_unpickle_block"), ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
    ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
    # disclosed extension (pyarrow-backed string columns):
    ("pyarrow.lib", "_restore_array"), ("pyarrow.lib", "py_buffer"), ("pyarrow.lib", "type_for_alias"),
})


class UnsafePickle(RuntimeError):
    """The pickle references a global outside the allowlist (or one that cannot be resolved statically). NOT loaded."""


_STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING"}
_NOPUSH = {"MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT", "FRAME", "PROTO", "STOP", "MARK", "POP", "POP_MARK"}
_UNRESOLVED = object()


def scan_globals(data: bytes) -> tuple[set[tuple[str, str]], list[str]]:
    """Static scan. Returns (resolved (module, name) pairs, problems). Never unpickles."""
    found: set[tuple[str, str]] = set(); problems: list[str] = []
    memo: dict[int, object] = {}; pushed: list[object] = []
    for op, arg, pos in pickletools.genops(data):
        n = op.name
        if n in _STR_OPS: pushed.append(arg if isinstance(arg, str) else arg.decode("latin-1") if isinstance(arg, bytes) else _UNRESOLVED)
        elif n in ("BINGET", "LONG_BINGET", "GET"): pushed.append(memo.get(int(arg), _UNRESOLVED))
        elif n == "MEMOIZE": memo[len(memo)] = pushed[-1] if pushed else _UNRESOLVED
        elif n in ("PUT", "BINPUT", "LONG_BINPUT"): memo[int(arg)] = pushed[-1] if pushed else _UNRESOLVED
        elif n in ("GLOBAL", "INST"):
            mod, _, name = str(arg).partition(" "); found.add((mod, name)); pushed.append(_UNRESOLVED)
        elif n == "STACK_GLOBAL":
            if len(pushed) >= 2 and isinstance(pushed[-2], str) and isinstance(pushed[-1], str):
                found.add((pushed[-2], pushed[-1]))
            else:
                problems.append(f"STACK_GLOBAL at byte {pos} has operands that are not literal strings; cannot verify")
            pushed.append(_UNRESOLVED)
        elif n not in _NOPUSH: pushed.append(_UNRESOLVED)
    return found, problems


def check(data: bytes, allowed: frozenset = ALLOWED) -> set[tuple[str, str]]:
    """Gate 1. Raises UnsafePickle unless every referenced global is resolvable and allowlisted."""
    found, problems = scan_globals(data)
    bad = sorted(g for g in found if g not in allowed)
    if problems or bad:
        raise UnsafePickle(f"refusing to load: disallowed globals {bad}; unresolved {problems}")
    return found


class RestrictedUnpickler(pickle.Unpickler):
    """Gate 2. find_class refuses anything outside the exact-pair allowlist."""
    allowed: frozenset = ALLOWED

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise UnsafePickle(f"find_class refused {module}.{name}")
        return super().find_class(module, name)


def safe_loads(data: bytes, allowed: frozenset = ALLOWED):
    check(data, allowed)
    u = RestrictedUnpickler(io.BytesIO(data)); u.allowed = allowed
    return u.load()
