"""Sandbox-side runner. Executed BY PATH with `python -I` inside scripts/coding_sandbox_run.sh; cwd = workspace.

It makes the WORKSPACE copy of engine/adapters importable ahead of the editable install of LAB, refuses to continue if
`engine` resolved anywhere else, then (a) runs pytest collecting a per-test outcome map and (b) computes the golden probes.
Everything it writes is produced by untrusted code and is treated as such by the host (see workflow.py).
"""
from __future__ import annotations
import importlib.util, json, os, sys, time

def main(argv: list[str]) -> int:
    ws, out_path, golden_path = os.path.realpath(argv[0]), argv[1], argv[2]
    pytest_args = argv[3:]
    sys.path.insert(0, ws); os.chdir(ws)
    res: dict = {"ws": ws, "started": time.time(), "tests": {}, "collect_errors": [], "golden": None, "runner_error": None}
    try:
        import engine
        ef = os.path.realpath(engine.__file__)
        res["engine_file"] = ef
        if not ef.startswith(ws + os.sep): raise RuntimeError(f"engine resolved outside the workspace: {ef}")
        import pytest

        class Collector:
            def pytest_runtest_logreport(self, report):
                cur = res["tests"].get(report.nodeid)
                if report.when == "call":
                    new = "passed" if report.passed else "failed" if report.failed else "skipped"
                elif report.failed: new = "error"
                elif report.skipped and report.when == "setup": new = "skipped"
                else: new = cur or None
                if new and cur not in ("failed", "error"): res["tests"][report.nodeid] = new
            def pytest_collectreport(self, report):
                if report.failed: res["collect_errors"].append({"nodeid": report.nodeid, "repr": str(report.longrepr)[-600:]})

        res["pytest_exit"] = int(pytest.main(["-p", "no:cacheprovider", "-q", "--no-header", "--tb=line", *pytest_args], plugins=[Collector()]))
        spec = importlib.util.spec_from_file_location("_coding_golden_trusted", golden_path)
        g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
        res["golden"] = g.compute()
    except BaseException as e:                                          # noqa: BLE001
        res["runner_error"] = f"{type(e).__name__}: {str(e)[:400]}"
    res["finished"] = time.time()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f: json.dump(res, f)
    return 0 if res["runner_error"] is None else 3

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
