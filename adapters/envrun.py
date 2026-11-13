"""Run a worker script inside an isolated per-library venv (envs/<name>). Core never imports heavy libs; data crosses via files/JSON."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
from engine.errors import AdapterUnavailable
ROOT = Path(__file__).resolve().parents[1]

def run_in_env(env: str, worker: str, payload: dict, timeout: int = 600) -> dict:
    py = ROOT / "envs" / env / "bin" / "python"
    if not py.exists(): raise AdapterUnavailable(f"env '{env}' not built (see scripts/sandbox_build.sh)")
    r = subprocess.run([str(py), "-I", "-W", "ignore", str(ROOT / "adapters/workers" / worker), json.dumps(payload)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0: raise AdapterUnavailable(f"{worker} failed in {env}: {r.stderr[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])
