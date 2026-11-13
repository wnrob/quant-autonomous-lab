import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_demo_end_to_end_synthetic(tmp_path):
    from engine.orchestration.pipeline import run_demo
    from engine.data.ingest import SyntheticProvider, load_prices
    import engine.data.ingest as ing
    ing.DATASETS = tmp_path
    p = run_demo(SyntheticProvider(), ["A", "B", "C"], "2000-01-03", "2020-12-31", outdir=tmp_path / "run")
    txt = p.read_text()
    assert "SYNTHETIC TEST DATA" in txt                      # banner must propagate
    assert (tmp_path / "run/equity.png").exists()
    res = json.loads((tmp_path / "run/results.json").read_text()); assert res["dataset"]["kind"] == "synthetic"

def test_cli_help_and_adapters():
    for args in (["--help"], ["adapters"]):
        r = subprocess.run([sys.executable, "-m", "engine.cli", *args], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
