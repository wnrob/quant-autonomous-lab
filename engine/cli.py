"""`qal` command line interface."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from engine.errors import QALError
from engine.logging_setup import setup

DEFAULT_SYMBOLS = ["SPY", "TLT", "GLD"]
ROOT = Path(__file__).resolve().parents[1]

def _provider(a):
    from engine.data.ingest import CSVProvider, SyntheticProvider, YahooProvider
    return {"yahoo": YahooProvider, "synthetic": SyntheticProvider}[a.provider]() if a.provider != "csv" else CSVProvider(a.csv)

def _common(p):
    p.add_argument("--provider", choices=["yahoo", "synthetic", "csv"], default="yahoo")
    p.add_argument("--csv", help="path for --provider csv")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default="2005-01-01"); p.add_argument("--end", default=None)
    p.add_argument("--offline", action="store_true", help="use cache only"); p.add_argument("--refresh", action="store_true")

def cmd_ingest(a):
    from engine.data.ingest import load_prices
    df, m = load_prices(_provider(a), a.symbols, a.start, a.end, offline=a.offline, refresh=a.refresh)
    print(json.dumps(m.__dict__, indent=2)); print(df.tail(3).to_string())

def cmd_demo(a):
    from engine.orchestration.pipeline import run_demo
    p = run_demo(_provider(a), a.symbols, a.start, a.end, hyp_asset=a.hyp_asset, cost_bps=a.cost_bps, offline=a.offline, refresh=a.refresh)
    print(f"report: {p}"); print(p.read_text())

def cmd_adapters(a):
    from adapters.registry import REGISTRY, health_check
    for r in REGISTRY:
        h = health_check(r) if a.check and r.status == "operational" else ""
        print(f"{r.status:12} {r.repo:30} {r.role:28} {h}")

def cmd_repos(a):
    m = json.loads((ROOT / "REPOSITORY_MANIFEST.json").read_text())
    for r in m["repositories"]:
        print(f'{r["id"]:30} {r["license"]["spdx"]:12} {r["acquisition"]["head_sha"][:10]}  {r["integration"]["role"]}')

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("qal", description="Quant Autonomous Lab"); ap.add_argument("--log", default="INFO")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("ingest", help="fetch/cache prices"); _common(p); p.set_defaults(f=cmd_ingest)
    p = sp.add_parser("demo", help="run all six demos and write a report"); _common(p)
    p.add_argument("--hyp-asset", default=None); p.add_argument("--cost-bps", type=float, default=2.0); p.set_defaults(f=cmd_demo)
    p = sp.add_parser("adapters", help="list adapter status"); p.add_argument("--check", action="store_true"); p.set_defaults(f=cmd_adapters)
    p = sp.add_parser("repos", help="list acquired repositories"); p.set_defaults(f=cmd_repos)
    a = ap.parse_args(argv); setup(a.log)
    try: a.f(a); return 0
    except QALError as e: print(f"error: {e}", file=sys.stderr); return 2

if __name__ == "__main__": sys.exit(main())
