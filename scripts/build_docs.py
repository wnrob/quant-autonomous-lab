"""Generates REPOSITORY_MANIFEST.json, CAPABILITY_MATRIX.md, LICENSE_AUDIT.md, DEPENDENCY_COMPATIBILITY.md
from artifacts/{acquisition_report,audit_raw,compat_probe}.json + scripts/curated.py. Read-only on clones."""
from __future__ import annotations
import json, sys
from datetime import date, datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from curated import CAPS, CUR, LICENSE_OVERRIDES
from repos import REPOS
ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "artifacts"
aud = json.loads((A/"audit_raw.json").read_text()); acq = {r["id"]: r for r in json.loads((A/"acquisition_report.json").read_text())}
probe = json.loads((A/"compat_probe.json").read_text()) if (A/"compat_probe.json").exists() else {}
from adapters.registry import REGISTRY
reg = {r.repo: r for r in REGISTRY}
TODAY = date(2026, 11, 11)

def maint(d):
    age = (TODAY - date.fromisoformat(d[:10])).days
    return ("active" if age < 120 else "slow" if age < 540 else "stale"), age
def commercial(spdx):
    if spdx.startswith(("MIT","Apache","BSD","CC0")): return "permitted with attribution/notice"
    if spdx.startswith("LGPL"): return "permitted if dynamically linked; modifications to the library must be shared"
    if spdx.startswith("GPL") or spdx.startswith("AGPL"): return "RESTRICTED: copyleft obligations on distribution/derivatives; get legal review"
    return "NOT GRANTED: no license => all rights reserved; do not copy/redistribute"
def pychk(rid, path):
    d = ROOT/acq[rid]["path"]; return (d/path).exists()

manifest = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "schema_version": 1,
            "note": "capability scores and roles are analyst judgements from README/static inspection; only fields under 'verified' are measured.", "repositories": []}
for rid, group, url, fb in REPOS:
    a, c, cu = aud[rid], CUR[rid], CUR[rid]
    spdx = LICENSE_OVERRIDES.get(rid, a["license_spdx_guess"])
    m, age = maint(a["last_commit_date"]); pr = probe.get(rid)
    adapter = reg[rid]
    manifest["repositories"].append({
      "id": rid, "group": group, "requested_url": url,
      "acquisition": {"status": acq[rid]["status"], "used_url": acq[rid]["used_url"], "branch": a["branch"], "head_sha": a["head_sha"], "size_mb": a["size_mb"], "shallow": True},
      "license": {"spdx": spdx, "file": a["license_file"], "commercial_use": commercial(spdx)},
      "maintenance": {"last_commit": a["last_commit_date"], "age_days": age, "status": m},
      "languages": cu["langs"], "file_counts": a["languages_file_counts"],
      "python": {"requires": a.get("requires_python"), "dependencies_sample": (a.get("dependencies") or [])[:15]},
      "summary": a["readme_summary"], "import_example": cu["imp"],
      "capabilities": {k: v for k, v in cu["caps"].items()}, "overlaps": cu["overlaps"], "limitations": cu["limits"],
      "static_risk_flags": a["risk_flags"] + ([f"setup.py flags: {a['setup_py_flags']}"] if a.get("setup_py_flags") else []),
      "integration": {"role": cu["role"], "suitable_for": cu["suit"], "policy": cu["policy"], "adapter_status": adapter.status, "env": adapter.env},
      "verified": {"api_source_path_exists": pychk(rid, cu["path"]), "pypi_install_import_py312": (None if not pr else {"install_ok": pr["install_ok"], "import_ok": pr.get("import_ok"), "version": pr.get("version"), "error": (pr.get("install_error") or pr.get("import_error") or "")[-240:] or None})}})
(ROOT/"REPOSITORY_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

# CAPABILITY MATRIX
ids = [r[0] for r in REPOS]
L = ["# Capability Matrix", "", "Scores: **3** primary strength · **2** solid · **1** minor · blank none. These are analyst judgements from READMEs and static inspection (see `scripts/curated.py`), **not benchmarks**.", "",
     "| capability | best-supported (3) | also solid (2) | minor (1) |", "|---|---|---|---|"]
for cap in CAPS:
    g = lambda s: ", ".join(i for i in ids if CUR[i]["caps"].get(cap, 0) == s) or "-"
    L.append(f"| {cap.replace('_',' ')} | {g(3)} | {g(2)} | {g(1)} |")
L += ["", "## Full grid", "", "| repo | " + " | ".join(c.replace('_', ' ') for c in CAPS) + " |", "|---|" + "---|" * len(CAPS)]
for i in ids: L.append(f"| {i} | " + " | ".join(str(CUR[i]["caps"].get(c, "")) for c in CAPS) + " |")
L += ["", "## Capabilities with NO strong (>=3) coverage in this set", ""] + [f"- {c.replace('_',' ')}" for c in CAPS if not any(CUR[i]['caps'].get(c,0)>=3 for i in ids)]
L += ["", "## Overlap clusters (pick ONE per cluster for the core)", "",
      "- **Portfolio optimisation**: PyPortfolioOpt, skfolio, Riskfolio-Lib (+ bt). Core uses own scipy optimiser; PyPortfolioOpt adapter operational; skfolio preferred next.",
      "- **Backtesting**: vectorbt (vectorised sweeps), bt (allocation), zipline-reloaded / backtrader (event-driven), nautilus_trader / Lean (institutional). Core uses own ~40-line vectorised engine.",
      "- **Performance analytics**: quantstats, QuantInvestStrats, vectorbt stats, alphalens (factor-level).",
      "- **Options/derivatives**: py_vollib (BS/Greeks/IV, tiny) vs QuantLib (full curve/exotics).",
      "- **RL**: FinRL vs FinRL-Trading. **ML alpha**: qlib (+ RD-Agent on top). **Data**: OpenBB vs yfinance (core).",
      "- **Directories**: awesome-quant / awesome-systematic-trading / awesome-open-source-trading (mostly overlapping link lists)."]
(ROOT/"CAPABILITY_MATRIX.md").write_text("\n".join(L) + "\n")

# LICENSE AUDIT
L = ["# License Audit", "", "Detected from each repo's LICENSE file text (then manually checked for ambiguous cases). **Not legal advice.**", "",
     "| repo | license | commercial use | notes |", "|---|---|---|---|"]
for r in manifest["repositories"]:
    note = {"backtrader": "GPL-3.0 strong copyleft + unmaintained -> reference only", "nautilus_trader": "LGPL-3.0: keep as separate installed dependency, do not vendor/modify",
            "awesome-quant": "No LICENSE file -> all rights reserved; read-only", "awesome-systematic-trading": "No LICENSE file -> all rights reserved; read-only",
            "Lean": "Apache-2.0 code; QuantConnect data/cloud services have separate terms", "OpenBB": "Apache-2.0 code; data providers have their own terms/keys",
            "arbitragelab": "BSD code; Hudson & Thames commercial docs/content separate", "QuantLib": "BSD-style; very many copyright holders"}.get(r["id"], "")
    L.append(f"| {r['id']} | {r['license']['spdx']} | {r['license']['commercial_use']} | {note} |")
L += ["", "## Summary", "", "- Permissive (MIT/Apache/BSD/CC0): " + ", ".join(r['id'] for r in manifest['repositories'] if r['license']['spdx'].startswith(('MIT','Apache','BSD','CC0'))),
      "- Weak copyleft: nautilus_trader (LGPL-3.0)", "- Strong copyleft: backtrader (GPL-3.0)", "- Unlicensed: awesome-quant, awesome-systematic-trading",
      "", "## Policy applied", "", "- External clones live under `external/` and are never copied into `engine/` or `adapters/`; the integration layer only `import`s released packages lazily.",
      "- Third-party data (Yahoo via yfinance) is subject to Yahoo's terms: personal/research use; check before commercial use."]
(ROOT/"LICENSE_AUDIT.md").write_text("\n".join(L) + "\n")

# DEPENDENCY COMPATIBILITY
L = ["# Dependency Compatibility", "", f"Host: macOS arm64, system Python 3.14. Probe: each library's **published PyPI wheel** installed with `--only-binary :all:` into a fresh isolated Python 3.12 venv (`scripts/probe_envs.py`), then `import` attempted. Clone source was **not** installed or executed. Results in `artifacts/compat_probe.json`.", "",
     "'import ok' means only that the package imports - **not** that an adapter works. Operational status requires a passing test (see IMPLEMENTATION_STATUS.md).", "",
     "| repo | declared python | PyPI py3.12 install | import | version/error | adapter status |", "|---|---|---|---|---|---|"]
for r in manifest["repositories"]:
    p = r["verified"]["pypi_install_ok"] if False else r["verified"]["pypi_install_import_py312"]
    if p is None: inst, imp, info = "not probed", "-", "no PyPI probe (" + ("directory" if r["group"] == "directories" else "see notes") + ")"
    else: inst, imp, info = ("yes" if p["install_ok"] else "NO"), ("yes" if p["import_ok"] else ("NO" if p["install_ok"] else "-")), (p["version"] or (p["error"] or "")[:110].replace("\n", " ").replace("|", "/"))
    L.append(f"| {r['id']} | {r['python']['requires'] or 'n/a'} | {inst} | {imp} | {info} | {r['integration']['adapter_status']} |")
L += ["", "## Environment strategy", "", "- `envs/core` (Python 3.12): numpy/pandas/scipy/pyarrow/matplotlib + yfinance, PyPortfolioOpt, quantstats. The only env the CLI needs.",
      "- One venv **per heavy library** (`envs/<name>`), never a shared global install. Conflicts seen in practice: numba/llvmlite pins (vectorbt), cvxpy solver stacks (portfolio libs), torch/gym pins (FinRL), old pins (arbitragelab, backtrader), `python_requires<3.13` (py_vollib).",
      "- System Python 3.14 is **too new** for several libs (numba-family, zipline, qlib classifiers): use 3.11/3.12 envs.",
      "- Not buildable here by design: Lean (needs .NET/Docker), QuantLib C++ source, nautilus_trader Rust source: use released wheels."]
(ROOT/"DEPENDENCY_COMPATIBILITY.md").write_text("\n".join(L) + "\n")
print("ok", sum(1 for r in manifest["repositories"] if r["verified"]["api_source_path_exists"]), "/24 api paths exist")
