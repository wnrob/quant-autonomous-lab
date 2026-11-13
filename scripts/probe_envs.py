"""Compatibility probe: one isolated venv per library, wheels only (no sdist build scripts),
install from PyPI (not from the cloned source), then try the import. Results -> artifacts/compat_probe.json"""
from __future__ import annotations
import json, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PROBES = [  # repo id, pip spec, import name
 ("skfolio","skfolio","skfolio"),("Riskfolio-Lib","riskfolio-lib","riskfolio"),("vectorbt","vectorbt","vectorbt"),
 ("bt","bt","bt"),("py_vollib","py_vollib","py_vollib"),("quantstats","quantstats","quantstats"),
 ("PyPortfolioOpt","pyportfolioopt","pypfopt"),("alphalens-reloaded","alphalens-reloaded","alphalens"),
 ("zipline-reloaded","zipline-reloaded","zipline"),("backtrader","backtrader","backtrader"),
 ("qlib","pyqlib","qlib"),("nautilus_trader","nautilus_trader","nautilus_trader"),("QuantLib","QuantLib","QuantLib"),
 ("RD-Agent","rdagent","rdagent"),("OpenBB","openbb","openbb"),("arbitragelab","arbitragelab","arbitragelab"),
 ("QuantInvestStrats","qis","qis"),("FinRL","finrl","finrl"),
]
PY = "3.12"
def sh(cmd, timeout):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout); return r.returncode, (r.stdout+r.stderr)[-600:]
    except subprocess.TimeoutExpired: return 124, "timeout"
if __name__ == "__main__":
    only = set(sys.argv[1:]); res = {}
    p = ROOT/"artifacts/compat_probe.json"
    if p.exists(): res = json.loads(p.read_text())
    base = ROOT/"envs"/"probe"
    for rid, spec, mod in PROBES:
        if only and rid not in only: continue
        d = base/rid; t = time.time()
        shutil.rmtree(d, ignore_errors=True)
        sh(["uv","venv","--python",PY,str(d),"-q"], 120)
        rc, out = sh(["uv","pip","install","-q","--python",str(d/"bin/python"),"--only-binary",":all:",spec], 900)
        rec = {"pip_spec":spec,"python":PY,"install_ok":rc==0,"install_secs":round(time.time()-t,1)}
        if rc != 0: rec["install_error"]=out
        else:
            rc2, out2 = sh([str(d/"bin/python"),"-I","-c",f"import {mod} as m; print(getattr(m,'__version__','?'))"], 180)
            rec["import_ok"]=rc2==0; rec["version" if rc2==0 else "import_error"]=out2.strip()[-300:]
        res[rid]=rec; p.write_text(json.dumps(res,indent=2)); print(rid, rec.get("install_ok"), rec.get("import_ok"), flush=True)
        shutil.rmtree(d, ignore_errors=True)  # keep disk small; result is what matters
