"""Static audit of cloned repos. READ-ONLY: never imports or executes third-party code."""
from __future__ import annotations
import json, re, subprocess, sys, tomllib
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
LANG = {".py":"Python",".cs":"C#",".cpp":"C++",".hpp":"C++",".h":"C/C++",".c":"C",".rs":"Rust",".pyx":"Cython",".ts":"TypeScript",".js":"JavaScript",".ipynb":"Jupyter",".md":"Markdown",".java":"Java",".go":"Go",".r":"R"}
SKIP = {".git","node_modules"}

def license_of(d: Path):
    for n in ("LICENSE","LICENSE.md","LICENSE.txt","LICENCE","COPYING","LICENSE.rst"):
        p = d/n
        if p.exists():
            t = p.read_text(errors="ignore"); tl = t.lower()
            for key, spdx in [("gnu affero","AGPL-3.0"),("gnu lesser general","LGPL"),("gnu general public license","GPL"),("apache license","Apache-2.0"),
                              ("mit license","MIT"),("permission is hereby granted, free of charge","MIT"),("bsd","BSD"),("mozilla public","MPL-2.0"),
                              ("creative commons","CC"),("business source","BUSL"),("elastic license","Elastic")]:
                if key in tl:
                    if spdx=="GPL" and "version 3" in tl: spdx="GPL-3.0"
                    return spdx, n
            return "UNKNOWN", n
    return "NONE_FOUND", None

def pyproject(d: Path):
    out = {}
    p = d/"pyproject.toml"
    if p.exists():
        try:
            t = tomllib.loads(p.read_text(errors="ignore"))
            pr = t.get("project", {})
            poet = t.get("tool",{}).get("poetry",{})
            out["requires_python"] = pr.get("requires-python") or (poet.get("dependencies",{}) or {}).get("python")
            deps = pr.get("dependencies") or list((poet.get("dependencies") or {}).keys())
            out["dependencies"] = [str(x) for x in deps][:60]
            out["build_backend"] = t.get("build-system",{}).get("build-backend")
            out["license_field"] = str(pr.get("license") or poet.get("license") or "")[:80] or None
        except Exception as e: out["pyproject_error"]=str(e)[:100]
    s = d/"setup.py"
    if s.exists():
        txt = s.read_text(errors="ignore")
        m = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)", txt)
        if m and not out.get("requires_python"): out["requires_python"]=m.group(1)
        out["has_setup_py"]=True
        out["setup_py_flags"]=[k for k in ("cmdclass","subprocess","os.system","urllib","requests","exec(","eval(","Cython","numpy.get_include") if k in txt]
    r = d/"requirements.txt"
    if r.exists() and not out.get("dependencies"):
        out["dependencies"]=[l.strip() for l in r.read_text(errors="ignore").splitlines() if l.strip() and not l.startswith("#")][:60]
    return out

def langs(d: Path):
    c = Counter()
    for p in d.rglob("*"):
        if any(s in p.parts for s in SKIP) or not p.is_file(): continue
        l = LANG.get(p.suffix.lower())
        if l and l not in ("Markdown",): c[l]+=1
    return dict(c.most_common(6))

def risk_flags(d: Path):
    flags=[]
    pat = re.compile(r"(curl|wget)[^\n|]{0,120}\|\s*(ba)?sh")
    for p in list(d.glob("*.sh"))+list(d.glob("scripts/*.sh"))+list(d.glob("*.md"))[:5]+list(d.glob("docker/*"))[:5]:
        try:
            if p.is_file() and pat.search(p.read_text(errors="ignore")): flags.append(f"pipe-to-shell in {p.relative_to(d)}")
        except Exception: pass
    for n in ("install.sh","setup.sh","build.sh","Makefile","Dockerfile"):
        if (d/n).exists(): flags.append(f"build/install script present: {n} (not executed)")
    txt = ""
    for n in ("README.md","readme.md"):
        if (d/n).exists(): txt=(d/n).read_text(errors="ignore").lower(); break
    for k,msg in [("api key","needs API keys"),("openai","calls external LLM APIs"),("broker","has broker/live-trading connectors"),("live trading","live trading features"),("docker","docker-oriented")]:
        if k in txt: flags.append(msg)
    return sorted(set(flags))

def readme_head(d: Path):
    for n in ("README.md","readme.md","README.rst","README"):
        if (d/n).exists():
            t=(d/n).read_text(errors="ignore")
            t=re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)|\[!\[.*?\]\(.*?\)\]\(.*?\)","",t)
            lines=[l.strip() for l in t.splitlines() if len(l.strip())>40 and not l.strip().startswith(("#","|","[","`","-","*",">"))]
            return " ".join(lines[:2])[:400]
    return ""

def git(d,*a): return subprocess.run(["git","-c","core.hooksPath=/dev/null",*a],cwd=d,capture_output=True,text=True).stdout.strip()

if __name__=="__main__":
    acq = {r["id"]:r for r in json.loads((ROOT/"artifacts/acquisition_report.json").read_text())}
    out={}
    for rid,r in acq.items():
        d = ROOT/r["path"]
        lic,licfile = license_of(d)
        py = pyproject(d)
        out[rid]={**{k:r.get(k) for k in ("group","requested_url","used_url","status","head_sha","branch","size_mb")},
                  "license_spdx_guess":lic,"license_file":licfile,"last_commit_date":git(d,"log","-1","--format=%cI"),
                  "languages_file_counts":langs(d),"readme_summary":readme_head(d),**py,"risk_flags":risk_flags(d)}
        print(rid,lic,out[rid]["last_commit_date"][:10],out[rid].get("requires_python"),sep=" | ")
    (ROOT/"artifacts/audit_raw.json").write_text(json.dumps(out,indent=2))
