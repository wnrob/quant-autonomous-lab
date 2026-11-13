"""Read-only view builders for the dashboard. They read STORED state only:
  - SQLite store tables: experiments, strategies, lessons, hypotheses, test_access, ml_models (if the table exists; never created here)
  - JSON files under <artifacts>/ml/<run>/results.json and <artifacts>/portfolio/<run>/results.json
Nothing is computed for display: values are passed through as stored (NaN/inf become null because JSON cannot carry them).
Run names are never joined into paths unless they are an exact match of a directory listed under the artifact root.
No pickle/joblib is ever opened. No function here writes to the store or to disk.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from engine.memory.store import Store
from engine.research import validation as V

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
NO_DATA = "no_data"


class UnknownRun(LookupError):
    """Requested run name is not one of the run directories that exist under the artifact root."""


def clean(x: Any) -> Any:
    """Recursively replace NaN/inf with None. Nothing else changes."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def no_data(detail: str, **extra: Any) -> dict:
    return {"status": NO_DATA, "detail": detail, **extra}


def _table_exists(s: Store, name: str) -> bool:
    return bool(s.q("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", name))


def _list_runs(base: Path, want_file: str) -> list[str]:
    """Run names = real (non-symlink) subdirectories of base holding a regular file `want_file`. Sorted; newest last."""
    if not base.is_dir():
        return []
    out = []
    for p in base.iterdir():
        if p.is_symlink() or not p.is_dir() or not RUN_NAME.match(p.name):
            continue
        f = p / want_file
        if f.is_file() and not f.is_symlink():
            out.append(p.name)
    return sorted(out)


def _select_run(base: Path, want_file: str, requested: str | None) -> tuple[str | None, Path | None]:
    """The requested name must EXACTLY equal a listed run; it is never used to build a path itself."""
    runs = _list_runs(base, want_file)
    if requested is None:
        return (runs[-1], base / runs[-1] / want_file) if runs else (None, None)
    if requested not in runs:
        raise UnknownRun(requested)
    return requested, base / requested / want_file


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Risk Center
def protected_test_status(s: Store, criteria: V.Criteria | None = None) -> dict:
    crit = criteria or V.Criteria()
    used = s.test_looks()
    return {"looks_used": used, "max_test_looks": crit.max_test_looks, "remaining": max(crit.max_test_looks - used, 0),
            "source": "store table test_access (count); limit = engine.research.validation.Criteria().max_test_looks"}


def risk_view(s: Store, criteria: V.Criteria | None = None) -> dict:
    crit = criteria or V.Criteria()
    protected = protected_test_status(s, crit)
    limits = {"max_drawdown": crit.risk.max_drawdown, "max_leverage": crit.risk.max_leverage,
              "max_annual_turnover": crit.risk.max_annual_turnover, "source": "engine.research.validation.RiskLimits (code constants)"}
    rows = s.q("SELECT e.id, e.decision, e.reasons, e.metrics, s.name, s.family FROM experiments e JOIN strategies s ON s.id=e.strategy_id ORDER BY e.id")
    if not rows:
        return no_data("No experiments in the store yet, so there are no risk-gate results to show.", protected_test=protected, risk_limits=limits)
    exps = []
    for r in rows:
        m = json.loads(r["metrics"] or "{}")
        reasons = json.loads(r["reasons"] or "[]")
        v = m.get("validation") or {}
        t = m.get("test")
        exps.append({
            "id": r["id"], "name": r["name"], "family": r["family"], "decision": r["decision"],
            "validation": {"max_drawdown": v.get("max_drawdown"), "cvar_95": v.get("cvar_95"), "sharpe": v.get("sharpe")},
            "test": ({"max_drawdown": t.get("max_drawdown"), "cvar_95": t.get("cvar_95"), "sharpe": t.get("sharpe")} if t else None),
            "turnover_annual": m.get("turnover_annual"),
            "risk_violations": [x for x in reasons if str(x).startswith("RISK")],
        })
    return clean({"status": "ok", "protected_test": protected, "risk_limits": limits, "experiments": exps})


# ---------------------------------------------------------------- Model Lab
ML_RUN_FIELDS = ("decision", "reasons", "n_trials", "dsr", "val_active_ann", "val_active_sharpe", "tracking_error",
                 "val_sharpe", "bench_sharpe", "max_drawdown", "turnover_annual", "avg_names", "rank_ic")


def _registry_rows(s: Store) -> list[dict]:
    if not _table_exists(s, "ml_models"):
        return []
    rows = s.q("SELECT id, name, family, seed, status, data_sha, code_version, created, params, metrics, parent_id, artifact_path FROM ml_models ORDER BY id")
    for r in rows:
        r["params"] = json.loads(r["params"] or "{}")
        r["metrics"] = json.loads(r["metrics"] or "{}")
        apath = r.pop("artifact_path")
        r["artifact_file"] = Path(apath).name if apath else None  # basename only; no directory leaks to the page
    return rows


def models_view(s: Store, artifacts: Path, run: str | None = None) -> dict:
    rows = _registry_rows(s)
    registry = {"status": "ok", "rows": rows} if rows else no_data("No rows in the ml_models table yet.")
    name, path = _select_run(artifacts / "ml", "results.json", run)
    if path is None:
        ml = no_data(f"No ML run under {artifacts / 'ml'} (expected <run>/results.json).", runs_available=[])
    else:
        d = _read_json(path)
        runs = []
        for k, v in (d.get("runs") or {}).items():
            row = {"name": v.get("name", k)}
            row.update({f: v.get(f) for f in ML_RUN_FIELDS})
            runs.append(row)
        ml = {"status": "ok", "run": name, "generated": d.get("generated"), "verdict": d.get("verdict"),
              "protected_test_consulted": d.get("protected_test_consulted"), "n_trials": d.get("n_trials"),
              "validation_passes": d.get("validation_passes"), "criteria_hash": d.get("criteria_hash"), "data": d.get("data"),
              "seed_stability": d.get("seed_stability"), "shuffled_label_controls": d.get("shuffled_label_controls"),
              "runs": runs, "runs_available": _list_runs(artifacts / "ml", "results.json")}
    if registry["status"] != "ok" and ml["status"] != "ok":
        return no_data("No model registry rows and no ML run results stored yet.", registry=registry, ml_run=ml)
    return clean({"status": "ok", "registry": registry, "ml_run": ml})


# ---------------------------------------------------------------- Portfolio Intelligence
PORTFOLIO_KEYS = ("label", "generated", "cost_bps", "n_trials", "table", "verdict_vs_equal_weight", "selected", "summary", "skfolio_used", "meta")


def portfolio_view(artifacts: Path, run: str | None = None) -> dict:
    name, path = _select_run(artifacts / "portfolio", "results.json", run)
    avail = _list_runs(artifacts / "portfolio", "results.json")
    if path is None:
        return no_data(f"No portfolio run under {artifacts / 'portfolio'} (expected <run>/results.json).", runs_available=[])
    d = _read_json(path)
    out = {"status": "ok", "run": name, "runs_available": avail}
    out.update({k: d.get(k) for k in PORTFOLIO_KEYS})
    return clean(out)


# ---------------------------------------------------------------- Research Lab
def research_view(s: Store) -> dict:
    exp_total = s.q("SELECT COUNT(*) n FROM experiments")[0]["n"]
    lessons = s.q("SELECT id, ts, experiment_id, family, kind, text FROM lessons ORDER BY id")
    hyps = s.q("SELECT id, created, text, rationale, source, parent_experiment FROM hypotheses ORDER BY id")
    if not (exp_total or lessons or hyps):
        return no_data("No experiments, lessons or hypotheses in the store yet.")
    return {
        "status": "ok",
        "experiments_total": exp_total,
        "decisions": s.q("SELECT decision, COUNT(*) n FROM experiments GROUP BY decision ORDER BY n DESC, decision"),
        "by_family": s.q("SELECT s.family AS family, e.decision AS decision, COUNT(*) AS n FROM experiments e JOIN strategies s ON s.id=e.strategy_id "
                         "GROUP BY s.family, e.decision ORDER BY s.family, e.decision"),
        "strategies_by_status": s.q("SELECT status, COUNT(*) n FROM strategies GROUP BY status ORDER BY n DESC"),
        "lesson_kinds": s.q("SELECT kind, COUNT(*) n FROM lessons GROUP BY kind ORDER BY n DESC"),
        "lessons": lessons,
        "hypotheses": hyps,
        "hypothesis_sources": s.q("SELECT source, COUNT(*) n FROM hypotheses GROUP BY source ORDER BY n DESC"),
        "protected_test": protected_test_status(s),
    }
