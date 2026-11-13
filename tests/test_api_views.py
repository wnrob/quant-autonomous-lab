"""Dashboard view endpoints (GET only, stored state). All data here is SYNTHETIC and lives in pytest tmp dirs.
Nothing reads artifacts/qae/ or the real artifact tree. Fixtures are labelled 'SYNTHETIC' in their verdict/label fields."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.api.app import create_app
from engine.memory.store import Store
from engine.models.registry import Registry
from engine.research import validation as V

NEW_GET = ["/api/risk", "/api/models", "/api/portfolio", "/api/research"]
ML_RUN = "20990101-000000"
PF_RUN = "20990101T000000"


@pytest.fixture
def art(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


@pytest.fixture
def empty_client(tmp_path, art):
    return TestClient(create_app(Store(tmp_path / "empty.db"), artifacts_root=art))


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=True), encoding="utf-8")


@pytest.fixture
def populated(tmp_path, art):
    """Tmp store + tmp artifacts with SYNTHETIC values. Returns (store, client, expected) where expected holds the exact stored numbers."""
    s = Store(tmp_path / "pop.db")
    sid, _ = s.get_or_create_strategy({"family": "SYN_family", "k": 1}, "SYN_strategy_A")
    ok_metrics = {"validation": {"sharpe": 0.5, "max_drawdown": -0.1234, "cvar_95": 0.0456, "dsr": 0.31},
                  "validation_benchmark": {"sharpe": 0.4}, "turnover_annual": 7.5,
                  "test": {"sharpe": 0.2, "max_drawdown": -0.2222, "cvar_95": 0.0333}}
    s.add_experiment(strategy_id=sid, hypothesis_id=None, spec_hash="sh1", criteria_hash="ch", data_sha="d", code_version="syn", seed=0,
                     secs=1.0, metrics=ok_metrics, decision="TEST_FAIL",
                     reasons=["test Sharpe degraded > 50% vs validation", "RISK: max_drawdown -35.0% < limit -30%"], report_path=None)
    sid2, _ = s.get_or_create_strategy({"family": "SYN_family", "k": 2}, "SYN_strategy_B")
    s.add_experiment(strategy_id=sid2, hypothesis_id=None, spec_hash="sh2", criteria_hash="ch", data_sha="d", code_version="syn", seed=0,
                     secs=1.0, metrics={"validation": {"sharpe": -0.1, "max_drawdown": -0.05, "cvar_95": 0.01, "dsr": 0.0}, "turnover_annual": 1.0},
                     decision="VALIDATION_FAIL", reasons=["DSR 0.00 < 0.9 (trials=2)"], report_path=None)
    hid = s.add_hypothesis("SYN hypothesis text", "SYN rationale", source="rule_based")
    s.add_lesson(1, "SYN_family", "negative", "SYN_strategy_A: TEST_FAIL. negative lesson text")
    s.record_test_access("sh1", "SYN_family", "d")
    s.record_test_access("sh2", "SYN_family", "d")

    reg = Registry(s, artifact_dir=art / "models")
    reg.register("SYN_model", "SYN_fam", {"depth": 3}, "fh", "d", "syn", 7, metrics={"dsr": 0.123, "synthetic": True})

    _write_json(art / "ml" / ML_RUN / "results.json", {
        "generated": "1999-01-01T00:00:00", "verdict": "SYNTHETIC FIXTURE: NULL", "protected_test_consulted": False, "n_trials": 2,
        "validation_passes": [], "criteria_hash": "syn", "data": {"kind": "synthetic"},
        "runs": {"syn-run-a": {"name": "syn-run-a", "decision": "VALIDATION_FAIL", "reasons": ["SYN reason"], "n_trials": 2, "dsr": 0.4242,
                               "val_active_ann": 0.0123, "val_active_sharpe": 0.31, "tracking_error": 0.05, "max_drawdown": -0.1,
                               "turnover_annual": 3.25, "rank_ic": {"mean": 0.02, "note": "diagnostic only"}},
                 "syn-run-nan": {"name": "syn-run-nan", "decision": "VALIDATION_FAIL", "reasons": [], "dsr": float("nan")}},
        "seed_stability": {"hgb": {"n_seeds": 5, "std": 0.1, "unstable": False}}})
    _write_json(art / "portfolio" / PF_RUN / "results.json", {
        "label": "SYNTHETIC FIXTURE", "cost_bps": 5.0, "n_trials": 2,
        "table": {"equal_weight": {"net_return_total": 0.5, "sharpe": 0.7, "max_drawdown": -0.2, "cvar_95": 0.01, "turnover_oneway_annual": 0.3},
                  "inverse_vol": {"net_return_total": 0.55, "sharpe": 0.8, "max_drawdown": -0.15, "cvar_95": 0.011, "turnover_oneway_annual": 0.4}},
        "verdict_vs_equal_weight": {"inverse_vol": {"better_than_equal_weight": False, "active_ann_return": 0.001, "active_dsr": 0.48,
                                                    "reasons": ["SYN: not better"]}},
        "selected": "equal_weight", "summary": "SYN summary", "skfolio_used": False, "meta": {"n_obs": 10}})
    expected = {"ml_dsr_a": 0.4242, "ml_nan_run_dsr": None, "pf_sharpe_ew": 0.7, "model_dsr": 0.123, "looks": 2,
                "exp1_maxdd": -0.1234, "exp1_cvar": 0.0456, "exp1_test_maxdd": -0.2222}
    return s, TestClient(create_app(s, artifacts_root=art)), expected


# (1) empty store / empty artifacts -> explicit no_data, HTTP 200, no invented numbers
def test_empty_store_every_endpoint_returns_no_data(empty_client):
    for url in NEW_GET:
        r = empty_client.get(url)
        assert r.status_code == 200, url
        body = r.json()
        assert body["status"] == "no_data", url
        assert body.get("detail"), url
    m = empty_client.get("/api/models").json()
    assert m["registry"]["status"] == "no_data" and m["ml_run"]["status"] == "no_data"
    assert "runs" not in m["ml_run"] and "table" not in empty_client.get("/api/portfolio").json()
    assert "experiments" not in empty_client.get("/api/risk").json()
    # protected-test count is a stored count (0 here), not an invented value
    assert empty_client.get("/api/risk").json()["protected_test"]["looks_used"] == 0


# (2) populated tmp store + tmp artifacts -> exactly the stored values
def test_populated_risk_center_returns_stored_values(populated):
    _, c, exp = populated
    d = c.get("/api/risk").json()
    assert d["status"] == "ok"
    assert d["protected_test"] == {"looks_used": 2, "max_test_looks": V.Criteria().max_test_looks,
                                   "remaining": V.Criteria().max_test_looks - 2,
                                   "source": d["protected_test"]["source"]}
    e1 = next(x for x in d["experiments"] if x["name"] == "SYN_strategy_A")
    assert e1["validation"]["max_drawdown"] == exp["exp1_maxdd"] and e1["validation"]["cvar_95"] == exp["exp1_cvar"]
    assert e1["test"]["max_drawdown"] == exp["exp1_test_maxdd"]
    assert e1["risk_violations"] == ["RISK: max_drawdown -35.0% < limit -30%"]
    e2 = next(x for x in d["experiments"] if x["name"] == "SYN_strategy_B")
    assert e2["test"] is None and e2["risk_violations"] == []


def test_populated_model_lab_returns_stored_values(populated):
    _, c, exp = populated
    d = c.get("/api/models").json()
    assert d["status"] == "ok"
    row = d["registry"]["rows"][0]
    assert row["name"] == "SYN_model" and row["status"] == "IDEA" and row["metrics"]["dsr"] == exp["model_dsr"]
    assert "dedup_key" not in row and "artifact_path" not in row and row["artifact_file"] is None  # registered without a model object: no artifact written
    run = d["ml_run"]
    assert run["run"] == ML_RUN and run["verdict"] == "SYNTHETIC FIXTURE: NULL" and run["protected_test_consulted"] is False
    a = next(r for r in run["runs"] if r["name"] == "syn-run-a")
    assert a["dsr"] == exp["ml_dsr_a"] and a["val_active_ann"] == 0.0123 and a["decision"] == "VALIDATION_FAIL"
    nan_run = next(r for r in run["runs"] if r["name"] == "syn-run-nan")
    assert nan_run["dsr"] is None  # NaN is not representable in JSON; reported as null, never a number


def test_populated_portfolio_returns_stored_values(populated):
    _, c, exp = populated
    d = c.get("/api/portfolio").json()
    assert d["status"] == "ok" and d["run"] == PF_RUN and d["selected"] == "equal_weight"
    assert d["table"]["equal_weight"]["sharpe"] == exp["pf_sharpe_ew"]
    assert d["verdict_vs_equal_weight"]["inverse_vol"]["active_dsr"] == 0.48
    assert d["verdict_vs_equal_weight"]["inverse_vol"]["better_than_equal_weight"] is False


def test_populated_research_lab_returns_stored_values_incl_negative(populated):
    _, c, _ = populated
    d = c.get("/api/research").json()
    assert d["status"] == "ok" and d["experiments_total"] == 2
    assert {"family": "SYN_family", "decision": "TEST_FAIL", "n": 1} in d["by_family"]
    assert {"kind": "negative", "n": 1} in d["lesson_kinds"]
    assert d["lessons"][0]["text"].endswith("negative lesson text") and d["lessons"][0]["kind"] == "negative"
    assert d["hypotheses"][0]["text"] == "SYN hypothesis text"
    assert d["protected_test"]["looks_used"] == 2


# (3) path traversal / arbitrary path cannot read outside the artifact root
def test_run_parameter_cannot_traverse_outside_artifact_root(tmp_path, art):
    outside = tmp_path / "secret"
    _write_json(outside / "results.json", {"verdict": "LEAK_SHOULD_NOT_APPEAR", "table": {"x": {"sharpe": 99}}})
    _write_json(art / "ml" / ML_RUN / "results.json", {"verdict": "SYNTHETIC inside", "runs": {}})
    _write_json(art / "portfolio" / PF_RUN / "results.json", {"label": "SYNTHETIC inside", "table": {}})
    os.symlink(outside, art / "ml" / "evil_link")  # symlinked run dir pointing outside
    c = TestClient(create_app(Store(tmp_path / "trav.db"), artifacts_root=art))
    attempts = ["../../secret", "../secret", "../../../../etc", "/etc/passwd", "..", "evil_link", "%2e%2e%2fsecret", "x/../../secret"]
    for url in ("/api/models", "/api/portfolio"):
        for a in attempts:
            r = c.get(url, params={"run": a})
            assert r.status_code == 404, (url, a, r.status_code)
            assert "LEAK_SHOULD_NOT_APPEAR" not in r.text and "99" not in r.text, (url, a)
    listed = c.get("/api/models").json()["ml_run"]["runs_available"]
    assert listed == [ML_RUN] and "evil_link" not in listed  # symlink is not a run
    assert c.get("/api/models", params={"run": ML_RUN}).json()["ml_run"]["verdict"] == "SYNTHETIC inside"


def test_unparsed_formats_are_not_read(tmp_path, art):
    """A run dir holding only a pickle/joblib file (no results.json) is not a run and its bytes are never opened."""
    (art / "ml" / "20990102").mkdir(parents=True)
    (art / "ml" / "20990102" / "model.joblib").write_bytes(b"not a real pickle \x80\x05")
    c = TestClient(create_app(Store(tmp_path / "pk.db"), artifacts_root=art))
    d = c.get("/api/models").json()
    assert d["ml_run"]["status"] == "no_data" and d["status"] == "no_data"
    assert c.get("/api/models", params={"run": "20990102"}).status_code == 404


# (4) GET only: POST/PUT/DELETE rejected
def test_new_endpoints_reject_writes_with_405(empty_client):
    for url in NEW_GET:
        assert empty_client.post(url, json={}).status_code == 405, url
        assert empty_client.put(url, json={}).status_code == 405, url
        assert empty_client.delete(url).status_code == 405, url


def test_nan_in_stored_json_serialises_as_null(populated):
    """Guard: a NaN in a results file must not turn into a 500 (JSON cannot carry NaN) and must not become a number."""
    _, c, _ = populated
    r = c.get("/api/models")
    assert r.status_code == 200
    runs = {x["name"]: x for x in r.json()["ml_run"]["runs"]}
    assert runs["syn-run-nan"]["dsr"] is None
    assert not any(isinstance(v, float) and math.isnan(v) for v in runs["syn-run-nan"].values())


def test_unknown_run_is_404_not_fake_data(populated):
    _, c, _ = populated
    assert c.get("/api/portfolio", params={"run": "nope"}).status_code == 404
    assert c.get("/api/models", params={"run": "nope"}).status_code == 404


# ---- (5) static XSS guard for engine/api/static/index.html ------------------------------------
# Every ${...} template interpolation in the dashboard script must be one of:
#   * esc(...)                              - the HTML escape helper (the normal case)
#   * a formatter call f(...) / p(...)      - numeric/percent output only (digits . - % and en-dash)
#   * encodeURIComponent(...)               - URL-path encoding inside the fetch URL
#   * protCard(...) / noData(...)           - in-file HTML builders; their own interpolations are checked here
#   * lim / reg / run                       - in-file HTML fragments assembled from checked interpolations
#   * a string literal, a template literal (its own ${} are checked), or a ternary whose branches are
#     themselves safe (the condition is not output), or a `.map(...).join("...")` chain whose map
#     callback returns a template (or uses .map(esc)), optionally followed by `|| "literal"`.
# Anything else (e.g. ${x.text} or ${o.experiments} without esc) is a violation.
INDEX_HTML = Path(__file__).resolve().parents[1] / "engine" / "api" / "static" / "index.html"

ALLOW_FORMATTERS = ("f", "p")
ALLOW_ENCODERS = ("encodeURIComponent",)
ALLOW_HTML_BUILDERS = ("protCard", "noData")
ALLOW_HTML_FRAGMENT_VARS = ("lim", "reg", "run")
_JOIN_TAIL = re.compile(r"""\.join\((?:"[^"]*"|'[^']*')\)(?:\s*\|\|\s*(?:"[^"]*"|'[^']*'))?\s*$""")
_REGEX_PREV = "(,=:[!&|?{};>"


def _skip_string(s, i):
    q, j = s[i], i + 1
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == q:
            return j + 1
        j += 1
    raise ValueError("unterminated string")


def _skip_regex(s, i):
    """s[i] == '/' opening a regex literal (not a comment). Handles escapes and [...] classes."""
    j, in_class = i + 1, False
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if in_class:
            in_class = c != "]"
        elif c == "[":
            in_class = True
        elif c == "/":
            j += 1
            while j < len(s) and s[j].isalpha():  # flags
                j += 1
            return j
        j += 1
    raise ValueError("unterminated regex")


def _regex_allowed(s, i):
    j = i - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    return j < 0 or s[j] in _REGEX_PREV


def _skip_atom(s, i, found):
    """If a string, template literal or regex literal starts at i, return the index just past it; else None."""
    c = s[i]
    if c in "'\"":
        return _skip_string(s, i)
    if c == "`":
        return _skip_template(s, i, found)
    if c == "/" and s[i + 1:i + 2] != "/" and _regex_allowed(s, i):
        return _skip_regex(s, i)
    return None


def _skip_template(s, i, found):
    """s[i] == '`'. Returns index after the closing backtick; records every ${} expression (any depth) in found."""
    j = i + 1
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "`":
            return j + 1
        if c == "$" and s[j + 1:j + 2] == "{":
            expr, j = _read_expr(s, j + 2, found)
            found.append(expr)
            continue
        j += 1
    raise ValueError("unterminated template")


def _read_expr(s, i, found):
    depth, j = 0, i
    while j < len(s):
        n = _skip_atom(s, j, found)
        if n is not None:
            j = n
            continue
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return s[i:j], j + 1
            depth -= 1
        j += 1
    raise ValueError("unterminated ${")


def collect_interpolations(html):
    """Every ${...} expression text in the <script> block, at any template nesting depth."""
    script = html[html.index("<script>") + len("<script>"):html.index("</script>")]
    found, i = [], 0
    while i < len(script):
        if script.startswith("//", i):
            nl = script.find("\n", i)
            i = len(script) if nl < 0 else nl
            continue
        n = _skip_atom(script, i, found)
        i = n if n is not None else i + 1
    return found


def _match_paren(e, i):
    depth, j = 0, i
    while j < len(e):
        n = _skip_atom(e, j, [])
        if n is not None:
            j = n
            continue
        c = e[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError("unbalanced parens")


def _split_ternary(e):
    """(true_branch, false_branch) for a top-level `cond ? a : b`, else None."""
    def is_ternary_q(j):
        return e[j] == "?" and e[j + 1:j + 2] not in ("?", ".") and e[j - 1:j] != "?"
    depth, j, qpos = 0, 0, None
    while j < len(e):
        n = _skip_atom(e, j, [])
        if n is not None:
            j = n
            continue
        c = e[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and is_ternary_q(j):
            qpos = j
            break
        j += 1
    if qpos is None:
        return None
    depth, pending, j = 0, 0, qpos + 1
    while j < len(e):
        n = _skip_atom(e, j, [])
        if n is not None:
            j = n
            continue
        c = e[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0:
            if is_ternary_q(j):
                pending += 1
            elif c == ":":
                if pending == 0:
                    return e[qpos + 1:j], e[j + 1:]
                pending -= 1
        j += 1
    return None


def _whole_call(e, names):
    m = re.match(r"(%s)\(" % "|".join(map(re.escape, names)), e)
    return bool(m) and _match_paren(e, m.end() - 1) == len(e) - 1


def _is_safe_interpolation(e):
    e = e.strip()
    if not e:
        return False
    if e[0] == "(" and _match_paren(e, 0) == len(e) - 1:
        return _is_safe_interpolation(e[1:-1])
    if _whole_call(e, ("esc",) + ALLOW_FORMATTERS + ALLOW_ENCODERS + ALLOW_HTML_BUILDERS):
        return True
    if e in ALLOW_HTML_FRAGMENT_VARS:
        return True
    if e[0] in "'\"" and _skip_string(e, 0) == len(e):
        return True
    if e[0] == "`" and _skip_template(e, 0, []) == len(e):
        return True
    t = _split_ternary(e)
    if t is not None:
        return _is_safe_interpolation(t[0]) and _is_safe_interpolation(t[1])
    m = _JOIN_TAIL.search(e)
    if m:
        prefix = e[:m.start()]
        if re.search(r"\.map\(esc\)\s*$", prefix):
            return True
        if ".map(" in prefix and re.search(r"(=>|return)\s*`", prefix):
            return True
    return False


def unescaped_interpolations(html):
    return [e.strip() for e in collect_interpolations(html) if not _is_safe_interpolation(e)]


def test_index_html_every_interpolation_is_escaped():
    html = INDEX_HTML.read_text(encoding="utf-8")
    interps = collect_interpolations(html)
    assert len(interps) >= 80, "scanner found too few interpolations; check the parser"
    assert unescaped_interpolations(html) == []


def test_xss_guard_mutation_unescaped_lessons_text_is_flagged(tmp_path):
    """Mutation check on a COPY: drop the esc() around the lessons text (Learning Center) and the guard must flag it."""
    src = INDEX_HTML.read_text(encoding="utf-8")
    needle = '<td>${esc(x.text)}</td></tr>`).join("")}</table></div>\n  <div class="card"><h3>Hypotheses</h3>'
    assert src.count(needle) == 1, "lessons row anchor moved; update the mutation anchor"
    mutated = src.replace(needle, needle.replace("${esc(x.text)}", "${x.text}"), 1)
    copy = tmp_path / "index.html"
    copy.write_text(mutated, encoding="utf-8")
    assert unescaped_interpolations(copy.read_text(encoding="utf-8")) == ["x.text"]
    assert unescaped_interpolations(src) == []
