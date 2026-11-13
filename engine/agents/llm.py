"""Local open-source LLM hypothesis generator (Ollama, loopback only).

What this IS: a proposal source. The model suggests strategy *specs* inside a strict schema, informed by memory.
What this is NOT: it never trains or updates weights; it cannot change costs, criteria, risk limits or assets; its proposals get
no special treatment (same dedup, same multiple-testing count, same frozen gates, same protected-test accounting).
"""
from __future__ import annotations
import json, logging, os, time, urllib.parse, urllib.request
from typing import Callable
from engine.memory.store import Store, h
from engine.orchestration.budget import BudgetGuard
from engine.research.validation import Criteria
from engine.errors import QALError
log = logging.getLogger(__name__)
DEFAULT_MODEL = os.environ.get("QAE_LLM_MODEL", "qwen2.5:14b")
LOOKBACK_RANGE, VT_RANGE = (3, 24), (0.05, 0.25)

class LLMUnavailable(QALError): pass
Transport = Callable[[str, dict], dict]

def _http(url: str, payload: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())

class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = "http://127.0.0.1:11434", transport: Transport = _http, seed: int = 0):
        u = urllib.parse.urlparse(host)
        if u.hostname not in ("127.0.0.1", "localhost", "::1"): raise ValueError("LLM host must be loopback: no data leaves this machine")
        self.model, self.url, self.transport, self.seed = model, host.rstrip("/") + "/api/chat", transport, seed
    def chat_json(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        try:
            r = self.transport(self.url, {"model": self.model, "stream": False, "format": schema,
                "options": {"temperature": 0, "seed": self.seed, "num_ctx": 4096}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        except Exception as e: raise LLMUnavailable(f"ollama call failed: {e}") from e
        usage = {"tokens": r.get("prompt_eval_count", 0) + r.get("eval_count", 0)}
        try: return json.loads(r["message"]["content"]), usage
        except Exception as e: raise LLMUnavailable(f"non-JSON model output: {e}") from e

SCHEMA = {"type": "object", "properties": {"proposals": {"type": "array", "maxItems": 3, "items": {"type": "object",
  "properties": {"lookback_months": {"type": "integer"}, "vol_target": {"type": ["number", "null"]}, "rationale": {"type": "string"}},
  "required": ["lookback_months", "vol_target", "rationale"]}}}, "required": ["proposals"]}
SYSTEM = ("You are a quantitative research assistant. Propose NEW falsifiable variants of a long/flat trend-following strategy on one ETF. "
          f"Allowed knobs only: lookback_months integer in {LOOKBACK_RANGE}, vol_target null or a number in {VT_RANGE} (annualised). "
          "Each rationale must state a testable hypothesis grounded in the prior results given. Do not repeat tested variants. Return JSON only.")

def validate_proposal(p: dict, base_spec: dict) -> tuple[dict, str] | None:
    """Strict schema gate: reject (never clamp) anything outside the allowed ranges; the LLM cannot touch asset, costs or family."""
    try:
        lb, vt, why = p["lookback_months"], p["vol_target"], str(p["rationale"]).strip()
        if isinstance(lb, bool) or not isinstance(lb, int) or not LOOKBACK_RANGE[0] <= lb <= LOOKBACK_RANGE[1]: return None
        if vt is not None and (isinstance(vt, bool) or not isinstance(vt, (int, float)) or not VT_RANGE[0] <= vt <= VT_RANGE[1]): return None
        if len(why) < 15: return None
    except (KeyError, TypeError): return None
    return {**base_spec, "lookback_months": lb, "vol_target": None if vt is None else round(float(vt), 3)}, why[:400]

class LLMProposer:
    def __init__(self, client: OllamaClient, store: Store, guard: BudgetGuard | None = None, max_calls_per_cycle: int = 1):
        self.c, self.store, self.guard, self.calls_left = client, store, guard, max_calls_per_cycle
        self.source = f"llm:{client.model}"
    def propose(self, base_spec: dict, base_result: dict, crit: Criteria, data_sha: str) -> list[tuple[dict, str]]:
        if self.calls_left <= 0: return []
        if self.guard: self.guard.check("llm_calls", 1)           # raises BudgetExceeded -> cycle stops cleanly
        self.calls_left -= 1
        v, vb = base_result["metrics"]["validation"], base_result["metrics"]["validation_benchmark"]
        tried = [json.loads(x["spec"]) for x in self.store.q("SELECT s.spec FROM strategies s")]
        lessons = [x["text"] for x in self.store.q("SELECT text FROM lessons ORDER BY id DESC LIMIT 12")]
        user = (f"Baseline spec: {json.dumps(base_spec)}\nBaseline validation: Sharpe {v['sharpe']:.2f}, maxDD {v['max_drawdown']:.1%}, vol {v['ann_vol']:.1%}, "
                f"DSR {v.get('dsr', float('nan')):.2f}; buy&hold Sharpe {vb['sharpe']:.2f}.\nAlready tested (do not repeat): "
                f"{[(t['lookback_months'], t.get('vol_target')) for t in tried]}\nRecent lessons:\n- " + "\n- ".join(lessons))
        t0 = time.time(); raw, usage = self.c.chat_json(SYSTEM, user, SCHEMA)
        self.store.spend("llm_calls", 1, self.source); self.store.spend("llm_tokens", usage["tokens"], self.source); self.store.spend("compute_s", time.time() - t0, "llm")
        out, rejected = [], 0
        for p in raw.get("proposals", []):
            ok = validate_proposal(p, base_spec)
            if ok is None: rejected += 1; continue
            spec, why = ok
            if self.store.find_experiment(h(spec), crit.digest(), data_sha) or any(spec == s for s, _ in out): continue
            out.append((spec, why))
        log.info("LLM %s: %d valid, %d schema-rejected", self.c.model, len(out), rejected)
        if rejected: self.store.add_lesson(None, base_spec["family"], "llm_schema_reject", f"{self.source}: {rejected} proposal(s) rejected by schema gate")
        return out
