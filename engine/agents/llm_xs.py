"""LLM proposer for the cross-sectional factor family (xs_factor). Local Ollama, loopback only (OllamaClient).

The model may choose ONLY: kind in {momentum,reversal,lowvol}, lookback_days int in [5,252], skip_days int in [0,63], vol_adj bool
(plus a free-text rationale). Universe, top_frac, min_assets, cost_bps, family are FIXED here and not LLM-controllable.
Validation is strict: reject, never clamp; unknown/extra keys, wrong types (incl. bool-as-int, float-as-int) and out-of-range values reject the proposal.
Proposals get no special treatment downstream (same dedup, same trial counting, same frozen criteria)."""
from __future__ import annotations
import json, logging, time
from engine.agents.llm import OllamaClient
from engine.memory.store import Store, h
from engine.orchestration.budget import BudgetGuard
from engine.research.validation import Criteria
log = logging.getLogger(__name__)
KINDS = ("momentum", "reversal", "lowvol"); LOOKBACK_RANGE, SKIP_RANGE = (5, 252), (0, 63)
FIXED = {"family": "xs_factor", "universe": "etf24", "top_frac": 0.33, "min_assets": 8, "cost_bps": 5.0}
ALLOWED_KEYS = {"kind", "lookback_days", "skip_days", "vol_adj", "rationale"}
MAX_PER_CALL = 4

SCHEMA = {"type": "object", "properties": {"proposals": {"type": "array", "maxItems": MAX_PER_CALL, "items": {"type": "object", "properties": {
    "kind": {"type": "string", "enum": list(KINDS)}, "lookback_days": {"type": "integer", "minimum": LOOKBACK_RANGE[0], "maximum": LOOKBACK_RANGE[1]},
    "skip_days": {"type": "integer", "minimum": SKIP_RANGE[0], "maximum": SKIP_RANGE[1]}, "vol_adj": {"type": "boolean"}, "rationale": {"type": "string"}},
    "required": ["kind", "lookback_days", "skip_days", "vol_adj", "rationale"]}}}, "required": ["proposals"]}
SYSTEM = ("You are a quantitative research assistant proposing cross-sectional factor variants for a monthly-rebalanced, long-only, top-third ETF portfolio "
          "(24 liquid ETFs, judged by active return versus the equal-weight universe after costs). "
          f"Allowed knobs ONLY: kind one of {list(KINDS)}; lookback_days integer in {list(LOOKBACK_RANGE)}; skip_days integer in {list(SKIP_RANGE)} "
          "(days skipped before the lookback window); vol_adj boolean (divide the score by realised volatility). Nothing else can be changed. "
          "Each rationale must state a short testable hypothesis grounded in the prior results given. Do not repeat tested specs. Return JSON only.")

def validate_proposal(p) -> tuple[dict, str] | None:
    """Strict gate. Returns (full spec, rationale) or None. Never clamps; never lets the model set a fixed field."""
    if not isinstance(p, dict) or set(p) != ALLOWED_KEYS: return None
    k, lb, sk, va, why = p["kind"], p["lookback_days"], p["skip_days"], p["vol_adj"], p["rationale"]
    if k not in KINDS or not isinstance(k, str): return None
    for v, (lo, hi) in ((lb, LOOKBACK_RANGE), (sk, SKIP_RANGE)):
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi: return None
    if not isinstance(va, bool) or not isinstance(why, str) or len(why.strip()) < 10: return None
    return {**FIXED, "kind": k, "lookback_days": lb, "skip_days": sk, "vol_adj": va}, why.strip()[:400]

class LLMXSProposer:
    def __init__(self, client: OllamaClient, store: Store, guard: BudgetGuard | None = None, max_calls: int = 1):
        self.c, self.store, self.guard, self.calls_left = client, store, guard, max_calls
        self.source = f"llm:{client.model}"; self.stats = {"calls": 0, "proposed": 0, "invalid": 0, "duplicate": 0, "valid": 0, "tokens": 0, "secs": 0.0}
    def build_prompt(self) -> str:
        tried = [json.loads(x["spec"]) for x in self.store.q("SELECT spec FROM strategies WHERE family='xs_factor'")]
        lessons = [x["text"] for x in self.store.q("SELECT text FROM lessons WHERE family='xs_factor' AND kind!='llm_schema_reject' ORDER BY id DESC LIMIT 15")]
        return (f"Already tested (do not repeat): {[(t['kind'], t['lookback_days'], t['skip_days'], t['vol_adj']) for t in tried]}\n"
                f"Prior lessons (newest first):\n- " + ("\n- ".join(lessons) if lessons else "none yet") +
                f"\nPropose up to {MAX_PER_CALL} NEW, diverse specs most likely to show positive active return after costs.")
    def propose(self, crit: Criteria, data_sha: str) -> list[tuple[dict, str]]:
        if self.calls_left <= 0: return []
        if self.guard: self.guard.check("llm_calls", 1)
        self.calls_left -= 1; t0 = time.time()
        raw, usage = self.c.chat_json(SYSTEM, self.build_prompt(), SCHEMA)
        dt = time.time() - t0
        self.store.spend("llm_calls", 1, self.source); self.store.spend("llm_tokens", usage["tokens"], self.source); self.store.spend("compute_s", dt, "llm")
        self.stats["calls"] += 1; self.stats["tokens"] += usage["tokens"]; self.stats["secs"] += dt
        items = raw.get("proposals", []) if isinstance(raw, dict) else []
        items = items if isinstance(items, list) else []
        out: list[tuple[dict, str]] = []
        for p in items[:MAX_PER_CALL]:
            self.stats["proposed"] += 1; ok = validate_proposal(p)
            if ok is None: self.stats["invalid"] += 1; continue
            spec, why = ok
            if self.store.find_experiment(h(spec), crit.digest(), data_sha) or self.store.q("SELECT 1 FROM strategies WHERE spec_hash=?", h(spec)) or any(spec == s for s, _ in out):
                self.stats["duplicate"] += 1; continue
            out.append((spec, why)); self.stats["valid"] += 1
        if self.stats["invalid"]: self.store.add_lesson(None, "xs_factor", "llm_schema_reject", f"{self.source}: {self.stats['invalid']} proposal(s) rejected by schema gate so far")
        return out
