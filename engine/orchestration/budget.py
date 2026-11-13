"""Resource budgets enforced from the persistent ledger (compute seconds, experiments, disk MB, LLM $, API calls)."""
from __future__ import annotations
import shutil, time
from dataclasses import dataclass
from pathlib import Path

class BudgetExceeded(RuntimeError): pass

@dataclass(frozen=True)
class Budget:
    max_compute_s_per_day: float = 1800; max_experiments_per_day: int = 20; max_disk_mb: float = 2048
    max_llm_usd_per_day: float = 0.0; max_api_calls_per_day: int = 0    # paid LLMs and external APIs stay OFF
    max_llm_calls_per_day: int = 24; max_llm_tokens_per_day: int = 200_000   # LOCAL open-source model only (loopback)

class BudgetGuard:
    def __init__(self, store, budget: Budget = Budget(), disk_path: Path | None = None):
        self.store, self.b, self.disk_path = store, budget, disk_path or store.path.parent
    def _day(self): return time.time() - 86400
    def check(self, resource: str, amount: float = 0) -> None:
        s = self.store
        if resource == "experiments" and s.spent("experiments", self._day()) + amount > self.b.max_experiments_per_day: raise BudgetExceeded("experiments/day")
        if resource == "compute_s" and s.spent("compute_s", self._day()) + amount > self.b.max_compute_s_per_day: raise BudgetExceeded("compute seconds/day")
        if resource == "llm_usd" and s.spent("llm_usd", self._day()) + amount > self.b.max_llm_usd_per_day: raise BudgetExceeded("LLM $/day (0 = disabled)")
        if resource == "api_calls" and s.spent("api_calls", self._day()) + amount > self.b.max_api_calls_per_day: raise BudgetExceeded("API calls/day (0 = disabled)")
        if resource == "llm_calls" and s.spent("llm_calls", self._day()) + amount > self.b.max_llm_calls_per_day: raise BudgetExceeded("local LLM calls/day")
        if resource == "llm_tokens" and s.spent("llm_tokens", self._day()) + amount > self.b.max_llm_tokens_per_day: raise BudgetExceeded("local LLM tokens/day")
        if resource == "disk":
            used = sum(f.stat().st_size for f in Path(self.disk_path).rglob("*") if f.is_file()) / 1e6
            if used > self.b.max_disk_mb or shutil.disk_usage(self.disk_path).free / 1e9 < 2: raise BudgetExceeded(f"disk ({used:.0f} MB used)")
    def status(self) -> dict:
        d = self._day(); s = self.store
        return {"limits": self.b.__dict__, "used_24h": {"experiments": s.spent("experiments", d), "compute_s": s.spent("compute_s", d), "llm_usd": s.spent("llm_usd", d), "api_calls": s.spent("api_calls", d), "llm_calls": s.spent("llm_calls", d), "llm_tokens": s.spent("llm_tokens", d)}}
