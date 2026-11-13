"""Paper broker: simulated orders/fills/positions/P&L only. There is NO live connectivity and no credentials are accepted."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Fill: symbol: str; qty: float; price: float; cost: float; ts: str
@dataclass
class PaperBroker:
    cash: float = 100_000.0; commission_bps: float = 1.0; slippage_bps: float = 2.0
    positions: dict[str, float] = field(default_factory=dict); fills: list[Fill] = field(default_factory=list)
    def order(self, symbol: str, qty: float, mkt_price: float, ts: str) -> Fill:
        if qty == 0 or mkt_price <= 0: raise ValueError("invalid paper order")
        px = mkt_price * (1 + self.slippage_bps / 1e4 * (1 if qty > 0 else -1))      # adverse slippage
        cost = abs(qty) * px * self.commission_bps / 1e4
        if qty > 0 and qty * px + cost > self.cash + 1e-9: raise ValueError("insufficient paper cash (no leverage)")
        if qty < 0 and self.positions.get(symbol, 0) + qty < -1e-9: raise ValueError("short selling disabled")
        self.cash -= qty * px + cost; self.positions[symbol] = self.positions.get(symbol, 0) + qty
        f = Fill(symbol, qty, px, cost, ts); self.fills.append(f); return f
    def equity(self, prices: dict[str, float]) -> float: return self.cash + sum(q * prices[s] for s, q in self.positions.items())
    def execution_cost_total(self, ref_prices: dict[str, float] | None = None) -> float: return sum(f.cost for f in self.fills)
