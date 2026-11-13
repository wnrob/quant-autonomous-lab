"""Contract specifications for the roots present in the user's local data.

SOURCE: "CME contract specs as known to the agent; verify before trading". Values were not fetched from
the exchange in this session. `confidence` marks fields the agent is less sure of; nothing here may be
used to size real positions without checking the current CME Group contract specification page.

multiplier = USD per 1.00 of QUOTED price (grains quote in cents/bushel, so 1.00 = 1 cent = $50).
tick_value = tick_size * multiplier (checked by a unit test, not typed by hand).
listed_months = the months the market actually trades as front month (some roots list every month but
liquidity concentrates in a subset, e.g. GC trades G/J/M/Q/V/Z).
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from engine.futures import calendar as cal

SOURCE = "CME contract specs as known to the agent; verify before trading"

QUARTERLY = (3, 6, 9, 12)
ALL_MONTHS = tuple(range(1, 13))


@dataclass(frozen=True)
class ContractSpec:
    root: str
    name: str
    exchange: str
    currency: str
    multiplier: float
    tick_size: float
    listed_months: tuple[int, ...]
    expiry_rule: str              # key into calendar.EXPIRY_RULES
    settlement: str               # "cash" | "physical"
    typical_roll: str             # human description of where the market rolls
    roll_ref: str                 # "last_trade" | "first_notice": what a days-before roll rule counts back from
    confidence: str = "high"      # "high" | "medium" (agent unsure of a field; see note)
    note: str = ""
    source: str = SOURCE

    @property
    def tick_value(self) -> float: return self.tick_size * self.multiplier

    def last_trade(self, year: int, month: int) -> dt.date:
        if month not in self.listed_months: raise ValueError(f"{self.root} does not trade month {month} in this table")
        return cal.EXPIRY_RULES[self.expiry_rule](year, month)

    def first_notice(self, year: int, month: int) -> dt.date | None:
        return cal.fnd_month_before(year, month) if self.settlement == "physical" and self.expiry_rule in ("comex_metal", "treasury", "grain") else None

    def roll_reference(self, year: int, month: int) -> dt.date:
        """Date a days-before roll rule counts back from: FND where one exists and roll_ref says so, else last trade."""
        if self.roll_ref == "first_notice":
            f = self.first_notice(year, month)
            if f is not None: return f
        return self.last_trade(year, month)


_EQ = "roll ~8 days before expiry (CME 'roll date' is the Thursday 8 days before the 3rd Friday)"
_FX = "roll ~1 week before last trade (volume migrates during the week before IMM date)"
_UST = "roll before first notice day (end of month before contract month); volume migrates ~last week of that month"
_METAL = "roll before first notice day (last business day of month before contract month)"
_GRAIN = "roll before first notice day (last business day of month before contract month)"

SPECS: dict[str, ContractSpec] = {s.root: s for s in [
    ContractSpec("MES", "Micro E-mini S&P 500", "CME", "USD", 5.0, 0.25, QUARTERLY, "equity_index", "cash", _EQ, "last_trade"),
    ContractSpec("MNQ", "Micro E-mini Nasdaq-100", "CME", "USD", 2.0, 0.25, QUARTERLY, "equity_index", "cash", _EQ, "last_trade"),
    ContractSpec("M2K", "Micro E-mini Russell 2000", "CME", "USD", 5.0, 0.10, QUARTERLY, "equity_index", "cash", _EQ, "last_trade"),
    ContractSpec("6E", "Euro FX", "CME", "USD", 125_000.0, 0.00005, QUARTERLY, "fx", "physical", _FX, "last_trade"),
    ContractSpec("6J", "Japanese Yen", "CME", "USD", 12_500_000.0, 0.0000005, QUARTERLY, "fx", "physical", _FX, "last_trade"),
    ContractSpec("6B", "British Pound", "CME", "USD", 62_500.0, 0.0001, QUARTERLY, "fx", "physical", _FX, "last_trade",
                 confidence="medium", note="tick believed 0.0001 ($6.25); verify"),
    ContractSpec("6A", "Australian Dollar", "CME", "USD", 100_000.0, 0.00005, QUARTERLY, "fx", "physical", _FX, "last_trade",
                 confidence="medium", note="tick believed 0.00005 ($5.00); the user's qa/contracts.py uses 0.0001 - one of them is wrong, verify"),
    ContractSpec("CL", "WTI Crude Oil", "NYMEX", "USD", 1_000.0, 0.01, ALL_MONTHS, "crude", "physical",
                 "roll several business days before last trade (front month expires ~20th of prior month)", "last_trade"),
    ContractSpec("NG", "Henry Hub Natural Gas", "NYMEX", "USD", 10_000.0, 0.001, ALL_MONTHS, "natgas", "physical",
                 "roll several business days before last trade (3 bdays before the contract month)", "last_trade"),
    ContractSpec("GC", "Gold", "COMEX", "USD", 100.0, 0.10, (2, 4, 6, 8, 10, 12), "comex_metal", "physical", _METAL, "first_notice",
                 note="all months listed; G/J/M/Q/V/Z are the active ones"),
    ContractSpec("MGC", "Micro Gold", "COMEX", "USD", 10.0, 0.10, (2, 4, 6, 8, 10, 12), "comex_metal", "physical", _METAL, "first_notice",
                 confidence="medium", note="micro gold delivery/expiry mechanics believed to mirror GC; verify"),
    ContractSpec("SI", "Silver", "COMEX", "USD", 5_000.0, 0.005, (3, 5, 7, 9, 12), "comex_metal", "physical", _METAL, "first_notice"),
    ContractSpec("HG", "Copper", "COMEX", "USD", 25_000.0, 0.0005, (3, 5, 7, 9, 12), "comex_metal", "physical", _METAL, "first_notice"),
    ContractSpec("ZN", "10-Year T-Note", "CBOT", "USD", 1_000.0, 1 / 64, QUARTERLY, "treasury", "physical", _UST, "first_notice",
                 note="quoted in points and 32nds; tick is half of 1/32"),
    ContractSpec("ZB", "30-Year T-Bond", "CBOT", "USD", 1_000.0, 1 / 32, QUARTERLY, "treasury", "physical", _UST, "first_notice"),
    ContractSpec("ZC", "Corn", "CBOT", "USD", 50.0, 0.25, (3, 5, 7, 9, 12), "grain", "physical", _GRAIN, "first_notice",
                 note="quoted in cents/bushel; 5,000 bu"),
    ContractSpec("ZW", "Chicago SRW Wheat", "CBOT", "USD", 50.0, 0.25, (3, 5, 7, 9, 12), "grain", "physical", _GRAIN, "first_notice",
                 note="quoted in cents/bushel; 5,000 bu"),
    ContractSpec("ZS", "Soybeans", "CBOT", "USD", 50.0, 0.25, (1, 3, 5, 7, 8, 9, 11), "grain", "physical", _GRAIN, "first_notice",
                 note="quoted in cents/bushel; 5,000 bu"),
]}


@dataclass(frozen=True, order=True)
class ContractId:
    """Sortable by (year, month): chronological expiry order. NEVER sort tickers alphabetically
    (MNQZ5 > MNQH6 alphabetically) - that ordering is what made the user's fetch_full pick dead contracts."""
    year: int
    month: int
    root: str

    @property
    def ticker(self) -> str: return f"{self.root}{cal.MONTH_CODES[self.month]}{self.year % 10}"
    def __str__(self) -> str: return self.ticker

    @property
    def spec(self) -> ContractSpec: return SPECS[self.root]
    def last_trade(self) -> dt.date: return self.spec.last_trade(self.year, self.month)
    def roll_reference(self) -> dt.date: return self.spec.roll_reference(self.year, self.month)


def parse_ticker(ticker: str, root: str, ref_date: dt.date) -> ContractId:
    """'MESZ4' -> ContractId(2024, 12, 'MES'). The single year digit is resolved to the decade that puts the
    contract's month within (-2, +8] years of ref_date (an observation date the ticker was seen on), because
    a contract is never observed years after it expires nor ~10 years before."""
    rest = ticker[len(root):] if ticker.startswith(root) else None
    if not rest or len(rest) != 2 or rest[0] not in cal.CODE_MONTH or not rest[1].isdigit():
        raise ValueError(f"cannot parse ticker {ticker!r} for root {root!r}")
    m, digit = cal.CODE_MONTH[rest[0]], int(rest[1])
    cands = [y for y in range(ref_date.year - 3, ref_date.year + 10) if y % 10 == digit]
    ok = [y for y in cands if -2 < (y - ref_date.year) + (m - ref_date.month) / 12 <= 8]
    if len(ok) != 1: raise ValueError(f"ambiguous year for {ticker} at {ref_date}: {ok}")
    return ContractId(ok[0], m, root)


def specs_table() -> list[dict]:
    rows = []
    for s in SPECS.values():
        rows.append({"root": s.root, "name": s.name, "exchange": s.exchange, "currency": s.currency, "multiplier": s.multiplier,
                     "tick_size": s.tick_size, "tick_value": round(s.tick_value, 6),
                     "listed_months": "".join(cal.MONTH_CODES[m] for m in s.listed_months), "expiry_rule": s.expiry_rule,
                     "expiry_rule_text": (cal.EXPIRY_RULES[s.expiry_rule].__doc__ or "").strip().split("\n")[0],
                     "settlement": s.settlement, "typical_roll": s.typical_roll, "confidence": s.confidence, "note": s.note, "source": s.source})
    return rows
