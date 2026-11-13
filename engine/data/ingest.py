"""Market-data ingestion: provider interface, parquet cache with provenance metadata.

Every dataset carries a `kind` of "real" or "synthetic". Synthetic data exists ONLY for software
tests and must never be presented as investment evidence; downstream reports propagate the flag.
"""
from __future__ import annotations
import hashlib, json, logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence
import numpy as np
import pandas as pd
from engine.errors import DataError

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

@dataclass(frozen=True)
class DatasetMeta:
    provider: str
    kind: str                      # "real" | "synthetic"
    symbols: list[str]
    start: str
    end: str | None
    rows: int
    first_date: str
    last_date: str
    fetched_at: str
    sha256: str
    price_field: str = "adjusted close (splits+dividends)"
    notes: list[str] = field(default_factory=list)

class DataProvider(Protocol):
    name: str
    kind: str
    def fetch(self, symbols: Sequence[str], start: str, end: str | None) -> pd.DataFrame: ...

def validate_prices(df: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Wide daily price frame: DatetimeIndex ascending/unique, float columns, positive, no all-NaN column."""
    if df is None or df.empty: raise DataError("empty price frame")
    missing = [s for s in symbols if s not in df.columns]
    if missing: raise DataError(f"missing symbols: {missing}")
    df = df[list(symbols)].copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index().astype(float)
    bad = [c for c in df if df[c].notna().sum() == 0]
    if bad: raise DataError(f"no data for: {bad}")
    if (df.stack() <= 0).any(): raise DataError("non-positive prices present")
    return df

class YahooProvider:
    name, kind = "yahoo_finance(yfinance)", "real"
    def fetch(self, symbols, start, end):
        try: import yfinance as yf
        except ImportError as e: raise DataError("yfinance not installed (pip install '.[data]')") from e
        raw = yf.download(list(symbols), start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty: raise DataError("yfinance returned no data (network or ticker problem)")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": symbols[0]})
        return close

class CSVProvider:
    """Wide CSV: first column = date, other columns = symbols (adjusted close)."""
    kind = "real"
    def __init__(self, path: str | Path): self.path = Path(path); self.name = f"csv:{self.path.name}"
    def fetch(self, symbols, start, end):
        if not self.path.exists(): raise DataError(f"CSV not found: {self.path}")
        df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        return df.loc[start:end] if end else df.loc[start:]

class SyntheticProvider:
    """Seeded geometric Brownian motion. FOR SOFTWARE TESTS ONLY - not real market data."""
    name, kind = "synthetic_gbm(TEST_ONLY)", "synthetic"
    def __init__(self, seed: int = 7): self.seed = seed
    def fetch(self, symbols, start, end):
        idx = pd.bdate_range(start, end or "2020-12-31")
        rng = np.random.default_rng(self.seed)
        out = {}
        for i, s in enumerate(symbols):
            mu, sig = 0.06 + 0.02 * i, 0.15 + 0.05 * i
            r = rng.normal(mu / 252, sig / np.sqrt(252), len(idx))
            out[s] = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame(out, index=idx)

def _key(provider: str, symbols: Sequence[str], start: str, end: str | None) -> str:
    return hashlib.sha1(json.dumps([provider, sorted(symbols), start, end]).encode()).hexdigest()[:12]

def load_prices(provider: DataProvider, symbols: Sequence[str], start: str, end: str | None = None,
                *, cache_dir: Path = DATASETS, refresh: bool = False, offline: bool = False) -> tuple[pd.DataFrame, DatasetMeta]:
    """Fetch (or load cached) validated prices plus provenance metadata."""
    symbols = list(symbols); cache_dir.mkdir(parents=True, exist_ok=True)
    k = _key(provider.name, symbols, start, end)
    pq, mj = cache_dir / f"{k}.parquet", cache_dir / f"{k}.meta.json"
    if pq.exists() and mj.exists() and not refresh:
        log.info("cache hit %s", pq.name)
        return pd.read_parquet(pq), DatasetMeta(**json.loads(mj.read_text()))
    if offline: raise DataError(f"offline mode and no cache for {symbols} {start}..{end}")
    log.info("fetching %s via %s", symbols, provider.name)
    df = validate_prices(provider.fetch(symbols, start, end), symbols)
    notes = []
    na = int(df.isna().sum().sum())
    if na: notes.append(f"{na} NaN cells (assets with different inception dates); not forward-filled")
    df.to_parquet(pq)
    sha = hashlib.sha256(pq.read_bytes()).hexdigest()
    meta = DatasetMeta(provider.name, provider.kind, symbols, start, end, len(df), str(df.index[0].date()),
                       str(df.index[-1].date()), datetime.now(timezone.utc).isoformat(timespec="seconds"), sha, notes=notes)
    mj.write_text(json.dumps(asdict(meta), indent=2))
    return df, meta

def check_availability(df: pd.DataFrame, universe: str = "etf24") -> dict:
    """Additive PIT check: flag (never fill) values dated before a symbol's availability date. See engine/data/pit.py."""
    from engine.data.pit import validate_frame_pit
    return validate_frame_pit(df, universe)
