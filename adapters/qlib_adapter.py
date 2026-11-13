"""qlib adapter: (1) write our price frame into qlib's binary provider format (own writer; qlib's clone scripts are NOT executed),
(2) evaluate qlib expressions via a worker in envs/ml. Contract: features(prices, {name: "$close expr"}) -> DataFrame[date x (symbol,name)]."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np, pandas as pd
from adapters.envrun import run_in_env

def write_qlib_dir(prices: pd.DataFrame, out: Path) -> Path:
    """qlib layout: calendars/day.txt, instruments/all.txt, features/<sym>/<field>.day.bin = float32[start_calendar_idx, v0, v1, ...].
    Missing observations stay missing (NaN gaps are not filled); each symbol is written from its first valid date."""
    cal = prices.index; (out / "calendars").mkdir(parents=True, exist_ok=True); (out / "instruments").mkdir(exist_ok=True)
    (out / "calendars/day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in cal))
    lines = []
    for sym in prices.columns:
        s = prices[sym]; first, last = s.first_valid_index(), s.last_valid_index()
        if first is None: continue
        i0, i1 = cal.get_loc(first), cal.get_loc(last); seg = s.iloc[i0:i1 + 1].to_numpy(dtype="<f4")
        d = out / "features" / sym.lower(); d.mkdir(parents=True, exist_ok=True)
        for field, vals in (("close", seg), ("factor", np.where(np.isnan(seg), np.nan, 1.0).astype("<f4"))):
            np.hstack([np.float32(i0), vals]).astype("<f4").tofile(d / f"{field}.day.bin")
        lines.append(f"{sym.upper()}\t{first:%Y-%m-%d}\t{last:%Y-%m-%d}")
    (out / "instruments/all.txt").write_text("\n".join(lines)); return out

def features(prices: pd.DataFrame, exprs: dict[str, str], workdir: Path | None = None) -> pd.DataFrame:
    wd = Path(workdir or tempfile.mkdtemp(prefix="qal_qlib_")); wd.mkdir(parents=True, exist_ok=True)
    qdir = write_qlib_dir(prices, wd / "qlib_data"); out = wd / "features.parquet"
    run_in_env("ml", "qlib_worker.py", {"provider_uri": str(qdir), "exprs": exprs, "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()), "out": str(out)})
    df = pd.read_parquet(out); df.index = pd.MultiIndex.from_arrays([pd.to_datetime(df["datetime"]), df["instrument"].str.upper()], names=["date", "symbol"])
    return df.drop(columns=["datetime", "instrument"]).unstack("symbol").swaplevel(axis=1).sort_index(axis=1)
