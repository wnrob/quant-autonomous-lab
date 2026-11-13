# Point-in-time universe and survivorship (task 6)

Files: `engine/data/pit.py`, `configs/universe_pit.json`, `engine/data/ingest.py::check_availability` (additive),
`tests/test_pit.py`, `artifacts/pit/{build_pit_json.py,mutation_check.py,impact.py,impact.json}`.
`ETF24` / `UNIVERSES` in `engine/data/universe.py` are untouched.

## Table
Per symbol: `official_inception` (agent knowledge, source "agent knowledge, verify", confidence medium - NOT checked against issuer
filings), `first_valid_bar` (first non-NaN bar in the cached Yahoo frame `datasets/dcfd243d77e6`, sha in the JSON, with a
`left_censored` flag), `availability_date = max(both)` (later wins, conservative), optional `delisting`
`{last_trade_date, successor, source, confidence}` (null for all 24; null means "still listed at cache date").
The cache starts 2005-01-03, so 20 of 24 symbols are left-censored: their availability date is the window start, not a real listing date.
Regenerate with `envs/core/bin/python artifacts/pit/build_pit_json.py` (cache only, no network).

## API (`engine.data.pit`)
- `universe_at(as_of, universe="etf24")` - members with availability <= as_of and as_of <= last_trade_date.
- `pit_prices(prices, as_of, members=None)` - rows <= as_of, member columns only; raises `PITViolation` for a non-member request, a
  missing column, or any non-NaN value before availability / after delisting. Never fills.
- `validate_frame_pit(df)` / `ingest.check_availability(df)` - report (`ok`, `violations`, `info`), never modifies the frame.
- `survivorship_flag(universe)` - dict to attach to experiment metrics: `survivorship_biased: True`, reason, member counts, table version.
- `availability_by_year(years, prices=None)`.

## Real impact (cached data, train/validation window <= 2019-12-31 only; `artifacts/pit/impact.json`)
Late starters: HYG first bar 2007-04-11 (official 2007-04-04, so Yahoo is later -> 2007-04-11), SLV 2006-04-28 (official 04-21),
DBC 2006-02-06 (official 02-03), USO 2006-04-10. All others predate the cache.

| year | available on first bday | available Dec 31 |
|---|---|---|
| 2005 | 20 | 20 |
| 2006 | 20 | 23 (DBC Feb, USO Apr, SLV Apr) |
| 2007 | 23 | 24 (HYG Apr) |
| 2008-2026 | 24 | 24 |

So every experiment whose train window starts 2005 includes 4 symbols that are NaN for 0.3-2.3 years. Existing code handles this correctly:
- `alpha.py` builds scores from `px.shift` ratios / rolling std, which are NaN until a full lookback exists; `row = f.loc[d].dropna()` drops them; `min_assets=8` is met by the 20 older ETFs. Over all 20 candidate specs, weight-days on a symbol before its availability date = **0**. First held: DBC 2007-12 (L252), USO 2008-01, SLV 2008-03, HYG 2008-09 (momentum L252 S21 va example).
- `features.py::build_panel` rows require all features non-NaN: 0 rows before availability; first rows HYG 2008-04-30, SLV 2007-05-31, DBC 2007-02-28, USO 2007-04-30.
- The equal-weight benchmark (`alpha.py:run_xs`, `avail = ret.notna()`) averages only assets with a return that day (20 names in Jan 2005), so it does not count unlisted symbols.
- Pandas 3.1.1 here: `pct_change()` no longer forward-fills (and leading NaNs would not be padded anyway).
- `ingest.load_prices` explicitly does not forward-fill and notes NaN cells.
- `portfolio.py` consumes sleeves built from the above; `px[sym].pct_change().dropna()` for buy-and-hold sleeves is fine. No fill of prices found.

No pre-inception leakage found. The PIT check on the cached frame returns `ok: True`, zero violations.

## Issues / residual bias (read-only findings, nothing edited)
1. **Survivorship / selection bias is real and unfixable with local data.** All 24 are survivors picked with hindsight (DBC, USO, SLV,
   HYG are large today). Effect: levels of long-only returns and the equal-weight benchmark are biased up; the direction for active
   returns is unknown. Not mitigated; flagged only. Any experiment should attach `survivorship_flag()`.
   Proposed (shared-file, CTO): in `engine/agents/alpha.py` evaluate_xs, `m["pit"] = survivorship_flag(spec["universe"])` right after `m = {...}` (alpha.py:~65), same in the ML/portfolio metrics dicts.
2. **Seasoning.** Newly launched ETFs (first months of HYG/SLV/DBC/USO) have thin liquidity; availability = first bar does not mean tradeable at modeled cost. Alpha needs L+ days of history anyway (>= 21 days; reversal L21 enters after about one month), so exposure is small but nonzero for short lookbacks. Proposal: an optional `seasoning_days` argument in `universe_at` (not implemented).
3. **Left-censoring.** `n_first_bday` for 2005 is 20 because the cache starts 2005-01-03, not because 20 ETFs existed. Extending the history back needs a refetch (not done; no network).
4. `alpha.py:run_xs` `W.ffill().fillna(0.0)` after month-end ranks holds weights through a later delisting; with real delistings a post-delist price would be NaN and `ret.fillna(0.0)` would hold the dead asset at 0% return (understating loss). Not triggered today (no delistings); propose using `pit.universe_at` per rebalance date once delisted funds exist.

## Delisted / merged ETFs: documented gap
Checked `LAB/datasets` (4 parquet files, all current ETFs or synthetic) and `/Users/newman/quantalpha/data` (futures/MES-type data only):
**no delisted-ETF price history exists locally.** No network call was made. Yahoo generally returns nothing for delisted tickers, so none was attempted.
Candidate delisted/merged funds in these asset classes, **known to the agent, unverified, no histories invented**: PowerShares DB G10 Currency
Harvest (DBV), PowerShares DB Silver (DBS) / other DB single-commodity funds closed in 2023-2024 (exact dates unknown), Guggenheim BRIC (EEB, closed/merged ~2018),
assorted closed leveraged/inverse and single-country iShares/SPDR funds. Treat names and dates as leads to verify against issuer notices, then add
to `configs/universe_pit.json` with a `delisting` record and a price source.

## Tests
`tests/test_pit.py`: 9 tests (synthetic fixtures labelled), plus table consistency (later-date rule, order equals ETF24) and real-cache check.
Mutation check (`artifacts/pit/mutation_check.py`): 8 mutants, all caught, file restored.
