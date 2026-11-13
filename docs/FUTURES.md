# Futures term structure + roll adapter (engine/futures)

Status: built, tested (28 tests in `tests/test_futures.py`, 13/13 mutants caught). Run on the user's real local data as a
**read-only, descriptive audit**. Real carry/roll yield could **not** be computed: the local data has no per-contract prices
(see `artifacts/futures/data_gap.md`).

> **Protected era.** All local futures data (2024-08..2026-08) is after 2019-12-31, i.e. inside the lab's protected test
> window. This work is data engineering only: no strategy was backtested, selected or evaluated on it, and the experiment
> store and its `test_access` ledger were not opened. Outputs go only to `artifacts/futures/`.

## Modules
| module | what |
|---|---|
| `specs.py` | `ContractSpec` table for MES, MNQ, M2K, 6E, 6J, 6B, 6A, CL, NG, GC, MGC, SI, HG, ZN, ZB, ZC, ZW, ZS: multiplier, tick size/value, currency, exchange, active months, expiry rule, settlement, typical roll, `confidence`. Source string: *"CME contract specs as known to the agent; verify before trading"*. Not fetched from CME in this session. `ContractId` sorts by (year, month), never alphabetically. `parse_ticker` resolves the one-digit year against an observation date. |
| `calendar.py` | Approximate CME holiday calendar (US exchange holidays incl. Good Friday; Juneteenth from 2022). Last-trade rules: equity index = 3rd Friday; FX = 2 bdays before 3rd Wednesday; CL = 3 bdays before the 25th of the prior month (or before the preceding bday); NG = 3 bdays before the 1st of the month; COMEX metals = 3rd-last bday; ZN/ZB = 7th bday before the last bday; grains = bday before the 15th; FND = last bday of the prior month. |
| `rules.py` | Frozen, hashed roll rules. `CalendarRoll(n_bdays, ref)` rolls n bdays before the last trade or FND. `VolumeRoll(confirm, backstop)` rolls once the next contract outtraded the held one on `confirm` consecutive **prior** sessions; a missing value neither advances nor resets the count; it never rolls back and never skips a contract. `held_contracts()` and `roll_events()` produce the schedule. |
| `adjust.py` | `build_continuous(prices, held, "ratio"/"difference")` from per-contract closes. On a roll day the series earns the **new** contract's own return. It raises `DataError` if the new contract was not observed the day before the roll, rather than inventing a gap. Missing bars stay NaN. |
| `carry.py` | `roll_yield()`: annualised (ln F_front − ln F_next)·365/Δexpiry, only where both contracts are observed; positive = backwardation. |
| `stale.py` | `detect_stale()`: reports problems and never repairs. Checks: multi-contract timestamps, **back-switches** (the fetch_full signature), bars past last trade, bars past a rule's roll date, stale end, frozen prices, unadjusted roll gaps (info on a raw splice, error if the series claims to be adjusted), thin volume, and **thin segments** (≥N consecutive same-contract bars under 5% of median volume). |
| `real_data.py`, `gap_report.py` | The read-only audit of `/Users/newman/quantalpha/data` and the data-gap report. |

```bash
envs/core/bin/python -m pytest -p no:cacheprovider tests/test_futures*.py   # 28 passed
envs/core/bin/python -m engine.futures.real_data    # ~40 s, writes artifacts/futures/real_data_audit.json, roll_events_daily.csv, specs_table.json
envs/core/bin/python -m engine.futures.gap_report   # data_gap.md / data_gap.json
envs/core/bin/python artifacts/futures/mutate.py    # 13 mutants, all must be CAUGHT; files restored
```

## What the real data is
- `data/daily/*.parquet` (16 roots; 6A 6B 6E 6J CL GC HG M2K MES NG SI ZB ZC ZN ZS ZW): each file is **one unadjusted
  front-month splice**, with columns `open high low close volume transactions dollar_volume contract`. There is exactly one
  contract per row and **no per-contract history**: the user's `qa/daily.py:load_product` fetched each contract but cached
  only the splice. The roll is chosen by **same-day** volume with a forward-only ratchet (`qa/roll.py:45-75`). The splice is
  **not back-adjusted**. Note that `qa/roll.py:130 back_adjust` exists but is not applied to these files, and it would
  zero out the real roll-day move (gap = new(t) − old(t−1)).
- **Date labels are not trade dates.** Daily bars carry the vendor's session `window_start` date. Labels fall only on
  Sun–Thu, and a label means the next business day's session (MES label 2024-09-12 closes at 5626.50, which is the
  2024-09-13 session). The audit maps labels to trade dates heuristically. Exceptions: grain daytime-only sessions after a
  holiday, and the final row. There is one label collision per root on 2026-04-06 (Good Friday) for FX/rates, and one for
  grains on 2026-08-28. Collisions are reported, not merged. Any date join of this data with other sources is off by one
  day unless the mapping is applied.
- `data/cache/*_1m_*.parquet` (MES, MNQ, MGC; RTH and ETH): also single-contract splices. In all 13 files the audit found
  0 back-switches, 0 multi-contract timestamps, 0 bars past last trade and 0 session-length thin runs. These caches are
  clean of the fetch_full defect.

## The fetch_full bug (evidence)
The original source (`convexity/delta/data.py:95`, per the user's memory note) is **no longer on disk**. A search of
`/Users/newman` found no `def fetch_full`, and `/Users/newman/convexity copy/convexity` has no `delta/`. The evidence is
`/Users/newman/quantalpha/README.md:10-24` plus the memory note:
`fetch_full()` fetched every quarterly contract (e.g. MNQH6/M6/U6) over the **full** window, concatenated them in ticker
order, then deduplicated timestamps with `duplicated(keep="first")`. So the **oldest** contract won every minute it printed,
and a dead month trading 1 lot/min overwrote the liquid front month. Symptoms: 35% of MNQ 1m bars were stale-contract
noise, 37% of consecutive close-to-close moves exceeded 100 points, and median 5m ATR(20) read 251 vs a true 42.8. Tick-rule
delta read contract switches as ticks. Five pre-registered "no edge" studies ran on this data and are uninterpretable. The
live watcher (`fetch_recent_1m`, single ticker) was not affected.
`tests/test_futures.py::fetch_full_reconstruction` rebuilds this mechanism synthetically. The detector flags it through
hundreds of back-switches, past-roll-date bars and thin volume. A correct splice of the same walk passes.

## New hazard found in the current data (descriptive, not repaired)
`qa/roll.py:42` zero-fills missing per-contract volume, `:72` takes the argmax and `:73` ratchets the floor forward. If the
front month has no bar for one day, a thin deferred contract wins and the series can never return. On the real data:
- **6E**: 6EM6 → 6EU6 on trade date 2026-03-27, when U6 traded 103 lots (the day before, M6 traded ~171k). That was 54
  bdays before M6's last trade. There are 49 consecutive sessions (2026-03-27..06-04) on U6 at median volume 597, against
  a series median of 151,645.
- **GC**: GCJ6 → **GCV6** on 2026-03-27, skipping M6 and Q6. There are 84 consecutive thin sessions (median volume 336)
  through 2026-07-28, and the series stays on V6 (median 557) to the panel end. The roll gap that day was +4.8% (|z|>3).
- **NG**: the series ends 2025-05-28, 457 days before the panel end (already handled by the user's `drop_stale`).
- `qa/trend.py:42-44 returns()` says "Roll gaps are excluded, not traded through", but it runs a plain `pct_change` on the
  unadjusted panel. Roll-day returns are therefore included: they are 1.0–4.8% of total absolute daily return per root, and
  11% for NG (max single roll day: NG 25%, ZW 7.4%, GC 5.2%). `run_trend.py:21` also forward-fills up to 3 days. Both are
  read-only findings. Nothing under /Users/newman/quantalpha outside LAB was changed.

## Roll timing and gaps on real data (trade-date basis, business days before reference)
Equity index rolls 3–4 bdays before the 3rd Friday. FX rolls 1 bday before last trade. CL rolls 2–3 bdays before last trade.
GC/SI roll 1–2 bdays before FND, HG 3, ZN/ZB 1–2, and ZC/ZW 5–10 (with new-crop jumps of 23 for ZC and 62 for ZS). Roll
gaps beyond 3 robust sigmas: 6J 1, GC 2, MES 1, NG 3, ZW 3, others 0. Full list: `artifacts/futures/roll_events_daily.csv`
(146 rolls).

**Roll-gap proxy (NOT carry):** the mean over rolls of −gap·365/Δexpiry. Each gap = calendar spread + one day of market
noise, so the standard errors are wide. This is descriptive only; see the table in `data_gap.md`. Equity index reads about
−6%/yr (contango, consistent with rates minus dividends), metals and grains −5 to −15%/yr, ZN/ZB/6A/6B are
indistinguishable from 0, and CL is +5% ± 7%. The 6E and GC numbers include the bad ratchet rolls.

## Known limitations
- Holiday calendar is approximate. It lacks one-off closures such as 2025-01-09 (which shows up as a missing day for
  MES/M2K), and Juneteenth handling for equity-index expiry (June 2026: 3rd Friday = 19 Jun) should be verified.
- Specs come from the agent's knowledge. 6A tick (0.00005 here vs 0.0001 in the user's `qa/contracts.py`) and 6B tick
  are marked `confidence=medium`. MGC delivery mechanics are assumed to match GC. ZF/ZT are not covered.
- No per-contract data, so `build_continuous` and `roll_yield` run only on labelled synthetic fixtures. No OI-based roll.
- The trade-date mapping is a heuristic, not vendor metadata.
