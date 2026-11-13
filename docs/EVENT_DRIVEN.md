# Event-driven backtest (nautilus_trader) - fidelity cross-check of the vector engine

**What this is:** the 12-month SPY long/flat trend baseline re-run event-driven in nautilus_trader's `BacktestEngine`, compared
with `engine/backtesting/vector.py` on the same cached real data, train+validation only (2005-01-03..2019-12-31).
**What it is not:** a strategy evaluation. No Criteria/DSR verdict, nothing written to any QAE store, no protected-test look.

## Run
```bash
envs/core/bin/python -m engine.eventsim.compare                 # -> artifacts/nautilus/comparison.json, trades_*.csv, daily_diff_*.csv
envs/core/bin/python -m pytest tests/test_nautilus.py           # 10 fast tests (synthetic) ; add -m slow for the real-data test
envs/core/bin/python artifacts/nautilus/mutation_check.py       # 8 mutants -> artifacts/nautilus/mutation_results.json
```

## Environment and licence
- `envs/nautilus`: `uv venv --python 3.12 envs/nautilus && uv pip install --python envs/nautilus/bin/python --only-binary :all: nautilus_trader pandas pyarrow`
  -> nautilus_trader **1.231.0** (PyPI wheel), pandas 3.1.1, pyarrow 26.0.1, CPython 3.12.12.
- LGPL-3.0: used unmodified as an installed wheel through its public Python API from a separate worker process
  (`adapters/workers/nautilus_worker.py`, launched by `adapters/envrun.run_in_env`). Nothing vendored, copied or patched; the
  `external/` clone and `/Users/newman/quantalpha/vendor/nautilus_trader` are not used.
- `adapters/nautilus_adapter.py` (core side) never imports nautilus; data crosses as a parquet file + a JSON result file.

## No execution path (ARCHITECTURE principle 7)
Only `BacktestEngine` with one simulated venue `SIM` (CASH account, NETTING OMS, $ starting balance). No `TradingNode`/`LiveNode`,
no live execution or data clients, no `nautilus_trader.adapters.*` / `nautilus_trader.live.*` imports, no credentials, no network.
`tests/test_nautilus.py::test_worker_imports_no_live_or_adapter_clients` parses the worker's AST and fails on any such import or name
(and on `importlib`/`__import__`, to close the dynamic-import loophole).

## Protected window
`nautilus_adapter.assert_dev_only` raises `ProtectedWindowAccess` for any date > 2019-12-31 (= `Criteria().splits.val_end`;
stricter than test-start because it also refuses the embargo days). It runs before any file is written or the worker launched.
There is **no override parameter** (tested). The worker re-checks independently and refuses (exit != 0) bars after 2019-12-31.

## Configuration (pre-registered in `artifacts/nautilus/prereg.json`, 21:38 UTC, before any real-data run)
| item | choice |
|---|---|
| data | cached Yahoo SPY adjusted close (`datasets/1018b88b67c0.parquet`, sha256 47c8d715...), `load_prices(offline=True)`, sliced to <= 2019-12-31 immediately |
| bars | close-only data, so each daily bar is O=H=L=C=adj close, ts = date 21:00 UTC, volume 1e9 (no liquidity limit) |
| instrument | `Equity` SPY.SIM, lot size 1, price precision 4 (adjusted prices have more decimals; 4dp keeps rounding < 0.005 bp) |
| signal | at a month-end session t: close_t / close_(month-end t-12) - 1 > 0 -> long, else flat. Month-end sessions are passed to the strategy in advance (role of a published exchange calendar), derived with the same rule as `trend_signal` |
| fee model | `MakerTakerFeeModel`, taker_fee = cost_bps/1e4 = 0.0002 of fill notional (commission in cents) |
| fill model | `FillModel(prob_slippage=0)`, `bar_execution=True`, L1 book |
| sizing | long = floor(equity / (close*(1+fee))) whole shares; flat = sell whole position; no rebalancing while long |
| primary mode | `same_bar_close` |
| sensitivities (declared in advance) | `next_bar_close` ($1m); `same_bar_close` with $10k (to expose integer-share/cash-residual effects) |

### Execution semantics (verified from fill prices, not assumed)
- **same_bar_close (primary):** the market order is submitted inside `on_bar(t)`. With bar execution the simulated book is at bar t's
  close when `on_bar` runs, so nautilus fills it immediately **at close t** (all 11 real fills within 0.005 bp of the close, the
  difference being 4dp price rounding). This is a market-on-close assumption - the same one the vector engine makes (weight decided
  at close t earns the t+1 return). It is optimistic: in reality the close is not known until it prints.
- **next_bar_close (sensitivity):** decision at close t, order submitted in `on_bar(t+1)`, filled at close t+1 (`fill_lag_sessions = 1` for every fill).
- Equity is recorded at the start of each `on_bar(t)` as account cash + qty x close_t, i.e. after the fill at close t-1, so a fee paid
  at close t-1 lands in day t's return - the same day the vector engine charges it.

## Results (real SPY, window 2006-02-01..2019-12-31, 3,503 daily returns)
| run | total return | Sharpe | trades | total costs | daily corr | tracking err (ann) | max abs daily diff |
|---|---|---|---|---|---|---|---|
| vector engine ($1m notional) | 216.985% | 0.69213 | 11 | $4,057.22 (0.22% of equity in return units) | - | - | - |
| nautilus same_bar_close, $1m (primary) | 216.970% | 0.69212 | 11 | $4,056.56 | 0.9999999992 | 0.00065% | 0.028 bp (2015-11-02) |
| nautilus next_bar_close, $1m | 202.025% | 0.66526 | 11 | $3,950.73 | 0.99751 | 0.94% | 250 bp (2012-06-01) |
| nautilus same_bar_close, $10k | 215.389% | 0.69149 | 11 | $40.28 (vector: $40.57) | 0.9999994 | 0.052% | 2.4 bp (2019-08-05) |

Primary: 0 days differ by more than 1 bp; sum of daily differences -0.55 bp. Final equity $3,169,695 (10,830 shares + $242.49 cash).
Per-trade tables: `artifacts/nautilus/trades_<run>.csv` (decision date, side, vector implied price = close at decision, nautilus fill date/qty/price, price diff bp, notionals, vector cost vs nautilus commission, cash left after the fill, fill lag).

## Divergence sources (each checked against the numbers)
1. **Fill timing.** Primary: none (both trade at the decision close). next_bar_close: every trade fills one session later; per-trade price
   differences range from -252 bp (sell decided 2012-05-31, filled 2012-06-01 after a -2.5% day) to +161 bp; this alone takes total
   return from 217% to 202% and Sharpe from 0.692 to 0.665. That is the size of the "you cannot trade at the close you computed on" haircut
   for this strategy (11 trades in 14 years); a strategy that trades more would show a bigger one.
2. **Integer shares + cash residual.** nautilus holds whole shares and leaves the remainder (plus room for the fee) in cash earning 0.
   With $1m the residual is $7-$176 after buys (mean cash weight while long 0.003%), cash-drag estimate -0.45 bp, vs observed total
   daily-diff sum -0.55 bp. With $10k it is $42-$224 (0.39% mean cash weight), drag estimate -59.70 bp vs observed sum -59.70 bp:
   the $10k gap is almost entirely cash drag.
3. **Fees on notional vs on weight change.** Vector charges 2 bp x |dw| x prior equity (always the full equity); nautilus charges 2 bp
   of the actual fill notional (slightly less than equity because of the residual cash) rounded to cents. Gap: $0.66 over 11 trades at $1m.
4. **Price rounding.** 4dp price precision: fill vs close differences <= 0.005 bp.
5. **Dividends / adjusted prices.** No divergence *between* the engines because both consume the same dividend-adjusted close (dividends
   are reinvested implicitly in the price). But this means the nautilus "fill prices" are adjusted prices, not prices that ever traded
   (e.g. 2006-01-31 fill at 87.33, while SPY actually closed near 128 that day), so the share counts are fictional. A realistic
   event-driven run would need unadjusted OHLC + dividend cash events, which the cached dataset does not contain. Not modelled.
6. **Open prices.** The cache has no open/high/low, so a next-bar-open execution could not be simulated; the bars are single-price.

## Disclosures (changes after seeing results)
- **No simulation configuration was changed after seeing results.** All three runs are the pre-registered ones.
- One diagnostic was corrected after the first run: the cash-drag estimate used the cash weight recorded *before* the fill at close t-1
  (shifted one bar too far), giving -4.2% instead of -0.45 bp, contradicting the observed 0.003% mean cash weight. It is a pure
  post-processing bug in `engine/eventsim/compare.py`; the fixed version reads the cash held over day t. Simulation outputs (returns, fills,
  costs) were identical between the two runs.

## Limitations / not done
- Single asset, long/flat, close-only data. No slippage, latency, partial fills, borrowing, short selling, or dividend cash flows modelled.
- The month-end calendar comes from the data's own index (proxy for an exchange calendar).
- The comparison window ends 2019-12-31; the protected window was never run.
- Not wired into the QAE cycle, registry status or CLI (shared files; proposed to CTO).
