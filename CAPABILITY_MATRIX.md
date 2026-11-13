# Capability Matrix

Scores: **3** primary strength · **2** solid · **1** minor · blank none. These are analyst judgements from READMEs and static inspection (see `scripts/curated.py`), **not benchmarks**.

| capability | best-supported (3) | also solid (2) | minor (1) |
|---|---|---|---|
| market data acquisition | OpenBB | Lean | machine-learning-for-trading, FinRL-Trading, FinRL |
| historical data normalization | OpenBB | qlib, Lean, nautilus_trader, zipline-reloaded | - |
| time series analytics | - | vectorbt, quantstats, machine-learning-for-trading, QuantInvestStrats, arbitragelab | qlib, QuantLib, OpenBB |
| statistical analysis | arbitragelab | quantstats, machine-learning-for-trading, alphalens-reloaded | PyPortfolioOpt, Riskfolio-Lib, skfolio |
| factor research | qlib, alphalens-reloaded | machine-learning-for-trading, zipline-reloaded | - |
| alpha discovery | qlib | alphalens-reloaded, arbitragelab, RD-Agent | machine-learning-for-trading |
| machine learning | qlib, machine-learning-for-trading | skfolio, RD-Agent | FinRL |
| reinforcement learning | FinRL | FinRL-Trading | machine-learning-for-trading |
| strategy generation | - | vectorbt, arbitragelab, RD-Agent, FinRL | qlib, Lean, backtrader, bt, FinRL-Trading |
| strategy optimization | vectorbt | skfolio | PyPortfolioOpt |
| portfolio construction | PyPortfolioOpt, Riskfolio-Lib, skfolio | bt | QuantInvestStrats, arbitragelab |
| risk management | Riskfolio-Lib, skfolio | quantstats | vectorbt, PyPortfolioOpt, QuantInvestStrats, QuantLib |
| derivatives pricing | QuantLib | Lean | py_vollib |
| options analytics | QuantLib, py_vollib | Lean | nautilus_trader, OpenBB |
| futures research | Lean | nautilus_trader | OpenBB |
| backtesting | vectorbt, Lean, nautilus_trader, zipline-reloaded | qlib, backtrader, bt | machine-learning-for-trading, QuantInvestStrats, FinRL-Trading, FinRL |
| execution simulation | Lean, nautilus_trader | zipline-reloaded, backtrader | vectorbt, FinRL-Trading |
| performance attribution | - | vectorbt, quantstats, Lean, QuantInvestStrats, alphalens-reloaded | qlib, Riskfolio-Lib, skfolio, bt |
| research automation | RD-Agent | qlib | - |
| agent memory evaluation | - | RD-Agent | - |

## Full grid

| repo | market data acquisition | historical data normalization | time series analytics | statistical analysis | factor research | alpha discovery | machine learning | reinforcement learning | strategy generation | strategy optimization | portfolio construction | risk management | derivatives pricing | options analytics | futures research | backtesting | execution simulation | performance attribution | research automation | agent memory evaluation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qlib |  | 2 | 1 |  | 3 | 3 | 3 |  | 1 |  |  |  |  |  |  | 2 |  | 1 | 2 |  |
| vectorbt |  |  | 2 |  |  |  |  |  | 2 | 3 |  | 1 |  |  |  | 3 | 1 | 2 |  |  |
| quantstats |  |  | 2 | 2 |  |  |  |  |  |  |  | 2 |  |  |  |  |  | 2 |  |  |
| Lean | 2 | 2 |  |  |  |  |  |  | 1 |  |  |  | 2 | 2 | 3 | 3 | 3 | 2 |  |  |
| machine-learning-for-trading | 1 |  | 2 | 2 | 2 | 1 | 3 | 1 |  |  |  |  |  |  |  | 1 |  |  | 0 |  |
| PyPortfolioOpt |  |  |  | 1 |  |  |  |  |  | 1 | 3 | 1 |  |  |  |  |  |  |  |  |
| Riskfolio-Lib |  |  |  | 1 |  |  |  |  |  |  | 3 | 3 |  |  |  |  |  | 1 |  |  |
| skfolio |  |  |  | 1 |  |  | 2 |  |  | 2 | 3 | 3 |  |  |  |  |  | 1 |  |  |
| QuantInvestStrats |  |  | 2 |  |  |  |  |  |  |  | 1 | 1 |  |  |  | 1 |  | 2 |  |  |
| alphalens-reloaded |  |  |  | 2 | 3 | 2 |  |  |  |  |  |  |  |  |  |  |  | 2 |  |  |
| arbitragelab |  |  | 2 | 3 |  | 2 |  |  | 2 |  | 1 |  |  |  |  |  |  |  |  |  |
| nautilus_trader |  | 2 |  |  |  |  |  |  |  |  |  |  |  | 1 | 2 | 3 | 3 |  |  |  |
| zipline-reloaded |  | 2 |  |  | 2 |  |  |  |  |  |  |  |  |  |  | 3 | 2 |  |  |  |
| backtrader |  |  |  |  |  |  |  |  | 1 |  |  |  |  |  |  | 2 | 2 |  |  |  |
| bt |  |  |  |  |  |  |  |  | 1 |  | 2 |  |  |  |  | 2 |  | 1 |  |  |
| QuantLib |  |  | 1 |  |  |  |  |  |  |  |  | 1 | 3 | 3 |  |  |  |  |  |  |
| py_vollib |  |  |  |  |  |  |  |  |  |  |  |  | 1 | 3 |  |  |  |  |  |  |
| RD-Agent |  |  |  |  |  | 2 | 2 |  | 2 |  |  |  |  |  |  |  |  |  | 3 | 2 |
| FinRL-Trading | 1 |  |  |  |  |  |  | 2 | 1 |  |  |  |  |  |  | 1 | 1 |  |  |  |
| FinRL | 1 |  |  |  |  |  | 1 | 3 | 2 |  |  |  |  |  |  | 1 |  |  |  |  |
| OpenBB | 3 | 3 | 1 |  |  |  |  |  |  |  |  |  |  | 1 | 1 |  |  |  |  |  |
| awesome-quant |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| awesome-systematic-trading |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| awesome-open-source-trading |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Capabilities with NO strong (>=3) coverage in this set

- time series analytics
- strategy generation
- performance attribution
- agent memory evaluation

## Overlap clusters (pick ONE per cluster for the core)

- **Portfolio optimisation**: PyPortfolioOpt, skfolio, Riskfolio-Lib (+ bt). Core uses own scipy optimiser; PyPortfolioOpt adapter operational; skfolio preferred next.
- **Backtesting**: vectorbt (vectorised sweeps), bt (allocation), zipline-reloaded / backtrader (event-driven), nautilus_trader / Lean (institutional). Core uses own ~40-line vectorised engine.
- **Performance analytics**: quantstats, QuantInvestStrats, vectorbt stats, alphalens (factor-level).
- **Options/derivatives**: py_vollib (BS/Greeks/IV, tiny) vs QuantLib (full curve/exotics).
- **RL**: FinRL vs FinRL-Trading. **ML alpha**: qlib (+ RD-Agent on top). **Data**: OpenBB vs yfinance (core).
- **Directories**: awesome-quant / awesome-systematic-trading / awesome-open-source-trading (mostly overlapping link lists).
