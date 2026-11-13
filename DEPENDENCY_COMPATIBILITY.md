# Dependency Compatibility

Host: macOS arm64, system Python 3.14. Probe: each library's **published PyPI wheel** installed with `--only-binary :all:` into a fresh isolated Python 3.12 venv (`scripts/probe_envs.py`), then `import` attempted. Clone source was **not** installed or executed. Results in `artifacts/compat_probe.json`.

'import ok' means only that the package imports - **not** that an adapter works. Operational status requires a passing test (see IMPLEMENTATION_STATUS.md).

| repo | declared python | PyPI py3.12 install | import | version/error | adapter status |
|---|---|---|---|---|---|
| qlib | >=3.8.0 | yes | yes | 0.9.7 (source build under sandbox-exec; needed --prerelease=allow for gym) | operational |
| vectorbt | >=3.11,<3.15 | yes | yes | 1.2.0 | untested |
| quantstats | >=3.10 | yes | yes | 0.0.86 | operational |
| Lean | n/a | not probed | - | no PyPI probe (see notes) | unsupported |
| machine-learning-for-trading | >=3.14,<3.15 | not probed | - | no PyPI probe (see notes) | reference_only |
| PyPortfolioOpt | >=3.10,<3.15 | yes | NO | -autonomous-lab/envs/probe/PyPortfolioOpt/lib/python3.12/site-packages/skbase/utils/dependencies/_dependencies | operational |
| Riskfolio-Lib | n/a | yes | yes | 7.4.1 | untested |
| skfolio | >=3.10 | yes | yes | 1.12.9 | untested |
| QuantInvestStrats | >=3.11 | yes | yes | ? | untested |
| alphalens-reloaded | >=3.11,<3.16 | yes | yes | 0.5.1 | untested |
| arbitragelab | ^3.8 | yes | yes | py3.12 FAILED (pins scikit-learn 1.3.0, no wheel); works on py3.11 only after manual pins setuptools==80.9.0 + typing_extensions==4.12.2 | untested |
| nautilus_trader | n/a | yes | yes | 1.231.0 | untested |
| zipline-reloaded | >=3.11,<3.15 | yes | yes | 3.2.1 | untested |
| backtrader | n/a | yes | yes | 75: SyntaxWarning: invalid escape sequence '\*'
  - callback(msg, \*args, \*\*kwargs)
/Users/newman/quantalpha/quant-autonomous-lab/envs/probe/backtrader/lib/python3.12/site-packages/backtrader/cerebro.py:717: SyntaxWarning: invalid escape sequence '\*'
  - callback(data, status, \*args, \*\*kwargs) | reference_only |
| bt | >=3.9 | yes | yes | 1.4.1 | untested |
| QuantLib | n/a | yes | yes | 1.44 | untested |
| py_vollib | >=3.9,<3.13 | yes | yes | ?
<string>:1: DeprecationWarning: py_vollib is deprecated and will be removed in a future release; please import from vollib instead. | untested |
| RD-Agent | >=3.10 | yes | yes | installed; no __version__ attr | operational |
| FinRL-Trading | >=3.11 | not probed | - | no PyPI probe (see notes) | untested |
| FinRL | ^3.7 | yes | NO | "/Users/newman/quantalpha/quant-autonomous-lab/envs/probe/FinRL/lib/python3.12/site-packages/finrl/meta/env_st | untested |
| OpenBB | n/a | yes | yes | ibit@2.0.0, ecb@2.0.0, eia@2.0.0, famafrench@2.0.0, federal_reserve@2.0.0, finra@2.0.0, fred@2.0.0, imf@3.0.0, jodi@2.0.0, nasdaq@2.0.0, news@2.0.0, oecd@2.0.0, openbb_charting@4.0.0, sec@2.0.0, tmx@2.0.0, us_eia@2.0.0, us_treasury@2.0.0, uscongress@2.0.0, usda@2.0.0, ustreasury@2.0.0

Building...
? | untested |
| awesome-quant | >=3.11 | not probed | - | no PyPI probe (directory) | reference_only |
| awesome-systematic-trading | n/a | not probed | - | no PyPI probe (directory) | reference_only |
| awesome-open-source-trading | n/a | not probed | - | no PyPI probe (directory) | reference_only |

## Environment strategy

- `envs/core` (Python 3.12): numpy/pandas/scipy/pyarrow/matplotlib + yfinance, PyPortfolioOpt, quantstats. The only env the CLI needs.
- One venv **per heavy library** (`envs/<name>`), never a shared global install. Conflicts seen in practice: numba/llvmlite pins (vectorbt), cvxpy solver stacks (portfolio libs), torch/gym pins (FinRL), old pins (arbitragelab, backtrader), `python_requires<3.13` (py_vollib).
- System Python 3.14 is **too new** for several libs (numba-family, zipline, qlib classifiers): use 3.11/3.12 envs.
- Not buildable here by design: Lean (needs .NET/Docker), QuantLib C++ source, nautilus_trader Rust source: use released wheels.
