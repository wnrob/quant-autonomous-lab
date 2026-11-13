# License Audit

Detected from each repo's LICENSE file text (then manually checked for ambiguous cases). **Not legal advice.**

| repo | license | commercial use | notes |
|---|---|---|---|
| qlib | MIT | permitted with attribution/notice |  |
| vectorbt | Apache-2.0 | permitted with attribution/notice |  |
| quantstats | Apache-2.0 | permitted with attribution/notice |  |
| Lean | Apache-2.0 | permitted with attribution/notice | Apache-2.0 code; QuantConnect data/cloud services have separate terms |
| machine-learning-for-trading | MIT | permitted with attribution/notice |  |
| PyPortfolioOpt | MIT | permitted with attribution/notice |  |
| Riskfolio-Lib | BSD-3-Clause | permitted with attribution/notice |  |
| skfolio | BSD-3-Clause | permitted with attribution/notice |  |
| QuantInvestStrats | MIT | permitted with attribution/notice |  |
| alphalens-reloaded | Apache-2.0 | permitted with attribution/notice |  |
| arbitragelab | BSD-3-Clause | permitted with attribution/notice | BSD code; Hudson & Thames commercial docs/content separate |
| nautilus_trader | LGPL-3.0 | permitted if dynamically linked; modifications to the library must be shared | LGPL-3.0: keep as separate installed dependency, do not vendor/modify |
| zipline-reloaded | Apache-2.0 | permitted with attribution/notice |  |
| backtrader | GPL-3.0 | RESTRICTED: copyleft obligations on distribution/derivatives; get legal review | GPL-3.0 strong copyleft + unmaintained -> reference only |
| bt | MIT | permitted with attribution/notice |  |
| QuantLib | BSD-style (QuantLib License) | permitted with attribution/notice | BSD-style; very many copyright holders |
| py_vollib | MIT | permitted with attribution/notice |  |
| RD-Agent | MIT | permitted with attribution/notice |  |
| FinRL-Trading | Apache-2.0 | permitted with attribution/notice |  |
| FinRL | MIT | permitted with attribution/notice |  |
| OpenBB | Apache-2.0 | permitted with attribution/notice | Apache-2.0 code; data providers have their own terms/keys |
| awesome-quant | NONE | NOT GRANTED: no license => all rights reserved; do not copy/redistribute | No LICENSE file -> all rights reserved; read-only |
| awesome-systematic-trading | NONE | NOT GRANTED: no license => all rights reserved; do not copy/redistribute | No LICENSE file -> all rights reserved; read-only |
| awesome-open-source-trading | CC0-1.0 | permitted with attribution/notice |  |

## Summary

- Permissive (MIT/Apache/BSD/CC0): qlib, vectorbt, quantstats, Lean, machine-learning-for-trading, PyPortfolioOpt, Riskfolio-Lib, skfolio, QuantInvestStrats, alphalens-reloaded, arbitragelab, zipline-reloaded, bt, QuantLib, py_vollib, RD-Agent, FinRL-Trading, FinRL, OpenBB, awesome-open-source-trading
- Weak copyleft: nautilus_trader (LGPL-3.0)
- Strong copyleft: backtrader (GPL-3.0)
- Unlicensed: awesome-quant, awesome-systematic-trading

## Policy applied

- External clones live under `external/` and are never copied into `engine/` or `adapters/`; the integration layer only `import`s released packages lazily.
- Third-party data (Yahoo via yfinance) is subject to Yahoo's terms: personal/research use; check before commercial use.
