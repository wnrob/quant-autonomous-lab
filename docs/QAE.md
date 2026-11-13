# Quant Autonomous Engine (QAE) - first vertical slice

## Run it
```bash
cd quant-autonomous-lab
envs/core/bin/qae cycle --max-experiments 4     # baseline + proposer-driven follow-ups (real SPY data, cached)
envs/core/bin/qae status                        # experiments + decisions
envs/core/bin/qae alpha --max-experiments 6     # cross-sectional ETF factor search
envs/core/bin/qae portfolio                     # allocator comparison (dev window only)
envs/core/bin/qae ml                            # ML ladder (dev window only)
envs/core/bin/qae search "drawdown"             # full-text memory search
envs/core/bin/qae serve --port 8000             # API + dashboard -> http://127.0.0.1:8000
envs/core/bin/qae scheduler tick|list|pause|resume|cancel|retry [job]
envs/core/bin/qae forward log|score|status|verify|demo   # forward SHADOW evaluation (approve = human-only, interactive TTY)
envs/core/bin/qae code propose|status|verify     # coding-agent patch workflow (approve/apply = human-only)
envs/core/bin/qae futures-audit                  # read-only audit of the user's futures data (descriptive only)
envs/core/bin/qae eventsim                       # vector vs nautilus event-driven trend baseline (dev window only)
envs/core/bin/qae families qlib|pairs            # qlib-model / arbitragelab-pairs families (scratch store only)
envs/core/bin/python -m pytest -m "not slow"     # see IMPLEMENTATION_STATUS.md for the current count
```
State lives in `artifacts/qae/` (`qae.db`, QuantStats HTML reports). Override with `QAE_HOME`.

## Loop (engine/orchestration/cycle.py)
```
OBSERVE (data quality) -> HYPOTHESIZE (stat agent) -> DESIGN (spec dict + frozen Criteria)
 -> EXPERIMENT (backtest w/ costs) -> EVALUATE (validation window, DSR, bootstrap, cost stress, risk gate)
 -> CRITIQUE (proposer reads failure analysis + memory) -> IMPROVE (new falsifiable variant, dedup'd)
 -> VALIDATE (protected test: once per spec, global look cap) -> PROMOTE/REJECT (only to CANDIDATE) -> REMEMBER (lessons incl. negative)
```
## What is real vs not
- **Real:** the statistics, splits, protected-test accounting, risk gate, dedup, budgets, scheduler state, paper broker, memory, API, dashboard.
- **Not ML:** no model is trained. The "improvement" mechanism is rule-based proposal + grid-neighbour search, bounded by multiple-testing correction. **Optional local open-source LLM (`--llm`, Ollama qwen2.5:14b, loopback only) proposes specs inside a strict schema; it never trains, and no LLM weights change.** Paid-LLM $ and external API budgets stay 0; local calls/tokens are capped per day.
- **Promotion stops at CANDIDATE in the research store.** `store.set_status` still raises `PermissionError` for SHADOW/PAPER_APPROVED. Forward status lives only in the human-approval chain of `engine/forward/approval.py` (interactive TTY + typed phrase; refuses scheduler/agent/API code paths). No strategy is in SHADOW; nothing in the main store is CANDIDATE.
- **Frozen rules:** `Criteria`/`RiskLimits` are frozen dataclasses; their hash is stored with every experiment, so changing a threshold never re-judges old results silently.
- **Protected test:** a spec's test window is evaluated at most once and only after validation passes; `max_test_looks` caps the total across all specs.

## Deviations from the brief (documented, not hidden)
| brief | status |
|---|---|
| Qlib, RD-Agent integration | UNBLOCKED (you approved): built from source under `scripts/sandbox_build.sh`, import verified in `envs/ml`, `envs/agents`. adapters now exist and are tested but narrow: qlib = expression engine + Linear/LGB model family (NULL result); arbitragelab = Engle-Granger pairs family with BH-FDR (NULL, 0/276); RD-Agent = LLM backend only (its loops stay off) |
| Next.js dashboard | static HTML dashboard served by FastAPI (8 views incl. Risk Center, Model Lab, Portfolio Intelligence, Research Lab; read-only endpoints, explicit no-data states; page NOT visually inspected) |
| MLflow, Optuna, APScheduler, PostgreSQL, Docker | not used; SQLite + own tracking tables + poll scheduler. Optuna fits the proposer later |
| Agents 4, 5, 9 | implemented but narrow (see IMPLEMENTATION_STATUS, docs/ML.md, docs/PORTFOLIO.md); results so far are null / risk-reduction only |
| Futures rolls | roll/back-adjust/carry/stale-contract tooling built (docs/FUTURES.md); user's data has no per-contract prices, so real carry NOT measured |
| Options | py_vollib + QuantLib IV/Greeks adapter operational (docs/OPTIONS.md); local options data is post-2024 only, so descriptive use only, no options strategy research |
| Event-driven sim | nautilus_trader BacktestEngine cross-check of the trend baseline, dev window only (docs/EVENT_DRIVEN.md) |
| Survivorship, point-in-time | PIT inception table + availability checks (docs/PIT_UNIVERSE.md); ETF24 flagged survivorship-biased; no delisted-ETF data available |
| Autonomous coding-agent/PR workflow (Part 6) | propose-only workflow built (docs/CODING_AGENT.md); apply needs a human approval bound to the patch hash; LLM patch path tested with a fake transport only |
| Forward shadow evaluation | built (docs/FORWARD.md), paper only; 1 demo signal in a scratch store, 0 scored days; nothing is being tracked in the main store |

## Local LLM (you approved open-source, high-parameter)
- Model: `qwen2.5:14b` (Apache-2.0, ~14.7B params, 9 GB) via Ollama. Hardware is 17 GB RAM, so 30B+ models would not fit; this is the largest practical size.
- `qae cycle --llm` -> `engine/agents/llm.py`. Guardrails (all tested with a fake transport): loopback-only host; strict schema (out-of-range values rejected, never clamped); cannot alter asset/costs/family/criteria; same dedup, same multiple-testing trial count, same protected-test accounting; per-day call/token caps; falls back to rule-based proposer if the model is down.
- Measure value, don't assume it: `GET /api/proposal_yield` compares experiments / validation passes / test passes per proposal source.
- First live run: 3 valid proposals, 0 improved the baseline. Controlled A/B (docs/AB_PROPOSERS.md, 5 rounds x 12 experiments, validation-only): **LLM not distinguishable from random-spec sampling (p>0.7)**; no demonstrated value yet.

## Sandboxed source builds
`scripts/sandbox_build.sh <env> <spec>`: macOS `sandbox-exec` + scrubbed env: secret dirs (`~/.ssh`, `~/.aws`, `~/.claude`, keychains, trading data dirs) unreadable, writes only inside the env/tmp/uv cache (blocking verified). **Network is allowed** (needed to fetch deps) and this is not a container (Docker is not installed); it limits a build's blast radius, it does not make untrusted code safe to run later. Import-time code from these libs is NOT sandboxed.
