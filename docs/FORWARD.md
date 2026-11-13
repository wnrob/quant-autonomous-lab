# Forward SHADOW evaluation (engine/forward)

Paper only. No real orders, no broker credentials, no network. Fills are simulated by `engine/backtesting/paper.py:PaperBroker`.

## What it does
```
CANDIDATE (research store)
   |  daily, after the close:   forward log     -> signals.jsonl   (hash-chained, written BEFORE the outcome exists)
   |  later, when bars exist:   forward score   -> reports/<spec_hash>.json (paper fills at the NEXT bar, costs, tracking)
   |  human at a terminal:      forward approve enter-shadow   -> approvals.jsonl  => forward status SHADOW
   |  >= frozen minimum evidence + human:  forward approve paper-approved          => PAPER_APPROVED (still paper)
```
Files live next to the store db: `<QAE_HOME>/forward/{signals.jsonl, approvals.jsonl, forward.db, reports/}`.

| module | role |
|---|---|
| `chain.py` | append-only JSONL hash chain; `verify()` detects edit / delete / reorder / insert / tail truncation |
| `signals.py` | `SignalLog.log()` with the backfill guards G1-G5; `LogPolicy` (frozen, hashed) |
| `scoring.py` | `score()` (S1-S4), `ForwardEvidence` (frozen, hashed), `load_forward_bars`, `backtest_expectation` |
| `approval.py` | `ApprovalLog.approve()` human gate (A1-A3), forward status = fold of the approval chain |
| `jobs.py` | automatable `forward_log` / `forward_score` handlers (offline cache only) |
| `cli.py`, `demo.py` | `python -m engine.forward.cli log|score|status|verify|approve|demo` |

## Signal record
`strategy_id, strategy_name, spec_hash, spec, as_of (last bar used), target_earliest, weights, ref_prices (close at as_of),
data_sha (sha256 of exactly the frame handed to the logger), dataset_sha, data_kind, n_bars, first_bar, code_version,
forward_code_version, logged_at (wall clock, UTC), data_fetched_at, as_of_bar_final, forward_status, log_policy_hash`,
wrapped as `{seq, prev, body, hash}` with `hash = sha256(prev + canonical_json({seq, prev, body}))`.

## Guards (each raises; each has a test; each was mutation-checked)
| id | rule |
|---|---|
| G1 | the frame handed to the logger contains a bar after `as_of` -> `BackfillError` |
| G2 | wall clock is at/after the close (16:00 New York) of the first weekday after `as_of` -> `BackfillError`. This is what stops "truncate the frame to 2019 and log a forward signal today". Holidays are not modelled; the rule can only refuse more |
| G3 | `as_of` later than today's market date -> `FutureBarError` |
| G4 | `as_of` not strictly later than the strategy's last logged `as_of` -> `DuplicateSignal` |
| G5 | strategy not CANDIDATE in the research store -> `PermissionError` |
| S1 | scorer handed any bar <= first logged `as_of` -> `ForwardWindowError`. The scorer therefore cannot read the protected window (2020-01-06 .. last bar at log time); `load_forward_bars` reads only rows `> as_of` from the parquet (predicate pushdown) |
| S2 | paper fill must be strictly after the signal's `as_of` -> `SameBarFillError` |
| S3 | chain must verify and every stored record must still satisfy G2/G3 |
| S4 | forward code never calls `record_test_access` / `test_seen` / `test_looks` and never reads `metrics["test"]`; the backtest expectation is the stored VALIDATION window only |
| A1 | approval refused if an automation env marker is set (`QAE_JOB_CONTEXT, QAE_AGENT_CONTEXT, QAE_SCHEDULER, CI, GITHUB_ACTIONS, CLAUDECODE, CLAUDE_CODE_ENTRYPOINT`) or any caller frame is in `engine.orchestration, engine.agents, engine.api, engine.coding, engine.models, engine.research, engine.forward.jobs, adapters`. Markers only add refusals; no flag, env var or config turns the gate off |
| A2 | approval needs stdin+stdout TTY and a controlling terminal; a one-time code is shown and the phrase `APPROVE <KIND> <spec_hash> <code>` must be typed on `/dev/tty` |
| A3 | PAPER_APPROVED: the report is recomputed inside `approve()` and must meet `ForwardEvidence` before the human is asked |

`logger.log()` has no clock parameter. The tests simulate time by monkeypatching `signals._utcnow`.

## Scoring
For each outcome bar `d`, the in-force signal is the latest with `as_of < d`.
- **Paper**: rebalance at `d`'s close through `PaperBroker` (1 bp commission + 2 bp adverse slippage, long-only, unlevered, buys scaled to cash) only when the target weights changed. P&L starts after the fill.
- **Vector equivalent**: what `engine/backtesting/vector.py` assumes: the weight at `as_of` earns bar `d` in full, cost = `cost_bps * |dw|`.
- **Cost analysis**: paper commission + slippage in $ and bps of traded notional vs the spec's `cost_bps`.
- **Tracking**: paper minus vector total return, daily tracking error. The gap is one bar of timing plus the cost difference.
- **Expectation**: z of forward daily mean vs the stored validation-window mean/sd. It has almost no power on few days.
- Missing bars are skipped and counted, never filled. Days without a fresh signal from the preceding bar are counted as stale.

## Evidence minimum (`ForwardEvidence`, frozen 2026-11-11 before any forward outcome existed; digest stored in reports and approvals)
`min_forward_days=60, min_calendar_days=84, max_stale_frac=0.10, max_drawdown=-0.20, max_abs_expectation_z=3.0`, real data only.
A day is *eligible* only if its signal was logged under an approved SHADOW status, on real data, from a final as_of bar
(data fetched after that session's close), and fresh. Signals logged before the ENTER_SHADOW approval are scored but do not count.

## Status semantics
`Store.set_status` still raises `PermissionError` for SHADOW / PAPER_APPROVED and is not called for them. The research store keeps
the strategy at CANDIDATE; forward status is derived from the verified approval chain. `forward.db:forward_status` is only a
mirror: `status()` raises if it disagrees with the chain. Each approval stores approver, reason, ts, spec_hash, evidence hash
(forward `report_hash` for PAPER_APPROVED; hash of the latest experiment row for ENTER_SHADOW), evidence-rule hash, the
signal-log head (anchor), and proof (tty, pid, OS user, parent process chain, sha of the one-time code).

## Honest limits
- **No secret key.** A process with write access can rewrite a whole chain file consistently, or append a well-formed approval
  record straight to `approvals.jsonl`. Mitigations in place: signal-log heads are anchored in approval records; illegal
  transition order is rejected. Not in place: signatures with a key held off the machine. Keep the head hashes from
  `forward verify` somewhere else if you need more.
- **A pseudo-terminal can impersonate a human.** The tests do this to exercise the real path. The gate stops accidental,
  scheduled, config- or flag-driven promotion and leaves an audit trail; it is not proof a human was present.
- A crash between writing a record and its `.head` sidecar leaves the chain failing verification (fails closed; manual repair).
- Daily bars, trend family (single asset) only in `jobs.py`; the logger and scorer accept multi-asset long-only weights via `weights_fn`.
- Yahoo adjusted closes are re-based on dividends, so `ref_prices` at `as_of` can differ from later downloads; the first forward day of the vector-equivalent uses the recorded value.
- If the cache was fetched during the session, the as_of bar is partial; the record says `as_of_bar_final: false` and is not eligible evidence. Run `data_refresh` after the close, then `forward log`.
- No next-open fills (only closes are cached), no holiday calendar, no automatic demotion on bad forward results.

## Demo (mechanics only, not a finding)
`QAE_HOME=artifacts/qae_scratch_forward envs/core/bin/python -m engine.forward.cli demo` seeds a scratch strategy
(`DEMO-mechanics-trend-SPY-L12 (NOT A FINDING)`, spec label `MECHANICS_DEMO_NOT_A_FINDING`; it passed no validation) as
CANDIDATE in the scratch store and logs one signal from the cached real SPY bars. Run 2026-11-11 21:23 UTC:
as_of 2026-11-11, weight SPY 1.0, `as_of_bar_final=false` (cache fetched 10:34 New York, mid-session), forward status CANDIDATE.
**Realised scoring is empty: 0 outcome bars exist after the signal, 0 fills, no P&L.** Refusals shown on real data: G1, G2
(frame truncated to 2019-12-31), `store.set_status(SHADOW)`, and the approval attempt. Output: `artifacts/forward/demo_*.json`.

## Tests
`envs/core/bin/python -m pytest -p no:cacheprovider tests/test_forward.py` (46 tests). All price data in tests is synthetic and labelled.
Mutation checks (30 mutants, all caught, sources restored): chain content-hash / prev-link / head anchor / external anchors /
append-without-verify; G1, G2, G2 off-by-one, G3, G4, G5; S1, S1 off-by-one, S2 guard, S2 replay `<` -> `<=`, S3, loader `>` -> `>=`,
paper fill at a non-outcome price; eligibility, evidence min-days; A1 env, A1 stack, A2 isatty, A2 auto-confirm, A2 any phrase,
A3, transition check, mirror trust, fold order, non-CANDIDATE approval.
