# Coding-agent improvement workflow (Part 6)

Code: `engine/coding/` + `scripts/coding_sandbox_run.sh`. Tests: `tests/test_coding_workflow.py`. Outputs: `artifacts/coding/`.

A patch (unified diff, from a person or from the local loopback LLM) is tested in an isolated copy of the lab source and
ends in exactly one of: `READY_FOR_HUMAN_REVIEW`, `ROLLED_BACK`, `BLOCKED_NEEDS_APPROVAL`, `REJECTED_INVALID_PATCH`.
`propose` never writes to the lab. Only `apply` does, and it needs a human approval record for the exact patch bytes.

## Run it
```bash
envs/core/bin/python -m engine.coding propose --patch my.diff --summary "what and why" [--declare-golden risk.metrics]
envs/core/bin/python -m engine.coding propose --llm --path engine/risk/metrics.py --instruction "..."   # optional, local Ollama
envs/core/bin/python -m engine.coding status [run_id]
envs/core/bin/python -m engine.coding approve <patch_sha256> --approver "Name" --reason "..."          # person at a terminal only
envs/core/bin/python -m engine.coding apply <run_id>        # backup -> write -> verify; restores on any failure
envs/core/bin/python -m engine.coding restore <run_id>      # undo an applied patch from its backup
envs/core/bin/python -m engine.coding verify                # hash-chained index + report hashes
```

## Pipeline (`engine/coding/workflow.py::CodingWorkflow.propose`)
1. **Validate** (`patch.py`): whitelist grammar. Rejects absolute paths, `..`, backslash/quoted/non-ASCII paths, any `.git*`
   component, binary patches, renames/copies, mode changes, exec bits, symlink/submodule modes, paths outside
   `engine/ adapters/ tests/ configs/ scripts/ pyproject.toml`, case-variant paths, and paths crossing a symlink.
2. **Gate** (`gate.py`): protected paths + added-line keyword scan (below). Blocked patches stop here: no workspace, no sandbox.
3. **Workspace**: copy of `engine/ adapters/ tests/ configs/ scripts/ pyproject.toml` (symlinks, caches, `.bak` skipped) into
   `artifacts/coding/ws/<run_id>/`, `git init`, baseline commit. Host-side git runs with hooks off and no global/system config.
4. **Baseline run** in the sandbox: per-test outcomes + golden probes. Results are read out of the workspace immediately.
5. **Apply in workspace** (`git apply`), verify the changed set equals the validated paths and no mode changed, commit, and
   stage the post-patch file contents from the commit (before any patched code runs).
6. **Patched run** in the same sandbox profile.
7. **Compare**: failure if any test passed before and does not pass after (failed, errored or missing), any test added by the
   patch fails, a new collection error appears, the runner fails/times out, or a golden probe moved beyond tolerance
   (rel 1e-9 / abs 1e-12) without being named in the intent's `golden_changes`. Pre-existing failures are baseline, not regressions.
8. **Failure => auto-rollback**: workspace `git reset --hard` + `git clean`, verified against the baseline tree; staged files
   discarded; workspace deleted. The lab was never written.
9. **Report**: `artifacts/coding/reports/<run_id>.{json,md}` + the sandbox profile used; entry appended to the hash-chained
   `artifacts/coding/index.jsonl`.

## Sandbox (`scripts/coding_sandbox_run.sh`)
macOS `sandbox-exec` with a **deny-default** profile (`(deny default)` + Apple's `system.sb` base + explicit allows).
What the tests demonstrate (each line is asserted from inside the sandbox in `tests/test_coding_workflow.py`):
- env: scrubbed with `env -i` before entering the sandbox; synthetic token variables set in the parent are not visible.
  `HOME`, `TMPDIR`, `QAE_HOME` point inside the workspace (never `artifacts/qae/`).
- process-exec: only the core interpreter (env path, its real path, and its framework directory). `/bin/sh`,
  `/usr/bin/open`, `osascript`, `curl` all fail to exec. An exec path inside the writable dir is refused by the script.
- mach-lookup: only what `system.sb` grants; a lookup of `com.apple.coreservices.launchservicesd` fails inside and
  succeeds outside (control test).
- LaunchServices escape (found in review: write an `.app` in the workspace, `open` it, it runs unsandboxed): blocked three
  ways - `open` cannot be exec'd, LaunchServices cannot be looked up, and no `*.app` path can be written. Regression test
  `test_launchservices_app_bundle_escape_is_blocked` tries `.app`, `.bundle`, a bare directory, `open <url>`, `/bin/sh`,
  direct exec of the written script and `osascript`; nothing is created outside the workspace.
- reads: `/Users`, `/Volumes`, `/private/tmp`, `/private/var/folders` denied except the workspace, `envs/core` (read-only)
  and the two trusted runner files; `~/.ssh ~/.aws ~/.claude ~/.config ~/.gnupg ~/.docker ~/Library ~/.netrc`,
  `quantalpha/data`, `quantalpha/tradesea` denied on top. The rest of the system volume (`/System`, `/usr`, `/opt/homebrew`,
  `/private/etc`, ...) IS readable: anything world-readable there is visible to patched code.
- writes: only inside the workspace (plus `/dev/null`); the workspace `.git` is not writable.
- network: only loopback IP. Outbound TCP, UDP and DNS fail.
- signals: only to processes in the same sandbox (signalling the host test process fails).
- the interpreter runs with `-I`; the runner refuses to continue unless `engine` resolved inside the workspace.

Host-side handling of what the sandbox produced:
- the result JSON is opened with `O_NOFOLLOW`, must be a single-link regular file, and is size-capped;
- on timeout the process group is killed (a denied `killpg` falls back to killing the child);
- after every run the host scans `ps` for processes carrying the run tag in their environment or the workspace path in
  their command line, kills them, and fails the run. This is best-effort: a process that removed both markers is not found
  (it would still be confined by the sandbox profile it inherited).

Not demonstrated / not claimed: resistance to kernel or `sandbox-exec` bugs; that no readable system file contains
anything sensitive; that loopback services are safe to expose (see Limits).

## Approval gate
The protected list is a frozen tuple in `engine/coding/gate.py`, pinned by sha256 (checked at import and again in a test).
It is not read from config, environment or CLI flags. Protected: `engine/coding/**`, `tests/test_coding*`,
`engine/memory/**`, `engine/research/**`, `engine/agents/**`, `engine/backtesting/**`, `engine/orchestration/**`,
`engine/forward/**`, `engine/api/**`, `engine/data/**`, `engine/models/**`, `engine/families/**`, `engine/derivs/**`,
`engine/futures/**`, `engine/eventsim/**`, `engine/strategies/**`, `engine/qae_cli.py`, `engine/cli.py`, `engine/errors.py`,
`engine/logging_setup.py`, `engine/__init__.py`, `adapters/**`, `scripts/**`, `configs/**`, `pyproject.toml`, any
`conftest.py` / `sitecustomize.py` / `usercustomize.py` / `*.pth` / `*.sh`.
**Not protected** (deliberately small): `engine/analytics/**`, `engine/risk/**`, `engine/optimization/**`, brand-new
top-level `engine/` packages, and test files other than `conftest.py` and `tests/test_coding*`. Patches outside
`engine/ adapters/ tests/ configs/ scripts/ pyproject.toml` (docs included) are rejected, not gated.
Added lines are also scanned for broker / order / credential / exec / secret / token / api key / account / subprocess /
socket / urllib / requests / http(s) URLs / os.system / eval( / `__import__` / importlib / pickle / ctypes / os.environ /
getenv / sandbox / chmod / symlink and similar. The scan is deliberately broad: the word "order" in a docstring asks for a human.

An approval is `artifacts/coding/approvals/<patch_sha256>.json` (approver, reason, patch sha256, self-hash). `approve`
refuses unless stdin and stdout are a terminal, no scheduler/CI/agent marker is set (`QAE_SCHEDULER`, `QAE_JOB_CONTEXT`,
`QAE_AGENT_CONTEXT`, `CI`, `CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, ...), no frame on the call stack belongs to
`engine.orchestration`, `engine.agents`, `engine.api`, `engine.forward`, `engine.models`, `engine.research`,
`engine.coding.workflow`, `engine.coding.llm_patch` or `adapters`, the approver is not an automation identity, and the
person re-types the first 12 hex characters of the hash. `CodingWorkflow` has no approve method; the CLI calls the gate directly. A record authorises one exact patch; editing it or
copying it under another hash invalidates it. An approved patch runs in the same sandbox profile as any other patch.
`apply` needs such a record for every patch, protected or not.

## Apply / restore
`apply` re-verifies the index, the staged manifest and every staged file hash, refuses if any lab file differs from what was
tested (drift), backs up to `artifacts/coding/backups/<run_id>/`, writes atomically, verifies post-hashes. Any failure
restores every file from backup and verifies by sha256 (`APPLY_ROLLED_BACK`). `restore` reverses a completed apply.
`apply` does not re-run the suite in the lab: run it afterwards.

## Golden probes (`engine/coding/golden.py`)
Nine probes over SYNTHETIC seeded random walks (not market data): `analytics.returns`, `risk.metrics`, `backtesting.vector`,
`strategies.trend`, `research.validation`, `research.risk_gate`, `optimization.portfolio`, `backtesting.paper`,
`memory.store_rules`. A probe that raises is recorded as an error value and compared like any other output.

## Local LLM (`engine/coding/llm_patch.py`, optional)
Reuses `engine.agents.llm.OllamaClient` (loopback hosts only). The model returns the full new text of one file (max 12 kB);
the diff is computed with difflib and then treated like any human patch. One request at a time (non-blocking flock:
a second caller gets `LLMBusy`). Only the fake-transport path is tested; no real model call was made while building this.

## Limits (read these)
- `sandbox-exec` limits blast radius; it is not a VM or container. An earlier allow-default version of this profile was escaped in review via LaunchServices; the current one is deny-default, but treat it as defence in depth, not proof.
- Loopback is reachable from the sandbox, so patched code can talk to local services (Ollama on 11434, a running `qae serve`).
- The test and golden results are produced by the code under test. A patch that edits tests or forges the result file can
  make a bad change look clean; baseline results are out of its reach, removed/added tests are listed in the report, and the
  end state is human review, never auto-apply.
- An approval record is a file and no signing key exists. Any process that can write `artifacts/coding/approvals/` can
  forge one (the record's self-hash is a consistency check, not a signature); the workflow itself has no code path that
  creates one without a terminal. A pty-driving program can impersonate a person (the tests do exactly that).
- The hash chain detects edits, deletions in the middle and reordering; truncating the tail of `index.jsonl` is not detectable
  without an external anchor of the head hash.
- Tests that need `datasets/`, `envs/*` other than core, or files outside the workspace fail or skip in both runs and are
  reported as pre-existing failures; they provide no regression coverage here.
- A full-suite run on the real lab copy takes about 3 minutes per pass (two passes per proposal).
