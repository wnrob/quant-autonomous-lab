"""Coding-agent improvement workflow (Part 6): isolated git workspace, hard approval gate, sandboxed before/after tests,
golden-result diff, auto-rollback, auditable hash-chained reports. See docs/CODING_AGENT.md."""
from engine.coding.gate import PROTECTED_PATHS, PROTECTED_SHA256, ApprovalRefused, GateError      # noqa: F401
from engine.coding.patch import PatchRejected, parse_patch                                        # noqa: F401
from engine.coding.workflow import ApplyRefused, CodingError, CodingWorkflow                      # noqa: F401
