"""Stable internal adapter contract. Third-party libs are imported lazily INSIDE adapter methods only."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
Status = Literal["operational", "untested", "unsupported", "reference_only"]

@dataclass(frozen=True)
class AdapterSpec:
    repo: str
    role: str
    status: Status
    env: str            # which isolated environment it needs
    note: str = ""
