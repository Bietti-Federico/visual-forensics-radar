"""One component's contribution to the final risk score — kept so the report shows its own math."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskComponentScore:
    name: str
    score: float
    weight: float
