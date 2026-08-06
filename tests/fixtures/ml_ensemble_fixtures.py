"""A small, fully deterministic synthetic labeled dataset shared by ML-ensemble tests.

Two clearly separable clusters — no randomness anywhere, so every model's
behavior on it is reproducible. `is_original=True` for the cluster near the
origin, `False` (manipulated) for the cluster far away.
"""

from __future__ import annotations

ORIGINAL_VECTORS: list[dict[str, float]] = [{"a": 0.01 * i, "b": 0.01 * i} for i in range(15)]
MANIPULATED_VECTORS: list[dict[str, float]] = [
    {"a": 10.0 + 0.01 * i, "b": 10.0 + 0.01 * i} for i in range(15)
]

TRAINING_VECTORS: list[dict[str, float]] = ORIGINAL_VECTORS + MANIPULATED_VECTORS
TRAINING_LABELS: list[bool] = [True] * len(ORIGINAL_VECTORS) + [False] * len(MANIPULATED_VECTORS)

HELD_OUT_ORIGINAL: dict[str, float] = {"a": 0.05, "b": 0.05}
HELD_OUT_MANIPULATED: dict[str, float] = {"a": 10.05, "b": 10.05}
