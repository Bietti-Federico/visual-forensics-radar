"""A small, fully deterministic synthetic dataset shared by anomaly-detector tests.

A 2D grid of "normal" vectors (not a degenerate line — a proper spread in
both dimensions so no single grid point looks artificially isolated to a
tree-based detector) plus one clear outlier far outside the grid's range in
both dimensions. `INLIER_VECTOR` is the grid's center point, deliberately not
a corner, so it's unambiguously the least isolated point in the set. No
randomness anywhere, so every detector's behavior on it is reproducible.
"""

from __future__ import annotations

_GRID_STEP = 0.1
_GRID_SIZE = 5  # 5x5 = 25 points, spread over [0.0, 0.4] x [0.0, 0.4]

NORMAL_VECTORS: list[dict[str, float]] = [
    {"a": _GRID_STEP * i, "b": _GRID_STEP * j} for i in range(_GRID_SIZE) for j in range(_GRID_SIZE)
]
INLIER_VECTOR: dict[str, float] = NORMAL_VECTORS[len(NORMAL_VECTORS) // 2]
OUTLIER_VECTOR: dict[str, float] = {"a": 100.0, "b": 100.0}
