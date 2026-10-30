"""Find the measure of a point on measured geometry."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Hit(NamedTuple):
    m: float        # measure at the nearest point, interpolated between its two vertices
    offset: float   # distance from the query point to the nearest point
    along: float    # shape length from the start of the polyline to the nearest point


def locate(xym: np.ndarray, x: float, y: float) -> Hit:
    """Nearest point on a measured polyline (an (n, 3) x/y/m array) to (x, y)."""
    a, b = xym[:-1, :2], xym[1:, :2]
    ab = b - a
    sq = (ab * ab).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(sq > 0, ((np.array([x, y]) - a) * ab).sum(axis=1) / sq, 0.0)
    t = t.clip(0.0, 1.0)
    near = a + t[:, None] * ab
    d = np.hypot(near[:, 0] - x, near[:, 1] - y)
    i = int(d.argmin())
    step = np.sqrt(sq)
    return Hit(float(xym[i, 2] + t[i] * (xym[i + 1, 2] - xym[i, 2])), float(d[i]),
               float(step[:i].sum() + t[i] * step[i]))


def locate_on_parts(parts: list[np.ndarray], x: float, y: float) -> tuple[Hit, int]:
    """:func:`locate` over every part of a route; returns the nearest hit and its part."""
    best, best_k = Hit(np.nan, np.inf, np.nan), -1
    for k, xym in enumerate(parts):
        hit = locate(xym, x, y)
        if hit.offset < best.offset:
            best, best_k = hit, k
    return best, best_k
