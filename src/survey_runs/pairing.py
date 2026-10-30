"""Pairing repeat runs position by position, and the repeatability metrics.

Only NumPy, pandas and Shapely, so the lab notebook can run it in the browser.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
import shapely

SEG_FT = 26.4
BLOCK = 10             # BRSD block: 10 positions, 264 ft
TOL_SEG = 0.75          # a segment pairs with a station within this many segments along the road
SKIP_SEG = 3.0          # what leaving a station unpaired costs, in segments of distance
WINDOW = 3              # rows either side of a run's expected row that may pair with a station
HELD_SEG = 0.25         # a segment start that moved less than this share of a segment is held
EXTEND_FT = 1000.0      # the reference line runs on past both ends, so along-road distance is signed


def icc(x: np.ndarray) -> tuple[float, float]:
    """ICC(2,k) and ICC(3,k) of a positions x runs table."""
    n, k = x.shape
    if n < 2 or k < 2:
        return np.nan, np.nan
    grand = x.mean()
    msr = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand
    mse = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (msc - mse) / n)), float((msr - mse) / msr)


def brsd(x: np.ndarray, block: int = BLOCK) -> float:
    """Mean over blocks of ``block`` positions of the between-run SD of the runs' block means."""
    if len(x) == 0 or x.shape[1] < 2:
        return np.nan
    means = pd.DataFrame(x).groupby(np.arange(len(x)) // block).mean().to_numpy()
    return float(means.std(axis=1, ddof=1).mean())


def tangents(p: np.ndarray) -> np.ndarray:
    """Unit direction of travel at each point, from the points two either side (fewer at the
    ends); a point with no span takes the direction of the one before."""
    i = np.arange(len(p))
    v = p[np.minimum(i + 2, len(p) - 1)] - p[np.maximum(i - 2, 0)]
    norm = np.hypot(*v.T)
    last = np.maximum.accumulate(np.where(norm > 0, i, 0))
    v, norm = v[last], norm[last]
    return np.where(norm[:, None] > 0, v / np.where(norm > 0, norm, 1.0)[:, None], [1.0, 0.0])


def unhold(p: np.ndarray, ch: np.ndarray) -> np.ndarray:
    """Held GPS fixes (a start that moved under HELD_SEG of a segment from the one before)
    re-placed along chainage between the good fixes either side."""
    step = np.r_[np.inf, np.hypot(*np.diff(p, axis=0).T)]
    good = step >= HELD_SEG * SEG_FT
    if good.all() or good.sum() < 2:
        return p
    return np.column_stack([np.interp(ch, ch[good], p[good, 0]), np.interp(ch, ch[good], p[good, 1])])


def along_road(ref: np.ndarray):
    """Signed distance along the reference track, as a function of points."""
    t = tangents(ref)
    line = shapely.linestrings(np.vstack([ref[0] - EXTEND_FT * t[0], ref, ref[-1] + EXTEND_FT * t[-1]]))
    return lambda pts: shapely.line_locate_point(line, shapely.points(np.atleast_2d(pts))) - EXTEND_FT


def pair_in_order(P, S, T, start, tol=TOL_SEG * SEG_FT, skip=SKIP_SEG * SEG_FT, window=WINDOW):
    """One run's row at each station (-1 for none), rows in surveyed order.

    ``P`` are the run's points, ``S`` the stations, ``T`` the road's unit direction at each
    station (distance is measured along it) and ``start`` the run's row nearest the first
    station. A greedy walk sets each station's window, ``window`` rows either side of where the
    run's last match predicts, so its drift along the road is followed. Within the windows a
    dynamic programme takes, of all in-order pairings, the one with the least total distance, a
    station left unpaired costing ``skip``.
    """
    def dist(k, rows):
        d = np.abs((P[rows] - S[k]) @ T[k])
        return np.where(np.isnan(d), np.inf, d)        # a point with no position never pairs

    windows, offset, last = [], start, start - 1
    for k in range(len(S)):
        lo, hi = max(k + offset - window, 0), min(k + offset + window, len(P) - 1)
        windows.append((lo, hi))
        rows = np.arange(max(lo, last + 1), hi + 1)
        if len(rows):
            d = dist(k, rows)
            b = int(np.argmin(d))
            if d[b] <= tol:
                last, offset = int(rows[b]), int(rows[b]) - k

    # states: (last row used, cost, path back); along the list rows rise and costs fall
    states = [(-1, 0.0, None)]
    for k, (lo, hi) in enumerate(windows):
        rows = np.arange(lo, hi + 1)
        used = [s[0] for s in states]
        moved = []
        for r, d in zip(rows, dist(k, rows)):
            j = bisect.bisect_left(used, r) - 1          # the cheapest state ending before row r
            if d <= tol and j >= 0:
                moved.append((int(r), states[j][1] + float(d), (k, int(r), states[j][2])))
        merged = sorted([(r, c + skip, p) for r, c, p in states] + moved, key=lambda s: (s[0], s[1]))
        states, best = [], np.inf
        for s in merged:
            if s[1] < best - 1e-9:
                states.append(s)
                best = s[1]
    out = np.full(len(S), -1)
    node = min(states, key=lambda s: s[1])[2]
    while node is not None:
        out[node[0]], node = node[1], node[2]
    return out
