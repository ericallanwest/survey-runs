"""Place survey segments on the LRS: route, measure and direction of travel.

An arc-projection matcher:

1. **Smooth the track.** Each run's segment midpoints get a centred 5-segment rolling mean,
   and the direction of travel is read off the smoothed track, so GPS jitter and held fixes
   don't swing the bearing.
2. **Find candidates.** Every route part within 150 ft; each midpoint is projected onto it for
   a distance, a measure and the part's local bearing there.
3. **Gate and score.** Keep candidates whose bearing is within 30 degrees of the travel
   direction, or within 30 degrees of its reverse (the same road driven the other way).
   Score = distance + 2 ft per *folded* degree of disagreement.
4. **Use the run's context.** A confident first pass (under 50 ft and 10 degrees) names the
   route each stretch of the run is on; the second pass adds 500 ft to any candidate that
   disagrees, so a segment isn't snapped onto a ramp or the opposite carriageway beside it.
5. **Measure and direction.** The segment's two end points are projected onto the winning
   part: their mean is the measure, and whether it grew says the direction of travel
   (``meas_sign`` +1 with the measure, -1 against it).

6. **Bridge held GPS fixes.** A receiver that stops updating repeats one position while the
   odometer keeps counting, then jumps when it recovers. Segments whose GPS step is near zero,
   or several segment lengths long, are left out of matching, and their measure is
   interpolated by DMI chainage between the matched segments on either side when both sides
   agree on route and direction (flag ``interpolated``).

Context fills within a run only (never across runs). Units are feet throughout (EPSG:3089).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from .network import EPSG

SEARCH_FT = 150.0
MAX_ANGLE = 30.0
ANGLE_PENALTY_FT = 2.0
CONTEXT_PENALTY_FT = 500.0
CONFIDENT_FT = 50.0
CONFIDENT_ANGLE = 10.0
CONTEXT_FILL = 10
SMOOTH = 5
STEP_MI = 0.005
HELD_FT = 0.5            # a GPS step this short while the odometer advances is a held fix
JUMP_FACTOR = 3.0        # a GPS step this many segment lengths long is a recovery jump
BRIDGE_FT = 2640.0       # longest stretch of held fixes bridged by chainage


class Network:
    """Every route part as a line, with the arrays a projection needs."""

    def __init__(self, parts: dict[str, list[np.ndarray]]):
        self.route = np.array([r for r, ps in parts.items() for _ in ps], dtype=object)
        xyms = [p for ps in parts.values() for p in ps]
        owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(xyms)])
        xym = np.vstack(xyms)
        self.lines = shapely.linestrings(xym[:, :2], indices=owner)
        step = np.hypot(*np.diff(xym[:, :2], axis=0).T)
        step[owner[1:] != owner[:-1]] = 0.0
        cum = np.concatenate([[0.0], np.cumsum(step)])
        first = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        self.first = first
        self.last = np.r_[first[1:] - 1, len(owner) - 1]
        self.cum = cum - cum[first][owner]                      # distance within each part
        self.m = xym[:, 2]
        dx, dy = np.diff(xym[:, 0]), np.diff(xym[:, 1])
        self.bearing = np.r_[(np.degrees(np.arctan2(dx, dy)) % 360), 0.0]   # compass, per vertex
        self.tree = shapely.STRtree(self.lines)

    def locate(self, part: np.ndarray, points) -> tuple[np.ndarray, np.ndarray]:
        """Measure (miles) and compass bearing where points project onto their parts."""
        pos = shapely.line_locate_point(self.lines[part], points)
        lo, hi = self.first[part], self.last[part]
        j = np.empty(len(part), dtype=np.int64)
        # binary search within each part's slice of the flat arrays
        a, b = lo.copy(), hi.copy()
        while True:
            open_ = a < b
            if not open_.any():
                break
            mid = (a + b + 1) // 2
            go_up = open_ & (self.cum[mid] <= pos)
            a = np.where(go_up, mid, a)
            b = np.where(open_ & ~go_up, mid - 1, b)
        j = np.minimum(a, hi - 1)
        span = self.cum[j + 1] - self.cum[j]
        t = np.clip(np.where(span > 0, (pos - self.cum[j]) / np.where(span > 0, span, 1), 0.0), 0, 1)
        return self.m[j] + t * (self.m[j + 1] - self.m[j]), self.bearing[j]


def _bridge(out: pd.DataFrame, df: pd.DataFrame, unreliable: np.ndarray) -> None:
    """Fill held-fix segments by chainage between matched neighbors on one route and direction."""
    good = (out["flag"] == "ok").to_numpy()
    ch = ((df["ch_start_ft"] + df["ch_end_ft"]) / 2).to_numpy()
    run = df["run"].to_numpy()
    idx = np.arange(len(df))
    prev = pd.Series(np.where(good, idx, np.nan)).groupby(run).ffill().to_numpy()
    nxt = pd.Series(np.where(good, idx, np.nan)).groupby(run).bfill().to_numpy()
    fill = np.flatnonzero(unreliable & np.isfinite(prev) & np.isfinite(nxt))
    if not len(fill):
        return
    a, b = prev[fill].astype(int), nxt[fill].astype(int)
    route, sign, m = out["route"].to_numpy(), out["meas_sign"].to_numpy(), out["m"].to_numpy(float)
    ok = (route[a] == route[b]) & (sign[a] == sign[b]) & (ch[b] - ch[a] <= BRIDGE_FT)
    fill, a, b = fill[ok], a[ok], b[ok]
    t = (ch[fill] - ch[a]) / (ch[b] - ch[a])
    mi = m[a] + t * (m[b] - m[a])
    out.loc[fill, "route"] = route[a]
    out.loc[fill, "m"] = mi
    out.loc[fill, "mp"] = np.round(np.round(mi / STEP_MI) * STEP_MI, 3)
    out.loc[fill, "meas_sign"] = sign[a]
    out.loc[fill, "flag"] = "interpolated"


def _smooth(v: pd.Series, run: pd.Series) -> pd.Series:
    return v.groupby(run).transform(lambda s: s.rolling(SMOOTH, center=True, min_periods=1).mean())


def match(segs: pd.DataFrame, net: Network) -> pd.DataFrame:
    """Match segments (columns run, seq, lon0, lat0, lon1, lat1). Returns a frame aligned with
    ``segs`` adding ``route``, ``m`` (raw measure), ``mp`` (snapped to 0.005 mi), ``meas_sign``,
    ``snap_ft``, ``angle_fold``, ``meas_len_mi`` and ``flag``."""
    order = np.lexsort((segs["seq"].to_numpy(), segs["run"].to_numpy()))
    df = segs.iloc[order].reset_index(drop=True)
    to_lrs = Transformer.from_crs(4326, EPSG, always_xy=True)
    x0, y0 = to_lrs.transform(df["lon0"].to_numpy(), df["lat0"].to_numpy())
    x1, y1 = to_lrs.transform(df["lon1"].to_numpy(), df["lat1"].to_numpy())
    run = df["run"]
    step = np.hypot(x1 - x0, y1 - y0)
    seg_len = (df["ch_end_ft"] - df["ch_start_ft"]).to_numpy()
    unreliable = (step < HELD_FT) | (step > JUMP_FACTOR * seg_len)
    mx = np.where(unreliable, np.nan, (x0 + x1) / 2)
    my = np.where(unreliable, np.nan, (y0 + y1) / 2)
    sx = _smooth(pd.Series(mx), run)
    sy = _smooth(pd.Series(my), run)
    sx[unreliable], sy[unreliable] = np.nan, np.nan
    g = run.to_numpy()
    nxt_ok = np.r_[g[1:] == g[:-1], False]
    prv_ok = np.r_[False, g[1:] == g[:-1]]
    xs, ys = sx.to_numpy(), sy.to_numpy()

    def neighbor(v, ok, shift):
        """The next (shift 1) or previous (-1) smoothed point in the run, else this one."""
        w = np.where(ok, np.roll(v, -shift), np.nan)
        return np.where(np.isfinite(w), w, v)

    dx = neighbor(xs, nxt_ok, 1) - neighbor(xs, prv_ok, -1)
    dy = neighbor(ys, nxt_ok, 1) - neighbor(ys, prv_ok, -1)
    lone = (dx == 0) & (dy == 0)            # no usable neighbor on either side: own direction
    bearing = np.degrees(np.arctan2(np.where(lone, x1 - x0, dx), np.where(lone, y1 - y0, dy))) % 360
    cent = shapely.points(xs, ys)
    usable = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))   # held fixes match nothing

    q_i, part_i = net.tree.query(cent[usable], predicate="dwithin", distance=SEARCH_FT)
    seg_i = usable[q_i]
    cand = pd.DataFrame({"seg": seg_i, "part": part_i,
                         "dist": shapely.distance(cent[seg_i], net.lines[part_i])})
    _, arc_brg = net.locate(part_i, cent[seg_i])
    diff = np.abs(bearing[seg_i] - arc_brg) % 360
    angle = np.minimum(diff, 360 - diff)
    keep = (angle <= MAX_ANGLE) | (angle >= 180 - MAX_ANGLE)     # every Kentucky part is two-way
    cand = cand[keep].copy()
    cand["fold"] = np.minimum(angle[keep], 180 - angle[keep])
    cand["route"] = net.route[cand["part"].to_numpy()]
    cand["base"] = cand["dist"] + ANGLE_PENALTY_FT * cand["fold"]

    first = cand.sort_values("base", kind="stable").drop_duplicates("seg").set_index("seg").sort_index()
    confident = (first["dist"] < CONFIDENT_FT) & (first["fold"] < CONFIDENT_ANGLE)
    context = pd.Series(np.nan, index=df.index, dtype=object)
    context.loc[first.index[confident]] = first.loc[confident, "route"]
    context = context.groupby(run).transform(lambda s: s.ffill(limit=CONTEXT_FILL).bfill(limit=CONTEXT_FILL))
    ctx = context.to_numpy()[cand["seg"].to_numpy()]
    disagree = pd.notna(ctx) & (cand["route"].to_numpy() != ctx)
    cand["final"] = cand["base"] + np.where(disagree, CONTEXT_PENALTY_FT, 0.0)
    best = cand.sort_values("final", kind="stable").drop_duplicates("seg").set_index("seg").sort_index()

    idx = best.index.to_numpy()
    part = best["part"].to_numpy()
    m0, _ = net.locate(part, shapely.points(x0[idx], y0[idx]))
    m1, _ = net.locate(part, shapely.points(x1[idx], y1[idx]))
    out = pd.DataFrame(index=df.index, columns=["route", "m", "mp", "meas_sign", "snap_ft",
                                                "angle_fold", "meas_len_mi"], dtype=object)
    out["m"] = out["mp"] = out["meas_sign"] = out["snap_ft"] = out["angle_fold"] = out["meas_len_mi"] = np.nan
    m = (m0 + m1) / 2
    out.loc[idx, "route"] = best["route"].to_numpy()
    out.loc[idx, "m"] = m
    out.loc[idx, "mp"] = np.round(np.round(m / STEP_MI) * STEP_MI, 3)
    out.loc[idx, "meas_sign"] = np.where(m1 >= m0, 1.0, -1.0)
    out.loc[idx, "snap_ft"] = best["dist"].to_numpy()
    out.loc[idx, "angle_fold"] = best["fold"].to_numpy()
    out.loc[idx, "meas_len_mi"] = np.abs(m1 - m0)
    out["flag"] = np.where(out["route"].isna(), "unmatched",
                           np.where(out["meas_len_mi"] < 0.0005, "zero_advance", "ok"))
    _bridge(out, df, unreliable)
    result = pd.concat([df, out], axis=1)
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    return result.iloc[back].reset_index(drop=True)
