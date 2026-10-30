"""Drive paths: the polylines a survey vehicle follows, and the truth address along them.

A *corridor* is one signed route (say US-23) chained across county lines: KYTC restarts
the milepoint and changes the route id at every county line, so a vehicle surveying US-23
crosses several ``-000`` routes in one run.

Driving with the inventory direction follows the ``-000`` mainlines with increasing M.
Driving the other way follows them backward, except where a ``-010`` twin carriageway
exists: there the vehicle is on the twin. The truth of every reading is kept on the ``-000``
measure axis, which is also the axis the year-over-year grid uses: a point on a twin takes
the mainline measure abeam it (perpendicular projection onto the mainline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LINK_FT = 300.0   # a part continues a corridor if it starts within this distance of the last end


def twin_of(route: str) -> str:
    return route[:-3] + "010"


def mainline_of(route: str) -> str:
    return route[:-3] + "000"


def cut(xym: np.ndarray, m0: float, m1: float) -> np.ndarray:
    """The stretch of a measured polyline (M increasing) between measures m0 < m1."""
    m = xym[:, 2]
    m0, m1 = max(m0, m[0]), min(m1, m[-1])
    if m1 - m0 <= 1e-9:
        return xym[:0]
    inside = xym[(m > m0) & (m < m1)]
    ends = []
    for target in (m0, m1):
        i = int(np.clip(np.searchsorted(m, target), 1, len(m) - 1))
        span = m[i] - m[i - 1]
        t = 0.0 if span <= 0 else (target - m[i - 1]) / span
        ends.append(xym[i - 1] + t * (xym[i] - xym[i - 1]))
    return np.vstack([ends[0], inside, ends[1]])


@dataclass
class Leg:
    route: str          # route driven: a -000 mainline or its -010 twin
    base: str           # the -000 route whose cells hold the truth
    xym: np.ndarray     # vertices in travel order; M on the driven route's scale
    base_m: np.ndarray | None = None   # mainline measure abeam each vertex (twins only)

    def __post_init__(self):
        if self.base_m is None:
            self.base_m = self.xym[:, 2]


def abeam(xy: np.ndarray, mainline: np.ndarray) -> np.ndarray:
    """The mainline's measure at the point abeam each (x, y): perpendicular projection."""
    import shapely

    line = shapely.linestrings(mainline[:, :2])
    pos = shapely.line_locate_point(line, shapely.points(xy))
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(mainline[:, :2], axis=0).T))])
    return np.interp(pos, cum, mainline[:, 2])


@dataclass
class DrivePath:
    corridor: str
    direction: str                  # "inv" (with increasing M) or "non" (against it)
    legs: list[Leg]
    xy: np.ndarray = field(init=False)
    s: np.ndarray = field(init=False)       # distance along the path, feet
    leg: np.ndarray = field(init=False)     # leg index of each vertex

    def __post_init__(self):
        self.xy = np.vstack([g.xym[:, :2] for g in self.legs])
        self.leg = np.concatenate([np.full(len(g.xym), k) for k, g in enumerate(self.legs)])
        step = np.hypot(*np.diff(self.xy, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(step)])
        self._m = np.concatenate([g.base_m for g in self.legs])
        self._mr = np.concatenate([g.xym[:, 2] for g in self.legs])
        self._bases = np.array([g.base for g in self.legs], dtype=object)
        self._twins = np.array([g.route != g.base for g in self.legs])

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def sample(self, s: np.ndarray):
        """Position, heading and truth address at distances ``s`` along the path.

        Returns ``xy`` (n, 2), ``heading`` (radians, math convention), ``base`` (object
        array of -000 route ids, None on the short connectors between legs), ``m`` (the
        -000 measure, NaN on connectors), ``on_twin`` (bool) and ``m_route`` (the measure
        on the route actually driven: the twin's own measure where ``on_twin``).
        """
        s = np.clip(np.asarray(s, float), 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2)
        span = self.s[i + 1] - self.s[i]
        t = np.where(span > 0, (s - self.s[i]) / np.where(span > 0, span, 1), 0.0)
        a, b = self.xy[i], self.xy[i + 1]
        xy = a + t[:, None] * (b - a)
        heading = np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
        same_leg = self.leg[i] == self.leg[i + 1]
        m = np.where(same_leg, self._m[i] + t * (self._m[i + 1] - self._m[i]), np.nan)
        m_route = np.where(same_leg, self._mr[i] + t * (self._mr[i + 1] - self._mr[i]), np.nan)
        base = np.where(same_leg, self._bases[self.leg[i]], None)
        return xy, heading, base, m, same_leg & self._twins[self.leg[i]], m_route


def corridors(table: pd.DataFrame, parts: dict[str, list[np.ndarray]], min_mi: float = 0.3):
    """Chain -000 parts of each signed route into corridors.

    Returns a list of ``(corridor_id, [(route, part_index), ...])`` in inventory order.
    """
    main = table[table["ROUTE"].str.endswith("-000")]
    out = []
    for key, grp in main.groupby(["RT_PREFIX", "RT_NUMBER", "RT_SUFFIX"], dropna=False, sort=True):
        nodes = [(r, k) for r in grp["ROUTE"] for k in range(len(parts[r]))]
        start = {n: parts[n[0]][n[1]][0, :2] for n in nodes}
        stop = {n: parts[n[0]][n[1]][-1, :2] for n in nodes}
        succ = {}
        for a in nodes:
            best = min(((float(np.hypot(*(start[b] - stop[a]))), b) for b in nodes if b != a),
                       default=(np.inf, None))
            if best[0] <= LINK_FT:
                succ[a] = best[1]
        has_pred = set(succ.values())
        seen = set()
        prefix, number, suffix = key
        name = f"{prefix}-{number}{'' if pd.isna(suffix) else suffix}"
        for n in sorted(nodes, key=lambda n: (n in has_pred, n)):
            if n in seen:
                continue
            chain = []
            while n is not None and n not in seen:
                seen.add(n)
                chain.append(n)
                n = succ.get(n)
            miles = sum(float(np.hypot(*np.diff(parts[r][k][:, :2], axis=0).T).sum())
                        for r, k in chain) / 5280
            if miles >= min_mi:
                out.append((f"{name}#{len([c for c in out if c[0].startswith(name + '#')]) + 1}", chain))
    return out


def drive_path(corridor_id: str, chain, parts: dict[str, list[np.ndarray]], direction: str) -> DrivePath:
    """The path for one corridor and direction, with twins substituted for "non" travel."""
    legs: list[Leg] = []
    if direction == "inv":
        for r, k in chain:
            legs.append(Leg(r, r, parts[r][k]))
        return DrivePath(corridor_id, direction, legs)

    for r, k in reversed(chain):
        xym = parts[r][k]
        lo, hi = xym[0, 2], xym[-1, 2]
        twin = [p for p in parts.get(twin_of(r), []) if p[-1, 2] > lo and p[0, 2] < hi]
        # stretches covered by the twin, then the uncovered remainder, walked from hi down
        spans = sorted(((max(p[0, 2], lo), min(p[-1, 2], hi), p) for p in twin), key=lambda t: -t[0])
        cursor = hi
        for a, b, p in spans:
            if b < cursor - 1e-6:
                piece = cut(xym, b, cursor)
                if len(piece) >= 2:
                    legs.append(Leg(r, r, piece[::-1]))
            piece = cut(p, a, min(b, cursor))
            if len(piece) >= 2:
                piece = piece[::-1]
                legs.append(Leg(twin_of(r), r, piece, abeam(piece[:, :2], xym)))
            cursor = min(cursor, a)
        if cursor > lo + 1e-6:
            piece = cut(xym, lo, cursor)
            if len(piece) >= 2:
                legs.append(Leg(r, r, piece[::-1]))
    return DrivePath(corridor_id, direction, legs)
