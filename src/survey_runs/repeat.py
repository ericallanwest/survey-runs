"""Repeatability: how closely repeat runs over one site agree, and how much the answer depends
on how the runs are lined up.

Each reference site is driven three times by each vehicle on one day a year. Per site and year
there are three baskets of runs to compare: each vehicle's own three (repeatability) and all six
(reproducibility, vehicle against vehicle).

**Pairing.** Runs are compared position by position, so each run's segments are first paired with
the other runs' segments at the same place on the road. The longest run's segments over the site
are the stations; the methods differ in where they think each other run's segments are:

* ``chainage``: by odometer. Cut every run at the site's start cross-section, then pair the k-th
  segment of each. Exact only while the odometers agree: one reading 0.5% short slides a segment
  against the others every mile.
* ``gps``: by GPS position along the road. Each run contributes, in surveyed order, the segment
  nearest each station (within 0.75 segment): of all in-order pairings, the closest in total, a
  station going unpaired only when that brings the rest closer by more than 3 segments.
* ``lrs``: by matched LRS measure, paired the same way.

``oracle`` scores them: the same pairing fed each segment's true position on the road, the best
the segments allow. A pairing is right where a run's segment lies within half a segment of the
station's.

**Metrics**, over the positions every run covers with a valid reading:

* BRSD: each run's mean over every 264-ft block (10 positions), the between-run SD of those means
  per block, averaged over the blocks: the typical run-to-run disagreement.
* ICC(2,k) and ICC(3,k), from a two-way ANOVA without replication on the positions x runs table:
  the share of the spread that is the road rather than the runs. (2,k) counts a run that reads
  high throughout against agreement, (3,k) does not, so a vehicle bias shows as the gap between.

Held GPS fixes are re-placed along chainage between the good fixes either side before pairing.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from . import match, network
from .network import EPSG

SEG_FT = 26.4
BLOCK = 10             # BRSD block: 10 positions, 264 ft
TOL_SEG = 0.75          # a segment pairs with a station within this many segments along the road
SKIP_SEG = 3.0          # what leaving a station unpaired costs, in segments of distance
WINDOW = 3              # rows either side of a run's expected row that may pair with a station
HELD_SEG = 0.25         # a segment start that moved less than this share of a segment is held
EXTEND_FT = 1000.0      # the reference line runs on past both ends, so along-road distance is signed
METHODS = ("chainage", "gps", "lrs", "oracle")


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


def point_at(parts, route: str, m: float) -> np.ndarray:
    for p in parts[route]:
        if p[0, 2] <= m <= p[-1, 2]:
            return np.array([np.interp(m, p[:, 2], p[:, 0]), np.interp(m, p[:, 2], p[:, 1])])
    raise ValueError(f"{route} has no measure {m}")


def stations(runs: list[pd.DataFrame], ref: int, site: dict) -> np.ndarray:
    """The reference run's rows whose segment starts lie between the site's two ends."""
    xy = runs[ref][["x", "y"]].to_numpy()
    along = along_road(xy)
    a = along(xy)
    a0, a1 = along(np.vstack([site["xy_start"], site["xy_end"]]))
    return np.flatnonzero((a >= a0) & (a < a1))


def pair(method: str, runs: list[pd.DataFrame], ref: int, st: np.ndarray) -> np.ndarray:
    """Each run's row at each station (runs x stations, -1 where a run has none).

    ``oracle`` needs ``s_true`` (true distance along the road, ft) on the runs."""
    if method in ("chainage", "gps"):
        P = [r[["x", "y"]].to_numpy() for r in runs]
        S, T = P[ref][st], tangents(P[ref])[st]
        along = along_road(P[ref])
        pos = [along(p) for p in P]                     # distance along the road, to find each start
    elif method in ("lrs", "oracle"):
        P = [r[["m"]].to_numpy(float) * 5280 if method == "lrs" else r[["s_true"]].to_numpy() for r in runs]
        S, T = P[ref][st], np.ones((len(st), 1))
        pos = [p[:, 0] for p in P]
    else:
        raise ValueError(method)
    k0 = int(np.flatnonzero(np.isfinite(pos[ref][st]))[0])            # first station with a position
    out = np.empty((len(runs), len(st)), dtype=int)
    for i, p in enumerate(P):
        d = np.abs(pos[i] - pos[ref][st[k0]])
        start = int(np.argmin(np.where(np.isnan(d), np.inf, d))) - k0    # this run's row at station 0
        if i == ref:
            out[i] = st
        elif method == "chainage":
            k = start + np.arange(len(st))
            out[i] = np.where((k >= 0) & (k < len(p)), k, -1)
        else:
            out[i] = pair_in_order(p, S, T, max(start, 0))
    return out


def score_basket(runs: list[pd.DataFrame], rows: np.ndarray, ref: int) -> dict:
    """Metrics on the stations every run covers with a valid reading, and how far (truth) each
    other run's paired segment lies from the station's."""
    have = (rows >= 0).all(axis=0)
    rows = rows[:, have]
    vals = np.column_stack([np.where(r["valid"].to_numpy()[k], r["value"].to_numpy()[k], np.nan)
                            for r, k in zip(runs, rows)])
    keep = np.isfinite(vals).all(axis=1)
    x = vals[keep]
    mp = runs[ref]["m"].astype(float).to_numpy()[rows[ref]][keep]
    s = np.column_stack([r["s_true"].to_numpy()[k] for r, k in zip(runs, rows)])[keep]
    off = np.abs(np.delete(s, ref, axis=1) - s[:, [ref]])
    icc2, icc3 = icc(x)
    some = len(x) > 0
    return {"positions": int(len(x)), "unpaired": int((~have).sum()), "BRSD": brsd(x),
            "ICC2k": icc2, "ICC3k": icc3,
            "paired_right": float((off <= SEG_FT / 2).mean()) if some else np.nan,
            "offset_p95_ft": float(np.quantile(off, 0.95)) if some else np.nan,
            "_x": x, "_mp": mp, "_off": off.max(axis=1)}


def analyze(data: Path, log=print) -> dict:
    data = Path(data)
    sim, out = data / "sim", data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(sim / "runs.parquet")
    runs = runs[runs["kind"] == "repeat"].sort_values("run")
    segs = pd.read_parquet(sim / "segments.parquet", filters=[("run", "in", runs["run"].tolist())])
    sites = pd.read_parquet(sim / "sites.parquet").set_index("site")
    _, parts = network.read(data / "network.parquet")
    m = match.match(segs, match.Network(parts)).sort_values(["run", "seq"])
    x, y = Transformer.from_crs(4326, EPSG, always_xy=True).transform(m["lon0"].to_numpy(), m["lat0"].to_numpy())
    m["x"], m["y"] = x, y
    # the truth: for the oracle and the scores only
    tru = pd.read_parquet(sim / "truth_segments.parquet", columns=["run", "seq", "s_true"],
                          filters=[("run", "in", runs["run"].tolist())])
    m = m.merge(tru, on=["run", "seq"])
    by_run = {}
    for name, r in m.groupby("run", sort=False):
        r = r.reset_index(drop=True)
        r[["x", "y"]] = unhold(r[["x", "y"]].to_numpy(), ((r["ch_start_ft"] + r["ch_end_ft"]) / 2).to_numpy())
        by_run[name] = r
    log(f"repeat runs: {len(by_run)} matched, {int((m['flag'] == 'unmatched').sum())} segments unmatched")

    rows_out, aligned = [], []
    for (site_id, year), grp in runs.groupby(["site", "year"]):
        s = sites.loc[site_id]
        site = {"route_start": s["route_start"], "m_start": s["m_start"], "m_end": s["m_end"],
                "xy_start": point_at(parts, s["route_start"], s["m_start"]),
                "xy_end": point_at(parts, s["route_end"], s["m_end"])}
        baskets = {v: g["run"].tolist() for v, g in grp.groupby("vehicle")} | {"both": grp["run"].tolist()}
        for basket, names in baskets.items():
            rr = [by_run[n] for n in names]
            ref = int(np.argmax([len(r) for r in rr]))
            st = stations(rr, ref, site)
            for method in METHODS:
                res = score_basket(rr, pair(method, rr, ref, st), ref)
                x_, mp, off = res.pop("_x"), res.pop("_mp"), res.pop("_off")
                rows_out.append({"site": site_id, "year": int(year), "basket": basket, "method": method,
                                 "runs": len(names), "stations": len(st), **res})
                if basket == "both":       # the six runs side by side, for the profiles
                    n = len(names)
                    aligned.append(pd.DataFrame({"site": site_id, "year": int(year), "method": method,
                                                 "position": np.repeat(np.arange(len(x_)), n),
                                                 "mp": np.repeat(mp, n), "run": np.tile(names, len(x_)),
                                                 "value": x_.ravel(), "worst_offset_ft": np.repeat(off, n)}))
    table = pd.DataFrame(rows_out)
    table.to_parquet(out / "repeat.parquet", index=False)
    pd.concat(aligned, ignore_index=True).to_parquet(out / "repeat_aligned.parquet", index=False)
    summary = summarize(table)
    (out / "repeat_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(json.dumps(summary, indent=1, default=float))
    return summary


def summarize(table: pd.DataFrame) -> dict:
    """Per method, within-vehicle and between-vehicle baskets apart: the metrics' means, and
    how far each sits from the oracle's on the same basket."""
    t = table.assign(kind=np.where(table["basket"] == "both", "between_vehicles", "within_vehicle"))
    oracle = t[t["method"] == "oracle"].set_index(["site", "year", "basket"])
    out = {}
    for (kind, method), g in t.groupby(["kind", "method"], sort=False):
        o = oracle.loc[g.set_index(["site", "year", "basket"]).index]
        out.setdefault(kind, {})[method] = {
            "baskets": len(g), "stations": int(g["stations"].sum()), "positions": int(g["positions"].sum()),
            "unpaired": int(g["unpaired"].sum()),
            "paired_right": float(np.average(g["paired_right"], weights=g["positions"])),
            "offset_p95_ft": float(g["offset_p95_ft"].median()),
            "BRSD": float(g["BRSD"].mean()), "ICC2k": float(g["ICC2k"].mean()), "ICC3k": float(g["ICC3k"].mean()),
            "BRSD_vs_oracle": float((g["BRSD"].to_numpy() - o["BRSD"].to_numpy()).mean()),
            "ICC2k_vs_oracle": float((g["ICC2k"].to_numpy() - o["ICC2k"].to_numpy()).mean()),
        }
    return out
