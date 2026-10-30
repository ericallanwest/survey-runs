"""The year-over-year panel: every survey year on one fixed 0.005-mi grid.

* **Grid.** One cell per exact 0.005-mi multiple of M on every route, integer keys so nothing
  drifts off the grid through float arithmetic, half-open (a route's terminus has no cell).
* **Direction.** A matched segment travelling with the measure fills its cell's ``inv`` slot,
  against it the ``non`` slot. On an undivided road both slots belong to the ``-000`` route.
  A divided road's other direction runs on a ``-010`` twin, measured with the mainline and
  so driven against its measure; each twin cell is paired with the nearest mainline cell
  within 300 ft (greedy, one to one) and its readings land in that cell's ``non`` slot.
* **Latest pass wins.** Where a year covers a cell and slot more than once (a re-drive), the
  reading from the latest run counts, by the timestamp in the run name. Within one run, a
  cell hit twice (odometer drift against the measure) keeps the reading nearest its center.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STEP_MI = 0.005
PAIR_FT = 300.0


def run_time(run: pd.Series) -> pd.Series:
    """The start time encoded in a run name like ``SV1_230417_114200``."""
    return pd.to_datetime(run.str[-13:], format="%y%m%d_%H%M%S")


def grid_points(parts: dict[str, list[np.ndarray]], routes) -> pd.DataFrame:
    rows = []
    for r in routes:
        for xym in parts[r]:
            lo, hi = xym[0, 2], xym[-1, 2]
            k = np.arange(int(np.ceil(lo / STEP_MI - 1e-9)), int(np.ceil(hi / STEP_MI - 1e-9)))
            if not len(k):
                continue
            m = k * STEP_MI
            rows.append(pd.DataFrame({"route": r, "grid": k, "x": np.interp(m, xym[:, 2], xym[:, 0]),
                                      "y": np.interp(m, xym[:, 2], xym[:, 1])}))
    return pd.concat(rows, ignore_index=True).drop_duplicates(["route", "grid"])


def pair_twins(grid: pd.DataFrame, radius_ft: float = PAIR_FT) -> pd.DataFrame:
    """Each -010 cell to its nearest -000 cell within ``radius_ft``, greedy one to one."""
    out = []
    by_route = dict(tuple(grid.groupby("route", sort=False)))
    for twin in [r for r in by_route if r.endswith("-010")]:
        main = twin[:-3] + "000"
        if main not in by_route:
            continue
        a, b = by_route[main], by_route[twin]
        dist, j = cKDTree(a[["x", "y"]].to_numpy()).query(b[["x", "y"]].to_numpy(),
                                                          distance_upper_bound=radius_ft)
        ok = np.isfinite(dist)
        cand = pd.DataFrame({"twin_grid": b["grid"].to_numpy()[ok], "grid": a["grid"].to_numpy()[j[ok]],
                             "pair_ft": dist[ok]}).sort_values("pair_ft", kind="stable")
        cand = cand.drop_duplicates("twin_grid").drop_duplicates("grid")
        cand["twin"], cand["route"] = twin, main
        out.append(cand)
    return pd.concat(out, ignore_index=True)[["twin", "twin_grid", "route", "grid", "pair_ft"]]


def to_cells(matched: pd.DataFrame, pairs: pd.DataFrame, year: int) -> pd.DataFrame:
    """One row per (cell, slot) for a year's matched segments, latest pass winning."""
    d = matched[matched["flag"].isin(["ok", "interpolated"]) & matched["valid"]].copy()
    d["grid"] = np.round(d["m"].astype(float) / STEP_MI).astype(np.int64)
    d["slot"] = np.where(d["meas_sign"] > 0, "inv", "non")
    d["key_route"], d["key_grid"], d["divided"] = d["route"], d["grid"], False
    twin_key = pairs.set_index(["twin", "twin_grid"])[["route", "grid"]]
    on_twin = d["route"].str.endswith("-010")
    hit = pd.MultiIndex.from_arrays([d.loc[on_twin, "route"], d.loc[on_twin, "grid"]])
    found = twin_key.reindex(hit)
    idx = d.index[on_twin]
    d.loc[idx, "key_route"] = found["route"].to_numpy()
    d.loc[idx, "key_grid"] = found["grid"].to_numpy()
    d.loc[idx, "divided"] = True
    d = d[d["key_route"].notna() & d["key_route"].str.endswith("-000")]
    d["key_grid"] = d["key_grid"].astype(np.int64)
    d["off_center"] = (d["m"].astype(float) / STEP_MI - d["grid"]).abs()
    d["when"] = run_time(d["run"])
    d = d.sort_values(["when", "off_center"], ascending=[False, True], kind="stable")
    d = d.drop_duplicates(["key_route", "key_grid", "slot"])
    d["year"] = year
    return d[["key_route", "key_grid", "slot", "year", "value", "run", "route", "m", "divided"]]


def panel(cells: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Wide panel: one row per mainline cell, ``value_<slot>_<year>`` and coverage counts."""
    wide = cells.pivot_table(index=["key_route", "key_grid"], columns=["slot", "year"],
                             values="value", aggfunc="first")
    wide.columns = [f"value_{s}_{y}" for s, y in wide.columns]
    for s in ("inv", "non"):
        cols = [f"value_{s}_{y}" for y in years if f"value_{s}_{y}" in wide.columns]
        wide[f"years_{s}"] = wide[cols].notna().sum(axis=1)
    divided = cells.groupby(["key_route", "key_grid"])["divided"].any()
    wide["divided"] = divided.reindex(wide.index).fillna(False).to_numpy()
    wide = wide.reset_index().rename(columns={"key_route": "route", "key_grid": "grid"})
    wide.insert(2, "mp", (wide["grid"] * STEP_MI).round(3))
    return wide


def run_qa(matched: pd.DataFrame) -> pd.DataFrame:
    """Per run: cells hit twice or skipped along the run, and the odometer scale implied by GPS.

    The GPS scale is the run's total DMI distance over its total GPS distance, held fixes and
    recovery jumps excluded: chainage = ground x (1 + scale error).

    Fix noise only ever adds length to a 26.4-ft GPS step, so this reads an odometer a little
    short, the more so the noisier the GPS and the slower the vehicle (a step at walking pace is
    mostly noise). With the simulated 3-ft GPS error the median run reads 0.004% short; at 5 ft
    it was 0.07%. Smoothing the track or measuring longer chords would remove it but cuts the
    corners of mountain curves, which costs more than it saves. A miscalibrated odometer (0.3%
    or more) stands well clear of either.
    """
    from pyproj import Transformer

    from .network import EPSG

    d = matched.sort_values(["run", "seq"])
    t = Transformer.from_crs(4326, EPSG, always_xy=True)
    x0, y0 = t.transform(d["lon0"].to_numpy(), d["lat0"].to_numpy())
    x1, y1 = t.transform(d["lon1"].to_numpy(), d["lat1"].to_numpy())
    step = np.hypot(x1 - x0, y1 - y0)
    seg_len = (d["ch_end_ft"] - d["ch_start_ft"]).to_numpy()
    good = (step > 0.5) & (step < 3 * seg_len)
    k = np.round(d["m"].astype(float).to_numpy() / STEP_MI)
    same = (d["run"].to_numpy()[1:] == d["run"].to_numpy()[:-1]) & \
           (d["route"].to_numpy()[1:] == d["route"].to_numpy()[:-1])
    dk = np.abs(np.diff(k))
    frame = pd.DataFrame({"run": d["run"].to_numpy(), "gps_ft": np.where(good, step, 0.0),
                          "dmi_ft": np.where(good, seg_len, 0.0),
                          "twice": np.r_[False, same & (dk == 0)], "skip": np.r_[False, same & (dk == 2)]})
    g = frame.groupby("run")
    out = pd.DataFrame({"segments": g.size(), "hit_twice": g["twice"].sum(), "skipped": g["skip"].sum(),
                        "dmi_scale_gps": g["dmi_ft"].sum() / g["gps_ft"].sum() - 1})
    return out.reset_index()


def changes(cells: pd.DataFrame, threshold: float = 8.0, window: int = 21) -> pd.DataFrame:
    """Likely resurfacing: a cell whose value rose by ``threshold`` or more since the previous
    year it was surveyed, as a centred rolling median over ``window`` cells (0.1 mi), so a
    single noisy reading can't trigger it."""
    d = cells.sort_values(["key_route", "slot", "key_grid", "year"])
    d["prev"] = d.groupby(["key_route", "slot", "key_grid"])["value"].shift(1)
    d["prev_year"] = d.groupby(["key_route", "slot", "key_grid"])["year"].shift(1)
    d = d[d["prev"].notna()].copy()
    d["delta"] = d["value"] - d["prev"]
    d = d.sort_values(["key_route", "slot", "year", "key_grid"])
    d["delta_smooth"] = d.groupby(["key_route", "slot", "year"])["delta"].transform(
        lambda s: s.rolling(window, center=True, min_periods=window // 2).median())
    d["flagged"] = d["delta_smooth"] >= threshold
    return d[["key_route", "key_grid", "slot", "year", "prev_year", "delta", "delta_smooth", "flagged"]]
