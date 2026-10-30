"""The project page's data: ``docs/data/`` from the analysis outputs.

* ``site.json``: headline numbers, the scores, the odometer check and the repeatability tables;
* ``changes.geojson``: every mainline cut into 0.1-mi pieces, with the change since the previous
  survey in each direction and year (the median of the smoothed cell deltas in the piece);
* ``routes/<route>.json``: each route's profile by year and direction in 0.02-mi bins, the
  stretches flagged as resurfaced, and the simulator's true resurfacing;
* ``repeat/<site>_<year>.json``: along each site and per pairing method, how far the six runs
  disagree and how far (by the truth) the worst-paired run sits from each station, and the runs
  side by side over the site's last 0.2 mi.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from . import network
from .network import EPSG
from .paths import cut

PIECE_MI = 0.1          # map pieces
BIN_MI = 0.02           # profile bins
CELL_MI = 0.005
BLOCK_MI = 0.05         # 264-ft blocks along the repeat sites, as BRSD uses
MIN_BLOCK = 5           # positions a block needs to be shown
ZOOM_MI = 0.2           # the side-by-side window at each site's far end
SIMPLIFY_FT = 10.0


def _r(v, nd=1):
    """JSON-ready rounding: None for NaN."""
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"))
    path.write_text(text)
    return len(text)


def _spans(grid: np.ndarray, gap: int = 4) -> list[list[float]]:
    """Runs of cell numbers as [m0, m1] milepoint spans, bridging gaps of up to ``gap`` cells
    (a cell one year missed)."""
    if not len(grid):
        return []
    g = np.sort(grid)
    brk = np.flatnonzero(np.diff(g) > gap + 1)
    starts, ends = np.r_[g[0], g[brk + 1]], np.r_[g[brk], g[-1]]
    return [[round(a * CELL_MI, 3), round((b + 1) * CELL_MI, 3)] for a, b in zip(starts, ends)]


def changes_geojson(table, parts, chg: pd.DataFrame, years) -> dict:
    to_wgs = Transformer.from_crs(EPSG, 4326, always_xy=True)
    per = int(round(PIECE_MI / CELL_MI))
    c = chg.dropna(subset=["delta_smooth"]).assign(piece=lambda d: d["key_grid"] // per)
    med = c.groupby(["key_route", "piece", "slot", "year"])["delta_smooth"].median()
    props: dict[tuple[str, int], dict] = {}
    for (route, piece, slot, year), v in med.items():
        props.setdefault((route, piece), {})[f"{slot[0]}{year % 100}"] = round(float(v), 1)
    feats = []
    for route in table.loc[table["ROUTE"].str.endswith("-000"), "ROUTE"]:
        for xym in parts[route]:
            lo, hi = xym[0, 2], xym[-1, 2]
            for piece in range(int(np.floor(lo / PIECE_MI + 1e-9)), int(np.ceil(hi / PIECE_MI - 1e-9))):
                seg = cut(xym, piece * PIECE_MI, (piece + 1) * PIECE_MI)
                if len(seg) < 2:
                    continue
                line = shapely.simplify(shapely.linestrings(seg[:, :2]), SIMPLIFY_FT)
                xy = shapely.get_coordinates(line)
                lon, lat = to_wgs.transform(xy[:, 0], xy[:, 1])
                feats.append({"type": "Feature",
                              "properties": {"r": route, "m": round(piece * PIECE_MI, 1),
                                             **props.get((route, piece), {})},
                              "geometry": {"type": "LineString",
                                           "coordinates": np.round(np.column_stack([lon, lat]), 5).tolist()}})
    return {"type": "FeatureCollection", "features": feats}


def route_profiles(table, panel: pd.DataFrame, chg: pd.DataFrame, treat: pd.DataFrame, years, out: Path) -> int:
    per = int(round(BIN_MI / CELL_MI))
    info = table.set_index("ROUTE")
    flagged = chg[chg["flagged"]].groupby(["key_route", "slot", "year"])["key_grid"].apply(np.asarray)
    size = 0
    for route, p in panel.groupby("route"):
        b = (p["grid"] // per).to_numpy()
        k0, k1 = int(b.min()), int(b.max())
        prof = {}
        for slot in ("inv", "non"):
            prof[slot] = {}
            for y in years:
                col = f"value_{slot}_{y}"
                med = pd.Series(p[col].to_numpy()).groupby(b).median().reindex(range(k0, k1 + 1))
                if med.notna().any():
                    prof[slot][str(y)] = [_r(v) for v in med.to_numpy()]
        flags = {slot: {str(y): _spans(flagged[(route, slot, y)]) for y in years
                        if (route, slot, y) in flagged.index} for slot in ("inv", "non")}
        r = info.loc[route]
        t = treat[treat["route"] == route]
        size += _write(out / f"{route}.json", {
            "route": route, "name": None if pd.isna(r["RD_NAME"]) else str(r["RD_NAME"]),
            "county": str(r["CO_NAME"]), "beg": _r(r["BEG_MP"], 3), "end": _r(r["END_MP"], 3),
            "x0": round(k0 * BIN_MI, 3), "dx": BIN_MI, "profile": prof, "flagged": flags,
            "treated": [[round(a, 3), round(z, 3), int(y)] for a, z, y in zip(t["m0"], t["m1"], t["year"])],
        })
    return size


def repeat_files(aligned: pd.DataFrame, runs: pd.DataFrame, sites: pd.DataFrame, out: Path) -> int:
    size = 0
    run_info = runs.set_index("run")
    for (site, year), g in aligned.groupby(["site", "year"]):
        s = sites.loc[site]
        names = sorted(g["run"].unique())
        methods = {}
        for method, h in g.groupby("method"):
            wide = h.pivot_table(index="mp", columns="run", values="value").reindex(columns=names)
            at = wide.index.to_numpy() - s["m_start"]
            block = np.floor(at / BLOCK_MI).astype(int)
            grouped = wide.groupby(block)
            # a block left with a few valid positions (a slowdown) has a mean too noisy to show
            sd = grouped.mean().std(axis=1, ddof=1).where(grouped.size() >= MIN_BLOCK)
            # how far (truth) the worst-paired run's segment lies from each station
            off = h.groupby("mp")["worst_offset_ft"].first().reindex(wide.index).groupby(block).median()
            zoom = wide[at >= (s["m_end"] - s["m_start"]) - ZOOM_MI]
            methods[method] = {
                "block_sd": [[round(int(k) * BLOCK_MI, 3), _r(v, 2)] for k, v in sd.items()],
                "block_off": [[round(int(k) * BLOCK_MI, 3), _r(v, 1)] for k, v in off.items()],
                "zoom_x": [round(float(v - s["m_start"]), 4) for v in zoom.index],
                "zoom": [[_r(v) for v in zoom[n].to_numpy()] for n in names],
            }
        size += _write(out / f"{site}_{year}.json", {
            "site": site, "year": int(year), "length_mi": _r(s["m_end"] - s["m_start"], 3),
            "runs": [{"run": n, "vehicle": run_info.loc[n, "vehicle"],
                      "dmi_pct": _r(100 * run_info.loc[n, "dmi_scale"], 3)} for n in names],
            "methods": methods,
        })
    return size


def build(data: Path, docs: Path, log=print) -> None:
    data, docs = Path(data), Path(docs)
    out = docs / "data"
    an, sim = data / "analysis", data / "sim"
    table, parts = network.read(data / "network.parquet")
    net = json.loads((data / "network.json").read_text())
    summary = json.loads((an / "summary.json").read_text())
    years = summary["years"]
    runs = pd.read_parquet(sim / "runs.parquet")
    sites = pd.read_parquet(sim / "sites.parquet").set_index("site")
    chg = pd.read_parquet(an / "changes.parquet")

    gj = changes_geojson(table, parts, chg, years)
    xy = np.vstack([f["geometry"]["coordinates"] for f in gj["features"]])
    bounds = [[float(xy[:, 0].min()), float(xy[:, 1].min())], [float(xy[:, 0].max()), float(xy[:, 1].max())]]
    n = _write(out / "changes.geojson", gj)
    log(f"changes.geojson: {n / 1e6:.1f} MB")
    treat = pd.read_parquet(sim / "treatments.parquet")
    # the page opens on the route-year with the longest stretch both resurfaced and flagged, of
    # the routes short enough (5-25 mi) for one chart
    hits = (chg[chg["flagged"] & (chg["slot"] == "inv")].groupby(["key_route", "year"]).size()
            .rename("cells").reset_index().merge(treat[["route", "year"]].drop_duplicates(),
                                                 left_on=["key_route", "year"], right_on=["route", "year"])
            .merge(table[["ROUTE", "LENGTH_MI"]], left_on="key_route", right_on="ROUTE"))
    best = hits[hits["LENGTH_MI"].between(5, 25)].sort_values("cells").iloc[-1]
    featured = {"route": best["key_route"], "year": int(best["year"])}
    n = route_profiles(table, pd.read_parquet(an / "panel.parquet"), chg, treat, years, out / "routes")
    log(f"routes/: {n / 1e6:.1f} MB")
    n = repeat_files(pd.read_parquet(an / "repeat_aligned.parquet"), runs, sites, out / "repeat")
    log(f"repeat/: {n / 1e6:.2f} MB")

    twins = table[table["ROUTE"].str.endswith("-010")]
    rep = pd.read_parquet(an / "repeat.parquet")
    settings = json.loads((sim / "settings.json").read_text())
    _write(out / "site.json", {
        "network": {"counties": net["counties"], "routes": net["routes"], "parts": net["parts"],
                    "miles": net["miles"], "fetched": net["fetched"][:10], "service": net["service"],
                    "mainline_routes": int(table["ROUTE"].str.endswith("-000").sum()),
                    "mainline_miles": round(float(table.loc[table["ROUTE"].str.endswith("-000"), "LENGTH_MI"].sum()), 1),
                    "twin_routes": len(twins), "twin_miles": round(float(twins["LENGTH_MI"].sum()), 1),
                    "bounds": bounds},
        "featured": featured,
        "program": {"years": years, "vehicles": settings["vehicles"],
                    "network_runs": int((runs["kind"] == "network").sum()),
                    "repeat_runs": int((runs["kind"] == "repeat").sum()),
                    "segments": summary["segments"], "panel_cells": summary["panel_cells"]},
        "settings": settings,
        "scores": summary["scores"],
        "repeat": {"summary": json.loads((an / "repeat_summary.json").read_text()),
                   "baskets": [{k: (_r(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                               for row in rep.to_dict("records")]},
        "sites": [{"site": k, "corridor": r["corridor"], "route": r["route_start"],
                   "county": str(table.set_index("ROUTE").loc[r["route_start"], "CO_NAME"]),
                   "m_start": round(r["m_start"], 3), "m_end": round(r["m_end"], 3)}
                  for k, r in sites.iterrows()],
    })
    log(f"site.json written to {out}")
