"""The road network: KYTC's public All Roads LRS for the counties of one highway district.

KYTC publishes its linear referencing system as an ArcGIS feature service with the
milepoint stored as the M ordinate (miles). Each route (``RT_UNIQUE``) arrives as several
features; they are chained into parts here. Divided highways carry a second carriageway as
its own route, ``-010`` beside the ``-000`` mainline, digitized in the same (inventory)
direction, so traffic on it drives against its measure.

Coordinates stay in the service's native CRS, EPSG:3089 (NAD83 / Kentucky Single Zone,
US survey feet).
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from pyproj import CRS

from . import geoparquet

SERVICE = ("https://services2.arcgis.com/CcI36Pduqd0OR4W9/ArcGIS/rest/services/"
           "KYTC_All_Roads/FeatureServer/0/query")
EPSG = 3089
DISTRICT_12 = ["Floyd", "Johnson", "Knott", "Lawrence", "Letcher", "Martin", "Pike"]
FIELDS = ["RT_UNIQUE", "RT_PREFIX", "RT_NUMBER", "RT_SUFFIX", "RT_SECTION", "CO_NAME",
          "RD_NAME", "BEGIN_MP", "END_MP", "GOV_LEVEL"]
PAGE = 2000
JOIN_FT = 1.0       # a feature continues the previous one if it starts within this distance
MP_TOL = 0.0015     # ... and its BEGIN_MP meets the previous END_MP within this, miles


def route_id(rt_unique: str) -> str:
    """KYTC's RT_UNIQUE with its padding removed: ``098-US-0023  -000`` -> ``098-US-0023-000``."""
    return rt_unique.replace(" ", "")


def fetch(counties: list[str], prefixes=("US", "KY"), log=print) -> list[dict]:
    """State-maintained route features (paths with M) for the counties, page by page."""
    names = ",".join(f"'{c}'" for c in counties)
    kinds = ",".join(f"'{p}'" for p in prefixes)
    where = f"CO_NAME IN ({names}) AND RT_PREFIX IN ({kinds}) AND GOV_LEVEL = '01'"
    feats, offset = [], 0
    while True:
        params = {"where": where, "outFields": ",".join(FIELDS), "returnM": "true",
                  "outSR": EPSG, "orderByFields": "OBJECTID", "resultOffset": offset,
                  "resultRecordCount": PAGE, "f": "json"}
        for attempt in range(4):
            try:
                r = requests.get(SERVICE, params=params, timeout=300)
                r.raise_for_status()
                page = r.json()
                if "error" in page:
                    raise RuntimeError(page["error"])
                break
            except (requests.RequestException, RuntimeError) as e:
                if attempt == 3:
                    raise
                log(f"retrying page at {offset}: {e}")
                time.sleep(5 * (attempt + 1))
        feats += page.get("features", [])
        log(f"fetched {len(feats):,} features")
        if not page.get("exceededTransferLimit") and len(page.get("features", [])) < PAGE:
            return feats
        offset += PAGE


def assemble(feats: list[dict]) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Chain each route's features into parts of (n, 3) x/y/m arrays.

    Features join when one starts where the last ended (``JOIN_FT``) and its milepoints
    continue (``MP_TOL``); anything else starts a new part, with no geometry invented.
    Returns the route table and ``{route_id: [part, ...]}``.
    """
    pieces = []
    for f in feats:
        a = f["attributes"]
        paths = [np.asarray(p, dtype=float) for p in f["geometry"]["paths"]]
        if any(p.shape[1] != 3 or len(p) < 2 for p in paths):
            raise ValueError(f"{a['RT_UNIQUE']}: path without M")
        # The service's M is local to each feature (0 at its start); anchor it to the
        # feature's own milepoints so it reads as the route's measure.
        lo = min(p[0, 2] for p in paths)
        hi = max(p[-1, 2] for p in paths)
        beg, end = a["BEGIN_MP"], a["END_MP"]
        scale = (end - beg) / (hi - lo) if hi > lo else 0.0
        for p in paths:
            p[:, 2] = beg + (p[:, 2] - lo) * scale
            if end < beg:
                # Twin carriageways (-010) come digitized in their direction of travel, with
                # falling milepoints. Flip them so every part runs with increasing M, in the
                # inventory direction like its -000 mainline; traffic drives them against M.
                p = p[::-1].copy()
            pieces.append((route_id(a["RT_UNIQUE"]), p[0, 2], p, a))
    pieces.sort(key=lambda p: (p[0], p[1]))

    parts: dict[str, list[np.ndarray]] = {}
    rows = {}
    for rid, _, xym, a in pieces:
        own = parts.setdefault(rid, [])
        if own:
            last = own[-1]
            if (np.hypot(*(xym[0, :2] - last[-1, :2])) <= JOIN_FT
                    and abs(xym[0, 2] - last[-1, 2]) <= MP_TOL):
                own[-1] = np.vstack([last, xym[1:]])
                continue
        own.append(xym)
        rows.setdefault(rid, {k: a[k] for k in ("RT_PREFIX", "RT_NUMBER", "RT_SUFFIX",
                                                "RT_SECTION", "CO_NAME", "RD_NAME")})
    for rid, own in parts.items():
        for xym in own:
            if (np.diff(xym[:, 2]) < -1e-9).any():
                raise ValueError(f"{rid}: M decreases inside a part")
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("ROUTE").reset_index()
    table["BEG_MP"] = [min(p[0, 2] for p in parts[r]) for r in table["ROUTE"]]
    table["END_MP"] = [max(p[-1, 2] for p in parts[r]) for r in table["ROUTE"]]
    table["N_PARTS"] = [len(parts[r]) for r in table["ROUTE"]]
    table["LENGTH_MI"] = [sum(float(np.hypot(*np.diff(p[:, :2], axis=0).T).sum()) for p in parts[r]) / 5280
                          for r in table["ROUTE"]]
    return table, parts


def write(table: pd.DataFrame, parts: dict[str, list[np.ndarray]], path: Path) -> None:
    """One MultiLineString M per route as GeoParquet."""
    wkb = [geoparquet.multilinestring_m(parts[r]) for r in table["ROUTE"]]
    xy = np.vstack([p[:, :2] for r in table["ROUTE"] for p in parts[r]])
    geoparquet.write(path, pa.Table.from_pandas(table, preserve_index=False), wkb,
                     "MultiLineString M", CRS.from_epsg(EPSG), (*xy.min(axis=0), *xy.max(axis=0)))


def read(path: Path) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    import pyarrow.parquet as pq
    import shapely

    t = pq.read_table(path)
    table = t.drop_columns(["geometry"]).to_pandas()
    geoms = shapely.from_wkb(t.column("geometry").to_numpy(False))
    sub, owner = shapely.get_parts(geoms, return_index=True)
    xym, which = shapely.get_coordinates(sub, include_m=True, return_index=True)
    cuts = np.flatnonzero(np.diff(which)) + 1
    parts: dict[str, list[np.ndarray]] = {}
    for r, coords in zip(owner, np.split(xym, cuts)):
        parts.setdefault(table["ROUTE"].iat[r], []).append(coords)
    return table, parts


def build(out_dir: Path, counties=DISTRICT_12, log=print) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = fetch(counties, log=log)
    table, parts = assemble(feats)
    path = out_dir / "network.parquet"
    write(table, parts, path)
    (out_dir / "network.json").write_text(json.dumps({
        "service": SERVICE, "counties": list(counties), "features": len(feats),
        "routes": len(table), "parts": int(table["N_PARTS"].sum()),
        "miles": round(float(table["LENGTH_MI"].sum()), 1),
        "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    log(f"{len(table):,} routes, {table['N_PARTS'].sum():,} parts, {table['LENGTH_MI'].sum():,.0f} mi "
        f"({table['ROUTE'].str.endswith('-010').sum()} -010 carriageways) -> {path}")
    return path
