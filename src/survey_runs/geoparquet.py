"""Write measured (M) geometry to GeoParquet without losing the measures.

GeoPandas and pyogrio can't write M coordinates, and a GeoParquet file whose
``geo`` header declares a plain ``LineString`` is read by GDAL and QGIS as
un-measured even when the WKB carries M. So the geometry column is built here
as ISO WKB (type codes 2002 / 2005) and the header declares ``"LineString M"``
/ ``"MultiLineString M"``, as GeoParquet 1.1 specifies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS

_LINESTRING_M = 2002
_MULTILINESTRING_M = 2005


def linestring_m(xym: np.ndarray) -> bytes:
    """ISO WKB (little-endian) for a LineString M from an (n, 3) x/y/m array."""
    xym = np.ascontiguousarray(xym, dtype="<f8")
    return b"\x01" + np.array([_LINESTRING_M, len(xym)], dtype="<u4").tobytes() + xym.tobytes()


def multilinestring_m(parts: list[np.ndarray]) -> bytes:
    """ISO WKB (little-endian) for a MultiLineString M from a list of (n, 3) arrays."""
    head = b"\x01" + np.array([_MULTILINESTRING_M, len(parts)], dtype="<u4").tobytes()
    return head + b"".join(linestring_m(p) for p in parts)


def write(path: Path, table: pa.Table, wkb: list[bytes], geometry_type: str, crs: CRS,
          bbox: tuple[float, float, float, float]) -> None:
    """Write ``table`` plus a WKB geometry column as GeoParquet 1.1 with an M geometry type."""
    table = table.append_column("geometry", pa.array(wkb, type=pa.binary()))
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {
            "encoding": "WKB",
            "geometry_types": [geometry_type],
            "crs": crs.to_json_dict(),
            "bbox": [float(v) for v in bbox],
        }},
    }
    meta = dict(table.schema.metadata or {})
    meta[b"geo"] = json.dumps(geo).encode()
    pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd",
                   compression_level=5)


def geometry_types(path: Path) -> list[str]:
    """The geometry types declared in a GeoParquet file's ``geo`` header."""
    geo = json.loads(pq.read_schema(path).metadata[b"geo"])
    return geo["columns"][geo["primary_column"]]["geometry_types"]
