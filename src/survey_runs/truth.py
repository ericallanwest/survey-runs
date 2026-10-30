"""Synthetic ground truth: a friction-like index on every 0.005-mi cell, by year and direction.

The index is invented, on a 0-100 scale, and behaves the way skid resistance tends to:

* the road is made of pavement *sections* (about a mile long on average), each with its own
  surface level;
* each direction of travel wears at its own rate, so the level drops a little every year;
* now and then a section is resurfaced between two survey years: its level jumps up for both
  directions and the wear starts over;
* within a section the value wanders around the level (spatially correlated noise), and a new
  surface brings new texture;
* on top of that, every cell has its own small local feature (a patch, a joint, a polished
  wheel path), the same for every run that reads it.

Truth lives on the ``-000`` mainline's measure axis. A reading taken on a ``-010`` twin
carriageway belongs to the mainline cell beside it, in the "non" (against-measure) direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CELL_MI = 0.005
DIRECTIONS = ("inv", "non")


@dataclass
class Truth:
    years: tuple[int, ...]
    routes: dict[str, tuple[int, int, int]]   # route -> (first cell number, n cells, offset)
    values: np.ndarray                        # (direction, year, flat cell) float32
    sections: pd.DataFrame
    treatments: pd.DataFrame

    def at(self, route, direction, year, m) -> np.ndarray:
        """Truth at measure ``m`` (miles) on -000 ``route``; vectorized, NaN where unknown."""
        route = np.asarray(route, dtype=object)
        direction = np.broadcast_to(np.asarray(direction, dtype=object), route.shape)
        m = np.asarray(m, dtype=float)
        out = np.full(route.shape, np.nan, dtype=np.float32)
        yi = self.years.index(year)
        codes, uniques = pd.factorize(route, use_na_sentinel=True)
        for code, r in enumerate(uniques):
            if r not in self.routes:
                continue
            k0, n, off = self.routes[r]
            sel = codes == code
            k = np.floor(m[sel] / CELL_MI + 1e-9).astype(np.int64) - k0
            ok = (k >= 0) & (k < n) & np.isfinite(m[sel])
            d = np.where(direction[sel] == "inv", 0, 1)
            vals = np.full(sel.sum(), np.nan, dtype=np.float32)
            vals[ok] = self.values[d[ok], yi, off + k[ok]]
            out[sel] = vals
        return out

    def save(self, path) -> None:
        np.savez_compressed(path, years=np.array(self.years), values=self.values,
                            routes=np.array(list(self.routes), dtype=object),
                            index=np.array(list(self.routes.values())), allow_pickle=True)

    @classmethod
    def load(cls, path, sections: pd.DataFrame, treatments: pd.DataFrame) -> Truth:
        z = np.load(path, allow_pickle=True)
        routes = {r: tuple(int(v) for v in idx) for r, idx in zip(z["routes"], z["index"])}
        return cls(tuple(int(y) for y in z["years"]), routes, z["values"], sections, treatments)


def _texture(rng, n: int, phi: float) -> np.ndarray:
    """Correlated wander (SD 3) plus each cell's own local feature (SD 1.5)."""
    return _ar1(rng, n, phi, 3.0) + rng.normal(0.0, 1.5, n)


def _ar1(rng, n: int, phi: float, sd: float) -> np.ndarray:
    """A stationary AR(1) series: spatially correlated noise along a road."""
    e = rng.normal(0.0, sd * np.sqrt(1 - phi * phi), n)
    e[0] = rng.normal(0.0, sd)
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -phi], e)


def generate(table: pd.DataFrame, parts: dict[str, list[np.ndarray]], years: tuple[int, ...],
             seed: int = 7, section_mi: float = 1.2, treat_p: float = 0.06) -> Truth:
    rng = np.random.default_rng(seed)
    phi = np.exp(-CELL_MI / 0.04)           # correlation length about 0.04 mi (200 ft)
    routes, blocks, sec_rows, treat_rows = {}, [], [], []
    offset = 0
    for r in sorted(table.loc[table["ROUTE"].str.endswith("-000"), "ROUTE"]):
        lo = min(p[0, 2] for p in parts[r])
        hi = max(p[-1, 2] for p in parts[r])
        k0 = int(np.floor(lo / CELL_MI + 1e-9))
        n = int(np.ceil(hi / CELL_MI - 1e-9)) - k0
        routes[r] = (k0, n, offset)
        offset += n
        block = np.empty((2, len(years), n), dtype=np.float32)

        # section boundaries along the route, in cells
        cuts = [0]
        while True:
            nxt = cuts[-1] + max(20, int(rng.exponential(section_mi) / CELL_MI))
            if nxt >= n:
                break
            cuts.append(nxt)
        cuts.append(n)
        for si, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            level = float(np.clip(rng.normal(52, 7), 30, 75))
            age = rng.uniform(0, 8)                       # years since this surface was laid
            wear = rng.uniform(0.4, 1.4, 2)               # per direction, points per year
            offset_dir = rng.normal(0, 2, 2)
            texture = [_texture(rng, b - a, phi) for _ in DIRECTIONS]
            sec_rows.append((r, si, (k0 + a) * CELL_MI, (k0 + b) * CELL_MI, level))
            for yi, year in enumerate(years):
                if yi > 0 and rng.random() < treat_p:     # resurfaced since last year's survey
                    level = float(np.clip(rng.normal(66, 4), 55, 80))
                    age = rng.uniform(0, 0.8)
                    wear = rng.uniform(0.4, 1.4, 2)
                    texture = [_texture(rng, b - a, phi) for _ in DIRECTIONS]
                    treat_rows.append((r, si, (k0 + a) * CELL_MI, (k0 + b) * CELL_MI, year))
                for d in range(2):
                    block[d, yi, a:b] = level - wear[d] * age + offset_dir[d] + texture[d]
                age += 1.0
        blocks.append(block)
    values = np.concatenate(blocks, axis=2)
    sections = pd.DataFrame(sec_rows, columns=["route", "section", "m0", "m1", "initial_level"])
    treatments = pd.DataFrame(treat_rows, columns=["route", "section", "m0", "m1", "year"])
    return Truth(tuple(years), routes, values, sections, treatments)
