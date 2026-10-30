"""Simulate a multi-year survey program over the corridors, keeping every reading's truth.

What the vehicle records, per 0.005-mi segment of DMI (odometer) distance: chainage, a GPS
fix at each end, speed and the measured index. What goes wrong, on purpose, because the
analysis has to cope with it:

* **DMI scale error.** Each vehicle-year has its own small odometer error, each run strays a
  little from it (tire temperature and pressure), and one vehicle-year is miscalibrated
  outright, so segments drift against the road's measure;
* **GPS error** that wanders slowly (correlated in time) and **held fixes**, where the
  receiver repeats one position for several seconds;
* **lane offset**: the vehicle drives its lane, not the centerline;
* **speed**: stops and slowdowns lower the speed, which shifts the reading, and readings
  below 15 mph are invalid;
* **season**: readings run high in spring and low in late summer;
* **coverage**: not every corridor and direction is surveyed every year, some runs start
  late or stop early, and some are driven twice in a year;
* **vehicles**: two survey vehicles, each with its own small bias.

Repeat runs over a handful of 3-mile reference sites, three passes per vehicle on one day a
year, feed the repeatability analysis. Each starts and stops a quarter mile either side of the
site, give or take, as an operator starting at a landmark would.

Outputs (``data/sim/``): ``segments.parquet`` (what the vehicle exports, WGS84, no truth),
``runs.parquet`` (run log), and the truth that scores the analysis: ``truth_segments.parquet``,
``truth.npz``, ``sections.parquet``, ``treatments.parquet``, ``sites.parquet``.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .network import EPSG
from .paths import DrivePath, corridors, drive_path
from .truth import Truth

FT_PER_S_PER_MPH = 5280 / 3600


@dataclass(frozen=True)
class Settings:
    years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
    vehicles: tuple[str, ...] = ("SV1", "SV2")
    vehicle_bias: tuple[float, ...] = (0.8, -0.6)
    coverage: float = 0.88
    redrive: float = 0.05
    partial: float = 0.08
    run_max_mi: float = 30.0
    seg_ft: float = 26.4
    speed_mph: tuple[float, float] = (35.0, 45.0)
    stop_every_mi: float = 1.5
    min_valid_mph: float = 15.0
    lane_offset_ft: float = 6.0
    gps_sd_ft: float = 3.0
    gps_tau_s: float = 20.0
    hold_p: float = 0.3
    dmi_sd: float = 0.0012
    dmi_run_sd: float = 0.0005
    dmi_fault: tuple = ("SV2", 2023, -0.005)
    noise_sd: float = 1.5
    speed_coef: float = -0.12
    season_amp: float = 3.0
    season_days: tuple[int, int] = (95, 295)
    sites: int = 6
    site_mi: float = 3.0
    repeats: int = 3
    run_in_ft: float = 1320.0      # repeat runs start and stop this far either side of the site,
    run_in_jitter_ft: float = 60.0  # give or take this much


def season(doy, amp: float) -> np.ndarray:
    """Seasonal shift: +amp in mid-April, falling through zero in late June to -amp in September."""
    return amp * np.cos(np.pi * (np.asarray(doy, float) - 100.0) / 150.0)


def _speed_profile(rng, length_ft: float, s: Settings):
    grid = np.arange(0.0, length_ft + 20.0, 10.0)
    v0 = rng.uniform(*s.speed_mph)
    v = np.full(grid.shape, v0)
    for at in rng.uniform(0, length_ft, rng.poisson(length_ft / 5280 / s.stop_every_mi)):
        floor, width = rng.uniform(0.0, 0.3), rng.uniform(150, 400)
        v *= 1 - (1 - floor) * np.exp(-((grid - at) / width) ** 2)
    v *= np.clip(0.55 + grid / 1200.0, 0.0, 1.0)     # run-up from the start
    v = np.maximum(v, 1.0)
    t = np.concatenate([[0.0], np.cumsum(np.diff(grid) / (0.5 * (v[1:] + v[:-1]) * FT_PER_S_PER_MPH))])
    return grid, v, t


def _gps_error(rng, t: np.ndarray, sd: float, tau: float) -> np.ndarray:
    err = np.empty((len(t), 2))
    err[0] = rng.normal(0, sd, 2)
    rho = np.exp(-np.diff(t) / tau)
    for i in range(1, len(t)):
        err[i] = rho[i - 1] * err[i - 1] + np.sqrt(1 - rho[i - 1] ** 2) * rng.normal(0, sd, 2)
    return err


def _hold(rng, t: np.ndarray, fix: np.ndarray) -> np.ndarray:
    """Freeze the GPS fix for a few seconds, once to three times in the run."""
    fix = fix.copy()
    for _ in range(rng.integers(1, 4)):
        i = rng.integers(0, len(t))
        held = (t >= t[i]) & (t <= t[i] + rng.uniform(3, 15))
        fix[held] = fix[i]
    return fix


def _run(rng, path: DrivePath, s0: float, s1: float, *, year, vehicle, bias, dmi, doy,
         truth: Truth, s: Settings, to_wgs):
    length = s1 - s0
    n = int(length * (1 + dmi) // s.seg_ft)
    if n < 4:
        return None
    grid, _, t_grid = _speed_profile(rng, length, s)
    chain = np.arange(n + 1) * s.seg_ft                 # DMI chainage at segment boundaries
    ground = chain / (1 + dmi)                           # true distance driven
    t = np.interp(ground, grid, t_grid)
    xy, heading, *_ = path.sample(s0 + ground)
    right = np.column_stack([np.sin(heading), -np.cos(heading)])
    fix = xy + s.lane_offset_ft * right + _gps_error(rng, t, s.gps_sd_ft, s.gps_tau_s)
    if rng.random() < s.hold_p:
        fix = _hold(rng, t, fix)
    lon, lat = to_wgs.transform(fix[:, 0], fix[:, 1])

    speed = (np.diff(ground) / np.diff(t)) / FT_PER_S_PER_MPH
    s_mid = s0 + 0.5 * (ground[:-1] + ground[1:])
    _, _, base, m, twin, m_route = path.sample(s_mid)
    true = pd.Series(truth.at(base, path.direction, year, m)).ffill().bfill().to_numpy()
    measured = (true + season(doy, s.season_amp) + s.speed_coef * (speed - 40.0) + bias
                + rng.normal(0, s.noise_sd, n))
    seg = pd.DataFrame({
        "seq": np.arange(n), "ch_start_ft": chain[:-1], "ch_end_ft": chain[1:],
        "lon0": lon[:-1], "lat0": lat[:-1], "lon1": lon[1:], "lat1": lat[1:],
        "speed_mph": speed.round(1), "value": measured.round(1),
        "valid": speed >= s.min_valid_mph,
    })
    tru = pd.DataFrame({"seq": np.arange(n), "route": base, "direction": path.direction,
                        "m": m, "on_twin": twin, "m_route": m_route, "s_true": s_mid.round(2),
                        "value_true": true.round(2)})
    return seg, tru, float(t[-1])


def simulate(table, parts, truth: Truth, s: Settings = Settings(), seed: int = 11, log=print):
    rng = np.random.default_rng(seed)
    to_wgs = Transformer.from_crs(EPSG, 4326, always_xy=True)
    corr = corridors(table, parts)
    dmi = {(v, y): rng.normal(0, s.dmi_sd) for v in s.vehicles for y in s.years}
    if s.dmi_fault:
        dmi[(s.dmi_fault[0], s.dmi_fault[1])] = s.dmi_fault[2]
    bias = dict(zip(s.vehicles, s.vehicle_bias))
    cache: dict[tuple[str, str], DrivePath] = {}

    def path_for(cid, chain, direction):
        if (cid, direction) not in cache:
            cache[(cid, direction)] = drive_path(cid, chain, parts, direction)
        return cache[(cid, direction)]

    segs, trus, runs = [], [], []
    free: dict[tuple[str, dt.date], dt.datetime] = {}   # each vehicle's next free time, per day

    def slot(vehicle: str, day: dt.date) -> dt.datetime:
        """When the vehicle can start its next run that day (or the next day after 6 pm)."""
        while True:
            start = free.get((vehicle, day), dt.datetime.combine(day, dt.time(8))
                             + dt.timedelta(minutes=int(rng.integers(0, 60))))
            if start.time() < dt.time(18):
                return start
            day += dt.timedelta(days=1)

    def done(vehicle: str, start: dt.datetime, seconds: float) -> None:
        free[(vehicle, start.date())] = start + dt.timedelta(seconds=seconds + int(rng.integers(600, 2400)))

    def record(run, vehicle, year, when, path, out, kind, scale, site=None, partial=False):
        seg, tru, seconds = out
        seg.insert(0, "run", run)
        tru.insert(0, "run", run)
        segs.append(seg)
        trus.append(tru)
        runs.append({"run": run, "vehicle": vehicle, "year": year, "start": when,
                     "corridor": path.corridor, "direction": path.direction, "kind": kind,
                     "site": site, "partial": partial, "segments": len(seg), "dmi_scale": scale,
                     "dmi_vehicle_year": dmi[(vehicle, year)], "duration_s": round(seconds)})
        return seconds

    def run_scale(vehicle, year):
        return dmi[(vehicle, year)] + rng.normal(0, s.dmi_run_sd)

    for year in s.years:
        jobs = []
        for cid, chain in corr:
            for direction in ("inv", "non"):
                if rng.random() < s.coverage:
                    jobs.append((cid, chain, direction))
                    if rng.random() < s.redrive:
                        jobs.append((cid, chain, direction))
        for cid, chain, direction in jobs:
            path = path_for(cid, chain, direction)
            vehicle = s.vehicles[rng.integers(len(s.vehicles))]
            day = dt.date(year, 1, 1) + dt.timedelta(days=int(rng.integers(*s.season_days)) - 1)
            n_runs = max(1, int(np.ceil(path.length / 5280 / s.run_max_mi)))
            cuts = np.sort(np.concatenate([[0, path.length], rng.uniform(0, path.length, n_runs - 1)]))
            for a, b in zip(cuts[:-1], cuts[1:]):
                partial = rng.random() < s.partial
                if partial:
                    a += rng.uniform(0, min(5280, 0.3 * (b - a)))
                    b -= rng.uniform(0, min(5280, 0.3 * (b - a)))
                clock = slot(vehicle, day)
                day = clock.date()
                run = f"{vehicle}_{clock:%y%m%d_%H%M%S}"
                scale = run_scale(vehicle, year)
                out = _run(rng, path, a, b, year=year, vehicle=vehicle, bias=bias[vehicle],
                           dmi=scale, doy=clock.timetuple().tm_yday, truth=truth, s=s, to_wgs=to_wgs)
                if out is None:
                    continue
                done(vehicle, clock, record(run, vehicle, year, clock, path, out, "network", scale,
                                            partial=partial))
        log(f"{year}: {len(jobs)} corridor-directions surveyed")

    # reference sites: stretches of long corridors, inventory direction, each with its run-in
    # and run-out on one route (no county line to restart the measure)
    reach = s.run_in_ft + s.run_in_jitter_ft
    long_ones = [(cid, chain) for cid, chain in corr
                 if path_for(cid, chain, "inv").length > (s.site_mi + 2) * 5280]
    site_rows, chosen = [], []
    for idx in rng.permutation(len(long_ones)):
        if len(chosen) == s.sites:
            break
        cid, chain = long_ones[idx]
        path = path_for(cid, chain, "inv")
        start = rng.uniform(5280, path.length - 5280 - s.site_mi * 5280)
        end = start + s.site_mi * 5280
        _, _, base, m, *_ = path.sample(np.linspace(start - reach, end + reach, 200))
        if len(set(base)) != 1 or base[0] is None:
            continue
        _, _, base, m, *_ = path.sample(np.array([start, end]))
        site = f"R{len(chosen) + 1}"
        site_rows.append({"site": site, "corridor": cid, "route_start": base[0], "m_start": m[0],
                          "route_end": base[1], "m_end": m[1]})
        chosen.append((site, path, start, end))
    for site, path, start, end in chosen:
        for year in s.years:
            day = dt.date(year, 1, 1) + dt.timedelta(days=int(rng.integers(*s.season_days)) - 1)
            for vehicle in s.vehicles:
                for _ in range(s.repeats):
                    clock = slot(vehicle, day)
                    run = f"{vehicle}_{clock:%y%m%d_%H%M%S}"
                    a, b = rng.uniform(-s.run_in_jitter_ft, s.run_in_jitter_ft, 2)
                    scale = run_scale(vehicle, year)
                    out = _run(rng, path, start - s.run_in_ft + a, end + s.run_in_ft + b, year=year,
                               vehicle=vehicle, bias=bias[vehicle], dmi=scale,
                               doy=clock.timetuple().tm_yday, truth=truth, s=s, to_wgs=to_wgs)
                    done(vehicle, clock, record(run, vehicle, year, clock, path, out, "repeat", scale,
                                                site=site))
    log(f"reference sites: {len(site_rows)}, {sum(r['kind'] == 'repeat' for r in runs)} repeat runs")
    names = pd.Series([r["run"] for r in runs])
    if names.duplicated().any():
        raise RuntimeError(f"duplicate run names: {names[names.duplicated()].tolist()[:5]}")

    return (pd.DataFrame(runs), pd.concat(segs, ignore_index=True),
            pd.concat(trus, ignore_index=True), pd.DataFrame(site_rows))


def write(out_dir: Path, truth: Truth, runs, segs, trus, sites, settings: Settings) -> None:
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segs.to_parquet(out / "segments.parquet", index=False)
    runs.to_parquet(out / "runs.parquet", index=False)
    trus.to_parquet(out / "truth_segments.parquet", index=False)
    sites.to_parquet(out / "sites.parquet", index=False)
    truth.sections.to_parquet(out / "sections.parquet", index=False)
    truth.treatments.to_parquet(out / "treatments.parquet", index=False)
    truth.save(out / "truth.npz")
    (out / "settings.json").write_text(json.dumps(asdict(settings), indent=2, default=list))
