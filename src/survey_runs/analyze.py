"""Run the analysis on the simulated surveys, then score every step against the truth.

The analysis side reads only what a survey vehicle would export (``segments.parquet``) plus
the public network. The truth files are opened afterwards, only to score it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import grid, match, network
from .paths import twin_of


def analyze(data: Path, log=print) -> dict:
    data = Path(data)
    out = data / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    table, parts = network.read(data / "network.parquet")
    net = match.Network(parts)
    segs = pd.read_parquet(data / "sim" / "segments.parquet")
    runs = pd.read_parquet(data / "sim" / "runs.parquet")
    years = sorted(runs["year"].unique().tolist())

    g = grid.grid_points(parts, [r for r in table["ROUTE"] if r.endswith(("-000", "-010"))])
    pairs = grid.pair_twins(g)
    log(f"grid: {len(g):,} cells; {len(pairs):,} twin cells paired")

    matched_all, cells_all = [], []
    network_runs = runs.loc[runs["kind"] == "network", ["run", "year"]]
    for year in years:
        s = segs[segs["run"].isin(network_runs.loc[network_runs["year"] == year, "run"])]
        m = match.match(s, net)
        matched_all.append(m)
        cells_all.append(grid.to_cells(m, pairs, year))
        log(f"{year}: {len(m):,} segments matched; {cells_all[-1].shape[0]:,} cell-slots")
    matched = pd.concat(matched_all, ignore_index=True)
    cells = pd.concat(cells_all, ignore_index=True)
    wide = grid.panel(cells, years)
    qa = grid.run_qa(matched)
    chg = grid.changes(cells)
    matched.to_parquet(out / "matched.parquet", index=False)
    cells.to_parquet(out / "cells.parquet", index=False)
    wide.to_parquet(out / "panel.parquet", index=False)
    qa.to_parquet(out / "run_qa.parquet", index=False)
    chg.to_parquet(out / "changes.parquet", index=False)
    two_plus = (wide["years_inv"] + wide["years_non"] >= 2)
    log(f"panel: {len(wide):,} cells, {two_plus.sum():,} with two or more surveyed years")

    scores = score(data, matched, cells, qa, chg, runs, pairs, log)
    summary = {"years": years, "grid_cells": len(g), "twin_pairs": len(pairs),
               "panel_cells": len(wide), "panel_cells_2plus": int(two_plus.sum()),
               "segments": len(matched), "scores": scores}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def score(data: Path, matched, cells, qa, chg, runs, pairs, log=print) -> dict:
    sim = Path(data) / "sim"
    tru = pd.read_parquet(sim / "truth_segments.parquet")
    treat = pd.read_parquet(sim / "treatments.parquet")
    d = matched.merge(tru, on=["run", "seq"], suffixes=("", "_true"))
    d = d[d["route_true"].notna()]
    driven = np.where(d["on_twin"], d["route_true"].map(twin_of), d["route_true"])
    right = (d["route"] == driven).to_numpy()
    err_ft = ((d["m"].astype(float) - d["m_route"]) * 5280).abs().to_numpy()   # on the route driven
    sign_true = np.where(d["direction"] == "inv", 1.0, -1.0)
    matching = {
        "route_exact": float(right.mean()),
        "route_up_to_carriageway": float((d["route"].str[:-3] == d["route_true"].str[:-3]).mean()),
        "unmatched": float(d["route"].isna().mean()),
        "direction_right": float((d["meas_sign"].to_numpy()[right] == sign_true[right]).mean()),
        "measure_error_ft": {f"p{q}": float(np.quantile(err_ft[right], q / 100)) for q in (50, 90, 99)},
        "interpolated": int((d["flag"] == "interpolated").sum()),
    }

    # cells: does each kept reading sit in the cell its truth says? (twins: the mainline cell abeam)
    c = cells.merge(d[["run", "m_true", "route_true", "direction"]].assign(m=d["m"].astype(float)),
                    on=["run", "m"], how="left")
    true_grid = np.round(c["m_true"] / grid.STEP_MI)
    cell_ok = (c["key_route"] == c["route_true"]) & (c["key_grid"] == true_grid)
    near_ok = (c["key_route"] == c["route_true"]) & ((c["key_grid"] - true_grid).abs() <= 1)
    twin = c["divided"].astype(bool) & c["route"].str.endswith("-010")
    cells_score = {"exact_cell": float(cell_ok.mean()), "within_one_cell": float(near_ok.mean()),
                   "twin_rows_exact": float(cell_ok[twin].mean()),
                   "twin_rows_within_one_cell": float(near_ok[twin].mean())}

    # odometer: GPS-implied scale vs the true scale, per run and per vehicle-year
    q = qa.merge(runs[["run", "dmi_scale", "dmi_vehicle_year", "vehicle", "year"]], on="run")
    q = q[q["segments"] >= 200]
    vy = q.groupby(["vehicle", "year"]).agg(true=("dmi_vehicle_year", "first"), gps=("dmi_scale_gps", "median"))
    dmi = {"runs": len(q), "run_abs_error_p50": float((q["dmi_scale_gps"] - q["dmi_scale"]).abs().median()),
           "run_abs_error_p95": float((q["dmi_scale_gps"] - q["dmi_scale"]).abs().quantile(0.95)),
           "vehicle_years": len(vy),
           "vehicle_year_error_mean": float((vy["gps"] - vy["true"]).mean()),
           "vehicle_year_abs_error_max": float((vy["gps"] - vy["true"]).abs().max()),
           "vehicle_years_flagged": vy.index[vy["gps"].abs() > 0.003].map(lambda t: f"{t[0]} {t[1]}").tolist(),
           "vehicle_years_miscalibrated": vy.index[vy["true"].abs() > 0.003].map(lambda t: f"{t[0]} {t[1]}").tolist(),
           "table": [{"vehicle": v, "year": int(y), "true": float(r["true"]), "gps": float(r["gps"])}
                     for (v, y), r in vy.iterrows()]}

    # change detection vs true treatments
    k0 = np.floor(treat["m0"] / grid.STEP_MI + 1e-9).astype(int)
    k1 = np.ceil(treat["m1"] / grid.STEP_MI - 1e-9).astype(int)
    treated = set()
    for r, a, b, y in zip(treat["route"], k0, k1, treat["year"]):
        treated.update((r, k, y) for k in range(a, b))
    ch = chg.dropna(subset=["delta_smooth"])

    def was_treated(row):
        return any((row.key_route, row.key_grid, y) in treated
                   for y in range(int(row.prev_year) + 1, int(row.year) + 1))

    truth_flag = np.array([was_treated(r) for r in ch.itertuples(index=False)])
    flagged = ch["flagged"].to_numpy()
    detect = {"cells_compared": int(len(ch)), "true_positive": int((flagged & truth_flag).sum()),
              "precision": float((flagged & truth_flag).sum() / max(flagged.sum(), 1)),
              "recall": float((flagged & truth_flag).sum() / max(truth_flag.sum(), 1))}
    result = {"matching": matching, "cells": cells_score, "dmi": dmi, "change_detection": detect}
    log(json.dumps(result, indent=1, default=float))
    return result
