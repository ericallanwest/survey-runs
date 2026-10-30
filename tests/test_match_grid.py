import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from survey_runs import grid, match
from survey_runs.network import EPSG

MILE = 5280.0
X0, Y0 = Transformer.from_crs(4326, EPSG, always_xy=True).transform(-82.5, 37.5)
MAIN = np.column_stack([X0 + np.linspace(0, 2 * MILE, 21), np.full(21, Y0), np.linspace(0, 2, 21)])
TWIN = np.column_stack([X0 + np.linspace(0.5 * MILE, 1.5 * MILE, 11), np.full(11, Y0 - 60),
                        np.linspace(0.5, 1.5, 11)])
PARTS = {"001-US-0001-000": [MAIN], "001-US-0001-010": [TWIN]}
TO_WGS = Transformer.from_crs(EPSG, 4326, always_xy=True)


def run_along(name, x_start, x_end, y, n_held=0, held_at=20, scale=0.0, seg_ft=26.4):
    """Segments of a run driven straight from x_start to x_end at constant y (EPSG:3089 feet).

    ``scale`` is the odometer error: chainage = ground x (1 + scale). ``n_held`` fixes after
    ``held_at`` repeat that boundary's position, as a receiver that stops updating does.
    """
    sign = np.sign(x_end - x_start)
    n = int(abs(x_end - x_start) * (1 + scale) // seg_ft)
    chain = np.arange(n + 1) * seg_ft
    x = x_start + sign * chain / (1 + scale)
    fx, fy = x.copy(), np.full(n + 1, float(y))
    fx[held_at + 1:held_at + 1 + n_held] = fx[held_at]
    lon, lat = TO_WGS.transform(fx, fy)
    return pd.DataFrame({"run": name, "seq": np.arange(n), "ch_start_ft": chain[:-1], "ch_end_ft": chain[1:],
                         "lon0": lon[:-1], "lat0": lat[:-1], "lon1": lon[1:], "lat1": lat[1:],
                         "x_mid": (x[:-1] + x[1:]) / 2})


def test_segments_land_on_the_route_and_direction_driven():
    net = match.Network(PARTS)
    east = run_along("SV1_230417_114200", X0 + 0.1 * MILE, X0 + 0.4 * MILE, Y0 - 6)      # lane right of center
    west = run_along("SV1_230417_120000", X0 + 1.4 * MILE, X0 + 0.6 * MILE, Y0 - 54)     # on the twin
    out = match.match(pd.concat([west, east], ignore_index=True), net)
    e, w = out[out["run"] == east["run"][0]], out[out["run"] == west["run"][0]]
    assert (e["route"] == "001-US-0001-000").all() and (e["meas_sign"] == 1).all()
    assert (w["route"] == "001-US-0001-010").all() and (w["meas_sign"] == -1).all()
    np.testing.assert_allclose(e["m"].astype(float), (e["x_mid"] - X0) / MILE, atol=1e-6)
    assert (out["flag"] == "ok").all()


def test_held_fixes_are_bridged_by_chainage():
    net = match.Network(PARTS)
    run = run_along("SV1_230417_114200", X0 + 0.1 * MILE, X0 + 0.4 * MILE, Y0 - 6, n_held=10)
    out = match.match(run, net)
    held = out.iloc[20:31]                                  # 9 still steps, then the step that jumps
    assert (held["flag"] == "interpolated").all()
    assert (held["route"] == "001-US-0001-000").all() and (held["meas_sign"] == 1).all()
    np.testing.assert_allclose(held["m"].astype(float), (held["x_mid"] - X0) / MILE, atol=1e-6)
    assert (out.drop(held.index)["flag"] == "ok").all()


def test_twin_cells_pair_one_to_one_with_the_nearest_mainline_cell():
    g = pd.DataFrame({"route": ["A-000"] * 3 + ["A-010"] * 3, "grid": [0, 1, 2, 0, 1, 2],
                      "x": [0.0, 26.4, 52.8, 0.0, 10.0, 26.4], "y": [0.0] * 3 + [-60.0] * 3})
    p = grid.pair_twins(g).sort_values("twin_grid")
    # twin cell 1 is nearest mainline cell 0, which twin cell 0 holds closer: it goes unpaired
    assert p[["twin_grid", "grid"]].values.tolist() == [[0, 0], [2, 1]]
    assert p["pair_ft"].tolist() == pytest.approx([60.0, 60.0])


def test_latest_pass_wins_then_nearest_center():
    pairs = pd.DataFrame({"twin": ["A-010"], "twin_grid": [10], "route": ["A-000"], "grid": [11]})
    matched = pd.DataFrame({
        "run": ["SV1_230417_114200", "SV2_230601_090000", "SV2_230601_090000", "SV1_230417_120000"],
        "route": ["A-000", "A-000", "A-000", "A-010"],
        "m": [0.0500, 0.0488, 0.0502, 0.0501],
        "meas_sign": [1.0, 1.0, 1.0, -1.0],
        "flag": ["ok", "ok", "interpolated", "ok"], "valid": True, "value": [40.0, 50.0, 51.0, 60.0]})
    cells = grid.to_cells(matched, pairs, 2023).set_index(["key_route", "key_grid", "slot"])
    assert cells.loc[("A-000", 10, "inv"), "value"] == 51.0        # June run, nearer the center
    assert cells.loc[("A-000", 11, "non"), "value"] == 60.0        # twin reading, keyed to its pair
    assert cells.loc[("A-000", 11, "non"), "divided"]
    assert len(cells) == 2


def test_change_needs_a_sustained_rise():
    k = np.arange(100)
    before = pd.DataFrame({"key_route": "A-000", "key_grid": k, "slot": "inv", "year": 2021, "value": 50.0})
    after = before.assign(year=2022, value=50.0 + np.where((k >= 40) & (k < 80), 10.0, 0.0)
                          + np.where(k == 10, 30.0, 0.0))
    chg = grid.changes(pd.concat([before, after], ignore_index=True))
    assert chg.loc[chg["flagged"], "key_grid"].tolist() == list(range(40, 80))   # not the one-cell spike
    assert (chg["prev_year"] == 2021).all()


def test_run_qa_reads_the_odometer_scale_from_gps():
    run = run_along("SV1_230417_114200", X0, X0 + 2 * MILE, Y0, scale=0.004)
    run["route"] = "001-US-0001-000"
    run["m"] = (run["x_mid"] - X0) / MILE
    qa = grid.run_qa(run).iloc[0]
    assert qa["dmi_scale_gps"] == pytest.approx(0.004, abs=1e-6)
    assert qa["skipped"] == 0 and qa["hit_twice"] == 1       # 0.4% long: one cell hit twice in 2 mi
