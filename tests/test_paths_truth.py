import numpy as np
import pandas as pd
import pytest

from survey_runs.paths import cut, drive_path
from survey_runs.simulate import season
from survey_runs.truth import CELL_MI, generate

MILE = 5280.0
MAIN = np.column_stack([np.linspace(0, 2 * MILE, 21), np.zeros(21), np.linspace(0, 2, 21)])
TWIN = np.column_stack([np.linspace(0.5 * MILE, 1.5 * MILE, 11), np.full(11, -60.0), np.linspace(0.5, 1.5, 11)])
PARTS = {"001-US-0001-000": [MAIN], "001-US-0001-010": [TWIN]}


def test_cut_interpolates_exact_ends():
    piece = cut(MAIN, 0.25, 0.75)
    assert piece[0, 2] == pytest.approx(0.25) and piece[-1, 2] == pytest.approx(0.75)
    assert piece[0, 0] == pytest.approx(0.25 * MILE) and piece[-1, 0] == pytest.approx(0.75 * MILE)


def test_inventory_direction_follows_the_mainline_with_measure():
    path = drive_path("US-1#1", [("001-US-0001-000", 0)], PARTS, "inv")
    _, _, base, m, twin, _ = path.sample(np.array([0.0, MILE, 2 * MILE]))
    np.testing.assert_allclose(m, [0, 1, 2])
    assert not twin.any() and set(base) == {"001-US-0001-000"}


def test_other_direction_drives_the_twin_where_it_exists():
    path = drive_path("US-1#1", [("001-US-0001-000", 0)], PARTS, "non")
    assert [g.route[-3:] for g in path.legs] == ["000", "010", "000"]
    s = np.array([0.25 * MILE, path.length / 2, path.length - 0.25 * MILE])
    xy, _, base, m, twin, m_route = path.sample(s)
    assert twin.tolist() == [False, True, False]
    assert set(base) == {"001-US-0001-000"}            # truth stays on the mainline's axis
    assert m[0] > m[1] > m[2]                            # travelling against the measure
    assert xy[1, 1] == pytest.approx(-60)                # on the twin's geometry
    assert m_route[1] == pytest.approx(m[1])             # twin measured abeam its mainline here


def test_truth_lookup_is_by_mainline_cell():
    table = pd.DataFrame({"ROUTE": ["001-US-0001-000", "001-US-0001-010"]})
    tr = generate(table, PARTS, (2021, 2022), seed=1)
    vals = tr.at(np.array(["001-US-0001-000"] * 3, dtype=object), "inv", 2021, np.array([0.0, 1.0, 9.0]))
    assert np.isfinite(vals[:2]).all() and np.isnan(vals[2])     # 9 mi is off the route
    assert tr.values.shape == (2, 2, int(round(2 / CELL_MI)))
    assert 0 < np.nanmin(tr.values) and np.nanmax(tr.values) < 100


def test_season_is_high_in_spring_and_low_in_late_summer():
    assert season(100, 3) == pytest.approx(3)
    assert season(250, 3) == pytest.approx(-3)
