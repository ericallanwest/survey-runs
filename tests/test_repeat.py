import numpy as np
import pytest

from survey_runs.repeat import SEG_FT, brsd, icc, pair_in_order, unhold

ROAD = np.cumsum(np.random.default_rng(3).normal(0, 3, 200))     # the road's own variation


def test_icc_counts_a_run_offset_only_for_absolute_agreement():
    x = np.column_stack([ROAD, ROAD + 2.0, ROAD - 1.0])            # same road, runs reading high/low
    icc2, icc3 = icc(x)
    assert icc3 == pytest.approx(1.0)                                 # perfectly consistent
    assert 0.9 < icc2 < 1.0                                           # but not in absolute agreement
    assert np.isnan(icc(x[:, :1])).all()                             # one run: nothing to compare


def test_brsd_is_the_mean_between_run_sd_of_block_means():
    x = np.column_stack([ROAD, ROAD + 1.0])
    assert brsd(x) == pytest.approx(1.0 / np.sqrt(2))                 # SD of (a, a + 1)
    x[:10, 1] += 3.0                                                  # one block disagrees more
    blocks = len(x) // 10
    assert brsd(x) == pytest.approx(((blocks - 1) * 1 + 4) / blocks / np.sqrt(2))


def test_pairing_follows_a_run_that_drifts_along_the_road():
    stations = np.column_stack([np.arange(100) * SEG_FT, np.zeros(100)])
    run = np.column_stack([5.0 + np.arange(110) * SEG_FT * 1.006, np.full(110, 3.0)])  # odometer 0.6% short
    rows = pair_in_order(run, stations, np.tile([1.0, 0.0], (100, 1)), start=0)
    paired = rows >= 0
    off = np.abs(run[rows[paired], 0] - stations[paired, 0])
    assert np.all(np.diff(rows[paired]) > 0)                         # in surveyed order
    assert off.max() < 0.55 * SEG_FT                                  # pairing by row would reach 21 ft
    assert paired.mean() > 0.95                                       # the odd station given up


def test_pairing_drops_a_station_rather_than_run_a_segment_out():
    # every segment of the run starts 18 ft past a station, 8.4 ft short of the next one.
    # Taking the nearest unused segment station by station would pair each 18 ft out.
    stations = np.column_stack([np.arange(30) * SEG_FT, np.zeros(30)])
    run = np.column_stack([np.arange(30) * SEG_FT + 18.0, np.zeros(30)])
    rows = pair_in_order(run, stations, np.tile([1.0, 0.0], (30, 1)), start=0)
    assert rows[0] == -1 and (rows[1:] == np.arange(29)).all()


def test_held_fixes_are_replaced_along_chainage():
    ch = np.arange(20) * SEG_FT
    p = np.column_stack([ch * 1.0, np.zeros(20)])
    held = p.copy()
    held[6:11] = held[5]                                              # receiver stuck for 5 fixes
    np.testing.assert_allclose(unhold(held, ch), p)
