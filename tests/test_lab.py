"""The lab notebook's cells, run outside the browser at their default settings."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as V

import numpy as np
import pandas as pd
import pytest

from survey_runs.pairing import SEG_FT, brsd, icc, pair_in_order, tangents

pytest.importorskip("marimo")
LAB = Path(__file__).parents[1] / "notebooks" / "lab.py"


def test_lab_pairs_by_gps_better_than_by_odometer():
    spec = importlib.util.spec_from_file_location("lab", LAB)
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    knobs = dict(gps_ft=3, odo_gap=0.5, odo_run=0.05, site_mi=3, local_sd=1.5, noise_sd=1.5, n_runs=3, seed=11)
    _, sim = asyncio.run(lab.simulate.run(SEG_FT=SEG_FT, np=np, **{k: V(value=v) for k, v in knobs.items()}))
    _, out = asyncio.run(lab.score.run(SEG_FT=SEG_FT, brsd=brsd, icc=icc, np=np, pair_in_order=pair_in_order,
                                       pd=pd, tangents=tangents, **{k: sim[k] for k in
                                                                    ("road", "runs", "s_road", "site0", "site1")}))
    right = out["table"].set_index("Paired by")["Pairs right"]
    assert right["True position"] > 0.95
    assert right["GPS"] > right["Odometer"] + 0.2      # SV2's odometer 0.5% short over 3 mi
