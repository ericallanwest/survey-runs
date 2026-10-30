# Same road, different day

**[Project page](https://ericallanwest.github.io/survey-runs/)** ·
**[Repeatability lab](https://ericallanwest.github.io/survey-runs/lab/)** (runs in the browser)

Two questions every road-survey program has to answer, worked end to end:

1. **Year over year.** Has this stretch of road changed since last year's survey? Every year's runs
   have to land on one fixed grid before the years can be compared.
2. **Repeatability.** When the same road is driven several times, how closely do the runs agree?
   The runs have to be lined up position by position before they can be compared.

The road network is real: the Kentucky Transportation Cabinet's public All Roads linear referencing
system (LRS) for District 12, seven counties in eastern Kentucky (684 US and KY routes, 2,089 miles,
including 199 miles of divided highway). Everything measured on it is synthetic: a friction-like
index from 0 to 100, with pavement sections, wear, resurfacing and local texture, read by two
simulated survey vehicles over five years. Because the truth is known, every step of the analysis
is scored against it.

## What goes wrong, on purpose

The simulator builds in the trouble real survey data has:

* **Odometer (DMI) scale error:** a small error per vehicle per year, a little more per run, and one
  vehicle-year 0.5% short;
* **GPS error** that wanders slowly, and **held fixes**, where the receiver repeats one position for
  several seconds and then jumps;
* **lane offset**, **speed** (slowdowns shift the reading; under 15 mph it is invalid) and
  **season** (readings run high in spring and low in late summer);
* **coverage:** not every road is surveyed every year, some runs start late or stop early, and some
  are driven twice;
* **two vehicles**, each with its own small bias.

## Pipeline

| Step | Command | What it does |
|---|---|---|
| Network | `survey-runs network` | Fetches District 12 from KYTC's public ArcGIS service; anchors each feature's measures to its milepoints and turns the `-010` twin carriageways to the mainline's direction; writes a measured GeoParquet. |
| Simulate | `survey-runs simulate` | Builds the truth, chains routes into corridors across county lines, and drives five years of survey runs plus 180 repeat runs over six 3-mile reference sites. |
| Year over year | `survey-runs analyze` | Matches every 26.4-ft segment to the LRS (route, measure, direction), builds the 0.005-mile year-over-year panel, checks each odometer against GPS, flags likely resurfacing, and scores it all against the truth. |
| Repeatability | `survey-runs repeat` | Pairs the repeat runs four ways (odometer, GPS, LRS measure, and the truth as an oracle), then computes BRSD and ICC per site, year and basket of runs. |
| Page data | `survey-runs site` | Writes `docs/data/` for the project page: the change map, route profiles, scores and repeatability views. |

## Results

Seed 11: 2,430 network runs (3.4 million segments) and 180 repeat runs.

**Year over year.** Every number is checked against the simulator's truth.

| Check | Result |
|---|---|
| Segment matched to the route actually driven | 99.92% (0.06% unmatched) |
| Direction of travel right | 99.9996% |
| Measure error, p50 / p90 / p99 | 2.0 / 5.0 / 7.9 ft |
| Reading in its true 0.005-mi cell / within one cell | 90.6% / 99.98% |
| Twin-carriageway readings in their true cell / within one cell | 77.7% / 99.89% |
| Odometer check against GPS | finds the one miscalibrated vehicle-year (SV2 2023: −0.497% estimated, −0.500% true), no false flags; worst vehicle-year error 0.007% |
| Resurfacing flagged, precision / recall | 98.7% / 78.8% |

**Repeatability.** Six 3-mile sites, five years. Each site-year gives three baskets: each vehicle's
three runs, plus all six runs together. A pair is right where a run's segment lies within half a
segment of the station's.

| Pairing | Pairs right, one vehicle | Pairs right, both vehicles | Both vehicles in 2023 (SV2's odometer 0.5% short) |
|---|---|---|---|
| Odometer (chainage) | 80.6% | 64.8% | 23–51% right; BRSD 5–9% high, ICC(2,k) up to 0.038 low |
| GPS position | 88.3% | 87.6% | 84–94% right; ICC(2,k) within 0.001 of the oracle |
| LRS measure | 88.4% | 88.0% | 84–94% right; ICC(2,k) within 0.002 of the oracle |
| Oracle (true position) | 99.6% | 99.6% | |

Pairing by odometer works until two odometers disagree. Pairing by position doesn't drift, and
what it misses (each run's GPS error wanders a few feet along the road) barely moves BRSD or ICC.

## Run it

```sh
uv sync
uv run survey-runs network      # about a minute: pages through the public service
uv run survey-runs simulate     # about 20 s
uv run survey-runs analyze      # about 5 min: 3.4 million segments
uv run survey-runs repeat       # about 10 s
uv run survey-runs site         # docs/data for the page
uv run pytest
```

Outputs land in `data/` (not tracked): `network.parquet`, `sim/`, `analysis/`. `docs/data/` is
tracked, so the page is reproducible from the repository.

The lab is `notebooks/lab.py` (`uv run marimo edit notebooks/lab.py`). The Pages workflow exports
it to WebAssembly together with this package's wheel, which the notebook installs in the browser:

```sh
uv build --wheel -o notebooks/public
uv run marimo export html-wasm notebooks/lab.py -o docs/lab --mode run
```

## Code

| Module | |
|---|---|
| `network.py` | Fetch and assemble the measured LRS |
| `paths.py` | Corridors and drive paths; twins substituted for travel against the measure |
| `truth.py` | The synthetic friction-like index, per 0.005-mi cell, direction and year |
| `simulate.py` | The survey program and everything that goes wrong in it |
| `match.py` | Segment-to-LRS matching, with held GPS fixes bridged by chainage |
| `grid.py` | The year-over-year panel, odometer check and change detection |
| `analyze.py` | Runs the year-over-year analysis and scores it |
| `repeat.py` | Repeatability per site, year and basket, scored against an oracle |
| `pairing.py` | Run pairing, BRSD and ICC (NumPy, pandas and Shapely only, so the lab can use it) |
| `site.py` | The project page's data |
| `notebooks/lab.py` | The repeatability lab (marimo) |

## Data

Road geometry and milepoints: Kentucky Transportation Cabinet, All Roads LRS, public ArcGIS
FeatureServer
`https://services2.arcgis.com/CcI36Pduqd0OR4W9/ArcGIS/rest/services/KYTC_All_Roads/FeatureServer/0`,
fetched at run time (the results here: 2026-10-30). All measured values are simulated; no real
survey data is used. This project isn't affiliated with KYTC.

## License

MIT
