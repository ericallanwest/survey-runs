import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Repeatability lab")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Repeatability lab

    Two survey vehicles drive one reference site a few times each on the same day. Before their
    readings can be compared, every run's segments have to be paired with the other runs' at the
    same place on the road. Change how the vehicles and their instruments behave, and see which
    way of pairing holds up, and what it does to the repeatability numbers.

    The pairing and the metrics are the same code the
    [project page](https://ericallanwest.github.io/survey-runs/) runs on five years of simulated
    surveys: `survey_runs.pairing`, installed into this page.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _(mo):
    import sys

    if "pyodide" in sys.modules:   # in the browser: install this project's own wheel
        import micropip

        await micropip.install(
            str(mo.notebook_location() / "public" / "survey_runs-0.1.0-py3-none-any.whl"), deps=False)

    import altair as alt
    import numpy as np
    import pandas as pd
    import shapely  # noqa: F401  (used by survey_runs.pairing)

    from survey_runs.pairing import SEG_FT, brsd, icc, pair_in_order, tangents
    return SEG_FT, alt, brsd, icc, np, pair_in_order, pd, tangents


@app.cell
def _(mo):
    gps_ft = mo.ui.slider(0, 10, step=0.5, value=3, show_value=True, label="GPS error (ft)")
    odo_gap = mo.ui.slider(0, 1, step=0.05, value=0.5, show_value=True, label="SV2's odometer short by (%)")
    odo_run = mo.ui.slider(0, 0.2, step=0.01, value=0.05, show_value=True, label="Odometer wander (%)")
    site_mi = mo.ui.slider(0.5, 5, step=0.5, value=3, show_value=True, label="Site length (mi)")
    local_sd = mo.ui.slider(0, 3, step=0.25, value=1.5, show_value=True, label="Local features (SD)")
    noise_sd = mo.ui.slider(0, 3, step=0.25, value=1.5, show_value=True, label="Reading noise (SD)")
    n_runs = mo.ui.slider(2, 5, step=1, value=3, show_value=True, label="Runs per vehicle")
    seed = mo.ui.number(start=1, stop=9999, value=11, label="Random seed")
    mo.vstack([
        mo.hstack([mo.vstack([gps_ft, odo_gap, odo_run, site_mi], align="end"),
                   mo.vstack([local_sd, noise_sd, n_runs, seed], align="end")], justify="start", gap=3),
        mo.md("GPS error is the SD of each axis, wandering slowly along the road. Odometer wander "
              "is each run's own SD around its vehicle's scale. Local features are each 26.4-ft cell's "
              "own reading, the same for every run that reads it."),
    ])
    return gps_ft, local_sd, n_runs, noise_sd, odo_gap, odo_run, seed, site_mi


@app.cell
def simulate(SEG_FT, gps_ft, local_sd, n_runs, noise_sd, np, odo_gap, odo_run, seed, site_mi):
    rng = np.random.default_rng(seed.value)
    LEAD = 1320.0                              # each run starts a quarter mile before the site
    site_ft = site_mi.value * 5280
    total = site_ft + 2 * LEAD + 600

    # a winding road, its centerline every 5 ft
    STEP = 5.0
    s_road = np.arange(0, total + STEP, STEP)
    bend = np.convolve(rng.normal(0, 1, s_road.size), np.ones(120) / 120, mode="same") * 0.006
    heading = np.cumsum(bend) * STEP
    road = np.column_stack([np.cumsum(np.cos(heading)), np.cumsum(np.sin(heading))]) * STEP

    def ar1(n, corr_cells, sd):
        """Correlated wander: AR(1) with the given correlation length (in steps)."""
        phi = np.exp(-1.0 / corr_cells)
        e = rng.normal(0, sd * np.sqrt(1 - phi * phi), n)
        out = np.empty(n)
        out[0] = rng.normal(0, sd)
        for i in range(1, n):
            out[i] = phi * out[i - 1] + e[i]
        return out

    # the truth on 26.4-ft cells: pavement sections, wander along the road, each cell's own feature
    n_cells = int(total // SEG_FT) + 1
    cuts = np.cumsum(rng.exponential(5280 / SEG_FT, 20)).astype(int)
    level = rng.normal(52, 7, len(cuts) + 1)[np.searchsorted(cuts, np.arange(n_cells), side="right")]
    truth = level + ar1(n_cells, 200 / SEG_FT, 3.0) + rng.normal(0, local_sd.value, n_cells)

    site0, site1 = LEAD + 300, LEAD + 300 + site_ft
    runs = []
    for vehicle, scale, bias in (("SV1", 0.0, 0.8), ("SV2", -odo_gap.value / 100, -0.6)):
        for k in range(n_runs.value):
            dmi = scale + rng.normal(0, odo_run.value / 100)
            s0 = 300 + rng.uniform(-60, 60)              # an operator starting at a landmark
            length = site_ft + 2 * LEAD + rng.uniform(-60, 60)
            n = int(length * (1 + dmi) // SEG_FT)
            ground = s0 + np.arange(n + 1) * SEG_FT / (1 + dmi)   # odometer counts 26.4 ft, the road gives this
            xy = np.column_stack([np.interp(ground, s_road, road[:, 0]), np.interp(ground, s_road, road[:, 1])])
            err = np.column_stack([ar1(n + 1, 1300 / SEG_FT, gps_ft.value), ar1(n + 1, 1300 / SEG_FT, gps_ft.value)])
            mid = (ground[:-1] + ground[1:]) / 2
            value = truth[(mid // SEG_FT).astype(int)] + bias + rng.normal(0, noise_sd.value, n)
            runs.append({"run": f"{vehicle} run {k + 1}", "vehicle": vehicle, "dmi": dmi,
                         "gps": (xy + err)[:-1], "s_true": mid, "value": value})
    return road, runs, s_road, site0, site1


@app.cell
def score(SEG_FT, brsd, icc, np, pair_in_order, pd, road, runs, s_road, site0, site1, tangents):
    def along(p):
        """Along-road position of GPS points: the nearest point of the known centerline (an LRS)."""
        out = np.empty(len(p))
        for i in range(0, len(p), 200):
            d = ((p[i:i + 200, None, :] - road[None, :, :]) ** 2).sum(axis=2)
            out[i:i + 200] = s_road[d.argmin(axis=1)]
        return out

    for r in runs:
        r["lrs"] = along(r["gps"])
    ref = int(np.argmax([len(r["value"]) for r in runs]))
    st = np.flatnonzero((runs[ref]["lrs"] >= site0) & (runs[ref]["lrs"] < site1))

    def pairing(method):
        if method == "GPS":
            P, S, T = [r["gps"] for r in runs], runs[ref]["gps"][st], tangents(runs[ref]["gps"])[st]
        else:
            key = "s_true" if method == "True position" else "lrs"
            P = [r[key][:, None] for r in runs]
            S, T = P[ref][st], np.ones((len(st), 1))
        rows = []
        for i, r in enumerate(runs):
            start = int(np.argmin(np.abs(r["lrs"] - runs[ref]["lrs"][st[0]])))
            if i == ref:
                rows.append(st)
            elif method == "Odometer":
                k = start + np.arange(len(st))
                rows.append(np.where(k < len(r["value"]), k, -1))
            else:
                rows.append(pair_in_order(P[i], S, T, start))
        return np.array(rows)

    results, offsets = [], []
    for method in ("Odometer", "GPS", "LRS measure", "True position"):
        rows = pairing(method)
        keep = (rows >= 0).all(axis=0)
        rows = rows[:, keep]
        x = np.column_stack([r["value"][k] for r, k in zip(runs, rows)])
        s = np.column_stack([r["s_true"][k] for r, k in zip(runs, rows)])
        off = np.abs(np.delete(s, ref, axis=1) - s[:, [ref]])
        icc2, icc3 = icc(x)
        results.append({"Paired by": method, "Pairs right": (off <= SEG_FT / 2).mean(),
                        "BRSD": brsd(x), "ICC(2,k)": icc2, "ICC(3,k)": icc3, "Positions": len(x)})
        at = (s[:, ref] - site0) / 5280
        offsets.append(pd.DataFrame({"method": method, "mi": np.floor(at / 0.05) * 0.05, "ft": off.max(axis=1)})
                       .groupby(["method", "mi"], as_index=False)["ft"].median())
    table = pd.DataFrame(results)
    offset = pd.concat(offsets, ignore_index=True)
    return offset, table


@app.cell
def _(alt, mo, offset):
    colors = alt.Scale(domain=["GPS", "Odometer", "LRS measure", "True position"],
                       range=["#2a78d6", "#eb6834", "#1baf7a", "#898781"])
    lines = alt.Chart(offset).mark_line(strokeWidth=2).encode(
        x=alt.X("mi:Q", title="miles along the site"),
        y=alt.Y("ft:Q", title="feet off"),
        color=alt.Color("method:N", scale=colors, title=None, legend=alt.Legend(orient="top")),
        strokeDash=alt.condition(alt.datum.method == "True position", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=["method", alt.Tooltip("mi:Q", format=".2f"), alt.Tooltip("ft:Q", format=".0f")],
    )
    half = alt.Chart(alt.Data(values=[{"y": 13.2}])).mark_rule(strokeDash=[3, 3], color="#c3c2b7").encode(y="y:Q")
    mo.vstack([
        mo.md("**How far off each pairing is, along the site:** the distance on the road between each "
              "station and the worst-paired run's segment, by the truth. The dashed line is half a segment."),
        mo.ui.altair_chart((half + lines).properties(width="container", height=260)),
    ])
    return


@app.cell
def _(mo, table):
    three = "{:.3f}".format
    mo.vstack([
        mo.md("**Repeatability, both vehicles' runs together.** A pair is right when the run's segment "
              "lies within half a segment (13.2 ft) of the station's. BRSD: the mean, over 264-ft blocks, of "
              "the between-run SD of the runs' block means. ICC(2,k) counts a vehicle that reads high "
              "throughout against agreement; ICC(3,k) does not."),
        mo.ui.table(table, selection=None, show_column_summaries=False, show_data_types=False,
                    show_download=False, show_search=False, pagination=False,
                    format_mapping={"Pairs right": "{:.1%}".format, "BRSD": three,
                                    "ICC(2,k)": three, "ICC(3,k)": three}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Things to try** (the figures are averages over ten seeds; one seed wanders around them):

    * Set SV2's odometer difference to 0. Odometer pairing nearly catches up with GPS, about 89% of
      pairs right against 92%.
    * Turn the GPS error up to 8 ft. GPS pairing slips to about two-thirds right, and odometer
      pairing slips too: its cut at the site's start is placed by GPS.
    * Set the local features to 0. The pairing gets no better, but a misaligned pair now costs
      ICC(2,k) about half as much, because neighboring cells read more alike.
    """)
    return


if __name__ == "__main__":
    app.run()
