"""Command line: ``survey-runs network|simulate|analyze|repeat|site``."""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="survey-runs")
    ap.add_argument("command", choices=["network", "simulate", "analyze", "repeat", "site"])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--docs", type=Path, default=Path("docs"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    from . import network
    if a.command == "network":
        network.build(a.data, log=log)
    if a.command == "simulate":
        from . import simulate, truth
        settings = simulate.Settings()
        table, parts = network.read(a.data / "network.parquet")
        tr = truth.generate(table, parts, settings.years, seed=a.seed)
        log(f"truth: {len(tr.routes):,} routes, {tr.values.shape[2]:,} cells, "
            f"{len(tr.sections):,} sections, {len(tr.treatments):,} treatments")
        runs, segs, trus, sites = simulate.simulate(table, parts, tr, settings, seed=a.seed, log=log)
        simulate.write(a.data / "sim", tr, runs, segs, trus, sites, settings)
        log(f"{len(runs):,} runs, {len(segs):,} segments -> {a.data / 'sim'}")
    if a.command == "analyze":
        from . import analyze
        analyze.analyze(a.data, log=log)
    if a.command == "repeat":
        from . import repeat
        repeat.analyze(a.data, log=log)
    if a.command == "site":
        from . import site
        site.build(a.data, a.docs, log=log)


if __name__ == "__main__":
    main()
