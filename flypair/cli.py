"""flypair CLI: `flypair run <scenario> [--connectome tiny] [--controls] [--out runs]`."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flypair")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a scenario YAML (+ optional controls) and write outputs")
    r.add_argument("scenario")
    r.add_argument("--cache", default="cache")
    r.add_argument("--out", default="runs")
    r.add_argument("--override-connectome", default=None, help="use this connectome for every fly (e.g. tiny)")
    r.add_argument("--duration-ms", type=float, default=None)
    r.add_argument("--controls", action="store_true", help="also run the scenario's controls")
    r.add_argument("--video", action="store_true")
    r.add_argument("--device", default="auto")
    g = sub.add_parser("groups", help="print the group-resolution report for a connectome")
    g.add_argument("connectome"); g.add_argument("--cache", default="cache")
    s = sub.add_parser("sanity", help="sugar GRN -> MN9 sanity check")
    s.add_argument("connectome"); s.add_argument("--cache", default="cache"); s.add_argument("--rate", type=float, default=100.0)
    w = sub.add_parser("web", help="local web UI (dashboard + 3D viewer) on http://localhost:8000")
    w.add_argument("--port", type=int, default=8000); w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--cache", default="cache"); w.add_argument("--out", default="runs")
    a = ap.parse_args(argv)
    if a.cmd == "web":
        from .web import serve
        serve(a.host, a.port, a.cache, a.out)
        return

    from .connectome import load_connectome
    if a.cmd == "groups":
        from .groups import resolve_for
        resolve_for(load_connectome(a.connectome, a.cache), strict=False)
        return
    if a.cmd == "sanity":
        from .sanity import sugar_mn9
        sugar_mn9(load_connectome(a.connectome, a.cache), rate_hz=a.rate)
        return
    from .scenario import load_scenario, run_scenario
    from .controls import run_controls
    from .metrics import coupling_report
    scn = load_scenario(a.scenario)
    if a.override_connectome:
        for f in scn["flies"]:
            if f.get("connectome"):
                f["connectome"] = a.override_connectome
    if a.duration_ms:
        scn["duration_ms"] = a.duration_ms
    conns = {n: load_connectome(n, a.cache) for n in {f["connectome"] for f in scn["flies"] if f.get("connectome")}}
    live = run_scenario(scn, conns, out_dir=a.out, device=a.device)
    if a.controls:
        runs = run_controls(scn, conns, live, out_dir=a.out, device=a.device)
        if len(scn["flies"]) >= 2:
            print(coupling_report(runs, scn["flies"][0]["name"], scn["flies"][1]["name"]))
    from .plots import rasters, rates, trajectories
    d = Path(a.out) / live.name
    try:
        rasters(live, out=d / "rasters.png")
    except ValueError as e:
        print(f"[plots] {e}")
    rates(live, out=d / "rates.png"); trajectories(live, out=d / "trajectories.png")
    if a.video:
        from .video import render
        print("[video]", render(live, d / "arena.mp4"))
    print(f"outputs in {d}")


if __name__ == "__main__":
    main()
