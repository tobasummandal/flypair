"""Local web UI: `flypair web` -> http://localhost:8000

Dashboard (web/index.html) to launch scenarios, watch progress, and view results:
3D viewer (viewer/index.html), figures, coupling metrics, run table. Runs execute in a
background thread; the API is a thin layer over scenario.run_scenario / controls / plots.
Only connectomes already in the cache (or `tiny`) can be used: this app never builds one.
"""
from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import PKG_ROOT, SCENARIOS_DIR
from .connectome import Connectome, load_connectome
from .record import Run
from .scenario import ScenarioError, load_scenario, run_scenario

app = FastAPI(title="flypair")
STATE = {"cache": Path("cache"), "out": Path("runs"), "jobs": {}, "conns": {}}


def get_conn(name: str) -> Connectome:
    if name not in STATE["conns"]:
        if name != "tiny" and not Connectome.cached(STATE["cache"], name):
            raise HTTPException(400, f"connectome {name!r} is not in the cache ({STATE['cache']}); build it on Colab "
                                     f"and copy cache/{name}/ here, or use 'tiny'")
        STATE["conns"][name] = load_connectome(name, STATE["cache"])
    return STATE["conns"][name]


SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_path(*parts: str) -> Path:
    """Path under the runs directory; 404 on unsafe names, traversal or symlink escape."""
    base = STATE["out"].resolve()
    if not all(SAFE.match(x) for x in parts):
        raise HTTPException(404, "not found")
    p = (base.joinpath(*parts)).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        raise HTTPException(404, "not found")
    return p


class RunRequest(BaseModel):
    scenario: str                       # name of a shipped scenario (scenarios/<name>.yaml), validated below
    connectome: str | None = "tiny"     # override for every fly; None keeps the YAML's
    duration_ms: float | None = None
    controls: list[str] = []
    video3d: bool = False
    camera: str = "orbit"


@app.get("/", response_class=HTMLResponse)
def index():
    return (PKG_ROOT / "web" / "index.html").read_text()


@app.get("/viewer", response_class=HTMLResponse)
def viewer():
    return (PKG_ROOT / "viewer" / "index.html").read_text()


@app.get("/api/scenarios")
def scenarios():
    out = []
    for p in sorted(SCENARIOS_DIR.glob("*.yaml")):
        try:
            d = load_scenario(p)
            out.append({"name": d["name"], "description": d.get("description", ""), "duration_ms": d["duration_ms"],
                        "flies": [{"name": f["name"], "connectome": f.get("connectome"), "sex": f["sex"]} for f in d["flies"]],
                        "controls": d["controls"], "yaml": p.read_text()})
        except ScenarioError as e:
            out.append({"name": p.stem, "error": str(e)})
    return out


@app.get("/api/connectomes")
def connectomes():
    return {n: (n == "tiny" or Connectome.cached(STATE["cache"], n)) for n in ("tiny", "malecns", "flywire")}


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(STATE["out"].glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = json.loads(d.read_text())
        out.append({"name": d.parent.name, "scenario": m.get("name"), "control": m.get("control"), "duration_ms": m.get("duration_ms"),
                    "flies": m.get("flies"), "wall_s": m.get("wall_s"), "mtime": d.stat().st_mtime,
                    "has_3d": (d.parent / "arena3d.mp4").exists(), "has_2d": (d.parent / "arena.mp4").exists()})
    return out


@app.get("/api/runs/{name}/summary", response_class=PlainTextResponse)
def run_summary(name: str):
    d = safe_path(name)
    if not (d / "meta.json").is_file():
        raise HTTPException(404, name)
    return Run.load(d).summary()


@app.get("/api/runs/{name}/{file}")
def run_file(name: str, file: str):
    p = safe_path(name, file)
    if not p.is_file():
        raise HTTPException(404, f"{name}/{file} not found")
    return FileResponse(p)


@app.get("/api/metrics/{name}", response_class=PlainTextResponse)
def metrics(name: str):
    """Coupling report for a live run and whatever controls exist next to it."""
    from .metrics import coupling_report
    base = safe_path(name)
    if not (base / "meta.json").is_file():
        raise HTTPException(404, name)
    live = Run.load(base)
    rs = {"live": live}
    for ctl in ("open_loop", "playback", "shuffled"):
        d = safe_path(f"{name}__{ctl}")
        if (d / "meta.json").is_file():
            rs[ctl] = Run.load(d)
    if len(live.flies) < 2:
        return "coupling metric needs at least two flies"
    return coupling_report(rs, live.flies[0], live.flies[1], groups=("p1", "pip10", "dna02", "dnp01"))


def _job(job: dict, req: RunRequest):
    try:
        if not SAFE.match(req.scenario) or not (SCENARIOS_DIR / f"{req.scenario}.yaml").is_file():
            raise ScenarioError(f"unknown scenario {req.scenario!r}")
        scn = load_scenario(SCENARIOS_DIR / f"{req.scenario}.yaml")
        if req.connectome:
            for f in scn["flies"]:
                if f.get("connectome"):
                    f["connectome"] = req.connectome
        if req.duration_ms:
            scn["duration_ms"] = req.duration_ms
        conns = {n: get_conn(n) for n in {f["connectome"] for f in scn["flies"] if f.get("connectome")}}
        job["stage"] = "live"

        def cb(t, n):
            job["tick"], job["n_ticks"] = t, n

        live = run_scenario(scn, conns, out_dir=STATE["out"], progress=False, verbose=False, progress_cb=cb)
        job["run"] = live.name
        from .plots import rasters, rates, trajectories
        d = STATE["out"] / live.name
        try:
            rasters(live, out=d / "rasters.png")
        except ValueError:
            pass
        rates(live, out=d / "rates.png"); trajectories(live, out=d / "trajectories.png")
        import matplotlib.pyplot as plt; plt.close("all")
        for ctl in req.controls:
            job["stage"] = ctl
            run_scenario(scn, conns, control=ctl, playback_from=live if ctl == "playback" else None,
                         out_dir=STATE["out"], progress=False, verbose=False, progress_cb=cb)
        if req.video3d:
            job["stage"] = "3d video"
            from .video3d import render3d
            render3d(live, d / "arena3d.mp4", camera=req.camera, verbose=False)
        job["stage"] = "done"; job["done"] = True
    except Exception as e:  # noqa: BLE001
        job["error"] = f"{e}\n{traceback.format_exc()}"; job["done"] = True; job["stage"] = "error"


@app.post("/api/run")
def start_run(req: RunRequest):
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "req": req.model_dump(), "tick": 0, "n_ticks": 1, "stage": "starting", "done": False, "error": None,
           "run": None, "started": time.time()}
    STATE["jobs"][jid] = job
    threading.Thread(target=_job, args=(job, req), daemon=True).start()
    return job


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    j = STATE["jobs"].get(jid)
    if not j:
        raise HTTPException(404, jid)
    return JSONResponse({**j, "elapsed_s": time.time() - j["started"]})


def serve(host: str = "127.0.0.1", port: int = 8000, cache: str = "cache", out: str = "runs"):
    import uvicorn
    STATE["cache"], STATE["out"] = Path(cache), Path(out)
    STATE["out"].mkdir(parents=True, exist_ok=True)
    print(f"flypair web UI -> http://{host}:{port}   (cache {cache}, runs {out})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
