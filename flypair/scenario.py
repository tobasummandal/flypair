"""Scenario YAML schema validation and the closed-loop runner."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import SCENARIOS_DIR
from .brain import Brain
from .channels import DEFAULT_CHANNELS, ChannelBank
from .constants import SHIU
from .groups import GroupRegistry, resolve_for
from .inputs import BrainSource, InputSource, RecordedSource, ScriptedSource
from .record import Run
from .world import Fly, World

KNOWN_CONNECTOMES = {"tiny", "malecns", "flywire"}
KNOWN_CONTROLS = {"open_loop", "playback", "shuffled"}
KNOWN_SOURCES = {"brain", "playback", "scripted"}
TOP_KEYS = {"name", "description", "duration_ms", "world_dt_ms", "arena", "flies", "channels", "motor",
            "record", "controls", "on_unresolved", "device", "propagate"}
FLY_KEYS = {"name", "connectome", "sex", "seed", "pose", "silence", "activate", "noise_hz", "gain", "source",
            "size_mm", "timeline", "sensory_timeline"}


class ScenarioError(ValueError):
    pass


def _err(path, msg):
    raise ScenarioError(f"scenario {path}: {msg}")


def validate(doc: dict, where: str = "<dict>") -> dict:
    """Validate a scenario document; returns it with defaults filled in. Raises ScenarioError."""
    if not isinstance(doc, dict):
        _err(where, "top level must be a mapping")
    unknown = set(doc) - TOP_KEYS
    if unknown:
        _err(where, f"unknown top-level keys {sorted(unknown)}; allowed: {sorted(TOP_KEYS)}")
    d = dict(doc)
    d.setdefault("name", Path(where).stem)
    d.setdefault("duration_ms", 1000)
    d.setdefault("world_dt_ms", 20.0)
    d.setdefault("arena", {"radius_mm": 20.0})
    d.setdefault("channels", "default")
    d.setdefault("motor", {})
    d.setdefault("record", {})
    d.setdefault("controls", [])
    d.setdefault("on_unresolved", "skip")
    if not isinstance(d["duration_ms"], (int, float)) or d["duration_ms"] <= 0:
        _err(where, f"duration_ms must be a positive number, got {d['duration_ms']!r}")
    if not isinstance(d["world_dt_ms"], (int, float)) or d["world_dt_ms"] <= 0:
        _err(where, f"world_dt_ms must be a positive number, got {d['world_dt_ms']!r}")
    if abs(round(d["world_dt_ms"] / SHIU.dt) * SHIU.dt - d["world_dt_ms"]) > 1e-9:
        _err(where, f"world_dt_ms ({d['world_dt_ms']}) must be a multiple of the brain dt ({SHIU.dt} ms)")
    flies = d.get("flies")
    if not isinstance(flies, list) or not flies:
        _err(where, "'flies' must be a non-empty list")
    names = set()
    for i, f in enumerate(flies):
        if not isinstance(f, dict):
            _err(where, f"flies[{i}] must be a mapping")
        unk = set(f) - FLY_KEYS
        if unk:
            _err(where, f"flies[{i}]: unknown keys {sorted(unk)}; allowed: {sorted(FLY_KEYS)}")
        f.setdefault("name", f"fly{i}")
        if f["name"] in names:
            _err(where, f"duplicate fly name {f['name']!r}")
        names.add(f["name"])
        f.setdefault("source", "brain")
        if f["source"] not in KNOWN_SOURCES:
            _err(where, f"flies[{i}] ({f['name']}): source must be one of {sorted(KNOWN_SOURCES)}")
        if f["source"] == "brain" or "connectome" in f:
            if f.get("connectome") not in KNOWN_CONNECTOMES:
                _err(where, f"flies[{i}] ({f['name']}): connectome must be one of {sorted(KNOWN_CONNECTOMES)}, got {f.get('connectome')!r}")
        f.setdefault("sex", "male")
        if f["sex"] not in ("male", "female", "none"):
            _err(where, f"flies[{i}] ({f['name']}): sex must be male, female or none")
        f.setdefault("seed", i)
        f.setdefault("pose", {})
        if not isinstance(f["pose"], dict) or set(f["pose"]) - {"x", "y", "heading"}:
            _err(where, f"flies[{i}] ({f['name']}): pose must be a mapping with keys x, y, heading")
        f.setdefault("silence", [])
        if not isinstance(f["silence"], list):
            _err(where, f"flies[{i}] ({f['name']}): silence must be a list of group names")
        f.setdefault("activate", {})
        if not isinstance(f["activate"], dict) or any(not isinstance(v, (int, float)) or v < 0 for v in f["activate"].values()):
            _err(where, f"flies[{i}] ({f['name']}): activate must map group name -> rate Hz (>= 0)")
        f.setdefault("noise_hz", 0.0)
        f.setdefault("gain", 1.0)
        f.setdefault("size_mm", 2.5)
    ch = d["channels"]
    if ch == "default":
        d["channels"] = [dict(c) for c in DEFAULT_CHANNELS]
    elif isinstance(ch, dict):
        lst = [dict(c) for c in DEFAULT_CHANNELS] if ch.get("default", True) else []
        d["channels"] = lst + list(ch.get("extra", []))
    elif isinstance(ch, list):
        d["channels"] = ch
    else:
        _err(where, "channels must be 'default', a list of channel mappings, or {default: bool, extra: [...]}")
    for c in d["channels"]:
        from .channels import Channel
        try:
            Channel.from_dict(c)
        except ValueError as e:
            _err(where, str(e))
    for c in d["controls"]:
        if c not in KNOWN_CONTROLS:
            _err(where, f"unknown control {c!r}; allowed: {sorted(KNOWN_CONTROLS)}")
    rec = d["record"]
    if not isinstance(rec, dict) or set(rec) - {"groups", "rasters"}:
        _err(where, "record must be a mapping with optional keys groups, rasters")
    rec.setdefault("groups", "default")
    rec.setdefault("rasters", [])
    if d["on_unresolved"] not in ("skip", "fail"):
        _err(where, "on_unresolved must be 'skip' or 'fail'")
    return d


def load_scenario(path: Path | str) -> dict:
    p = Path(path)
    if not p.exists() and (SCENARIOS_DIR / f"{path}.yaml").exists():
        p = SCENARIOS_DIR / f"{path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"scenario file not found: {path}")
    try:
        doc = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ScenarioError(f"scenario {p}: invalid YAML: {e}") from e
    return validate(doc, str(p))


# ---------------------------------------------------------------- runner
@dataclass
class BrainSlot:
    brain: Brain
    flies: list = field(default_factory=list)       # Fly objects in batch order
    registry: GroupRegistry = None
    record_groups: list = field(default_factory=list)
    raster_groups: list = field(default_factory=list)


def _record_groups(reg: GroupRegistry, spec) -> list:
    if spec == "default":
        return [n for n, g in reg.groups.items() if g.ok and g.role in ("internal", "output")]
    return [g for g in spec if reg.has(g)]


def run_scenario(scn: dict, connectomes: dict, control: str | None = None, playback_from: Run | None = None,
                 device: str = "auto", propagate: str | None = None, progress: bool = True,
                 out_dir: Path | str | None = None, verbose: bool = True, progress_cb=None) -> Run:
    """Run one scenario. `connectomes` maps name -> Connectome (already built).
    control: None | 'open_loop' | 'playback' | 'shuffled'."""
    scn = validate(scn, scn.get("name", "<dict>"))
    t_wall = time.time()
    dt_world = float(scn["world_dt_ms"])
    steps_per_tick = int(round(dt_world / SHIU.dt))
    n_ticks = int(round(scn["duration_ms"] / dt_world))
    propagate = propagate or scn.get("propagate", "auto")
    device = scn.get("device", device)

    # connectomes (shuffled control swaps W)
    needed = {f["connectome"] for f in scn["flies"] if f.get("connectome")}
    conns = {}
    for name in needed:
        if name not in connectomes:
            raise ScenarioError(f"scenario needs connectome {name!r} but it was not provided (have {sorted(connectomes)})")
        c = connectomes[name]
        if control == "shuffled":
            from .controls import shuffle_connectome
            c = shuffle_connectome(c, seed=0)
        conns[name] = c
    registries = {name: resolve_for(c, strict=False, verbose=False) for name, c in conns.items()}
    for name, c in conns.items():
        registries[name].connectome_name = name    # shuffled connectome keeps its group file name

    # world + flies
    world = World(scn["arena"], scn["motor"], registries, dt_world)
    flies = []
    for fd in scn["flies"]:
        f = Fly(fd["name"], fd["sex"], fd.get("connectome", ""), size_mm=fd["size_mm"])
        f.x = float(fd["pose"].get("x", 0.0)); f.y = float(fd["pose"].get("y", 0.0)); f.heading = float(fd["pose"].get("heading", 0.0))
        src = fd["source"]
        if control == "playback" and fd is scn["flies"][0]:
            src = "playback"
        if src == "brain":
            f.source = BrainSource()
        elif src == "playback":
            if playback_from is None:
                raise ScenarioError("playback control/source needs `playback_from=<Run of the live scenario>`")
            f.source = RecordedSource(playback_from.fly_frames(f.name))
        else:
            f.source = ScriptedSource(fd.get("timeline"), fd.get("sensory_timeline"))
        flies.append(f)
        world.add_fly(f)

    # brains: one per connectome, batch = flies with a brain on it
    slots: dict[str, BrainSlot] = {}
    for f, fd in zip(flies, scn["flies"]):
        if not f.source.needs_brain:
            continue
        slots.setdefault(f.connectome, BrainSlot(None, [], registries[f.connectome])).flies.append(f)
    for cname, slot in slots.items():
        reg = slot.registry
        slot.record_groups = _record_groups(reg, scn["record"]["groups"])
        slot.raster_groups = [g for g in scn["record"]["rasters"] if reg.has(g)]
        missing = [g for g in scn["record"]["rasters"] if not reg.has(g)]
        if missing and verbose:
            warnings.warn(f"raster groups unresolved on {cname}: {missing}")
        rec_idx = np.concatenate([reg[g] for g in slot.raster_groups]) if slot.raster_groups else None
        seeds = [scn["flies"][flies.index(f)]["seed"] for f in slot.flies]
        brain = Brain(conns[cname], len(slot.flies), device=device, seeds=seeds, propagate=propagate, record_indices=rec_idx)
        slot.brain = brain
        for b, f in enumerate(slot.flies):
            fd = scn["flies"][flies.index(f)]
            f.brain_key = (cname, b)
            for g in fd["silence"]:
                if reg.has(g):
                    brain.set_silence(b, reg[g])
                else:
                    msg = f"silence group {g!r} unresolved on {cname} for fly {f.name}"
                    if scn["on_unresolved"] == "fail":
                        raise ScenarioError(msg)
                    warnings.warn(msg)
            brain.set_noise(b, float(fd["noise_hz"]))
            brain.set_gain(b, float(fd["gain"]))
        if verbose:
            print(f"[scenario] brain '{cname}' batch={len(slot.flies)} flies={[f.name for f in slot.flies]} "
                  f"device={brain.device} recording {len(slot.record_groups)} groups, rasters {slot.raster_groups}")

    bank = ChannelBank(scn["channels"], world, registries, on_unresolved=scn["on_unresolved"],
                       global_gain=0.0 if control == "open_loop" else 1.0)
    if verbose:
        print(bank.describe() if control != "open_loop" else "[scenario] open_loop control: all channels off")

    # group index tensors for fast rate readout
    group_index = {cname: {g: slot.registry[g] for g in slot.record_groups} for cname, slot in slots.items()}

    def apply_drive(f: Fly, drive: dict):
        cname, b = f.brain_key
        slot = slots[cname]; reg = slot.registry; brain = slot.brain
        fd = scn["flies"][flies.index(f)]
        brain.clear_rates(b)
        for g, hz in fd["activate"].items():
            if reg.has(g):
                brain.set_rates(b, reg[g], float(hz))
            elif scn["on_unresolved"] == "fail":
                raise ScenarioError(f"activate group {g!r} unresolved on {cname}")
        for g, hz in drive.items():
            if hz > 0 and reg.has(g):
                brain.set_rates(b, reg[g], float(hz))

    # initial drive: constant activations only
    for f in flies:
        if f.brain_key:
            apply_drive(f, {})
    for slot in slots.values():
        slot.brain.commit_rates()

    rows = []
    rates_by_fly = {f.name: {} for f in flies}
    drive_by_fly = {f.name: {} for f in flies}
    it = range(n_ticks)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc=f"{scn['name']}{'/' + control if control else ''}", unit="tick")
    window_s = dt_world / 1000.0
    for tick in it:
        if progress_cb is not None and (tick % 5 == 0 or tick == n_ticks - 1):
            progress_cb(tick + 1, n_ticks)
        # 1. brains run one window
        for cname, slot in slots.items():
            slot.brain.run(steps_per_tick)
            counts = slot.brain.take_counts()
            for b, f in enumerate(slot.flies):
                rates_by_fly[f.name] = {g: float(counts[b, idx].mean() / window_s) for g, idx in group_index[cname].items()}
        for f in flies:
            if isinstance(f.source, RecordedSource):
                rates_by_fly[f.name] = f.source.recorded_rates(tick)   # replayed, not simulated
        # 2. motor commands -> world
        cmds = {f.name: f.source.motor(f, rates_by_fly[f.name], world, tick) for f in flies}
        world.apply_all(cmds)
        # 3. record
        for f in flies:
            row = {"tick": tick, "t_ms": world.t_ms, "fly": f.name}
            row.update(world.snapshot(f))
            for g, r in rates_by_fly[f.name].items():
                row[f"rate_{g}"] = r
            for g, hz in drive_by_fly[f.name].items():
                row[f"drive_{g}"] = hz
            rows.append(row)
        # 4. sensory drive for the next window
        sens = bank.evaluate(rates_by_fly)
        for f in flies:
            extra = f.source.sensory(f, world, tick)
            for g, hz in extra.items():
                sens[f.name][g] = sens[f.name].get(g, 0.0) + hz
            drive_by_fly[f.name] = sens[f.name]
            if f.brain_key:
                apply_drive(f, sens[f.name])
        for slot in slots.values():
            slot.brain.commit_rates()
        world.advance()

    frames = pd.DataFrame(rows)
    # spikes for rasters
    sp_rows = []
    for cname, slot in slots.items():
        slot.brain.flush_record()
        t, b, n = slot.brain.record.arrays()
        if len(t):
            neuron_group = {}
            for g in slot.raster_groups:
                for i in slot.registry[g]:
                    neuron_group.setdefault(int(i), g)
            sp_rows.append(pd.DataFrame({"fly": [slot.flies[i].name for i in b], "t_ms": t * SHIU.dt,
                                         "neuron": n, "group": [neuron_group.get(int(i), "") for i in n]}))
    spikes = pd.concat(sp_rows, ignore_index=True) if sp_rows else pd.DataFrame(columns=["fly", "t_ms", "neuron", "group"])
    meta = {"name": scn["name"], "control": control, "duration_ms": scn["duration_ms"], "world_dt_ms": dt_world,
            "flies": [f.name for f in flies], "fly_specs": scn["flies"], "channels": scn["channels"],
            "motor": world.motor.cfg, "arena": scn["arena"], "wall_s": time.time() - t_wall,
            "connectomes": {n: c.meta for n, c in conns.items()},
            "record_groups": {n: s.record_groups for n, s in slots.items()},
            "raster_groups": scn["record"]["rasters"],
            "skipped_channels": bank.skipped, "lif": SHIU.__dict__}
    run = Run(scn["name"] + (f"__{control}" if control else ""), frames, spikes, meta)
    if out_dir:
        run.save(Path(out_dir) / run.name)
    if verbose:
        print(run.summary())
    return run
