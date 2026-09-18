"""2D arena, fly bodies, emitters and the hand-designed motor decoders.

Everything in `MotorDecoder` is a modelling choice, not biology. The full list
(also in README):
  speed   = gain_fwd * rate(forward_dn)                    [mm/s], clipped to max_speed
  turn    = gain_turn * (rate(dna02_L) - rate(dna02_R))   [deg/s], positive = left turn
  reverse : if rate(mdn) > mdn_threshold: speed = -gain_rev * rate(mdn)
  jump    : if rate(dnp01) > dnp01_threshold (and not within jump_refractory): jump away
            from the nearest fly by jump_mm and flash
  song    = clip(rate(song_group) / song_ref_hz, 0, 1)   song_group = pip10, else wing_mn
All rates are per-neuron mean Hz over the world tick, EMA-smoothed with `smoothing_ms`.
A group missing from the connectome disables that pathway (warned once).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MOTOR = {
    "forward": {"group": "forward_dn", "gain": 0.25, "max_speed": 25.0},      # mm/s per Hz
    "turn": {"left": "dna02_L", "right": "dna02_R", "gain": 8.0, "max_turn": 500.0},  # deg/s per Hz
    "reverse": {"group": "mdn", "threshold": 20.0, "gain": 0.2},
    "escape": {"group": "dnp01", "threshold": 30.0, "jump_mm": 6.0, "refractory_ms": 300.0},
    "song": {"group": "pip10", "fallback_group": "wing_mn", "ref_hz": 40.0},
    "smoothing_ms": 60.0,
}

PHEROMONE_PROFILES = {           # emitter values by sex (hand-designed, binary)
    "male": {"pheromone_male": 1.0, "pheromone_female": 0.0},
    "female": {"pheromone_male": 0.0, "pheromone_female": 1.0},
    "none": {"pheromone_male": 0.0, "pheromone_female": 0.0},
}


@dataclass
class Fly:
    name: str
    sex: str
    connectome: str
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0            # degrees, 0 = +x, counter-clockwise positive
    size_mm: float = 2.5
    speed: float = 0.0
    turn: float = 0.0
    song: float = 0.0
    jump_flag: bool = False
    last_jump_ms: float = -1e9
    emitters: dict = field(default_factory=dict)
    smoothed: dict = field(default_factory=dict)
    brain_key: tuple | None = None   # (connectome name, batch index) or None if no brain
    source: object = None

    def __post_init__(self):
        prof = PHEROMONE_PROFILES.get(self.sex, PHEROMONE_PROFILES["none"])
        self.emitters = {"song_intensity": 0.0, "size": self.size_mm, "speed": 0.0, **prof}

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def heading_rad(self):
        return math.radians(self.heading)


class MotorDecoder:
    def __init__(self, config: dict | None = None, registries: dict | None = None):
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULT_MOTOR.items()}
        for k, v in (config or {}).items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
        self.cfg = cfg
        self.registries = registries or {}
        self._warned = set()

    def _has(self, fly, group):
        reg = self.registries.get(fly.connectome)
        ok = reg is not None and reg.has(group)
        if not ok and (fly.connectome, group) not in self._warned:
            warnings.warn(f"[motor] group {group!r} unresolved on {fly.connectome}: pathway disabled for {fly.name}")
            self._warned.add((fly.connectome, group))
        return ok

    def _smooth(self, fly, key, value, dt_ms):
        tau = float(self.cfg["smoothing_ms"])
        a = 1.0 if tau <= 0 else 1.0 - math.exp(-dt_ms / tau)
        prev = fly.smoothed.get(key, value)
        out = prev + a * (value - prev)
        fly.smoothed[key] = out
        return out

    def decode(self, fly, rates: dict, dt_ms: float):
        from .inputs import MotorCommand
        c = self.cfg
        cmd = MotorCommand()
        f = c["forward"]
        if self._has(fly, f["group"]):
            r = self._smooth(fly, "fwd", rates.get(f["group"], 0.0), dt_ms)
            cmd.speed = min(f["gain"] * r, f["max_speed"])
        t = c["turn"]
        if self._has(fly, t["left"]) and self._has(fly, t["right"]):
            rl = self._smooth(fly, "turnL", rates.get(t["left"], 0.0), dt_ms)
            rr = self._smooth(fly, "turnR", rates.get(t["right"], 0.0), dt_ms)
            cmd.turn = float(np.clip(t["gain"] * (rl - rr), -t["max_turn"], t["max_turn"]))
        rv = c["reverse"]
        if self._has(fly, rv["group"]):
            r = self._smooth(fly, "rev", rates.get(rv["group"], 0.0), dt_ms)
            if r > rv["threshold"]:
                cmd.speed = -rv["gain"] * r
        e = c["escape"]
        if self._has(fly, e["group"]):
            r = rates.get(e["group"], 0.0)      # not smoothed: bursts matter
            cmd.jump = r > e["threshold"]
        s = c["song"]
        sg = s["group"] if self._has(fly, s["group"]) else (s.get("fallback_group") if s.get("fallback_group") and self._has(fly, s["fallback_group"]) else None)
        if sg is not None:
            r = self._smooth(fly, "song", rates.get(sg, 0.0), dt_ms)
            cmd.song = float(np.clip(r / s["ref_hz"], 0.0, 1.0))
        return cmd


class World:
    def __init__(self, arena: dict | None = None, motor: dict | None = None, registries=None, dt_ms: float = 20.0):
        arena = arena or {}
        self.radius = float(arena.get("radius_mm", 20.0))
        self.dt_ms = float(dt_ms)
        self.flies: list[Fly] = []
        self.motor = MotorDecoder(motor, registries)
        self.t_ms = 0.0
        self.tick = 0

    def add_fly(self, fly: Fly):
        self.flies.append(fly)
        return fly

    # ---- geometry helpers ------------------------------------------------
    @staticmethod
    def wrap_deg(a):
        return (a + 180.0) % 360.0 - 180.0

    def distance(self, a: Fly, b: Fly) -> float:
        return float(math.hypot(a.x - b.x, a.y - b.y))

    def bearing(self, observer: Fly, target: Fly) -> float:
        """Angle (deg) of target relative to observer's heading; positive = target on observer's LEFT."""
        ang = math.degrees(math.atan2(target.y - observer.y, target.x - observer.x))
        return self.wrap_deg(ang - observer.heading)

    def angular_size(self, observer: Fly, target: Fly) -> float:
        d = max(self.distance(observer, target), 1e-3)
        return math.degrees(2 * math.atan2(target.size_mm / 2, d))

    def nearest_at_tick_start(self, fly: Fly):
        """Nearest other fly using positions frozen at the start of this tick (order-independent)."""
        snap = getattr(self, "_tick_pos", None) or {f.name: (f.x, f.y) for f in self.flies}
        me = snap[fly.name]
        best, bd = None, None
        for o in self.flies:
            if o is fly:
                continue
            ox, oy = snap[o.name]
            d = math.hypot(me[0] - ox, me[1] - oy)
            if bd is None or d < bd:
                best, bd = (ox, oy), d
        return best

    def apply_all(self, cmds: dict):
        """Apply one command per fly; jump directions use the poses frozen at tick start."""
        self._tick_pos = {f.name: (f.x, f.y) for f in self.flies}
        for f in self.flies:
            self.apply(f, cmds[f.name])
        self._tick_pos = None

    # ---- integrate one tick ----------------------------------------------
    def apply(self, fly: Fly, cmd):
        dt = self.dt_ms / 1000.0
        fly.jump_flag = False
        if cmd.pose_override is not None:
            fly.x, fly.y, fly.heading = cmd.pose_override
            fly.speed = cmd.speed; fly.turn = 0.0
        else:
            fly.turn = cmd.turn
            fly.heading = (fly.heading + cmd.turn * dt) % 360.0
            fly.speed = cmd.speed
            h = fly.heading_rad
            fly.x += fly.speed * dt * math.cos(h)
            fly.y += fly.speed * dt * math.sin(h)
            e = self.motor.cfg["escape"]
            if cmd.jump and (self.t_ms - fly.last_jump_ms) > e["refractory_ms"]:
                fly.jump_flag = True
                fly.last_jump_ms = self.t_ms
                near = self.nearest_at_tick_start(fly)
                if near is not None:
                    away = math.atan2(fly.y - near[1], fly.x - near[0])
                else:
                    away = h
                fly.x += e["jump_mm"] * math.cos(away)
                fly.y += e["jump_mm"] * math.sin(away)
                fly.heading = math.degrees(away) % 360.0
            # circular arena: clamp to the wall and slide along it
            r = math.hypot(fly.x, fly.y)
            if r > self.radius:
                fly.x *= self.radius / r; fly.y *= self.radius / r
        fly.song = cmd.song
        fly.emitters["song_intensity"] = cmd.song
        fly.emitters["speed"] = fly.speed
        fly.emitters.update(cmd.emitter_override)

    def advance(self):
        self.tick += 1
        self.t_ms += self.dt_ms

    def snapshot(self, fly: Fly) -> dict:
        d = {"x": fly.x, "y": fly.y, "heading": fly.heading, "speed": fly.speed, "turn": fly.turn,
             "song": fly.song, "jump": fly.jump_flag}
        d.update({f"em_{k}": v for k, v in fly.emitters.items()})
        return d
