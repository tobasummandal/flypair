"""Channels: declarative source-fly -> target-fly sensory mappings.

channels:
  - name: song_to_jo
    source: song_intensity     # an emitter of the source fly (song_intensity, pheromone_male,
                               # pheromone_female, size, speed), a geometric quantity seen from the
                               # target (angular_size, looming, distance, presence), or "rate:<group>"
    target: jo_ab              # sensory group of the target fly (Hz of Poisson drive)
    gain: 120                  # Hz per unit source at zero distance
    falloff: {type: exp, scale_mm: 15}   # none | exp | inverse_square | linear | contact
    fov_deg: 360               # only if the source lies within +/- fov/2 of the target's heading
    split_lr: false            # drive <target>_L or <target>_R by the source's bearing
    delay_ms: 0                # transport delay (world ticks)
    clip_hz: 250
    from: all                  # source fly names (or all); to: all (every OTHER fly)
    to: all
Multiple channels onto one target group sum, then clip. Unresolved target groups
are skipped with a warning (or raise with on_unresolved: fail).
"""
from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DEFAULT_CHANNELS = [
    {"name": "song_to_jo", "source": "song_intensity", "target": "jo_ab", "gain": 120.0,
     "falloff": {"type": "exp", "scale_mm": 15.0}, "fov_deg": 360, "split_lr": True, "clip_hz": 200},
    {"name": "body_to_lc10a", "source": "angular_size", "target": "lc10a", "gain": 4.0,
     "falloff": {"type": "none"}, "fov_deg": 180, "split_lr": True, "clip_hz": 150},
    {"name": "looming_to_lplc2", "source": "looming", "target": "lplc2", "gain": 1.5,
     "falloff": {"type": "none"}, "fov_deg": 240, "split_lr": True, "clip_hz": 200},
    {"name": "male_pheromone_to_gr32a", "source": "pheromone_male", "target": "gr32a", "gain": 80.0,
     "falloff": {"type": "contact", "radius_mm": 3.0}, "fov_deg": 360, "split_lr": False, "clip_hz": 150},
    {"name": "male_pheromone_to_or67d", "source": "pheromone_male", "target": "or67d", "gain": 60.0,
     "falloff": {"type": "exp", "scale_mm": 5.0}, "fov_deg": 360, "split_lr": True, "clip_hz": 150},
    {"name": "female_pheromone_to_ppk23", "source": "pheromone_female", "target": "ppk23", "gain": 80.0,
     "falloff": {"type": "contact", "radius_mm": 3.0}, "fov_deg": 360, "split_lr": False, "clip_hz": 150},
]

FALLOFFS = {"none", "exp", "inverse_square", "linear", "contact"}
EMITTER_SOURCES = {"song_intensity", "pheromone_male", "pheromone_female", "size", "speed"}
GEOM_SOURCES = {"angular_size", "looming", "distance", "presence"}


def falloff(spec: dict, d: float) -> float:
    t = (spec or {}).get("type", "none")
    if t == "none":
        return 1.0
    if t == "exp":
        return math.exp(-d / float(spec.get("scale_mm", 10.0)))
    if t == "inverse_square":
        r0 = float(spec.get("r0_mm", 2.0))
        return 1.0 / (1.0 + (d / r0) ** 2)
    if t == "linear":
        rmax = float(spec.get("rmax_mm", 20.0))
        return max(0.0, 1.0 - d / rmax)
    if t == "contact":
        return 1.0 if d <= float(spec.get("radius_mm", 3.0)) else 0.0
    raise ValueError(f"unknown falloff type {t!r}; choose from {sorted(FALLOFFS)}")


@dataclass
class Channel:
    name: str
    source: str
    target: str
    gain: float = 1.0
    falloff: dict = field(default_factory=lambda: {"type": "none"})
    fov_deg: float = 360.0
    split_lr: bool = False
    delay_ms: float = 0.0
    clip_hz: float = 250.0
    sources: object = "all"      # 'from'
    targets: object = "all"      # 'to'
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Channel":
        d = dict(d)
        d.setdefault("name", f"{d.get('source')}->{d.get('target')}")
        d["sources"] = d.pop("from", "all"); d["targets"] = d.pop("to", "all")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"channel {d['name']!r}: unknown keys {sorted(unknown)}")
        for k in ("source", "target"):
            if k not in d:
                raise ValueError(f"channel {d['name']!r}: missing required key {k!r}")
        if (d.get("falloff") or {}).get("type", "none") not in FALLOFFS:
            raise ValueError(f"channel {d['name']!r}: falloff.type must be one of {sorted(FALLOFFS)}")
        src = d["source"]
        if not (src in EMITTER_SOURCES or src in GEOM_SOURCES or src.startswith("rate:")):
            raise ValueError(f"channel {d['name']!r}: source {src!r} must be an emitter {sorted(EMITTER_SOURCES)}, "
                             f"a geometric quantity {sorted(GEOM_SOURCES)}, or 'rate:<group>'")
        return cls(**d)


class ChannelBank:
    """Evaluates all channels each world tick and returns per-fly sensory rates."""

    def __init__(self, specs, world, registries: dict, on_unresolved: str = "skip", global_gain: float = 1.0):
        self.world = world
        self.registries = registries
        self.global_gain = global_gain
        self.channels: list[Channel] = []
        self.skipped: list[tuple] = []
        for s in specs:
            ch = Channel.from_dict(s)
            self.channels.append(ch)
        self._prev_angular = {}
        self._queues = {}
        self.on_unresolved = on_unresolved

    def target_group(self, ch: Channel, fly, bearing: float):
        """Resolve the target group name for this fly (L/R split); None if unresolved."""
        reg = self.registries.get(fly.connectome)
        if reg is None:
            return None
        name = ch.target
        if ch.split_lr:
            side = "L" if bearing > 0 else "R"
            cand = f"{name}_{side}"
            if reg.has(cand):
                return cand
        if reg.has(name):
            return name
        key = (ch.name, fly.connectome)
        if key not in self.skipped:
            self.skipped.append(key)
            msg = f"[channels] {ch.name}: target group {name!r} unresolved on {fly.connectome} -> channel skipped for {fly.name}"
            if self.on_unresolved == "fail":
                raise RuntimeError(msg)
            warnings.warn(msg)
        return None

    def source_value(self, ch: Channel, src, tgt, rates_by_fly: dict) -> float:
        w = self.world
        if ch.source in EMITTER_SOURCES:
            return float(src.emitters.get(ch.source, 0.0))
        if ch.source.startswith("rate:"):
            return float(rates_by_fly.get(src.name, {}).get(ch.source[5:], 0.0))
        if ch.source == "distance":
            return w.distance(tgt, src)
        if ch.source == "presence":
            return 1.0
        ang = w.angular_size(tgt, src)
        if ch.source == "angular_size":
            return ang
        if ch.source == "looming":       # deg/s of angular expansion, positive part
            key = (tgt.name, src.name)
            prev = self._prev_angular.get(key, ang)
            self._prev_angular[key] = ang
            return max(0.0, (ang - prev) / (w.dt_ms / 1000.0))
        raise ValueError(ch.source)

    def evaluate(self, rates_by_fly: dict) -> dict:
        """-> {fly.name: {group: Hz}} for this tick."""
        w = self.world
        out = {f.name: {} for f in w.flies}
        for ch in self.channels:
            if not ch.enabled:
                continue
            for src in w.flies:
                if ch.sources != "all" and src.name not in ch.sources:
                    continue
                for tgt in w.flies:
                    if tgt is src or tgt.brain_key is None:
                        continue
                    if ch.targets != "all" and tgt.name not in ch.targets:
                        continue
                    bearing = w.bearing(tgt, src)
                    if abs(bearing) > ch.fov_deg / 2.0:
                        continue
                    grp = self.target_group(ch, tgt, bearing)
                    if grp is None:
                        continue
                    val = self.source_value(ch, src, tgt, rates_by_fly)
                    hz = self.global_gain * ch.gain * val * falloff(ch.falloff, w.distance(tgt, src))
                    if ch.delay_ms > 0:
                        q = self._queues.setdefault((ch.name, src.name, tgt.name), deque())
                        q.append(hz)
                        n = max(1, int(round(ch.delay_ms / w.dt_ms)))
                        hz = q.popleft() if len(q) > n else 0.0
                    hz = min(hz, ch.clip_hz)
                    if hz > 0:
                        out[tgt.name][grp] = min(out[tgt.name].get(grp, 0.0) + hz, ch.clip_hz)
        return out

    def describe(self) -> str:
        lines = ["Channels (source fly -> target fly sensory group):"]
        for ch in self.channels:
            lines.append(f"  {ch.name:28s} {ch.source:16s} -> {ch.target}{'(L/R)' if ch.split_lr else ''}"
                         f"  gain {ch.gain} Hz/unit, falloff {ch.falloff}, fov {ch.fov_deg}, clip {ch.clip_hz} Hz"
                         + (f", delay {ch.delay_ms} ms" if ch.delay_ms else ""))
        return "\n".join(lines)
