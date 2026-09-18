"""InputSource: the seam between a fly's body/senses and whatever drives them.

Every fly in a scenario has one InputSource. Each world tick the runner asks it:

  motor(fly, rates, world, tick)   -> MotorCommand   (how the body moves / emits this tick)
  sensory(fly, world, tick)        -> dict group -> Hz (extra Poisson drive for its brain)

The default `BrainSource` decodes motor commands from the fly's own brain output
rates (World.motor decoders) and adds nothing to the channel-derived sensory drive.
`RecordedSource` (the `playback` control) replays poses/emitters from a previous
run and needs no brain. Other sources (scripted or interactive players) plug in
here without touching the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MotorCommand:
    speed: float = 0.0          # mm/s, negative = backward
    turn: float = 0.0           # deg/s, positive = left (counter-clockwise)
    jump: bool = False          # escape jump this tick
    song: float = 0.0           # 0..1 song intensity
    pose_override: tuple | None = None   # (x, y, heading_deg) to set directly (playback/puppet)
    emitter_override: dict = field(default_factory=dict)


class InputSource:
    name = "abstract"
    needs_brain = True

    def motor(self, fly, rates: dict, world, tick: int) -> MotorCommand:
        raise NotImplementedError

    def sensory(self, fly, world, tick: int) -> dict:
        return {}

    def reset(self):
        pass


class BrainSource(InputSource):
    """Default: the fly's brain drives its body through the world's motor decoders."""
    name = "brain"
    needs_brain = True

    def motor(self, fly, rates, world, tick):
        return world.motor.decode(fly, rates, world.dt_ms)


class RecordedSource(InputSource):
    """Playback control: pose and emitters come from a recorded run; no brain is simulated."""
    name = "playback"
    needs_brain = False

    def __init__(self, table):
        """table: DataFrame rows for ONE fly with columns tick, x, y, heading, song, plus emitter columns."""
        self.table = table.sort_values("tick").reset_index(drop=True)
        self.by_tick = {int(r.tick): r for r in self.table.itertuples()}
        self.last = self.table.iloc[-1]

    def recorded_rates(self, tick: int) -> dict:
        """The recorded fly's own group rates at this tick (for metrics/plots; it has no brain now)."""
        r = self.by_tick.get(int(tick), self.last)
        return {c[5:]: float(getattr(r, c)) for c in self.table.columns if c.startswith("rate_")}

    def motor(self, fly, rates, world, tick):
        r = self.by_tick.get(int(tick), self.last)
        em = {c: float(getattr(r, c)) for c in self.table.columns if c.startswith("em_")}
        return MotorCommand(speed=float(getattr(r, "speed", 0.0)), song=float(r.song),
                            jump=bool(getattr(r, "jump", False)),
                            pose_override=(float(r.x), float(r.y), float(r.heading)),
                            emitter_override={k[3:]: v for k, v in em.items()})


class ScriptedSource(InputSource):
    """A stationary (or scripted) emitter with no brain: `timeline` maps tick -> MotorCommand kwargs."""
    name = "scripted"
    needs_brain = False

    def __init__(self, timeline: dict | None = None, sensory_timeline: dict | None = None):
        self.timeline = {int(k): v for k, v in (timeline or {}).items()}
        self.sensory_timeline = {int(k): v for k, v in (sensory_timeline or {}).items()}
        self.current = MotorCommand()

    def motor(self, fly, rates, world, tick):
        if tick in self.timeline:
            self.current = MotorCommand(**self.timeline[tick])
        return self.current

    def sensory(self, fly, world, tick):
        return dict(self.sensory_timeline.get(tick, {}))
