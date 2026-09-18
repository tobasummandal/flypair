"""Run outputs: per-tick frames (poses, emitters, group rates), spikes, metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Run:
    name: str
    frames: pd.DataFrame            # one row per (tick, fly)
    spikes: pd.DataFrame            # columns: fly, t_ms, neuron, group
    meta: dict = field(default_factory=dict)

    @property
    def flies(self) -> list:
        return list(self.meta.get("flies", self.frames["fly"].unique().tolist()))

    def fly_frames(self, name: str) -> pd.DataFrame:
        return self.frames[self.frames["fly"] == name].sort_values("tick").reset_index(drop=True)

    def rate_columns(self) -> list:
        return [c for c in self.frames.columns if c.startswith("rate_")]

    def rate(self, fly: str, group: str) -> np.ndarray:
        return self.fly_frames(fly)[f"rate_{group}"].to_numpy()

    def save(self, out_dir: Path | str) -> Path:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        self.frames.to_parquet(d / "run.parquet", index=False)
        self.spikes.to_parquet(d / "spikes.parquet", index=False)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        # JSON copy of the frames for people without parquet readers
        self.frames.to_json(d / "run.json", orient="records")
        return d

    @classmethod
    def load(cls, d: Path | str) -> "Run":
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        return cls(meta.get("name", d.name), pd.read_parquet(d / "run.parquet"),
                   pd.read_parquet(d / "spikes.parquet"), meta)

    def summary(self) -> str:
        lines = [f"Run {self.name}: {self.meta.get('duration_ms')} ms, {len(self.flies)} flies, "
                 f"control={self.meta.get('control')}, wall {self.meta.get('wall_s', 0):.1f}s"]
        for f in self.flies:
            ff = self.fly_frames(f)
            rates = {c[5:]: ff[c].mean() for c in self.rate_columns() if ff[c].notna().any()}
            top = sorted(rates.items(), key=lambda kv: -kv[1])[:6]
            lines.append(f"  {f}: mean song {ff['song'].mean():.2f}, path {np.hypot(ff.x.diff(), ff.y.diff()).sum():.1f} mm, "
                         f"jumps {int(ff['jump'].sum())}; top rates " + ", ".join(f"{k} {v:.1f}Hz" for k, v in top))
        return "\n".join(lines)
