"""Arena animation -> MP4 (matplotlib + ffmpeg, headless). Flies = oriented triangles,
song = pulsing ring, escape jump = flash, right panel = live rate bars of key groups."""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .record import Run

KEY_GROUPS = ["p1", "pip10", "wing_mn", "dna02_L", "dna02_R", "dnp01", "mdn", "vpodn", "dnp13", "mn9"]
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]


def _triangle(x, y, heading_deg, size):
    h = np.radians(heading_deg)
    pts = np.array([[size, 0], [-size * 0.6, size * 0.5], [-size * 0.6, -size * 0.5]])
    rot = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    return pts @ rot.T + [x, y]


def render(run: Run, out: Path | str, fps: int = 25, speed: float = 1.0, captions: dict | None = None,
           watermark: str | None = None, groups=None, dpi: int = 100) -> Path:
    """captions: {tick: {fly_name: text}} overlay (used by dub.py). speed: playback speed vs real time."""
    out = Path(out)
    flies = run.flies
    frames = {f: run.fly_frames(f) for f in flies}
    n_ticks = len(frames[flies[0]])
    dt_ms = float(run.meta.get("world_dt_ms", 20))
    R = float(run.meta.get("arena", {}).get("radius_mm", 20))
    groups = groups or [g for g in KEY_GROUPS if f"rate_{g}" in run.frames.columns and run.frames[f"rate_{g}"].notna().any()]
    tick_stride = max(1, int(round(1000.0 / fps / dt_ms * speed)))
    ticks = list(range(0, n_ticks, tick_stride))
    rate_max = {g: max(float(np.nanmax(run.frames[f"rate_{g}"])), 1.0) for g in groups}

    fig = plt.figure(figsize=(11, 6), dpi=dpi)
    ax = fig.add_axes([0.03, 0.05, 0.55, 0.9]); ax.set_aspect("equal")
    ax.set_xlim(-R - 1, R + 1); ax.set_ylim(-R - 1, R + 1); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="gray", lw=1.5))
    axb = fig.add_axes([0.66, 0.1, 0.32, 0.8])
    axb.set_xlim(0, 1); axb.set_yticks(range(len(groups))); axb.set_yticklabels(groups, fontsize=8)
    axb.set_xlabel("rate / max"); axb.invert_yaxis()
    title = ax.set_title("")
    bodies, rings, flashes, labels, bars, caps = {}, {}, {}, {}, {}, {}
    for k, f in enumerate(flies):
        col = COLORS[k % len(COLORS)]
        bodies[f] = plt.Polygon(_triangle(0, 0, 0, 1.5), color=col)
        ax.add_patch(bodies[f])
        rings[f] = plt.Circle((0, 0), 0.1, fill=False, color=col, lw=2, alpha=0.0)
        ax.add_patch(rings[f])
        flashes[f] = plt.Circle((0, 0), 4, color="yellow", alpha=0.0)
        ax.add_patch(flashes[f])
        labels[f] = ax.text(0, 0, f, fontsize=9, color=col, ha="center", va="bottom")
        caps[f] = ax.text(0, 0, "", fontsize=8, color="black", ha="center", va="top",
                          bbox=dict(boxstyle="round", fc="white", ec=col, alpha=0.85))
        width = 0.8 / len(flies)
        bars[f] = axb.barh(np.arange(len(groups)) + (k - (len(flies) - 1) / 2) * width, np.zeros(len(groups)),
                           height=width, color=col, label=f)
    axb.legend(fontsize=8, loc="lower right")
    if watermark:
        fig.text(0.5, 0.01, watermark, ha="center", fontsize=9, color="red", alpha=0.8)

    def update(i):
        tick = ticks[i]
        title.set_text(f"{run.name}   t = {tick * dt_ms / 1000:.2f} s")
        arts = [title]
        for k, f in enumerate(flies):
            r = frames[f].iloc[tick]
            bodies[f].set_xy(_triangle(r.x, r.y, r.heading, 1.5))
            s = float(r.song)
            rings[f].center = (r.x, r.y); rings[f].radius = 2.0 + 2.5 * s * (0.5 + 0.5 * np.sin(tick * 0.9))
            rings[f].set_alpha(min(1.0, s * 1.2))
            flashes[f].center = (r.x, r.y); flashes[f].set_alpha(0.6 if bool(r.jump) else 0.0)
            labels[f].set_position((r.x, r.y + 2.2))
            if captions and tick in captions and f in captions[tick]:
                caps[f].set_text(captions[tick][f]); caps[f].set_position((r.x, r.y - 2.2))
            for j, g in enumerate(groups):
                v = r.get(f"rate_{g}", np.nan)
                bars[f][j].set_width(0.0 if np.isnan(v) else v / rate_max[g])
            arts += [bodies[f], rings[f], flashes[f], labels[f], caps[f]]
        return arts

    anim = animation.FuncAnimation(fig, update, frames=len(ticks), blit=False)
    if shutil.which("ffmpeg"):
        anim.save(str(out), writer=animation.FFMpegWriter(fps=fps, bitrate=2000))
    else:
        out = out.with_suffix(".gif")
        warnings.warn("ffmpeg not found: writing a GIF instead of MP4")
        anim.save(str(out), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out
