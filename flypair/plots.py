"""Figures: side-by-side spike rasters and rate time series (matplotlib, headless-safe)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .record import Run


def rasters(run: Run, groups=None, flies=None, out: Path | str | None = None, max_neurons_per_group: int = 40):
    flies = flies or run.flies
    if groups is None:
        rg = run.meta.get("record_groups", {})
        spec = run.meta.get("fly_specs")
        groups = [g for g in run.spikes["group"].unique() if g]
        wanted = run.meta.get("raster_groups") or []
        groups = list(dict.fromkeys(list(wanted) + groups))
    if not groups:
        raise ValueError("no raster spikes recorded; set record.rasters in the scenario")
    fig, axes = plt.subplots(len(groups), len(flies), figsize=(4.5 * len(flies), 1.6 * len(groups) + 1),
                             sharex=True, squeeze=False)
    for j, f in enumerate(flies):
        sp = run.spikes[run.spikes.fly == f]
        for i, g in enumerate(groups):
            ax = axes[i, j]
            s = sp[sp.group == g]
            neurons = np.sort(s.neuron.unique())[:max_neurons_per_group]
            row = {n: k for k, n in enumerate(neurons)}
            s = s[s.neuron.isin(neurons)]
            ax.scatter(s.t_ms, s.neuron.map(row), s=2, c="k", marker="|")
            ax.set_ylabel(f"{g}\n(n={len(neurons)})", fontsize=8)
            ax.set_ylim(-1, max(len(neurons), 1))
            if i == 0:
                ax.set_title(f"fly {f}")
            if i == len(groups) - 1:
                ax.set_xlabel("time (ms)")
    fig.suptitle(f"{run.name}: spike rasters")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    return fig


def rates(run: Run, groups=None, flies=None, out: Path | str | None = None):
    flies = flies or run.flies
    if groups is None:
        key = ["p1", "pip10", "wing_mn", "dna02_L", "dna02_R", "forward_dn", "dnp01", "mdn", "vpodn", "dnp13", "mal", "mn9"]
        have = [c[5:] for c in run.rate_columns()]
        active = [g for g in have if run.frames[f"rate_{g}"].fillna(0).max() > 0]
        groups = [g for g in key if g in active] + [g for g in active if g not in key] + [g for g in key if g in have and g not in active]
        groups = list(dict.fromkeys(groups))[:10]
    fig, axes = plt.subplots(len(groups) + 1, 1, figsize=(9, 1.5 * (len(groups) + 1) + 1), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for f in flies:
        ff = run.fly_frames(f)
        axes[0].plot(ff.t_ms, ff.song, label=f"{f} song")
    axes[0].set_ylabel("song\n(0-1)"); axes[0].legend(fontsize=7, loc="upper right")
    for i, g in enumerate(groups, 1):
        col = f"rate_{g}"
        for f in flies:
            ff = run.fly_frames(f)
            if col in ff and ff[col].notna().any():
                axes[i].plot(ff.t_ms, ff[col], label=f)
        axes[i].set_ylabel(f"{g}\n(Hz)", fontsize=8)
    axes[-1].set_xlabel("time (ms)")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{run.name}: group rates (mean Hz per neuron, per {run.meta.get('world_dt_ms')} ms tick)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    return fig


def trajectories(run: Run, out: Path | str | None = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    R = run.meta.get("arena", {}).get("radius_mm", 20)
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="gray"))
    for f in run.flies:
        ff = run.fly_frames(f)
        ax.plot(ff.x, ff.y, label=f, lw=1)
        ax.plot(ff.x.iloc[0], ff.y.iloc[0], "o", ms=4, color=ax.lines[-1].get_color())
    ax.set_aspect("equal"); ax.set_xlim(-R - 1, R + 1); ax.set_ylim(-R - 1, R + 1); ax.legend(fontsize=8)
    ax.set_title(f"{run.name}: trajectories (mm)")
    if out:
        fig.savefig(out, dpi=120)
    return fig
