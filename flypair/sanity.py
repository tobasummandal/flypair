"""Single-fly sanity check: Shiu et al. sugar GRN -> MN9.

FlyWire reference (their shipped results, 30 trials x 1 s, 21 right sugar GRNs):
  100 Hz drive -> MN9 ~ 67 Hz ; 150 Hz -> ~93 Hz.
On MaleCNS the same constants are reused unchanged (mlx reports MN9_L ~ 53 Hz at
100 Hz on LB3b_R/LB3c_R); there is no published reference, so the number is
reported, not asserted.
"""
from __future__ import annotations

import time

import numpy as np

from .brain import Brain
from .connectome.base import Connectome
from .constants import SHIU
from .groups import resolve_for


def sugar_mn9(c: Connectome, rate_hz: float = 100.0, duration_ms: float = 1000.0, n_trials: int = 3,
              use_shiu_ids: bool = True, device: str = "auto", propagate: str = "auto", verbose: bool = True) -> dict:
    reg = resolve_for(c, strict=False, verbose=False)
    if c.name == "flywire" and use_shiu_ids:
        from .connectome.flywire import SHIU_SUGAR_IDS, SHIU_MN9_ID
        ids = c.neurons["id"].to_numpy()
        sugar = np.flatnonzero(np.isin(ids, SHIU_SUGAR_IDS))
        mn9 = np.flatnonzero(ids == SHIU_MN9_ID)
        label = f"Shiu's {len(sugar)} sugar GRN ids -> MN9 id {SHIU_MN9_ID}"
    else:
        sugar = reg["sugar_grn_R"] if reg.has("sugar_grn_R") else reg["sugar_grn"]
        mn9 = reg["mn9"]
        label = f"group sugar_grn_R ({len(sugar)} neurons) -> mn9 ({len(mn9)} neurons)"
    n_steps = int(round(duration_ms / SHIU.dt))
    b = Brain(c, n_trials, device=device, seeds=list(range(100, 100 + n_trials)), propagate=propagate)
    for t in range(n_trials):
        b.set_rates(t, sugar, rate_hz)
    b.commit_rates()
    t0 = time.time()
    b.run(n_steps)
    wall = time.time() - t0
    cnt = b.take_counts()
    per_s = 1000.0 / duration_ms
    res = {"connectome": c.name, "label": label, "rate_hz": rate_hz, "n_trials": n_trials, "duration_ms": duration_ms,
           "sugar_rate_hz": float(cnt[:, sugar].mean() * per_s),
           "mn9_rate_hz_per_neuron": [float(x) for x in cnt[:, mn9].mean(axis=0) * per_s],
           "mn9_rate_hz_mean": float(cnt[:, mn9].mean() * per_s),
           "active_neurons": int((cnt.sum(axis=0) > 0).sum()), "total_spikes_per_trial": float(cnt.sum() / n_trials),
           "steps_per_s": n_steps / wall, "wall_s": wall, "device": str(b.device), "propagate": b.propagate_mode}
    if c.name == "flywire":
        ref = {100.0: 67.0, 150.0: 93.3}.get(rate_hz)
        res["shiu_reference_mn9_hz"] = ref
    if verbose:
        print(f"[sanity] {c.name}: {label}, {rate_hz:.0f} Hz x {duration_ms:.0f} ms x {n_trials} trials on {b.device} ({b.propagate_mode})")
        print(f"         sugar GRNs fire at {res['sugar_rate_hz']:.1f} Hz; MN9 per neuron: "
              + ", ".join(f"{x:.1f}" for x in res["mn9_rate_hz_per_neuron"]) + " Hz"
              + (f"  (Shiu reference {res['shiu_reference_mn9_hz']} Hz)" if res.get("shiu_reference_mn9_hz") else ""))
        print(f"         {res['active_neurons']:,} neurons active, {res['total_spikes_per_trial']:,.0f} spikes/trial; "
              f"{res['steps_per_s']:,.0f} steps/s ({res['steps_per_s'] / 10000:.2f}x real time for batch {n_trials})")
    return res


def benchmark_devices(c: Connectome, n_flies: int = 2, n_steps: int = 500) -> dict:
    """steps/s on cpu and (if available) cuda, each with its faster propagate kernel."""
    import torch
    out = {}
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        b = Brain(c, n_flies, device=dev, seeds=list(range(n_flies)), propagate="auto")
        reg = resolve_for(c, strict=False, verbose=False)
        b.set_rates(0, reg["sugar_grn"], 100.0); b.commit_rates()
        b.run(50)
        sps = b.steps_per_second(n_steps)
        out[dev] = {"steps_per_s": sps, "propagate": b.propagate_mode, "bio_s_per_wall_s": sps / 10000.0}
        print(f"[benchmark] {dev}: {sps:,.0f} steps/s ({b.propagate_mode}) = {sps/10000:.3f} bio-s per wall-s for {n_flies} flies")
    best = max(out, key=lambda k: out[k]["steps_per_s"])
    out["best"] = best
    print(f"[benchmark] -> using {best}")
    return out
