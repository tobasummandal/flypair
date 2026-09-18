"""Common connectome format shared by every loader.

A Connectome is:
  W        scipy CSR float32, shape [N, N], rows = POSTsynaptic, cols = PREsynaptic,
           values = signed synapse counts (sign from the presynaptic neurotransmitter).
           The runtime multiplies by w_syn.
  neurons  pandas DataFrame with N rows in W order: id (int64), type, class, side
           ('L','R','M' or ''), nt, plus any extra annotation columns the registry may query.
  meta     dict (name, source, counts, sign convention, build notes)

Cached as <cache>/<name>/W.npz + neurons.parquet + meta.json so a Colab session
builds once.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

REQUIRED_COLS = ("id", "type", "class", "side", "nt")


@dataclass
class Connectome:
    name: str
    W: sparse.csr_matrix
    neurons: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = self.W.shape[0]
        if self.W.shape != (n, n):
            raise ValueError(f"W must be square, got {self.W.shape}")
        if len(self.neurons) != n:
            raise ValueError(f"neuron table has {len(self.neurons)} rows but W has {n}")
        for c in REQUIRED_COLS:
            if c not in self.neurons.columns:
                raise ValueError(f"neuron table lacks required column {c!r}")
        if self.W.dtype != np.float32:
            self.W = self.W.astype(np.float32)
        self.W.sort_indices()
        self.neurons = self.neurons.reset_index(drop=True)

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def nnz(self) -> int:
        return int(self.W.nnz)

    def summary(self) -> str:
        pos = int((self.W.data > 0).sum())
        neg = int((self.W.data < 0).sum())
        return (f"{self.name}: {self.n:,} neurons, {self.nnz:,} connections "
                f"({pos:,} excitatory, {neg:,} inhibitory), "
                f"sum|w| = {float(np.abs(self.W.data).sum()):,.0f} synapses")

    # ---- cache -----------------------------------------------------------
    def save(self, cache_dir: Path | str) -> Path:
        d = Path(cache_dir) / self.name
        d.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(d / "W.npz", self.W, compressed=False)
        self.neurons.to_parquet(d / "neurons.parquet", index=False)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        return d

    @classmethod
    def load(cls, cache_dir: Path | str, name: str) -> "Connectome":
        d = Path(cache_dir) / name
        W = sparse.load_npz(d / "W.npz").tocsr()
        neurons = pd.read_parquet(d / "neurons.parquet")
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        return cls(name, W, neurons, meta)

    @staticmethod
    def cached(cache_dir: Path | str, name: str) -> bool:
        d = Path(cache_dir) / name
        return (d / "W.npz").exists() and (d / "neurons.parquet").exists()


def normalize_side(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().upper()
    return {"LEFT": "L", "RIGHT": "R", "CENTER": "M", "CENTRE": "M", "MIDDLE": "M",
            "M": "M", "L": "L", "R": "R"}.get(s, "")


def load_connectome(name: str, cache_dir: Path | str = "cache", force: bool = False,
                    **kwargs) -> Connectome:
    """Build-or-load by name: 'tiny', 'malecns', 'flywire'. Prints RAM/timing."""
    import psutil

    cache_dir = Path(cache_dir)
    if not force and Connectome.cached(cache_dir, name):
        t0 = time.time()
        c = Connectome.load(cache_dir, name)
        print(f"[connectome] loaded {c.summary()} from cache in {time.time()-t0:.1f}s")
        return c
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 2**30
    t0 = time.time()
    if name == "tiny":
        from .tiny import build_tiny
        c = build_tiny(**kwargs)
    elif name == "malecns":
        from .malecns import build_malecns
        c = build_malecns(cache_dir=cache_dir, **kwargs)
    elif name == "flywire":
        from .flywire import build_flywire
        c = build_flywire(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f"unknown connectome {name!r}; choose tiny, malecns or flywire")
    c.save(cache_dir)
    rss1 = proc.memory_info().rss / 2**30
    print(f"[connectome] built {c.summary()} in {time.time()-t0:.1f}s; "
          f"RSS {rss0:.2f} -> {rss1:.2f} GB; cached to {cache_dir / name}")
    return c
