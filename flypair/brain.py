"""Batched leaky integrate-and-fire brain on a shared connectome (PyTorch).

One Brain holds one connectome's W and simulates B flies that share it. All
state tensors are [B, N]. Dynamics and tick order follow Shiu et al. 2024 as
implemented in Brian2 (see constants.py and PLAN.md):

  1. rfc -= 1 ; not_ref = rfc == 0
  2. exact linear update of v, g where not_ref
  3. spike = not_ref & (v > v_th) ; spike &= ~silence[b]
  4. delayed = ring[t % delay_ticks] ; I = W @ delayed  (signed synapse counts)
  5. g += w_syn * gain[b] * I  where not_ref ; v += w_ext * poisson  where not_ref
  6. reset v, g ; rfc reload (0 for Poisson-driven neurons, as in Shiu's poi()) ;
     ring[t % delay_ticks] = spike ; counts += spike

Per-fly knobs are masks and scalars (silence mask, Poisson drive rates, gain,
background noise, RNG seed) so W is never edited and stays shared.

Two propagation kernels: 'spmm' (CSR sparse @ dense, good on GPU / dense
activity) and 'gather' (event-driven row gather over firing neurons, good on
CPU / sparse activity). 'auto' benchmarks both on the first run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
warnings.filterwarnings("ignore", message="Sparse invariant checks")
from scipy import sparse

from .constants import SHIU, LIFParams


def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class SpikeRecord:
    """Spikes of recorded neurons: arrays of (step, fly, neuron_index)."""
    steps: list = field(default_factory=list)
    flies: list = field(default_factory=list)
    neurons: list = field(default_factory=list)

    def extend(self, t: np.ndarray, b: np.ndarray, n: np.ndarray):
        self.steps.append(t); self.flies.append(b); self.neurons.append(n)

    def arrays(self):
        if not self.steps:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z
        return (np.concatenate(self.steps), np.concatenate(self.flies), np.concatenate(self.neurons))


class Brain:
    def __init__(self, connectome, n_flies: int = 1, params: LIFParams = SHIU,
                 device: str = "auto", seeds=None, propagate: str = "auto",
                 record_indices=None, dtype=torch.float32):
        self.c = connectome
        self.p = params
        self.B = int(n_flies)
        self.N = connectome.n
        self.device = pick_device(device)
        self.dtype = dtype
        self.propagate_mode = propagate
        self._build_matrices(connectome.W)
        # per-fly knobs
        self.gain = torch.ones(self.B, device=self.device, dtype=dtype)
        self.silence = torch.zeros(self.B, self.N, device=self.device, dtype=torch.bool)
        self.noise_hz = torch.zeros(self.B, device=self.device, dtype=dtype)
        self.rates = np.zeros((self.B, self.N), dtype=np.float32)   # host-side Poisson rates (Hz)
        self.drive_idx = torch.zeros(0, dtype=torch.long, device=self.device)
        self.drive_p = torch.zeros(self.B, 0, device=self.device, dtype=dtype)
        self.rfc_reload = torch.full((self.B, self.N), params.rfc_ticks, dtype=torch.int32, device=self.device)
        self.set_record_indices(record_indices)
        self.seeds = list(seeds) if seeds is not None else list(range(self.B))
        if len(self.seeds) != self.B:
            raise ValueError(f"need {self.B} seeds, got {len(self.seeds)}")
        self.reset()
        # constants as tensors
        p = params
        self.k = {name: torch.tensor(getattr(p, name), dtype=dtype, device=self.device)
                  for name in ("decay_v", "decay_g", "v0_term", "couple_g", "v_0", "v_rst", "v_th", "w_syn", "w_ext")}

    # ------------------------------------------------------------ matrices
    def _build_matrices(self, W: sparse.csr_matrix):
        W = W.tocsr().astype(np.float32)
        # spmm: W [post, pre] as torch CSR
        self.W_csr = torch.sparse_csr_tensor(
            torch.from_numpy(W.indptr.astype(np.int64)), torch.from_numpy(W.indices.astype(np.int64)),
            torch.from_numpy(W.data), size=(self.N, self.N), dtype=self.dtype).to(self.device)
        # gather: W^T [pre, post] CSR arrays (row = presynaptic neuron's outgoing edges)
        WT = W.T.tocsr()
        WT.sort_indices()
        self.wt_indptr = torch.from_numpy(WT.indptr.astype(np.int64)).to(self.device)
        self.wt_indices = torch.from_numpy(WT.indices.astype(np.int64)).to(self.device)
        self.wt_data = torch.from_numpy(WT.data.astype(np.float32)).to(self.device, self.dtype)

    # ------------------------------------------------------------ state
    def reset(self, seeds=None):
        if seeds is not None:
            self.seeds = list(seeds)
        p, B, N, d = self.p, self.B, self.N, self.device
        self.v = torch.full((B, N), p.v_0, dtype=self.dtype, device=d)
        self.g = torch.zeros((B, N), dtype=self.dtype, device=d)
        self.rfc = torch.zeros((B, N), dtype=torch.int32, device=d)
        self.ring = torch.zeros((p.delay_ticks, B, N), dtype=torch.bool, device=d)
        self.counts = torch.zeros((B, N), dtype=torch.int32, device=d)
        self.t = 0
        self.gens = []
        for s in self.seeds:
            g = torch.Generator(device=d)
            g.manual_seed(int(s))
            self.gens.append(g)
        self.record = SpikeRecord()
        self._rec_buf = None
        self._rec_t0 = 0

    # ------------------------------------------------------------ per-fly knobs
    def set_silence(self, fly: int, indices, on: bool = True):
        idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        self.silence[fly, idx] = on

    def clear_silence(self, fly: int | None = None):
        if fly is None:
            self.silence.zero_()
        else:
            self.silence[fly].zero_()

    def set_gain(self, fly: int, gain: float):
        self.gain[fly] = float(gain)

    def set_noise(self, fly: int, hz: float):
        self.noise_hz[fly] = float(hz)

    def set_rates(self, fly: int, indices, rate_hz: float):
        """Set Poisson drive rate (Hz) for neuron indices of one fly (overwrites those entries)."""
        self.rates[fly, np.asarray(indices, dtype=np.int64)] = float(rate_hz)

    def clear_rates(self, fly: int | None = None):
        if fly is None:
            self.rates[:] = 0
        else:
            self.rates[fly] = 0

    def commit_rates(self):
        """Upload the host-side rate table; call after set_rates/clear_rates."""
        p = self.p
        nz = np.flatnonzero(self.rates.any(axis=0))
        prob = self.rates[:, nz] * (p.dt / 1000.0)
        if prob.size and prob.max() > 1.0:
            raise ValueError(f"Poisson rate too high for dt={p.dt} ms: max {self.rates.max():.0f} Hz > {1000/p.dt:.0f} Hz")
        self.drive_idx = torch.as_tensor(nz, dtype=torch.long, device=self.device)
        self.drive_p = torch.as_tensor(prob, dtype=self.dtype, device=self.device)
        # Shiu: Poisson-driven neurons have no refractory period
        self.rfc_reload.fill_(p.rfc_ticks)
        if nz.size:
            driven = torch.as_tensor(self.rates[:, nz] > 0, device=self.device)
            sub = self.rfc_reload[:, self.drive_idx]
            sub[driven] = 0
            self.rfc_reload[:, self.drive_idx] = sub

    def set_record_indices(self, indices):
        if indices is None or len(indices) == 0:
            self.record_idx = None
        else:
            self.record_idx = torch.as_tensor(np.unique(np.asarray(indices, dtype=np.int64)), device=self.device)

    # ------------------------------------------------------------ propagation
    def _propagate_spmm(self, delayed: torch.Tensor) -> torch.Tensor:
        S = delayed.to(self.dtype).T.contiguous()                 # [N, B]
        return torch.sparse.mm(self.W_csr, S).T                   # [B, N]

    def _propagate_gather(self, delayed: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.B * self.N, dtype=self.dtype, device=self.device)
        nz = delayed.nonzero(as_tuple=False)                      # [K, 2] (fly, pre)
        if nz.shape[0] == 0:
            return out.view(self.B, self.N)
        b, pre = nz[:, 0], nz[:, 1]
        starts = self.wt_indptr[pre]
        lens = self.wt_indptr[pre + 1] - starts
        tot = int(lens.sum())
        if tot == 0:
            return out.view(self.B, self.N)
        rep_start = torch.repeat_interleave(starts, lens)
        rep_b = torch.repeat_interleave(b, lens)
        offs = torch.arange(tot, device=self.device) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        e = rep_start + offs
        tgt = rep_b * self.N + self.wt_indices[e]
        out.index_add_(0, tgt, self.wt_data[e])
        return out.view(self.B, self.N)

    def propagate(self, delayed: torch.Tensor) -> torch.Tensor:
        if self.propagate_mode == "auto":
            self.propagate_mode = self.benchmark_propagate()
        if self.propagate_mode == "spmm":
            return self._propagate_spmm(delayed)
        return self._propagate_gather(delayed)

    def benchmark_propagate(self, n_iter: int = 20, density: float = 0.002, verbose: bool = True) -> str:
        """Time both kernels on a random spike pattern of the given density; return the faster."""
        gen = torch.Generator(device=self.device); gen.manual_seed(0)
        delayed = torch.rand(self.B, self.N, generator=gen, device=self.device) < density
        res = {}
        for mode, fn in (("spmm", self._propagate_spmm), ("gather", self._propagate_gather)):
            fn(delayed)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn(delayed)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            res[mode] = (time.perf_counter() - t0) / n_iter
        best = min(res, key=res.get)
        if verbose:
            print(f"[brain] propagate benchmark on {self.device} (B={self.B}, N={self.N:,}, density {density}): "
                  + ", ".join(f"{k} {v*1e3:.3f} ms" for k, v in res.items()) + f" -> using {best}")
        return best

    # ------------------------------------------------------------ step
    @torch.no_grad()
    def step(self):
        p, k = self.p, self.k
        slot = self.t % p.delay_ticks
        # 1. refractory decrement
        self.rfc = torch.clamp(self.rfc - 1, min=0)
        not_ref = self.rfc == 0
        # 2. exact linear update (non-refractory only)
        v_upd = k["v0_term"] + (self.g * k["couple_g"] + self.v * k["decay_v"])
        g_upd = self.g * k["decay_g"]
        self.v = torch.where(not_ref, v_upd, self.v)
        self.g = torch.where(not_ref, g_upd, self.g)
        # 3. threshold, silence mask
        spike = not_ref & (self.v > k["v_th"]) & ~self.silence
        # 4. delayed spikes -> synaptic input
        delayed = self.ring[slot]
        I = self.propagate(delayed)
        # 5. synaptic input onto g (shielded while refractory); external Poisson onto v
        self.g = self.g + torch.where(not_ref, I * (k["w_syn"] * self.gain[:, None]), torch.zeros((), dtype=self.dtype, device=self.device))
        if self.drive_idx.numel():
            draws = torch.empty_like(self.drive_p)
            for b in range(self.B):
                torch.rand(self.drive_p.shape[1], generator=self.gens[b], device=self.device, out=draws[b])
            hit = (draws < self.drive_p) & not_ref[:, self.drive_idx]
            self.v[:, self.drive_idx] += hit.to(self.dtype) * k["w_ext"]
        if bool((self.noise_hz > 0).any()):
            for b in range(self.B):
                if self.noise_hz[b] > 0:
                    pn = float(self.noise_hz[b]) * p.dt / 1000.0
                    r = torch.rand(self.N, generator=self.gens[b], device=self.device)
                    self.v[b] += ((r < pn) & not_ref[b]).to(self.dtype) * k["w_ext"]
        # 6. reset, refractory reload, ring store, counts
        self.v = torch.where(spike, k["v_rst"], self.v)
        self.g = torch.where(spike, torch.zeros((), dtype=self.dtype, device=self.device), self.g)
        self.rfc = torch.where(spike, self.rfc_reload, self.rfc)
        self.ring[slot] = spike
        self.counts += spike.to(torch.int32)
        if self.record_idx is not None:
            self._record(spike)
        self.t += 1

    def _record(self, spike):
        if self._rec_buf is None:
            self._rec_buf = []
            self._rec_t0 = self.t
        self._rec_buf.append(spike[:, self.record_idx])
        if len(self._rec_buf) >= 256:
            self.flush_record()

    def flush_record(self):
        if not self._rec_buf:
            return
        buf = torch.stack(self._rec_buf)               # [T, B, R]
        nz = buf.nonzero(as_tuple=False).cpu().numpy()
        if nz.shape[0]:
            self.record.extend(nz[:, 0] + self._rec_t0, nz[:, 1], self.record_idx.cpu().numpy()[nz[:, 2]])
        self._rec_buf = []
        self._rec_t0 = self.t

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self.step()

    def take_counts(self) -> np.ndarray:
        """Return accumulated spike counts [B, N] since the last call and zero them."""
        c = self.counts.cpu().numpy().copy()
        self.counts.zero_()
        return c

    def state_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for t in (self.v, self.g, self.rfc):
            h.update(t.cpu().numpy().tobytes())
        return h.hexdigest()

    def steps_per_second(self, n_steps: int = 200) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.run(n_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return n_steps / (time.perf_counter() - t0)
