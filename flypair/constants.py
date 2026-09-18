"""LIF constants of Shiu et al. 2024 (reference/Drosophila_brain_model/model.py).

Times in ms, voltages in mV. Nothing here is tuned; every value is copied from
`default_params` in that file. dt is Brian2's default clock (0.1 ms), which the
model never overrides (confirmed tick-by-tick by drosophila-brain-mlx).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LIFParams:
    dt: float = 0.1        # ms, Brian2 default clock
    v_0: float = -52.0     # resting potential
    v_rst: float = -52.0   # reset potential
    v_th: float = -45.0    # threshold, strict v > v_th
    t_mbr: float = 20.0    # membrane time constant
    tau: float = 5.0       # synaptic time constant (dg/dt = -g/tau)
    t_rfc: float = 2.2     # refractory period
    t_dly: float = 1.8     # synaptic delay
    w_syn: float = 0.275   # mV per synaptic contact
    f_poi: float = 250.0   # Poisson input weight factor: w_ext = w_syn * f_poi = 68.75 mV on v
    r_poi: float = 150.0   # default Poisson rate (Hz) in the Shiu experiments

    @property
    def w_ext(self) -> float:
        return self.w_syn * self.f_poi

    @property
    def rfc_ticks(self) -> int:
        return round(self.t_rfc / self.dt)   # 22

    @property
    def delay_ticks(self) -> int:
        return round(self.t_dly / self.dt)   # 18

    # Exact update of the linear system over one dt, in the association order
    # Brian2 (method='linear') generates. Transcribed from mlx core.py, which
    # inspected the generated code; the order matters for float32 parity.
    @property
    def decay_v(self) -> float:
        return math.exp(-self.dt / self.t_mbr)

    @property
    def decay_g(self) -> float:
        return math.exp(-self.dt / self.tau)

    @property
    def v0_term(self) -> float:
        return self.v_0 - self.v_0 * math.exp(-self.dt / self.t_mbr)

    @property
    def couple_g(self) -> float:
        return (((self.tau / (self.t_mbr - self.tau))
                 * (-math.exp(self.dt / self.t_mbr) + math.exp(self.dt / self.tau)))
                * math.exp(-self.dt / self.t_mbr)) * math.exp(-self.dt / self.tau)


SHIU = LIFParams()
