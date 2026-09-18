import numpy as np
import pytest

from flypair.brain import Brain
from flypair.constants import SHIU
from tests.conftest import micro_connectome


def rate_of(brain, idx, n_steps):
    brain.run(n_steps)
    c = brain.take_counts()
    return c[:, idx] * 1000.0 / (n_steps * SHIU.dt)


def test_constants_match_shiu():
    assert SHIU.dt == 0.1 and SHIU.v_th == -45.0 and SHIU.v_0 == -52.0
    assert SHIU.rfc_ticks == 22 and SHIU.delay_ticks == 18
    assert abs(SHIU.w_ext - 68.75) < 1e-9


def test_poisson_driven_neuron_fires_at_commanded_rate():
    c = micro_connectome([], n=2)
    b = Brain(c, 1, seeds=[3], propagate="gather")
    b.set_rates(0, [0], 150.0); b.commit_rates()
    r = rate_of(b, 0, 20000)[0]
    assert 130 < r < 170          # every Poisson draw fires the neuron (68.75 mV >> 7 mV gap)


def test_synaptic_fI_curve_monotonic():
    """Neuron 1 gets synapses from Poisson-driven neuron 0; more synapses -> higher rate."""
    rates = []
    for w in (5, 15, 40, 100):
        c = micro_connectome([(0, 1, w)], n=2)
        b = Brain(c, 1, seeds=[0], propagate="gather")
        b.set_rates(0, [0], 100.0); b.commit_rates()
        rates.append(rate_of(b, 1, 10000)[0])
    assert rates[0] < rates[-1]
    assert all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))
    assert rates[0] < 5            # 5 synapses = 1.4 mV per spike: essentially silent
    assert rates[-1] > 20          # 100 synapses = 27.5 mV on g -> ~5 mV on v: needs 2 inputs within ~10 ms


def test_refractory_caps_rate():
    c = micro_connectome([(0, 1, 5000)], n=2)   # 1375 mV per input spike: fires within one tick of arrival
    b = Brain(c, 1, seeds=[0], propagate="gather")
    b.set_rates(0, [0], 5000.0); b.commit_rates()  # input every ~2 ticks
    r = rate_of(b, 1, 20000)[0]
    assert r <= 1000.0 / SHIU.t_rfc + 1   # 454 Hz ceiling from the 2.2 ms refractory period
    assert r > 300                        # ~ 1000 / (2.2 ms refractory + ~0.3 ms to next input)


def test_delay_is_18_ticks():
    c = micro_connectome([(0, 1, 5000)], n=2)   # huge weight: v crosses threshold on the tick after arrival
    b = Brain(c, 1, seeds=[0], propagate="gather", record_indices=[0, 1])
    b.set_rates(0, [0], 20.0); b.commit_rates()
    b.run(5000); b.flush_record()
    t, fl, nrn = b.record.arrays()
    t0 = t[nrn == 0]; t1 = set(t[nrn == 1].tolist())
    assert len(t0) > 5 and len(t1) > 5
    # spike at s arrives at s + 18 (added to g after that tick's threshold test) -> fires at s + 19
    isolated = [s for i, s in enumerate(t0) if i == 0 or s - t0[i - 1] > 60]
    assert len(isolated) > 3
    assert all((s + SHIU.delay_ticks + 1) in t1 for s in isolated)


def test_inhibitory_sign_reduces_firing():
    exc = micro_connectome([(0, 2, 40)], n=3)
    both = micro_connectome([(0, 2, 40), (1, 2, -40)], n=3, nt=["acetylcholine", "gaba", "acetylcholine"])
    r = []
    for c in (exc, both):
        b = Brain(c, 1, seeds=[0], propagate="gather")
        b.set_rates(0, [0, 1], 100.0); b.commit_rates()
        r.append(rate_of(b, 2, 10000)[0])
    assert r[1] < r[0] * 0.5


def test_spmm_and_gather_agree():
    from flypair.connectome.tiny import build_tiny
    c = build_tiny()
    sugar = np.flatnonzero((c.neurons.type == "LB3") & (c.neurons.side == "R"))
    hashes = []
    for mode in ("spmm", "gather"):
        b = Brain(c, 2, seeds=[1, 2], propagate=mode)
        b.set_rates(0, sugar, 100.0); b.commit_rates()
        b.run(2000)
        hashes.append((b.state_hash(), b.take_counts().sum()))
    assert hashes[0] == hashes[1]
