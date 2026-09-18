import numpy as np

from flypair.brain import Brain


def _sugar(tiny):
    return np.flatnonzero((tiny.neurons.type == "LB3") & (tiny.neurons.side == "R"))


def test_batch_of_two_equals_two_separate_runs(tiny):
    sugar = _sugar(tiny)
    # separate runs
    singles = []
    for seed in (11, 22):
        b = Brain(tiny, 1, seeds=[seed], propagate="gather")
        b.set_rates(0, sugar, 100.0); b.commit_rates()
        b.run(3000)
        singles.append((b.v[0].cpu().numpy().copy(), b.take_counts()[0]))
    # batch
    bb = Brain(tiny, 2, seeds=[11, 22], propagate="gather")
    for f in range(2):
        bb.set_rates(f, sugar, 100.0)
    bb.commit_rates()
    bb.run(3000)
    cnt = bb.take_counts()
    for f in range(2):
        assert np.array_equal(bb.v[f].cpu().numpy(), singles[f][0])
        assert np.array_equal(cnt[f], singles[f][1])
    assert cnt.sum() > 0


def test_same_seed_same_result_different_seed_differs(tiny):
    sugar = _sugar(tiny)
    b = Brain(tiny, 3, seeds=[5, 5, 6], propagate="gather")
    for f in range(3):
        b.set_rates(f, sugar, 100.0)
    b.commit_rates(); b.run(2000)
    cnt = b.take_counts()
    assert np.array_equal(cnt[0], cnt[1])
    assert not np.array_equal(cnt[0], cnt[2])


def test_silence_mask_zeroes_spikes(tiny, tiny_groups):
    sugar = tiny_groups["sugar_grn_R"]
    b = Brain(tiny, 2, seeds=[1, 1], propagate="gather")
    for f in range(2):
        b.set_rates(f, sugar, 100.0)
    b.commit_rates()
    b.set_silence(1, tiny_groups["mn9"])
    b.run(5000)
    cnt = b.take_counts()
    assert cnt[0, tiny_groups["mn9"]].sum() > 50
    assert cnt[1, tiny_groups["mn9"]].sum() == 0
    # everything upstream identical (same seed)
    assert np.array_equal(cnt[0, sugar], cnt[1, sugar])


def test_silenced_neurons_send_nothing(tiny, tiny_groups):
    """Silencing the SEZ relay must cut MN9 (no output from silenced neurons)."""
    sugar = tiny_groups["sugar_grn_R"]
    relay = np.flatnonzero(tiny.neurons.type == "SEZ_int")
    b = Brain(tiny, 2, seeds=[1, 1], propagate="gather")
    for f in range(2):
        b.set_rates(f, sugar, 100.0)
    b.commit_rates()
    b.set_silence(1, relay)
    b.run(5000)
    cnt = b.take_counts()
    assert cnt[0, tiny_groups["mn9"]].sum() > 50
    assert cnt[1, tiny_groups["mn9"]].sum() == 0


def test_gain_override_scales_input(tiny, tiny_groups):
    sugar = tiny_groups["sugar_grn_R"]
    b = Brain(tiny, 2, seeds=[1, 1], propagate="gather")
    for f in range(2):
        b.set_rates(f, sugar, 100.0)
    b.commit_rates()
    b.set_gain(1, 0.05)
    b.run(5000)
    cnt = b.take_counts()
    assert cnt[1, tiny_groups["mn9"]].sum() < cnt[0, tiny_groups["mn9"]].sum()
