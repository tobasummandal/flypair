import numpy as np

from flypair.metrics import te_lite, xcorr_max


def test_xcorr_finds_lag():
    rng = np.random.default_rng(0)
    a = rng.random(300)
    b = np.roll(a, 3) + 0.01 * rng.random(300)
    r, lag = xcorr_max(a, b, max_lag=10)
    assert r > 0.9 and lag == 3


def test_te_lite_detects_driver():
    rng = np.random.default_rng(1)
    a = rng.random(500)
    b = np.zeros(500)
    for t in range(2, 500):
        b[t] = 0.5 * b[t - 1] + 0.8 * a[t - 1] + 0.05 * rng.random()
    assert te_lite(a, b) > 0.5
    assert te_lite(b, a) < 0.2
