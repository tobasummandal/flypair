"""Simple coupling metrics between two flies' rate time series.

xcorr_max     : max |normalized cross-correlation| over lags within +/- max_lag ticks
                (positive lag = the second series follows the first)
te_lite       : "transfer-entropy-lite" = fractional variance reduction of B_t from adding
                A's past (k lags) to B's own past (linear Granger-style). 0 = A's past adds nothing.
"""
from __future__ import annotations

import numpy as np


def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def xcorr_max(a, b, max_lag: int = 25):
    a, b = _z(a), _z(b)
    n = len(a)
    best, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:                 # positive lag: b follows a by `lag` ticks
            x, y = a[:n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:n + lag]
        if len(x) < 5:
            continue
        r = float(np.mean(x * y))
        if abs(r) > abs(best):
            best, best_lag = r, lag
    return best, best_lag


def te_lite(source, target, k: int = 5) -> float:
    s, t = _z(source), _z(target)
    n = len(t)
    if n <= 2 * k + 5:
        return 0.0
    Y = t[k:]
    X_self = np.column_stack([t[k - i - 1:n - i - 1] for i in range(k)])
    X_full = np.column_stack([X_self] + [s[k - i - 1:n - i - 1] for i in range(k)])
    def resid(X):
        X1 = np.column_stack([np.ones(len(Y)), X])
        beta, *_ = np.linalg.lstsq(X1, Y, rcond=None)
        return float(np.mean((Y - X1 @ beta) ** 2))
    v_self, v_full = resid(X_self), resid(X_full)
    return 0.0 if v_self <= 1e-12 else max(0.0, 1.0 - v_full / v_self)


def coupling_report(runs: dict, fly_a: str, fly_b: str, groups=("p1", "pip10")) -> str:
    """runs: {'live': Run, 'playback': Run, 'shuffled': Run, 'open_loop': Run}."""
    lines = [f"Coupling metrics between {fly_a} and {fly_b} (rates per world tick)",
             f"{'run':10s} {'signal':22s} {'xcorr':>7s} {'lag':>5s} {'TE A->B':>8s} {'TE B->A':>8s}"]
    for name, run in runs.items():
        for g in groups:
            col = f"rate_{g}"
            fa, fb = run.fly_frames(fly_a), run.fly_frames(fly_b)
            if col not in fa.columns or col not in fb.columns or fa[col].isna().all() or fb[col].isna().all():
                lines.append(f"{name:10s} {g:22s} {'n/a':>7s}")
                continue
            a, b = fa[col].to_numpy(), fb[col].to_numpy()
            r, lag = xcorr_max(a, b)
            lines.append(f"{name:10s} {g:22s} {r:7.3f} {lag:5d} {te_lite(a, b):8.3f} {te_lite(b, a):8.3f}")
    lines.append("Read: live >> playback and live >> shuffled suggests genuine bidirectional coupling via wiring;\n"
                 "      live ~ playback means B just follows A (one-way); live ~ shuffled means the wiring is not what matters.")
    return "\n".join(lines)
