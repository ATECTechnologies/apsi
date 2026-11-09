"""Adaptive ODE integration (Dormand–Prince 5(4), the method behind ode45/RK45)."""
import numpy as np

from .errors import RuntimeErr

# Dormand–Prince coefficients
C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
Acoef = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
E = [b5 - b4 for b5, b4 in zip(B5, B4)]


def _raw(x):
    from .units import Quantity
    return np.asarray(x.v if isinstance(x, Quantity) else x, dtype=float)


def rk45(deriv, t0, t1, h0, state, tol, on_step=None, max_steps=10_000_000):
    """Integrate from t0 to t1 with error control `tol` (relative, per component).

    deriv(t, state_list) -> list of derivatives. Values may be plain numbers,
    tensors or Quantities (SI magnitudes are used for the error norm).
    Returns (final_state, accepted_steps, rejected_steps).
    """
    t, h, st = t0, h0, list(state)
    span_sign = 1.0 if float(_raw(t1 - t0)) >= 0 else -1.0
    scale0 = [np.maximum(np.abs(_raw(v)), 1e-300) for v in st]
    k1 = deriv(t, st)
    accepted = rejected = 0
    while float(_raw(t1 - t)) * span_sign > 1e-15 * float(np.abs(_raw(t1)) + np.abs(_raw(h))):
        if float(_raw(t + h - t1)) * span_sign > 0:
            h = t1 - t
        ks = [k1]
        for i in range(1, 7):
            yi = [y + h * sum_k(Acoef[i], ks, j) for j, y in enumerate(st)]
            ks.append(deriv(t + C[i] * h, yi))
        y5 = [y + h * sum_k(B5, ks, j) for j, y in enumerate(st)]
        errs = []
        for j, (y, yn) in enumerate(zip(st, y5)):
            e = _raw(h * sum_k(E, ks, j))
            sc = tol * np.maximum(np.maximum(np.abs(_raw(y)), np.abs(_raw(yn))), scale0[j] * 1e-3)
            errs.append(float(np.max(np.abs(e) / sc)) if np.size(e) else 0.0)
        err = max(errs) if errs else 0.0
        if err <= 1.0:
            t = t + h
            st = y5
            k1 = ks[6]                     # FSAL: last stage equals first stage of next step
            accepted += 1
            if on_step is not None:
                on_step(t, st)
            factor = 5.0 if err == 0 else min(5.0, max(0.2, 0.9 * err ** -0.2))
        else:
            rejected += 1
            if not np.isfinite(err):
                factor = 0.1
            else:
                factor = max(0.1, 0.9 * err ** -0.25)
        h = h * factor
        if accepted + rejected > max_steps:
            raise RuntimeErr("rk45: too many steps (is the system stiff, or the tolerance too strict?)")
        if abs(float(_raw(h))) < 1e-300:
            raise RuntimeErr("rk45: step size underflow")
    return st, accepted, rejected


def sum_k(coefs, ks, j):
    total = None
    for c, k in zip(coefs, ks):
        if c == 0:
            continue
        term = c * k[j]
        total = term if total is None else total + term
    return total
