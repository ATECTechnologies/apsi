"""`use sci` — scientific computing: einsum, FFT, integration, roots, optimisation,
curve fitting with correlated uncertainties, interpolation.

Everything is unit-aware and (where it makes sense) differentiable.
"""
import math

import numpy as np

from . import autodiff as ad
from . import ops
from .autodiff import Node
from .errors import RuntimeErr, UnitError, ShapeError
from .uncertain import Uncertain, _ids
from .units import Quantity, make, DIMLESS, dim_name
from .values import tree_map, tree_leaves


# ------------------------------------------------------------------ einsum
def _parse_spec(spec, n):
    spec = spec.replace(" ", "")
    if "->" not in spec:
        ins = spec.split(",")
        letters = [c for c in "".join(ins)]
        out = "".join(sorted(c for c in set(letters) if letters.count(c) == 1))
    else:
        lhs, out = spec.split("->")
        ins = lhs.split(",")
    if len(ins) != n:
        raise ShapeError(f"einsum spec has {len(ins)} operands but {n} were given")
    if any("." in s for s in ins):
        raise ShapeError("einsum: '...' is not supported yet")
    return ins, out


def _ad_einsum(ins, out, ops_):
    spec = ",".join(ins) + "->" + out
    shapes = [np.shape(ad._v(o)) for o in ops_]

    def vjp(g, outv, need, *xs):
        res = []
        for i in range(len(xs)):
            if not need[i]:
                res.append(None)
                continue
            others = [xs[j] for j in range(len(xs)) if j != i]
            subs = [out] + [ins[j] for j in range(len(xs)) if j != i]
            target = ins[i]
            # indices of operand i that appear nowhere else must be broadcast back
            avail = set("".join(subs))
            missing = [c for c in target if c not in avail]
            if missing:
                tmp_target = "".join(c for c in target if c in avail)
                r = einsum(",".join(subs) + "->" + tmp_target, g, *others)
                shp = list(shapes[i])
                kd = [1 if c in missing else shp[k] for k, c in enumerate(target)]
                r = ad.reshape(r, tuple(kd)) if isinstance(r, Node) else np.reshape(r, kd)
                r = ad.broadcast_to(r, tuple(shp))
            else:
                r = einsum(",".join(subs) + "->" + target, g, *others)
            res.append(r)
        return tuple(res)
    return ad._op(lambda *vals: np.einsum(spec, *vals), vjp, list(ops_))


def einsum(spec, *operands):
    """Einstein summation, e.g. einsum("bij,bjk->bik", A, B). Units multiply; differentiable."""
    ins, out = _parse_spec(spec, len(operands))
    dim = DIMLESS
    raws = []
    for o in operands:
        o = ops._arr(o)
        if isinstance(o, Quantity):
            dim = dim * o.dim
            raws.append(o.v)
        else:
            raws.append(o)
    if any(isinstance(r, Node) for r in raws):
        r = _ad_einsum(ins, out, raws)
    else:
        r = np.einsum(",".join(ins) + "->" + out, *[np.asarray(x) for x in raws])
    return make(ops.py(r), dim)


# ------------------------------------------------------------------ FFT
def _wrap(r, like):
    return Quantity(r, like.dim) if isinstance(like, Quantity) else r


def fft(x, axis=-1):
    return _wrap(np.fft.fft(ops.raw(x), axis=axis), x)


def ifft(x, axis=-1):
    return _wrap(np.fft.ifft(ops.raw(x), axis=axis), x)


def rfft(x, axis=-1):
    return _wrap(np.fft.rfft(ops.raw(x), axis=axis), x)


def irfft(x, n=None, axis=-1):
    return _wrap(np.fft.irfft(ops.raw(x), n=n, axis=axis), x)


def fftfreq(n, d=1.0):
    r = np.fft.fftfreq(int(n), ops.raw(d))
    return Quantity(r, DIMLESS / d.dim) if isinstance(d, Quantity) else r


def rfftfreq(n, d=1.0):
    r = np.fft.rfftfreq(int(n), ops.raw(d))
    return Quantity(r, DIMLESS / d.dim) if isinstance(d, Quantity) else r


def power_spectrum(x, d=1.0):
    """One-sided power spectrum |rfft|² with matching frequencies -> (freqs, power)."""
    n = np.shape(ops.raw(x))[-1]
    spec = np.abs(np.fft.rfft(ops.raw(x))) ** 2
    p = Quantity(spec, x.dim ** 2) if isinstance(x, Quantity) else spec
    return rfftfreq(n, d), p


def real(x):
    return _wrap(np.real(ops.raw(x)), x)


def imag(x):
    return _wrap(np.imag(ops.raw(x)), x)


def conj(x):
    return _wrap(np.conj(ops.raw(x)), x)


# ------------------------------------------------------------------ integration
def trapz(y, x=None, dx=None):
    """Trapezoidal integral of samples y over x (or uniform spacing dx). Units: [y]·[x]."""
    y = ops._arr(y)
    if x is not None:
        x = ops._arr(x)
        return ops.sum_((y[1:] + y[:-1]) * (x[1:] - x[:-1])) * 0.5
    dx = 1.0 if dx is None else dx
    return (ops.sum_(y) - 0.5 * (y[0] + y[-1])) * dx


_GL = {}


def integrate(f, a, b, points=64):
    """∫_a^b f(x) dx by Gauss–Legendre quadrature. f receives a tensor of points.
    Exact for polynomials up to degree 2·points−1; units are [f]·[x]."""
    n = int(points)
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    xs, ws = _GL[n]
    half = (b - a) * 0.5
    mid = (b + a) * 0.5
    pts = mid + half * xs
    vals = f(pts)
    return ops.sum_(vals * ws) * half


def cumtrapz(y, x):
    y, x = ops._arr(y), ops._arr(x)
    seg = (y[1:] + y[:-1]) * (x[1:] - x[:-1]) * 0.5
    ry = ops.raw(seg)
    out = np.concatenate([[0.0], np.cumsum(ry)])
    return Quantity(out, seg.dim) if isinstance(seg, Quantity) else out


# ------------------------------------------------------------------ interpolation
def interp(x, xp, fp):
    """Piecewise-linear interpolation (units of xp and x must match)."""
    if ops.dim_of(x) != ops.dim_of(xp):
        raise UnitError(f"interp: x is {dim_name(ops.dim_of(x))} but xp is {dim_name(ops.dim_of(xp))}")
    r = np.interp(ops.raw(x), ops.raw(xp), ops.raw(fp))
    return make(ops.py(r), ops.dim_of(fp))


# ------------------------------------------------------------------ roots
def find_root(f, x0, tol=1e-12, max_iter=100):
    """Newton's method with the exact derivative from autodiff (unit-aware)."""
    from .stdlib import value_and_grad
    vg = value_and_grad(f)
    x = x0
    for _ in range(int(max_iter)):
        fx, d = vg(x)
        if ops.raw(d) == 0:
            raise RuntimeErr("find_root: zero derivative; try another starting point")
        step = fx / d
        x = x - step
        if abs(float(ops.raw(step))) <= tol * max(1.0, abs(float(ops.raw(x)))):
            return x
    raise RuntimeErr("find_root did not converge")


def bisect(f, a, b, tol=1e-12, max_iter=200):
    fa = f(a)
    if float(ops.raw(fa)) * float(ops.raw(f(b))) > 0:
        raise RuntimeErr("bisect: f(a) and f(b) must have opposite signs")
    for _ in range(int(max_iter)):
        m = (a + b) * 0.5
        fm = f(m)
        if float(ops.raw(fm)) * float(ops.raw(fa)) <= 0:
            b = m
        else:
            a, fa = m, fm
        if abs(float(ops.raw(b - a))) <= tol * max(1.0, abs(float(ops.raw(m)))):
            break
    return (a + b) * 0.5


# ------------------------------------------------------------------ optimisation
def _flatten(tree):
    leaves = tree_leaves(tree)
    numeric = [i for i, l in enumerate(leaves) if ops.dim_of(l) is not None]
    raws = [np.asarray(ops.raw(leaves[i]), dtype=float) for i in numeric]
    vec = np.concatenate([r.ravel() for r in raws]) if raws else np.zeros(0)

    def unflatten(v):
        pos = 0
        out = list(leaves)
        for i, r in zip(numeric, raws):
            chunk = v[pos:pos + r.size].reshape(r.shape)
            pos += r.size
            orig = leaves[i]
            val = chunk if r.ndim else float(chunk)
            out[i] = Quantity(val, orig.dim, orig.disp) if isinstance(orig, Quantity) else val
        it = iter(out)
        return tree_map(lambda _: next(it), tree)
    return vec, unflatten


def minimize(f, x0, method="lbfgs", steps=500, tol=1e-10, lr=0.01, history=10, verbose=False):
    """Minimise a scalar function of a parameter tree using exact gradients.

    method: "lbfgs" (default, quasi-Newton with line search) or "adam".
    Returns record(x, value, steps, converged).
    """
    from .stdlib import value_and_grad
    from .values import Record
    vg = value_and_grad(f)
    vec, unflat = _flatten(x0)

    def fg(v):
        val, g = vg(unflat(v))
        gv, _ = _flatten(g)
        return float(ops.raw(val)), gv

    if method == "adam":
        m = np.zeros_like(vec)
        s = np.zeros_like(vec)
        val = None
        for t in range(1, int(steps) + 1):
            val, g = fg(vec)
            m = 0.9 * m + 0.1 * g
            s = 0.999 * s + 0.001 * g * g
            vec = vec - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(s / (1 - 0.999 ** t)) + 1e-12)
            if np.linalg.norm(g) < tol:
                return Record(x=unflat(vec), value=val, steps=t, converged=True)
        return Record(x=unflat(vec), value=fg(vec)[0], steps=int(steps), converged=False)

    if method != "lbfgs":
        raise RuntimeErr("minimize: method must be \"lbfgs\" or \"adam\"")
    val, g = fg(vec)
    S, Y = [], []
    for it in range(int(steps)):
        gnorm = np.linalg.norm(g)
        if gnorm < tol:
            return Record(x=unflat(vec), value=val, steps=it, converged=True)
        q = g.copy()
        alphas = []
        for s_, y_ in reversed(list(zip(S, Y))):
            rho = 1.0 / np.dot(y_, s_)
            a = rho * np.dot(s_, q)
            alphas.append((rho, a, s_, y_))
            q -= a * y_
        if Y:
            q *= np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1])
        else:
            q *= min(1.0, 1.0 / max(gnorm, 1e-300))
        for rho, a, s_, y_ in reversed(alphas):
            b = rho * np.dot(y_, q)
            q += s_ * (a - b)
        d = -q
        if np.dot(d, g) >= 0:
            d = -g
            S, Y = [], []
        step = 1.0
        for _ in range(40):                        # Armijo backtracking
            nv, ng = fg(vec + step * d)
            if np.isfinite(nv) and nv <= val + 1e-4 * step * np.dot(g, d):
                break
            step *= 0.5
        else:
            return Record(x=unflat(vec), value=val, steps=it, converged=False)
        s_, y_ = step * d, ng - g
        vec, val, g = vec + s_, nv, ng
        if np.dot(s_, y_) > 1e-12:
            S.append(s_)
            Y.append(y_)
            if len(S) > history:
                S.pop(0)
                Y.pop(0)
        if verbose and it % 50 == 0:
            print(f"[minimize] step {it}  value {val:.6g}  |grad| {gnorm:.3g}")
        if abs(np.dot(s_, g)) < tol * tol and np.linalg.norm(s_) < tol * (1 + np.linalg.norm(vec)):
            return Record(x=unflat(vec), value=val, steps=it + 1, converged=True)
    return Record(x=unflat(vec), value=val, steps=int(steps), converged=False)


# ------------------------------------------------------------------ curve fitting
def curve_fit(model, x, y, p0, sigma=None, max_iter=200, tol=1e-12):
    """Levenberg–Marquardt least squares with an exact autodiff Jacobian.

    model(x, p) -> predictions, p is a list of parameters (units allowed).
    Returns record(params, cov, chi2, dof) where params are Uncertain values whose
    correlations follow the covariance matrix exactly.
    """
    from .stdlib import jacobian
    from .values import Record
    p0 = list(p0)
    dims = [ops.dim_of(p) for p in p0]
    disp = [p.disp if isinstance(p, Quantity) else None for p in p0]
    pv = np.array([float(ops.raw(p)) for p in p0])
    ydim = ops.dim_of(y)
    yr = np.asarray(ops.raw(y), dtype=float)
    w = np.ones_like(yr) if sigma is None else 1.0 / np.asarray(ops.raw(sigma), dtype=float)

    def build(v):
        return [make(float(a), d) for a, d in zip(v, dims)]

    def resid(v):
        pred = model(x, build(v))
        if ops.dim_of(pred) != ydim:
            raise UnitError(f"curve_fit: model returns {dim_name(ops.dim_of(pred))} but y is {dim_name(ydim)}")
        return (np.asarray(ops.raw(pred), dtype=float) - yr) * w

    def J(v):
        def r_of(vec):
            pred = model(x, [make(vec[i], d) for i, d in enumerate(dims)])
            pr = pred.v if isinstance(pred, Quantity) else pred
            return (pr - yr) * w
        return np.asarray(ops.raw(jacobian(r_of)(np.asarray(v, dtype=float))), dtype=float)

    lam = 1e-3
    r = resid(pv)
    cost = float(r @ r)
    for _ in range(int(max_iter)):
        Jm = J(pv)
        A = Jm.T @ Jm
        gvec = Jm.T @ r
        improved = False
        for _ in range(30):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-300), -gvec)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            nv = pv + step
            nr = resid(nv)
            nc = float(nr @ nr)
            if np.isfinite(nc) and nc < cost:
                pv, r, lam, improved = nv, nr, max(lam / 10, 1e-15), True
                rel = (cost - nc) / max(cost, 1e-300)
                cost = nc
                break
            lam *= 10
        if not improved or rel < tol:
            break
    Jm = J(pv)
    dof = max(1, yr.size - pv.size)
    try:
        cov = np.linalg.inv(Jm.T @ Jm)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Jm.T @ Jm)
    if sigma is None:
        cov = cov * cost / dof
    # correlated uncertain parameters: p = μ + L z with independent unit sources z
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    vals = np.clip(vals, 0, None)
    L = vecs * np.sqrt(vals)
    src = [next(_ids) for _ in range(len(pv))]
    params = []
    for i, (mu, d) in enumerate(zip(pv, dims)):
        u = Uncertain(float(mu), {src[j]: (float(L[i, j]), 1.0) for j in range(len(pv)) if L[i, j] != 0})
        params.append(make(u, d, disp[i]))
    cov_q = cov
    return Record(params=params, cov=cov_q, chi2=cost, dof=dof)


# ------------------------------------------------------------------ initial value problems (stiff and non-stiff)
def _num(x):
    """Plain float array from numbers/tensors/quantities (SI values); also returns the unit to restore."""
    if isinstance(x, (list, tuple)):
        x = ops.tensor(list(x))
    if isinstance(x, Quantity):
        return np.atleast_1d(np.asarray(ops.raw(x.v), dtype=float)), x.dim
    return np.atleast_1d(np.asarray(ops.raw(x), dtype=float)), None


def solve_ivp(f, t_span, y0, method="bdf", rtol=1e-6, atol=1e-9, t_eval=None, first_step=None, max_steps=200000):
    """Solve y' = f(t, y) from t_span[0] to t_span[1].

    method "bdf"  : TR-BDF2 — implicit, L-stable, adaptive; for stiff problems (chemistry, circuits, diffusion)
    method "rk45" : Dormand–Prince — explicit, adaptive; for smooth non-stiff problems
    Returns record(t, y, steps, rejected, evaluations, success) where y has one row per time in t.
    With t_eval the solution is reported exactly at those times (cubic Hermite interpolation).
    Works on plain numbers and tensors (SI values for quantities).
    """
    t0, tu = _num(t_span[0])[0][0], _num(t_span[0])[1]
    t1 = _num(t_span[1])[0][0]
    y, yu = _num(y0)
    shape = np.shape(ops.raw(y0.v if isinstance(y0, Quantity) else y0))
    nfev = [0]

    def F(t, yy):
        nfev[0] += 1
        tt = make(t, tu) if tu is not None else t
        val = make(yy.reshape(shape) if len(shape) else float(yy[0]), yu) if yu is not None else (
            yy.reshape(shape) if len(shape) else float(yy[0]))
        out = f(tt, val)
        return np.atleast_1d(np.asarray(ops.raw(out.v if isinstance(out, Quantity) else out), dtype=float)).ravel()

    y = y.ravel().copy()
    direction = 1.0 if t1 >= t0 else -1.0
    span = abs(t1 - t0)
    h = abs(float(ops.raw(first_step))) if first_step is not None else max(span * 1e-4, 1e-12)
    ts, ys, fs = [t0], [y.copy()], [F(t0, y)]
    t, steps, rejected = t0, 0, 0

    def err_norm(e, a, b):
        sc = atol + rtol * np.maximum(np.abs(a), np.abs(b))
        return float(np.sqrt(np.mean((e / sc) ** 2)))

    with np.errstate(all="ignore"):         # rejected trial steps may overflow; they are discarded
        return _ivp_loop(F, t, t1, y, h, direction, method, rtol, atol, max_steps, ts, ys, fs, nfev, shape,
                         tu, yu, t_eval, err_norm)


def _ivp_loop(F, t, t1, y, h, direction, method, rtol, atol, max_steps, ts, ys, fs, nfev, shape, tu, yu,
              t_eval, err_norm):
    steps = rejected = 0
    if method == "rk45":
        A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
             [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
             [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
             [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
        C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
        B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
        B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
        k0 = fs[0]
        while direction * (t1 - t) > 1e-14 * max(1.0, abs(t1)):
            if steps + rejected > max_steps:
                break
            h = min(h, abs(t1 - t))
            K = [k0]
            for i in range(1, 7):
                yi = y + direction * h * sum(a * K[j] for j, a in enumerate(A[i]))
                K.append(F(t + direction * C[i] * h, yi))
            y5 = y + direction * h * sum(b * k for b, k in zip(B5, K))
            y4 = y + direction * h * sum(b * k for b, k in zip(B4, K))
            e = err_norm(y5 - y4, y, y5)
            if e <= 1.0:
                t, y, k0 = t + direction * h, y5, K[6]
                ts.append(t); ys.append(y.copy()); fs.append(k0)
                steps += 1
            else:
                rejected += 1
            h *= min(5.0, max(0.2, 0.9 * (1.0 / max(e, 1e-12)) ** 0.2))
    elif method == "bdf":
        g = 2.0 - math.sqrt(2.0)
        d = (1.0 - g) / (2.0 - g)
        c1 = 1.0 / (g * (2.0 - g))
        c0 = (1.0 - g) ** 2 / (g * (2.0 - g))
        n = y.size

        def jac(tt, yy, fy):
            J = np.empty((n, n))
            for j in range(n):
                dy = 1e-8 * max(1.0, abs(yy[j]))
                yp = yy.copy()
                yp[j] += dy
                J[:, j] = (F(tt, yp) - fy) / dy
            return J

        def newton(tt, rhs, c, guess, J):
            z = guess.copy()
            M = np.eye(n) - c * J
            for _ in range(10):
                r = z - c * F(tt, z) - rhs
                dz = np.linalg.solve(M, -r)
                z += dz
                if np.max(np.abs(dz) / (atol + rtol * np.abs(z))) < 1e-3:
                    return z, True
            return z, False

        fy = fs[0]
        J = jac(t, y, fy)
        while direction * (t1 - t) > 1e-14 * max(1.0, abs(t1)):
            if steps + rejected > max_steps:
                break
            h = min(h, abs(t1 - t))
            hh = direction * h
            # stage 1: trapezoidal rule to t + g h; stage 2: BDF2 to t + h
            yg, ok1 = newton(t + g * hh, y + g * hh / 2 * fy, g * hh / 2, y + g * hh * fy, J)
            yn, ok2 = (newton(t + hh, c1 * yg - c0 * y, d * hh, yg, J) if ok1 else (yg, False))
            if not (ok1 and ok2):
                rejected += 1
                h *= 0.25
                J = jac(t, y, fy)
                continue
            fn_ = F(t + hh, yn)
            # local error estimate of TR-BDF2 (Hosea & Shampine 1996), filtered through (I - d h J)
            fg = F(t + g * hh, yg)
            C = (-3 * g ** 2 + 4 * g - 2) / (6 * (2 - g))
            e_raw = C * hh * (fy / g - fg / (g * (1 - g)) + fn_ / (1 - g))
            e_vec = np.linalg.solve(np.eye(n) - d * hh * J, e_raw)
            e = err_norm(e_vec, y, yn)
            if e <= 1.0:
                t, y, fy = t + hh, yn, fn_
                ts.append(t); ys.append(y.copy()); fs.append(fy)
                steps += 1
                if steps % 5 == 0:
                    J = jac(t, y, fy)
            else:
                rejected += 1
                J = jac(t, y, fy)
            h *= min(4.0, max(0.2, 0.9 * (1.0 / max(e, 1e-12)) ** (1.0 / 3.0)))
    else:
        raise RuntimeErr("solve_ivp(): method must be \"bdf\" or \"rk45\"")
    success = direction * (t1 - t) <= 1e-12 * max(1.0, abs(t1)) + 1e-300
    T = np.array(ts)
    Y = np.array(ys)
    if t_eval is not None:
        te = _num(t_eval)[0]
        out = []
        for tq in te:
            i = int(np.clip(np.searchsorted(T * direction, tq * direction) - 1, 0, len(T) - 2)) if len(T) > 1 else 0
            if len(T) == 1:
                out.append(Y[0])
                continue
            h_ = T[i + 1] - T[i]
            s_ = (tq - T[i]) / h_
            h00, h10, h01, h11 = 2 * s_ ** 3 - 3 * s_ ** 2 + 1, s_ ** 3 - 2 * s_ ** 2 + s_, -2 * s_ ** 3 + 3 * s_ ** 2, s_ ** 3 - s_ ** 2
            out.append(h00 * Y[i] + h10 * h_ * fs[i] + h01 * Y[i + 1] + h11 * h_ * fs[i + 1])
        T, Y = te, np.array(out)
    Y = Y.reshape((len(T),) + shape)
    from .values import Record
    return Record(t=make(T, tu) if tu is not None else T, y=make(Y, yu) if yu is not None else Y,
                  steps=steps, rejected=rejected, evaluations=nfev[0], success=bool(success))


def module():
    return {"einsum": einsum, "fft": fft, "ifft": ifft, "rfft": rfft, "irfft": irfft,
            "fftfreq": fftfreq, "rfftfreq": rfftfreq, "power_spectrum": power_spectrum,
            "real": real, "imag": imag, "conj": conj,
            "trapz": trapz, "cumtrapz": cumtrapz, "integrate": integrate, "interp": interp,
            "find_root": find_root, "bisect": bisect, "minimize": minimize, "curve_fit": curve_fit,
            "solve_ivp": solve_ivp}
