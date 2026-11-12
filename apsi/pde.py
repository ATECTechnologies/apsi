"""`use pde` — partial differential equations and large linear systems.

Grid solvers (uniform grids, SI or any consistent units):
  heat1d(u0, dx, dt, steps, alpha, bc, left, right, every)    Crank–Nicolson (unconditionally stable)
  heat2d(u0, dx, dt, steps, alpha, every)                     ADI (Peaceman–Rachford), fixed boundary values
  wave1d(u0, v0, c, dx, dt, steps, bc, every)                 leapfrog; bc "fixed", "free" or "periodic"
  advection1d(u0, c, dx, dt, steps, scheme, every)            "upwind" or "lax-wendroff", periodic
  poisson2d(f, dx, bc)                                        ∇²u = f, fast sine/Fourier transforms (exact
                                                              solution of the 5-point scheme)
  navier_stokes2d(omega0, nu, length, dt, steps, every)       incompressible, periodic, pseudo-spectral
Iterative solvers (A can be a matrix or a function x -> A·x):
  cg(A, b, tol, max_iter)        symmetric positive definite
  bicgstab(A, b, tol, max_iter)  general square systems
  tridiag_solve(a, b, c, d)      Thomas algorithm (batched)
"""
import math

import numpy as np

from . import ops
from .errors import RuntimeErr
from .units import Quantity
from .values import Record


def _v(x):
    if isinstance(x, Quantity):
        x = x.v
    return np.asarray(ops.raw(x), dtype=float)


def _s(x):
    return float(_v(x))


def tridiag_solve(a, b, c, d):
    """Solve tridiagonal systems: a sub-diagonal (a[0] unused), b diagonal, c super-diagonal (c[-1] unused).
    d may have extra trailing dimensions (several right-hand sides solved at once)."""
    a, b, c, d = _v(a), _v(b), _v(c), _v(d)
    n = b.shape[0]
    cp = np.empty_like(b)
    dp = np.empty_like(d)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty_like(dp)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _periodic_tridiag(a, b, c, d):
    """Cyclic tridiagonal system (Sherman–Morrison)."""
    n = b.size
    gamma = -b[0]
    bb = b.copy()
    bb[0] -= gamma
    bb[-1] -= c[-1] * a[0] / gamma
    x = tridiag_solve(a, bb, c, d)
    u = np.zeros(n)
    u[0], u[-1] = gamma, c[-1]
    z = tridiag_solve(a, bb, c, u)
    fact = (x[0] + a[0] * x[-1] / gamma) / (1 + z[0] + a[0] * z[-1] / gamma)
    return x - fact * z


# =================================================================== diffusion
def heat1d(u0, dx, dt, steps, alpha, bc="dirichlet", left=None, right=None, every=1):
    """u_t = α u_xx by Crank–Nicolson. bc: "dirichlet" (end values fixed at left/right, default: the initial
    end values), "neumann" (zero flux) or "periodic". Returns record(t, x, u) with u[frame, i]."""
    u = _v(u0).copy()
    h, k, n, al = _s(dx), _s(dt), int(steps), _s(alpha)
    N = u.size
    r = al * k / (h * h)
    every = max(1, int(every))
    if bc == "dirichlet":
        uL = u[0] if left is None else _s(left)
        uR = u[-1] if right is None else _s(right)
        u[0], u[-1] = uL, uR
    a = np.full(N, -r / 2)
    b = np.full(N, 1 + r)
    c = np.full(N, -r / 2)
    if bc == "dirichlet":
        b[0] = b[-1] = 1.0
        c[0] = a[-1] = 0.0
    elif bc == "neumann":
        c[0] = -r
        a[-1] = -r
    elif bc != "periodic":
        raise RuntimeErr('bc must be "dirichlet", "neumann" or "periodic"')
    frames, ts = [u.copy()], [0.0]
    for i in range(n):
        if bc == "periodic":
            lap = np.roll(u, 1) - 2 * u + np.roll(u, -1)
            u = _periodic_tridiag(a, b, c, u + r / 2 * lap)
        else:
            rhs = u.copy()
            rhs[1:-1] = u[1:-1] + r / 2 * (u[:-2] - 2 * u[1:-1] + u[2:])
            if bc == "neumann":
                rhs[0] = u[0] + r * (u[1] - u[0])
                rhs[-1] = u[-1] + r * (u[-2] - u[-1])
            u = tridiag_solve(a, b, c, rhs)
        if (i + 1) % every == 0:
            frames.append(u.copy())
            ts.append((i + 1) * k)
    return Record(t=np.asarray(ts), x=np.arange(N) * h, u=np.asarray(frames))


def heat2d(u0, dx, dt, steps, alpha, every=1):
    """u_t = α ∇²u on a square grid by the ADI method (unconditionally stable, second order).
    Boundary values stay at their initial values. Returns record(t, u) with u[frame, i, j]."""
    u = _v(u0).copy()
    if u.ndim != 2:
        raise RuntimeErr("heat2d needs a 2-D initial field")
    h, k, n, al = _s(dx), _s(dt), int(steps), _s(alpha)
    r = al * k / (2 * h * h)
    every = max(1, int(every))

    def implicit(rhs):                                   # (1 − r δ²) along axis 0, end values kept
        N = rhs.shape[0]
        a = np.full(N, -r)
        b = np.full(N, 1 + 2 * r)
        c = np.full(N, -r)
        b[0] = b[-1] = 1.0
        c[0] = a[-1] = 0.0
        return tridiag_solve(a, b, c, rhs)

    def fix(w, ref):
        w[0, :], w[-1, :], w[:, 0], w[:, -1] = ref[0, :], ref[-1, :], ref[:, 0], ref[:, -1]
        return w

    frames, ts = [u.copy()], [0.0]
    for i in range(n):
        rhs = u.copy()                                    # half step 1: implicit in x, explicit in y
        rhs[1:-1, :] = u[1:-1, :] + r * (u[:-2, :] - 2 * u[1:-1, :] + u[2:, :])
        half = fix(implicit(fix(rhs, u).T).T, u)
        rhs = half.copy()                                 # half step 2: implicit in y, explicit in x
        rhs[:, 1:-1] = half[:, 1:-1] + r * (half[:, :-2] - 2 * half[:, 1:-1] + half[:, 2:])
        u = fix(implicit(fix(rhs, u)), u)
        if (i + 1) % every == 0:
            frames.append(u.copy())
            ts.append((i + 1) * k)
    return Record(t=np.asarray(ts), u=np.asarray(frames))


# =================================================================== waves and transport
def wave1d(u0, v0, c, dx, dt, steps, bc="fixed", every=1):
    """u_tt = c² u_xx by the leapfrog scheme (needs c·dt ≤ dx). bc: "fixed", "free" or "periodic".
    Returns record(t, x, u)."""
    u = _v(u0).copy()
    v = _v(v0) * np.ones_like(u)
    cc, h, k, n = _s(c), _s(dx), _s(dt), int(steps)
    C2 = (cc * k / h) ** 2
    if C2 > 1 + 1e-12:
        raise RuntimeErr(f"unstable: c·dt/dx = {math.sqrt(C2):.3g} > 1 (use a smaller dt)")
    every = max(1, int(every))

    def lap(w):
        if bc == "periodic":
            return np.roll(w, 1) - 2 * w + np.roll(w, -1)
        L = np.zeros_like(w)
        L[1:-1] = w[:-2] - 2 * w[1:-1] + w[2:]
        if bc == "free":
            L[0] = 2 * (w[1] - w[0])
            L[-1] = 2 * (w[-2] - w[-1])
        return L
    if bc not in ("fixed", "free", "periodic"):
        raise RuntimeErr('bc must be "fixed", "free" or "periodic"')
    prev = u.copy()
    cur = u + k * v + 0.5 * C2 * lap(u)
    if bc == "fixed":
        cur[0], cur[-1] = u[0], u[-1]
    frames, ts = [u.copy()], [0.0]
    for i in range(1, n + 1):
        if i % every == 0:
            frames.append(cur.copy())
            ts.append(i * k)
        nxt = 2 * cur - prev + C2 * lap(cur)
        if bc == "fixed":
            nxt[0], nxt[-1] = u[0], u[-1]
        prev, cur = cur, nxt
    return Record(t=np.asarray(ts), x=np.arange(u.size) * h, u=np.asarray(frames))


def advection1d(u0, c, dx, dt, steps, scheme="lax-wendroff", every=1):
    """u_t + c u_x = 0 on a periodic grid ("upwind": first order, monotone; "lax-wendroff": second order)."""
    u = _v(u0).copy()
    cc, h, k, n = _s(c), _s(dx), _s(dt), int(steps)
    nu = cc * k / h
    if abs(nu) > 1 + 1e-12:
        raise RuntimeErr(f"unstable: |c|·dt/dx = {abs(nu):.3g} > 1")
    every = max(1, int(every))
    frames, ts = [u.copy()], [0.0]
    for i in range(n):
        if scheme == "upwind":
            u = u - nu * (u - np.roll(u, 1)) if nu >= 0 else u - nu * (np.roll(u, -1) - u)
        elif scheme == "lax-wendroff":
            up, um = np.roll(u, -1), np.roll(u, 1)
            u = u - nu / 2 * (up - um) + nu * nu / 2 * (up - 2 * u + um)
        else:
            raise RuntimeErr('scheme must be "upwind" or "lax-wendroff"')
        if (i + 1) % every == 0:
            frames.append(u.copy())
            ts.append((i + 1) * k)
    return Record(t=np.asarray(ts), x=np.arange(u.size) * h, u=np.asarray(frames))


# =================================================================== Poisson
def _dst1(a, axis):
    """Type-I discrete sine transform along an axis (via a real FFT of the odd extension)."""
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    ext = np.zeros(a.shape[:-1] + (2 * (n + 1),))
    ext[..., 1:n + 1] = a
    ext[..., n + 2:] = -a[..., ::-1]
    r = -np.fft.rfft(ext, axis=-1).imag[..., 1:n + 1] / 2
    return np.moveaxis(r, -1, axis)


def poisson2d(f, dx, bc="dirichlet"):
    """Solve ∇²u = f on a uniform grid with the 5-point Laplacian.
    "dirichlet": u = 0 on the boundary (f gives the interior points); "periodic": f must have zero mean,
    the solution has zero mean. Fast transforms: O(N log N), exact for the discrete problem."""
    F = _v(f)
    h = _s(dx)
    if F.ndim != 2:
        raise RuntimeErr("poisson2d needs a 2-D right-hand side")
    ny, nx = F.shape
    if bc == "dirichlet":
        lx = 2 * (np.cos(np.pi * np.arange(1, nx + 1) / (nx + 1)) - 1)
        ly = 2 * (np.cos(np.pi * np.arange(1, ny + 1) / (ny + 1)) - 1)
        Fh = _dst1(_dst1(F, 1), 0)
        U = Fh * h * h / (ly[:, None] + lx[None, :])
        return _dst1(_dst1(U, 1), 0) * (2 / (nx + 1)) * (2 / (ny + 1))
    if bc == "periodic":
        Fh = np.fft.fft2(F)
        kx = 2 * (np.cos(2 * np.pi * np.arange(nx) / nx) - 1)
        ky = 2 * (np.cos(2 * np.pi * np.arange(ny) / ny) - 1)
        den = ky[:, None] + kx[None, :]
        den[0, 0] = 1.0
        U = Fh * h * h / den
        U[0, 0] = 0.0
        return np.real(np.fft.ifft2(U))
    raise RuntimeErr('bc must be "dirichlet" or "periodic"')


# =================================================================== Navier–Stokes
def navier_stokes2d(omega0, nu, length=2 * math.pi, dt=1e-3, steps=1000, every=0):
    """2-D incompressible Navier–Stokes in vorticity form ω_t + u·∇ω = ν∇²ω on a periodic square, pseudo-spectral
    with 2/3 de-aliasing and RK4 (viscous term integrated exactly). Returns record(t, omega, u, v, energy) for
    the stored frames (every=0 stores only the final state)."""
    w = _v(omega0).copy()
    n = w.shape[0]
    if w.shape != (n, n):
        raise RuntimeErr("navier_stokes2d needs a square vorticity field")
    Lx, nuv, h, N = _s(length), _s(nu), _s(dt), int(steps)
    k = np.fft.fftfreq(n, d=Lx / (2 * math.pi * n))
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K2 = KX ** 2 + KY ** 2
    K2i = np.where(K2 == 0, 0, 1 / np.where(K2 == 0, 1, K2))
    kmax = np.max(np.abs(k))
    mask = (np.abs(KX) < 2 / 3 * kmax) & (np.abs(KY) < 2 / 3 * kmax)
    E = np.exp(-nuv * K2 * h / 2)

    def vel(wh):
        psi = wh * K2i
        return np.real(np.fft.ifft2(1j * KY * psi)), np.real(np.fft.ifft2(-1j * KX * psi))

    def rhs(wh):
        u, v = vel(wh)
        wx = np.real(np.fft.ifft2(1j * KX * wh))
        wy = np.real(np.fft.ifft2(1j * KY * wh))
        return -np.fft.fft2(u * wx + v * wy) * mask

    wh = np.fft.fft2(w)
    every = int(every)
    out_t, out_w = [], []

    def store(t, wh):
        out_t.append(t)
        out_w.append(np.real(np.fft.ifft2(wh)))
    store(0.0, wh)
    for i in range(N):                                   # integrating-factor RK4
        k1 = rhs(wh)
        k2 = rhs(E * (wh + h / 2 * k1))
        k3 = rhs(E * wh + h / 2 * k2)
        k4 = rhs(E * E * wh + h * E * k3)
        wh = E * E * wh + h / 6 * (E * E * k1 + 2 * E * (k2 + k3) + k4)
        if every and (i + 1) % every == 0 and i + 1 < N:
            store((i + 1) * h, wh)
    store(N * h, wh)
    u, v = vel(wh)
    en = [0.5 * float(np.mean(sum(c ** 2 for c in vel(np.fft.fft2(ww))))) for ww in out_w]
    return Record(t=np.asarray(out_t), omega=np.asarray(out_w), u=u, v=v, energy=np.asarray(en))


# =================================================================== Krylov solvers
def _op(A):
    if callable(A):
        return lambda x: _v(A(x))
    M = _v(A)
    return lambda x: M @ x


def cg(A, b, tol=1e-10, max_iter=None, x0=None):
    """Conjugate gradients for symmetric positive-definite A. Returns record(x, iterations, residual, converged)."""
    Ax = _op(A)
    bb = _v(b)
    x = np.zeros_like(bb) if x0 is None else _v(x0).copy()
    r = bb - Ax(x)
    p = r.copy()
    rs = float(r @ r)
    nb = float(np.linalg.norm(bb)) or 1.0
    it = 0
    mi = int(max_iter) if max_iter else 10 * bb.size
    while math.sqrt(rs) > tol * nb and it < mi:
        Ap = Ax(p)
        al = rs / float(p @ Ap)
        x += al * p
        r -= al * Ap
        rn = float(r @ r)
        p = r + rn / rs * p
        rs = rn
        it += 1
    return Record(x=x, iterations=it, residual=math.sqrt(rs) / nb, converged=math.sqrt(rs) <= tol * nb)


def bicgstab(A, b, tol=1e-10, max_iter=None, x0=None):
    """BiCGSTAB for general square A. Returns record(x, iterations, residual, converged)."""
    Ax = _op(A)
    bb = _v(b)
    x = np.zeros_like(bb) if x0 is None else _v(x0).copy()
    r = bb - Ax(x)
    rh = r.copy()
    rho = al = om = 1.0
    v = p = np.zeros_like(bb)
    nb = float(np.linalg.norm(bb)) or 1.0
    mi = int(max_iter) if max_iter else 10 * bb.size
    it = 0
    while np.linalg.norm(r) > tol * nb and it < mi:
        rho1 = float(rh @ r)
        if rho1 == 0:
            break
        beta = rho1 / rho * al / om
        p = r + beta * (p - om * v)
        v = Ax(p)
        al = rho1 / float(rh @ v)
        s = r - al * v
        if np.linalg.norm(s) <= tol * nb:
            x += al * p
            r = s
            it += 1
            break
        t = Ax(s)
        om = float(t @ s) / float(t @ t)
        x += al * p + om * s
        r = s - om * t
        rho = rho1
        it += 1
    res = float(np.linalg.norm(r)) / nb
    return Record(x=x, iterations=it, residual=res, converged=res <= tol)


def module():
    return {k: v for k, v in globals().items()
            if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__ and k != "module"}
