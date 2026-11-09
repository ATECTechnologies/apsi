"""`use mechanics` — classical mechanics: integrators, Hamiltonian and Lagrangian dynamics (equations of motion
derived by automatic differentiation), N-body gravity, orbits, rigid bodies, vibrations and structures.

Inputs may be numbers, tensors or quantities; computations run in SI units and results are SI values.

Trajectories
  simulate(accel, x0, v0, dt, steps, method)   method: "verlet" (default), "yoshida4", "rk4"
  hamiltonian_flow(H, q0, p0, dt, steps, method) H(q, p); method "leapfrog" (separable H) or "rk4"
  lagrangian_flow(L, q0, qdot0, dt, steps)      L(q, qdot); Euler–Lagrange equations by autodiff
  nbody(masses, x0, v0, dt, steps, G, softening, method, every)
Orbits
  kepler_solve(M, e), orbit_elements(r, v, mu), orbit_state(a, e, i, raan, argp, nu, mu),
  orbital_period(a, mu), vis_viva(r, a, mu), hohmann(r1, r2, mu)
Rigid bodies
  inertia_tensor(masses, positions), principal_axes(I), rotation_matrix(axis, angle),
  quat(axis, angle), quat_mul, quat_conj, quat_rotate, quat_to_matrix, rigid_body(I, omega0, q0, dt, steps, torque)
Vibrations and structures
  normal_modes(K, M), beam(length, E, I, support, loads, distributed, n), truss2d(nodes, members, E, A, supports, loads)
"""
import inspect
import math

import numpy as np

from . import ops
from .errors import RuntimeErr
from .units import Quantity
from .values import Record

G_NEWTON = 6.67430e-11


def _v(x):
    """SI float array from a number, list, tensor or quantity."""
    if isinstance(x, Quantity):
        x = x.v
    if isinstance(x, (list, tuple)):
        x = [_v(e) for e in x]
    return np.asarray(ops.raw(x), dtype=float)


def _s(x):
    return float(_v(x))


def _arity(f, default):
    try:
        ps = [p for p in inspect.signature(f).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
        return len(ps)
    except (TypeError, ValueError):
        return default


def _call_accel(f):
    n = _arity(f, 3)
    if n >= 3:
        return lambda x, v, t: _v(f(x, v, t))
    if n == 2:
        return lambda x, v, t: _v(f(x, v))
    return lambda x, v, t: _v(f(x))


# =================================================================== generic second-order integration
_Y4 = (1 / (2 - 2 ** (1 / 3)), -2 ** (1 / 3) / (2 - 2 ** (1 / 3)))


def simulate(accel, x0, v0, dt, steps, method="verlet", t0=0.0, every=1):
    """Integrate x'' = accel(x, v, t) (accel may also take (x) or (x, v)).

    "verlet"   velocity Verlet: 2nd order, symplectic for position-only forces (orbits, molecules)
    "yoshida4" 4th-order symplectic (Yoshida); long-term energy behaviour of Verlet with higher accuracy
    "rk4"      classical Runge–Kutta: 4th order, for velocity-dependent forces (drag, damping, magnetism)
    Returns record(t, x, v) with one row per stored step (every `every` steps)."""
    a = _call_accel(accel)
    x, v = _v(x0).copy(), _v(v0).copy()
    h, n, t = _s(dt), int(steps), _s(t0)
    every = max(1, int(every))
    ts, xs, vs = [t], [x.copy()], [v.copy()]

    def verlet(x, v, t, h, acc):
        vh = v + 0.5 * h * acc
        x = x + h * vh
        acc = a(x, vh, t + h)
        return x, vh + 0.5 * h * acc, acc

    if method == "verlet":
        acc = a(x, v, t)
        for i in range(n):
            x, v, acc = verlet(x, v, t, h, acc)
            t += h
            if (i + 1) % every == 0:
                ts.append(t); xs.append(x.copy()); vs.append(v.copy())
    elif method == "yoshida4":
        w1, w0 = _Y4
        acc = a(x, v, t)
        for i in range(n):
            for w in (w1, w0, w1):
                x, v, acc = verlet(x, v, t, w * h, acc)
                t += w * h
            if (i + 1) % every == 0:
                ts.append(t); xs.append(x.copy()); vs.append(v.copy())
    elif method == "rk4":
        for i in range(n):
            k1x, k1v = v, a(x, v, t)
            k2x, k2v = v + 0.5 * h * k1v, a(x + 0.5 * h * k1x, v + 0.5 * h * k1v, t + 0.5 * h)
            k3x, k3v = v + 0.5 * h * k2v, a(x + 0.5 * h * k2x, v + 0.5 * h * k2v, t + 0.5 * h)
            k4x, k4v = v + h * k3v, a(x + h * k3x, v + h * k3v, t + h)
            x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            t += h
            if (i + 1) % every == 0:
                ts.append(t); xs.append(x.copy()); vs.append(v.copy())
    else:
        raise RuntimeErr('method must be "verlet", "yoshida4" or "rk4"')
    return Record(t=np.asarray(ts), x=np.asarray(xs), v=np.asarray(vs))


# =================================================================== Hamiltonian and Lagrangian dynamics
def _grads():
    from .stdlib import grad, jacobian
    return grad, jacobian


def hamiltonian_flow(H, q0, p0, dt, steps, method="leapfrog", every=1):
    """Hamilton's equations q' = ∂H/∂p, p' = −∂H/∂q with gradients from automatic differentiation.
    "leapfrog" is symplectic for separable H = T(p) + V(q); "rk4" works for any H.
    Returns record(t, q, p, energy)."""
    grad, _ = _grads()
    dHq, dHp = grad(H, 0), grad(H, 1)
    q, p = _v(q0).copy(), _v(p0).copy()
    h, n = _s(dt), int(steps)
    every = max(1, int(every))
    Hq = lambda q, p: _v(dHq(q, p))
    Hp = lambda q, p: _v(dHp(q, p))
    ts, qs, ps = [0.0], [q.copy()], [p.copy()]
    t = 0.0
    for i in range(n):
        if method == "leapfrog":
            p = p - 0.5 * h * Hq(q, p)
            q = q + h * Hp(q, p)
            p = p - 0.5 * h * Hq(q, p)
        elif method == "rk4":
            k1q, k1p = Hp(q, p), -Hq(q, p)
            k2q, k2p = Hp(q + h / 2 * k1q, p + h / 2 * k1p), -Hq(q + h / 2 * k1q, p + h / 2 * k1p)
            k3q, k3p = Hp(q + h / 2 * k2q, p + h / 2 * k2p), -Hq(q + h / 2 * k2q, p + h / 2 * k2p)
            k4q, k4p = Hp(q + h * k3q, p + h * k3p), -Hq(q + h * k3q, p + h * k3p)
            q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        else:
            raise RuntimeErr('method must be "leapfrog" or "rk4"')
        t += h
        if (i + 1) % every == 0:
            ts.append(t); qs.append(q.copy()); ps.append(p.copy())
    E = np.array([_s(H(qq, pp)) for qq, pp in zip(qs, ps)])
    return Record(t=np.asarray(ts), q=np.asarray(qs), p=np.asarray(ps), energy=E)


def lagrangian_flow(L, q0, qdot0, dt, steps, every=1):
    """Euler–Lagrange equations of L(q, qdot), derived by automatic differentiation:
    M(q, q̇) q̈ = ∂L/∂q − (∂²L/∂q̇∂q) q̇ with M = ∂²L/∂q̇². Integrated with RK4.
    Returns record(t, q, qdot, energy) where energy = q̇·∂L/∂q̇ − L."""
    grad, jacobian = _grads()
    dLq, dLv = grad(L, 0), grad(L, 1)
    scalar = np.ndim(_v(q0)) == 0

    def acc(q, v):
        qt, vt = (float(q[0]), float(v[0])) if scalar else (q, v)
        gq = np.atleast_1d(_v(dLq(qt, vt)))
        if scalar:
            M = np.atleast_2d(_v(grad(lambda w: dLv(qt, w))(vt)))
            C = np.atleast_2d(_v(grad(lambda u: dLv(u, vt))(qt)))
        else:
            M = np.atleast_2d(_v(jacobian(lambda w: dLv(qt, w))(vt)))
            C = np.atleast_2d(_v(jacobian(lambda u: dLv(u, vt))(qt)))
        return np.linalg.solve(M, gq - C @ v)

    q, v = np.atleast_1d(_v(q0)).copy(), np.atleast_1d(_v(qdot0)).copy()
    h, n = _s(dt), int(steps)
    every = max(1, int(every))
    ts, qs, vs = [0.0], [q.copy()], [v.copy()]
    t = 0.0
    for i in range(n):
        k1q, k1v = v, acc(q, v)
        k2q, k2v = v + h / 2 * k1v, acc(q + h / 2 * k1q, v + h / 2 * k1v)
        k3q, k3v = v + h / 2 * k2v, acc(q + h / 2 * k2q, v + h / 2 * k2v)
        k4q, k4v = v + h * k3v, acc(q + h * k3q, v + h * k3v)
        q = q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += h
        if (i + 1) % every == 0:
            ts.append(t); qs.append(q.copy()); vs.append(v.copy())

    def energy(q, v):
        qt, vt = (float(q[0]), float(v[0])) if scalar else (q, v)
        return float(np.dot(np.atleast_1d(_v(dLv(qt, vt))), v) - _s(L(qt, vt)))
    E = np.array([energy(qq, vv) for qq, vv in zip(qs, vs)])
    Q, V = np.asarray(qs), np.asarray(vs)
    if scalar:
        Q, V = Q[:, 0], V[:, 0]
    return Record(t=np.asarray(ts), q=Q, qdot=V, energy=E)


# =================================================================== N-body gravity
def _nbody_acc(x, m, G, eps2):
    d = x[None, :, :] - x[:, None, :]                       # d[i, j] = x_j − x_i
    r2 = (d * d).sum(-1) + eps2
    np.fill_diagonal(r2, np.inf)
    inv3 = r2 ** -1.5
    return G * (d * (m[None, :, None] * inv3[:, :, None])).sum(1)


def nbody(masses, x0, v0, dt, steps, G=G_NEWTON, softening=0.0, method="yoshida4", every=1):
    """Gravitational N-body integration (direct summation, symplectic). Positions/velocities have shape [N, 3]
    (or [N, 2]). Returns record(t, x, v, energy, momentum, angular_momentum, energy_drift)."""
    m = _v(masses).ravel()
    x, v = _v(x0).copy(), _v(v0).copy()
    if x.shape != v.shape or x.shape[0] != m.size:
        raise RuntimeErr("masses, positions and velocities must describe the same bodies")
    G, eps2 = _s(G), _s(softening) ** 2
    h, n = _s(dt), int(steps)
    every = max(1, int(every))
    ws = (1.0,) if method == "verlet" else _Y4[:1] + _Y4[1:] + _Y4[:1] if method == "yoshida4" else None
    if ws is None:
        raise RuntimeErr('method must be "yoshida4" or "verlet"')

    def energy(x, v):
        d = x[None] - x[:, None]
        r = np.sqrt((d * d).sum(-1) + eps2)
        iu = np.triu_indices(m.size, 1)
        return 0.5 * (m * (v * v).sum(-1)).sum() - G * (m[iu[0]] * m[iu[1]] / r[iu]).sum()

    def ang(x, v):
        if x.shape[1] == 2:
            return float((m * (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])).sum())
        return (m[:, None] * np.cross(x, v)).sum(0)

    acc = _nbody_acc(x, m, G, eps2)
    ts, xs, vs = [0.0], [x.copy()], [v.copy()]
    E, P, Lm = [energy(x, v)], [(m[:, None] * v).sum(0)], [ang(x, v)]
    t = 0.0
    for i in range(n):
        for w in ws:
            vh = v + 0.5 * w * h * acc
            x = x + w * h * vh
            acc = _nbody_acc(x, m, G, eps2)
            v = vh + 0.5 * w * h * acc
        t += h
        if (i + 1) % every == 0:
            ts.append(t); xs.append(x.copy()); vs.append(v.copy())
            E.append(energy(x, v)); P.append((m[:, None] * v).sum(0)); Lm.append(ang(x, v))
    E = np.asarray(E)
    return Record(t=np.asarray(ts), x=np.asarray(xs), v=np.asarray(vs), energy=E, momentum=np.asarray(P),
                  angular_momentum=np.asarray(Lm), energy_drift=float(np.max(np.abs(E - E[0])) / abs(E[0])))


# =================================================================== orbits
def kepler_solve(M, e):
    """Eccentric anomaly E from the mean anomaly M (Kepler's equation M = E − e sin E), 0 ≤ e < 1."""
    Mv, e = _v(M), _s(e)
    if not 0 <= e < 1:
        raise RuntimeErr("kepler_solve needs 0 ≤ e < 1")
    Mw = np.mod(Mv + np.pi, 2 * np.pi) - np.pi
    E = np.where(e > 0.8, np.pi * np.sign(Mw), Mw + e * np.sin(Mw))
    for _ in range(50):
        f = E - e * np.sin(E) - Mw
        d = f / (1 - e * np.cos(E))
        E = E - d
        if np.all(np.abs(d) < 1e-15):
            break
    E = E + (Mv - Mw)
    return float(E) if np.ndim(E) == 0 else E


def orbit_elements(r, v, mu):
    """Classical orbital elements from a state vector: a, e, i, raan, argp, nu (angles in radians)."""
    r, v, mu = _v(r), _v(v), _s(mu)
    h = np.cross(r, v)
    nvec = np.cross([0, 0, 1.0], h)
    rn, vn, hn, nn = (float(np.linalg.norm(z)) for z in (r, v, h, nvec))
    evec = ((vn * vn - mu / rn) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(evec))
    energy = vn * vn / 2 - mu / rn
    a = -mu / (2 * energy) if abs(e - 1) > 1e-12 else math.inf
    inc = math.acos(np.clip(h[2] / hn, -1, 1))
    raan = math.atan2(nvec[1], nvec[0]) % (2 * math.pi) if nn > 1e-12 else 0.0
    if e > 1e-12:
        argp = math.acos(np.clip(np.dot(nvec, evec) / (nn * e), -1, 1)) if nn > 1e-12 else \
            math.atan2(evec[1], evec[0]) % (2 * math.pi)
        if nn > 1e-12 and evec[2] < 0:
            argp = 2 * math.pi - argp
        nu = math.acos(np.clip(np.dot(evec, r) / (e * rn), -1, 1))
        if np.dot(r, v) < 0:
            nu = 2 * math.pi - nu
    else:
        argp = 0.0
        ref = nvec / nn if nn > 1e-12 else np.array([1.0, 0, 0])
        nu = math.atan2(np.dot(np.cross(ref, r), h) / hn, np.dot(ref, r)) % (2 * math.pi)
    return Record(a=a, e=e, i=inc, raan=raan, argp=argp, nu=nu, energy=energy, h=hn,
                  period=2 * math.pi * math.sqrt(a ** 3 / mu) if e < 1 else math.inf)


def orbit_state(a, e, i, raan, argp, nu, mu):
    """Position and velocity from classical orbital elements -> record(r, v)."""
    a, e, i, raan, argp, nu, mu = (_s(z) for z in (a, e, i, raan, argp, nu, mu))
    p = a * (1 - e * e)
    r_pf = p / (1 + e * math.cos(nu)) * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    R = _rz(raan) @ _rx(i) @ _rz(argp)
    return Record(r=R @ r_pf, v=R @ v_pf)


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def orbital_period(a, mu):
    """T = 2π √(a³/μ) in seconds."""
    return 2 * math.pi * math.sqrt(_s(a) ** 3 / _s(mu))


def vis_viva(r, a, mu):
    """Orbital speed at distance r on an orbit of semi-major axis a."""
    return math.sqrt(_s(mu) * (2 / _s(r) - 1 / _s(a)))


def hohmann(r1, r2, mu):
    """Hohmann transfer between circular orbits: record(dv1, dv2, dv_total, time)."""
    r1, r2, mu = _s(r1), _s(r2), _s(mu)
    at = (r1 + r2) / 2
    dv1 = math.sqrt(mu / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = math.sqrt(mu / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))
    return Record(dv1=dv1, dv2=dv2, dv_total=abs(dv1) + abs(dv2), time=math.pi * math.sqrt(at ** 3 / mu))


# =================================================================== rigid bodies
def inertia_tensor(masses, positions, about="com"):
    """Inertia tensor of point masses about their centre of mass (about="com") or the origin ("origin")."""
    m = _v(masses).ravel()
    x = _v(positions).reshape(-1, 3)
    if about == "com":
        x = x - (m[:, None] * x).sum(0) / m.sum()
    r2 = (x * x).sum(1)
    return (m[:, None, None] * (r2[:, None, None] * np.eye(3) - x[:, :, None] * x[:, None, :])).sum(0)


def principal_axes(I):
    """Principal moments (ascending) and axes (columns) of an inertia tensor."""
    w, V = np.linalg.eigh(_v(I))
    return Record(moments=w, axes=V)


def rotation_matrix(axis, angle):
    """Rotation matrix for a right-handed rotation by `angle` about `axis` (Rodrigues)."""
    k = _v(axis)
    k = k / np.linalg.norm(k)
    t = _s(angle)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def quat(axis, angle):
    """Unit quaternion [w, x, y, z] for a rotation by `angle` about `axis`."""
    k = _v(axis)
    k = k / np.linalg.norm(k)
    t = _s(angle) / 2
    return np.r_[math.cos(t), math.sin(t) * k]


def quat_mul(a, b):
    """Hamilton product a·b."""
    w1, x1, y1, z1 = _v(a)
    w2, x2, y2, z2 = _v(b)
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    q = _v(q)
    return np.r_[q[0], -q[1:]]


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q."""
    return quat_mul(quat_mul(q, np.r_[0.0, _v(v)]), quat_conj(q))[1:]


def quat_to_matrix(q):
    w, x, y, z = _v(q) / np.linalg.norm(_v(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rigid_body(I, omega0, q0, dt, steps, torque=None, every=1):
    """Free (or torqued) rotation of a rigid body: Euler's equations in the body frame + quaternion attitude, RK4.
    I: 3×3 body inertia tensor (or its 3 principal moments); torque(t, omega, q) in the body frame.
    Returns record(t, omega, q, L_world, energy)."""
    Iv = _v(I)
    Im = np.diag(Iv) if Iv.ndim == 1 else Iv
    Iinv = np.linalg.inv(Im)
    w, q = _v(omega0).copy(), _v(q0 if q0 is not None else [1, 0, 0, 0]).copy()
    q /= np.linalg.norm(q)
    h, n = _s(dt), int(steps)
    every = max(1, int(every))
    tq = (lambda t, w, q: _v(torque(t, w, q))) if torque is not None else (lambda t, w, q: 0.0)

    def f(t, w, q):
        dw = Iinv @ (tq(t, w, q) - np.cross(w, Im @ w))
        dq = 0.5 * quat_mul(q, np.r_[0.0, w])
        return dw, dq
    ts, ws, qs = [0.0], [w.copy()], [q.copy()]
    t = 0.0
    for i in range(n):
        a1, b1 = f(t, w, q)
        a2, b2 = f(t + h / 2, w + h / 2 * a1, q + h / 2 * b1)
        a3, b3 = f(t + h / 2, w + h / 2 * a2, q + h / 2 * b2)
        a4, b4 = f(t + h, w + h * a3, q + h * b3)
        w = w + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        q = q + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        q /= np.linalg.norm(q)
        t += h
        if (i + 1) % every == 0:
            ts.append(t); ws.append(w.copy()); qs.append(q.copy())
    W, Q = np.asarray(ws), np.asarray(qs)
    Lw = np.array([quat_to_matrix(qq) @ (Im @ ww) for ww, qq in zip(W, Q)])
    E = 0.5 * np.einsum("ni,ij,nj->n", W, Im, W)
    return Record(t=np.asarray(ts), omega=W, q=Q, L_world=Lw, energy=E)


# =================================================================== vibrations
def normal_modes(K, M):
    """Normal modes of M x'' + K x = 0: angular frequencies ω (rad/s), frequencies (Hz) and mass-normalized
    mode shapes (columns), from the generalized symmetric eigenproblem K φ = ω² M φ."""
    K, M = _v(K), _v(M)
    if M.ndim == 1:
        M = np.diag(M)
    Lc = np.linalg.cholesky(M)
    Li = np.linalg.inv(Lc)
    w2, Y = np.linalg.eigh(Li @ K @ Li.T)
    phi = Li.T @ Y
    w = np.sqrt(np.clip(w2, 0, None))
    return Record(omega=w, freq=w / (2 * math.pi), modes=phi)


# =================================================================== structures
def beam(length, E, I, support="cantilever", loads=None, distributed=0.0, n=100):
    """Euler–Bernoulli beam by finite elements (cubic Hermite elements, exact at the nodes for these loads).

    support: "cantilever" (fixed at x=0), "simply" (pinned at both ends), "fixed" (clamped at both ends),
             "propped" (clamped at 0, pinned at L)
    loads: list of [x, F] point forces (positive upward); distributed: uniform load per length (positive upward).
    Returns record(x, deflection, slope, moment, shear, reactions, max_deflection)."""
    L, E, I = _s(length), _s(E), _s(I)
    ne = max(2, int(n))
    pts = sorted({0.0, L} | {min(max(_s(p[0]), 0.0), L) for p in (loads or [])})
    xs = np.unique(np.concatenate([np.linspace(a, b, max(2, int(round((b - a) / L * ne)) + 1))
                                   for a, b in zip(pts[:-1], pts[1:])]))
    nn = xs.size
    K = np.zeros((2 * nn, 2 * nn))
    Fv = np.zeros(2 * nn)
    q = _s(distributed)
    for e in range(nn - 1):
        le = xs[e + 1] - xs[e]
        k = E * I / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le * le, -6 * le, 2 * le * le],
                                         [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le * le, -6 * le, 4 * le * le]])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += k
        Fv[idx] += q * np.array([le / 2, le * le / 12, le / 2, -le * le / 12])
    for p in (loads or []):
        i = int(np.argmin(np.abs(xs - min(max(_s(p[0]), 0.0), L))))
        Fv[2 * i] += _s(p[1])
    fixed = {"cantilever": [0, 1], "simply": [0, 2 * nn - 2], "fixed": [0, 1, 2 * nn - 2, 2 * nn - 1],
             "propped": [0, 1, 2 * nn - 2]}.get(support)
    if fixed is None:
        raise RuntimeErr('support must be "cantilever", "simply", "fixed" or "propped"')
    free = np.setdiff1d(np.arange(2 * nn), fixed)
    u = np.zeros(2 * nn)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], Fv[free])
    R = K @ u - Fv
    w, th = u[0::2], u[1::2]
    # bending moment and shear from the element curvature (evaluated at the nodes, averaged)
    M = np.zeros(nn)
    cnt = np.zeros(nn)
    for e in range(nn - 1):
        le = xs[e + 1] - xs[e]
        d = u[[2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]]
        for side, xi in ((e, 0.0), (e + 1, 1.0)):
            B = np.array([-6 + 12 * xi, (-4 + 6 * xi) * le, 6 - 12 * xi, (-2 + 6 * xi) * le]) / le ** 2
            M[side] += E * I * B @ d
            cnt[side] += 1
    M /= cnt
    V = np.gradient(M, xs)
    reactions = Record(**{f"at_{xs[j // 2]:g}_{'force' if j % 2 == 0 else 'moment'}": float(R[j]) for j in fixed})
    return Record(x=xs, deflection=w, slope=th, moment=M, shear=V, reactions=reactions,
                  max_deflection=float(w[np.argmax(np.abs(w))]))


def truss2d(nodes, members, E, A, supports, loads):
    """Plane pin-jointed truss by the direct stiffness method.
    nodes: [[x, y], ...]; members: [[i, j], ...] (0-based node indices); E, A: numbers or one per member;
    supports: [[node, fix_x, fix_y], ...] with true/false; loads: [[node, Fx, Fy], ...].
    Returns record(displacements [n, 2], forces (tension > 0), stresses, reactions [n, 2])."""
    X = _v(nodes).reshape(-1, 2)
    mem = [(int(a), int(b)) for a, b in members]
    m = len(mem)
    Ev = np.broadcast_to(_v(E), (m,)).astype(float)
    Av = np.broadcast_to(_v(A), (m,)).astype(float)
    nn = X.shape[0]
    K = np.zeros((2 * nn, 2 * nn))
    geo = []
    for k, (i, j) in enumerate(mem):
        d = X[j] - X[i]
        Lk = float(np.hypot(*d))
        c, s = d / Lk
        T = np.array([-c, -s, c, s])
        idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(idx, idx)] += Ev[k] * Av[k] / Lk * np.outer(T, T)
        geo.append((idx, T, Lk))
    F = np.zeros(2 * nn)
    for ld in loads:
        F[2 * int(ld[0])] += _s(ld[1])
        F[2 * int(ld[0]) + 1] += _s(ld[2])
    fixed = []
    for sdef in supports:
        nd = int(sdef[0])
        if sdef[1]:
            fixed.append(2 * nd)
        if sdef[2]:
            fixed.append(2 * nd + 1)
    free = np.setdiff1d(np.arange(2 * nn), fixed)
    Kff = K[np.ix_(free, free)]
    if np.linalg.matrix_rank(Kff) < free.size:
        raise RuntimeErr("the truss is a mechanism (not enough members or supports)")
    u = np.zeros(2 * nn)
    u[free] = np.linalg.solve(Kff, F[free])
    R = K @ u - F
    forces = np.array([Ev[k] * Av[k] / Lk * (T @ u[idx]) for k, (idx, T, Lk) in enumerate(geo)])
    return Record(displacements=u.reshape(-1, 2), forces=forces, stresses=forces / Av, reactions=R.reshape(-1, 2))


def module():
    return {k: v for k, v in globals().items()
            if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__ and k != "module"}
