"""`use quantum` — quantum mechanics: Schrödinger equation, operators, spins and qubits.

Wave mechanics (SI units by default; pass hbar=1, mass=1 for natural units):
  schrodinger1d(V, x, mass, n, hbar)           bound states: energies (J and eV) and normalized ψ_n(x)
  evolve1d(psi0, V, x, dt, steps, mass, hbar)  time evolution by the split-operator FFT method
  gaussian_packet(x, x0, sigma, k0)            normalized wave packet
  expect_x(psi, x), expect_p(psi, x, hbar), uncertainty(psi, x, hbar), probability(psi, x, a, b)
Operators and matrices:
  dagger, commutator, anticommutator, kron(a, b, ...), expm(A), expm_hermitian(H, t, hbar),
  spin_ops(s) -> record(x, y, z, plus, minus), pauli() -> record(I, X, Y, Z)
  time_evolution(H, psi0, times, hbar)          exact, by diagonalization
Qubits (state vectors, qubit 0 = leftmost bit):
  ket("0110"), bra, gates.I/X/Y/Z/H/S/T/CNOT/CZ/SWAP, rx(θ), ry(θ), rz(θ), phase(φ)
  apply(state, gate, targets), controlled(gate), probabilities(state), measure(state, shots, seed),
  density(state), partial_trace(rho, keep, n), entropy(rho), purity(rho), fidelity(a, b), bloch(state),
  ghz(n), bell(k), concurrence(rho)
Atoms:
  hydrogen_energy(n, Z) in eV, hydrogen_radial(n, l, r) (normalized R_nl in units of a0)
"""
import math

import numpy as np

from . import ops
from .errors import RuntimeErr
from .units import Quantity
from .values import Record

HBAR = 1.054571817e-34
M_E = 9.1093837015e-31
EV = 1.602176634e-19


def _v(x):
    if isinstance(x, Quantity):
        x = x.v
    return np.asarray(ops.raw(x))


def _f(x):
    return np.asarray(_v(x), dtype=float)


def _c(x):
    return np.asarray(_v(x), dtype=complex)


def _s(x):
    return float(_f(x))


def _potential(V, x):
    if callable(V):
        return np.broadcast_to(_f(V(x)), x.shape).astype(float)
    return _f(V) * np.ones_like(x)


# =================================================================== wave mechanics
def schrodinger1d(V, x, mass=M_E, n=5, hbar=HBAR):
    """Lowest n bound states of −ħ²/(2m) ψ'' + V ψ = E ψ on the grid x (ψ = 0 beyond the ends;
    fourth-order finite differences).
    V: function of x or values on the grid. Returns record(energies, energies_ev, psi[n, len(x)], x)."""
    xg = _f(x)
    if xg.ndim != 1 or xg.size < 3:
        raise RuntimeErr("x must be a 1-D grid with at least 3 points")
    h = float(xg[1] - xg[0])
    if not np.allclose(np.diff(xg), h, rtol=1e-6, atol=0):
        raise RuntimeErr("x must be uniformly spaced")
    m, hb, k = _s(mass), _s(hbar), int(n)
    Vg = _potential(V, xg)
    t = hb * hb / (2 * m * h * h)
    N = xg.size                                           # fourth-order 5-point kinetic energy stencil
    Hm = (np.diag(Vg + 2.5 * t) - (4 / 3) * t * (np.eye(N, k=1) + np.eye(N, k=-1))
          + (1 / 12) * t * (np.eye(N, k=2) + np.eye(N, k=-2)))
    w, U = np.linalg.eigh(Hm)
    psi = U[:, :k].T / math.sqrt(h)
    for i in range(psi.shape[0]):                        # sign convention: positive just right of the left end
        j = int(np.argmax(np.abs(psi[i]) > 1e-3 * np.abs(psi[i]).max()))
        if psi[i, j] < 0:
            psi[i] = -psi[i]
    return Record(energies=w[:k], energies_ev=w[:k] / EV, psi=psi, x=xg)


def gaussian_packet(x, x0, sigma, k0=0.0):
    """Normalized Gaussian wave packet centred at x0 with width sigma and mean wavenumber k0."""
    xg = _f(x)
    s = _s(sigma)
    psi = np.exp(-((xg - _s(x0)) ** 2) / (4 * s * s) + 1j * _s(k0) * xg)
    return psi / math.sqrt(np.sum(np.abs(psi) ** 2) * (xg[1] - xg[0]))


def evolve1d(psi0, V, x, dt, steps, mass=M_E, hbar=HBAR, every=1):
    """Time evolution with the second-order split-operator method (periodic box, norm-preserving).
    Returns record(t, psi[frame, i], norm, x_mean)."""
    xg = _f(x)
    psi = _c(psi0).copy()
    h = float(xg[1] - xg[0])
    m, hb, k, n = _s(mass), _s(hbar), _s(dt), int(steps)
    Vg = _potential(V, xg)
    kk = 2 * np.pi * np.fft.fftfreq(xg.size, d=h)
    half_v = np.exp(-0.5j * Vg * k / hb)
    kin = np.exp(-1j * hb * kk * kk * k / (2 * m))
    every = max(1, int(every))
    frames, ts = [psi.copy()], [0.0]
    for i in range(n):
        psi = half_v * np.fft.ifft(kin * np.fft.fft(half_v * psi))
        if (i + 1) % every == 0:
            frames.append(psi.copy())
            ts.append((i + 1) * k)
    F = np.asarray(frames)
    P = np.abs(F) ** 2
    return Record(t=np.asarray(ts), psi=F, norm=P.sum(1) * h, x_mean=(P * xg).sum(1) * h)


def probability(psi, x, a, b):
    """∫_a^b |ψ|² dx."""
    xg = _f(x)
    p = np.abs(_c(psi)) ** 2
    m = (xg >= _s(a)) & (xg <= _s(b))
    return float(p[m].sum() * (xg[1] - xg[0]))


def expect_x(psi, x):
    xg = _f(x)
    p = np.abs(_c(psi)) ** 2
    return float((p * xg).sum() / p.sum())


def expect_p(psi, x, hbar=HBAR):
    """⟨p⟩ = ∫ ψ* (−iħ ∂ψ/∂x) dx (spectral derivative)."""
    xg = _f(x)
    ps = _c(psi)
    h = xg[1] - xg[0]
    kk = 2 * np.pi * np.fft.fftfreq(xg.size, d=h)
    d = np.fft.ifft(1j * kk * np.fft.fft(ps))
    return float(np.real(np.sum(np.conj(ps) * (-1j * _s(hbar)) * d)) / np.sum(np.abs(ps) ** 2))


def uncertainty(psi, x, hbar=HBAR):
    """record(dx, dp, product) — standard deviations of position and momentum."""
    xg = _f(x)
    ps = _c(psi)
    p = np.abs(ps) ** 2
    p = p / p.sum()
    mx = (p * xg).sum()
    dx = math.sqrt(max((p * (xg - mx) ** 2).sum(), 0))
    h = xg[1] - xg[0]
    kk = 2 * np.pi * np.fft.fftfreq(xg.size, d=h)
    pk = np.abs(np.fft.fft(ps)) ** 2
    pk /= pk.sum()
    hb = _s(hbar)
    mk = (pk * kk).sum()
    dp = hb * math.sqrt(max((pk * (kk - mk) ** 2).sum(), 0))
    return Record(dx=dx, dp=dp, product=dx * dp)


# =================================================================== operators
def dagger(A):
    return np.conj(_c(A)).T


def commutator(A, B):
    A, B = _c(A), _c(B)
    return A @ B - B @ A


def anticommutator(A, B):
    A, B = _c(A), _c(B)
    return A @ B + B @ A


def kron(*mats):
    """Tensor (Kronecker) product of any number of matrices or vectors."""
    if len(mats) == 1 and isinstance(mats[0], (list, tuple)):
        mats = mats[0]
    out = _c(mats[0])
    for m in mats[1:]:
        out = np.kron(out, _c(m))
    return out


def expm(A):
    """Matrix exponential (scaling and squaring with a Padé(13) approximant)."""
    A = _c(A)
    nrm = np.linalg.norm(A, 1)
    s = max(0, int(math.ceil(math.log2(nrm / 5.4)))) if nrm > 5.4 else 0
    A = A / (2 ** s)
    b = [64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000.,
         10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.]
    I = np.eye(A.shape[0], dtype=complex)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    U = A @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
    Vm = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I
    R = np.linalg.solve(Vm - U, Vm + U)
    for _ in range(s):
        R = R @ R
    return R


def expm_hermitian(H, t=1.0, hbar=1.0):
    """exp(−i H t / ħ) for Hermitian H (exact, by diagonalization)."""
    w, U = np.linalg.eigh(_c(H))
    return (U * np.exp(-1j * w * _s(t) / _s(hbar))) @ np.conj(U).T


def time_evolution(H, psi0, times, hbar=1.0):
    """ψ(t) = exp(−iHt/ħ) ψ0 for every t in `times` -> array [len(times), dim]."""
    w, U = np.linalg.eigh(_c(H))
    c = np.conj(U).T @ _c(psi0)
    ts = np.atleast_1d(_f(times))
    return np.array([U @ (np.exp(-1j * w * t / _s(hbar)) * c) for t in ts])


def pauli():
    """Pauli matrices: record(I, X, Y, Z)."""
    return Record(I=np.eye(2, dtype=complex), X=np.array([[0, 1], [1, 0]], complex),
                  Y=np.array([[0, -1j], [1j, 0]], complex), Z=np.array([[1, 0], [0, -1]], complex))


def spin_ops(s=0.5, hbar=1.0):
    """Spin operators for spin s (in units of ħ by default): record(x, y, z, plus, minus, squared)."""
    s2 = round(2 * float(s))
    if s2 < 1 or abs(2 * float(s) - s2) > 1e-12:
        raise RuntimeErr("spin must be a positive multiple of 1/2")
    s = s2 / 2
    m = s - np.arange(s2 + 1)
    hb = _s(hbar)
    Sp = np.zeros((s2 + 1, s2 + 1), complex)
    for i in range(1, s2 + 1):
        Sp[i - 1, i] = hb * math.sqrt(s * (s + 1) - m[i] * (m[i] + 1))
    Sm = Sp.conj().T
    Sz = hb * np.diag(m).astype(complex)
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / 2j
    return Record(x=Sx, y=Sy, z=Sz, plus=Sp, minus=Sm, squared=Sx @ Sx + Sy @ Sy + Sz @ Sz)


# =================================================================== qubits
_SQ2 = 1 / math.sqrt(2)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
H = np.array([[1, 1], [1, -1]], complex) * _SQ2
S = np.array([[1, 0], [0, 1j]], complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)


def rx(theta):
    t = _s(theta) / 2
    return np.array([[math.cos(t), -1j * math.sin(t)], [-1j * math.sin(t), math.cos(t)]])


def ry(theta):
    t = _s(theta) / 2
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], complex)


def rz(theta):
    t = _s(theta) / 2
    return np.array([[np.exp(-1j * t), 0], [0, np.exp(1j * t)]])


def phase(phi):
    return np.array([[1, 0], [0, np.exp(1j * _s(phi))]])


def controlled(gate):
    """Controlled version of a gate (control = first qubit)."""
    G = _c(gate)
    d = G.shape[0]
    out = np.eye(2 * d, dtype=complex)
    out[d:, d:] = G
    return out


def ket(bits):
    """Computational basis state, e.g. ket("010") or ket([0, 1, 0])."""
    b = [int(c) for c in (bits if isinstance(bits, str) else bits)]
    if any(c not in (0, 1) for c in b):
        raise RuntimeErr("ket() takes a string or list of 0/1")
    v = np.zeros(2 ** len(b), complex)
    v[int("".join(map(str, b)), 2) if b else 0] = 1
    return v


def bra(bits):
    return np.conj(ket(bits))


def _nq(state):
    n = int(round(math.log2(state.shape[0])))
    if 2 ** n != state.shape[0]:
        raise RuntimeErr("state dimension must be a power of 2")
    return n


def apply(state, gate, targets):
    """Apply a k-qubit gate to the given target qubits (list, in the gate's qubit order) of an n-qubit state."""
    psi = _c(state)
    G = _c(gate)
    tg = [int(targets)] if np.ndim(targets) == 0 else [int(t) for t in targets]
    n = _nq(psi)
    k = len(tg)
    if G.shape != (2 ** k, 2 ** k):
        raise RuntimeErr(f"a gate on {k} qubit(s) must be {2 ** k}×{2 ** k}")
    if len(set(tg)) != k or any(not 0 <= t < n for t in tg):
        raise RuntimeErr("invalid target qubits")
    T_ = psi.reshape([2] * n)
    rest = [i for i in range(n) if i not in tg]
    T_ = np.transpose(T_, tg + rest).reshape(2 ** k, -1)
    T_ = (G @ T_).reshape([2] * n)
    inv = np.argsort(tg + rest)
    return np.transpose(T_, inv).reshape(-1)


def probabilities(state):
    """Measurement probabilities in the computational basis."""
    return np.abs(_c(state)) ** 2


def measure(state, shots=1000, seed=None):
    """Sample measurement outcomes: record of bitstring -> count."""
    p = probabilities(state)
    n = _nq(_c(state))
    rng = np.random.default_rng(None if seed is None else int(seed))
    out = rng.choice(p.size, size=int(shots), p=p / p.sum())
    counts = np.bincount(out, minlength=p.size)
    return Record(**{"b" + format(i, f"0{n}b"): int(c) for i, c in enumerate(counts) if c})


def density(state):
    """Density matrix |ψ⟩⟨ψ|."""
    v = _c(state)
    return np.outer(v, np.conj(v))


def partial_trace(rho, keep, n=None):
    """Reduced density matrix of the qubits in `keep` (list of indices)."""
    R = _c(rho)
    n = int(n) if n is not None else _nq(R)
    keep = [int(keep)] if np.ndim(keep) == 0 else [int(k) for k in keep]
    drop = [i for i in range(n) if i not in keep]
    T_ = R.reshape([2] * (2 * n))
    perm = keep + drop + [n + i for i in keep] + [n + i for i in drop]
    T_ = np.transpose(T_, perm).reshape(2 ** len(keep), 2 ** len(drop), 2 ** len(keep), 2 ** len(drop))
    return np.einsum("ajbj->ab", T_)


def entropy(rho, base=2):
    """Von Neumann entropy −Tr ρ log ρ (bits by default)."""
    w = np.linalg.eigvalsh(_c(rho))
    w = w[w > 1e-15]
    return float(-(w * np.log(w)).sum() / math.log(float(base)))


def purity(rho):
    R = _c(rho)
    return float(np.real(np.trace(R @ R)))


def fidelity(a, b):
    """Fidelity between two pure states |⟨a|b⟩|², or two density matrices (Uhlmann)."""
    A, B = _c(a), _c(b)
    if A.ndim == 1 and B.ndim == 1:
        return float(abs(np.vdot(A, B)) ** 2)
    if A.ndim == 1:
        A = density(A)
    if B.ndim == 1:
        B = density(B)
    w, U = np.linalg.eigh(A)
    sA = (U * np.sqrt(np.clip(w, 0, None))) @ np.conj(U).T
    ev = np.linalg.eigvalsh(sA @ B @ sA)
    return float(np.sum(np.sqrt(np.clip(ev, 0, None))) ** 2)


def bloch(state):
    """Bloch vector [x, y, z] of a single-qubit state or density matrix."""
    R = _c(state)
    if R.ndim == 1:
        R = density(R)
    return np.real([np.trace(R @ X), np.trace(R @ Y), np.trace(R @ Z)])


def ghz(n):
    """GHZ state (|0…0⟩ + |1…1⟩)/√2."""
    v = np.zeros(2 ** int(n), complex)
    v[0] = v[-1] = _SQ2
    return v


def bell(k=0):
    """Bell states: 0 Φ+, 1 Φ−, 2 Ψ+, 3 Ψ−."""
    v = {0: [1, 0, 0, 1], 1: [1, 0, 0, -1], 2: [0, 1, 1, 0], 3: [0, 1, -1, 0]}[int(k)]
    return np.asarray(v, complex) * _SQ2


def concurrence(rho):
    """Wootters concurrence of a two-qubit state (0 = separable, 1 = maximally entangled)."""
    R = _c(rho)
    if R.ndim == 1:
        R = density(R)
    yy = np.kron(Y, Y)
    Rt = yy @ np.conj(R) @ yy
    ev = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(R @ Rt)))[::-1], 0, None))
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


# =================================================================== atoms
def hydrogen_energy(n, Z=1):
    """Bohr energy levels −13.605693 eV · Z²/n² (infinite nuclear mass)."""
    r = -13.605693122994 * float(Z) ** 2 / _f(n) ** 2
    return float(r) if np.ndim(r) == 0 else r


def hydrogen_radial(n, l, r):
    """Normalized radial function R_nl(r) of hydrogen, r in Bohr radii (∫ R² r² dr = 1)."""
    from .special import laguerre
    n, l = int(n), int(l)
    if not 0 <= l < n:
        raise RuntimeErr("need 0 ≤ l < n")
    rr = _f(r)
    rho = 2 * rr / n
    norm = math.sqrt((2 / n) ** 3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho ** l * np.asarray(laguerre(n - l - 1, rho, 2 * l + 1))


def module():
    out = {k: v for k, v in globals().items()
           if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__ and k != "module"}
    out["gates"] = Record(I=I, X=X, Y=Y, Z=Z, H=H, S=S, T=T, CNOT=CNOT, CZ=CZ, SWAP=SWAP)
    return out
