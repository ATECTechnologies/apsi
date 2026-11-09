"""`use special` — special functions of mathematical physics and statistics.

Every function works on numbers and tensors (element-wise) and needs nothing beyond numpy. Accuracy is close to
double precision; the test suite compares each function against SciPy over wide argument ranges.

  gamma, lgamma, digamma, beta, lbeta, factorial, comb, perm
  erf, erfc, erfinv, erfcinv
  gammainc, gammaincc (regularized incomplete gamma P, Q), gammaincinv
  betainc (regularized incomplete beta I_x(a, b)), betaincinv
  besselj(n, x), besseli(n, x)               integer order, exponentially convergent quadrature
  ellipk(m), ellipe(m)                        complete elliptic integrals (parameter m = k²), by the AGM
  legendre(n, x), assoc_legendre(l, m, x), sph_harm(l, m, theta, phi)
  hermite(n, x) (physicists'), hermite_e(n, x) (probabilists'), laguerre(n, x, alpha), chebyshev(n, x)
  zeta(s) (s > 1), expint(x) (E1), sinc(x)
"""
import math

import numpy as np

from .errors import RuntimeErr
from .units import Quantity, DIMLESS


def _a(x):
    """Plain float ndarray from a number, list or tensor (quantities must be dimensionless)."""
    if isinstance(x, Quantity):
        if x.dim != DIMLESS:
            raise RuntimeErr("special functions need dimensionless arguments")
        x = x.v
    return np.asarray(x, dtype=float)


def _out(r, *inputs):
    """Return a Python float for scalar inputs, an array otherwise."""
    if all(np.ndim(i) == 0 for i in inputs):
        return float(np.asarray(r).reshape(()))
    return np.asarray(r, dtype=float)


def _vec(f):
    return np.vectorize(f, otypes=[float])


# ------------------------------------------------------------------ gamma family
_lg = _vec(math.lgamma)


def _gamma1(x):
    if x == math.floor(x) and x <= 0:
        return math.nan
    try:
        return math.gamma(x)
    except OverflowError:
        return math.inf


_g = _vec(_gamma1)


def gamma(x):
    """Γ(x)."""
    return _out(_g(_a(x)), x)


def lgamma(x):
    """ln|Γ(x)|."""
    return _out(_lg(_a(x)), x)


def _digamma1(x):
    if x <= 0 and x == math.floor(x):
        return math.nan
    r = 0.0
    if x < 0:                                       # reflection
        return _digamma1(1 - x) - math.pi / math.tan(math.pi * x)
    while x < 12:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def digamma(x):
    """ψ(x) = Γ'(x)/Γ(x)."""
    return _out(_vec(_digamma1)(_a(x)), x)


def lbeta(a, b):
    a, b = _a(a), _a(b)
    return _lg(a) + _lg(b) - _lg(a + b)


def beta(a, b):
    """B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
    return _out(np.exp(lbeta(a, b)), a, b)


def factorial(n):
    """n! (exact integer for integers)."""
    if isinstance(n, (int, np.integer)):
        return math.factorial(int(n))
    return _out(_g(_a(n) + 1), n)


def comb(n, k):
    """Binomial coefficient C(n, k) (exact for integers)."""
    if isinstance(n, (int, np.integer)) and isinstance(k, (int, np.integer)):
        return math.comb(int(n), int(k)) if 0 <= k <= n else 0
    n, k = _a(n), _a(k)
    return _out(np.exp(_lg(n + 1) - _lg(k + 1) - _lg(n - k + 1)), n, k)


def perm(n, k):
    """Number of ordered selections n!/(n-k)! (exact for integers)."""
    if isinstance(n, (int, np.integer)) and isinstance(k, (int, np.integer)):
        return math.perm(int(n), int(k)) if 0 <= k <= n else 0
    n, k = _a(n), _a(k)
    return _out(np.exp(_lg(n + 1) - _lg(n - k + 1)), n, k)


# ------------------------------------------------------------------ error function
_erf = _vec(math.erf)
_erfc = _vec(math.erfc)


def erf(x):
    return _out(_erf(_a(x)), x)


def erfc(x):
    return _out(_erfc(_a(x)), x)


def _erfinv1(y):
    if not -1 <= y <= 1:
        return math.nan
    if y == 1:
        return math.inf
    if y == -1:
        return -math.inf
    # Giles (2010) single-precision start, then Newton/Halley on erf for full double precision
    w = -math.log((1 - y) * (1 + y))
    if w < 5:
        w -= 2.5
        p = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503,
                  -0.00417768164, 0.246640727, 1.50140941):
            p = c + p * w
    else:
        w = math.sqrt(w) - 3
        p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613,
                  0.00943887047, 1.00167406, 2.83297682):
            p = c + p * w
    x = p * y
    for _ in range(3):
        err = (math.erf(x) - y) if abs(y) < 0.5 else -(math.erfc(x) - (1 - y)) if y > 0 else \
            (math.erfc(-x) - (1 + y))
        d = 2 / math.sqrt(math.pi) * math.exp(-x * x)
        if d == 0:
            break
        x -= err / (d + x * err)                     # Halley step
    return x


def erfinv(y):
    """Inverse error function."""
    return _out(_vec(_erfinv1)(_a(y)), y)


def erfcinv(y):
    """Inverse complementary error function."""
    y = _a(y)
    small = _vec(lambda v: -_erfinv1(v - 1) if v > 0.5 else _erfcinv_tail(v))
    return _out(small(y), y)


def _erfcinv_tail(v):
    if v <= 0:
        return math.inf if v == 0 else math.nan
    x = _erfinv1(1 - v) if v > 1e-300 else 26.0
    if not math.isfinite(x):
        x = math.sqrt(-math.log(v))
    for _ in range(50):                             # Newton on erfc directly (accurate in the far tail)
        f = math.erfc(x) - v
        d = -2 / math.sqrt(math.pi) * math.exp(-x * x)
        if d == 0:
            break
        step = f / d
        x -= step
        if abs(step) < 1e-15 * abs(x):
            break
    return x


# ------------------------------------------------------------------ incomplete gamma / beta
def _gser(a, x):
    s = term = 1.0 / a
    ap = a
    for _ in range(100000):
        ap += 1
        term *= x / ap
        s += term
        if abs(term) < abs(s) * 1e-17:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x):                                     # modified Lentz for Q(a, x)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _P(a, x):
    if math.isnan(a) or math.isnan(x) or a <= 0 or x < 0:
        return math.nan
    if x == 0:
        return 0.0
    if math.isinf(x):
        return 1.0
    return _gser(a, x) if x < a + 1 else 1 - _gcf(a, x)


def _Q(a, x):
    if math.isnan(a) or math.isnan(x) or a <= 0 or x < 0:
        return math.nan
    if x == 0:
        return 1.0
    if math.isinf(x):
        return 0.0
    return 1 - _gser(a, x) if x < a + 1 else _gcf(a, x)


def gammainc(a, x):
    """Regularized lower incomplete gamma P(a, x)."""
    return _out(_vec(_P)(_a(a), _a(x)), a, x)


def gammaincc(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x)."""
    return _out(_vec(_Q)(_a(a), _a(x)), a, x)


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1 / d
    h = d
    for m in range(1, 100000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-16:
            break
    return h


def _I(a, b, x):
    if math.isnan(a) or math.isnan(b) or math.isnan(x) or a <= 0 or b <= 0 or x < 0 or x > 1:
        return math.nan
    if x == 0 or x == 1:
        return x
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    return _out(_vec(_I)(_a(a), _a(b), _a(x)), a, b, x)


def _invert(f, y, lo, hi, increasing=True, grow_hi=False):
    """Root of f(x) = y by bracketing + bisection/secant (Illinois); robust and to full precision."""
    if grow_hi:
        while f(hi) < y and hi < 1e300:
            lo, hi = hi, hi * 4
    flo, fhi = f(lo) - y, f(hi) - y
    if flo * fhi > 0:
        return math.nan
    side = 0
    for _ in range(400):
        x = (lo * fhi - hi * flo) / (fhi - flo) if fhi != flo else 0.5 * (lo + hi)
        if not lo < x < hi:
            x = 0.5 * (lo + hi)
        fx = f(x) - y
        if fx == 0 or abs(hi - lo) <= 4e-16 * max(abs(x), 1e-300):
            return x
        if fx * fhi > 0:
            hi, fhi = x, fx
            if side == -1:
                flo /= 2
            side = -1
        else:
            lo, flo = x, fx
            if side == 1:
                fhi /= 2
            side = 1
    return 0.5 * (lo + hi)


def _gammaincinv1(a, p):
    if not 0 <= p <= 1 or a <= 0:
        return math.nan
    if p == 0:
        return 0.0
    if p == 1:
        return math.inf
    if p > 0.5:                                    # upper tail: solve Q(a, x) = 1 − p (well conditioned there)
        return _invert(lambda x: -_Q(a, x), -(1 - p), 0.0, max(1.0, 2 * a), grow_hi=True)
    return _invert(lambda x: _P(a, x), p, 0.0, max(1.0, 2 * a), grow_hi=True)


def gammaincinv(a, p):
    """x with P(a, x) = p."""
    return _out(_vec(_gammaincinv1)(_a(a), _a(p)), a, p)


def _betaincinv1(a, b, p):
    if not 0 <= p <= 1 or a <= 0 or b <= 0:
        return math.nan
    if p in (0, 1):
        return float(p)
    if p > 0.5:                                    # upper tail via the symmetry I_x(a, b) = 1 − I_{1−x}(b, a)
        return _invert(lambda x: -_I(b, a, 1 - x), -(1 - p), 0.0, 1.0)
    return _invert(lambda x: _I(a, b, x), p, 0.0, 1.0)


def betaincinv(a, b, p):
    """x with I_x(a, b) = p."""
    return _out(_vec(_betaincinv1)(_a(a), _a(b), _a(p)), a, b, p)


# ------------------------------------------------------------------ Bessel functions (integer order)
def _quad_periodic(integrand, x, n):
    """(1/π)∫_0^π g(τ) dτ for a smooth even-periodic g: the trapezoid rule converges exponentially."""
    xmax = float(np.max(np.abs(x))) if np.size(x) else 0.0
    N = int(2 * (xmax + abs(n)) + 64)
    tau = (np.arange(N) + 0.5) * (np.pi / N)          # midpoint rule on [0, π]
    return integrand(np.asarray(x)[..., None], tau).mean(axis=-1)


def _bessel_series(n, x, sign):
    """Σ_k sign^k (x/2)^{2k+n} / (k! (k+n)!): no cancellation where it is used (small |x| relative to n)."""
    m = abs(n)
    h = x / 2
    term = h ** m / math.factorial(m)
    tot = term.copy()
    q = sign * h * h
    for k in range(1, 160):
        term = term * q / (k * (k + m))
        tot = tot + term
    return tot


def besselj(n, x):
    """Bessel function of the first kind J_n(x), integer n."""
    n = int(n)
    xa = _a(x)
    r = _quad_periodic(lambda xx, t: np.cos(n * t - xx * np.sin(t)), xa, n)
    small = np.abs(xa) <= max(2.0, math.sqrt(abs(n) + 1))
    if np.any(small):
        ser = _bessel_series(n, np.where(small, xa, 0.0), -1.0)
        if n < 0 and n % 2:
            ser = -ser
        r = np.where(small, ser, r)
    return _out(r, x)


def besseli(n, x):
    """Modified Bessel function of the first kind I_n(x), integer n."""
    n = abs(int(n))
    xa = _a(x)
    r = _quad_periodic(lambda xx, t: np.exp(xx * np.cos(t)) * np.cos(n * t), xa, n)
    small = np.abs(xa) <= 50.0
    if np.any(small):
        r = np.where(small, _bessel_series(n, np.where(small, xa, 0.0), 1.0), r)
    return _out(r, x)


# ------------------------------------------------------------------ elliptic integrals
def ellipk(m):
    """Complete elliptic integral of the first kind K(m), m = k² < 1 (AGM)."""
    m = _a(m)
    with np.errstate(all="ignore"):
        a, b = np.ones_like(m), np.sqrt(1 - m)
        for _ in range(40):
            a, b = (a + b) / 2, np.sqrt(a * b)
        r = np.where(m < 1, np.pi / (2 * a), np.where(m == 1, np.inf, np.nan))
    return _out(r, m)


def ellipe(m):
    """Complete elliptic integral of the second kind E(m), m = k² ≤ 1 (AGM)."""
    m = _a(m)
    with np.errstate(all="ignore"):
        a, b = np.ones_like(m), np.sqrt(1 - m)
        c2sum = m / 2
        p = 0.5
        for _ in range(40):
            c = (a - b) / 2
            a, b = (a + b) / 2, np.sqrt(a * b)
            p *= 2
            c2sum = c2sum + p * c * c
        r = np.where(m < 1, np.pi / (2 * a) * (1 - c2sum), np.where(m == 1, 1.0, np.nan))
    return _out(r, m)


# ------------------------------------------------------------------ orthogonal polynomials
def _rec(n, x, p0, p1, step):
    n = int(n)
    x = _a(x)
    if n < 0:
        raise RuntimeErr("polynomial degree must be ≥ 0")
    a, b = p0(x), p1(x)
    if n == 0:
        return a
    for k in range(1, n):
        a, b = b, step(k, x, a, b)
    return b


def legendre(n, x):
    """Legendre polynomial P_n(x)."""
    return _out(_rec(n, x, np.ones_like, lambda x: x, lambda k, x, a, b: ((2 * k + 1) * x * b - k * a) / (k + 1)), x)


def hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) (H_1 = 2x)."""
    return _out(_rec(n, x, np.ones_like, lambda x: 2 * x, lambda k, x, a, b: 2 * x * b - 2 * k * a), x)


def hermite_e(n, x):
    """Probabilists' Hermite polynomial He_n(x) (He_1 = x)."""
    return _out(_rec(n, x, np.ones_like, lambda x: x, lambda k, x, a, b: x * b - k * a), x)


def laguerre(n, x, alpha=0.0):
    """Generalized Laguerre polynomial L_n^(α)(x)."""
    al = float(alpha)
    return _out(_rec(n, x, np.ones_like, lambda x: 1 + al - x,
                     lambda k, x, a, b: ((2 * k + 1 + al - x) * b - (k + al) * a) / (k + 1)), x)


def chebyshev(n, x):
    """Chebyshev polynomial of the first kind T_n(x)."""
    return _out(_rec(n, x, np.ones_like, lambda x: x, lambda k, x, a, b: 2 * x * b - a), x)


def assoc_legendre(l, m, x):
    """Associated Legendre function P_l^m(x) with the Condon–Shortley phase (as in physics texts)."""
    l, m = int(l), int(m)
    x = _a(x)
    am = abs(m)
    if am > l:
        return _out(np.zeros_like(x), x)
    pmm = np.ones_like(x)
    s = np.sqrt(np.clip(1 - x * x, 0, None))
    fact = 1.0
    for _ in range(am):
        pmm = -pmm * fact * s
        fact += 2
    if l == am:
        r = pmm
    else:
        pmm1 = x * (2 * am + 1) * pmm
        if l == am + 1:
            r = pmm1
        else:
            for ll in range(am + 2, l + 1):
                pll = (x * (2 * ll - 1) * pmm1 - (ll + am - 1) * pmm) / (ll - am)
                pmm, pmm1 = pmm1, pll
            r = pmm1
    if m < 0:
        r = (-1) ** am * math.factorial(l - am) / math.factorial(l + am) * r
    return _out(r, x)


def sph_harm(l, m, theta, phi):
    """Spherical harmonic Y_l^m(θ, φ) (θ polar, φ azimuth), orthonormal, Condon–Shortley phase. Complex."""
    l, m = int(l), int(m)
    th, ph = _a(theta), _a(phi)
    norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m)) if abs(m) <= l \
        else 0.0
    plm = np.asarray(assoc_legendre(l, m, np.cos(th)))
    r = norm * plm * np.exp(1j * m * ph)
    return complex(r) if np.ndim(r) == 0 else r


# ------------------------------------------------------------------ others
def zeta(s):
    """Riemann zeta ζ(s) for real s > 1 (Euler–Maclaurin)."""
    def z1(s):
        if s <= 1:
            return math.inf if s == 1 else math.nan
        N = 20
        tot = sum(k ** -s for k in range(1, N))
        tot += N ** (1 - s) / (s - 1) + 0.5 * N ** -s
        B = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6]
        fac, term = 1.0, s * N ** (-s - 1)
        for j, b in enumerate(B):
            fac = math.factorial(2 * j + 2)
            tot += b / fac * term
            term *= (s + 2 * j + 1) * (s + 2 * j + 2) / (N * N)
        return tot
    return _out(_vec(z1)(_a(s)), s)


def expint(x):
    """Exponential integral E1(x) = ∫_x^∞ e^{-t}/t dt, x > 0."""
    def e1(x):
        if x <= 0:
            return math.inf if x == 0 else math.nan
        if x < 1:
            s, term, k = 0.0, 1.0, 0
            for k in range(1, 200):
                term *= -x / k
                s += term / k
                if abs(term / k) < 1e-17:
                    break
            return -0.5772156649015329 - math.log(x) - s
        return math.exp(-x) * _cf_e1(x)
    return _out(_vec(e1)(_a(x)), x)


def _cf_e1(x):
    tiny = 1e-300
    b = x + 1
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * i
        b += 2
        d = 1 / (an * d + b)
        c = b + an / c
        de = c * d
        h *= de
        if abs(de - 1) < 1e-16:
            break
    return h


def sinc(x):
    """sin(πx)/(πx) (normalized)."""
    return _out(np.sinc(_a(x)), x)


def module():
    return {k: v for k, v in globals().items()
            if callable(v) and not k.startswith("_") and v.__module__ == __name__ and k != "module"}
