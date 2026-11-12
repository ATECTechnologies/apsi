"""`use hp` — high-precision arithmetic: 30, 50, 100 or 10 000 significant digits.

    use hp
    hp.digits(60)                      # every hp calculation now carries 60 significant digits
    let x = hp.num("0.1")              # exact decimal input (a string keeps every digit you write)
    print(hp.sqrt(2))                  # 1.41421356237309504880168872420969807856967187537694807317668
    print(hp.pi())                     # π to 60 digits
    print(hp.exp(1) - hp.e())          # 0E-59  (agreement to the last digit)
    print(hp.sum([0.1] * 10))          # 1 — the exact sum of the binary values, then rounded once
    let A = [[4, 1], [1, 3]]
    print(hp.solve(A, [1, 2]))         # linear systems without rounding drift
    hp.float(x)                        # back to an ordinary 64-bit float

Ordinary floats (IEEE 754 double) carry about 16 significant digits, so a relative error below ~1e-16 cannot be
expressed in them at all. hp numbers are Python decimals: they work with + - * / ** < == and with print,
format and f-strings, and every function below is computed with guard digits and rounded once to the precision
you chose — results are accurate to within one unit in the last digit (1e-30 needs about 32 digits, 1e-100 about
102). Cost grows with the number of digits; 50 digits are still millions of operations per second for + - * /.

Functions: num, digits, float, sqrt, cbrt, root, exp, ln/log, log10, log2, pow, sin, cos, tan, asin, acos, atan,
atan2, sinh, cosh, tanh, pi, e, sum, dot, mean, prod, solve, det, inv, matmul, factorial, gamma (integers and
halves), equal (comparison to a number of digits), str (fixed digits).
"""
import decimal
import math
from decimal import Decimal, localcontext

_DIGITS = [50]


def _ctx(extra=10):
    c = decimal.Context(prec=_DIGITS[0] + extra, rounding=decimal.ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999)
    return c


def digits(n=None):
    """Set (or read) the number of significant digits of hp results (default 50, at most 100 000)."""
    if n is not None:
        n = int(n)
        if not 2 <= n <= 100000:
            raise ValueError("hp.digits must be between 2 and 100000")
        _DIGITS[0] = n
    _sync()
    return _DIGITS[0]


def _sync():
    """+ - * / between hp numbers use Python's decimal context: give it the same precision."""
    c = decimal.getcontext()
    c.prec = _DIGITS[0]
    c.Emax, c.Emin = 999999999, -999999999


def _round(x):
    with localcontext(decimal.Context(prec=_DIGITS[0], rounding=decimal.ROUND_HALF_EVEN, Emax=999999999,
                                      Emin=-999999999)):
        return +x


def num(x):
    """An hp number. Strings keep every digit you write ("0.1" is exactly one tenth); floats keep their exact
    binary value (0.1 becomes 0.1000000000000000055511…), integers and fractions are exact."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, str):
        return Decimal(x.strip().replace("_", ""))
    if isinstance(x, bool):
        return Decimal(int(x))
    if isinstance(x, int):
        return Decimal(x)
    try:
        from fractions import Fraction
        if isinstance(x, Fraction):
            with localcontext(_ctx()):
                return _round(Decimal(x.numerator) / Decimal(x.denominator))
    except ImportError:                      # pragma: no cover
        pass
    if hasattr(x, "item"):                   # numpy scalars
        x = x.item()
    if hasattr(x, "v") and hasattr(x, "dim"):
        raise TypeError("hp works with plain numbers; take magnitude(x) of a quantity first")
    return Decimal(float(x))


def to_float(x):
    return float(x)


# ------------------------------------------------------------------------------------------------ constants
def pi():
    """π to the current number of digits (Chudnovsky-free: Machin's formula with binary splitting of atan)."""
    with localcontext(_ctx()):
        return _round(16 * _atan_inv(5) - 4 * _atan_inv(239))


def e():
    return exp(1)


def _atan_inv(n):
    """atan(1/n) for an integer n > 1 (series converges fast)."""
    ctx = decimal.getcontext()
    eps = Decimal(10) ** (-(ctx.prec + 2))
    x = Decimal(1) / n
    x2 = x * x
    term, total, k = x, x, 1
    while abs(term) > eps:
        term *= -x2
        k += 2
        total += term / k
    return total


# ------------------------------------------------------------------------------------------------ roots & logs
def sqrt(x):
    with localcontext(_ctx()):
        return _round(num(x).sqrt())


def root(x, n):
    """The real n-th root (negative x allowed for odd n)."""
    n = int(n)
    with localcontext(_ctx()):
        x = num(x)
        if x < 0:
            if n % 2 == 0:
                raise ValueError("even root of a negative number")
            return _round(-root(-x, n))
        if x == 0:
            return Decimal(0)
        return _round((x.ln() / n).exp())


def cbrt(x):
    return root(x, 3)


def exp(x):
    with localcontext(_ctx()):
        return _round(num(x).exp())


def ln(x):
    with localcontext(_ctx()):
        return _round(num(x).ln())


log = ln


def log10(x):
    with localcontext(_ctx()):
        return _round(num(x).log10())


def log2(x):
    with localcontext(_ctx()):
        return _round(num(x).ln() / Decimal(2).ln())


def pow(x, y):
    with localcontext(_ctx()):
        x, y = num(x), num(y)
        if y == y.to_integral_value() and abs(y) < 10 ** 6:
            return _round(x ** int(y))
        if x <= 0:
            if x == 0:
                return Decimal(0)
            raise ValueError("a negative number to a non-integer power")
        return _round((y * x.ln()).exp())


# ------------------------------------------------------------------------------------------------ trigonometry
def _reduce(x):
    """x reduced to [-π, π] using π with extra digits (so large arguments keep their accuracy)."""
    p = pi_raw()
    two = 2 * p
    k = (x / two).to_integral_value(rounding=decimal.ROUND_HALF_EVEN)
    return x - k * two, p


def pi_raw():
    return 16 * _atan_inv(5) - 4 * _atan_inv(239)


def _sin_series(x):
    ctx = decimal.getcontext()
    eps = Decimal(10) ** (-(ctx.prec + 2))
    term, total, k, x2 = x, x, 1, x * x
    while abs(term) > eps:
        term *= -x2 / ((k + 1) * (k + 2))
        k += 2
        total += term
    return total


def _cos_series(x):
    ctx = decimal.getcontext()
    eps = Decimal(10) ** (-(ctx.prec + 2))
    term, total, k, x2 = Decimal(1), Decimal(1), 0, x * x
    while abs(term) > eps:
        term *= -x2 / ((k + 1) * (k + 2))
        k += 2
        total += term
    return total


def sin(x):
    with localcontext(_ctx(20 + max(0, num(x).adjusted()))):
        r, p = _reduce(num(x))
        return _round(_sin_series(r))


def cos(x):
    with localcontext(_ctx(20 + max(0, num(x).adjusted()))):
        r, p = _reduce(num(x))
        return _round(_cos_series(r))


def tan(x):
    with localcontext(_ctx(20 + max(0, num(x).adjusted()))):
        r, p = _reduce(num(x))
        return _round(_sin_series(r) / _cos_series(r))


def _atan(x):
    """atan with argument halving so the series converges quickly for any x."""
    if x < 0:
        return -_atan(-x)
    if x > 1:
        return pi_raw() / 2 - _atan(1 / x)
    halvings = 0
    while x > Decimal("0.1"):
        x = x / (1 + (1 + x * x).sqrt())
        halvings += 1
    ctx = decimal.getcontext()
    eps = Decimal(10) ** (-(ctx.prec + 2))
    term, total, k, x2 = x, x, 1, x * x
    while abs(term) > eps:
        term *= -x2
        k += 2
        total += term / k
    return total * (2 ** halvings)


def atan(x):
    with localcontext(_ctx(20)):
        return _round(_atan(num(x)))


def atan2(y, x):
    with localcontext(_ctx(20)):
        y, x = num(y), num(x)
        if x > 0:
            return _round(_atan(y / x))
        p = pi_raw()
        if x < 0:
            return _round(_atan(y / x) + (p if y >= 0 else -p))
        if y > 0:
            return _round(p / 2)
        if y < 0:
            return _round(-p / 2)
        return Decimal(0)


def asin(x):
    with localcontext(_ctx(20)):
        x = num(x)
        if abs(x) > 1:
            raise ValueError("asin of a number outside [-1, 1]")
        if abs(x) == 1:
            return _round(pi_raw() / 2 * (1 if x > 0 else -1))
        return _round(_atan(x / (1 - x * x).sqrt()))


def acos(x):
    with localcontext(_ctx(20)):
        return _round(pi_raw() / 2 - asin(x))


def sinh(x):
    with localcontext(_ctx()):
        ex = num(x).exp()
        return _round((ex - 1 / ex) / 2)


def cosh(x):
    with localcontext(_ctx()):
        ex = num(x).exp()
        return _round((ex + 1 / ex) / 2)


def tanh(x):
    with localcontext(_ctx()):
        e2 = (2 * num(x)).exp()
        return _round((e2 - 1) / (e2 + 1))


# ------------------------------------------------------------------------------------------------ collections
def _items(xs):
    if hasattr(xs, "tolist"):
        xs = xs.tolist()
    out = []
    for v in xs:
        if isinstance(v, (list, tuple)):
            out.extend(_items(v))
        else:
            out.append(num(v))
    return out


def sum(xs):
    """The exact sum (no rounding error accumulates), rounded once to the current precision."""
    with localcontext(decimal.Context(prec=decimal.MAX_PREC, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)):
        total = Decimal(0)
        for v in _items(xs):
            total += v
    return _round(total)


def prod(xs):
    with localcontext(_ctx(len(xs) if hasattr(xs, "__len__") else 20)):
        p = Decimal(1)
        for v in _items(xs):
            p *= v
        return _round(p)


def mean(xs):
    items = _items(xs)
    with localcontext(_ctx()):
        return _round(sum(items) / len(items))


def dot(a, b):
    a, b = _items(a), _items(b)
    if len(a) != len(b):
        raise ValueError(f"dot: lengths differ ({len(a)} and {len(b)})")
    with localcontext(decimal.Context(prec=decimal.MAX_PREC, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)):
        total = Decimal(0)
        for x, y in zip(a, b):
            total += x * y
    return _round(total)


def _matrix(m):
    if hasattr(m, "tolist"):
        m = m.tolist()
    return [[num(v) for v in row] for row in m]


def matmul(a, b):
    A, B = _matrix(a), _matrix(b)
    if len(A[0]) != len(B):
        raise ValueError("matmul: inner sizes differ")
    cols = list(zip(*B))
    return [[dot(row, col) for col in cols] for row in A]


def _lu(A):
    n = len(A)
    A = [row[:] for row in A]
    perm, sign = list(range(n)), 1
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(A[i][k]))
        if A[p][k] == 0:
            raise ValueError("the matrix is singular")
        if p != k:
            A[k], A[p] = A[p], A[k]
            perm[k], perm[p] = perm[p], perm[k]
            sign = -sign
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            A[i][k] = f
            for j in range(k + 1, n):
                A[i][j] -= f * A[k][j]
    return A, perm, sign


def solve(a, b):
    """x with a·x = b, by LU factorisation with partial pivoting at the current precision (+ guard digits)."""
    with localcontext(_ctx(20)):
        A = _matrix(a)
        n = len(A)
        rhs = [num(v) for v in _items(b)]
        LU, perm, _ = _lu(A)
        y = [rhs[perm[i]] for i in range(n)]
        for i in range(n):
            for j in range(i):
                y[i] -= LU[i][j] * y[j]
        x = [Decimal(0)] * n
        for i in reversed(range(n)):
            s = y[i]
            for j in range(i + 1, n):
                s -= LU[i][j] * x[j]
            x[i] = s / LU[i][i]
        return [_round(v) for v in x]


def det(a):
    with localcontext(_ctx(20)):
        LU, _, sign = _lu(_matrix(a))
        d = Decimal(sign)
        for i in range(len(LU)):
            d *= LU[i][i]
        return _round(d)


def inv(a):
    n = len(_matrix(a))
    cols = [solve(a, [1 if i == j else 0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def factorial(n):
    return Decimal(math.factorial(int(n)))


def gamma(x):
    """Γ(x) for positive integers and half-integers (exact formulas)."""
    x = num(x)
    with localcontext(_ctx()):
        if x == x.to_integral_value() and x > 0:
            return _round(Decimal(math.factorial(int(x) - 1)))
        if (2 * x) == (2 * x).to_integral_value() and x > 0:
            k = int(x - Decimal("0.5"))
            return _round(Decimal(math.factorial(2 * k)) / (Decimal(4) ** k * math.factorial(k)) * pi_raw().sqrt())
    raise ValueError("hp.gamma supports positive integers and halves; use special.gamma for other values")


def equal(a, b, digits_=None):
    """True when a and b agree to `digits` significant digits (default: the current precision minus 2)."""
    d = _DIGITS[0] - 2 if digits_ is None else int(digits_)
    a, b = num(a), num(b)
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    with localcontext(_ctx()):
        return abs(a - b) <= scale * Decimal(10) ** (-d)


def text(x, n=None):
    """x as text with exactly n significant digits (default: the current precision)."""
    n = _DIGITS[0] if n is None else int(n)
    with localcontext(decimal.Context(prec=n)):
        return format(+num(x), "")


def module():
    _sync()
    return {"digits": digits, "num": num, "float": to_float, "pi": pi, "e": e, "sqrt": sqrt, "root": root,
            "cbrt": cbrt, "exp": exp, "ln": ln, "log": log, "log10": log10, "log2": log2, "pow": pow, "sin": sin,
            "cos": cos, "tan": tan, "asin": asin, "acos": acos, "atan": atan, "atan2": atan2, "sinh": sinh,
            "cosh": cosh, "tanh": tanh, "sum": sum, "prod": prod, "mean": mean, "dot": dot, "matmul": matmul,
            "solve": solve, "det": det, "inv": inv, "factorial": factorial, "gamma": gamma, "equal": equal,
            "str": text}
