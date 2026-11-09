"""Measurement uncertainty with exact linear error propagation.

`9.81[m/s^2] ± 0.02[m/s^2]` creates an Uncertain value. Each independent
measurement is a *source*; every result keeps its partial derivatives with
respect to all sources, so correlations are handled exactly to first order:
`x - x` is exactly 0 ± 0, and `x * x` has twice the relative error of `x`.
"""
import itertools
import math

import numpy as np

_ids = itertools.count(1)


class Uncertain:
    __slots__ = ("x", "d")          # nominal value, {source_id: (partial, sigma)}
    __array_ufunc__ = None
    __array_priority__ = 998

    def __init__(self, x, d):
        self.x = x
        self.d = d

    @staticmethod
    def measure(x, sigma):
        if sigma < 0:
            raise ValueError("uncertainty must be non-negative")
        return Uncertain(float(x), {next(_ids): (1.0, float(sigma))})

    @property
    def sigma(self):
        return math.sqrt(sum((p * s) ** 2 for p, s in self.d.values()))

    @property
    def nominal(self):
        return self.x

    # -- propagation helpers
    def _lin(self, value, da, other=None, db=0.0):
        d = {k: (p * da, s) for k, (p, s) in self.d.items()}
        if isinstance(other, Uncertain):
            for k, (p, s) in other.d.items():
                if k in d:
                    d[k] = (d[k][0] + p * db, s)
                else:
                    d[k] = (p * db, s)
        return Uncertain(value, d)

    def apply(self, f, df):
        return self._lin(f(self.x), df(self.x))

    # -- arithmetic
    def __add__(self, o):
        if isinstance(o, Uncertain):
            return self._lin(self.x + o.x, 1.0, o, 1.0)
        if not isinstance(o, (int, float, np.number)):
            return NotImplemented
        return self._lin(self.x + o, 1.0)

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        if isinstance(o, Uncertain):
            return self._lin(self.x - o.x, 1.0, o, -1.0)
        if not isinstance(o, (int, float, np.number)):
            return NotImplemented
        return self._lin(self.x - o, 1.0)

    def __rsub__(self, o):
        if not isinstance(o, (int, float, np.number)):
            return NotImplemented
        return self._lin(o - self.x, -1.0)

    def __mul__(self, o):
        if isinstance(o, Uncertain):
            return self._lin(self.x * o.x, o.x, o, self.x)
        if not isinstance(o, (int, float, np.number)):
            return NotImplemented
        return self._lin(self.x * o, o)

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        if isinstance(o, Uncertain):
            return self._lin(self.x / o.x, 1.0 / o.x, o, -self.x / (o.x * o.x))
        if not isinstance(o, (int, float, np.number)):
            return NotImplemented
        return self._lin(self.x / o, 1.0 / o)

    def __rtruediv__(self, o):
        if not isinstance(o, (int, float, np.number)):
            return NotImplemented
        return self._lin(o / self.x, -o / (self.x * self.x))

    def __pow__(self, o):
        if isinstance(o, Uncertain):
            v = self.x ** o.x
            return self._lin(v, o.x * self.x ** (o.x - 1), o, v * math.log(self.x) if self.x > 0 else 0.0)
        if not isinstance(o, (int, float, np.number)):
            return NotImplemented
        return self._lin(self.x ** o, o * self.x ** (o - 1) if o != 0 else 0.0)

    def __rpow__(self, o):
        v = o ** self.x
        return self._lin(v, v * math.log(o))

    def __neg__(self):
        return self._lin(-self.x, -1.0)

    def __pos__(self):
        return self

    def __abs__(self):
        return self._lin(abs(self.x), 1.0 if self.x >= 0 else -1.0)

    # -- comparisons use nominal values
    def _v(self, o):
        return o.x if isinstance(o, Uncertain) else o

    def __lt__(self, o): return self.x < self._v(o)
    def __le__(self, o): return self.x <= self._v(o)
    def __gt__(self, o): return self.x > self._v(o)
    def __ge__(self, o): return self.x >= self._v(o)

    def __eq__(self, o):
        if isinstance(o, Uncertain):
            return self.x == o.x and math.isclose(self.sigma, o.sigma)
        return False

    __hash__ = object.__hash__

    def __float__(self):
        return float(self.x)

    def __repr__(self):
        return format_uncertain(self.x, self.sigma)


def format_uncertain(x, s, scale=1.0):
    x, s = x / scale, s / scale
    if s == 0 or not math.isfinite(s):
        return f"{x:.12g} ± {s:.2g}"
    digits = max(0, 1 - int(math.floor(math.log10(abs(s)))))
    if digits > 12:
        return f"{x:.6e} ± {s:.2e}"
    return f"{x:.{digits}f} ± {s:.{digits}f}"


# derivative table for elementwise math
DERIVS = {
    "exp": (math.exp, math.exp),
    "log": (math.log, lambda x: 1.0 / x),
    "log10": (math.log10, lambda x: 1.0 / (x * math.log(10.0))),
    "sin": (math.sin, math.cos),
    "cos": (math.cos, lambda x: -math.sin(x)),
    "tan": (math.tan, lambda x: 1.0 / math.cos(x) ** 2),
    "asin": (math.asin, lambda x: 1.0 / math.sqrt(1 - x * x)),
    "acos": (math.acos, lambda x: -1.0 / math.sqrt(1 - x * x)),
    "atan": (math.atan, lambda x: 1.0 / (1 + x * x)),
    "sinh": (math.sinh, math.cosh),
    "cosh": (math.cosh, math.sinh),
    "tanh": (math.tanh, lambda x: 1.0 - math.tanh(x) ** 2),
    "sqrt": (math.sqrt, lambda x: 0.5 / math.sqrt(x)),
    "sigmoid": (lambda x: 1 / (1 + math.exp(-x)), lambda x: math.exp(-x) / (1 + math.exp(-x)) ** 2),
}
