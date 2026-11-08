"""Apsi physical units: SI dimensions, unit parsing and the Quantity value type.

Every Quantity stores its magnitude in SI base units plus a 7-component
dimension vector (m, kg, s, A, K, mol, cd). Results that become
dimensionless collapse back into plain numbers automatically.
"""
import math
import re
from fractions import Fraction
from functools import lru_cache

import numpy as np

from .errors import UnitError

BASE_NAMES = ("m", "kg", "s", "A", "K", "mol", "cd")


class Dim(tuple):
    """Dimension vector: exponents of the 7 SI base units (as Fractions)."""
    __slots__ = ()

    def __new__(cls, exps=()):
        e = [Fraction(x) for x in exps]
        e += [Fraction(0)] * (7 - len(e))
        return super().__new__(cls, e)

    def __mul__(self, o):
        return Dim([a + b for a, b in zip(self, o)])

    def __truediv__(self, o):
        return Dim([a - b for a, b in zip(self, o)])

    def __pow__(self, p):
        p = Fraction(p).limit_denominator(1000)
        return Dim([a * p for a in self])

    @property
    def dimensionless(self):
        return not any(self)

    def __str__(self):
        return dim_name(self)

    def __repr__(self):
        return f"Dim({unit_string(self)})"


def D(m=0, kg=0, s=0, A=0, K=0, mol=0, cd=0):
    return Dim((m, kg, s, A, K, mol, cd))


DIMLESS = D()

# ---------------------------------------------------------------- dimension names
# Names usable as type annotations: fn f(m: Mass) -> Energy
_DIM_TABLE = [
    ("Dimensionless", DIMLESS), ("Angle", DIMLESS),
    ("Length", D(m=1)), ("Mass", D(kg=1)), ("Time", D(s=1)), ("Current", D(A=1)),
    ("Temperature", D(K=1)), ("Amount", D(mol=1)), ("LuminousIntensity", D(cd=1)),
    ("Area", D(m=2)), ("Volume", D(m=3)),
    ("Velocity", D(m=1, s=-1)), ("Speed", D(m=1, s=-1)),
    ("Acceleration", D(m=1, s=-2)), ("Jerk", D(m=1, s=-3)),
    ("Force", D(m=1, kg=1, s=-2)),
    ("Energy", D(m=2, kg=1, s=-2)), ("Work", D(m=2, kg=1, s=-2)), ("Torque", D(m=2, kg=1, s=-2)),
    ("Power", D(m=2, kg=1, s=-3)),
    ("Pressure", D(m=-1, kg=1, s=-2)), ("Stress", D(m=-1, kg=1, s=-2)),
    ("Momentum", D(m=1, kg=1, s=-1)), ("Impulse", D(m=1, kg=1, s=-1)),
    ("AngularMomentum", D(m=2, kg=1, s=-1)), ("Action", D(m=2, kg=1, s=-1)),
    ("Frequency", D(s=-1)), ("AngularVelocity", D(s=-1)),
    ("Density", D(m=-3, kg=1)), ("Charge", D(s=1, A=1)),
    ("Voltage", D(m=2, kg=1, s=-3, A=-1)), ("Resistance", D(m=2, kg=1, s=-3, A=-2)),
    ("Capacitance", D(m=-2, kg=-1, s=4, A=2)), ("Inductance", D(m=2, kg=1, s=-2, A=-2)),
    ("MagneticField", D(kg=1, s=-2, A=-1)), ("MagneticFlux", D(m=2, kg=1, s=-2, A=-1)),
    ("ElectricField", D(m=1, kg=1, s=-3, A=-1)),
    ("Entropy", D(m=2, kg=1, s=-2, K=-1)), ("HeatCapacity", D(m=2, kg=1, s=-2, K=-1)),
    ("Viscosity", D(m=-1, kg=1, s=-1)), ("Diffusivity", D(m=2, s=-1)),
    ("KinematicViscosity", D(m=2, s=-1)), ("WaveNumber", D(m=-1)),
    ("Concentration", D(m=-3, mol=1)), ("MolarMass", D(kg=1, mol=-1)),
    ("ThermalConductivity", D(m=1, kg=1, s=-3, K=-1)),
    ("SpringConstant", D(kg=1, s=-2)),
]
DIM_NAMES = dict(_DIM_TABLE)
_ALIASES = {"Speed", "Work", "Torque", "Stress", "Impulse", "Action", "AngularVelocity",
            "HeatCapacity", "KinematicViscosity", "Angle", "SpringConstant"}
_DIM_TO_NAME = {}
for _n, _d in _DIM_TABLE:
    if _n not in _ALIASES and _d not in _DIM_TO_NAME:
        _DIM_TO_NAME[_d] = _n

# ---------------------------------------------------------------- unit table
_J = D(m=2, kg=1, s=-2)
_PA = D(m=-1, kg=1, s=-2)
UNITS = {
    "m": (1.0, D(m=1)), "g": (1e-3, D(kg=1)), "s": (1.0, D(s=1)), "A": (1.0, D(A=1)),
    "K": (1.0, D(K=1)), "mol": (1.0, D(mol=1)), "cd": (1.0, D(cd=1)),
    "Hz": (1.0, D(s=-1)), "N": (1.0, D(m=1, kg=1, s=-2)), "Pa": (1.0, _PA),
    "J": (1.0, _J), "W": (1.0, D(m=2, kg=1, s=-3)), "Wh": (3600.0, _J),
    "C": (1.0, D(s=1, A=1)), "V": (1.0, D(m=2, kg=1, s=-3, A=-1)),
    "F": (1.0, D(m=-2, kg=-1, s=4, A=2)), "ohm": (1.0, D(m=2, kg=1, s=-3, A=-2)),
    "Ω": (1.0, D(m=2, kg=1, s=-3, A=-2)), "S": (1.0, D(m=-2, kg=-1, s=3, A=2)),
    "Wb": (1.0, D(m=2, kg=1, s=-2, A=-1)), "T": (1.0, D(kg=1, s=-2, A=-1)),
    "H": (1.0, D(m=2, kg=1, s=-2, A=-2)),
    "eV": (1.602176634e-19, _J), "L": (1e-3, D(m=3)), "bar": (1e5, _PA),
    "rad": (1.0, DIMLESS), "sr": (1.0, DIMLESS), "deg": (math.pi / 180, DIMLESS),
    "min": (60.0, D(s=1)), "h": (3600.0, D(s=1)), "day": (86400.0, D(s=1)),
    "yr": (365.25 * 86400.0, D(s=1)),
    "au": (1.495978707e11, D(m=1)), "ly": (9.4607304725808e15, D(m=1)),
    "pc": (3.0856775814913673e16, D(m=1)), "Å": (1e-10, D(m=1)),
    "atm": (101325.0, _PA), "mmHg": (133.322387415, _PA), "psi": (6894.757293168, _PA),
    "cal": (4.184, _J), "kcal": (4184.0, _J), "Da": (1.66053906660e-27, D(kg=1)),
    "ft": (0.3048, D(m=1)), "inch": (0.0254, D(m=1)), "mi": (1609.344, D(m=1)),
    "lb": (0.45359237, D(kg=1)), "mph": (0.44704, D(m=1, s=-1)),
    "kn": (1852.0 / 3600.0, D(m=1, s=-1)), "rpm": (2 * math.pi / 60, D(s=-1)),
}
PREFIXABLE = {"m", "g", "s", "A", "K", "mol", "cd", "Hz", "N", "Pa", "J", "W", "Wh", "C", "V",
              "F", "ohm", "Ω", "S", "Wb", "T", "H", "eV", "L", "bar", "rad"}
PREFIXES = {"Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9, "M": 1e6,
            "k": 1e3, "h": 1e2, "da": 1e1, "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6,
            "µ": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}
_PREFIX_ORDER = sorted(PREFIXES, key=len, reverse=True)


def lookup_unit(name):
    if name in UNITS:
        return UNITS[name]
    for p in _PREFIX_ORDER:
        if name.startswith(p) and name[len(p):] in PREFIXABLE:
            scale, dim = UNITS[name[len(p):]]
            return PREFIXES[p] * scale, dim
    raise UnitError(f"unknown unit '{name}'")


# ---------------------------------------------------------------- unit parsing
_SUPER = [("⁻¹", "^-1"), ("⁻²", "^-2"), ("⁻³", "^-3"), ("¹", "^1"), ("²", "^2"), ("³", "^3"), ("⁴", "^4")]
_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)|([A-Za-zµμΩÅ_][A-Za-z0-9µμΩÅ_]*)|(\*\*|[*·/^()\-]))")


def _tokenize_unit(text):
    for a, b in _SUPER:
        text = text.replace(a, b)
    pos, toks = 0, []
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise UnitError(f"cannot parse unit '{text}' near '{text[pos:]}'")
        num, name, op = m.groups()
        toks.append(("num", num) if num else ("name", name) if name else ("op", op))
        pos = m.end()
    return toks


@lru_cache(maxsize=None)
def parse_unit(text):
    """Parse a unit expression like 'kg*m/s^2' or 'J/(mol K)' -> (scale, Dim)."""
    toks = _tokenize_unit(text)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else (None, None)

    def take():
        t = peek()
        pos[0] += 1
        return t

    def exponent():
        sign = 1
        if peek() == ("op", "-"):
            take()
            sign = -1
        k, v = take()
        if k == "num":
            return sign * Fraction(v)
        if (k, v) == ("op", "("):
            s2 = 1
            if peek() == ("op", "-"):
                take()
                s2 = -1
            k1, a = take()
            if peek() == ("op", "/"):
                take()
                _, b = take()
                val = Fraction(int(a), int(b))
            else:
                val = Fraction(a)
            if take() != ("op", ")"):
                raise UnitError(f"bad exponent in unit '{text}'")
            return sign * s2 * val
        raise UnitError(f"bad exponent in unit '{text}'")

    def atom():
        k, v = take()
        if k == "name":
            return lookup_unit(v)
        if k == "num":
            return float(v), DIMLESS
        if (k, v) == ("op", "("):
            r = expr()
            if take() != ("op", ")"):
                raise UnitError(f"missing ')' in unit '{text}'")
            return r
        raise UnitError(f"cannot parse unit '{text}'")

    def factor():
        scale, dim = atom()
        if peek() in (("op", "^"), ("op", "**")):
            take()
            e = exponent()
            scale, dim = scale ** float(e), dim ** e
        return scale, dim

    def expr():
        scale, dim = factor()
        while True:
            t = peek()
            if t in (("op", "*"), ("op", "·")):
                take()
                s2, d2 = factor()
                scale, dim = scale * s2, dim * d2
            elif t == ("op", "/"):
                take()
                s2, d2 = factor()
                scale, dim = scale / s2, dim / d2
            elif t[0] in ("name", "num") or t == ("op", "("):
                s2, d2 = factor()
                scale, dim = scale * s2, dim * d2
            else:
                return scale, dim

    if not toks:
        raise UnitError("empty unit")
    result = expr()
    if pos[0] != len(toks):
        raise UnitError(f"cannot parse unit '{text}'")
    return result


# ---------------------------------------------------------------- display
_DISPLAY = [
    ("m", D(m=1)), ("kg", D(kg=1)), ("s", D(s=1)), ("A", D(A=1)), ("K", D(K=1)),
    ("mol", D(mol=1)), ("cd", D(cd=1)), ("Hz", D(s=-1)), ("m/s", D(m=1, s=-1)), ("m/s^2", D(m=1, s=-2)),
    ("N", D(m=1, kg=1, s=-2)), ("J", _J), ("W", D(m=2, kg=1, s=-3)), ("Pa", _PA),
    ("kg·m/s", D(m=1, kg=1, s=-1)), ("J·s", D(m=2, kg=1, s=-1)), ("C", D(s=1, A=1)),
    ("V", D(m=2, kg=1, s=-3, A=-1)), ("Ω", D(m=2, kg=1, s=-3, A=-2)),
    ("F", D(m=-2, kg=-1, s=4, A=2)), ("H", D(m=2, kg=1, s=-2, A=-2)),
    ("T", D(kg=1, s=-2, A=-1)), ("Wb", D(m=2, kg=1, s=-2, A=-1)), ("S", D(m=-2, kg=-1, s=3, A=2)),
    ("J/K", D(m=2, kg=1, s=-2, K=-1)), ("V/m", D(m=1, kg=1, s=-3, A=-1)),
    ("W/(m·K)", D(m=1, kg=1, s=-3, K=-1)), ("N/m", D(kg=1, s=-2)),
]
_DISPLAY_MAP = {}
for _n, _d in _DISPLAY:
    _DISPLAY_MAP.setdefault(_d, _n)


def _fmt_exp(e):
    return str(e.numerator) if e.denominator == 1 else f"({e.numerator}/{e.denominator})"


def unit_string(dim):
    if dim in _DISPLAY_MAP:
        return _DISPLAY_MAP[dim]
    num, den = [], []
    for name, e in zip(BASE_NAMES, dim):
        if e > 0:
            num.append(name if e == 1 else f"{name}^{_fmt_exp(e)}")
        elif e < 0:
            den.append(name if e == -1 else f"{name}^{_fmt_exp(-e)}")
    s = "·".join(num) or "1"
    if den:
        s += "/" + (den[0] if len(den) == 1 else "(" + "·".join(den) + ")")
    return s


def dim_name(dim):
    if dim.dimensionless:
        return "Dimensionless"
    n = _DIM_TO_NAME.get(dim)
    return f"{n} [{unit_string(dim)}]" if n else f"[{unit_string(dim)}]"


# ---------------------------------------------------------------- Quantity
def _is_zero(x):
    return isinstance(x, (int, float, Fraction)) and not isinstance(x, bool) and x == 0


def make(v, dim, disp=None):
    """Build a Quantity, or return the bare value if dimensionless."""
    if dim.dimensionless:
        return v
    return Quantity(v, dim, disp)


class Quantity:
    __slots__ = ("v", "dim", "disp")
    __array_ufunc__ = None          # make numpy defer to our reflected operators
    __array_priority__ = 1000

    def __init__(self, v, dim, disp=None):
        self.v = v
        self.dim = dim
        self.disp = disp            # (unit text, scale) preferred for display

    # -- helpers
    def _same(self, o, verb):
        if isinstance(o, Quantity):
            if o.dim != self.dim:
                raise UnitError(f"cannot {verb} {dim_name(self.dim)} and {dim_name(o.dim)}")
            return o.v
        if _is_zero(o):
            return o
        raise UnitError(f"cannot {verb} {dim_name(self.dim)} and a dimensionless number")

    # -- arithmetic
    def __add__(self, o):
        return Quantity(self.v + self._same(o, "add"), self.dim)

    def __radd__(self, o):
        return Quantity(self._same(o, "add") + self.v, self.dim)

    def __sub__(self, o):
        return Quantity(self.v - self._same(o, "subtract"), self.dim)

    def __rsub__(self, o):
        return Quantity(self._same(o, "subtract") - self.v, self.dim)

    def __mul__(self, o):
        if isinstance(o, Quantity):
            return make(self.v * o.v, self.dim * o.dim)
        return Quantity(self.v * o, self.dim)

    def __rmul__(self, o):
        return Quantity(o * self.v, self.dim)

    def __truediv__(self, o):
        if isinstance(o, Quantity):
            return make(self.v / o.v, self.dim / o.dim)
        return Quantity(self.v / o, self.dim)

    def __rtruediv__(self, o):
        return Quantity(o / self.v, DIMLESS / self.dim)

    def __floordiv__(self, o):
        return self.v // self._same(o, "floor-divide")

    def __mod__(self, o):
        return Quantity(self.v % self._same(o, "take modulo of"), self.dim)

    def __pow__(self, p):
        if isinstance(p, Quantity):
            raise UnitError("exponent must be dimensionless")
        if isinstance(p, np.ndarray) and p.size != 1:
            raise UnitError("a Quantity can only be raised to a single scalar power")
        pf = float(p)
        return make(self.v ** p, self.dim ** Fraction(pf).limit_denominator(1000))

    def __rpow__(self, o):
        raise UnitError(f"exponent must be dimensionless, got {dim_name(self.dim)}")

    def __matmul__(self, o):
        if isinstance(o, Quantity):
            return make(self.v @ o.v, self.dim * o.dim)
        return Quantity(self.v @ o, self.dim)

    def __rmatmul__(self, o):
        return Quantity(o @ self.v, self.dim)

    def __neg__(self):
        return Quantity(-self.v, self.dim)

    def __pos__(self):
        return self

    def __abs__(self):
        return Quantity(abs(self.v), self.dim)

    # -- comparisons
    def __lt__(self, o):
        return self.v < self._same(o, "compare")

    def __le__(self, o):
        return self.v <= self._same(o, "compare")

    def __gt__(self, o):
        return self.v > self._same(o, "compare")

    def __ge__(self, o):
        return self.v >= self._same(o, "compare")

    def __eq__(self, o):
        if isinstance(o, Quantity):
            return self.dim == o.dim and self.v == o.v
        if _is_zero(o):
            return self.v == 0
        return False

    def __ne__(self, o):
        r = self.__eq__(o)
        return ~r if isinstance(r, np.ndarray) else not r

    def __hash__(self):
        try:
            return hash((self.v, self.dim))
        except TypeError:
            return id(self)

    # -- containers
    def __getitem__(self, i):
        return Quantity(self.v[i], self.dim)

    def __setitem__(self, i, val):
        self.v[i] = self._same(val, "assign")

    def __len__(self):
        return len(self.v)

    def __iter__(self):
        for x in self.v:
            yield Quantity(x, self.dim)

    def __bool__(self):
        return bool(self.v)

    def __float__(self):
        raise UnitError(f"cannot convert {dim_name(self.dim)} to a plain number; "
                        f"use x.magnitude or 'x as [unit]'")

    # -- properties
    @property
    def shape(self):
        v = getattr(self.v, "value", self.v)
        return tuple(np.shape(v))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def T(self):
        return Quantity(self.v.T, self.dim)

    @property
    def value(self):
        """Magnitude in SI base units."""
        return self.v

    @property
    def magnitude(self):
        """Magnitude in the display unit (e.g. km after 'as [km]')."""
        return self.v / self.disp[1] if self.disp else self.v

    @property
    def unit(self):
        return self.disp[0] if self.disp else unit_string(self.dim)

    def __repr__(self):
        from .fmt import fmt
        return fmt(self)
