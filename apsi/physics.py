"""Physics formulas added to `use physics` (alongside the constants and field operators).

All functions take numbers or quantities (SI values are used) and return quantities with the right unit.

Thermal radiation : planck(wavelength, T), wien_peak(T), blackbody_power(T, area)
Relativity        : lorentz_factor(v), relativistic_energy(m, v), relativistic_momentum(m, v),
                    lorentz_boost(t, x, v), time_dilation(dt, v), length_contraction(L, v), velocity_add(u, v)
Waves and optics  : doppler(f, v_source, v_observer, c_wave), relativistic_doppler(f, v), snell(n1, n2, theta),
                    critical_angle(n1, n2), fresnel(n1, n2, theta), brewster_angle(n1, n2), thin_lens(f, d_o),
                    single_slit(theta, width, wavelength), double_slit(theta, spacing, width, wavelength),
                    photon_energy(wavelength), de_broglie(m, v)
Electromagnetism  : coulomb_field(charges, positions, points), coulomb_potential(...), biot_savart(path, I, points),
                    lorentz_force(q, E, B, v), cyclotron_frequency(q, B, m), skin_depth(f, sigma, mu_r)
Thermodynamics    : ideal_gas_pressure(n, T, V), carnot_efficiency(T_hot, T_cold), maxwell_speed_pdf(v, m, T),
                    mean_speed(m, T), rms_speed(m, T), heat_conduction(k, area, dT, thickness)
Nuclear           : decay(N0, half_life, t), activity(N, half_life), half_life_from_rate(lam)
"""
import math

import numpy as np

from . import ops
from .errors import RuntimeErr
from .units import Quantity, make, parse_unit

C = 299792458.0
H = 6.62607015e-34
KB = 1.380649e-23
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
SIGMA = 5.670374419e-8
NA = 6.02214076e23
R = 8.314462618
WIEN = 2.897771955e-3


def _v(x):
    if isinstance(x, Quantity):
        x = x.v
    return np.asarray(ops.raw(x), dtype=float)


def _u(value, unit):
    v = np.asarray(value, dtype=float)
    v = float(v) if v.ndim == 0 else v
    _, dim = parse_unit(unit)                        # values are already SI
    return make(v, dim)


def _n(value):
    v = np.asarray(value, dtype=float)
    return float(v) if v.ndim == 0 else v


# --------------------------------------------------------------- thermal radiation
def planck(wavelength, T):
    """Spectral radiance B_λ(λ, T) in W/(m²·sr·m)."""
    lam, t = _v(wavelength), _v(T)
    with np.errstate(over="ignore"):
        b = 2 * H * C * C / lam ** 5 / np.expm1(H * C / (lam * KB * t))
    return _u(b, "W/m^3")


def wien_peak(T):
    """Wavelength of maximum spectral radiance (Wien's law)."""
    return _u(WIEN / _v(T), "m")


def blackbody_power(T, area=1.0, emissivity=1.0):
    """Radiated power ε σ A T⁴."""
    return _u(float(_v(emissivity)) * SIGMA * _v(area) * _v(T) ** 4, "W")


# --------------------------------------------------------------- relativity
def _beta(v):
    b = _v(v) / C
    if np.any(np.abs(b) >= 1):
        raise RuntimeErr("speed must be below the speed of light")
    return b


def lorentz_factor(v):
    """γ = 1/√(1 − v²/c²)."""
    return _n(1 / np.sqrt(1 - _beta(v) ** 2))


def relativistic_energy(m, v):
    """Total energy γ m c²."""
    return _u(lorentz_factor(v) * _v(m) * C * C, "J")


def relativistic_momentum(m, v):
    return _u(lorentz_factor(v) * _v(m) * _v(v), "kg*m/s")


def lorentz_boost(t, x, v):
    """Coordinates (t', x') in a frame moving with speed v along x -> [t', x']."""
    b = _beta(v)
    g = 1 / np.sqrt(1 - b * b)
    tt, xx = _v(t), _v(x)
    return [_u(g * (tt - b * xx / C), "s"), _u(g * (xx - b * C * tt), "m")]


def time_dilation(dt, v):
    """Dilated time γ·Δt of a moving clock's proper interval Δt."""
    return _u(lorentz_factor(v) * _v(dt), "s")


def length_contraction(L, v):
    return _u(_v(L) / lorentz_factor(v), "m")


def velocity_add(u, v):
    """Relativistic velocity addition (u + v)/(1 + uv/c²)."""
    a, b = _v(u), _v(v)
    return _u((a + b) / (1 + a * b / (C * C)), "m/s")


# --------------------------------------------------------------- waves and optics
def doppler(f, v_source=0.0, v_observer=0.0, c_wave=343.0):
    """Classical Doppler shift; speeds positive when source and observer approach each other."""
    return _u(_v(f) * (_v(c_wave) + _v(v_observer)) / (_v(c_wave) - _v(v_source)), "Hz")


def relativistic_doppler(f, v):
    """Frequency received from a source receding at speed v (negative v: approaching)."""
    b = _beta(v)
    return _u(_v(f) * np.sqrt((1 - b) / (1 + b)), "Hz")


def snell(n1, n2, theta):
    """Refraction angle (radians); NaN for total internal reflection."""
    s = _v(n1) * np.sin(_v(theta)) / _v(n2)
    with np.errstate(invalid="ignore"):
        return _n(np.where(np.abs(s) <= 1, np.arcsin(np.clip(s, -1, 1)), np.nan))


def critical_angle(n1, n2):
    r = _v(n2) / _v(n1)
    if np.any(r >= 1):
        raise RuntimeErr("total internal reflection needs n1 > n2")
    return _n(np.arcsin(r))


def brewster_angle(n1, n2):
    return _n(np.arctan(_v(n2) / _v(n1)))


def fresnel(n1, n2, theta):
    """Fresnel power coefficients for s and p polarization -> [Rs, Rp, Ts, Tp]."""
    a, b, th = _v(n1), _v(n2), _v(theta)
    ci = np.cos(th)
    st = a / b * np.sin(th)
    ct = np.sqrt(np.clip(1 - st * st, 0, None) + 0j)
    rs = (a * ci - b * ct) / (a * ci + b * ct)
    rp = (b * ci - a * ct) / (b * ci + a * ct)
    Rs, Rp = np.abs(rs) ** 2, np.abs(rp) ** 2
    return [_n(Rs), _n(Rp), _n(1 - Rs), _n(1 - Rp)]


def thin_lens(f, d_o):
    """Image distance and magnification for a thin lens -> [d_i, m]."""
    fo, do = _v(f), _v(d_o)
    with np.errstate(divide="ignore"):
        di = 1 / (1 / fo - 1 / do)
    return [_u(di, "m"), _n(-di / do)]


def single_slit(theta, width, wavelength):
    """Relative Fraunhofer intensity of a single slit: sinc²(π a sinθ / λ)."""
    beta = np.pi * _v(width) * np.sin(_v(theta)) / _v(wavelength)
    return _n(np.sinc(beta / np.pi) ** 2)


def double_slit(theta, spacing, width, wavelength):
    """Relative intensity of two slits of finite width."""
    s = np.sin(_v(theta))
    lam = _v(wavelength)
    return _n(np.cos(np.pi * _v(spacing) * s / lam) ** 2 * np.sinc(_v(width) * s / lam) ** 2)


def photon_energy(wavelength):
    return _u(H * C / _v(wavelength), "J")


def de_broglie(m, v):
    """Matter wavelength h/(m v) (non-relativistic)."""
    return _u(H / (_v(m) * _v(v)), "m")


# --------------------------------------------------------------- electromagnetism
def _pts(x):
    a = _v(x)
    return a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)


def coulomb_field(charges, positions, points, softening=0.0):
    """Electric field (V/m) of point charges at the given points (2-D or 3-D), shape [points, dim]."""
    q = _v(charges).ravel()
    P = _pts(positions)
    X = _pts(points)
    d = X[:, None, :] - P[None, :, :]
    r2 = (d * d).sum(-1) + float(_v(softening)) ** 2
    E = (q[None, :, None] * d / r2[..., None] ** 1.5).sum(1) / (4 * math.pi * EPS0)
    return _u(E, "V/m")


def coulomb_potential(charges, positions, points, softening=0.0):
    q = _v(charges).ravel()
    P = _pts(positions)
    X = _pts(points)
    d = X[:, None, :] - P[None, :, :]
    r = np.sqrt((d * d).sum(-1) + float(_v(softening)) ** 2)
    return _u((q[None, :] / r).sum(1) / (4 * math.pi * EPS0), "V")


def biot_savart(path, current, points):
    """Magnetic field (T) of a current along a polyline `path` [[x, y, z], ...] at the given points.
    Each straight segment uses the exact finite-wire formula."""
    Pth = _pts(path)
    if Pth.shape[1] != 3:
        raise RuntimeErr("biot_savart needs 3-D points")
    I = float(_v(current))
    X = _pts(points)
    B = np.zeros_like(X)
    for a, b in zip(Pth[:-1], Pth[1:]):                  # exact finite segment, cancellation-free form
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L == 0:
            continue
        u = seg / L
        r1 = X - a
        p1 = r1 @ u
        p2 = p1 - L
        dvec = r1 - p1[:, None] * u
        dn = np.linalg.norm(dvec, axis=1)
        n1 = np.hypot(dn, p1)
        n2 = np.hypot(dn, p2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.where(dn > 1e-300, MU0 * I / (4 * math.pi * dn * dn) * (p1 / n1 - p2 / n2), 0.0)
        B += mag[:, None] * np.cross(u, dvec)
    return _u(B, "T")


def lorentz_force(q, E, B, v):
    """F = q (E + v × B)."""
    return _u(_v(q) * (_v(E) + np.cross(_v(v), _v(B))), "N")


def cyclotron_frequency(q, B, m):
    """Angular cyclotron frequency |q|B/m (rad/s)."""
    return _u(abs(_v(q)) * _v(B) / _v(m), "1/s")


def skin_depth(f, sigma, mu_r=1.0):
    """Electromagnetic skin depth √(2/(ω μ σ)) of a good conductor."""
    w = 2 * math.pi * _v(f)
    return _u(np.sqrt(2 / (w * MU0 * _v(mu_r) * _v(sigma))), "m")


# --------------------------------------------------------------- thermodynamics
def ideal_gas_pressure(n, T, V):
    """p = n R T / V (n in mol)."""
    return _u(_v(n) * R * _v(T) / _v(V), "Pa")


def carnot_efficiency(T_hot, T_cold):
    return _n(1 - _v(T_cold) / _v(T_hot))


def maxwell_speed_pdf(v, m, T):
    """Maxwell–Boltzmann speed distribution f(v) (s/m)."""
    vv, mm, t = _v(v), _v(m), _v(T)
    a = mm / (2 * KB * t)
    return _u(4 * math.pi * (a / math.pi) ** 1.5 * vv * vv * np.exp(-a * vv * vv), "s/m")


def mean_speed(m, T):
    return _u(np.sqrt(8 * KB * _v(T) / (math.pi * _v(m))), "m/s")


def rms_speed(m, T):
    return _u(np.sqrt(3 * KB * _v(T) / _v(m)), "m/s")


def heat_conduction(k, area, dT, thickness):
    """Fourier's law: heat flow k A ΔT / d (W)."""
    return _u(_v(k) * _v(area) * _v(dT) / _v(thickness), "W")


# --------------------------------------------------------------- nuclear
def decay(N0, half_life, t):
    """Remaining amount N0 · 2^(−t/T½) (same unit as N0)."""
    f = np.exp2(-_v(t) / _v(half_life))
    if isinstance(N0, Quantity):
        return Quantity(N0.v * f, N0.dim, N0.disp)
    return _n(_v(N0) * f)


def activity(N, half_life):
    """Activity λN in becquerel."""
    return _u(math.log(2) / _v(half_life) * _v(N), "1/s")


def half_life_from_rate(lam):
    return _u(math.log(2) / _v(lam), "s")


def module():
    return {k: v for k, v in globals().items()
            if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__
            and k not in ("module", "make", "parse_unit")}

