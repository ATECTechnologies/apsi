"""`use signal` — digital signal processing: filters, spectra, peaks, smoothing, resampling.

Filters are designed as second-order sections (numerically stable at any order). Zero-phase filtering
(filtfilt) pads and starts in steady state like SciPy, so results agree with SciPy to rounding error.
SciPy is used to speed up the inner filter loop when it is installed; otherwise a built-in loop is used.

  butter(order, cutoff, kind, fs) -> sos     kind: "lowpass", "highpass", "bandpass", "bandstop"
  sosfilt(sos, x), filtfilt(sos, x), freqz(sos, n, fs)
  lowpass(x, cutoff, fs, order), highpass(...), bandpass(x, low, high, fs, order), bandstop(...)
  notch(x, freq, fs, q) (IIR notch, zero phase)
  hann(n), hamming(n), blackman(n), kaiser(n, beta)
  convolve(a, b, mode), correlate(a, b, mode)
  find_peaks(x, height, distance, prominence) -> record(indices, heights, prominences)
  welch(x, fs, nperseg) -> record(f, psd); spectrogram(x, fs, nperseg, noverlap) -> record(f, t, power)
  savgol(x, window, order), moving_average(x, n), detrend(x, kind), resample(x, num)
  hilbert(x) (analytic signal), envelope(x)
"""
import math

import numpy as np

from .errors import RuntimeErr
from .units import Quantity
from .values import Record


def _x(v):
    if isinstance(v, Quantity):
        v = v.v
    return np.asarray(v, dtype=float)


def _hz(v):
    """Frequencies may be given as quantities in Hz (or 1/s)."""
    if isinstance(v, Quantity):
        v = v.v
    return float(v)


_UNSET = object()
_sp_sosfilt = _UNSET          # scipy.signal.sosfilt once loaded (None: not available or switched off)


def _sp_sosfilt_get():
    """scipy's sosfilt as an optional accelerator, imported on first use (scipy.signal takes ~0.6 s to load, so
    loading it at start-up made every program start slower)."""
    global _sp_sosfilt
    if _sp_sosfilt is _UNSET:
        try:
            from scipy.signal import sosfilt
        except Exception:                # pragma: no cover - depends on the machine
            sosfilt = None
        _sp_sosfilt = sosfilt
    return _sp_sosfilt


# =================================================================== filter design
def _zpk2sos(z, p, k):
    """Pair conjugate poles (closest to the unit circle last) with the nearest zeros into biquads."""
    z, p = list(np.asarray(z, complex)), list(np.asarray(p, complex))
    sections = []
    while p:
        i = int(np.argmin([abs(1 - abs(q)) for q in p]))         # least damped pole first
        p1 = p.pop(i)
        if abs(p1.imag) > 1e-12:
            j = int(np.argmin([abs(q - p1.conjugate()) for q in p]))
            pp = [p1, p.pop(j)]
        else:
            reals = [q for q in p if abs(q.imag) <= 1e-12]
            if reals:
                j = int(np.argmin([abs(q - p1) for q in p]))
                pp = [p1, p.pop(j)]
            else:
                pp = [p1]
        zz = []
        if z:
            j = int(np.argmin([abs(q - pp[0]) for q in z]))
            z0 = z.pop(j)
            zz.append(z0)
            if abs(z0.imag) > 1e-12:                            # keep complex zeros in conjugate pairs
                j = int(np.argmin([abs(q - z0.conjugate()) for q in z]))
                zz.append(z.pop(j))
            elif len(pp) == 2 and z:
                reals = [i for i, q in enumerate(z) if abs(q.imag) <= 1e-12]
                if reals:
                    j = min(reals, key=lambda i: abs(z[i] - pp[0]))
                    zz.append(z.pop(j))
        a = np.real(np.poly(pp))
        b = np.real(np.poly(zz)) if zz else np.array([1.0])
        a = np.pad(a, (0, 3 - a.size))
        if len(pp) == 1:                                        # first-order section: b0 + b1 z^-1
            a = np.r_[np.real(np.poly(pp)), 0.0]
            b = np.r_[b, np.zeros(3 - b.size)]
        b = np.pad(b, (0, 3 - b.size))
        sections.append(np.r_[b, a])
    sections = sections[::-1]                                   # most damped section first (like SciPy)
    sos = np.array(sections, dtype=float)
    sos[0, :3] *= k
    return sos


def _bilinear(z, p, k, fs):
    fs2 = 2.0 * fs
    zd = (fs2 + z) / (fs2 - z)
    pd = (fs2 + p) / (fs2 - p)
    zd = np.r_[zd, -np.ones(len(p) - len(z))]
    kd = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    return zd, pd, kd


def butter(order, cutoff, kind="lowpass", fs=2.0):
    """Butterworth filter as second-order sections. cutoff in Hz (with fs) — a [low, high] pair for band filters.
    Default fs=2 means cutoff is a fraction of the Nyquist frequency, as in SciPy."""
    N = int(order)
    if N < 1:
        raise RuntimeErr("filter order must be ≥ 1")
    fs = _hz(fs)
    wn = np.atleast_1d([_hz(c) for c in (cutoff if isinstance(cutoff, (list, tuple, np.ndarray)) else [cutoff])])
    if np.any(wn <= 0) or np.any(wn >= fs / 2):
        raise RuntimeErr(f"cutoff must be between 0 and fs/2 = {fs / 2:g}")
    m = np.arange(-N + 1, N, 2)
    p = -np.exp(1j * np.pi * m / (2 * N))                      # analog prototype poles
    z = np.array([], complex)
    k = 1.0
    fs_d = 2.0
    warped = 2 * fs_d * np.tan(np.pi * (wn / (fs / 2)) / fs_d)
    if kind in ("lowpass", "low"):
        p, k = p * warped[0], k * warped[0] ** N
    elif kind in ("highpass", "high"):
        k = k * np.real(np.prod(-z) / np.prod(-p))
        z, p = np.zeros(N, complex), warped[0] / p
    elif kind in ("bandpass", "band", "bandstop", "stop"):
        if wn.size != 2:
            raise RuntimeErr("band filters need cutoff = [low, high]")
        bw = warped[1] - warped[0]
        wo = math.sqrt(warped[0] * warped[1])
        if kind in ("bandpass", "band"):
            pl = p * bw / 2
            p = np.r_[pl + np.sqrt(pl ** 2 - wo ** 2), pl - np.sqrt(pl ** 2 - wo ** 2)]
            z = np.zeros(N, complex)
            k = k * bw ** N
        else:
            k = k * np.real(np.prod(-z) / np.prod(-p))
            ph = (bw / 2) / p
            p = np.r_[ph + np.sqrt(ph ** 2 - wo ** 2), ph - np.sqrt(ph ** 2 - wo ** 2)]
            z = np.r_[1j * wo * np.ones(N), -1j * wo * np.ones(N)]
    else:
        raise RuntimeErr('kind must be "lowpass", "highpass", "bandpass" or "bandstop"')
    zd, pd, kd = _bilinear(z, p, k, fs_d)
    return _zpk2sos(zd, pd, kd)


def _sosfilt_py(sos, x, zi):
    y = np.array(x, dtype=float)
    zf = np.zeros((len(sos), 2))
    for s, (b0, b1, b2, a0, a1, a2) in enumerate(sos):
        b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
        z0, z1 = (zi[s] if zi is not None else (0.0, 0.0))
        out = y.tolist()
        for n, v in enumerate(out):                         # transposed direct form II
            o = b0 * v + z0
            z0 = b1 * v - a1 * o + z1
            z1 = b2 * v - a2 * o
            out[n] = o
        y = np.asarray(out)
        zf[s] = (z0, z1)
    return y, zf


def _sosfilt(sos, x, zi=None):
    f = _sp_sosfilt_get()
    if f is not None:
        if zi is None:
            return f(sos, x), None
        return f(sos, x, zi=zi)
    return _sosfilt_py(sos, x, zi)


def sosfilt(sos, x):
    """Causal filtering with second-order sections (starts from rest)."""
    s = np.atleast_2d(_x(sos))
    a = _x(x)
    if a.ndim != 1:
        return np.apply_along_axis(lambda r: _sosfilt(s, r)[0], -1, a)
    return _sosfilt(s, a)[0]


def _sos_zi(sos):
    """Initial state of each section for a unit step in steady state (as scipy.signal.sosfilt_zi)."""
    zi = np.zeros((len(sos), 2))
    scale = 1.0
    for s, (b0, b1, b2, a0, a1, a2) in enumerate(sos):
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        IminusA = np.eye(2) - np.array([[-a[1], 1], [-a[2], 0]])
        B = b[1:] - a[1:] * b[0]
        z = np.linalg.solve(IminusA, B)
        zi[s] = scale * z
        scale *= b.sum() / a.sum()
    return zi


def filtfilt(sos, x, padlen=None):
    """Zero-phase filtering: forward and backward, odd-extended edges, steady-state start (SciPy compatible)."""
    s = np.atleast_2d(_x(sos))
    a = _x(x)
    if a.ndim != 1:
        return np.apply_along_axis(lambda r: filtfilt(s, r, padlen), -1, a)
    if padlen is None:
        nz = min((s[:, 2] == 0).sum(), (s[:, 5] == 0).sum())
        padlen = 3 * (2 * len(s) + 1 - nz)
    padlen = int(padlen)
    if a.size <= padlen:
        raise RuntimeErr(f"the signal needs more than {padlen} samples for this filter")
    if padlen:
        ext = np.r_[2 * a[0] - a[padlen:0:-1], a, 2 * a[-1] - a[-2:-padlen - 2:-1]]
    else:
        ext = a
    zi = _sos_zi(s)
    y, _ = _sosfilt(s, ext, zi * ext[0])
    y = y[::-1]
    y, _ = _sosfilt(s, y, zi * y[0])
    y = y[::-1]
    return y[padlen:-padlen] if padlen else y


def freqz(sos, n=512, fs=2 * math.pi):
    """Frequency response: record(f, h (complex), magnitude, magnitude_db, phase)."""
    s = np.atleast_2d(_x(sos))
    fs = _hz(fs)
    w = np.linspace(0, math.pi, int(n), endpoint=False)
    zm1 = np.exp(-1j * w)
    h = np.ones_like(zm1)
    for b0, b1, b2, a0, a1, a2 in s:
        h *= (b0 + b1 * zm1 + b2 * zm1 ** 2) / (a0 + a1 * zm1 + a2 * zm1 ** 2)
    mag = np.abs(h)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(mag)
    return Record(f=w * fs / (2 * math.pi), h=h, magnitude=mag, magnitude_db=db, phase=np.unwrap(np.angle(h)))


def lowpass(x, cutoff, fs, order=4):
    """Zero-phase Butterworth low-pass."""
    return filtfilt(butter(order, cutoff, "lowpass", fs), x)


def highpass(x, cutoff, fs, order=4):
    """Zero-phase Butterworth high-pass."""
    return filtfilt(butter(order, cutoff, "highpass", fs), x)


def bandpass(x, low, high, fs, order=4):
    """Zero-phase Butterworth band-pass between low and high."""
    return filtfilt(butter(order, [low, high], "bandpass", fs), x)


def bandstop(x, low, high, fs, order=4):
    """Zero-phase Butterworth band-stop."""
    return filtfilt(butter(order, [low, high], "bandstop", fs), x)


def notch(x, freq, fs, q=30.0):
    """Remove one frequency (e.g. 50/60 Hz mains) with a second-order IIR notch, zero phase."""
    f0, fs = _hz(freq), _hz(fs)
    w0 = 2 * math.pi * f0 / fs
    bw = w0 / float(q)
    g = 1 / (1 + math.tan(bw / 2))
    b = np.array([g, -2 * g * math.cos(w0), g])
    a = np.array([1, -2 * g * math.cos(w0), 2 * g - 1])
    return filtfilt(np.r_[b, a][None, :], x)


# =================================================================== windows
def hann(n, sym=True):
    """Hann window."""
    n = int(n)
    m = n if sym else n + 1
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(m) / (m - 1)) if m > 1 else np.ones(m)
    return w[:n]


def hamming(n, sym=True):
    n = int(n)
    m = n if sym else n + 1
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(m) / (m - 1)) if m > 1 else np.ones(m)
    return w[:n]


def blackman(n, sym=True):
    n = int(n)
    m = n if sym else n + 1
    k = np.arange(m) / (m - 1) if m > 1 else np.zeros(m)
    w = 0.42 - 0.5 * np.cos(2 * np.pi * k) + 0.08 * np.cos(4 * np.pi * k)
    return w[:n]


def kaiser(n, beta=8.6):
    """Kaiser window."""
    return np.kaiser(int(n), float(beta))


# =================================================================== convolution
def convolve(a, b, mode="full"):
    """Linear convolution ("full", "same" or "valid"), FFT-based for long inputs."""
    x, y = _x(a).ravel(), _x(b).ravel()
    n = x.size + y.size - 1
    if x.size * y.size > 50000:
        L = 1 << (n - 1).bit_length()
        r = np.fft.irfft(np.fft.rfft(x, L) * np.fft.rfft(y, L), L)[:n]
    else:
        r = np.convolve(x, y)
    if mode == "full":
        return r
    if mode == "same":
        st = (y.size - 1) // 2
        return r[st:st + x.size]
    if mode == "valid":
        return r[min(x.size, y.size) - 1:max(x.size, y.size)]
    raise RuntimeErr('mode must be "full", "same" or "valid"')


def correlate(a, b, mode="full"):
    """Cross-correlation (as numpy.correlate / scipy.signal.correlate for real data)."""
    return convolve(a, _x(b).ravel()[::-1], mode)


# =================================================================== peaks
def _prominence(x, peaks):
    proms, lbase, rbase = [], [], []
    n = x.size
    for p in peaks:
        h = x[p]
        i = p
        lmin = h
        li = p
        while i > 0 and x[i - 1] <= h:
            i -= 1
            if x[i] < lmin:
                lmin, li = x[i], i
        j = p
        rmin = h
        ri = p
        while j < n - 1 and x[j + 1] <= h:
            j += 1
            if x[j] < rmin:
                rmin, ri = x[j], j
        base = max(lmin, rmin)
        proms.append(h - base)
        lbase.append(li)
        rbase.append(ri)
    return np.asarray(proms, float), lbase, rbase


def find_peaks(x, height=None, distance=None, prominence=None):
    """Local maxima (flat tops: the middle sample), filtered by minimum height, minimum distance in samples
    (taller peaks win) and minimum prominence — the same rules as scipy.signal.find_peaks."""
    a = _x(x).ravel()
    n = a.size
    peaks = []
    i = 1
    while i < n - 1:
        if a[i - 1] < a[i]:
            j = i
            while j + 1 < n and a[j + 1] == a[i]:
                j += 1
            if j + 1 < n and a[j + 1] < a[i]:
                peaks.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    peaks = np.asarray(peaks, dtype=np.int64)
    if height is not None and peaks.size:
        hmin = height[0] if isinstance(height, (list, tuple)) else height
        hmax = height[1] if isinstance(height, (list, tuple)) and len(height) > 1 else None
        keep = a[peaks] >= float(hmin)
        if hmax is not None:
            keep &= a[peaks] <= float(hmax)
        peaks = peaks[keep]
    if distance is not None and peaks.size:
        d = float(distance)
        if d < 1:
            raise RuntimeErr("distance must be ≥ 1 sample")
        keep = np.ones(peaks.size, bool)
        order = np.argsort(a[peaks], kind="stable")[::-1]
        for k in order:
            if not keep[k]:
                continue
            j = k - 1
            while j >= 0 and peaks[k] - peaks[j] < d:
                keep[j] = False
                j -= 1
            j = k + 1
            while j < peaks.size and peaks[j] - peaks[k] < d:
                keep[j] = False
                j += 1
        peaks = peaks[keep]
    proms, lb, rb = _prominence(a, peaks) if peaks.size else (np.zeros(0), [], [])
    if prominence is not None and peaks.size:
        keep = proms >= float(prominence)
        peaks, proms = peaks[keep], proms[keep]
    return Record(indices=peaks, heights=a[peaks], prominences=proms, count=int(peaks.size))


# =================================================================== spectra
def _segments(x, nperseg, noverlap):
    step = nperseg - noverlap
    n = 1 + (x.size - nperseg) // step
    idx = np.arange(nperseg)[None, :] + step * np.arange(n)[:, None]
    return x[idx]


def welch(x, fs=1.0, nperseg=256, noverlap=None, detrend="constant"):
    """Power spectral density by Welch's method (Hann window, one-sided, density scaling)."""
    a = _x(x).ravel()
    fs = _hz(fs)
    nperseg = min(int(nperseg), a.size)
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    w = hann(nperseg, sym=False)
    seg = _segments(a, nperseg, noverlap)
    if detrend == "constant":
        seg = seg - seg.mean(axis=1, keepdims=True)
    elif detrend == "linear":
        seg = np.apply_along_axis(lambda r: _detrend(r, "linear"), 1, seg)
    X = np.fft.rfft(seg * w, axis=1)
    P = (np.abs(X) ** 2) / (fs * (w * w).sum())
    if nperseg % 2:
        P[:, 1:] *= 2
    else:
        P[:, 1:-1] *= 2
    return Record(f=np.fft.rfftfreq(nperseg, 1 / fs), psd=P.mean(axis=0))


def spectrogram(x, fs=1.0, nperseg=256, noverlap=None):
    """Short-time power spectrum (Hann window, density scaling): record(f, t, power[f, t])."""
    a = _x(x).ravel()
    fs = _hz(fs)
    nperseg = min(int(nperseg), a.size)
    noverlap = nperseg // 8 if noverlap is None else int(noverlap)
    w = hann(nperseg, sym=False)
    seg = _segments(a, nperseg, noverlap)
    seg = seg - seg.mean(axis=1, keepdims=True)
    X = np.fft.rfft(seg * w, axis=1)
    P = (np.abs(X) ** 2) / (fs * (w * w).sum())
    if nperseg % 2:
        P[:, 1:] *= 2
    else:
        P[:, 1:-1] *= 2
    step = nperseg - noverlap
    t = (np.arange(seg.shape[0]) * step + nperseg / 2) / fs
    return Record(f=np.fft.rfftfreq(nperseg, 1 / fs), t=t, power=P.T)


# =================================================================== smoothing and resampling
def _detrend(a, kind):
    if kind == "constant":
        return a - a.mean()
    t = np.arange(a.size, dtype=float)
    A = np.column_stack([t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    return a - A @ coef


def detrend(x, kind="linear"):
    """Remove the mean ("constant") or the least-squares line ("linear")."""
    if kind not in ("linear", "constant"):
        raise RuntimeErr('kind must be "linear" or "constant"')
    return _detrend(_x(x).ravel(), kind)


def moving_average(x, n):
    """Centered moving average over n samples (edges use the available samples)."""
    a = _x(x).ravel()
    n = int(n)
    c = np.cumsum(np.r_[0.0, a])
    half = n // 2
    lo = np.clip(np.arange(a.size) - half, 0, a.size)
    hi = np.clip(np.arange(a.size) - half + n, 0, a.size)
    return (c[hi] - c[lo]) / (hi - lo)


def savgol(x, window, order):
    """Savitzky–Golay smoothing (least-squares polynomial of `order` over `window` samples; polynomial fit at the
    edges, like SciPy's mode="interp")."""
    a = _x(x).ravel()
    w, k = int(window), int(order)
    if w % 2 == 0 or w <= k or w > a.size:
        raise RuntimeErr("window must be odd, larger than order and not longer than the signal")
    h = w // 2
    t = np.arange(-h, h + 1, dtype=float)
    A = np.vander(t, k + 1, increasing=True)
    coef = np.linalg.pinv(A)[0]
    y = np.convolve(a, coef[::-1], mode="same")
    tt = np.arange(w, dtype=float)
    V = np.vander(tt, k + 1, increasing=True)
    pl = np.linalg.lstsq(V, a[:w], rcond=None)[0]
    pr = np.linalg.lstsq(V, a[-w:], rcond=None)[0]
    y[:h] = (V @ pl)[:h]
    y[-h:] = (V @ pr)[-h:]
    return y


def resample(x, num):
    """Change the number of samples with the Fourier method (periodic signals; as scipy.signal.resample)."""
    a = _x(x).ravel()
    n, m = a.size, int(num)
    X = np.fft.rfft(a)
    Y = np.zeros(m // 2 + 1, complex)
    N = min(m, n)
    nyq = N // 2 + 1
    Y[:nyq] = X[:nyq]
    if N % 2 == 0:                       # the shared Nyquist bin is split/merged between ±N/2
        if m < n:
            Y[N // 2] *= 2
        elif m > n:
            Y[N // 2] *= 0.5
    return np.fft.irfft(Y, m) * (m / n)


def hilbert(x):
    """Analytic signal x + i·H[x] (FFT method)."""
    a = _x(x).ravel()
    n = a.size
    X = np.fft.fft(a)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(X * h)


def envelope(x):
    """Amplitude envelope |analytic signal|."""
    return np.abs(hilbert(x))


def module():
    out = {k: v for k, v in globals().items()
           if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__ and k != "module"}
    return out
