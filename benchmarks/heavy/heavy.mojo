# Heavy CPU benchmarks in Mojo (raw pointers into List storage, as fast Mojo code does) — same algorithms, sizes and random numbers as heavy.apsi. Run: mojo build heavy.mojo
from std.time import perf_counter_ns
from std.math import sqrt, exp, log

comptime F = Float64


@always_inline
def rng_next(s: Int) -> Int:
    return (s * 16807) % 2147483647


@always_inline
def u(s: Int) -> F:
    return F(s) / 2147483647.0


def zeros(n: Int) -> List[F]:
    return List[F](length=n, fill=0.0)


def nbody_dense(n: Int, steps: Int) -> F:
    var x_l = zeros(n); var x = x_l.unsafe_ptr(); var y_l = zeros(n); var y = y_l.unsafe_ptr(); var z_l = zeros(n); var z = z_l.unsafe_ptr()
    var vx_l = zeros(n); var vx = vx_l.unsafe_ptr(); var vy_l = zeros(n); var vy = vy_l.unsafe_ptr(); var vz_l = zeros(n); var vz = vz_l.unsafe_ptr()
    var ax_l = zeros(n); var ax = ax_l.unsafe_ptr(); var ay_l = zeros(n); var ay = ay_l.unsafe_ptr(); var az_l = zeros(n); var az = az_l.unsafe_ptr(); var m_l = zeros(n); var m = m_l.unsafe_ptr()
    var s = 42
    for i in range(n):
        s = rng_next(s); x[i] = u(s) - 0.5
        s = rng_next(s); y[i] = u(s) - 0.5
        s = rng_next(s); z[i] = u(s) - 0.5
        s = rng_next(s); m[i] = 1.0 / F(n) * (0.5 + u(s))
    var dt: F = 0.001
    var eps2: F = 0.0001
    for _ in range(steps):
        for i in range(n):
            var fx: F = 0; var fy: F = 0; var fz: F = 0
            var xi = x[i]; var yi = y[i]; var zi = z[i]
            for j in range(n):
                var dx = x[j] - xi; var dy = y[j] - yi; var dz = z[j] - zi
                var d2 = dx * dx + dy * dy + dz * dz + eps2
                var inv = m[j] / (d2 * sqrt(d2))
                fx += dx * inv; fy += dy * inv; fz += dz * inv
            ax[i] = fx; ay[i] = fy; az[i] = fz
        for i in range(n):
            vx[i] += dt * ax[i]; vy[i] += dt * ay[i]; vz[i] += dt * az[i]
            x[i] += dt * vx[i]; y[i] += dt * vy[i]; z[i] += dt * vz[i]
    var e: F = 0
    for i in range(n):
        e += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
    _ = x_l[0]
    _ = y_l[0]
    _ = z_l[0]
    _ = vx_l[0]
    _ = vy_l[0]
    _ = vz_l[0]
    _ = ax_l[0]
    _ = ay_l[0]
    _ = az_l[0]
    _ = m_l[0]
    return e


def jacobi(n: Int, iters: Int) -> F:
    var a_l = zeros(n * n); var a = a_l.unsafe_ptr(); var b_l = zeros(n * n); var b = b_l.unsafe_ptr()
    for j in range(n):
        a[j] = 100.0; b[j] = 100.0
    for _ in range(iters):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                b[i * n + j] = 0.25 * (a[(i - 1) * n + j] + a[(i + 1) * n + j] + a[i * n + j - 1] + a[i * n + j + 1])
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                a[i * n + j] = 0.25 * (b[(i - 1) * n + j] + b[(i + 1) * n + j] + b[i * n + j - 1] + b[i * n + j + 1])
    var s: F = 0
    for i in range(n * n):
        s += a[i]
    _ = a_l[0]
    _ = b_l[0]
    return s


def matmul(n: Int) -> F:
    var a_l = zeros(n * n); var a = a_l.unsafe_ptr(); var b_l = zeros(n * n); var b = b_l.unsafe_ptr(); var c_l = zeros(n * n); var c = c_l.unsafe_ptr()
    var s = 7
    for i in range(n * n):
        s = rng_next(s); a[i] = u(s) - 0.5
        s = rng_next(s); b[i] = u(s) - 0.5
    for i in range(n):
        for k in range(n):
            var aik = a[i * n + k]
            for j in range(n):
                c[i * n + j] += aik * b[k * n + j]
    var t: F = 0
    for i in range(n):
        t += c[i * n + (i * 7) % n]
    _ = a_l[0]
    _ = b_l[0]
    _ = c_l[0]
    return t


def lu_logdet(n: Int) -> F:
    var a_l = zeros(n * n); var a = a_l.unsafe_ptr()
    var s = 11
    for i in range(n * n):
        s = rng_next(s); a[i] = u(s) - 0.5
    var logdet: F = 0
    for k in range(n):
        var p = k
        var best = abs(a[k * n + k])
        for i in range(k + 1, n):
            var v = abs(a[i * n + k])
            if v > best:
                best = v; p = i
        if p != k:
            for j in range(n):
                var t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t
        var piv = a[k * n + k]
        logdet += log(abs(piv))
        for i in range(k + 1, n):
            var f = a[i * n + k] / piv
            a[i * n + k] = f
            for j in range(k + 1, n):
                a[i * n + j] -= f * a[k * n + j]
    _ = a_l[0]
    return logdet


def ising(n: Int, sweeps: Int, beta: F) -> Int:
    var sp_l = zeros(n * n); var sp = sp_l.unsafe_ptr()
    var s = 5
    for i in range(n * n):
        s = rng_next(s)
        sp[i] = 1.0 if s < 1073741824 else -1.0
    var w4 = exp(-2.0 * beta * 4.0); var w8 = exp(-2.0 * beta * 8.0)
    for _ in range(sweeps):
        for i in range(n):
            for j in range(n):
                var nb = sp[((i + 1) % n) * n + j] + sp[((i + n - 1) % n) * n + j] + sp[i * n + (j + 1) % n] + sp[i * n + (j + n - 1) % n]
                var de = 2.0 * sp[i * n + j] * nb
                s = rng_next(s)
                var r = u(s)
                if de <= 0.0 or (de == 4.0 and r < w4) or (de == 8.0 and r < w8):
                    sp[i * n + j] = -sp[i * n + j]
    var mag = 0
    for i in range(n * n):
        mag += Int(sp[i])
    _ = sp_l[0]
    return mag


def levenshtein(n: Int) -> Int:
    var a_l = zeros(n); var a = a_l.unsafe_ptr(); var b_l = zeros(n); var b = b_l.unsafe_ptr()
    var s = 3
    for i in range(n):
        s = rng_next(s); a[i] = F(s % 4)
        s = rng_next(s); b[i] = F(s % 4)
    var prev_l = zeros(n + 1); var prev = prev_l.unsafe_ptr(); var cur_l = zeros(n + 1); var cur = cur_l.unsafe_ptr()
    for j in range(n + 1):
        prev[j] = F(j)
    for i in range(1, n + 1):
        cur[0] = F(i)
        var ai = a[i - 1]
        for j in range(1, n + 1):
            var cost: F = 0.0 if ai == b[j - 1] else 1.0
            cur[j] = min(min(prev[j] + 1.0, cur[j - 1] + 1.0), prev[j - 1] + cost)
        for j in range(n + 1):
            prev[j] = cur[j]
    _ = a_l[0]
    _ = b_l[0]
    _ = prev_l[0]
    _ = cur_l[0]
    return Int(prev[n])


def vdp_rk4(steps: Int) -> F:
    var x: F = 2.0; var v: F = 0.0
    var h: F = 0.0001; var mu: F = 2.0
    for _ in range(steps):
        var k1x = v; var k1v = mu * (1.0 - x * x) * v - x
        var x2 = x + 0.5 * h * k1x; var v2 = v + 0.5 * h * k1v
        var k2x = v2; var k2v = mu * (1.0 - x2 * x2) * v2 - x2
        var x3 = x + 0.5 * h * k2x; var v3 = v + 0.5 * h * k2v
        var k3x = v3; var k3v = mu * (1.0 - x3 * x3) * v3 - x3
        var x4 = x + h * k3x; var v4 = v + h * k3v
        var k4x = v4; var k4v = mu * (1.0 - x4 * x4) * v4 - x4
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x + v


def kmeans(n: Int, k: Int, iters: Int) -> F:
    var px_l = zeros(n); var px = px_l.unsafe_ptr(); var py_l = zeros(n); var py = py_l.unsafe_ptr()
    var cx_l = zeros(k); var cx = cx_l.unsafe_ptr(); var cy_l = zeros(k); var cy = cy_l.unsafe_ptr(); var sx_l = zeros(k); var sx = sx_l.unsafe_ptr(); var sy_l = zeros(k); var sy = sy_l.unsafe_ptr(); var cnt_l = zeros(k); var cnt = cnt_l.unsafe_ptr()
    var s = 9
    for i in range(n):
        s = rng_next(s)
        var c = s % k
        s = rng_next(s); px[i] = F(c % 4) * 3.0 + u(s)
        s = rng_next(s); py[i] = F(c // 4) * 3.0 + u(s)
    for c in range(k):
        cx[c] = px[c]; cy[c] = py[c]
    var inertia: F = 0
    for _ in range(iters):
        for c in range(k):
            sx[c] = 0; sy[c] = 0; cnt[c] = 0
        inertia = 0
        for i in range(n):
            var bc = 0
            var bd: F = 1e300
            for c in range(k):
                var dx = px[i] - cx[c]; var dy = py[i] - cy[c]
                var d = dx * dx + dy * dy
                if d < bd:
                    bd = d; bc = c
            sx[bc] += px[i]; sy[bc] += py[i]; cnt[bc] += 1.0; inertia += bd
        for c in range(k):
            if cnt[c] > 0:
                cx[c] = sx[c] / cnt[c]; cy[c] = sy[c] / cnt[c]
    _ = px_l[0]
    _ = py_l[0]
    _ = cx_l[0]
    _ = cy_l[0]
    _ = sx_l[0]
    _ = sy_l[0]
    _ = cnt_l[0]
    return inertia


def report(name: String, best: F, r: F, is_int: Bool):
    var ms = Int(best * 10.0 + 0.5)
    var shown = String(Int(r)) if is_int else String(r)
    print(name + ": " + String(ms // 10) + "." + String(ms % 10) + " ms  [check " + shown + "]")


def main():
    var r: F
    var best: F
    var t: Int
    r = nbody_dense(2000, 10)
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = nbody_dense(2000, 10)
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("nbody-dense 2000x10", best, r, False)
    r = jacobi(1024, 200)
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = jacobi(1024, 200)
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("jacobi-2d 1024x200", best, r, False)
    r = matmul(1024)
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = matmul(1024)
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("matmul 1024", best, r, False)
    r = lu_logdet(800)
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = lu_logdet(800)
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("lu-pivot 800", best, r, False)
    r = F(ising(512, 50, 0.44))
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = F(ising(512, 50, 0.44))
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("ising 512x50", best, r, True)
    r = F(levenshtein(10000))
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = F(levenshtein(10000))
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("levenshtein 10k", best, r, True)
    r = vdp_rk4(20000000)
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = vdp_rk4(20000000)
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("ode-rk4 20M", best, r, False)
    r = kmeans(200000, 16, 20)
    best = 1e18
    for _ in range(3):
        t = perf_counter_ns()
        r = kmeans(200000, 16, 20)
        best = min(best, F(perf_counter_ns() - t) / 1e6)
    report("kmeans 200k k16 x20", best, r, False)
