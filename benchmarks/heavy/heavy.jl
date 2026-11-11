# Heavy CPU benchmarks in Julia — same algorithms, sizes and random numbers as heavy.c. Run: julia -O3 heavy.jl
using Printf
rng(s::Int64) = (s * 16807) % 2147483647
U(s::Int64) = Float64(s) / 2147483647.0

function nbody_dense(n::Int, steps::Int)
    x = zeros(n); y = zeros(n); z = zeros(n); vx = zeros(n); vy = zeros(n); vz = zeros(n)
    ax = zeros(n); ay = zeros(n); az = zeros(n); m = zeros(n)
    s = 42
    for i in 1:n
        s = rng(s); x[i] = U(s) - 0.5; s = rng(s); y[i] = U(s) - 0.5; s = rng(s); z[i] = U(s) - 0.5
        s = rng(s); m[i] = 1.0 / n * (0.5 + U(s))
    end
    dt = 0.001; eps2 = 0.0001
    @inbounds for st in 1:steps
        for i in 1:n
            fx = 0.0; fy = 0.0; fz = 0.0; xi = x[i]; yi = y[i]; zi = z[i]
            for j in 1:n
                dx = x[j] - xi; dy = y[j] - yi; dz = z[j] - zi; d2 = dx * dx + dy * dy + dz * dz + eps2
                inv = m[j] / (d2 * sqrt(d2)); fx += dx * inv; fy += dy * inv; fz += dz * inv
            end
            ax[i] = fx; ay[i] = fy; az[i] = fz
        end
        for i in 1:n
            vx[i] += dt * ax[i]; vy[i] += dt * ay[i]; vz[i] += dt * az[i]; x[i] += dt * vx[i]; y[i] += dt * vy[i]; z[i] += dt * vz[i]
        end
    end
    e = 0.0
    for i in 1:n
        e += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
    end
    e
end

function jacobi(n::Int, iters::Int)
    a = zeros(n * n); b = zeros(n * n)
    for j in 1:n
        a[j] = 100.0; b[j] = 100.0
    end
    @inbounds for it in 1:iters
        for i in 1:n-2, j in 1:n-2
            b[i*n+j+1] = 0.25 * (a[(i-1)*n+j+1] + a[(i+1)*n+j+1] + a[i*n+j] + a[i*n+j+2])
        end
        for i in 1:n-2, j in 1:n-2
            a[i*n+j+1] = 0.25 * (b[(i-1)*n+j+1] + b[(i+1)*n+j+1] + b[i*n+j] + b[i*n+j+2])
        end
    end
    s = 0.0
    for i in 1:n*n
        s += a[i]
    end
    s
end

function matmul(n::Int)
    a = zeros(n * n); b = zeros(n * n); c = zeros(n * n); s = 7
    for i in 0:n-1, j in 0:n-1
        s = rng(s); a[i*n+j+1] = U(s) - 0.5; s = rng(s); b[i*n+j+1] = U(s) - 0.5
    end
    @inbounds for i in 0:n-1, k in 0:n-1
        aik = a[i*n+k+1]
        for j in 0:n-1
            c[i*n+j+1] += aik * b[k*n+j+1]
        end
    end
    t = 0.0
    for i in 0:n-1
        t += c[i*n+(i*7)%n+1]
    end
    t
end

function lu_logdet(n::Int)
    a = zeros(n * n); s = 11
    for i in 0:n-1, j in 0:n-1
        s = rng(s); a[i*n+j+1] = U(s) - 0.5
    end
    logdet = 0.0
    @inbounds for k in 0:n-1
        p = k; best = abs(a[k*n+k+1])
        for i in k+1:n-1
            v = abs(a[i*n+k+1])
            if v > best
                best = v; p = i
            end
        end
        if p != k
            for j in 0:n-1
                t = a[k*n+j+1]; a[k*n+j+1] = a[p*n+j+1]; a[p*n+j+1] = t
            end
        end
        piv = a[k*n+k+1]; logdet += log(abs(piv))
        for i in k+1:n-1
            f = a[i*n+k+1] / piv; a[i*n+k+1] = f
            for j in k+1:n-1
                a[i*n+j+1] -= f * a[k*n+j+1]
            end
        end
    end
    logdet
end

function ising(n::Int, sweeps::Int, beta::Float64)
    sp = zeros(n * n); s = 5
    for i in 0:n-1, j in 0:n-1
        s = rng(s); sp[i*n+j+1] = s < 1073741824 ? 1.0 : -1.0
    end
    w4 = exp(-2.0 * beta * 4.0); w8 = exp(-2.0 * beta * 8.0)
    @inbounds for sw in 1:sweeps, i in 0:n-1, j in 0:n-1
        nb = sp[((i+1)%n)*n+j+1] + sp[((i+n-1)%n)*n+j+1] + sp[i*n+(j+1)%n+1] + sp[i*n+(j+n-1)%n+1]
        de = 2.0 * sp[i*n+j+1] * nb; s = rng(s); r = U(s)
        if de <= 0.0 || (de == 4.0 && r < w4) || (de == 8.0 && r < w8)
            sp[i*n+j+1] = -sp[i*n+j+1]
        end
    end
    mag = 0
    for i in 1:n*n
        mag += Int(sp[i])
    end
    mag
end

function levenshtein(n::Int)
    a = zeros(n); b = zeros(n); prev = zeros(n + 1); cur = zeros(n + 1); s = 3
    for i in 1:n
        s = rng(s); a[i] = Float64(s % 4); s = rng(s); b[i] = Float64(s % 4)
    end
    for j in 0:n
        prev[j+1] = j
    end
    @inbounds for i in 1:n
        cur[1] = i; ai = a[i]
        for j in 1:n
            cost = ai == b[j] ? 0.0 : 1.0
            cur[j+1] = min(min(prev[j+1] + 1.0, cur[j] + 1.0), prev[j] + cost)
        end
        for j in 1:n+1
            prev[j] = cur[j]
        end
    end
    Int(prev[n+1])
end

function vdp_rk4(steps::Int)
    x = 2.0; v = 0.0; h = 0.0001; mu = 2.0
    for i in 1:steps
        k1x = v; k1v = mu * (1.0 - x * x) * v - x
        x2 = x + 0.5 * h * k1x; v2 = v + 0.5 * h * k1v; k2x = v2; k2v = mu * (1.0 - x2 * x2) * v2 - x2
        x3 = x + 0.5 * h * k2x; v3 = v + 0.5 * h * k2v; k3x = v3; k3v = mu * (1.0 - x3 * x3) * v3 - x3
        x4 = x + h * k3x; v4 = v + h * k3v; k4x = v4; k4v = mu * (1.0 - x4 * x4) * v4 - x4
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x); v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    end
    x + v
end

function kmeans(n::Int, k::Int, iters::Int)
    px = zeros(n); py = zeros(n); cx = zeros(k); cy = zeros(k); sx = zeros(k); sy = zeros(k); cnt = zeros(k); s = 9
    for i in 1:n
        s = rng(s); c = s % k; s = rng(s); px[i] = Float64(c % 4) * 3.0 + U(s); s = rng(s); py[i] = Float64(div(c, 4)) * 3.0 + U(s)
    end
    for c in 1:k
        cx[c] = px[c]; cy[c] = py[c]
    end
    inertia = 0.0
    @inbounds for it in 1:iters
        fill!(sx, 0.0); fill!(sy, 0.0); fill!(cnt, 0.0)
        inertia = 0.0
        for i in 1:n
            bc = 1; bd = 1e300
            for c in 1:k
                dx = px[i] - cx[c]; dy = py[i] - cy[c]; d = dx * dx + dy * dy
                if d < bd
                    bd = d; bc = c
                end
            end
            sx[bc] += px[i]; sy[bc] += py[i]; cnt[bc] += 1.0; inertia += bd
        end
        for c in 1:k
            if cnt[c] > 0
                cx[c] = sx[c] / cnt[c]; cy[c] = sy[c] / cnt[c]
            end
        end
    end
    inertia
end

fmt(r::Int) = string(r)
fmt(r::Float64) = (t = @sprintf("%.12g", r); t)

const SINK = Ref{Any}(nothing)        # every result is kept (like C's volatile), so no run can be removed

function T(name, f)
    r = f(); best = 1e9
    for k in 1:3
        t = time_ns(); SINK[] = f(); d = (time_ns() - t) * 1e-9
        best = min(best, d)
    end
    @printf("%s: %.1f ms  [check %s]\n", name, best * 1000, fmt(r))
end

# sizes read through Refs (like C's volatile): the compiler cannot fold a whole run into a constant
const N = Ref([2000, 10, 1024, 200, 800, 512, 50, 10000, 20000000, 200000, 16, 20])
T("nbody-dense 2000x10", () -> nbody_dense(N[][1], N[][2]))
T("jacobi-2d 1024x200", () -> jacobi(N[][3], N[][4]))
T("matmul 1024", () -> matmul(N[][3]))
T("lu-pivot 800", () -> lu_logdet(N[][5]))
T("ising 512x50", () -> ising(N[][6], N[][7], 0.44))
T("levenshtein 10k", () -> levenshtein(N[][8]))
T("ode-rk4 20M", () -> vdp_rk4(N[][9]))
T("kmeans 200k k16 x20", () -> kmeans(N[][10], N[][11], N[][12]))
