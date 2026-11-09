// Heavy CPU benchmarks in Rust — same algorithms, sizes and random numbers as heavy.apsi.
// Build: rustc -C opt-level=3 -C target-cpu=native heavy.rs
use std::hint::black_box;
use std::time::Instant;

#[inline]
fn rng_next(s: i64) -> i64 { (s * 16807) % 2147483647 }
#[inline]
fn u(s: i64) -> f64 { s as f64 / 2147483647.0 }

fn nbody_dense(n: usize, steps: usize) -> f64 {
    let (mut x, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut vx, mut vy, mut vz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut ax, mut ay, mut az, mut m) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut s = 42i64;
    for i in 0..n {
        s = rng_next(s); x[i] = u(s) - 0.5; s = rng_next(s); y[i] = u(s) - 0.5;
        s = rng_next(s); z[i] = u(s) - 0.5; s = rng_next(s); m[i] = 1.0 / n as f64 * (0.5 + u(s));
    }
    let (dt, eps2) = (0.001, 0.0001);
    for _ in 0..steps {
        for i in 0..n {
            let (mut fx, mut fy, mut fz) = (0.0, 0.0, 0.0);
            let (xi, yi, zi) = (x[i], y[i], z[i]);
            for j in 0..n {
                let (dx, dy, dz) = (x[j] - xi, y[j] - yi, z[j] - zi);
                let d2 = dx * dx + dy * dy + dz * dz + eps2;
                let inv = m[j] / (d2 * d2.sqrt());
                fx += dx * inv; fy += dy * inv; fz += dz * inv;
            }
            ax[i] = fx; ay[i] = fy; az[i] = fz;
        }
        for i in 0..n {
            vx[i] += dt * ax[i]; vy[i] += dt * ay[i]; vz[i] += dt * az[i];
            x[i] += dt * vx[i]; y[i] += dt * vy[i]; z[i] += dt * vz[i];
        }
    }
    (0..n).map(|i| 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])).sum()
}

fn jacobi(n: usize, iters: usize) -> f64 {
    let (mut a, mut b) = (vec![0.0; n * n], vec![0.0; n * n]);
    for j in 0..n { a[j] = 100.0; b[j] = 100.0; }
    for _ in 0..iters {
        for i in 1..n - 1 { for j in 1..n - 1 {
            b[i * n + j] = 0.25 * (a[(i - 1) * n + j] + a[(i + 1) * n + j] + a[i * n + j - 1] + a[i * n + j + 1]);
        } }
        for i in 1..n - 1 { for j in 1..n - 1 {
            a[i * n + j] = 0.25 * (b[(i - 1) * n + j] + b[(i + 1) * n + j] + b[i * n + j - 1] + b[i * n + j + 1]);
        } }
    }
    a.iter().sum()
}

fn matmul(n: usize) -> f64 {
    let (mut a, mut b, mut c) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
    let mut s = 7i64;
    for i in 0..n * n { s = rng_next(s); a[i] = u(s) - 0.5; s = rng_next(s); b[i] = u(s) - 0.5; }
    for i in 0..n { for k in 0..n {
        let aik = a[i * n + k];
        for j in 0..n { c[i * n + j] += aik * b[k * n + j]; }
    } }
    (0..n).map(|i| c[i * n + (i * 7) % n]).sum()
}

fn lu_logdet(n: usize) -> f64 {
    let mut a = vec![0.0; n * n];
    let mut s = 11i64;
    for v in a.iter_mut() { s = rng_next(s); *v = u(s) - 0.5; }
    let mut logdet = 0.0;
    for k in 0..n {
        let (mut p, mut best) = (k, a[k * n + k].abs());
        for i in k + 1..n { let v = a[i * n + k].abs(); if v > best { best = v; p = i; } }
        if p != k { for j in 0..n { a.swap(k * n + j, p * n + j); } }
        let piv = a[k * n + k];
        logdet += piv.abs().ln();
        for i in k + 1..n {
            let f = a[i * n + k] / piv; a[i * n + k] = f;
            for j in k + 1..n { a[i * n + j] -= f * a[k * n + j]; }
        }
    }
    logdet
}

fn ising(n: usize, sweeps: usize, beta: f64) -> i64 {
    let mut sp = vec![0.0; n * n];
    let mut s = 5i64;
    for v in sp.iter_mut() { s = rng_next(s); *v = if s < 1073741824 { 1.0 } else { -1.0 }; }
    let (w4, w8) = ((-2.0 * beta * 4.0f64).exp(), (-2.0 * beta * 8.0f64).exp());
    for _ in 0..sweeps { for i in 0..n { for j in 0..n {
        let nb = sp[((i + 1) % n) * n + j] + sp[((i + n - 1) % n) * n + j] + sp[i * n + (j + 1) % n] + sp[i * n + (j + n - 1) % n];
        let de = 2.0 * sp[i * n + j] * nb;
        s = rng_next(s); let r = u(s);
        if de <= 0.0 || (de == 4.0 && r < w4) || (de == 8.0 && r < w8) { sp[i * n + j] = -sp[i * n + j]; }
    } } }
    sp.iter().map(|&v| v as i64).sum()
}

fn levenshtein(n: usize) -> i64 {
    let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
    let mut s = 3i64;
    for i in 0..n { s = rng_next(s); a[i] = (s % 4) as f64; s = rng_next(s); b[i] = (s % 4) as f64; }
    let mut prev: Vec<f64> = (0..=n).map(|j| j as f64).collect();
    let mut cur = vec![0.0; n + 1];
    for i in 1..=n {
        cur[0] = i as f64; let ai = a[i - 1];
        for j in 1..=n {
            let cost = if ai == b[j - 1] { 0.0 } else { 1.0 };
            cur[j] = (prev[j] + 1.0).min(cur[j - 1] + 1.0).min(prev[j - 1] + cost);
        }
        prev.copy_from_slice(&cur);
    }
    prev[n] as i64
}

fn vdp_rk4(steps: usize) -> f64 {
    let (mut x, mut v) = (2.0f64, 0.0f64);
    let (h, mu) = (0.0001, 2.0);
    for _ in 0..steps {
        let (k1x, k1v) = (v, mu * (1.0 - x * x) * v - x);
        let (x2, v2) = (x + 0.5 * h * k1x, v + 0.5 * h * k1v); let (k2x, k2v) = (v2, mu * (1.0 - x2 * x2) * v2 - x2);
        let (x3, v3) = (x + 0.5 * h * k2x, v + 0.5 * h * k2v); let (k3x, k3v) = (v3, mu * (1.0 - x3 * x3) * v3 - x3);
        let (x4, v4) = (x + h * k3x, v + h * k3v); let (k4x, k4v) = (v4, mu * (1.0 - x4 * x4) * v4 - x4);
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }
    x + v
}

fn kmeans(n: usize, k: usize, iters: usize) -> f64 {
    let (mut px, mut py) = (vec![0.0; n], vec![0.0; n]);
    let (mut cx, mut cy, mut sx, mut sy, mut cnt) = (vec![0.0; k], vec![0.0; k], vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    let mut s = 9i64;
    for i in 0..n {
        s = rng_next(s); let c = s % k as i64;
        s = rng_next(s); px[i] = (c % 4) as f64 * 3.0 + u(s);
        s = rng_next(s); py[i] = (c / 4) as f64 * 3.0 + u(s);
    }
    for c in 0..k { cx[c] = px[c]; cy[c] = py[c]; }
    let mut inertia = 0.0;
    for _ in 0..iters {
        for c in 0..k { sx[c] = 0.0; sy[c] = 0.0; cnt[c] = 0.0; }
        inertia = 0.0;
        for i in 0..n {
            let (mut bc, mut bd) = (0usize, 1e300f64);
            for c in 0..k {
                let (dx, dy) = (px[i] - cx[c], py[i] - cy[c]);
                let d = dx * dx + dy * dy;
                if d < bd { bd = d; bc = c; }
            }
            sx[bc] += px[i]; sy[bc] += py[i]; cnt[bc] += 1.0; inertia += bd;
        }
        for c in 0..k { if cnt[c] > 0.0 { cx[c] = sx[c] / cnt[c]; cy[c] = sy[c] / cnt[c]; } }
    }
    inertia
}

fn timeit<T: std::fmt::Display + Copy, F: Fn() -> T>(name: &str, f: F, fmt: fn(T) -> String) {
    let r = f();
    let mut best = f64::MAX;
    for _ in 0..3 { let t = Instant::now(); black_box(f()); best = best.min(t.elapsed().as_secs_f64()); }
    println!("{}: {:.1} ms  [check {}]", name, best * 1000.0, fmt(r));
}

fn g12(x: f64) -> String { // %.12g
    if x == 0.0 { return "0".into(); }
    let e = x.abs().log10().floor() as i32;
    if e < -4 || e >= 12 { let s = format!("{:.11e}", x); return s; }
    let d = (11 - e).max(0) as usize;
    let s = format!("{:.*}", d, x);
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

fn main() {
    let b = |v: usize| black_box(v);
    timeit("nbody-dense 2000x10", || nbody_dense(b(2000), b(10)), g12);
    timeit("jacobi-2d 1024x200", || jacobi(b(1024), b(200)), g12);
    timeit("matmul 1024", || matmul(b(1024)), g12);
    timeit("lu-pivot 800", || lu_logdet(b(800)), g12);
    timeit("ising 512x50", || ising(b(512), b(50), 0.44), |v| v.to_string());
    timeit("levenshtein 10k", || levenshtein(b(10000)), |v| v.to_string());
    timeit("ode-rk4 20M", || vdp_rk4(b(20000000)), g12);
    timeit("kmeans 200k k16 x20", || kmeans(b(200000), b(16), b(20)), g12);
}
