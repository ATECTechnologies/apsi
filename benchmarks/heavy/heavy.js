// Heavy CPU benchmarks in JavaScript (Node.js) — same algorithms, sizes and random numbers as heavy.c. Run: node heavy.js
"use strict";
const now = () => Number(process.hrtime.bigint()) / 1e9;
const rng = (s) => (s * 16807) % 2147483647;
const U = (s) => s / 2147483647.0;

function nbodyDense(n, steps) {
  const x = new Float64Array(n), y = new Float64Array(n), z = new Float64Array(n), vx = new Float64Array(n),
    vy = new Float64Array(n), vz = new Float64Array(n), ax = new Float64Array(n), ay = new Float64Array(n),
    az = new Float64Array(n), m = new Float64Array(n);
  let s = 42;
  for (let i = 0; i < n; i++) { s = rng(s); x[i] = U(s) - 0.5; s = rng(s); y[i] = U(s) - 0.5; s = rng(s); z[i] = U(s) - 0.5;
    s = rng(s); m[i] = 1.0 / n * (0.5 + U(s)); }
  const dt = 0.001, eps2 = 0.0001;
  for (let st = 0; st < steps; st++) {
    for (let i = 0; i < n; i++) {
      let fx = 0, fy = 0, fz = 0; const xi = x[i], yi = y[i], zi = z[i];
      for (let j = 0; j < n; j++) { const dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi; const d2 = dx * dx + dy * dy + dz * dz + eps2;
        const inv = m[j] / (d2 * Math.sqrt(d2)); fx += dx * inv; fy += dy * inv; fz += dz * inv; }
      ax[i] = fx; ay[i] = fy; az[i] = fz;
    }
    for (let i = 0; i < n; i++) { vx[i] += dt * ax[i]; vy[i] += dt * ay[i]; vz[i] += dt * az[i]; x[i] += dt * vx[i]; y[i] += dt * vy[i]; z[i] += dt * vz[i]; }
  }
  let e = 0; for (let i = 0; i < n; i++) e += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
  return e;
}

function jacobi(n, iters) {
  const a = new Float64Array(n * n), b = new Float64Array(n * n);
  for (let j = 0; j < n; j++) { a[j] = 100.0; b[j] = 100.0; }
  for (let it = 0; it < iters; it++) {
    for (let i = 1; i < n - 1; i++) for (let j = 1; j < n - 1; j++) b[i * n + j] = 0.25 * (a[(i - 1) * n + j] + a[(i + 1) * n + j] + a[i * n + j - 1] + a[i * n + j + 1]);
    for (let i = 1; i < n - 1; i++) for (let j = 1; j < n - 1; j++) a[i * n + j] = 0.25 * (b[(i - 1) * n + j] + b[(i + 1) * n + j] + b[i * n + j - 1] + b[i * n + j + 1]);
  }
  let s = 0; for (let i = 0; i < n * n; i++) s += a[i]; return s;
}

function matmul(n) {
  const a = new Float64Array(n * n), b = new Float64Array(n * n), c = new Float64Array(n * n); let s = 7;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { s = rng(s); a[i * n + j] = U(s) - 0.5; s = rng(s); b[i * n + j] = U(s) - 0.5; }
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { const aik = a[i * n + k]; for (let j = 0; j < n; j++) c[i * n + j] += aik * b[k * n + j]; }
  let t = 0; for (let i = 0; i < n; i++) t += c[i * n + (i * 7) % n]; return t;
}

function luLogdet(n) {
  const a = new Float64Array(n * n); let s = 11;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { s = rng(s); a[i * n + j] = U(s) - 0.5; }
  let logdet = 0;
  for (let k = 0; k < n; k++) {
    let p = k, best = Math.abs(a[k * n + k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(a[i * n + k]); if (v > best) { best = v; p = i; } }
    if (p !== k) for (let j = 0; j < n; j++) { const t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t; }
    const piv = a[k * n + k]; logdet += Math.log(Math.abs(piv));
    for (let i = k + 1; i < n; i++) { const f = a[i * n + k] / piv; a[i * n + k] = f; for (let j = k + 1; j < n; j++) a[i * n + j] -= f * a[k * n + j]; }
  }
  return logdet;
}

function ising(n, sweeps, beta) {
  const sp = new Float64Array(n * n); let s = 5;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { s = rng(s); sp[i * n + j] = s < 1073741824 ? 1.0 : -1.0; }
  const w4 = Math.exp(-2.0 * beta * 4.0), w8 = Math.exp(-2.0 * beta * 8.0);
  for (let sw = 0; sw < sweeps; sw++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const nb = sp[((i + 1) % n) * n + j] + sp[((i + n - 1) % n) * n + j] + sp[i * n + (j + 1) % n] + sp[i * n + (j + n - 1) % n];
    const de = 2.0 * sp[i * n + j] * nb; s = rng(s); const r = U(s);
    if (de <= 0.0 || (de === 4.0 && r < w4) || (de === 8.0 && r < w8)) sp[i * n + j] = -sp[i * n + j];
  }
  let mag = 0; for (let i = 0; i < n * n; i++) mag += sp[i]; return mag;
}

function levenshtein(n) {
  const a = new Float64Array(n), b = new Float64Array(n), prev = new Float64Array(n + 1), cur = new Float64Array(n + 1); let s = 3;
  for (let i = 0; i < n; i++) { s = rng(s); a[i] = s % 4; s = rng(s); b[i] = s % 4; }
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= n; i++) {
    cur[0] = i; const ai = a[i - 1];
    for (let j = 1; j <= n; j++) { const cost = ai === b[j - 1] ? 0.0 : 1.0; cur[j] = Math.min(Math.min(prev[j] + 1.0, cur[j - 1] + 1.0), prev[j - 1] + cost); }
    for (let j = 0; j <= n; j++) prev[j] = cur[j];
  }
  return prev[n];
}

function vdpRk4(steps) {
  let x = 2.0, v = 0.0; const h = 0.0001, mu = 2.0;
  for (let i = 0; i < steps; i++) {
    const k1x = v, k1v = mu * (1.0 - x * x) * v - x;
    const x2 = x + 0.5 * h * k1x, v2 = v + 0.5 * h * k1v, k2x = v2, k2v = mu * (1.0 - x2 * x2) * v2 - x2;
    const x3 = x + 0.5 * h * k2x, v3 = v + 0.5 * h * k2v, k3x = v3, k3v = mu * (1.0 - x3 * x3) * v3 - x3;
    const x4 = x + h * k3x, v4 = v + h * k3v, k4x = v4, k4v = mu * (1.0 - x4 * x4) * v4 - x4;
    x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x); v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
  }
  return x + v;
}

function kmeans(n, k, iters) {
  const px = new Float64Array(n), py = new Float64Array(n), cx = new Float64Array(k), cy = new Float64Array(k),
    sx = new Float64Array(k), sy = new Float64Array(k), cnt = new Float64Array(k); let s = 9;
  for (let i = 0; i < n; i++) { s = rng(s); const c = s % k; s = rng(s); px[i] = (c % 4) * 3.0 + U(s); s = rng(s); py[i] = Math.floor(c / 4) * 3.0 + U(s); }
  for (let c = 0; c < k; c++) { cx[c] = px[c]; cy[c] = py[c]; }
  let inertia = 0;
  for (let it = 0; it < iters; it++) {
    for (let c = 0; c < k; c++) { sx[c] = 0; sy[c] = 0; cnt[c] = 0; }
    inertia = 0;
    for (let i = 0; i < n; i++) {
      let bc = 0, bd = 1e300;
      for (let c = 0; c < k; c++) { const dx = px[i] - cx[c], dy = py[i] - cy[c]; const d = dx * dx + dy * dy; if (d < bd) { bd = d; bc = c; } }
      sx[bc] += px[i]; sy[bc] += py[i]; cnt[bc] += 1.0; inertia += bd;
    }
    for (let c = 0; c < k; c++) if (cnt[c] > 0) { cx[c] = sx[c] / cnt[c]; cy[c] = sy[c] / cnt[c]; }
  }
  return inertia;
}

function T(name, f) {
  const r = f(); let best = 1e9;
  for (let k = 0; k < 3; k++) { const t = now(); f(); const d = now() - t; if (d < best) best = d; }
  console.log(`${name}: ${(best * 1000).toFixed(1)} ms  [check ${Number.isInteger(r) ? r : r.toPrecision(12).replace(/\.?0+$/, "")}]`);
}
T("nbody-dense 2000x10", () => nbodyDense(2000, 10));
T("jacobi-2d 1024x200", () => jacobi(1024, 200));
T("matmul 1024", () => matmul(1024));
T("lu-pivot 800", () => luLogdet(800));
T("ising 512x50", () => ising(512, 50, 0.44));
T("levenshtein 10k", () => levenshtein(10000));
T("ode-rk4 20M", () => vdpRk4(20000000));
T("kmeans 200k k16 x20", () => kmeans(200000, 16, 20));
