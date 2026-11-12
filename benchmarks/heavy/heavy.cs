// Heavy CPU benchmarks in C# (.NET) — same algorithms, sizes and random numbers as heavy.c. Built in Release by run.py.
using System;
using System.Diagnostics;
using System.Globalization;

static class Heavy {
  static double Now() => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;
  static long Rng(long s) => (s * 16807) % 2147483647L;
  static double U(long s) => (double)s / 2147483647.0;

  static double NbodyDense(int n, int steps) {
    double[] x = new double[n], y = new double[n], z = new double[n], vx = new double[n], vy = new double[n],
      vz = new double[n], ax = new double[n], ay = new double[n], az = new double[n], m = new double[n];
    long s = 42;
    for (int i = 0; i < n; i++) { s = Rng(s); x[i] = U(s) - 0.5; s = Rng(s); y[i] = U(s) - 0.5; s = Rng(s); z[i] = U(s) - 0.5;
      s = Rng(s); m[i] = 1.0 / n * (0.5 + U(s)); }
    double dt = 0.001, eps2 = 0.0001;
    for (int st = 0; st < steps; st++) {
      for (int i = 0; i < n; i++) {
        double fx = 0, fy = 0, fz = 0, xi = x[i], yi = y[i], zi = z[i];
        for (int j = 0; j < n; j++) { double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi; double d2 = dx * dx + dy * dy + dz * dz + eps2;
          double inv = m[j] / (d2 * Math.Sqrt(d2)); fx += dx * inv; fy += dy * inv; fz += dz * inv; }
        ax[i] = fx; ay[i] = fy; az[i] = fz;
      }
      for (int i = 0; i < n; i++) { vx[i] += dt * ax[i]; vy[i] += dt * ay[i]; vz[i] += dt * az[i]; x[i] += dt * vx[i]; y[i] += dt * vy[i]; z[i] += dt * vz[i]; }
    }
    double e = 0; for (int i = 0; i < n; i++) e += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    return e;
  }

  static double Jacobi(int n, int iters) {
    double[] a = new double[n * n], b = new double[n * n];
    for (int j = 0; j < n; j++) { a[j] = 100.0; b[j] = 100.0; }
    for (int it = 0; it < iters; it++) {
      for (int i = 1; i < n - 1; i++) for (int j = 1; j < n - 1; j++) b[i * n + j] = 0.25 * (a[(i - 1) * n + j] + a[(i + 1) * n + j] + a[i * n + j - 1] + a[i * n + j + 1]);
      for (int i = 1; i < n - 1; i++) for (int j = 1; j < n - 1; j++) a[i * n + j] = 0.25 * (b[(i - 1) * n + j] + b[(i + 1) * n + j] + b[i * n + j - 1] + b[i * n + j + 1]);
    }
    double s = 0; for (int i = 0; i < n * n; i++) s += a[i]; return s;
  }

  static double Matmul(int n) {
    double[] a = new double[n * n], b = new double[n * n], c = new double[n * n]; long s = 7;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { s = Rng(s); a[i * n + j] = U(s) - 0.5; s = Rng(s); b[i * n + j] = U(s) - 0.5; }
    for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) { double aik = a[i * n + k]; for (int j = 0; j < n; j++) c[i * n + j] += aik * b[k * n + j]; }
    double t = 0; for (int i = 0; i < n; i++) t += c[i * n + (i * 7) % n]; return t;
  }

  static double LuLogdet(int n) {
    double[] a = new double[n * n]; long s = 11;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { s = Rng(s); a[i * n + j] = U(s) - 0.5; }
    double logdet = 0;
    for (int k = 0; k < n; k++) {
      int p = k; double best = Math.Abs(a[k * n + k]);
      for (int i = k + 1; i < n; i++) { double v = Math.Abs(a[i * n + k]); if (v > best) { best = v; p = i; } }
      if (p != k) for (int j = 0; j < n; j++) { double t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t; }
      double piv = a[k * n + k]; logdet += Math.Log(Math.Abs(piv));
      for (int i = k + 1; i < n; i++) { double f = a[i * n + k] / piv; a[i * n + k] = f; for (int j = k + 1; j < n; j++) a[i * n + j] -= f * a[k * n + j]; }
    }
    return logdet;
  }

  static long Ising(int n, int sweeps, double beta) {
    double[] sp = new double[n * n]; long s = 5;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { s = Rng(s); sp[i * n + j] = s < 1073741824L ? 1.0 : -1.0; }
    double w4 = Math.Exp(-2.0 * beta * 4.0), w8 = Math.Exp(-2.0 * beta * 8.0);
    for (int sw = 0; sw < sweeps; sw++) for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      double nb = sp[((i + 1) % n) * n + j] + sp[((i + n - 1) % n) * n + j] + sp[i * n + (j + 1) % n] + sp[i * n + (j + n - 1) % n];
      double de = 2.0 * sp[i * n + j] * nb; s = Rng(s); double r = U(s);
      if (de <= 0.0 || (de == 4.0 && r < w4) || (de == 8.0 && r < w8)) sp[i * n + j] = -sp[i * n + j];
    }
    long mag = 0; for (int i = 0; i < n * n; i++) mag += (long)sp[i]; return mag;
  }

  static long Levenshtein(int n) {
    double[] a = new double[n], b = new double[n], prev = new double[n + 1], cur = new double[n + 1]; long s = 3;
    for (int i = 0; i < n; i++) { s = Rng(s); a[i] = (double)(s % 4); s = Rng(s); b[i] = (double)(s % 4); }
    for (int j = 0; j <= n; j++) prev[j] = j;
    for (int i = 1; i <= n; i++) {
      cur[0] = i; double ai = a[i - 1];
      for (int j = 1; j <= n; j++) { double cost = ai == b[j - 1] ? 0.0 : 1.0; cur[j] = Math.Min(Math.Min(prev[j] + 1.0, cur[j - 1] + 1.0), prev[j - 1] + cost); }
      for (int j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return (long)prev[n];
  }

  static double VdpRk4(int steps) {
    double x = 2.0, v = 0.0, h = 0.0001, mu = 2.0;
    for (int i = 0; i < steps; i++) {
      double k1x = v, k1v = mu * (1.0 - x * x) * v - x;
      double x2 = x + 0.5 * h * k1x, v2 = v + 0.5 * h * k1v, k2x = v2, k2v = mu * (1.0 - x2 * x2) * v2 - x2;
      double x3 = x + 0.5 * h * k2x, v3 = v + 0.5 * h * k2v, k3x = v3, k3v = mu * (1.0 - x3 * x3) * v3 - x3;
      double x4 = x + h * k3x, v4 = v + h * k3v, k4x = v4, k4v = mu * (1.0 - x4 * x4) * v4 - x4;
      x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x); v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }
    return x + v;
  }

  static double Kmeans(int n, int k, int iters) {
    double[] px = new double[n], py = new double[n], cx = new double[k], cy = new double[k], sx = new double[k], sy = new double[k], cnt = new double[k];
    long s = 9;
    for (int i = 0; i < n; i++) { s = Rng(s); long c = s % k; s = Rng(s); px[i] = (double)(c % 4) * 3.0 + U(s); s = Rng(s); py[i] = (double)(c / 4) * 3.0 + U(s); }
    for (int c = 0; c < k; c++) { cx[c] = px[c]; cy[c] = py[c]; }
    double inertia = 0;
    for (int it = 0; it < iters; it++) {
      for (int c = 0; c < k; c++) { sx[c] = 0; sy[c] = 0; cnt[c] = 0; }
      inertia = 0;
      for (int i = 0; i < n; i++) {
        int bc = 0; double bd = 1e300;
        for (int c = 0; c < k; c++) { double dx = px[i] - cx[c], dy = py[i] - cy[c]; double d = dx * dx + dy * dy; if (d < bd) { bd = d; bc = c; } }
        sx[bc] += px[i]; sy[bc] += py[i]; cnt[bc] += 1.0; inertia += bd;
      }
      for (int c = 0; c < k; c++) if (cnt[c] > 0) { cx[c] = sx[c] / cnt[c]; cy[c] = sy[c] / cnt[c]; }
    }
    return inertia;
  }

  static string Fmt(object r) {
    if (r is long l) return l.ToString(CultureInfo.InvariantCulture);
    return ((double)r).ToString("G12", CultureInfo.InvariantCulture);
  }

  static void T(string name, Func<object> f) {
    object r = f(); double best = 1e9;
    for (int k = 0; k < 3; k++) { double t = Now(); f(); double d = Now() - t; if (d < best) best = d; }
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0}: {1:F1} ms  [check {2}]", name, best * 1000, Fmt(r)));
  }

  static void Main() {
    T("nbody-dense 2000x10", () => NbodyDense(2000, 10));
    T("jacobi-2d 1024x200", () => Jacobi(1024, 200));
    T("matmul 1024", () => Matmul(1024));
    T("lu-pivot 800", () => LuLogdet(800));
    T("ising 512x50", () => Ising(512, 50, 0.44));
    T("levenshtein 10k", () => Levenshtein(10000));
    T("ode-rk4 20M", () => VdpRk4(20000000));
    T("kmeans 200k k16 x20", () => Kmeans(200000, 16, 20));
  }
}
