"""`use stats` — statistics for research: distributions, hypothesis tests, regression, effect sizes, power,
multiple-testing correction, bootstrap and permutation tests.

Numpy only. Every test returns a record with the statistic, the p-value and the quantities a paper needs
(degrees of freedom, confidence intervals, effect sizes). The test suite checks each result against SciPy and
statsmodels.

Distributions: normal, student_t, chi2, f_dist, gamma_dist, beta_dist, exponential, uniform, lognormal,
               binomial, poisson. Each is a record with pdf (pmf), cdf, sf, ppf, sample, mean, var, std.
Descriptive:   describe, quantile, zscore, sem, mean_ci, corr_matrix, cov_matrix
Tests:         ttest_1samp, ttest_ind, ttest_rel, anova_oneway, chi2_contingency, chi2_gof, fisher_exact,
               mannwhitneyu, wilcoxon, kruskal, ks_2samp, jarque_bera, pearsonr, spearmanr, kendalltau,
               proportion_ci, binom_test
Models:        ols (linear regression with SE, t, p, CI, R², F), logit (logistic regression, odds ratios)
Design:        cohen_d, hedges_g, power_ttest, sample_size_ttest
Resampling:    bootstrap_ci, permutation_test
Corrections:   adjust_pvalues (bonferroni, holm, bh, by)
"""
import math

import numpy as np

from . import special as sp
from .errors import RuntimeErr
from .units import Quantity
from .values import Record

ALT = ("two-sided", "less", "greater")


def _x(v, name="data"):
    if isinstance(v, Quantity):
        v = v.v
    a = np.asarray(v, dtype=float).ravel()
    if a.size == 0:
        raise RuntimeErr(f"{name} is empty")
    return a


def _f(v):
    return float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float)


def _alt(a):
    if a not in ALT:
        raise RuntimeErr(f"alternative must be one of {', '.join(ALT)}")
    return a


# =================================================================== distributions
def _rng(seed):
    return np.random.default_rng(None if seed is None else int(seed))


def _dist(name, pdf, cdf, sf, ppf, sample, mean, var, discrete=False, **params):
    fields = dict(name=name, cdf=lambda x: _f(cdf(np.asarray(x, float))),
                  sf=lambda x: _f(sf(np.asarray(x, float))),
                  ppf=lambda p: _f(ppf(np.asarray(p, float))),
                  sample=lambda n=1, seed=None: sample(int(n), _rng(seed)),
                  mean=mean, var=var, std=math.sqrt(var) if var == var and var != math.inf else var,
                  interval=lambda level=0.95: [float(ppf(np.asarray((1 - level) / 2))),
                                               float(ppf(np.asarray((1 + level) / 2)))])
    fields["pmf" if discrete else "pdf"] = lambda x: _f(pdf(np.asarray(x, float)))
    fields.update(params)
    return Record(**fields)


def _phi_cdf(z):
    return 0.5 * np.asarray(sp.erfc(-np.asarray(z) / math.sqrt(2)))


def _phi_ppf(p):
    return -math.sqrt(2) * np.asarray(sp.erfcinv(2 * np.asarray(p, float)))


def normal(mu=0.0, sigma=1.0):
    """Normal distribution N(mu, sigma²)."""
    mu, s = float(mu), float(sigma)
    if s <= 0:
        raise RuntimeErr("sigma must be > 0")
    return _dist("normal", lambda x: np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi)),
                 lambda x: _phi_cdf((x - mu) / s), lambda x: _phi_cdf(-(x - mu) / s),
                 lambda p: mu + s * _phi_ppf(p), lambda n, r: r.normal(mu, s, n), mu, s * s, mu=mu, sigma=s)


def _t_cdf(t, df):
    t = np.asarray(t, float)
    x = df / (df + t * t)
    tail = 0.5 * np.asarray(sp.betainc(df / 2, 0.5, x))
    return np.where(t > 0, 1 - tail, tail)


def _t_sf(t, df):
    return _t_cdf(-np.asarray(t, float), df)


def _t_ppf(p, df):
    p = np.asarray(p, float)
    q = np.minimum(p, 1 - p)
    x = np.asarray(sp.betaincinv(df / 2, 0.5, 2 * q))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sqrt(df * (1 - x) / x)
    return np.where(p < 0.5, -t, np.where(p == 0.5, 0.0, t))


def student_t(df, loc=0.0, scale=1.0):
    """Student's t distribution with df degrees of freedom."""
    df, m, s = float(df), float(loc), float(scale)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    return _dist("t", lambda x: c * (1 + ((x - m) / s) ** 2 / df) ** (-(df + 1) / 2) / s,
                 lambda x: _t_cdf((x - m) / s, df), lambda x: _t_sf((x - m) / s, df),
                 lambda p: m + s * _t_ppf(p, df), lambda n, r: m + s * r.standard_t(df, n),
                 m if df > 1 else math.nan, s * s * df / (df - 2) if df > 2 else math.inf, df=df)


def chi2(df):
    """Chi-squared distribution."""
    k = float(df)
    return _dist("chi2", lambda x: np.where(x > 0, np.exp((k / 2 - 1) * np.log(np.where(x > 0, x, 1)) - x / 2
                                                          - k / 2 * math.log(2) - math.lgamma(k / 2)), 0.0),
                 lambda x: np.asarray(sp.gammainc(k / 2, np.maximum(x, 0) / 2)),
                 lambda x: np.asarray(sp.gammaincc(k / 2, np.maximum(x, 0) / 2)),
                 lambda p: 2 * np.asarray(sp.gammaincinv(k / 2, p)), lambda n, r: r.chisquare(k, n), k, 2 * k, df=k)


def f_dist(d1, d2):
    """F distribution (Fisher–Snedecor)."""
    a, b = float(d1), float(d2)

    def pdf(x):
        xx = np.where(x > 0, x, 1.0)
        lp = (a / 2 * math.log(a) + b / 2 * math.log(b) + (a / 2 - 1) * np.log(xx) - (a + b) / 2 * np.log(b + a * xx)
              - sp.lbeta(a / 2, b / 2))
        return np.where(x > 0, np.exp(lp), 0.0)

    def cdf(x):
        x = np.maximum(x, 0)
        return np.asarray(sp.betainc(a / 2, b / 2, a * x / (a * x + b)))

    def sf(x):
        x = np.maximum(x, 0)
        return np.asarray(sp.betainc(b / 2, a / 2, b / (a * x + b)))

    def ppf(p):
        z = np.asarray(sp.betaincinv(a / 2, b / 2, p))
        with np.errstate(divide="ignore"):
            return b * z / (a * (1 - z))
    mean = b / (b - 2) if b > 2 else math.inf
    var = 2 * b * b * (a + b - 2) / (a * (b - 2) ** 2 * (b - 4)) if b > 4 else math.inf
    return _dist("f", pdf, cdf, sf, ppf, lambda n, r: r.f(a, b, n), mean, var, d1=a, d2=b)


def gamma_dist(shape, scale=1.0):
    """Gamma distribution (shape k, scale θ)."""
    k, th = float(shape), float(scale)
    return _dist("gamma", lambda x: np.where(x > 0, np.exp((k - 1) * np.log(np.where(x > 0, x, 1)) - x / th
                                                           - math.lgamma(k) - k * math.log(th)), 0.0),
                 lambda x: np.asarray(sp.gammainc(k, np.maximum(x, 0) / th)),
                 lambda x: np.asarray(sp.gammaincc(k, np.maximum(x, 0) / th)),
                 lambda p: th * np.asarray(sp.gammaincinv(k, p)), lambda n, r: r.gamma(k, th, n),
                 k * th, k * th * th, shape=k, scale=th)


def beta_dist(a, b):
    """Beta distribution on [0, 1]."""
    a, b = float(a), float(b)
    lb = float(sp.lbeta(a, b))
    return _dist("beta", lambda x: np.where((x > 0) & (x < 1), np.exp((a - 1) * np.log(np.clip(x, 1e-300, 1))
                                                                       + (b - 1) * np.log1p(-np.clip(x, 0, 1 - 1e-16))
                                                                       - lb), 0.0),
                 lambda x: np.asarray(sp.betainc(a, b, np.clip(x, 0, 1))),
                 lambda x: np.asarray(sp.betainc(b, a, 1 - np.clip(x, 0, 1))),
                 lambda p: np.asarray(sp.betaincinv(a, b, p)), lambda n, r: r.beta(a, b, n),
                 a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1)), a=a, b=b)


def exponential(rate=1.0):
    """Exponential distribution with the given rate λ."""
    lam = float(rate)
    return _dist("exponential", lambda x: np.where(x >= 0, lam * np.exp(-lam * x), 0.0),
                 lambda x: np.where(x > 0, -np.expm1(-lam * np.maximum(x, 0)), 0.0),
                 lambda x: np.where(x > 0, np.exp(-lam * np.maximum(x, 0)), 1.0),
                 lambda p: -np.log1p(-p) / lam, lambda n, r: r.exponential(1 / lam, n), 1 / lam, 1 / lam ** 2,
                 rate=lam)


def uniform(low=0.0, high=1.0):
    """Continuous uniform distribution on [low, high]."""
    a, b = float(low), float(high)
    w = b - a
    return _dist("uniform", lambda x: np.where((x >= a) & (x <= b), 1 / w, 0.0),
                 lambda x: np.clip((x - a) / w, 0, 1), lambda x: np.clip((b - x) / w, 0, 1),
                 lambda p: a + w * p, lambda n, r: r.uniform(a, b, n), (a + b) / 2, w * w / 12, low=a, high=b)


def lognormal(mu=0.0, sigma=1.0):
    """Log-normal: log(X) ~ N(mu, sigma²)."""
    m, s = float(mu), float(sigma)
    return _dist("lognormal", lambda x: np.where(x > 0, np.exp(-0.5 * ((np.log(np.where(x > 0, x, 1)) - m) / s) ** 2)
                                                 / (np.where(x > 0, x, 1) * s * math.sqrt(2 * math.pi)), 0.0),
                 lambda x: np.where(x > 0, _phi_cdf((np.log(np.where(x > 0, x, 1)) - m) / s), 0.0),
                 lambda x: np.where(x > 0, _phi_cdf(-(np.log(np.where(x > 0, x, 1)) - m) / s), 1.0),
                 lambda p: np.exp(m + s * _phi_ppf(p)), lambda n, r: r.lognormal(m, s, n),
                 math.exp(m + s * s / 2), (math.exp(s * s) - 1) * math.exp(2 * m + s * s), mu=m, sigma=s)


def _discrete_ppf(cdf, p, hi0):
    p = np.atleast_1d(np.asarray(p, float))
    out = np.empty_like(p)
    for i, q in enumerate(p):
        lo, hi = -1, max(1, int(hi0))
        while cdf(np.asarray(float(hi))) < q:
            hi *= 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cdf(np.asarray(float(mid))) >= q * (1 - 1e-15):
                hi = mid
            else:
                lo = mid
        out[i] = hi
    return out if out.size > 1 else out[0]


def binomial(n, p):
    """Binomial distribution: successes in n trials with probability p."""
    n, p = int(n), float(p)

    def pmf(k):
        k = np.asarray(k, float)
        ok = (k >= 0) & (k <= n) & (k == np.floor(k))
        kk = np.where(ok, k, 0)
        with np.errstate(divide="ignore"):
            lp = (sp.lgamma(n + 1) - np.asarray(sp.lgamma(kk + 1)) - np.asarray(sp.lgamma(n - kk + 1))
                  + kk * math.log(p if p > 0 else 1e-300) + (n - kk) * math.log1p(-p if p < 1 else -1 + 1e-300))
        return np.where(ok, np.exp(lp), 0.0)

    def cdf(k):
        k = np.floor(np.asarray(k, float))
        return np.where(k < 0, 0.0, np.where(k >= n, 1.0,
                        np.asarray(sp.betainc(np.maximum(n - k, 1e-300), k + 1, 1 - p))))

    def sf(k):
        return 1 - cdf(k)
    return _dist("binomial", pmf, cdf, sf, lambda q: _discrete_ppf(cdf, q, n * p + 1),
                 lambda m, r: r.binomial(n, p, m), n * p, n * p * (1 - p), discrete=True, n=n, p=p)


def poisson(rate):
    """Poisson distribution with mean `rate`."""
    lam = float(rate)

    def pmf(k):
        k = np.asarray(k, float)
        ok = (k >= 0) & (k == np.floor(k))
        kk = np.where(ok, k, 0)
        return np.where(ok, np.exp(kk * math.log(lam) - lam - np.asarray(sp.lgamma(kk + 1))), 0.0)

    def cdf(k):
        k = np.floor(np.asarray(k, float))
        return np.where(k < 0, 0.0, np.asarray(sp.gammaincc(np.maximum(k + 1, 1), lam)))
    return _dist("poisson", pmf, cdf, lambda k: 1 - cdf(k), lambda q: _discrete_ppf(cdf, q, lam + 1),
                 lambda m, r: r.poisson(lam, m), lam, lam, discrete=True, rate=lam)


# =================================================================== descriptive
def describe(x):
    """n, mean, std (n−1), sem, min, quartiles, max, skewness, excess kurtosis and a 95 % CI of the mean."""
    a = _x(x)
    n = a.size
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else math.nan
    d = a - m
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    q = np.quantile(a, [0.25, 0.5, 0.75])
    ci = mean_ci(a) if n > 1 else [math.nan, math.nan]
    return Record(n=n, mean=m, std=sd, var=sd * sd, sem=sd / math.sqrt(n) if n > 1 else math.nan,
                  min=float(a.min()), q1=float(q[0]), median=float(q[1]), q3=float(q[2]), max=float(a.max()),
                  skew=float(m3 / m2 ** 1.5) if m2 > 0 else math.nan,
                  kurtosis=float(m4 / m2 ** 2 - 3) if m2 > 0 else math.nan, ci95=ci)


def quantile(x, q):
    """Quantiles (linear interpolation, as numpy/R type 7)."""
    return _f(np.quantile(_x(x), np.asarray(q, float)))


def zscore(x):
    a = _x(x)
    return (a - a.mean()) / a.std(ddof=1)


def sem(x):
    a = _x(x)
    return float(a.std(ddof=1) / math.sqrt(a.size))


def mean_ci(x, level=0.95):
    """Confidence interval of the mean (t distribution)."""
    a = _x(x)
    n = a.size
    h = float(_t_ppf((1 + level) / 2, n - 1)) * a.std(ddof=1) / math.sqrt(n)
    return [float(a.mean() - h), float(a.mean() + h)]


def corr_matrix(X):
    """Pearson correlation matrix of the columns of X."""
    return np.corrcoef(np.asarray(X, float), rowvar=False)


def cov_matrix(X):
    """Sample covariance matrix (n−1) of the columns of X."""
    return np.cov(np.asarray(X, float), rowvar=False)


# =================================================================== tests
def _p_from_t(t, df, alternative):
    if alternative == "two-sided":
        return float(2 * _t_sf(abs(t), df))
    return float(_t_sf(t, df) if alternative == "greater" else _t_cdf(t, df))


def _p_from_z(z, alternative):
    if alternative == "two-sided":
        return float(2 * _phi_cdf(-abs(z)))
    return float(_phi_cdf(-z) if alternative == "greater" else _phi_cdf(z))


def _t_ci(est, se, df, level, alternative):
    if alternative == "two-sided":
        h = float(_t_ppf((1 + level) / 2, df)) * se
        return [est - h, est + h]
    h = float(_t_ppf(level, df)) * se
    return [est - h, math.inf] if alternative == "greater" else [-math.inf, est + h]


def ttest_1samp(x, mu=0.0, alternative="two-sided", level=0.95):
    """One-sample t-test of mean(x) = mu."""
    a, alt = _x(x), _alt(alternative)
    n = a.size
    se = a.std(ddof=1) / math.sqrt(n)
    t = (a.mean() - float(mu)) / se
    return Record(statistic=float(t), p=_p_from_t(t, n - 1, alt), df=n - 1, estimate=float(a.mean()),
                  ci=_t_ci(float(a.mean()), float(se), n - 1, level, alt), se=float(se),
                  cohen_d=float((a.mean() - float(mu)) / a.std(ddof=1)), test="one-sample t-test")


def ttest_ind(a, b, equal_var=False, alternative="two-sided", level=0.95):
    """Two-sample t-test (Welch by default; equal_var=true for Student's)."""
    x, y, alt = _x(a, "a"), _x(b, "b"), _alt(alternative)
    n1, n2 = x.size, y.size
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    diff = x.mean() - y.mean()
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t = diff / se
    return Record(statistic=float(t), p=_p_from_t(t, df, alt), df=float(df), estimate=float(diff),
                  ci=_t_ci(float(diff), se, df, level, alt), se=se, cohen_d=cohen_d(x, y),
                  test="Student t-test" if equal_var else "Welch t-test")


def ttest_rel(a, b, alternative="two-sided", level=0.95):
    """Paired t-test."""
    x, y = _x(a, "a"), _x(b, "b")
    if x.size != y.size:
        raise RuntimeErr("paired samples must have the same length")
    r = ttest_1samp(x - y, 0.0, alternative, level)
    r.fields["test"] = "paired t-test"
    return r


def anova_oneway(*groups):
    """One-way ANOVA: F, p, degrees of freedom, η² and ω²."""
    gs = [_x(g, "group") for g in (groups[0] if len(groups) == 1 and isinstance(groups[0], list)
                                   and np.ndim(groups[0][0]) > 0 else groups)]
    k = len(gs)
    if k < 2:
        raise RuntimeErr("anova_oneway needs at least two groups")
    allv = np.concatenate(gs)
    N = allv.size
    gm = allv.mean()
    ssb = sum(g.size * (g.mean() - gm) ** 2 for g in gs)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in gs)
    dfb, dfw = k - 1, N - k
    F = (ssb / dfb) / (ssw / dfw)
    p = float(f_dist(dfb, dfw).sf(F))
    sst = ssb + ssw
    return Record(statistic=float(F), p=p, df_between=dfb, df_within=dfw, eta2=float(ssb / sst),
                  omega2=float((ssb - dfb * ssw / dfw) / (sst + ssw / dfw)), test="one-way ANOVA")


def chi2_contingency(table, correction=True):
    """Chi-squared test of independence for an r×c table (Yates' correction for 2×2 by default)."""
    O = np.asarray(table, float)
    if O.ndim != 2 or min(O.shape) < 2:
        raise RuntimeErr("chi2_contingency needs a table with at least 2 rows and 2 columns")
    E = O.sum(1, keepdims=True) * O.sum(0, keepdims=True) / O.sum()
    df = (O.shape[0] - 1) * (O.shape[1] - 1)
    d = np.abs(O - E)
    if correction and df == 1:
        d = np.maximum(d - 0.5, 0)
    stat = float((d ** 2 / E).sum())
    n = O.sum()
    v = math.sqrt(stat / (n * (min(O.shape) - 1)))
    return Record(statistic=stat, p=float(chi2(df).sf(stat)), df=df, expected=E, cramers_v=v,
                  test="chi-squared test of independence")


def chi2_gof(observed, expected=None, ddof=0):
    """Chi-squared goodness-of-fit test (expected counts default to uniform)."""
    O = _x(observed)
    E = np.full_like(O, O.mean()) if expected is None else _x(expected) * (O.sum() / _x(expected).sum())
    stat = float(((O - E) ** 2 / E).sum())
    df = O.size - 1 - int(ddof)
    return Record(statistic=stat, p=float(chi2(df).sf(stat)), df=df, test="chi-squared goodness of fit")


def _hyper_pmf(k, M, n, N):
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + math.lgamma(M - n + 1) - math.lgamma(N - k + 1) - math.lgamma(M - n - N + k + 1)
                    - math.lgamma(M + 1) + math.lgamma(N + 1) + math.lgamma(M - N + 1))


def fisher_exact(table, alternative="two-sided"):
    """Fisher's exact test for a 2×2 table. Returns the sample odds ratio and the exact p-value."""
    t = np.asarray(table, dtype=np.int64)
    if t.shape != (2, 2) or (t < 0).any():
        raise RuntimeErr("fisher_exact needs a 2×2 table of counts")
    a, b, c, d = (int(v) for v in t.ravel())
    alt = _alt(alternative)
    M, n, N = a + b + c + d, a + b, a + c
    lo, hi = max(0, N - (M - n)), min(n, N)
    probs = {k: _hyper_pmf(k, M, n, N) for k in range(lo, hi + 1)}
    if alt == "less":
        p = sum(v for k, v in probs.items() if k <= a)
    elif alt == "greater":
        p = sum(v for k, v in probs.items() if k >= a)
    else:
        pa = probs[a]
        p = sum(v for v in probs.values() if v <= pa * (1 + 1e-7))
    with np.errstate(divide="ignore", invalid="ignore"):
        orr = (a * d) / (b * c) if b * c else (math.inf if a * d else math.nan)
    return Record(odds_ratio=float(orr), p=min(1.0, float(p)), test="Fisher's exact test")


def _rank(a):
    """Average ranks (1-based) with ties, and the tie-group sizes."""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(a.size)
    i = 0
    ties = []
    while i < a.size:
        j = i
        while j + 1 < a.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    return ranks, np.asarray(ties, float)


def mannwhitneyu(a, b, alternative="two-sided", continuity=True):
    """Mann–Whitney U test (normal approximation with tie correction). U is for sample a."""
    x, y, alt = _x(a, "a"), _x(b, "b"), _alt(alternative)
    n1, n2 = x.size, y.size
    r, ties = _rank(np.concatenate([x, y]))
    U1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    N = n1 + n2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - (ties ** 3 - ties).sum() / (N * (N - 1))))
    if alt == "two-sided":
        z = (abs(U1 - mu) - (0.5 if continuity else 0)) / sd
        p = float(min(1.0, 2 * _phi_cdf(-z)))
    elif alt == "greater":
        p = float(_phi_cdf(-((U1 - mu) - (0.5 if continuity else 0)) / sd))
    else:
        p = float(_phi_cdf(((U1 - mu) + (0.5 if continuity else 0)) / sd))
    return Record(statistic=float(U1), p=p, rank_biserial=float(2 * U1 / (n1 * n2) - 1),
                  test="Mann-Whitney U")


def wilcoxon(x, y=None, alternative="two-sided", continuity=False):
    """Wilcoxon signed-rank test (normal approximation, zeros dropped, tie correction)."""
    d = _x(x) - (_x(y) if y is not None else 0)
    d = d[d != 0]
    n = d.size
    if n == 0:
        raise RuntimeErr("all differences are zero")
    r, ties = _rank(np.abs(d))
    Wp = r[d > 0].sum()
    Wm = r[d < 0].sum()
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - (ties ** 3 - ties).sum() / 48)
    alt = _alt(alternative)
    c = 0.5 if continuity else 0.0
    if alt == "two-sided":
        T = min(Wp, Wm)
        z = (T - mu + c) / sd if T < mu else 0.0
        p = float(min(1.0, 2 * _phi_cdf(min(z, 0))))
        stat = T
    elif alt == "greater":
        p = float(_phi_cdf(-(Wp - mu - c) / sd))
        stat = Wp
    else:
        p = float(_phi_cdf((Wp - mu + c) / sd))
        stat = Wp
    return Record(statistic=float(stat), p=p, n=n, test="Wilcoxon signed-rank")


def kruskal(*groups):
    """Kruskal–Wallis H test (tie-corrected)."""
    gs = [_x(g, "group") for g in groups]
    allv = np.concatenate(gs)
    N = allv.size
    r, ties = _rank(allv)
    H, i = 0.0, 0
    for g in gs:
        H += r[i:i + g.size].sum() ** 2 / g.size
        i += g.size
    H = 12 / (N * (N + 1)) * H - 3 * (N + 1)
    H /= 1 - (ties ** 3 - ties).sum() / (N ** 3 - N)
    df = len(gs) - 1
    return Record(statistic=float(H), p=float(chi2(df).sf(H)), df=df, test="Kruskal-Wallis H")


def _kolmogorov_sf(x):
    if x <= 0:
        return 1.0
    if x < 1.18:                                       # Jacobi form converges fast for small x
        w = math.sqrt(2 * math.pi) / x
        s = sum(math.exp(-(2 * k - 1) ** 2 * math.pi ** 2 / (8 * x * x)) for k in range(1, 50))
        return 1 - w * s
    return 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * x * x) for k in range(1, 101))


def _ks_exact_inside(m, n, d):
    """P(D < d) for two samples of sizes m, n under H0: lattice paths that stay inside the band (row DP)."""
    tol = 1e-9 / (m * n)
    j = np.arange(n + 1)
    row = np.where(np.abs(0 / m - j / n) < d - tol, 1.0, 0.0)
    row = np.cumprod(row)                       # first row: once outside, every later point is unreachable
    logscale = 0.0
    for i in range(1, m + 1):
        ok = np.abs(i / m - j / n) < d - tol
        new = np.empty(n + 1)
        acc = row[0] if ok[0] else 0.0
        new[0] = acc
        for jj in range(1, n + 1):                 # paths come from the left (same row) or from below
            acc = (new[jj - 1] + row[jj]) if ok[jj] else 0.0
            new[jj] = acc
        mx = new.max()
        if mx > 0:
            new /= mx
            logscale += math.log(mx)
        row = new
    lt = math.lgamma(m + n + 1) - math.lgamma(m + 1) - math.lgamma(n + 1)
    return math.exp(math.log(row[n]) + logscale - lt) if row[n] > 0 else 0.0


def ks_2samp(a, b, method="auto"):
    """Two-sample Kolmogorov–Smirnov test. method "exact" (default when m·n ≤ 10 000), or "asymptotic"."""
    x, y = np.sort(_x(a, "a")), np.sort(_x(b, "b"))
    m, n = x.size, y.size
    allv = np.concatenate([x, y])
    D = float(np.max(np.abs(np.searchsorted(x, allv, "right") / m - np.searchsorted(y, allv, "right") / n)))
    if method == "exact" or (method == "auto" and m * n <= 10000):
        p = 1 - _ks_exact_inside(m, n, D)
        method = "exact"
    else:
        p = _kolmogorov_sf(math.sqrt(m * n / (m + n)) * D)
        method = "asymptotic"
    return Record(statistic=D, p=float(min(1.0, max(0.0, p))), method=method, test="Kolmogorov-Smirnov")


def jarque_bera(x):
    """Jarque–Bera normality test."""
    a = _x(x)
    n = a.size
    d = a - a.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    S, K = m3 / m2 ** 1.5, m4 / m2 ** 2
    jb = n / 6 * (S * S + (K - 3) ** 2 / 4)
    return Record(statistic=float(jb), p=float(chi2(2).sf(jb)), skew=float(S), kurtosis=float(K - 3),
                  test="Jarque-Bera")


def pearsonr(x, y, alternative="two-sided", level=0.95):
    """Pearson correlation with p-value (t test) and Fisher-z confidence interval."""
    a, b = _x(x, "x"), _x(y, "y")
    n = a.size
    if n != b.size or n < 3:
        raise RuntimeErr("pearsonr needs two samples of equal length ≥ 3")
    r = float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))
    df = n - 2
    with np.errstate(divide="ignore"):
        t = r * math.sqrt(df / max(1 - r * r, 1e-300))
    p = _p_from_t(t, df, _alt(alternative))
    z, se = math.atanh(min(max(r, -1 + 1e-16), 1 - 1e-16)), 1 / math.sqrt(n - 3) if n > 3 else math.inf
    h = float(_phi_ppf((1 + level) / 2)) * se
    return Record(r=r, statistic=r, p=p, df=df, ci=[math.tanh(z - h), math.tanh(z + h)], test="Pearson correlation")


def spearmanr(x, y, alternative="two-sided"):
    """Spearman rank correlation (t approximation for the p-value, as SciPy)."""
    a, b = _x(x, "x"), _x(y, "y")
    ra, _ = _rank(a)
    rb, _ = _rank(b)
    r = float(np.clip(np.corrcoef(ra, rb)[0, 1], -1, 1))
    n = a.size
    df = n - 2
    t = r * math.sqrt(df / max(1 - r * r, 1e-300))
    return Record(rho=r, statistic=r, p=_p_from_t(t, df, _alt(alternative)), test="Spearman correlation")


def kendalltau(x, y):
    """Kendall's τ-b with the asymptotic p-value (ties handled)."""
    a, b = _x(x, "x"), _x(y, "y")
    n = a.size
    dx = np.sign(a[:, None] - a[None, :])
    dy = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(n, 1)
    s = float((dx * dy)[iu].sum())
    n0 = n * (n - 1) / 2
    _, tx = _rank(a)
    _, ty = _rank(b)
    n1 = (tx * (tx - 1) / 2).sum()
    n2 = (ty * (ty - 1) / 2).sum()
    tau = s / math.sqrt((n0 - n1) * (n0 - n2))
    v0 = n * (n - 1) * (2 * n + 5)
    vt = (tx * (tx - 1) * (2 * tx + 5)).sum()
    vu = (ty * (ty - 1) * (2 * ty + 5)).sum()
    v1 = (tx * (tx - 1)).sum() * (ty * (ty - 1)).sum() / (2 * n * (n - 1))
    v2 = (tx * (tx - 1) * (tx - 2)).sum() * (ty * (ty - 1) * (ty - 2)).sum() / (9 * n * (n - 1) * (n - 2))
    var = (v0 - vt - vu) / 18 + v1 + v2
    z = s / math.sqrt(var)
    return Record(tau=float(tau), statistic=float(tau), p=float(2 * _phi_cdf(-abs(z))), test="Kendall tau-b")


def proportion_ci(k, n, level=0.95, method="wilson"):
    """Confidence interval of a proportion k/n: "wilson" (default) or "clopper-pearson" (exact)."""
    k, n = int(k), int(n)
    if n <= 0 or not 0 <= k <= n:
        raise RuntimeErr("need 0 ≤ k ≤ n and n > 0")
    a = 1 - level
    if method == "wilson":
        z = float(_phi_ppf(1 - a / 2))
        ph = k / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return [max(0.0, c - h), min(1.0, c + h)]
    if method in ("clopper-pearson", "exact"):
        lo = 0.0 if k == 0 else float(sp.betaincinv(k, n - k + 1, a / 2))
        hi = 1.0 if k == n else float(sp.betaincinv(k + 1, n - k, 1 - a / 2))
        return [lo, hi]
    raise RuntimeErr('method must be "wilson" or "clopper-pearson"')


def binom_test(k, n, p=0.5, alternative="two-sided"):
    """Exact binomial test."""
    k, n, p = int(k), int(n), float(p)
    d = binomial(n, p)
    alt = _alt(alternative)
    if alt == "less":
        pv = float(d.cdf(k))
    elif alt == "greater":
        pv = float(d.sf(k - 1))
    else:
        pk = float(d.pmf(k))
        allp = np.asarray(d.pmf(np.arange(n + 1)))
        pv = float(allp[allp <= pk * (1 + 1e-7)].sum())
    return Record(p=min(1.0, pv), estimate=k / n, ci=proportion_ci(k, n, method="clopper-pearson"),
                  test="exact binomial test")


# =================================================================== regression
def _design(X, add_intercept, names):
    A = np.asarray(X, float)
    if A.ndim == 1:
        A = A[:, None]
    nm = list(names) if names is not None else [f"x{i + 1}" for i in range(A.shape[1])]
    if add_intercept:
        A = np.column_stack([np.ones(A.shape[0]), A])
        nm = ["intercept"] + nm
    return A, nm


def ols(X, y, add_intercept=True, names=None, level=0.95):
    """Ordinary least squares: coefficients with standard errors, t, p and confidence intervals; R², adjusted R²,
    F test, residuals and a predict(X) function."""
    A, nm = _design(X, add_intercept, names)
    yv = _x(y, "y")
    n, k = A.shape
    if n <= k:
        raise RuntimeErr(f"ols needs more observations ({n}) than coefficients ({k})")
    Q, R = np.linalg.qr(A)
    beta = np.linalg.solve(R, Q.T @ yv)
    resid = yv - A @ beta
    df = n - k
    s2 = float(resid @ resid / df)
    Rinv = np.linalg.inv(R)
    cov = s2 * Rinv @ Rinv.T
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = np.array([_p_from_t(v, df, "two-sided") for v in t])
    h = float(_t_ppf((1 + level) / 2, df)) * se
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum()) if add_intercept else float(yv @ yv)
    r2 = 1 - ss_res / ss_tot
    k0 = k - (1 if add_intercept else 0)
    adj = 1 - (1 - r2) * (n - (1 if add_intercept else 0)) / df
    F = ((ss_tot - ss_res) / k0) / (ss_res / df) if k0 > 0 else math.nan
    coef = Record(**{nmi: float(b) for nmi, b in zip(nm, beta)})

    def predict(Xn):
        B, _ = _design(Xn, add_intercept, None)
        return B @ beta
    return Record(coef=coef, params=beta, se=se, t=t, p=p, ci=np.column_stack([beta - h, beta + h]), names=nm,
                  r2=r2, adj_r2=adj, f=float(F), f_p=float(f_dist(k0, df).sf(F)) if k0 > 0 else math.nan,
                  df=df, sigma=math.sqrt(s2), residuals=resid, cov=cov, n=n, predict=predict,
                  aic=float(n * math.log(ss_res / n) + 2 * k), test="ordinary least squares")


def logit(X, y, add_intercept=True, names=None, level=0.95, max_iter=100):
    """Logistic regression by Newton–Raphson (IRLS): coefficients, SE, z, p, CI, odds ratios with CI,
    log-likelihood, AIC and McFadden's pseudo-R²."""
    A, nm = _design(X, add_intercept, names)
    yv = _x(y, "y")
    if not np.all((yv == 0) | (yv == 1)):
        raise RuntimeErr("logit needs a 0/1 outcome")
    beta = np.zeros(A.shape[1])
    ll_old = -math.inf
    for it in range(int(max_iter)):
        eta = A @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-12, None)
        H = A.T @ (A * W[:, None])
        g = A.T @ (yv - mu)
        beta = beta + np.linalg.solve(H, g)
        eta = A @ beta
        ll = float(np.sum(yv * eta - np.logaddexp(0, eta)))
        if abs(ll - ll_old) < 1e-12 * (1 + abs(ll)):
            break
        ll_old = ll
    else:
        raise RuntimeErr("logit did not converge (perfect separation?)")
    mu = 1 / (1 + np.exp(-(A @ beta)))
    H = A.T @ (A * (mu * (1 - mu))[:, None])
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = np.array([_p_from_z(v, "two-sided") for v in z])
    h = float(_phi_ppf((1 + level) / 2)) * se
    p0 = yv.mean()
    ll0 = float(yv.size * (p0 * math.log(p0) + (1 - p0) * math.log(1 - p0))) if 0 < p0 < 1 else 0.0
    return Record(coef=Record(**{a: float(b) for a, b in zip(nm, beta)}), params=beta, se=se, z=z, p=p,
                  ci=np.column_stack([beta - h, beta + h]), odds_ratio=np.exp(beta),
                  odds_ratio_ci=np.exp(np.column_stack([beta - h, beta + h])), names=nm, loglik=ll,
                  aic=-2 * ll + 2 * len(beta), pseudo_r2=1 - ll / ll0 if ll0 else math.nan, iterations=it + 1,
                  predict=lambda Xn: 1 / (1 + np.exp(-(_design(Xn, add_intercept, None)[0] @ beta))),
                  test="logistic regression")


# =================================================================== effect size and power
def cohen_d(a, b):
    """Cohen's d with the pooled standard deviation."""
    x, y = _x(a, "a"), _x(b, "b")
    n1, n2 = x.size, y.size
    sp2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    return float((x.mean() - y.mean()) / math.sqrt(sp2))


def hedges_g(a, b):
    """Hedges' g (small-sample corrected d)."""
    n = _x(a).size + _x(b).size
    return cohen_d(a, b) * (1 - 3 / (4 * n - 9))


def _nct_sf(t, df, nc):
    """P(T > t) for the noncentral t distribution (integration over the chi distribution)."""
    # T = (Z + nc) / sqrt(V/df), V ~ chi2(df): P(T > t) = E_V[ Φ(nc − t·sqrt(V/df)) ]
    m = df + 40 * math.sqrt(2 * df) + 60
    v = np.linspace(0, m, 20001)[1:]
    lpdf = (df / 2 - 1) * np.log(v) - v / 2 - df / 2 * math.log(2) - math.lgamma(df / 2)
    w = np.exp(lpdf)
    f = w * _phi_cdf(nc - t * np.sqrt(v / df))
    h = v[1] - v[0]
    return float(np.sum(f) * h - 0.5 * h * (f[-1]))      # trapezoid (f → 0 at both ends)


def power_ttest(effect, n, alpha=0.05, alternative="two-sided", kind="two-sample"):
    """Power of a t-test for a standardized effect size (Cohen's d), n per group (exact noncentral t)."""
    d, n = float(effect), float(n)
    if kind == "two-sample":
        df, nc = 2 * n - 2, d * math.sqrt(n / 2)
    elif kind in ("one-sample", "paired"):
        df, nc = n - 1, d * math.sqrt(n)
    else:
        raise RuntimeErr('kind must be "two-sample", "one-sample" or "paired"')
    alt = _alt(alternative)
    if alt == "two-sided":
        tc = float(_t_ppf(1 - alpha / 2, df))
        return _nct_sf(tc, df, nc) + (1 - _nct_sf(-tc, df, nc))
    tc = float(_t_ppf(1 - alpha, df))
    return _nct_sf(tc, df, nc) if alt == "greater" else _nct_sf(tc, df, -nc)      # "less": mirror image


def sample_size_ttest(effect, power=0.8, alpha=0.05, alternative="two-sided", kind="two-sample"):
    """Smallest n (per group for two-sample) that reaches the requested power."""
    d = abs(float(effect))
    if d == 0:
        raise RuntimeErr("effect size must be non-zero")
    z = float(_phi_ppf(1 - alpha / (2 if alternative == "two-sided" else 1)) + _phi_ppf(power))
    n = max(2, int((2 if kind == "two-sample" else 1) * (z / d) ** 2) - 2)
    while power_ttest(d, n, alpha, alternative, kind) < power:
        n += 1
    while n > 2 and power_ttest(d, n - 1, alpha, alternative, kind) >= power:
        n -= 1
    return Record(n=n, power=power_ttest(d, n, alpha, alternative, kind), effect=d, alpha=alpha, kind=kind)


# =================================================================== resampling
def bootstrap_ci(data, statistic=None, n=10000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval of statistic(data) (default: the mean)."""
    a = _x(data)
    f = statistic or (lambda s: np.mean(s))
    rng = _rng(seed)
    idx = rng.integers(0, a.size, size=(int(n), a.size))
    if statistic is None:
        vals = a[idx].mean(axis=1)
    else:
        vals = np.array([float(f(a[i])) for i in idx])
    lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
    return Record(estimate=float(f(a)), ci=[float(lo), float(hi)], se=float(vals.std(ddof=1)), n=int(n))


def permutation_test(a, b, statistic=None, n=10000, seed=0, alternative="two-sided"):
    """Two-sample permutation test of statistic(a, b) (default: difference of means)."""
    x, y = _x(a, "a"), _x(b, "b")
    f = statistic or (lambda u, v: np.mean(u) - np.mean(v))
    obs = float(f(x, y))
    allv = np.concatenate([x, y])
    rng = _rng(seed)
    cnt = 0
    alt = _alt(alternative)
    for _ in range(int(n)):
        p = rng.permutation(allv)
        s = float(f(p[:x.size], p[x.size:]))
        cnt += (abs(s) >= abs(obs) - 1e-12) if alt == "two-sided" else (s >= obs - 1e-12) if alt == "greater" \
            else (s <= obs + 1e-12)
    return Record(statistic=obs, p=(cnt + 1) / (int(n) + 1), n=int(n), test="permutation test")


# =================================================================== multiple testing
def adjust_pvalues(p, method="holm"):
    """Adjusted p-values: "bonferroni", "holm", "bh" (Benjamini–Hochberg FDR) or "by" (Benjamini–Yekutieli)."""
    pv = _x(p, "p")
    m = pv.size
    o = np.argsort(pv)
    ps = pv[o]
    if method == "bonferroni":
        adj = np.minimum(1, ps * m)
    elif method == "holm":
        adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * ps))
    elif method in ("bh", "fdr", "by"):
        c = (1 / np.arange(1, m + 1)).sum() if method == "by" else 1.0
        adj = np.minimum(1, np.minimum.accumulate((ps * m * c / np.arange(1, m + 1))[::-1])[::-1])
    else:
        raise RuntimeErr('method must be "bonferroni", "holm", "bh" or "by"')
    out = np.empty(m)
    out[o] = adj
    return out


def module():
    return {k: v for k, v in globals().items()
            if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__
            and k != "module"}
