"""`use medical` — computation for medical and health research.

For research and education. These functions are not a medical device and must not be used on their own to make
decisions about the care of a patient.

Pharmacokinetics
  pk_one_compartment(dose, V, t, CL, ke, route, ka, F, tau, doses, infusion_time)
  pk_two_compartment(dose, V1, k10, k12, k21, t)
  nca(t, c, dose, route)            non-compartmental analysis: Cmax, Tmax, AUC, λz, t½, CL, Vz, MRT
Epidemiology
  sir(beta, gamma, S0, I0, R0, days, dt), seir(beta, sigma, gamma, S0, E0, I0, R0, days, dt), final_size(R0)
  incidence_rate(cases, person_time, level)          exact Poisson confidence interval
  risk_measures(a, b, c, d, level)                   RR, OR, RD, NNT with confidence intervals
  mantel_haenszel(tables, level)                     pooled OR and RR over strata, CMH test
Diagnostic accuracy
  diagnostic_metrics(tp, fp, fn, tn, level)          sensitivity, specificity, PPV, NPV, LR+, LR−, DOR …
  roc(scores, labels, level)                         ROC curve, AUC with DeLong CI, Youden threshold
  roc_compare(scores1, scores2, labels)              paired DeLong test
  cohen_kappa(a, b), brier_score(prob, outcome)
Survival analysis
  kaplan_meier(time, event, level), logrank(time, event, group), cox_ph(time, event, X, names, level),
  concordance_index(time, risk, event)
Clinical formulas
  bmi, bsa (mosteller | dubois), egfr_ckd_epi_2021, cockcroft_gault, qtc (bazett | fridericia | framingham |
  hodges), mean_arterial_pressure, ideal_body_weight, anion_gap
Physiological signals
  ecg_rpeaks(ecg, fs) -> record(indices, times, rr, heart_rate), hrv(rr) -> time- and frequency-domain HRV
Medical images
  read_dicom(path), read_dicom_series(folder), window_level(image, center, width)
"""
import math
import os
import struct

import numpy as np

from . import dsp
from . import stats as st
from .errors import RuntimeErr
from .units import Quantity
from .values import Record

DISCLAIMER = ("For research and education. Not a medical device; do not use on its own to make decisions about "
              "the care of a patient.")


def _v(x):
    if isinstance(x, Quantity):
        x = x.v
    return np.asarray(x, dtype=float)


def _s(x):
    return float(_v(x))


def _out(r):
    r = np.asarray(r, dtype=float)
    return float(r) if r.ndim == 0 else r


def _z(level):
    return float(st._phi_ppf((1 + float(level)) / 2))


# =================================================================== pharmacokinetics
def pk_one_compartment(dose, V, t, CL=None, ke=None, route="iv", ka=None, F=1.0, tau=None, doses=1,
                       infusion_time=None):
    """Plasma concentration of a one-compartment model with first-order elimination.

    route "iv" (bolus), "infusion" (constant rate over infusion_time) or "oral" (first-order absorption ka,
    bioavailability F). Repeated dosing: `doses` doses every `tau` (superposition).
    Give either CL (clearance) or ke (elimination rate constant). Use consistent units (e.g. mg, L, h)."""
    D, Vd, tt = _s(dose), _s(V), _v(t)
    if ke is None:
        if CL is None:
            raise RuntimeErr("give CL or ke")
        k = _s(CL) / Vd
    else:
        k = _s(ke)
    n = int(doses)
    T = _s(tau) if tau is not None else 0.0
    if n > 1 and T <= 0:
        raise RuntimeErr("repeated dosing needs tau > 0")
    c = np.zeros_like(tt, dtype=float)
    for i in range(n):
        s = tt - i * T
        on = s >= 0
        sv = np.where(on, s, 0.0)
        if route == "iv":
            ci = D / Vd * np.exp(-k * sv)
        elif route == "infusion":
            Ti = _s(infusion_time) if infusion_time is not None else None
            if not Ti or Ti <= 0:
                raise RuntimeErr("infusion needs infusion_time > 0")
            R = D / Ti
            ci = np.where(sv <= Ti, R / (k * Vd) * (1 - np.exp(-k * sv)),
                          R / (k * Vd) * (1 - math.exp(-k * Ti)) * np.exp(-k * (sv - Ti)))
        elif route == "oral":
            if ka is None:
                raise RuntimeErr("oral dosing needs ka")
            a = _s(ka)
            f = float(F)
            if abs(a - k) < 1e-12 * max(a, k):
                ci = f * D * k * sv * np.exp(-k * sv) / Vd
            else:
                ci = f * D * a / (Vd * (a - k)) * (np.exp(-k * sv) - np.exp(-a * sv))
        else:
            raise RuntimeErr('route must be "iv", "infusion" or "oral"')
        c += np.where(on, ci, 0.0)
    return _out(c)


def pk_two_compartment(dose, V1, k10, k12, k21, t):
    """IV bolus into a two-compartment model: C(t) = A e^{−αt} + B e^{−βt}. Returns record(c, alpha, beta, A, B,
    t_half_alpha, t_half_beta)."""
    D, V, a10, a12, a21, tt = _s(dose), _s(V1), _s(k10), _s(k12), _s(k21), _v(t)
    s = a10 + a12 + a21
    disc = math.sqrt(s * s - 4 * a10 * a21)
    al, be = (s + disc) / 2, (s - disc) / 2
    A = D / V * (al - a21) / (al - be)
    B = D / V * (a21 - be) / (al - be)
    c = A * np.exp(-al * tt) + B * np.exp(-be * tt)
    return Record(c=_out(c), alpha=al, beta=be, A=A, B=B, t_half_alpha=math.log(2) / al,
                  t_half_beta=math.log(2) / be)


def _lambda_z(t, c, tmax_index, npoints=None):
    """Terminal slope: the last ≥3 positive points after Cmax with the best adjusted R² (ties within 1e-4: more
    points), the rule used by common NCA software."""
    idx = [i for i in range(tmax_index + 1, t.size) if c[i] > 0]
    if len(idx) < 3:
        return None
    best = None
    choices = [int(npoints)] if npoints else range(3, len(idx) + 1)
    for n in choices:
        sel = idx[-n:]
        x, y = t[sel], np.log(c[sel])
        A = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if coef[1] >= 0:
            continue
        res = y - A @ coef
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (res @ res) / ss if ss > 0 else 1.0
        adj = 1 - (1 - r2) * (n - 1) / (n - 2)
        if best is None or adj > best[0] + 1e-4 or (abs(adj - best[0]) <= 1e-4 and n > best[3]):
            best = (adj, -coef[1], r2, n, sel[0])
    return best


def nca(t, c, dose=None, route="extravascular", lambda_points=None):
    """Non-compartmental analysis of one concentration–time profile (linear-up / log-down trapezoids).
    Returns Cmax, Tmax, AUC_last, AUC_inf, AUC_%extrap, AUMC, λz (with R² and points used), t½, MRT and,
    when dose is given, CL (CL/F for extravascular) and Vz (Vz/F)."""
    tt, cc = _v(t), _v(c)
    if tt.size != cc.size or tt.size < 3:
        raise RuntimeErr("nca needs matching time and concentration arrays (at least 3 points)")
    if np.any(np.diff(tt) <= 0):
        raise RuntimeErr("times must be increasing")
    imax = int(np.argmax(cc))
    auc = aumc = 0.0
    for i in range(tt.size - 1):
        dt = tt[i + 1] - tt[i]
        c1, c2 = cc[i], cc[i + 1]
        if c2 < c1 and c2 > 0 and c1 > 0 and i >= imax:            # log-down
            k = math.log(c1 / c2) / dt
            auc += (c1 - c2) / k
            aumc += (c1 * tt[i] - c2 * tt[i + 1]) / k + (c1 - c2) / k ** 2
        else:
            auc += (c1 + c2) / 2 * dt
            aumc += (c1 * tt[i] + c2 * tt[i + 1]) / 2 * dt
    last = int(np.max(np.nonzero(cc > 0)[0])) if np.any(cc > 0) else tt.size - 1
    lz = _lambda_z(tt, cc, imax, lambda_points)
    out = dict(cmax=float(cc[imax]), tmax=float(tt[imax]), auc_last=float(auc), aumc_last=float(aumc),
               clast=float(cc[last]), tlast=float(tt[last]))
    if lz:
        adj, lam, r2, n, first = lz
        auc_inf = auc + cc[last] / lam
        aumc_inf = aumc + cc[last] * tt[last] / lam + cc[last] / lam ** 2
        out.update(lambda_z=float(lam), lambda_z_r2=float(r2), lambda_z_adj_r2=float(adj), lambda_z_points=int(n),
                   t_half=math.log(2) / lam, auc_inf=float(auc_inf), aumc_inf=float(aumc_inf),
                   auc_extrap_percent=float(100 * (auc_inf - auc) / auc_inf), mrt=float(aumc_inf / auc_inf))
        if dose is not None:
            D = _s(dose)
            out.update(cl=D / auc_inf, vz=D / (lam * auc_inf))
            if route == "iv":
                out.update(vss=D / auc_inf * aumc_inf / auc_inf)
    else:
        out.update(lambda_z=math.nan, t_half=math.nan, auc_inf=math.nan, note="terminal phase not identifiable")
    return Record(**out)


# =================================================================== epidemiology
def _rk4_system(f, y0, days, dt):
    n = int(round(_s(days) / _s(dt)))
    h = _s(days) / n
    y = np.asarray(y0, float)
    out = [y.copy()]
    for _ in range(n):
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y.copy())
    return np.arange(n + 1) * h, np.asarray(out)


def sir(beta, gamma, S0, I0, R0=0.0, days=160, dt=0.1):
    """SIR epidemic model (counts or fractions). Returns record(t, S, I, R, peak_day, peak_infected, R0,
    attack_rate)."""
    b, g = _s(beta), _s(gamma)
    N = _s(S0) + _s(I0) + _s(R0)

    def f(y):
        s, i, _ = y
        inf = b * s * i / N
        return np.array([-inf, inf - g * i, g * i])
    t, Y = _rk4_system(f, [_s(S0), _s(I0), _s(R0)], days, dt)
    k = int(np.argmax(Y[:, 1]))
    return Record(t=t, S=Y[:, 0], I=Y[:, 1], R=Y[:, 2], peak_day=float(t[k]), peak_infected=float(Y[k, 1]),
                  R0=b / g, attack_rate=float((Y[-1, 2] - _s(R0)) / N))


def seir(beta, sigma, gamma, S0, E0, I0, R0=0.0, days=200, dt=0.1):
    """SEIR model with incubation rate sigma (1/latent period). Returns record(t, S, E, I, R, peak_day, R0)."""
    b, s_, g = _s(beta), _s(sigma), _s(gamma)
    N = _s(S0) + _s(E0) + _s(I0) + _s(R0)

    def f(y):
        s, e, i, _ = y
        inf = b * s * i / N
        return np.array([-inf, inf - s_ * e, s_ * e - g * i, g * i])
    t, Y = _rk4_system(f, [_s(S0), _s(E0), _s(I0), _s(R0)], days, dt)
    k = int(np.argmax(Y[:, 2]))
    return Record(t=t, S=Y[:, 0], E=Y[:, 1], I=Y[:, 2], R=Y[:, 3], peak_day=float(t[k]),
                  peak_infected=float(Y[k, 2]), R0=b / g, attack_rate=float((Y[-1, 3] - _s(R0)) / N))


def final_size(R0):
    """Final epidemic size z of the SIR model: z = 1 − exp(−R0·z) (fully susceptible population)."""
    r = _s(R0)
    if r <= 1:
        return 0.0
    z = 1 - math.exp(-r)
    for _ in range(200):
        g = z - 1 + math.exp(-r * z)
        d = 1 - r * math.exp(-r * z)
        z2 = z - g / d
        if abs(z2 - z) < 1e-15:
            return z2
        z = z2
    return z


def incidence_rate(cases, person_time, level=0.95, per=1.0):
    """Incidence rate with the exact (Poisson / chi-squared) confidence interval, scaled per `per` person-time."""
    k, T = int(cases), _s(person_time)
    a = 1 - float(level)
    lo = 0.0 if k == 0 else float(st.chi2(2 * k).ppf(a / 2)) / 2
    hi = float(st.chi2(2 * k + 2).ppf(1 - a / 2)) / 2
    f = _s(per)
    return Record(rate=k / T * f, ci=[lo / T * f, hi / T * f], cases=k, person_time=T)


def risk_measures(a, b, c, d, level=0.95):
    """2×2 table: exposed (a cases, b non-cases), unexposed (c cases, d non-cases).
    Relative risk (log CI), odds ratio (Woolf CI), risk difference (Wald CI), NNT and the attributable fraction."""
    a, b, c, d = (float(x) for x in (a, b, c, d))
    z = _z(level)
    r1, r0 = a / (a + b), c / (c + d)
    rr = r1 / r0 if r0 > 0 else math.inf
    se_lrr = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)) if a > 0 and c > 0 else math.inf
    orr = a * d / (b * c) if b * c > 0 else math.inf
    se_lor = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d) if min(a, b, c, d) > 0 else math.inf
    rd = r1 - r0
    se_rd = math.sqrt(r1 * (1 - r1) / (a + b) + r0 * (1 - r0) / (c + d))
    rd_ci = [rd - z * se_rd, rd + z * se_rd]
    nnt = 1 / rd if rd != 0 else math.inf
    nnt_ci = [1 / rd_ci[1], 1 / rd_ci[0]] if rd_ci[0] * rd_ci[1] > 0 else [math.nan, math.nan]
    return Record(risk_exposed=r1, risk_unexposed=r0,
                  rr=rr, rr_ci=[rr * math.exp(-z * se_lrr), rr * math.exp(z * se_lrr)],
                  odds_ratio=orr, or_ci=[orr * math.exp(-z * se_lor), orr * math.exp(z * se_lor)],
                  rd=rd, rd_ci=rd_ci, nnt=nnt, nnt_ci=nnt_ci,
                  attributable_fraction=(rr - 1) / rr if rr not in (0, math.inf) else math.nan)


def mantel_haenszel(tables, level=0.95):
    """Pooled odds ratio and risk ratio over strata of 2×2 tables [[a, b], [c, d]] (Mantel–Haenszel) with the
    Robins–Breslow–Greenland CI, the Cochran–Mantel–Haenszel test and the Breslow–Day-free homogeneity summary."""
    T = np.asarray(tables, float)
    if T.ndim != 3 or T.shape[1:] != (2, 2):
        raise RuntimeErr("mantel_haenszel needs a list of 2×2 tables")
    a, b, c, d = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    n = a + b + c + d
    R = a * d / n
    S = b * c / n
    or_mh = R.sum() / S.sum()
    P = (a + d) / n
    Q = (b + c) / n
    var = ((P * R).sum() / (2 * R.sum() ** 2) + ((P * S + Q * R)).sum() / (2 * R.sum() * S.sum())
           + (Q * S).sum() / (2 * S.sum() ** 2))
    z = _z(level)
    se = math.sqrt(var)
    n1, n0 = a + b, c + d
    m1 = a + c
    rr_num = (a * n0 / n).sum()
    rr_den = (c * n1 / n).sum()
    rr_mh = rr_num / rr_den
    var_rr = ((n1 * n0 * m1 - a * c * n) / n ** 2).sum() / (rr_num * rr_den)
    se_rr = math.sqrt(var_rr)
    Ea = n1 * m1 / n
    Va = n1 * n0 * m1 * (n - m1) / (n ** 2 * (n - 1))
    cmh = float((abs(a.sum() - Ea.sum()) - 0.5) ** 2 / Va.sum())
    return Record(odds_ratio=float(or_mh), or_ci=[or_mh * math.exp(-z * se), or_mh * math.exp(z * se)],
                  rr=float(rr_mh), rr_ci=[rr_mh * math.exp(-z * se_rr), rr_mh * math.exp(z * se_rr)],
                  cmh_statistic=cmh, p=float(st.chi2(1).sf(cmh)), strata=int(T.shape[0]))


# =================================================================== diagnostic accuracy
def _wilson(k, n, level):
    return st.proportion_ci(int(k), int(n), level) if n > 0 else [math.nan, math.nan]


def diagnostic_metrics(tp, fp, fn, tn, level=0.95):
    """Accuracy of a binary test against a reference standard, with confidence intervals
    (Wilson for proportions, log method for likelihood ratios and the diagnostic odds ratio)."""
    tp, fp, fn, tn = (int(x) for x in (tp, fp, fn, tn))
    z = _z(level)
    sens = tp / (tp + fn) if tp + fn else math.nan
    spec = tn / (tn + fp) if tn + fp else math.nan
    ppv = tp / (tp + fp) if tp + fp else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    n = tp + fp + fn + tn
    lrp = sens / (1 - spec) if spec < 1 else math.inf
    lrn = (1 - sens) / spec if spec > 0 else math.inf

    def lr_ci(lr, x1, n1, x2, n2):
        if not (x1 > 0 and x2 > 0) or not math.isfinite(lr):
            return [math.nan, math.nan]
        se = math.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
        return [lr * math.exp(-z * se), lr * math.exp(z * se)]
    dor = (tp * tn) / (fp * fn) if fp * fn else math.inf
    se_dor = math.sqrt(1 / tp + 1 / fp + 1 / fn + 1 / tn) if min(tp, fp, fn, tn) > 0 else math.inf
    return Record(sensitivity=sens, sensitivity_ci=_wilson(tp, tp + fn, level),
                  specificity=spec, specificity_ci=_wilson(tn, tn + fp, level),
                  ppv=ppv, ppv_ci=_wilson(tp, tp + fp, level), npv=npv, npv_ci=_wilson(tn, tn + fn, level),
                  accuracy=(tp + tn) / n, accuracy_ci=_wilson(tp + tn, n, level), prevalence=(tp + fn) / n,
                  lr_positive=lrp, lr_positive_ci=lr_ci(lrp, tp, tp + fn, fp, fp + tn),
                  lr_negative=lrn, lr_negative_ci=lr_ci(lrn, fn, tp + fn, tn, fp + tn),
                  dor=dor, dor_ci=[dor * math.exp(-z * se_dor), dor * math.exp(z * se_dor)]
                  if math.isfinite(se_dor) else [math.nan, math.nan],
                  youden=sens + spec - 1, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def _midrank(x):
    r, _ = st._rank(np.asarray(x, float))
    return r


def _delong_components(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    m, n = pos.size, neg.size
    tz = _midrank(np.r_[pos, neg])
    tx, ty = _midrank(pos), _midrank(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1 - (tz[m:] - ty) / m
    return auc, v10, v01


def roc(scores, labels, level=0.95):
    """ROC analysis: record(fpr, tpr, thresholds, auc, auc_ci (DeLong, logit-free), auc_se, youden_threshold,
    youden_sensitivity, youden_specificity). labels: 1 = disease/positive, 0 = negative."""
    s, y = _v(scores).ravel(), _v(labels).ravel()
    if s.size != y.size or not np.all((y == 0) | (y == 1)) or y.min() == y.max():
        raise RuntimeErr("roc needs scores and 0/1 labels of the same length, with both classes present")
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    distinct = np.r_[np.nonzero(np.diff(ss))[0], ss.size - 1]
    tps = np.cumsum(yy)[distinct]
    fps = (distinct + 1) - tps
    P, N = yy.sum(), yy.size - yy.sum()
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    thr = np.r_[np.inf, ss[distinct]]
    auc, v10, v01 = _delong_components(s, y.astype(int))
    var = v10.var(ddof=1) / v10.size + v01.var(ddof=1) / v01.size
    se = math.sqrt(var)
    z = _z(level)
    j = int(np.argmax(tpr - fpr))
    return Record(fpr=fpr, tpr=tpr, thresholds=thr, auc=float(auc), auc_se=se,
                  auc_ci=[max(0.0, auc - z * se), min(1.0, auc + z * se)],
                  youden_threshold=float(thr[j]), youden_sensitivity=float(tpr[j]),
                  youden_specificity=float(1 - fpr[j]))


def roc_compare(scores1, scores2, labels):
    """Paired DeLong test of two AUCs measured on the same subjects."""
    y = _v(labels).ravel().astype(int)
    a1, x1, y1 = _delong_components(_v(scores1).ravel(), y)
    a2, x2, y2 = _delong_components(_v(scores2).ravel(), y)
    S10 = np.cov(np.vstack([x1, x2]))
    S01 = np.cov(np.vstack([y1, y2]))
    S = S10 / x1.size + S01 / y1.size
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    zs = (a1 - a2) / math.sqrt(var) if var > 0 else 0.0
    return Record(auc1=float(a1), auc2=float(a2), difference=float(a1 - a2), z=float(zs),
                  p=float(2 * st._phi_cdf(-abs(zs))), test="DeLong paired AUC test")


def cohen_kappa(a, b, level=0.95):
    """Cohen's kappa for two raters' categorical labels, with its asymptotic standard error and CI."""
    x, y = list(np.asarray(a).ravel()), list(np.asarray(b).ravel())
    if len(x) != len(y):
        raise RuntimeErr("both raters must rate the same items")
    cats = sorted(set(x) | set(y), key=str)
    k = len(cats)
    ix = {c: i for i, c in enumerate(cats)}
    M = np.zeros((k, k))
    for u, v in zip(x, y):
        M[ix[u], ix[v]] += 1
    n = M.sum()
    p = M / n
    po = np.trace(p)
    r, c = p.sum(1), p.sum(0)
    pe = float(r @ c)
    kappa = (po - pe) / (1 - pe)
    # Fleiss, Cohen & Everitt (1969) standard error
    term1 = sum(p[i, i] * (1 - (r[i] + c[i]) * (1 - kappa)) ** 2 for i in range(k))
    term2 = (1 - kappa) ** 2 * sum(p[i, j] * (c[i] + r[j]) ** 2 for i in range(k) for j in range(k) if i != j)
    term3 = (kappa - pe * (1 - kappa)) ** 2
    se = math.sqrt(max(term1 + term2 - term3, 0) / (n * (1 - pe) ** 2))
    z = _z(level)
    return Record(kappa=float(kappa), se=se, ci=[kappa - z * se, kappa + z * se], observed=float(po), expected=pe)


def brier_score(prob, outcome):
    """Mean squared error of predicted probabilities against 0/1 outcomes."""
    p, y = _v(prob).ravel(), _v(outcome).ravel()
    return float(np.mean((p - y) ** 2))


# =================================================================== survival analysis
def kaplan_meier(time, event, level=0.95):
    """Kaplan–Meier estimate: record(times, at_risk, events, censored, survival, se (Greenwood), ci (log-log),
    median). event: 1 = event observed, 0 = censored."""
    t, e = _v(time).ravel(), _v(event).ravel()
    if t.size != e.size:
        raise RuntimeErr("time and event must have the same length")
    times = np.unique(t)
    at_risk, events, cens, surv, var_sum, ses, lo, hi = [], [], [], [], [], [], [], []
    S, V = 1.0, 0.0
    z = _z(level)
    for u in times:
        n = int((t >= u).sum())
        d = int(((t == u) & (e == 1)).sum())
        c = int(((t == u) & (e == 0)).sum())
        if d:
            S *= 1 - d / n
            V += d / (n * (n - d)) if n > d else math.inf
        at_risk.append(n)
        events.append(d)
        cens.append(c)
        surv.append(S)
        ses.append(S * math.sqrt(V) if math.isfinite(V) else math.nan)
        if 0 < S < 1 and math.isfinite(V):
            lls = math.log(-math.log(S))
            se_l = math.sqrt(V) / abs(math.log(S))
            lo.append(math.exp(-math.exp(lls + z * se_l)))
            hi.append(math.exp(-math.exp(lls - z * se_l)))
        else:
            lo.append(S if S == 1 else 0.0)
            hi.append(S if S == 1 else (1.0 if S > 0 else 0.0))
    surv = np.asarray(surv)
    below = np.nonzero(surv <= 0.5)[0]
    median = float(times[below[0]]) if below.size else math.inf
    return Record(times=times, at_risk=np.asarray(at_risk), events=np.asarray(events), censored=np.asarray(cens),
                  survival=surv, se=np.asarray(ses), ci=np.column_stack([lo, hi]), median=median)


def logrank(time, event, group):
    """Log-rank test comparing survival between two or more groups: record(statistic, p, df, observed,
    expected, groups)."""
    t, e = _v(time).ravel(), _v(event).ravel()
    g = np.asarray(group).ravel()
    labels = sorted(set(g.tolist()), key=str)
    k = len(labels)
    if k < 2:
        raise RuntimeErr("logrank needs at least two groups")
    G = np.array([[1.0 if gi == lab else 0.0 for lab in labels] for gi in g])
    O = np.zeros(k)
    E = np.zeros(k)
    Vm = np.zeros((k, k))
    for u in np.unique(t[e == 1]):
        risk = t >= u
        n = risk.sum()
        nj = G[risk].sum(0)
        dmask = (t == u) & (e == 1)
        d = dmask.sum()
        dj = G[dmask].sum(0)
        O += dj
        E += d * nj / n
        if n > 1:
            Vm += d * (n - d) / (n * n * (n - 1)) * (np.diag(nj) * n - np.outer(nj, nj))
    diff = (O - E)[:-1]
    stat = float(diff @ np.linalg.solve(Vm[:-1, :-1], diff))
    return Record(statistic=stat, p=float(st.chi2(k - 1).sf(stat)), df=k - 1, observed=O, expected=E,
                  groups=labels, test="log-rank")


def cox_ph(time, event, X, names=None, level=0.95, max_iter=50):
    """Cox proportional-hazards regression (Efron's method for ties), Newton–Raphson.
    Returns coefficients, SE, z, p, hazard ratios with CI, log-likelihood, likelihood-ratio test and the
    concordance index."""
    t, e = _v(time).ravel(), _v(event).ravel()
    Z = _v(X)
    if Z.ndim == 1:
        Z = Z[:, None]
    n, p = Z.shape
    nm = list(names) if names is not None else [f"x{i + 1}" for i in range(p)]
    order = np.argsort(-t, kind="mergesort")
    t, e, Z = t[order], e[order], Z[order]
    mu = Z.mean(0)
    Zc = Z - mu

    def loglik(beta, need_derivs=True):
        eta = Zc @ beta
        w = np.exp(eta)
        ll = 0.0
        g = np.zeros(p)
        Hm = np.zeros((p, p))
        rs = 0.0
        rz = np.zeros(p)
        rzz = np.zeros((p, p))
        i = 0
        while i < n:
            j = i
            while j < n and t[j] == t[i]:
                j += 1
            for k in range(i, j):                                # add the whole tied block to the risk set
                rs += w[k]
                rz += w[k] * Zc[k]
                rzz += w[k] * np.outer(Zc[k], Zc[k])
            dmask = e[i:j] == 1
            d = int(dmask.sum())
            if d:
                dz = Zc[i:j][dmask]
                dw = w[i:j][dmask]
                ts = dw.sum()
                tz = (dw[:, None] * dz).sum(0)
                tzz = (dw[:, None, None] * dz[:, :, None] * dz[:, None, :]).sum(0)
                ll += eta[i:j][dmask].sum()
                g += dz.sum(0)
                for l in range(d):
                    f = l / d
                    s0 = rs - f * ts
                    s1 = rz - f * tz
                    s2 = rzz - f * tzz
                    ll -= math.log(s0)
                    g -= s1 / s0
                    Hm -= s2 / s0 - np.outer(s1, s1) / s0 ** 2
            i = j
        return ll, g, Hm

    beta = np.zeros(p)
    ll0, g, Hm = loglik(beta)
    ll = ll0
    def _solve(Hm, g):
        try:
            st_ = np.linalg.solve(-Hm, g)
        except np.linalg.LinAlgError:
            st_ = None
        if st_ is None or not np.all(np.isfinite(st_)):
            raise RuntimeErr("cox_ph: the likelihood has no finite maximum (a covariate separates events from "
                             "non-events, so a hazard ratio is 0 or infinite); remove or combine that covariate, "
                             "or use a penalized model")
        return st_
    for it in range(int(max_iter)):
        step = _solve(Hm, g)
        nb = beta + step
        nll, ng, nH = loglik(nb)
        half = 0
        while nll < ll - 1e-12 and half < 30:                      # step halving keeps the likelihood rising
            step /= 2
            nb = beta + step
            nll, ng, nH = loglik(nb)
            half += 1
        beta, g, Hm = nb, ng, nH
        conv = abs(nll - ll) < 1e-12 * (1 + abs(nll))
        ll = nll
        if conv and np.max(np.abs(step)) < 1e-9:
            break
    if np.max(np.abs(beta)) > 30:
        raise RuntimeErr("cox_ph: the likelihood has no finite maximum (a covariate separates events from "
                         "non-events, so a hazard ratio is 0 or infinite); remove or combine that covariate")
    cov = np.linalg.inv(-Hm)
    se = np.sqrt(np.diag(cov))
    zv = beta / se
    pv = np.array([float(2 * st._phi_cdf(-abs(v))) for v in zv])
    zq = _z(level)
    lr = 2 * (ll - ll0)
    risk = Z @ beta
    return Record(coef=Record(**{a: float(b) for a, b in zip(nm, beta)}), params=beta, se=se, z=zv, p=pv,
                  hazard_ratio=np.exp(beta), hr_ci=np.exp(np.column_stack([beta - zq * se, beta + zq * se])),
                  names=nm, loglik=float(ll), loglik_null=float(ll0), lr_statistic=float(lr),
                  lr_p=float(st.chi2(p).sf(lr)), concordance=concordance_index(t, Z @ beta, e),
                  cov=cov, iterations=it + 1, test="Cox proportional hazards (Efron ties)")


def concordance_index(time, risk, event, higher_is_worse=True):
    """Harrell's C: the fraction of comparable pairs whose predicted risk orders the event times correctly
    (ties in prediction count ½)."""
    t, r, e = _v(time).ravel(), _v(risk).ravel(), _v(event).ravel()
    if not higher_is_worse:
        r = -r
    num = den = 0.0
    for i in range(t.size):
        if e[i] != 1:
            continue
        later = t > t[i]
        tie_t = (t == t[i]) & (e == 0)                           # censored at the same time: still at risk
        comp = later | tie_t
        den += comp.sum()
        num += (r[i] > r[comp]).sum() + 0.5 * (r[i] == r[comp]).sum()
    return float(num / den) if den else math.nan


# =================================================================== clinical formulas
def _kg(x):
    return _s(x)                       # quantities are SI already (kg)


def _cm(x):
    return _s(x) * 100 if isinstance(x, Quantity) else _s(x)


def bmi(weight, height):
    """Body-mass index in kg/m². weight in kg, height in m (numbers) or as quantities (70kg, 175cm)."""
    h = _s(height)
    if not isinstance(height, Quantity) and h > 3:
        raise RuntimeErr("height must be in metres (or give a quantity such as 175cm)")
    return _kg(weight) / (h * h)


def bsa(weight, height, method="mosteller"):
    """Body-surface area in m² (Mosteller or DuBois & DuBois). weight in kg, height in cm (numbers) or quantities."""
    w, hcm = _kg(weight), _cm(height)
    if method == "mosteller":
        return math.sqrt(w * hcm / 3600)
    if method == "dubois":
        return 0.007184 * w ** 0.425 * hcm ** 0.725
    raise RuntimeErr('method must be "mosteller" or "dubois"')


def egfr_ckd_epi_2021(creatinine, age, sex):
    """eGFR (mL/min/1.73 m²) by the race-free CKD-EPI 2021 creatinine equation. creatinine in mg/dL."""
    scr, a = _s(creatinine), _s(age)
    female = str(sex).lower() in ("f", "female", "woman")
    if str(sex).lower() not in ("f", "female", "woman", "m", "male", "man"):
        raise RuntimeErr('sex must be "female" or "male"')
    k, al = (0.7, -0.241) if female else (0.9, -0.302)
    return 142 * min(scr / k, 1) ** al * max(scr / k, 1) ** -1.200 * 0.9938 ** a * (1.012 if female else 1.0)


def cockcroft_gault(age, weight, creatinine, sex):
    """Creatinine clearance (mL/min) by Cockcroft–Gault. weight in kg, creatinine in mg/dL."""
    female = str(sex).lower() in ("f", "female", "woman")
    return (140 - _s(age)) * _kg(weight) / (72 * _s(creatinine)) * (0.85 if female else 1.0)


def qtc(qt, rr=None, hr=None, method="fridericia"):
    """Heart-rate-corrected QT (ms). qt in ms; give rr (s) or hr (beats/min)."""
    q = _s(qt) * (1000 if isinstance(qt, Quantity) else 1)
    if rr is None and hr is None:
        raise RuntimeErr("give rr (s) or hr (bpm)")
    r = _s(rr) if rr is not None else 60.0 / _s(hr)
    h = 60.0 / r
    if method == "bazett":
        return q / math.sqrt(r)
    if method == "fridericia":
        return q / r ** (1 / 3)
    if method == "framingham":
        return q + 154 * (1 - r)
    if method == "hodges":
        return q + 1.75 * (h - 60)
    raise RuntimeErr('method must be "bazett", "fridericia", "framingham" or "hodges"')


def mean_arterial_pressure(systolic, diastolic):
    """MAP ≈ DBP + (SBP − DBP)/3 (same unit as the inputs, usually mmHg)."""
    return _s(diastolic) + (_s(systolic) - _s(diastolic)) / 3


def ideal_body_weight(height, sex):
    """Devine ideal body weight (kg); height in cm (number) or as a quantity."""
    inches = _cm(height) / 2.54
    base = 45.5 if str(sex).lower() in ("f", "female", "woman") else 50.0
    return base + 2.3 * (inches - 60)


def anion_gap(sodium, chloride, bicarbonate, potassium=None):
    """Serum anion gap (mmol/L): Na − (Cl + HCO3) [+ K if given]."""
    return _s(sodium) + (_s(potassium) if potassium is not None else 0.0) - _s(chloride) - _s(bicarbonate)


# =================================================================== physiological signals
def ecg_rpeaks(ecg, fs, min_rr=0.25):
    """R-peak detection in a single-lead ECG (Pan–Tompkins style: 5–15 Hz band-pass, derivative, squaring,
    150 ms integration, adaptive threshold, refinement to the true maximum).
    Returns record(indices, times, rr (s), heart_rate (bpm), mean_heart_rate)."""
    x = _v(ecg).ravel()
    f = _s(fs)
    if x.size < int(2 * f):
        raise RuntimeErr("need at least 2 seconds of ECG")
    y = dsp.bandpass(x, 5.0, min(15.0, 0.45 * f), f, order=2)
    dy = np.gradient(y)
    sq = dy * dy
    integ = dsp.moving_average(sq, max(1, int(0.15 * f)))
    dist = max(1, int(min_rr * f))
    cand = dsp.find_peaks(integ, distance=dist)
    if cand.count == 0:
        return Record(indices=np.zeros(0, int), times=np.zeros(0), rr=np.zeros(0), heart_rate=np.zeros(0),
                      mean_heart_rate=math.nan)
    h = integ[cand.indices]
    thr = 0.35 * np.percentile(h, 90)
    peaks = cand.indices[h >= thr]
    w = max(1, int(0.075 * f))
    sgn = 1.0 if np.abs(np.max(y)) >= np.abs(np.min(y)) else -1.0
    refined = []
    for p in peaks:
        lo, hi = max(0, p - w), min(x.size, p + w + 1)
        seg = sgn * (x[lo:hi] - np.median(x[lo:hi]))
        refined.append(lo + int(np.argmax(seg)))
    idx = np.unique(np.asarray(refined, dtype=np.int64))
    keep = [idx[0]] if idx.size else []
    for p in idx[1:]:
        if p - keep[-1] >= dist:
            keep.append(p)
    idx = np.asarray(keep, dtype=np.int64)
    times = idx / f
    rr = np.diff(times)
    hr = 60.0 / rr if rr.size else np.zeros(0)
    return Record(indices=idx, times=times, rr=rr, heart_rate=hr,
                  mean_heart_rate=float(60.0 / rr.mean()) if rr.size else math.nan)


def hrv(rr, frequency_domain=True):
    """Heart-rate variability from RR intervals in seconds: mean RR, SDNN, RMSSD, SDSD, pNN50, mean HR (ms and
    bpm) and, with ≥ 60 s of data, LF (0.04–0.15 Hz), HF (0.15–0.4 Hz) power and LF/HF from a 4 Hz resampled
    tachogram (Welch)."""
    r = _v(rr).ravel()
    if r.size < 3:
        raise RuntimeErr("need at least 3 RR intervals")
    ms = r * 1000
    d = np.diff(ms)
    out = dict(mean_rr=float(ms.mean()), sdnn=float(ms.std(ddof=1)), rmssd=float(np.sqrt(np.mean(d * d))),
               sdsd=float(d.std(ddof=1)), pnn50=float(100 * np.mean(np.abs(d) > 50)),
               mean_hr=float(60000 / ms.mean()))
    if frequency_domain and r.sum() >= 60:
        t = np.cumsum(r)
        fs = 4.0
        grid = np.arange(t[0], t[-1], 1 / fs)
        tach = np.interp(grid, t, ms)
        w = dsp.welch(tach - tach.mean(), fs, nperseg=min(256, grid.size))
        df = w.f[1] - w.f[0]
        lf = float(w.psd[(w.f >= 0.04) & (w.f < 0.15)].sum() * df)
        hf = float(w.psd[(w.f >= 0.15) & (w.f < 0.4)].sum() * df)
        out.update(lf=lf, hf=hf, lf_hf=lf / hf if hf > 0 else math.inf)
    return Record(**out)


# =================================================================== DICOM
_LONG_VR = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"UC", b"UR", b"OD", b"OL", b"OV", b"SV", b"UV"}
_WANT = {
    (0x0008, 0x0060): "modality", (0x0008, 0x0070): "manufacturer", (0x0008, 0x0020): "study_date",
    (0x0008, 0x103E): "series_description", (0x0018, 0x0050): "slice_thickness", (0x0018, 0x0088): "slice_spacing",
    (0x0020, 0x0013): "instance_number", (0x0020, 0x0032): "image_position", (0x0020, 0x0037): "image_orientation",
    (0x0028, 0x0002): "samples_per_pixel", (0x0028, 0x0004): "photometric", (0x0028, 0x0008): "frames",
    (0x0028, 0x0010): "rows", (0x0028, 0x0011): "columns", (0x0028, 0x0030): "pixel_spacing",
    (0x0028, 0x0100): "bits_allocated", (0x0028, 0x0101): "bits_stored", (0x0028, 0x0103): "pixel_representation",
    (0x0028, 0x1050): "window_center", (0x0028, 0x1051): "window_width", (0x0028, 0x1052): "rescale_intercept",
    (0x0028, 0x1053): "rescale_slope", (0x0002, 0x0010): "transfer_syntax", (0x0028, 0x0006): "planar",
}
_IDENT = {(0x0010, 0x0010): "patient_name", (0x0010, 0x0020): "patient_id", (0x0010, 0x0030): "patient_birth_date",
          (0x0010, 0x0040): "patient_sex", (0x0008, 0x0080): "institution"}
_US = {(0x0028, 0x0002), (0x0028, 0x0010), (0x0028, 0x0011), (0x0028, 0x0100), (0x0028, 0x0101),
       (0x0028, 0x0103), (0x0028, 0x0006)}
_TS = {"1.2.840.10008.1.2": ("<", False), "1.2.840.10008.1.2.1": ("<", True), "1.2.840.10008.1.2.2": (">", True)}


def _dcm_value(tag, raw, end, vr):
    if tag in _US:
        return struct.unpack(end + "H", raw[:2])[0] if len(raw) >= 2 else None
    s = raw.decode("latin-1").strip("\x00 ")
    if tag in ((0x0028, 0x0030), (0x0020, 0x0032), (0x0020, 0x0037), (0x0028, 0x1050), (0x0028, 0x1051)):
        try:
            v = [float(p) for p in s.split("\\") if p.strip()]
            return v if len(v) != 1 else v[0]
        except ValueError:
            return s
    if tag in ((0x0018, 0x0050), (0x0018, 0x0088), (0x0028, 0x1052), (0x0028, 0x1053)):
        try:
            return float(s.split("\\")[0])
        except ValueError:
            return s
    if tag in ((0x0020, 0x0013), (0x0028, 0x0008)):
        try:
            return int(float(s))
        except ValueError:
            return s
    return s


def _parse_dicom(buf, include_identifiers):
    if len(buf) >= 132 and buf[128:132] == b"DICM":
        pos = 132
    else:
        pos = 0                                          # files without the preamble (old ACR-NEMA style)
    info, ident = {}, {}
    end, explicit = "<", True
    meta_end = None
    pixel = None
    n = len(buf)

    def need(k):
        if pos + k > n:
            raise RuntimeErr("DICOM file is truncated or damaged")

    while pos + 8 <= n:
        in_meta = meta_end is None or pos < meta_end
        e_end, e_exp = ("<", True) if in_meta and buf[pos:pos + 2] == b"\x02\x00" else (end, explicit)
        g, el = struct.unpack(e_end + "HH", buf[pos:pos + 4])
        tag = (g, el)
        if g == 0xFFFE:                                  # item / delimiter at top level: skip header
            pos += 8
            continue
        if e_exp:
            vr = buf[pos + 4:pos + 6]
            if vr in _LONG_VR:
                need(12)
                ln = struct.unpack(e_end + "I", buf[pos + 8:pos + 12])[0]
                pos += 12
            else:
                ln = struct.unpack(e_end + "H", buf[pos + 6:pos + 8])[0]
                pos += 8
        else:
            vr = b"??"
            ln = struct.unpack(e_end + "I", buf[pos + 4:pos + 8])[0]
            pos += 8
        if tag == (0x7FE0, 0x0010):
            if ln == 0xFFFFFFFF:
                raise RuntimeErr("compressed (encapsulated) DICOM pixel data is not supported; convert the file "
                                 "to an uncompressed transfer syntax first (e.g. with dcmdjpeg or gdcmconv)")
            need(ln)
            pixel = buf[pos:pos + ln]
            pos += ln
            continue
        if ln == 0xFFFFFFFF:                             # sequence of undefined length: skip to its delimiter
            depth = 1
            while depth and pos + 8 <= n:
                gg, ee = struct.unpack(e_end + "HH", buf[pos:pos + 4])
                ll = struct.unpack(e_end + "I", buf[pos + 4:pos + 8])[0]
                if (gg, ee) == (0xFFFE, 0xE0DD):
                    depth -= 1
                    pos += 8
                elif (gg, ee) == (0xFFFE, 0xE000):
                    pos += 8 if ll == 0xFFFFFFFF else 8 + ll     # undefined-length items end with E00D
                elif (gg, ee) == (0xFFFE, 0xE00D):
                    pos += 8
                else:                                     # element inside an undefined-length item
                    if e_exp:
                        v2 = buf[pos + 4:pos + 6]
                        if v2 in _LONG_VR:
                            l2 = struct.unpack(e_end + "I", buf[pos + 8:pos + 12])[0]
                            pos += 12
                        else:
                            l2 = struct.unpack(e_end + "H", buf[pos + 6:pos + 8])[0]
                            pos += 8
                    else:
                        l2 = struct.unpack(e_end + "I", buf[pos + 4:pos + 8])[0]
                        pos += 8
                    if l2 == 0xFFFFFFFF:
                        depth += 1
                    else:
                        pos += l2
            continue
        need(ln)
        raw = buf[pos:pos + ln]
        pos += ln
        if tag == (0x0002, 0x0000) and ln == 4:
            meta_end = pos + struct.unpack("<I", raw)[0]
        if tag in _WANT:
            info[_WANT[tag]] = _dcm_value(tag, raw, e_end, vr)
            if tag == (0x0002, 0x0010):
                ts = info["transfer_syntax"]
                if ts not in _TS:
                    raise RuntimeErr(f"DICOM transfer syntax {ts} (compressed) is not supported; convert the file to "
                                     f"an uncompressed syntax first (e.g. with dcmdjpeg or gdcmconv)")
                end, explicit = _TS[ts]
        elif include_identifiers and tag in _IDENT:
            ident[_IDENT[tag]] = _dcm_value(tag, raw, e_end, vr)
    return info, ident, pixel, end


def _dicom_pixels(info, pixel, end):
    rows, cols = info.get("rows"), info.get("columns")
    if pixel is None or not rows or not cols:
        return None
    bits = info.get("bits_allocated", 16)
    signed = info.get("pixel_representation", 0) == 1
    spp = info.get("samples_per_pixel", 1) or 1
    frames = info.get("frames", 1) or 1
    dt = {8: "i1" if signed else "u1", 16: "i2" if signed else "u2", 32: "i4" if signed else "u4"}.get(bits)
    if dt is None:
        raise RuntimeErr(f"{bits}-bit DICOM pixels are not supported")
    need = rows * cols * spp * frames * (bits // 8)
    if len(pixel) < need:
        raise RuntimeErr(f"DICOM pixel data is shorter than {rows}×{cols}×{frames} images (file damaged?)")
    a = np.frombuffer(pixel, dtype=np.dtype(dt).newbyteorder(end), count=rows * cols * spp * frames)
    if spp > 1:
        if info.get("planar", 0) == 1:
            a = a.reshape(frames, spp, rows, cols).transpose(0, 2, 3, 1)
        else:
            a = a.reshape(frames, rows, cols, spp)
    else:
        a = a.reshape(frames, rows, cols)
    if frames == 1:
        a = a[0]
    stored = info.get("bits_stored", bits)
    a = a.astype(np.float64)
    if signed and stored < bits:                      # sign-extend from the stored bit width
        a = np.where(a >= 2 ** (stored - 1), a - 2 ** stored, a)
    slope = info.get("rescale_slope", 1.0)
    icpt = info.get("rescale_intercept", 0.0)
    slope = slope if isinstance(slope, float) else 1.0
    icpt = icpt if isinstance(icpt, float) else 0.0
    return a * slope + icpt


def window_level(image, center, width):
    """Map intensities (e.g. Hounsfield units) to [0, 1] for display with a window centre and width
    (e.g. soft tissue 40/400, lung −600/1500, bone 400/1800, brain 40/80)."""
    a = _v(image)
    c, w = _s(center), _s(width)
    return np.clip((a - (c - 0.5 - (w - 1) / 2)) / (w - 1), 0, 1)


def install(interp):
    """Medical functions that read files go through the program's read permissions."""
    from .security import check_path

    def read_dicom(path, include_identifiers=False):
        """Read an uncompressed DICOM image: record(pixels (rescaled, e.g. Hounsfield units), rows, columns,
        pixel_spacing (mm), slice_thickness, modality, … ). Patient-identifying fields are left out unless
        include_identifiers=true."""
        full = check_path(interp, path, "read", "reading DICOM files")
        with open(full, "rb") as fh:
            buf = fh.read()
        try:
            info, ident, pixel, end = _parse_dicom(buf, include_identifiers)
            px = _dicom_pixels(info, pixel, end)
        except (struct.error, ValueError, IndexError, TypeError, OverflowError) as e:
            raise RuntimeErr(f"{os.path.basename(full)} is not a readable DICOM file ({e})")
        if px is None:
            raise RuntimeErr(f"{os.path.basename(full)} has no image data")
        info.update(ident)
        info["pixels"] = px
        return Record(**info)

    def read_dicom_series(folder, include_identifiers=False):
        """Read every DICOM slice in a folder and stack them into a volume sorted by position along the slice
        normal (or instance number): record(volume[z, y, x], spacing [z, y, x] mm, positions, info)."""
        full = check_path(interp, folder, "read", "reading DICOM files")
        slices = []
        for name in sorted(os.listdir(full)):
            p = os.path.join(full, name)
            if not os.path.isfile(p):
                continue
            with open(p, "rb") as fh:
                head = fh.read(132)
            if len(head) < 132 or head[128:132] != b"DICM":
                continue
            slices.append(read_dicom(p, include_identifiers))
        if not slices:
            raise RuntimeErr("no DICOM files found")
        ori = slices[0].fields.get("image_orientation")
        if isinstance(ori, list) and len(ori) == 6:
            nrm = np.cross(ori[:3], ori[3:])
            key = lambda s: float(np.dot(nrm, s.fields.get("image_position", [0, 0, 0])))
        else:
            key = lambda s: float(s.fields.get("instance_number", 0))
        slices.sort(key=key)
        vol = np.stack([s.pixels for s in slices])
        pos = np.array([key(s) for s in slices])
        dz = float(np.median(np.diff(pos))) if len(slices) > 1 else float(slices[0].fields.get("slice_thickness", 1.0))
        ps = slices[0].fields.get("pixel_spacing", [1.0, 1.0])
        ps = ps if isinstance(ps, list) else [ps, ps]
        meta = {k: v for k, v in slices[0].fields.items() if k != "pixels"}
        return Record(volume=vol, spacing=[abs(dz), ps[0], ps[1]], positions=pos, count=len(slices),
                      info=Record(**meta))

    return {"read_dicom": read_dicom, "read_dicom_series": read_dicom_series}


def module():
    out = {k: v for k, v in globals().items()
           if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == __name__
           and k not in ("module", "install")}
    out["DISCLAIMER"] = DISCLAIMER
    return out
