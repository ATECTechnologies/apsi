"""Science, engineering and medical modules, checked against SciPy / statsmodels / lifelines / pydicom when they
are installed and against exact analytic results otherwise."""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                                       # noqa: E402
from apsi import special as SP, stats as ST, dsp as DS, mechanics as ME, pde as PD, quantum as QU   # noqa: E402
from apsi import physics as PH, medical as MD, lab as LB                                            # noqa: E402
from apsi.errors import ApsiError                                  # noqa: E402


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


SCIPY = _has("scipy")
STATSMODELS = _has("statsmodels")
LIFELINES = _has("lifelines")
PYDICOM = _has("pydicom")
SKLEARN = _has("sklearn")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300))) if m.any() else 0.0


class Special(unittest.TestCase):
    def test_exact_values(self):
        self.assertEqual(SP.factorial(20), math.factorial(20))
        self.assertEqual(SP.comb(60, 30), math.comb(60, 30))
        self.assertAlmostEqual(SP.gamma(0.5) ** 2, math.pi, places=13)
        self.assertAlmostEqual(SP.erfinv(SP.erf(0.7)), 0.7, places=14)
        self.assertAlmostEqual(SP.ellipk(0.0), math.pi / 2, places=15)
        self.assertAlmostEqual(SP.zeta(2.0), math.pi ** 2 / 6, places=14)

    @unittest.skipUnless(SCIPY, "SciPy not installed")
    def test_against_scipy(self):
        import scipy.special as S
        rng = np.random.default_rng(0)
        x, a, b, p = (rng.uniform(0.01, 50, 400), rng.uniform(0.05, 60, 400), rng.uniform(0.05, 60, 400),
                      rng.uniform(0, 1, 400))
        self.assertLess(rel(SP.gammainc(a, x), S.gammainc(a, x)), 1e-12)
        self.assertLess(rel(SP.betainc(a, b, p), S.betainc(a, b, p)), 1e-11)
        self.assertLess(rel(SP.gammaincinv(a, p), S.gammaincinv(a, p)), 1e-12)
        self.assertLess(rel(SP.betaincinv(a, b, p), S.betaincinv(a, b, p)), 1e-11)
        y = 10 ** rng.uniform(-300, 0.3, 300)
        self.assertLess(rel(SP.erfcinv(y), S.erfcinv(y)), 1e-14)
        xs = rng.uniform(-60, 60, 300)
        for n in (0, 1, 4, -3, 25):
            self.assertLess(np.max(np.abs(SP.besselj(n, xs) - S.jv(n, xs))), 1e-14)
        for n in (0, 2, 9):
            self.assertLess(rel(SP.besseli(n, xs / 2), S.iv(n, xs / 2)), 1e-13)
        m = rng.uniform(-5, 0.9999, 200)
        self.assertLess(rel(SP.ellipe(m), S.ellipe(m)), 1e-14)
        t = rng.uniform(-1, 1, 200)
        self.assertLess(rel(SP.assoc_legendre(6, 3, t), S.lpmv(3, 6, t)), 1e-12)
        self.assertLess(rel(SP.digamma(x), S.digamma(x)), 1e-12)


class Stats(unittest.TestCase):
    @unittest.skipUnless(SCIPY, "SciPy not installed")
    def test_distributions_and_tests_match_scipy(self):
        import scipy.stats as S
        q = np.linspace(1e-9, 1 - 1e-9, 41)
        for ours, ref in [(ST.normal(1, 2), S.norm(1, 2)), (ST.student_t(4.5), S.t(4.5)), (ST.chi2(7), S.chi2(7)),
                          (ST.f_dist(4, 11), S.f(4, 11)), (ST.beta_dist(2, 5), S.beta(2, 5)),
                          (ST.gamma_dist(2.5, 3), S.gamma(2.5, scale=3))]:
            self.assertLess(rel(ours.ppf(q), ref.ppf(q)), 1e-9, ours.name)
            xx = ref.ppf(q[1:-1])
            self.assertLess(rel(ours.cdf(xx), ref.cdf(xx)), 1e-9, ours.name)
        rng = np.random.default_rng(1)
        a, b, c = rng.normal(0, 1, 30), rng.normal(0.5, 2, 25), rng.normal(0.2, 1, 40)
        for alt in ("two-sided", "less", "greater"):
            r, s = ST.ttest_ind(a, b, alternative=alt), S.ttest_ind(a, b, equal_var=False, alternative=alt)
            self.assertLess(rel([r.statistic, r.p], [s.statistic, s.pvalue]), 1e-10)
            ai, bi = np.round(a * 3), np.round(b * 3)
            r, s = ST.mannwhitneyu(ai, bi, alt), S.mannwhitneyu(ai, bi, alternative=alt, method="asymptotic")
            self.assertLess(rel([r.statistic, r.p], [s.statistic, s.pvalue]), 1e-10)
            r, s = ST.fisher_exact([[3, 9], [11, 2]], alt), S.fisher_exact([[3, 9], [11, 2]], alt)
            self.assertLess(rel([r.p], [s.pvalue]), 1e-10)
        r, s = ST.anova_oneway(a, b, c), S.f_oneway(a, b, c)
        self.assertLess(rel([r.statistic, r.p], [s.statistic, s.pvalue]), 1e-10)
        r, s = ST.chi2_contingency([[12, 5, 9], [7, 14, 3]]), S.chi2_contingency([[12, 5, 9], [7, 14, 3]])
        self.assertLess(rel([r.statistic, r.p], [s.statistic, s.pvalue]), 1e-10)
        r, s = ST.ks_2samp(a, b), S.ks_2samp(a, b, method="exact")
        self.assertLess(rel([r.statistic, r.p], [s.statistic, s.pvalue]), 1e-9)
        r, s = ST.kendalltau(np.round(a[:25]), np.round(b)), S.kendalltau(np.round(a[:25]), np.round(b),
                                                                          method="asymptotic")
        self.assertLess(rel([r.tau, r.p], [s.statistic, s.pvalue]), 1e-10)

    @unittest.skipUnless(STATSMODELS, "statsmodels not installed")
    def test_regression_power_and_corrections_match_statsmodels(self):
        import statsmodels.api as sm
        from statsmodels.stats.multitest import multipletests
        from statsmodels.stats.power import TTestIndPower
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 3))
        y = X @ [1, -2, 0.5] + 0.3 + rng.normal(size=60)
        r, m = ST.ols(X, y), sm.OLS(y, sm.add_constant(X)).fit()
        self.assertLess(rel(np.r_[r.params, r.se, r.p, r.r2, r.f], np.r_[m.params, m.bse, m.pvalues, m.rsquared,
                                                                         m.fvalue]), 1e-8)
        X2 = rng.normal(size=(200, 2))
        yb = (rng.random(200) < 1 / (1 + np.exp(-(X2 @ [1.2, -0.7] - 0.3)))).astype(float)
        r, m = ST.logit(X2, yb), sm.Logit(yb, sm.add_constant(X2)).fit(disp=0)
        self.assertLess(rel(np.r_[r.params, r.se, r.loglik], np.r_[m.params, m.bse, m.llf]), 1e-7)
        p = rng.random(15) ** 3
        for ours, theirs in (("holm", "holm"), ("bh", "fdr_bh"), ("by", "fdr_by")):
            self.assertLess(rel(ST.adjust_pvalues(p, ours), multipletests(p, method=theirs)[1]), 1e-12)
        self.assertLess(abs(ST.power_ttest(0.5, 30) - TTestIndPower().power(0.5, 30, 0.05)), 1e-7)


class Signal(unittest.TestCase):
    @unittest.skipUnless(SCIPY, "SciPy not installed")
    def test_filters_peaks_and_spectra_match_scipy(self):
        import scipy.signal as S
        rng = np.random.default_rng(2)
        x = rng.normal(size=2000) + np.sin(np.arange(2000) * 0.05)
        for N in (1, 3, 4, 6):
            for kind, c in (("lowpass", 40), ("highpass", 5), ("bandpass", [5, 40]), ("bandstop", [45, 55])):
                a, s = DS.butter(N, c, kind, 500), S.butter(N, c, kind, fs=500, output="sos")
                self.assertLess(np.max(np.abs(DS.filtfilt(a, x) - S.sosfiltfilt(s, x))), 1e-8, (N, kind))
        saved = DS._sp_sosfilt
        DS._sp_sosfilt = None                                  # the built-in loop gives the same answer
        try:
            a, s = DS.butter(4, [5, 40], "bandpass", 500), S.butter(4, [5, 40], "bandpass", fs=500, output="sos")
            self.assertLess(np.max(np.abs(DS.filtfilt(a, x) - S.sosfiltfilt(s, x))), 1e-9)
        finally:
            DS._sp_sosfilt = saved
        for kw in ({}, {"distance": 20}, {"prominence": 1.0}, {"height": 0.3, "distance": 7, "prominence": 0.5}):
            self.assertTrue(np.array_equal(DS.find_peaks(x, **kw).indices, S.find_peaks(x, **kw)[0]), kw)
        self.assertLess(rel(DS.welch(x, 500, 256).psd, S.welch(x, 500, nperseg=256)[1]), 1e-10)
        self.assertLess(np.max(np.abs(DS.savgol(x, 31, 3) - S.savgol_filter(x, 31, 3))), 1e-9)
        for n, m in ((2000, 700), (2001, 3000), (2000, 3001)):
            self.assertLess(np.max(np.abs(DS.resample(x[:n], m) - S.resample(x[:n], m))), 1e-10)
        self.assertLess(np.max(np.abs(DS.hilbert(x) - S.hilbert(x))), 1e-10)


class Mechanics(unittest.TestCase):
    def test_integrators_orbits_structures(self):
        r = ME.simulate(lambda x: -x, [1.0], [0.0], 0.01, 628, "yoshida4")
        self.assertLess(abs(r.x[-1, 0] - math.cos(6.28)), 1e-9)
        mu = 3.986e14
        st = ME.orbit_state(7e6, 0.1, 0.5, 1.0, 2.0, 0.3, mu)
        el = ME.orbit_elements(st.r, st.v, mu)
        self.assertLess(max(abs(el.a / 7e6 - 1), abs(el.e - 0.1), abs(el.argp - 2.0), abs(el.nu - 0.3)), 1e-9)
        nb = ME.nbody([5.972e24, 1.0], [[0, 0, 0], [7e6, 0, 0]], [[0, 0, 0], [0, math.sqrt(mu / 7e6), 0]], 10.0, 500,
                      G=mu / 5.972e24)
        self.assertLess(nb.energy_drift, 1e-10)
        rb = ME.rigid_body([1, 2, 3], [0.01, 1, 0.01], [1, 0, 0, 0], 0.01, 2000)
        self.assertLess(np.ptp(rb.energy) / rb.energy[0], 1e-9)
        E, I, L, P = 200e9, 8e-6, 3.0, 1000.0
        self.assertAlmostEqual(ME.beam(L, E, I, "cantilever", [[L, -P]]).max_deflection / (-P * L ** 3 / (3 * E * I)),
                               1.0, places=6)
        self.assertAlmostEqual(ME.beam(L, E, I, "simply", None, -500).max_deflection /
                               (-5 * 500 * L ** 4 / (384 * E * I)), 1.0, places=6)
        t = ME.truss2d([[0, 0], [4, 0], [2, 3]], [[0, 1], [1, 2], [0, 2]], 200e9, 1e-3,
                       [[0, True, True], [1, False, True]], [[2, 0, -10000]])
        self.assertAlmostEqual(t.forces[0], 10000 / 3, places=4)
        self.assertTrue(np.allclose(ME.normal_modes([[2, -1], [-1, 2]], [1, 1]).omega, [1, math.sqrt(3)]))

    def test_lagrangian_and_hamiltonian_from_apsi_functions(self):
        src = """use mechanics
let g = 9.81
fn L(q, qd) = 0.5 * qd * qd - g * (1 - cos(q))
fn H(q, p) = 0.5 * p * p + g * (1 - cos(q))
let a = lagrangian_flow(L, 2.0, 0.0, 0.002, 400)
let b = hamiltonian_flow(H, 2.0, 0.0, 0.002, 400)
assert max(a.energy) - min(a.energy) < 1e-8
assert max(b.energy) - min(b.energy) < 5e-4
assert abs(a.q[400] - b.q[400]) < 1e-4
"""
        Interpreter().run(src, "<t>")


class PDEs(unittest.TestCase):
    def test_solvers_against_exact_solutions(self):
        x = np.linspace(0, 1, 201)
        r = PD.heat1d(np.sin(np.pi * x), x[1], 1e-3, 100, 0.1)
        self.assertLess(np.max(np.abs(r.u[-1] - np.exp(-0.1 * np.pi ** 2 * 0.1) * np.sin(np.pi * x))), 1e-5)
        n = 41
        g = np.linspace(0, 1, n)
        X, Y = np.meshgrid(g, g)
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        r = PD.heat2d(u0, g[1], 1e-3, 40, 0.05)
        self.assertLess(np.max(np.abs(r.u[-1] - np.exp(-0.1 * np.pi ** 2 * 0.04) * u0)), 2e-4)
        r = PD.wave1d(np.sin(np.pi * x), 0, 1.0, x[1], x[1], 300)
        self.assertLess(np.max(np.abs(r.u[-1] - np.cos(np.pi * r.t[-1]) * np.sin(np.pi * x))), 1e-12)
        m, h = 47, 1 / 48
        f = np.random.default_rng(0).normal(size=(m, m))
        u = np.pad(PD.poisson2d(f, h), 1)
        lap = (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]) / h ** 2
        self.assertLess(np.max(np.abs(lap - f)), 1e-9)
        g = np.arange(32) * 2 * np.pi / 32
        X, Y = np.meshgrid(g, g)
        w0 = 2 * np.sin(X) * np.sin(Y)
        r = PD.navier_stokes2d(w0, 0.1, 2 * np.pi, 1e-2, 50)
        self.assertLess(np.max(np.abs(r.omega[-1] - w0 * np.exp(-2 * 0.1 * 0.5))), 1e-12)
        A = np.random.default_rng(2).normal(size=(40, 40))
        S = A @ A.T + 40 * np.eye(40)
        self.assertLess(np.max(np.abs(PD.cg(S, np.ones(40)).x - np.linalg.solve(S, np.ones(40)))), 1e-9)


class Quantum(unittest.TestCase):
    def test_bound_states_evolution_and_qubits(self):
        x = np.linspace(-10, 10, 2001)
        r = QU.schrodinger1d(lambda x: 0.5 * x ** 2, x, mass=1, n=5, hbar=1)
        self.assertLess(np.max(np.abs(r.energies - (np.arange(5) + 0.5))), 1e-6)
        X = np.linspace(-30, 30, 2048, endpoint=False)
        ev = QU.evolve1d(QU.gaussian_packet(X, 0, 1, 2.0), 0.0, X, 0.01, 200, mass=1, hbar=1)
        self.assertLess(abs(ev.x_mean[-1] - 4.0), 1e-9)
        self.assertLess(abs(ev.norm[-1] - 1), 1e-12)
        psi = QU.apply(QU.apply(QU.ket("00"), QU.H, 0), QU.CNOT, [0, 1])
        self.assertTrue(np.allclose(psi, QU.bell(0)))
        self.assertAlmostEqual(QU.entropy(QU.partial_trace(QU.density(psi), [0])), 1.0, places=12)
        self.assertAlmostEqual(QU.concurrence(psi), 1.0, places=12)
        sp = QU.spin_ops(1.5)
        self.assertTrue(np.allclose(QU.commutator(sp.x, sp.y), 1j * sp.z))

    @unittest.skipUnless(SCIPY, "SciPy not installed")
    def test_expm(self):
        import scipy.linalg as SL
        A = np.random.default_rng(0).normal(size=(6, 6)) * 3
        self.assertLess(np.max(np.abs(QU.expm(A) - SL.expm(A))) / np.max(np.abs(SL.expm(A))), 1e-12)


class Physics(unittest.TestCase):
    def test_formulas(self):
        self.assertAlmostEqual(PH.lorentz_factor(0.6 * PH.C), 1.25, places=12)
        self.assertAlmostEqual(float(PH.biot_savart([[0, 0, -1e4], [0, 0, 1e4]], 10.0, [[0.1, 0, 0]]).v[0, 1]),
                               PH.MU0 * 10 / (2 * math.pi * 0.1), delta=1e-12)
        self.assertAlmostEqual(PH.fresnel(1, 1.5, PH.brewster_angle(1, 1.5))[1], 0.0, places=14)
        self.assertAlmostEqual(float(PH.coulomb_field([1e-9], [[0, 0, 0]], [[1, 0, 0]]).v[0, 0]), 8.98755, places=4)


class Medical(unittest.TestCase):
    @unittest.skipUnless(LIFELINES, "lifelines not installed")
    def test_survival_matches_lifelines(self):
        import pandas as pd
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import multivariate_logrank_test
        rng = np.random.default_rng(5)
        n = 150
        T = np.round(rng.exponential(10, n), 1)
        E = (rng.random(n) < 0.7).astype(int)
        G = rng.integers(0, 3, n)
        km, k = MD.kaplan_meier(T, E), KaplanMeierFitter().fit(T, E)
        self.assertLess(rel(km.survival, k.survival_function_.loc[km.times].iloc[:, 0].values), 1e-12)
        self.assertLess(rel(km.ci, k.confidence_interval_.loc[km.times].values), 1e-9)
        lr, r = MD.logrank(T, E, G), multivariate_logrank_test(T, G, E)
        self.assertLess(rel([lr.statistic, lr.p], [r.test_statistic, r.p_value]), 1e-10)
        X = rng.normal(size=(n, 2))
        T2 = np.round(rng.exponential(1 / np.exp(X @ [0.7, -0.4])) * 10) + 1
        E2 = (rng.random(n) < 0.8).astype(int)
        c = MD.cox_ph(T2, E2, X)
        cf = CoxPHFitter().fit(pd.DataFrame({"T": T2, "E": E2, "a": X[:, 0], "b": X[:, 1]}), "T", "E",
                               fit_options={"precision": 1e-14, "max_steps": 200})
        self.assertLess(rel(np.r_[c.params, c.se, c.loglik], np.r_[cf.params_.values, cf.standard_errors_.values,
                                                                    cf.log_likelihood_]), 1e-8)

    @unittest.skipUnless(SKLEARN and STATSMODELS, "scikit-learn / statsmodels not installed")
    def test_diagnostic_accuracy_and_tables(self):
        from sklearn.metrics import roc_auc_score, cohen_kappa_score
        from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
        rng = np.random.default_rng(6)
        y = (rng.random(300) < 0.4).astype(int)
        s = np.round(y * 0.8 + rng.normal(size=300), 1)
        self.assertAlmostEqual(MD.roc(s, y).auc, roc_auc_score(y, s), places=13)
        a = rng.integers(0, 3, 100)
        b = np.where(rng.random(100) < 0.7, a, rng.integers(0, 3, 100))
        self.assertAlmostEqual(MD.cohen_kappa(a, b).kappa, cohen_kappa_score(a, b), places=13)
        tabs = [[[10, 20], [5, 30]], [[12, 18], [8, 25]], [[7, 15], [3, 22]]]
        mh, st_ = MD.mantel_haenszel(tabs), StratifiedTable([np.array(t) for t in tabs])
        self.assertLess(rel([mh.odds_ratio, *mh.or_ci], [st_.oddsratio_pooled, *st_.oddsratio_pooled_confint()]),
                        1e-10)
        rm, t2 = MD.risk_measures(30, 70, 15, 85), Table2x2(np.array([[30, 70], [15, 85]]))
        self.assertLess(rel([rm.rr, *rm.rr_ci, rm.odds_ratio, *rm.or_ci],
                            [t2.riskratio, *t2.riskratio_confint(), t2.oddsratio, *t2.oddsratio_confint()]), 1e-10)

    def test_pharmacokinetics_and_formulas(self):
        t = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24.0, 36, 48])
        c = MD.pk_one_compartment(100, 20, t, ke=0.15, route="oral", ka=1.2)
        r = MD.nca(t, c, dose=100, lambda_points=4)
        self.assertAlmostEqual(r.lambda_z, 0.15, places=6)
        self.assertLess(abs(MD.nca(t, c, dose=100).lambda_z - 0.15), 1e-3)     # automatic point selection
        self.assertLess(abs(r.auc_inf / (100 / 3.0) - 1), 0.01)
        self.assertAlmostEqual(MD.final_size(2.0), 0.7968121300200202, places=12)
        self.assertAlmostEqual(MD.egfr_ckd_epi_2021(0.7, 40, "female"),
                               142 * 0.9938 ** 40 * 1.012, places=10)
        self.assertAlmostEqual(MD.bsa(70, 170), math.sqrt(70 * 170 / 3600), places=12)
        self.assertAlmostEqual(MD.qtc(400, rr=1.0, method="bazett"), 400, places=12)

    def test_ecg_rpeaks(self):
        rng = np.random.default_rng(3)
        fs = 360.0
        beats = np.cumsum(0.8 + rng.normal(0, 0.03, 80))
        beats = beats[beats < 59]
        t = np.arange(int(60 * fs)) / fs
        ecg = 0.15 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.03, t.size)
        for b in beats:
            ecg += 1.2 * np.exp(-((t - b) / 0.012) ** 2) + 0.3 * np.exp(-((t - b - 0.25) / 0.05) ** 2)
        r = MD.ecg_rpeaks(ecg, fs)
        self.assertEqual(len(r.times), len(beats))
        self.assertLess(max(np.min(np.abs(r.times - b)) for b in beats), 0.02)

    @unittest.skipUnless(PYDICOM, "pydicom not installed")
    def test_dicom_matches_pydicom_and_rejects_damage(self):
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian, generate_uid
        d = tempfile.mkdtemp()
        try:
            rng = np.random.default_rng(0)
            fns = MD.install(Interpreter())
            for ts in (ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian):
                meta = FileMetaDataset()
                meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
                meta.MediaStorageSOPInstanceUID = generate_uid()
                meta.TransferSyntaxUID = ts
                ds = Dataset()
                ds.file_meta = meta
                ds.PatientName, ds.Modality = "Doe^Jane", "CT"
                ds.Rows, ds.Columns, ds.SamplesPerPixel = 16, 12, 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 1
                ds.RescaleSlope, ds.RescaleIntercept, ds.PixelSpacing = "1.5", "-1024", [0.7, 0.8]
                item = Dataset()
                item.CodeValue = "1"
                ds.ProcedureCodeSequence = [item]
                arr = rng.integers(-2048, 2047, size=(16, 12)).astype(np.int16)
                ds.PixelData = arr.tobytes() if ts != ExplicitVRBigEndian else arr.byteswap().tobytes()
                p = os.path.join(d, "x.dcm")
                ds.save_as(p, implicit_vr=ts == ImplicitVRLittleEndian, little_endian=ts != ExplicitVRBigEndian,
                           enforce_file_format=True)
                r = fns["read_dicom"](p)
                self.assertTrue(np.array_equal(r.pixels, pydicom.dcmread(p).pixel_array * 1.5 - 1024))
                self.assertNotIn("patient_name", r.fields)
            raw = open(p, "rb").read()
            for k in range(150):
                b = bytearray(raw)
                for _ in range(3):
                    b[int(rng.integers(128, min(600, len(b))))] = int(rng.integers(0, 256))
                if k % 4 == 0:
                    b = b[:int(rng.integers(0, len(b)))]
                with open(p, "wb") as fh:
                    fh.write(b)
                try:
                    fns["read_dicom"](p)
                except ApsiError:
                    pass
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Research(unittest.TestCase):
    def test_lab_tools_in_a_program(self):
        d = tempfile.mkdtemp()
        dq = d.replace("\\", "/")          # a Windows path inside an Apsi string: "\t" would be a tab
        try:
            src = f'''use lab
set_seed(7)
let run = experiment("trial", record(lr=0.1), folder="{dq}")
run.log(1, record(loss=0.4))
run.result(record(acc=0.9))
run.finish()
let runs = load_runs("{dq}")
assert len(runs) == 1
assert runs[0].result.acc == 0.9
assert format_measurement(9.8123, 0.0312) == "9.81(3)"
'''
            Interpreter(permissions={"fs"}).run(src, "<t>")
            with self.assertRaises(ApsiError):
                Interpreter(permissions=set()).run(src, "<t>")
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertIn(r"\toprule", LB.table_latex([[1, "a_b"]], ["x", "y"]))
        self.assertIn(r"a\_b", LB.table_latex([[1, "a_b"]], ["x", "y"]))

    def test_all_modules_load_together(self):
        src = "use special\nuse stats\nuse signal\nuse mechanics\nuse pde\nuse quantum\nuse physics\n" \
              "use medical\nuse lab\nprint(gamma(5.0), normal().ppf(0.5), lorentz_factor(0.0))\n"
        Interpreter().run(src, "<t>")


if __name__ == "__main__":
    unittest.main()
