# Science, engineering and medicine

Apsi ships the numerical tools of a research lab as standard modules. They need nothing beyond NumPy, accept
plain numbers, tensors and physical quantities, and are checked in the test suite against SciPy, statsmodels,
lifelines, scikit-learn, pydicom and exact analytic solutions (`tests/test_science.py`).

| Module | What it covers |
|---|---|
| [`use special`](#special-functions) | gamma, beta, error functions, incomplete gamma/beta and their inverses, Bessel J/I, elliptic integrals, orthogonal polynomials, spherical harmonics, ζ, E1 |
| [`use stats`](#statistics) | 11 distributions, descriptive statistics, 19 hypothesis tests, OLS and logistic regression, power and sample size, bootstrap, permutation tests, multiple-testing correction |
| [`use signal`](#signal-processing) | Butterworth filters (zero-phase), notch, windows, convolution, peak detection, Welch spectra, spectrograms, Savitzky–Golay, resampling, Hilbert envelope |
| [`use mechanics`](#mechanics) | symplectic integrators, Hamiltonian and Lagrangian dynamics derived by autodiff, N-body gravity, orbits, rigid bodies, normal modes, beams, trusses |
| [`use pde`](#partial-differential-equations) | heat (1-D Crank–Nicolson, 2-D ADI), wave, advection, Poisson (fast transforms), 2-D Navier–Stokes, CG and BiCGSTAB |
| [`use quantum`](#quantum-mechanics) | Schrödinger bound states and time evolution, operators, spins, qubit circuits, density matrices, entanglement, hydrogen |
| [`use physics`](#physics-formulas) | constants and fields, plus radiation, relativity, optics, electromagnetism, thermodynamics and decay formulas with units |
| [`use medical`](#medical-and-health-research) | pharmacokinetics and NCA, epidemic models, diagnostic accuracy and ROC, survival analysis (Kaplan–Meier, log-rank, Cox), clinical formulas, ECG and HRV, DICOM images |
| [`use lab`](#reproducible-research) | seeds, provenance, experiment logs, result formatting, Markdown/LaTeX/CSV tables |

The existing `use sci` (FFT, integration, roots, optimisation, curve fitting with uncertainties, stiff ODEs),
`use linalg`, units, measurement uncertainty and automatic differentiation work together with all of them.

## Special functions

```apsi
use special
print(gamma(4.5), lgamma(200.0), beta(2.0, 3.0), factorial(25), comb(60, 30))
print(erf(1.0), erfinv(0.5), erfcinv(1e-300))
print(gammainc(2.5, 1.0), betainc(2.0, 3.0, 0.4), gammaincinv(2.5, 0.9))
print(besselj(0, 2.404825557695773), besseli(1, 3.0), ellipk(0.5), ellipe(0.5))
print(legendre(5, 0.3), hermite(4, 1.2), laguerre(3, 0.5, 1.0), sph_harm(2, 1, 0.7, 1.1))
```

Every function works element-wise on tensors. Accuracy is close to double precision (relative errors of
1e-16 to 1e-13 against SciPy). `gamma`, `lgamma`, `erf`, `erfc`, `erfinv`, `factorial`, `comb`, `beta`, `besselj` and
`legendre` are also in `use math`; `erf` is a differentiable built-in.

## Statistics

```apsi
use stats
let a = randn(40) + 0.5
let b = randn(35)
let t = ttest_ind(a, b)                       # Welch by default
print(t.statistic, t.p, t.df, t.ci, t.cohen_d)

let d = student_t(12)                          # pdf, cdf, sf, ppf, sample, mean, var, interval
print(d.ppf(0.975), d.interval(0.95))

let fit = ols(X, y, names=["dose", "age"])     # coefficients, SE, t, p, CI, R², adjusted R², F, AIC
print(fit.coef.dose, fit.p, fit.r2)
let lr = logit(X, outcome)                     # odds ratios with CI
print(lr.odds_ratio, lr.odds_ratio_ci)

print(sample_size_ttest(0.5, power=0.8).n)     # 64 per group, exact noncentral t
print(adjust_pvalues([0.01, 0.04, 0.03, 0.2], "bh"))
```

| Area | Functions |
|---|---|
| distributions | `normal`, `student_t`, `chi2`, `f_dist`, `gamma_dist`, `beta_dist`, `exponential`, `uniform`, `lognormal`, `binomial`, `poisson` |
| descriptive | `describe` (n, mean, SD, SEM, quartiles, skewness, excess kurtosis, 95 % CI), `quantile`, `zscore`, `sem`, `mean_ci`, `corr_matrix`, `cov_matrix` |
| tests | `ttest_1samp`, `ttest_ind`, `ttest_rel`, `anova_oneway` (η², ω²), `chi2_contingency` (Cramér's V), `chi2_gof`, `fisher_exact`, `mannwhitneyu`, `wilcoxon`, `kruskal`, `ks_2samp` (exact for small samples), `jarque_bera`, `pearsonr` (Fisher-z CI), `spearmanr`, `kendalltau` (τ-b), `binom_test`, `proportion_ci` (Wilson, Clopper–Pearson) |
| models | `ols`, `logit` (each with `predict`) |
| design | `cohen_d`, `hedges_g`, `power_ttest`, `sample_size_ttest` |
| resampling | `bootstrap_ci`, `permutation_test` |
| corrections | `adjust_pvalues(p, "bonferroni" \| "holm" \| "bh" \| "by")` |

Every test takes `alternative="two-sided" | "less" | "greater"` where it applies, and returns a record with the
statistic, the p-value and what a paper reports (degrees of freedom, confidence intervals, effect sizes). The
results agree with SciPy and statsmodels to 1e-10 or better.

## Signal processing

```apsi
use signal
let sos = butter(4, [0.5, 40.0], "bandpass", 500.0)     # second-order sections
let clean = filtfilt(sos, ecg)                          # zero phase, SciPy-compatible edges
let y = notch(clean, 50.0, 500.0)                        # mains hum
let pk = find_peaks(y, height=0.5, distance=100, prominence=0.3)
let psd = welch(y, 500.0, nperseg=1024)                 # record(f, psd)
let sg = spectrogram(y, 500.0, 256, 128)                 # record(f, t, power)
```

Also: `sosfilt`, `freqz`, `lowpass`, `highpass`, `bandpass`, `bandstop`, `hann`, `hamming`, `blackman`, `kaiser`,
`convolve`, `correlate`, `savgol`, `moving_average`, `detrend`, `resample`, `hilbert`, `envelope`. Filter design,
filtering, peaks, Welch spectra, Savitzky–Golay, resampling and the Hilbert transform agree with `scipy.signal`
to 1e-8 or better. When SciPy is installed its compiled filter loop is used for speed; otherwise a built-in loop
gives the same result.

## Mechanics

```apsi
use mechanics

# any force law: x'' = accel(x, v, t)
fn spring(x) = -x
let r = simulate(spring, tensor([1.0]), tensor([0.0]), 0.01, 10000, "yoshida4")

# Lagrangian mechanics: the equations of motion are derived by automatic differentiation
let g = 9.81
fn L(q, qd) = qd[0]^2 + 0.5*qd[1]^2 + qd[0]*qd[1]*cos(q[0]-q[1]) + 2*g*cos(q[0]) + g*cos(q[1])
let pend = lagrangian_flow(L, tensor([1.5, -0.5]), tensor([0.0, 0.0]), 0.001, 5000)
print(max(pend.energy) - min(pend.energy))            # energy is conserved to ~1e-8

# gravity
let solar = nbody(masses, positions, velocities, 3600.0, 24 * 365, softening=1e3)
print(solar.energy_drift)
```

| Area | Functions |
|---|---|
| integration | `simulate` (Verlet, 4th-order Yoshida, RK4), `hamiltonian_flow(H, q0, p0, …)`, `lagrangian_flow(L, q0, qdot0, …)` |
| gravity and orbits | `nbody`, `kepler_solve`, `orbit_elements`, `orbit_state`, `orbital_period`, `vis_viva`, `hohmann` |
| rigid bodies | `inertia_tensor`, `principal_axes`, `rotation_matrix`, `quat`, `quat_mul`, `quat_conj`, `quat_rotate`, `quat_to_matrix`, `rigid_body` (Euler's equations with torques) |
| vibrations | `normal_modes(K, M)` |
| structures | `beam` (Euler–Bernoulli finite elements: cantilever, simply supported, fixed, propped; point and distributed loads; deflection, slope, moment, shear, reactions), `truss2d` (direct stiffness: displacements, member forces, reactions) |

## Partial differential equations

```apsi
use pde
let r = heat1d(u0, dx, dt, 1000, alpha, bc="neumann", every=100)   # record(t, x, u[frame, i])
let w = wave1d(u0, 0.0, c, dx, dt, 2000, bc="fixed")
let phi = poisson2d(rho / epsilon, dx)                             # exact solution of the 5-point scheme
let flow = navier_stokes2d(omega0, nu=1e-3, dt=1e-3, steps=5000, every=500)
let s = cg(A, b)                                                   # A: matrix or function x -> A·x
```

`heat2d` (ADI), `advection1d` (upwind, Lax–Wendroff), `bicgstab` and `tridiag_solve` (batched Thomas) are also
available. The tests compare every solver with exact solutions: decaying sine modes, standing waves, the discrete
Laplacian and the Taylor–Green vortex (to 1e-15).

## Quantum mechanics

```apsi
use quantum
let x = linspace(-10.0, 10.0, 2001)
let s = schrodinger1d(fn(x) => 0.5 * x * x, x, mass=1.0, n=5, hbar=1.0)   # 0.5, 1.5, 2.5, … (1e-8)
let wave = evolve1d(gaussian_packet(x, -3.0, 0.5, 4.0), fn(x) => 0.0, x, 0.001, 2000, mass=1.0, hbar=1.0)

var psi = ket("00")
psi = apply(psi, gates.H, 0)
psi = apply(psi, gates.CNOT, [0, 1])
print(measure(psi, shots=1000, seed=1), concurrence(psi), entropy(partial_trace(density(psi), [0])))
```

SI units are the default (`mass = m_e`, `hbar = ħ`); pass `hbar=1.0, mass=1.0` for natural units. Bound states use
fourth-order finite differences with ψ = 0 beyond the ends of the grid. Also: `expect_x`, `expect_p`,
`uncertainty`, `probability`, `dagger`, `commutator`, `anticommutator`, `kron`, `expm`, `expm_hermitian`,
`time_evolution`, `pauli`, `spin_ops(s)`, `rx`, `ry`, `rz`, `phase`, `controlled`, `probabilities`, `purity`,
`fidelity`, `bloch`, `ghz`, `bell`, `hydrogen_energy`, `hydrogen_radial`.

## Physics formulas

`use physics` now also contains formulas that take and return quantities with units:

```apsi
use physics
print(wien_peak(5778K))                      # 5.0e-07 m
print(planck(500e-9[m], 5778K))               # spectral radiance
print(lorentz_factor(0.9 * c), time_dilation(1[s], 0.9 * c))
print(coulomb_field([1e-9], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]))
print(biot_savart(loop_points, 2.0, [[0.0, 0.0, 0.0]]))
```

Thermal radiation (`planck`, `wien_peak`, `blackbody_power`), relativity (`lorentz_factor`, `relativistic_energy`,
`relativistic_momentum`, `lorentz_boost`, `time_dilation`, `length_contraction`, `velocity_add`), waves and optics
(`doppler`, `relativistic_doppler`, `snell`, `critical_angle`, `brewster_angle`, `fresnel`, `thin_lens`, `single_slit`,
`double_slit`, `photon_energy`, `de_broglie`), electromagnetism (`coulomb_field`, `coulomb_potential`, `biot_savart`
(exact finite segments), `lorentz_force`, `cyclotron_frequency`, `skin_depth`), thermodynamics
(`ideal_gas_pressure`, `carnot_efficiency`, `maxwell_speed_pdf`, `mean_speed`, `rms_speed`, `heat_conduction`) and
decay (`decay`, `activity`, `half_life_from_rate`).

## Medical and health research

> These functions are for research and education. They are not a medical device and must not be used on their
> own to make decisions about the care of a patient. The text is also available in programs as `DISCLAIMER`.

```apsi
use medical

# pharmacokinetics
let c = pk_one_compartment(500, 40, times, CL=5.0, route="oral", ka=1.1, tau=12, doses=6)
let r = nca(times, measured, dose=500)        # Cmax, Tmax, AUC (linear-up/log-down), λz, t½, CL/F, Vz/F, MRT

# survival analysis
let km = kaplan_meier(time, event)            # survival, Greenwood SE, log-log CI, median
let lr = logrank(time, event, group)
let cox = cox_ph(time, event, X, names=["age", "treatment"])
print(cox.hazard_ratio, cox.hr_ci, cox.p, cox.concordance)

# diagnostic accuracy
print(diagnostic_metrics(90, 20, 10, 180))    # sensitivity, specificity, PPV, NPV, LR±, DOR with CIs
let roc_ = roc(scores, labels)                 # AUC with DeLong CI, Youden threshold
print(roc_compare(model_a, model_b, labels).p)

# signals and images
let beats = ecg_rpeaks(ecg, 360.0)
print(hrv(beats.rr))                           # SDNN, RMSSD, pNN50, LF, HF, LF/HF
let ct = read_dicom("slice.dcm")               # pixels in Hounsfield units, spacing in mm
let view = window_level(ct.pixels, 40, 400)    # soft-tissue window
```

| Area | Functions |
|---|---|
| pharmacokinetics | `pk_one_compartment` (IV bolus, infusion, oral; repeated dosing), `pk_two_compartment`, `nca` |
| epidemiology | `sir`, `seir`, `final_size`, `incidence_rate` (exact Poisson CI), `risk_measures` (RR, OR, RD, NNT with CIs), `mantel_haenszel` (pooled OR/RR, CMH test) |
| diagnostic accuracy | `diagnostic_metrics`, `roc`, `roc_compare` (paired DeLong), `cohen_kappa`, `brier_score` |
| survival | `kaplan_meier`, `logrank` (any number of groups), `cox_ph` (Efron ties), `concordance_index` |
| clinical formulas | `bmi`, `bsa` (Mosteller, DuBois), `egfr_ckd_epi_2021` (race-free), `cockcroft_gault`, `qtc` (Bazett, Fridericia, Framingham, Hodges), `mean_arterial_pressure`, `ideal_body_weight`, `anion_gap` |
| signals | `ecg_rpeaks` (Pan–Tompkins style), `hrv` |
| images | `read_dicom`, `read_dicom_series` (3-D volume sorted along the slice normal), `window_level` |

Kaplan–Meier, the log-rank test and Cox regression agree with lifelines to 1e-9 or better. ROC AUC and kappa
agree with scikit-learn, and the 2×2 and Mantel–Haenszel measures with statsmodels. DICOM pixels are
bit-identical to pydicom for explicit/implicit little-endian and big-endian files.

`read_dicom` reads uncompressed DICOM files and leaves patient-identifying fields out unless you pass
`include_identifiers=true`. Compressed transfer syntaxes (JPEG, JPEG 2000, RLE) give a clear message asking you to
decompress the file first. Reading DICOM needs `--allow-read`, like any other file access.

## Reproducible research

```apsi
use lab
set_seed(42)                                   # Apsi, numpy and Python's random module
let run = experiment("dropout-sweep", record(dropout=0.1, lr=3e-4))   # needs --allow-write
for epoch in 0..20:
    run.log(epoch, record(loss=loss, val_auc=auc))
run.result(record(test_auc=0.912))
run.finish()                                   # experiments/dropout-sweep.jsonl

print(format_measurement(9.8123, 0.0312, "m/s^2"))         # 9.81(3) m/s^2
print(table_latex([record(model="A", auc=0.91), record(model="B", auc=0.87)], caption="Results"))
let runs = load_runs("experiments")                          # every run with params, logs and provenance
```

Every run records its provenance: the Apsi, Python and NumPy versions, OS, CPU, GPUs, seed and time (UTC).
User names, host names and paths are not recorded. Tables come as Markdown (`table_markdown`), LaTeX with
booktabs and escaping (`table_latex`) or CSV (`table_csv`). `sigfig` rounds to significant figures.
