# AGENTS.md — Molten Salt Optical Absorption and Radiative Heat Transfer Model

## Purpose and scope

This repository develops physically informed, temperature-dependent models of optical absorption in molten salts. Its central objective is to connect molten-salt structure—especially pair distribution functions (PDFs/RDFs)—to spectral absorption, refractive index, Planck-mean radiative properties, and engineering heat-transfer consequences.

The intended long-term workflow is:

```text
molten-salt composition + temperature range
        ↓
structural information (PDF/RDF, MD, or validated surrogate descriptors)
        ↓
spectral absorption coefficient κλ(T) and refractive index nλ(T)
        ↓
Planck-mean κP(T) and nეფf(T)
        ↓
radiative heat-transfer calculations, measurement corrections, and CFD inputs
```

The repository currently includes or may include workflows for:

- PDF-informed calculation of cation–anion vibrational frequencies and equilibrium separations;
- Lorentz-oscillator modeling of fundamental vibrational absorption;
- Urbach-tail electronic absorption;
- multiphonon absorption modeling;
- inhomogeneous broadening derived from PDF peak widths;
- fitting of damping and multiphonon parameters to experimental optical data;
- correlation-based prediction for salts without direct optical measurements;
- Planck-mean absorption coefficient and refractive-index calculations;
- radiative-conductivity corrections for laser-flash analysis (LFA), variable-gap, and related thermal-conductivity measurements;
- coupling of temperature-dependent optical properties to radiative heat-transfer models.

This project must distinguish carefully between:

1. **quantities directly constrained by experiment**;
2. **quantities calculated from molecular-dynamics structure**;
3. **parameters fitted to optical measurements**;
4. **correlations used for extrapolation or prediction**; and
5. **hypotheses about molecular energy-transfer pathways requiring validation**.

The agent’s role is to improve scientific reliability, interpretability, reproducibility, extensibility, and software quality—without overstating what the model demonstrates.

---

# Core scientific objectives

## 1. Build a physically reasonable absorption model

The absorption model should represent the total spectral absorption coefficient as:

\[
\kappa_{\lambda}(T)
=
\kappa_{\mathrm{vib},\lambda}(T)
+
\kappa_{\mathrm{elec},\lambda}(T)
+
\kappa_{\mathrm{multi},\lambda}(T)
\]

where the terms represent fundamental vibrational absorption, electronic absorption, and multiphonon absorption, respectively. The existing model uses Lorentz dielectric oscillators for vibrational absorption, an Urbach tail for electronic absorption, and a multiphonon exponential-tail model. (Source: abs_model.txt, Page: 1; Source: abs_model.txt, Page: 4)

The agent should preserve this decomposition unless a scientific-method change is explicitly proposed, justified, documented, and tested.

## 2. Connect optical response to molten-salt structure

The intended structural pathway is:

\[
g_{ij}(r)
\rightarrow
V_{\mathrm{PMF}}(r)
\rightarrow
k_{ij}
\rightarrow
\omega_{0,ij}
\rightarrow
\kappa_{\lambda}(T)
\]

The current absorption workflow derives the potential of mean force from:

\[
V_{\mathrm{PMF}}(r) = -k_B T \ln g(r)
\]

and obtains a local harmonic force constant from PMF curvature near the first RDF peak, followed by:

\[
\omega_0=\sqrt{\frac{k}{\mu}}
\]

where \(\mu\) is the pair reduced mass. (Source: abs_model.txt, Page: 2; Source: abs_model.txt, Page: 6)

The agent should recognize that the PMF is an effective, environment-dependent quantity rather than a bare pair potential. It includes many-body, screening, entropic, and cage effects. Therefore, PDF-derived frequencies and force constants are **model-based structural descriptors**, not automatically exact vibrational observables.

## 3. Develop validated prediction capability

The eventual goal is not merely to fit a few known salts. It is to support either:

1. a generalized, composition- and temperature-aware optical-absorption model for broad molten-salt composition space; or  
2. a lower-fidelity but physics-informed model that has been adequately benchmarked against higher-fidelity calculations and experimental optical data.

Prediction must be accompanied by explicit uncertainty, applicability limits, and provenance. A predicted \(\kappa_P(T)\) must never be presented as equivalent in confidence to a directly measured value.

## 4. Enable radiative heat-transfer applications

The principal engineering output is the Planck-mean absorption coefficient:

\[
\kappa_P(T)
=
\frac{\int \kappa_{\lambda}(T) I_{b,\lambda}(T)\,d\lambda}
{\int I_{b,\lambda}(T)\,d\lambda}
\]

together with a Planck-weighted effective refractive index. (Source: abs_model.txt, Page: 5)

These quantities are intended for radiative heat-transfer calculations, including corrections to apparent thermal conductivity measured in semi-transparent molten salts.

For LFA-scale samples, the radiative contribution should use an optical-thickness-aware bridging formulation:

\[
k_{\mathrm{rad}}
=
\frac{16 n^2 \sigma T^3}
{3\kappa_P + 4/L}
\]

rather than indiscriminately applying the optically thick Rosseland limit. (Source: LFA_case_study.txt, Page: 1; Source: LFA_case_study.txt, Page: 4)

---

# Core principles

## 1. Preserve scientific traceability

Every output value must be traceable to:

1. requested composition and temperature;
2. structural input source;
3. PDF/RDF source and preprocessing;
4. molecular-dynamics potential and simulation protocol, where applicable;
5. experimental optical dataset(s), where applicable;
6. fitted parameters and fit configuration;
7. spectral grid and wavelength range;
8. Planck-mean integration method;
9. analysis-code version and settings;
10. uncertainty or sensitivity treatment;
11. output files from which final values were calculated.

A Planck-mean absorption coefficient must never be a “magic number.” A user should be able to determine whether it came from:

```text
direct experimental fit
PDF-informed model constrained by experiment
correlation-based predicted parameters
endmember/interpolation approximation
user-supplied empirical correlation
```

and identify the associated evidence and limitations.

---

## 2. Do not confuse a fit with validation

A successful fit to an optical dataset does not validate:

- the model at unmeasured wavelengths;
- the model at different temperatures;
- the model in a different salt family;
- the PDF-to-frequency mapping;
- the multiphonon mechanism;
- the transferability of fitted parameters; or
- the model’s predicted Planck mean.

Use careful language:

- “is fitted to,”
- “reproduces the selected dataset within the stated metric,”
- “is consistent with the available data,”
- “suggests a possible structural relationship,”
- “is a candidate transferable descriptor,”
- “requires validation against independent spectra,”
- “is extrapolated beyond the measured wavelength range,”
- “is sensitive to the structural input and parameter-transfer assumptions.”

Avoid:

- “proves,”
- “demonstrates the mechanism,”
- “validates generally,”
- “predicts accurately” without independent test data,
- “phonon transport” as a literal description of liquid-state heat transfer unless the context and approximation are carefully stated.

---

## 3. Treat molecular energy-transfer mechanisms cautiously

Molten salts are disordered liquids. Concepts such as individual bonds, optical phonons, normal modes, coherence, and mode overlap can be useful approximations or diagnostics, but should not be treated as crystal-like facts without qualification.

When connecting structure to absorption or radiative heat transfer:

- distinguish local cation–anion vibrational descriptors from propagating lattice phonons;
- distinguish optical absorption from thermal conduction;
- distinguish radiative transport from molecular energy transport;
- identify whether a proposed mechanism is directly modeled, inferred, or speculative;
- avoid interpreting correlations as causal mechanisms without independent evidence.

For example, a correlation between \(\alpha/\omega_0\) and \(r_0\) may be physically motivated by anharmonic interionic-potential arguments, but it remains a transferability hypothesis that requires independent validation for additional salts and mixtures. The existing workflow explicitly uses family-dependent correlations and a shifted chloride intercept; this should be treated as a model assumption with limited validation, especially where chloride data are sparse. (Source: abs_model.txt, Page: 10; Source: abs_model.txt, Page: 14; Source: abs_model.txt, Page: 15)

---

## 4. Prefer explicit uncertainty and sensitivity analyses

Optical properties may vary by orders of magnitude across wavelength. Their engineering impact may also depend strongly on temperature, optical thickness, sample geometry, impurities, and spectral averaging.

Where practical, report:

- fitted parameter covariance or bootstrap uncertainty;
- sensitivity to experimental dataset selection;
- sensitivity to wavelength range and spectral resolution;
- sensitivity to PDF peak location and peak-width treatment;
- sensitivity to damping, multiphonon prefactor, and anharmonicity;
- sensitivity to density and refractive-index models;
- uncertainty in \(\kappa_P(T)\);
- uncertainty in \(k_{\mathrm{rad}}\), radiative fraction, and corrected conduction conductivity;
- applicability flags for extrapolation outside calibration temperature, wavelength, composition, or optical-depth ranges.

Do not report excessive significant figures. If input data or model assumptions support only order-of-magnitude confidence, the output must communicate that.

---

## 5. Fail clearly rather than silently substituting physics

The code must not silently replace invalid or absent physical data with arbitrary defaults in production calculations.

Examples requiring an explicit warning or error include:

- missing PDF/RDF file;
- malformed PDF columns or inconsistent pair labels;
- nonpositive or nonfinite RDF values where a logarithm is evaluated;
- ambiguous first RDF peak;
- ambiguous first-shell minimum;
- invalid peak width or Gaussian fit;
- nonpositive reduced mass, force constant, \(\omega_0\), density, or temperature;
- missing refractive-index or absorption data;
- Planck integration range that does not overlap the wavelength grid;
- nonconverged fit;
- unphysical predicted parameters;
- use of a parameter default because no same-pair or same-family data exist;
- use of a correlation outside its calibrated \(r_0\) range;
- use of the Rosseland approximation when \(\tau \not\gg1\);
- use of an LFA correction without sample thickness.

Defaults may be used only when explicitly requested, clearly labeled, and written to metadata and output summaries.

---

# Structural input requirements

## PDF/RDF provenance

Every structural dataset used by the absorption model must record:

```text
composition
temperature
source/reference
simulation method or experiment
interatomic potential, if simulated
system size
equilibration and production durations
RDF bin width
partial-pair definition
normalization convention
units
preprocessing operations
```

A PDF-derived quantity must not be combined with optical data from a substantially different composition or temperature without an explicit statement of the approximation.

## Pair-label normalization

Pair names must be normalized consistently across all modules. For example:

```text
Li-F
Na-F
K-F
Na-Cl
U-Cl
```

Do not allow equivalent pair labels such as `F-Li`, `LiF`, or `Li_F` to propagate independently through model code, fitting routines, CSV export, and plots.

## PDF-derived equilibrium distance

The first cation–anion RDF peak location should be stored as:

```text
r0_angstrom
```

It is a local structural descriptor and should not be called a “bond length” without qualification in a molten salt. Preferred language:

```text
first-shell cation–anion separation
PDF-derived equilibrium separation
local pair-separation descriptor
```

---

# Dual-Gaussian / split-Gaussian bond-strength requirement

## Required methodological direction

The improved absorption-model workflow should support the split-Gaussian PDF-peak method implemented in the SCL analysis code as the preferred structural route for estimating a local bond-strength descriptor.

This method separately represents the steep inner side and broader outer side of the first RDF peak. It extracts:

- \(r_{\mathrm{peak}}\);
- \(g_{\mathrm{peak}}\);
- first-shell baseline \(g_{\mathrm{base}}\);
- left and right half-widths at half maximum;
- \(\sigma_L\) and \(\sigma_R\);
- PMF slopes at Gaussian inflection points; and
- an effective local PMF curvature / bond-strength factor.

The SCL workflow computes the PMF as:

\[
V_{\mathrm{PMF}}(r)=-k_B T\ln g(r)
\]

and estimates a bond-strength quantity from the change in analytical PMF slope across the left and right split-Gaussian inflection points. (Source: SCL_calc.txt, `calculate_pmf_and_bond_strength` and `_fit_gaussian_to_peak`)

This approach should be evaluated as a replacement or alternative to directly applying noisy numerical second derivatives to a raw PMF.

## Implementation requirements

When integrating this method into the absorption model:

1. Preserve the current raw-PMF-curvature method as an optional comparison path until validation is complete.
2. Implement the split-Gaussian method in a dedicated, tested function rather than duplicating SCL code.
3. Store all intermediate peak-fit quantities in machine-readable output.
4. Record whether bond strength came from:
   ```text
   raw_pmf_second_derivative
   split_gaussian_pmf_slope_difference
   gaussian_left_width_fallback
   ```
5. Flag fits with poor peak definition, missing half-maximum crossing, negative curvature, or implausible width.
6. Compare \(\omega_0\) from the old and new methods for salts where experimental far-IR absorption peaks are available.
7. Do not claim that the split-Gaussian approach is more accurate until comparison against independent spectral data demonstrates an improvement.

## Scientific caution

The split-Gaussian method is physically appealing because molten-salt RDF peaks are often asymmetric. However, it remains a structural approximation. It does not independently prove that a fitted local PMF curvature maps uniquely to the observed optical resonance frequency.

The agent should describe this as:

```text
a potentially more robust PDF-derived local stiffness estimator
```

rather than:

```text
the exact bond force constant
```

until validated.

---

# Optical absorption model requirements

## Vibrational term

The vibrational contribution should be calculated from a complex dielectric function that sums cation–anion oscillator contributions:

\[
\varepsilon(\omega,T)
=
\varepsilon_{\infty}
+
\sum_j
\frac{\Delta\varepsilon_j\omega_{0,j}^{2}}
{\omega_{0,j}^{2}-\omega^2-i\gamma_j(T)\omega}
\]

with temperature-dependent damping and density-scaled oscillator strength where configured. (Source: abs_model.txt, Page: 4)

The conversion from dielectric response to vibrational absorption and refractive index must remain unit-tested.

## Oscillator strengths and mixtures

For mixtures, oscillator-strength treatment must clearly document:

- pair mole fractions;
- dielectric constants used;
- mixture rule;
- density correction;
- whether pair cross-coupling is neglected;
- whether complex-ion or network structures are omitted.

The current independent-pair treatment is a modeling assumption, not an established fact for all molten salts. It may be especially limited for strongly associating or complex-forming melts such as BeF\(_2\)-, actinide-fluoride-, or multivalent-chloride-containing systems.

## Electronic absorption

Electronic/Urbach-tail terms should be explicitly labeled as:

```text
measured
fitted
nominal
transferred
assumed
```

A nominal electronic prefactor must not be presented as experimentally established for compositions where data are unavailable.

## Multiphonon absorption

The multiphonon term should retain clear separation of roles:

- \(C_0\): magnitude;
- \(\alpha_{\mathrm{anh}}\): spectral-tail slope / anharmonicity descriptor;
- \(\bar n(\omega_0,T)\): thermal occupation contribution;
- \(p(\omega)\): multiphonon order approximation.

The current implementation computes multiphonon absorption in log space and smooths discrete-\(p\) artifacts using PDF-derived frequency broadening. (Source: abs_model.txt, Page: 3)

The agent should preserve the distinction between:

```text
physical model parameter
numerical smoothing choice
visualization choice
```

A smoothing operation must not be silently allowed to alter fitted physical conclusions.

## Inhomogeneous broadening

PDF peak width may be used to estimate frequency spread through an approximation of the form:

\[
\frac{\sigma_\omega}{\omega_0}
\approx
\frac{\sigma_r}{r_0}
\]

The existing model extracts a left-side HWHM to reduce sensitivity to the asymmetric escaped-neighbor tail and caps fractional broadening at 5%. (Source: abs_model.txt, Page: 3; Source: abs_model.txt, Page: 6)

This is a reasonable operational approximation, but it must be documented as such. The agent should investigate sensitivity to:

- left-only width;
- right-only width;
- average width;
- split-Gaussian \(\sigma_L\) and \(\sigma_R\);
- cap value;
- spectral-grid resolution;
- smoothing-window choice.

---

# Fitting and validation requirements

## Fitting strategy

Fits should generally minimize a log-space residual because absorption spans many orders of magnitude:

\[
\mathcal{L}
=
\sum_d w_d
\sum_i
\left[
\log_{10}\kappa_{\mathrm{model},i}
-
\log_{10}\kappa_{\mathrm{exp},i}
\right]^2
\]

The existing code uses a Nelder–Mead optimizer with non-negativity handling. (Source: abs_model.txt, Page: 7)

The agent should consider, but not blindly introduce:

- bounded optimization;
- multi-start optimization;
- profile likelihoods;
- bootstrap refits;
- Bayesian inference;
- parameter regularization;
- separate training and validation datasets.

Any fitting-method change is a scientific-method change if it changes inferred parameters or conclusions.

## Parameter identifiability

The damping relation:

\[
\gamma(T)=\gamma_0+\gamma' T
\]

is not fully identifiable from data at one temperature. The code and documentation must acknowledge this degeneracy. The existing model recommends retaining a crystal-derived \(\gamma_0\) while fitting \(\gamma'\), \(\alpha\), and \(C_0\). (Source: abs_model.txt, Page: 7)

Do not fit both \(\gamma_0\) and \(\gamma'\) from a single-temperature dataset and then imply they are independently determined.

## Required fit outputs

For every fitted salt, save:

```text
composition
temperature(s)
experimental dataset identifiers
wavelength points used
fit weighting
fitted parameter values
initial parameter values
fixed parameters
optimizer and settings
objective-function value
fit status
model spectrum
residual spectrum
residual summary statistics
parameter uncertainty or sensitivity estimate
warnings and applicability notes
```

## Independent validation

Whenever possible, reserve data for validation by one or more of:

- wavelength-range holdout;
- temperature holdout;
- composition holdout;
- same-pair transfer test;
- anion-family transfer test;
- comparison to independent experimental source;
- comparison to AIMD/DFT optical-response calculations, when available.

A model should not be described as predictive for a salt if its parameters were fitted directly to the only available spectrum for that salt.

---

# Parameter transfer and generalized prediction

## Prediction hierarchy

When predicting a salt with no direct optical data, use and record a transparent hierarchy:

1. direct fitted parameter for that composition;
2. same ion-pair average from fitted compositions;
3. same anion-family transfer;
4. regression-based structural prediction;
5. explicitly marked global default.

The current code applies a correlation for \(\alpha/\omega_0\) versus \(r_0\), transfers \(C_0\), \(\gamma_0\), and \(\gamma'\) from same-pair or same-family information, and uses global defaults as a last resort. (Source: abs_model.txt, Page: 14; Source: abs_model.txt, Page: 15)

Every predicted parameter must include its source label, for example:

```text
alpha_anh_source = "fluoride_r0_regression"
C0_source = "same_pair_average_n=2"
gamma_slope_source = "chloride_family_average"
gamma0_source = "user_configured_default"
```

## Applicability-domain checks

Prediction code must warn when:

- \(r_0\) lies outside the regression training range;
- composition includes an ion pair not represented in the training data;
- an anion family has too few calibration points;
- predicted \(\alpha_{\mathrm{anh}}\), \(C_0\), or damping differs substantially from observed training distributions;
- target temperature exceeds the model’s calibrated temperature range;
- target wavelength range substantially extends beyond measured data;
- predicted \(\kappa_P\) is dominated by an extrapolated spectral region.

## Generalized low-fidelity model

A lower-fidelity generalized model may be appropriate if it is explicit about its inputs and limits. Possible inputs include:

```text
composition
temperature
anion family
cation charge
cation mass
reduced mass
PDF-derived r0
PDF-derived local stiffness
density
refractive-index estimate
melting temperature
structural-network descriptors
```

The agent may propose reduced-order models, but should prioritize:

- interpretability;
- physical dimensional consistency;
- out-of-sample validation;
- uncertainty estimates;
- clear distinction between empirical correlation and mechanistic model.

---

# Connection to molecular-dynamics simulations

## Role of MD information

Prior MD work can provide valuable structural and dynamical information to improve or test the absorption model. It should be treated as a source of model inputs and diagnostics, not as automatic validation.

Potentially useful MD-derived quantities include:

- partial RDFs and their temperature dependence;
- first-shell peak positions and asymmetry;
- first-shell coordination numbers;
- RDF peak widths;
- local cluster/speciation statistics;
- density and thermal expansion;
- dielectric-response estimates, if available;
- velocity autocorrelation functions;
- vibrational density of states;
- current spectra;
- local force distributions;
- structure-factor data;
- temperature-dependent local-environment distributions.

## High-priority MD-to-optics connections

The agent should consider whether the following can improve the optical model:

1. **Temperature-dependent PDFs**  
   Directly calculate \(r_0(T)\), peak widths, asymmetry, and local stiffness descriptors from MD at multiple temperatures rather than assuming only damping changes with temperature.

2. **Split-Gaussian peak analysis**  
   Use the SCL workflow’s left/right PDF-peak widths and PMF-slope approach to obtain a more robust local stiffness metric.

3. **Speciation and complex-ion structure**  
   Identify whether nominal independent cation–anion pairs are inadequate because local complexes dominate, such as tetrahedral BeF\(_4^{2-}\), actinide-halide coordination environments, or multivalent-chloride complexes.

4. **VDOS comparison**  
   Compare PDF-derived \(\omega_0\) with peaks or centroids in MD-derived velocity-density-of-states or current-spectrum diagnostics. This is a consistency test, not proof of a one-to-one optical-mode assignment.

5. **Dielectric response**  
   Where feasible, compare the model dielectric function to dielectric properties inferred from polarizable simulations, ab initio MD, or independent optical/refractive-index data.

6. **Density and thermal expansion**  
   Use validated MD density trends where experimental density data are unavailable, while recording that oscillator-strength density scaling is still an approximation.

7. **Impurity and composition sensitivity**  
   Use MD to test whether small compositional changes qualitatively alter local coordination/speciation in ways that invalidate simple endmember or pairwise interpolation.

## Interpretation limits

Do not state that SCL, VDOS overlap, dispersion overlap, or local mode matching directly determines optical absorption unless an explicit model and independent validation support that conclusion.

Preferred wording:

```text
The MD-derived descriptor is evaluated as a candidate predictor or consistency diagnostic for the optical model.
```

---

# Planck-mean property requirements

## Spectral resolution and coverage

Planck means must be calculated over a wavelength grid that adequately covers the thermal-emission range for each temperature. The current implementation integrates around the Wien peak using a configurable range factor. (Source: abs_model.txt, Page: 5)

Outputs must record:

```text
temperature_K
wavelength_min_um
wavelength_max_um
number_of_wavelength_points
integration_rule
Wien_range_factor
spectral_extrapolation_flag
kappa_planck_m1
n_planck
uncertainty_or_sensitivity_band
```

## Spectral versus scalar radiative models

The scalar Planck-mean Deissler approximation is useful for engineering studies, but the nonlinear denominator means that averaging absorption before applying the radiative-conductivity formula is not generally equivalent to spectrally resolving the calculation.

The repository already includes a spectral Deissler-style calculation that weights wavelength-specific radiative conductivity by the Planck spectrum. (Source: LFA_case_study.txt, Page: 4)

Where spectral data are available, the agent should support comparison between:

```text
Planck-mean scalar approximation
spectrally resolved bridging calculation
Rosseland approximation
```

and clearly state the assumptions and differences.

---

# Radiative heat-transfer and measurement-correction requirements

## LFA and variable-gap interpretation

For semi-transparent salts:

\[
k_{\mathrm{measured}}
=
k_{\mathrm{conduction}}
+
k_{\mathrm{rad}}
\]

An experiment that does not separate the radiative contribution may overestimate conduction conductivity. (Source: LFA_case_study.txt, Page: 1)

The agent should support careful analysis of whether observed thickness dependence, apparent positive temperature coefficients, or disagreement among techniques are **consistent with** radiative transport. It should not claim that radiation is the demonstrated explanation unless supported by controlled experimental evidence.

## Optical thickness

Always calculate:

\[
\tau = \kappa_P L
\]

before selecting or interpreting a radiative-transfer approximation.

- \(\tau \gg 1\): Rosseland diffusion may be appropriate.
- \(\tau \ll 1\): optically thin gap behavior is important.
- intermediate \(\tau\): use a bridging or spectral transport method.

The LFA case-study code explicitly uses a Deissler bridging expression to connect these limits. (Source: LFA_case_study.txt, Page: 4; Source: LFA_case_study.txt, Page: 7)

## Required outputs for a measurement-correction calculation

```text
composition
temperature_K
sample_thickness_m
kappa_planck_m1
n_planck
optical_thickness
conduction_model_and_source
k_rad_W_mK
k_total_W_mK
radiative_fraction_percent
radiative_model_used
spectral_or_scalar_flag
uncertainty_or_sensitivity_summary
```

## Conduction model caution

A conduction baseline used to demonstrate radiative effects is not automatically a validated conductivity model. The current case-study code uses configured linear conduction relations despite comments referring to a power-law model. Any such inconsistency must be resolved and documented before publication-quality conclusions are drawn. (Source: LFA_case_study.txt, Page: 4; Source: LFA_case_study.txt, Page: 5)

---

# Repository conventions

## Recommended project layout

```text
Molten_Salt_Absorption_Model/
├── AGENTS.md
├── README.md
├── config/
│   ├── salts_config.py
│   ├── datasets.yaml
│   └── model_defaults.yaml
├── data/
│   ├── experimental_absorption/
│   ├── experimental_refractive_index/
│   ├── pdf_rdf/
│   ├── thermophysical/
│   └── processed/
├── src/
│   ├── absorption_model.py
│   ├── pdf_features.py
│   ├── split_gaussian_pdf.py
│   ├── dielectric_model.py
│   ├── multiphonon_model.py
│   ├── planck_means.py
│   ├── radiative_transport.py
│   ├── lfa_corrections.py
│   ├── parameter_transfer.py
│   ├── validation.py
│   └── io_utils.py
├── scripts/
│   ├── run_single_salt.py
│   ├── run_batch_fit.py
│   ├── run_prediction.py
│   ├── run_lfa_case_study.py
│   └── compare_pdf_methods.py
├── tests/
├── outputs/
│   └── <composition>/<model_run_label>/
└── docs/
```

Do not write generated spectra, fit products, figures, CSV databases, or temporary files into `src/`, `config/`, or `scripts/`.

## Canonical identifiers

Use a normalized composition identifier and temperature label consistently:

```text
composition: 0.465LiF-0.115NaF-0.420KF
temperature: 973.0 K
run_label: 0.465LiF-0.115NaF-0.420KF_973.0K
```

Do not independently reconstruct composition labels in separate modules.

---

# Required outputs and provenance

Each model run should create a run manifest, for example:

```json
{
  "composition": "0.465LiF-0.115NaF-0.420KF",
  "temperature_range_K": [737.0, 1300.0],
  "run_label": "0.465LiF-0.115NaF-0.420KF_737-1300K",
  "model_version": "git_commit_or_tag",
  "pdf_source": "Frandsen_2020",
  "pdf_temperature_K": 873.0,
  "bond_strength_method": "split_gaussian_pmf_slope_difference",
  "optical_data_sources": ["Chaleff_2016"],
  "fit_configuration": {
    "fit_target": ["gamma_slope", "alpha_anh", "C0_multi"]
  },
  "parameter_status": "fitted",
  "generated_files": [],
  "warnings": []
}
```

Required output products should include:

```text
manifest.json
spectral_absorption_<label>.csv
refractive_index_<label>.csv
planck_mean_properties_<label>.csv
fit_summary_<label>.csv
parameter_provenance_<label>.csv
pdf_feature_summary_<label>.csv
validation_summary_<label>.csv
```

For LFA or radiative-conductivity studies:

```text
radiative_conductivity_<label>.csv
lfa_thickness_sensitivity_<label>.csv
```

---

# Plotting requirements

All plots should be publication-ready by default.

Use:

- Times New Roman or an approved project-wide serif alternative;
- clear units;
- distinguishable colors and marker shapes;
- experimental points visually distinct from model lines;
- no misleading logarithmic axes without clear labeling;
- legends that do not obscure data;
- exported PDF and high-resolution PNG versions.

Plots should clearly distinguish:

```text
experimental data
fitted model
held-out validation prediction
fully transferred prediction
default-parameter estimate
```

For spectral plots, include the relevant Planck radiance curves or Wien-region indication when the purpose is radiative heat transfer.

For structural plots, show:

- raw RDF;
- fitted split-Gaussian peak representation;
- \(r_0\);
- left/right widths;
- first-shell baseline/minimum;
- the bond-strength method used.

---

# Testing requirements

Before large fitting campaigns or publication calculations, run lightweight tests for:

- composition normalization;
- pair-label normalization;
- unit conversions;
- PDF loading and column validation;
- raw PMF curvature calculation;
- split-Gaussian peak fitting;
- left/right HWHM extraction;
- reduced-mass calculation;
- \(\omega_0\) calculation;
- dielectric-to-absorption conversion;
- refractive-index calculation;
- Planck-function numerical stability;
- Planck-mean integration;
- Deissler and Rosseland limiting behavior;
- \(\tau \ll 1\) and \(\tau \gg 1\) limiting behavior;
- missing-data failure modes;
- parameter-provenance export;
- correlation-domain warning behavior;
- reproducible fit configuration.

Tests must include physically meaningful limiting checks, such as:

```text
kappa_P > 0
n > 0
k_rad > 0 for T > 0 and finite optical properties
k_rad approaches Rosseland behavior at large tau
k_rad scales with L in the optically thin bridging limit
```

---

# Communication requirements for the agent

When proposing a change, the agent must state:

1. what the present code does;
2. the concern or opportunity;
3. whether it is:
   - a confirmed implementation issue,
   - a likely robustness issue,
   - a methodological limitation,
   - a proposed scientific extension,
   - a hypothesis requiring validation;
4. the proposed change;
5. expected benefits;
6. risks and tradeoffs;
7. validation needed before scientific conclusions are updated.

The agent must explicitly label conclusions as:

```text
confirmed from code
supported by existing input data
inferred from model assumptions
plausible but unverified
requires independent validation
```

---

# Definition of done for a model result

A molten-salt absorption-model result is complete only when:

1. composition and temperature range are unambiguous;
2. structural input provenance is recorded;
3. PDF feature extraction is saved;
4. the bond-strength method is identified;
5. all fitted and transferred parameters have provenance labels;
6. spectral absorption and refractive-index outputs exist;
7. Planck-mean properties are exported with integration metadata;
8. wavelength and temperature extrapolations are flagged;
9. fit and/or validation diagnostics are saved;
10. radiative-transfer calculations state the optical-thickness regime and geometry;
11. warnings are preserved rather than hidden;
12. results are described at a confidence level justified by the available structural and optical evidence.

This project’s central contribution should be framed carefully: it provides a structured route for connecting molten-salt molecular structure, optical absorption, and radiative heat-transfer implications. It does not eliminate the need for optical measurements, independent validation, sensitivity analysis, or explicit acknowledgment of uncertainty.