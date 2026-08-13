# Workflow and physics reference

This document is the complete user-level specification for public schema 7.
The release is a multi-Q2 synthetic posterior laboratory. It is ready for the
user's DD closure and fresh native-model campaigns; real-data inference is
still disabled.

## 1. End-to-end workflow

```text
80 independent DD controls at Q0² = 1 GeV²
  (H,E,Htilde,Etilde × u,d,s,g × N,a,c,b,B)
                         │
                         ▼
project C++ input GPD + fixed shadow settings
                         │
                         ▼
PARTONS GPDEvolutionApfel / APFEL++ to each datum Q²
                         │
                         ▼
PARTONS DVCSCFFStandard → 4 complex CFFs
                         │
                         ▼
DVCSProcessGV08 → 6 native observables at each site
                         │
                         ▼
declared full covariance + 2 nuisance shifts + random noise
                         │
                         ▼
synthetic dataset D = unordered set of 36 point tokens
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
DeepSets encoder + NPE flow    exact likelihood on DD bank
neural qφ(θ,η|D)              “conventional” comparison
             └───────────┬───────────┘
                         ▼
coverage, PPC, posterior/CFF/GPD exact-native reevaluation
                         │
                         ▼  only after synthetic gates pass
fresh output-blind GK/VGG external validation
```

Python supplies priors, covariance/noise, contexts, NPE, metrics, and plots.
It contains no GPD, CFF, or DVCS observable formula. No gradient through
PARTONS is required.

The standalone `plot` action occurs only after the saved evaluation and
conventional-comparison records exist. It hashes and renders those records;
it does not revisit any box in the physics/inference chain or change a gate.

During evaluation, empirical coverage is calculated from neural posterior
draws first. The framework then selects the configured number of posterior
physics vectors and exactly reevaluates them through PARTONS at every datum
kinematic. A separate exact pass evaluates injected truth plus every retained
posterior vector on the GPD diagnostic x-grid. These two native passes use
ordered, process-isolated two-sample batches across all affinity-permitted
workers; completion order never changes saved posterior order.

## 2. Kinematics and evolution

The observable arguments are Bjorken \(x_B\),
\(t=(p'-p)^2\le0\) in GeV², photon virtuality \(Q^2=-q^2>0\) in GeV²,
beam energy \(E\) in GeV, and azimuth \(\phi\) in radians.

For the fixed proton target used by the native observable modules, PARTONS
defines

$$
y=\frac{Q^2}{2M_p E x_B},\qquad M_p=0.938272013\ \mathrm{GeV}.
$$

Every synthetic and holdout point must satisfy $0<y<1$. Individual JSON
ranges alone do not guarantee this coupled condition. Database-backed project
creation filters on it, the public loader rechecks it, and the C++ bridge
rejects it before evolution. This gate prevents a physically impossible point
from producing non-finite cross sections after expensive native work.

The GPD is parameterized once at

$$Q_0^2=1\ \mathrm{GeV}^2.$$

For datum \(i\), PARTONS/APFEL++ computes

$$F_i(x,\xi,t,Q_i^2)=U(Q_i^2,Q_0^2)\otimes F(x,\xi,t,Q_0^2).$$

The admitted range is \(1\le Q_i^2\le80\,\mathrm{GeV}^2\). At
\(Q_i^2=Q_0^2\), this is the tested identity route; at larger scales it is
native forward evolution. Backward evolution is rejected. Experimental
measurements are never evolved or Q2-projected: data are never evolved.

The configured evolution is LO, MSbar, fixed \(n_f=3\), with
`RunningAlphaStrongApfel`, \(\alpha_s(M_Z)=0.118\), and zero light-flavor
threshold entries. H/E use the installed unpolarized GPD kernels;
Htilde/Etilde use polarized kernels. LO qg and gq operators mean a gluon input
can affect evolved quarks and therefore the LO CFF indirectly. Heavy-quark
input and threshold matching are not included.

## 3. Full independent DD parameterization

For each field \(F\in\{H,E,\widetilde H,\widetilde E\}\) and channel
\(p\in\{u,d,s,g\}\),

$$
F^p(x,\xi,t)=\int_0^1d\beta
\int_{-1+\beta}^{1-\beta}d\alpha\,
\delta(x-\beta-\xi\alpha)f_{F,p}(\beta,t)\pi_{b_{F,p}}(\beta,\alpha),
$$

$$
f_{F,p}(\beta,t)=N_{F,p}
\frac{\beta^{a_{F,p}}(1-\beta)^{c_{F,p}}}
{B(a_{F,p}+1,c_{F,p}+1)}e^{B_{F,p}t},
$$

$$
\pi_b(\beta,\alpha)=
\frac{\Gamma(b+\tfrac32)}{\sqrt\pi\,\Gamma(b+1)}
\frac{[(1-\beta)^2-\alpha^2]^b}{(1-\beta)^{2b+1}}.
$$

Every one of the 16 blocks has five inferred controls:

| JSON | Meaning | Prior support |
|---|---|---|
| `normalization` | signed forward zeroth moment \(N\) | \((-4,4)\) |
| `a` | small-\(\beta\) power | \((0.1,1)\) |
| `c` | large-\(\beta\) power | \((2,6)\) |
| `profile_b` | skewness profile power | \((1,4)\) |
| `t_slope_GeV_minus2` | exponential slope \(B\) | \((0,2)\) GeV⁻² |

Wire order is field, channel, then the table order. The 80 physics controls
are followed by two standard-normal nuisance coordinates. Uniform physical
priors use an analytic probit transform; finite flow coordinates never
represent endpoints.

Calling the shapes “independent” means each block has its own controls. It
does not mean model-independent physics: this is one DD representation family
with declared priors. Native VGG/GK models are not used in training.

## 4. Physics constraints

Enforced and tested constraints include finite bounded controls, integrable
endpoint powers, DD support \(|x|\le1\), normalized DD profile, continuity,
the representation's polynomiality construction, finite native outputs,
physical observable domains, one fixed input scale, upward-only evolution,
the identity limit, and explicit singlet/gluon mixing. Native regression tests
cover support, symmetry, endpoints, forward/moment behavior, evolution, and
shadow composition at their declared scopes.

The framework does not claim general positivity, physical proton form-factor
sum rules for arbitrary independently signed normalizations, a D-term,
Etilde pion pole, heavy flavors, evolved shadow cancellation, uniqueness, or
a verified twist label. Those are omissions, not implicit constraints.

## 5. Fixed shadow settings

The simulator adds one type-level coefficient times channel amplitudes. H/u
uses installed PARTONS `GPDBDMMS21`; the remaining directions are project DD
differences with zero forward zeroth moment. They are stress directions, not
native shadows and not proved CFF-null. In this release all shadow settings
are fixed within a posterior run and are not among the 82 inferred
coordinates. Editing them changes the simulator family; use a new project.

## 6. CFFs and observables

PARTONS returns

$$\mathcal H,\quad\mathcal E,\quad
\widetilde{\mathcal H},\quad\widetilde{\mathcal E},\qquad
\mathcal F=\Re\mathcal F+i\Im\mathcal F.$$

The coefficient function is configured LO with
\(\mu_F^2=\mu_R^2=Q^2\). The six directly used installed observables are:

1. `DVCSCrossSectionUUMinus` — unpolarized/beam-spin half-sum cross section;
2. `DVCSCrossSectionDifferenceLUMinus` — beam-spin difference;
3. `DVCSAc` — beam-charge asymmetry;
4. `DVCSAluMinus` — beam-spin asymmetry;
5. `DVCSAulMinus` — longitudinal target-spin asymmetry;
6. `DVCSAllMinus` — longitudinal double-spin asymmetry.

Cross sections use the native nb/GeV⁴ route; asymmetries are dimensionless.
No unverified Fourier-moment method is invented. Despite LO evolution and LO
coefficient modules, the aggregate order and twist labels remain null until a
full published-route benchmark establishes the complete combination.

## 7. Synthetic covariance and nuisance parameters

For normalized observable vector \(\mu(\theta)\), the simulator creates a
fixed positive-definite covariance from diagonal relative/floor uncertainty,
a local correlation kernel, and a correlated phi-shape source. The observed
dataset is

$$y=\mu(\theta_*)+Lz,\qquad z\sim\mathcal N(0,I),\quad LL^T=C.$$

Two explicit standard-normal nuisances act multiplicatively:

$$
\mu_i(\theta,\eta)=\mu_i(\theta)
[1+r_g\eta_g+r_{LU,i}\eta_{LU}],
$$

where the LU response is nonzero only for the beam-spin difference. The
likelihood uses full \(C^{-1}\), not diagonal chi-square. These uncertainties
are synthetic design choices, not claims about the experimental covariance.

## 8. NPE and data splits

Each token contains kinematics, one-hot observable metadata, normalized value,
marginal uncertainty, a symmetric-whitened full-covariance encoding, and
nuisance responses. A shared point network maps tokens to embeddings;
DeepSets sums them, giving permutation invariance; global contract features
are appended. A zuko MAF estimates

$$q_\phi(\theta,\eta\mid D,m)$$

by minimizing

$$\mathcal L_{\mathrm{NPE}}=-\mathbb E\log q_\phi(\theta,\eta\mid D,m).$$

Observable MSE or chi-square is not the neural posterior loss. Exact PARTONS
observables are used only to generate training pairs and reevaluate samples.

The release default is a three-layer 128-wide point network, a two-layer
128-wide dataset network, 96 embedding features, and a zuko MAF with 96 hidden
features, six transforms, and eight bins. Training uses batch size 256,
learning rate (5\times10^{-4}), and a 10% internal-validation fraction.
These values are frozen for the release campaign, not asserted optimal.

Native parameter draws—not noisy replicas—are partitioned into train,
internal validation, and a 20% outer test. Replicas from one draw cannot cross
roles. Internal validation controls early stopping and optional Optuna. The
outer test is not used for selection.

## 9. Conventional versus neural posterior

The neural posterior is the learned conditional flow. The conventional
comparison evaluates the same full-covariance likelihood and prior on the
retained exact DD bank and self-normalizes its importance weights. It is a
useful reference only if its effective sample size passes; 82 dimensions can
make it degenerate. A failed ESS blocks comparison—it does not make the NPE
correct by default. Production conventional closure may require MCMC or
nested sampling beyond this importance reference.

## 10. Validation sequence

For a new configuration, require in order:

1. native validity/invalid-map and deterministic cache reproduction;
2. disjoint grouped split and held-out density checks;
3. coverage/SBC across many injected truths;
4. prior and posterior predictive checks with full covariance;
5. neural versus conventional comparison with adequate ESS;
6. exact-backend reevaluation of posterior samples;
7. fresh output-blind GK/VGG external validation;
8. representation and shadow tests when their campaigns are defined.

The external manifest was selected from catalog kinematics with no
measurements, uncertainties, or named-model outputs. It contains six fresh
sites; all six observables make the same 36-token shape used in training.
Named outputs are post-training only and may not drive Optuna or architecture
changes. A mismatch reports lack of robustness.

## 11. Real-data mapping readiness

The protected catalog is read-only and revision-pinned. New projects use
measurement-free catalog coordinates and retain Q2/beam values. The mapping
inventory marks direct-name UU, LU-difference, Ac, ALU, AUL, and ALL routes as
candidates pending dataset audits. Harmonics, transverse components,
virtual-photon cross sections, and t slopes remain unmapped. Before fitting,
the project still requires definition/sign/unit benchmarks, systematic and
normalization nuisance assignment, and covariance/missing-correlation policy.
No real likelihood or posterior update is enabled.

## 12. Reproducibility and devices

Every JSON is indented and finite. Native requests/responses, bridge/source
hashes, stderr, exit codes, cache keys, split arrays, seeds, checkpoints, and
metrics are retained. Native failures are isolated without imputation or
Python fallback. CUDA is used for neural work when selected and available;
PARTONS remains CPU-only. Process workers never share one PARTONS instance.
