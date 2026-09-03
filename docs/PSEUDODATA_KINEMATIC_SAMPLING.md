# Pseudodata kinematic sampling strategy

## Status and scope

This document defines the proposed large-corpus kinematic design implemented
as a separate cell in
[`scripts/visualize_gpddatabase_dvcs.ipynb`](../scripts/visualize_gpddatabase_dvcs.ipynb).
It is a design and validation artifact, not yet the production corpus
generator. The existing empirical-resampling cell remains unchanged so the two
schemes can be compared directly.

This strategy selects observable kinematics. It does not change the declared
80-dimensional GPD parameter prior, parameter-group split, measurement-noise
model, or PARTONS physics implementation. Experimental measurement values and
uncertainties are quarantined throughout.

## Design objective

A useful master design must satisfy five goals simultaneously:

1. remain inside the exact fixed-target physical domain;
2. represent the support of existing experiments without reproducing their
   highly uneven point density;
3. cover physically valid gaps between experimental anchors;
4. retain an explicit, controlled edge/stress population;
5. be deterministic, unique, auditable, and nestable for later scaling.

Independent uniform draws from a rectangular box do not satisfy the first
three goals. Bootstrap resampling of experimental anchors preserves their
joint correlations, but produces duplicate kinematics and overweights dense
experiments. The proposed design therefore combines unique maximin anchor
selection with conditional scrambled-Sobol gap filling.

## Declared envelope

The notebook cell uses:

| Variable | Range or policy |
|---|---|
| `x_b` | `[0.10, 0.50]` |
| `Q2_GeV2` | `[1.10, 4.00]` GeV2 |
| `minus_t_GeV2` | `[0.10, 1.50]` GeV2 |
| `beam_energy_GeV` | Discrete recorded energies with complete physical anchors |
| `phi_rad` | Eight balanced mid-bin angles for each selected four-tuple |

The Bjorken-limit mapping used for display is

```text
xi = x_b / (2 - x_b),
```

so this envelope corresponds to approximately `xi in [0.0526, 0.3333]`.
`xi` is derived; it is not sampled independently from `x_b`.

## Necessary physical constraints

For every candidate `(x_b, Q2, minus_t, beam_energy)`, the cell computes

```text
y = Q2 / (2 M_p E_beam x_b)
W2 = M_p^2 + Q2 (1 - x_b) / x_b
epsilon2 = 4 M_p^2 x_b^2 / Q2
```

and the positive exact finite-`Q2` boundary

```text
minus_t_min = Q2 *
    [2 (1 - x_b) (1 - sqrt(1 + epsilon2)) + epsilon2]
    / [4 x_b (1 - x_b) + epsilon2].
```

A point is called physical only when

```text
0 < y < 1
minus_t >= minus_t_min + 0.02 GeV2.
```

The `0.02 GeV2` offset is a numerical/design margin, not a new physical law.
It avoids placing training points exactly on the kinematic boundary. Its effect
must be checked against smaller/larger margins before a scientific production
claim.

The beam energy cannot be assigned independently after sampling `x_b` and
`Q2`. For example, at `E_beam = 3.355 GeV`, `Q2 = 4 GeV2` requires
`x_b > 0.635` from `y < 1`, outside the declared envelope. Conditioning on
energy is therefore mandatory.

## Core and edge/stress labels

Physical points are divided into two explicit strata. A core point satisfies

```text
0.02 <= y <= 0.98
W >= 2 GeV
minus_t / Q2 <= 0.5.
```

Every other physical point is labeled `edge/stress`. Thus low-`W`, relatively
large-transfer, and near-`y`-boundary points remain available for robustness
testing without silently dominating the main design. `Edge/stress` does not
mean mathematically unphysical; it identifies a less conservative analysis
region.

The notebook asserts that every selected core and edge point has the expected
classification. Core and edge metrics must be reported separately in later
validation.

## Default 1,000-group allocation

| Stratum | Count | Selection |
|---|---:|---|
| Experimental-anchor core | 700 | Unique, without replacement, deterministic maximin |
| Conditional-Sobol core | 150 | Continuous physical gap filling, then deterministic maximin |
| Experimental-anchor edge/stress | 150 | Unique, without replacement, deterministic maximin |

This is a starting design, not a claim that `700/150/150` is universally
optimal. It gives 85% core coverage and 15% explicit stress coverage while
keeping most points tied to experimentally occupied kinematics. Change the
allocation only through a new recorded design identity and compare coverage
and native stability against this baseline.

The earlier empirical bootstrap drew 1,000 times with replacement. In the
broader notebook envelope it had 3,029 available unique anchors; under equal
probabilities, only about 850 distinct anchors are expected. The proposed
anchor strata never spend native evaluations on duplicate four-tuples.

## Measurement quarantine and candidate pool

The candidate pool is constructed from unique

```text
(x_b, Q2_GeV2, minus_t_GeV2, beam_energy_GeV)
```

tuples found in experimental GPDdatabase rows. Lattice/pseudodata sources are
excluded. Observable values, statistical/systematic uncertainties, covariance,
and normalization uncertainties are not read into selection features or
scores. The database supplies only measurement-free kinematics and metadata.

With the pinned database and current settings, a read-only precheck found
2,897 unique physical anchors: 2,554 core and 343 edge/stress. This is enough
capacity for both anchor allocations without replacement.

## Distance coordinates

Maximin selection uses four coordinates:

```text
x_b
log(Q2_GeV2)
u_t
log(beam_energy_GeV)
```

where

```text
L = max(0.10, minus_t_min + 0.02)
u_t = (minus_t - L) / (1.50 - L).
```

Each coordinate is normalized with the clipped 1st-to-99th-percentile span of
the complete physical anchor pool. Clipping prevents one extreme point from
defining nearly all distances.

The logarithms are distance coordinates, not declarations that the target
data distribution is log-uniform. `log(Q2)` compares relative scale changes
and `log(E)` prevents the largest numerical beam energies from dominating
Euclidean distance. Recorded anchor values are retained exactly. Only the
continuous Sobol component interpolates uniformly in conditional `log(Q2)`.

`u_t` is preferable to either raw or logarithmic `minus_t`: the physically
allowed lower boundary varies with `(x_b,Q2)`, so the same `u_t` represents the
same relative position within each conditional transfer interval.

## Tempered beam-energy quotas

Beam energy is a discrete stratum; the strategy does not invent continuously
interpolated beam settings. If `N_E` is the unique candidate count at energy
`E`, the target quota is proportional to

```text
w_E = sqrt(N_E).
```

At least one point is assigned to each supported energy when the total budget
allows it. Integer quotas are filled deterministically and capped by available
unique anchors for anchor strata.

Pure empirical weighting (`w_E = N_E`) would allow the densest experiments to
dominate. Equal weighting would overrepresent energies supported by only one
or two anchors. Square-root weighting is a transparent compromise between
target realism and coverage; energy-resolved diagnostics must still be
inspected.

## Deterministic maximin selection

For each energy stratum:

1. sort and deduplicate candidate four-tuples;
2. choose one reproducible seed point;
3. compute every candidate's distance to that point;
4. repeatedly choose the candidate with the largest distance to its nearest
   already-selected point;
5. resolve numerical ties by stable sorted order.

This farthest-point rule spreads selections across the available joint domain
instead of reproducing local experimental density. The fixed seed
`20260901`, quota policy, sorted input, and explicit tie rule make reruns
identical for the same database and code.

## Conditional scrambled-Sobol core fill

The Sobol component uses three unit coordinates `(u_x,u_Q,u_t)` separately at
each recorded energy. It constructs core points in physical order:

```text
x_b = x_low + u_x (x_high - x_low)

Q2_low = max(
    1.10,
    (W_core^2 - M_p^2) x_b / (1 - x_b)
)

Q2_high = min(
    4.00,
    2 M_p E_beam x_b (1 - y_margin)
)

log(Q2) = log(Q2_low)
          + u_Q [log(Q2_high) - log(Q2_low)]

minus_t_low = max(0.10, minus_t_min + 0.02)
minus_t_high = min(1.50, 0.5 Q2)
minus_t = minus_t_low + u_t (minus_t_high - minus_t_low).
```

Candidates with an empty conditional interval are skipped. A larger scrambled
Sobol candidate pool is generated, and the same deterministic maximin rule
selects the requested quota. This avoids the severe rejection distortion that
would result from drawing an independent rectangular sample and discarding
most forbidden combinations.

The Sobol component fills physically valid holes; it is not evidence that an
experiment has measured those exact coordinates.

## Azimuthal design

Every selected four-tuple is expanded to eight balanced mid-bin angles:

```text
phi_k = 2 pi (k + 1/2) / 8,  k = 0,...,7.
```

The cell therefore exports 1,000 unique four-tuples and 8,000 five-dimensional
sites. Mid-bin placement avoids duplicating `0` and `2 pi`. If a campaign is
intended to reproduce a particular experiment, its audited angular bins may
replace this grid; the replacement becomes part of corpus identity.

Never split individual `phi` bins from one four-tuple across training and a
kinematic holdout. The complete angular group must remain together.

## Notebook outputs

The cell defines:

```text
STRATIFIED_SAMPLING_SETTINGS
STRATIFIED_PARTONS_KINEMATIC_POINTS   # [1000, 4]
STRATIFIED_PARTONS_LABELS             # [1000]
STRATIFIED_PARTONS_KINEMATIC_SITES    # [8000, 5]
STRATIFIED_PARTONS_SITE_COLUMNS
```

It prints pool sizes, selected stratum counts, unique/site totals, and
energy-resolved counts. Its four diagnostic panels show joint `(x_b,Q2)`,
`(Q2,minus_t)`, `(minus_t,xi)`, and stacked energy coverage. Assertions fail if
the requested count, uniqueness, or core/edge classification is violated.

## Training and validation use

Kinematic design and GPD-parameter sampling are distinct:

- keep parameter groups sampled from the declared prior unless a separately
  validated prior-sampling change is approved;
- keep the existing grouped parameter train/internal-validation/outer-test
  split;
- create a separate output-blind fresh-core kinematic holdout;
- create a separate edge-heavy stress holdout;
- freeze holdout coordinates before inspecting their PARTONS outputs;
- report performance by anchor/Sobol source, core/edge label, energy, and
  kinematic region, not only as one aggregate.

The proposed 15% edge fraction is part of the design being evaluated. It does
not replace an independent edge-heavy stress test.

## Computational scaling

The notebook cell performs only selection and plotting. It does not launch
PARTONS. Expanding 1,000 groups to eight angles creates 8,000 kinematic sites
per GPD parameter group. For 4,096 parameter groups, that is 32,768,000
parameter-site combinations before counting multiple observables or rejected
native draws.

Before production, run `corpus-preflight` and a small timed shard using the
intended observables. Record native success rate, wall time per parameter-site,
memory, storage, and output size. Scale only from those measured quantities;
do not infer feasibility from the inexpensive notebook plot.

## Production integration

`visualize_gpddatabase_dvcs.ipynb` is the authoritative user interface for the
sampling envelope and selected coordinates. Exported points are not clipped to
the former `t_GeV2 >= -0.90` corpus envelope. Python and native validation both
enforce `0<x_B<1`, `Q2>=Q0^2`, positive beam energy, `0<y<1`, azimuth range,
and exact finite-`Q2` forward/backward transfer limits.

Before a large run, use `corpus-preflight` and a timed native shard. This tests
PARTONS numerical stability and measures runtime/storage; it does not replace
the notebook's kinematic choice.

## Reproducing the design check

Run the notebook from the repository root in the pinned Python environment
with `DVCS_GPDDATABASE_ROOT` pointing to the installed clean database checkout.
Execute the database-loading cells, the existing envelope cell, and then the
new recommended-design cell. A valid run must report:

```text
Selected four-tuple strata: 700 anchor core, 150 Sobol core,
                             150 anchor edge/stress
Unique groups: 1,000
phi-expanded sites: 8,000
```

For repository-level checks after editing the notebook or this document:

```bash
python3 -m json.tool scripts/visualize_gpddatabase_dvcs.ipynb >/dev/null
PYTHONDONTWRITEBYTECODE=1 python3 tests/verify_documentation.py
git diff --check
```

These commands verify notebook JSON and documentation integrity. They do not
execute PARTONS or establish production acceptance.
