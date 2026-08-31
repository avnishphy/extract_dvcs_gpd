# Development configuration and testing

This page is for contributors and operators who need to verify the checked-in
implementation. It does not add a second user workflow: the supported
scientific interface remains `./dvcs` and the editable project
`experiment.json` described in [the user guide](USER_GUIDE.md).

## Source and configuration authority

The public CLI creates its private engine files from exactly these checked-in
inputs:

```text
configs/inference/stage10_pseudodata_workflow.json
configs/physics/stage10_pseudodata_systematics_v1.json
```

They are implementation templates, not user configuration files. `init`
copies their supported controls into an isolated schema-8 `experiment.json`;
the CLI validates that document and regenerates `.engine/workflow.json` and
`.engine/physics.json` before each action. Edit the project document, not the
Stage 10 templates or generated `.engine/` files. A template change is a code
and scientific-method change: review its effect on corpus identity, result
contracts, validation gates, documentation, and regression evidence.

`configs/schemas/` defines the versioned JSON protocol for direct native bridge
requests and responses and the synthetic-dataset contract. The files in
`configs/examples/` are example bridge payloads for supported native operation
families. They are useful when developing or diagnosing the bridge, but they
are not a replacement for corpus creation, materialization, training, or
validation through `./dvcs`.

Other `configs/inference/`, `configs/physics/`, and `configs/validation/`
files preserve explicitly named campaign, representation, evolution, shadow,
and holdout contracts used by the native bridge and retained validation
material. Their presence does not make every PARTONS module, representation,
or stage file a selectable production workflow. The currently supported public
production path is the full-independent DD pseudodata workflow documented in
[Workflow and physics](WORKFLOW_AND_PHYSICS.md).

## Implementation map

`src/extract_dvcs_cff/cli/user.py` owns containment of project, corpus, and
archive paths plus public-command validation. `corpus.py` owns immutable native
shards, selections, portable archives, verified checkpoint batches, merging,
and deterministic materialization. `workflows/pseudodata.py` coordinates
training, evaluation, comparison, holdout, real-data diagnostic, and saved
plots. The `data/`, `simulation/`, `inference/`, `likelihood/`, and
`optimization/` packages provide the corresponding focused components.

`cpp/partons_bridge/` is the authoritative C++/PARTONS boundary. Its C++ tests
cover retained native physics routes. Python never implements a fallback GPD,
CFF, or DVCS-observable calculation. Container recipes, launchers, and JLab
wrappers are described in [Architecture](ARCHITECTURE.md),
[Containers](CONTAINERS.md), and [JLab SWIF2](JLAB_SWIF2.md).

## Verification entry points

Run commands from the repository root. The package requires Python 3.11 or
newer; use the installed container runtime or an equivalent development
environment with the pinned runtime dependencies for Python tests.

```bash
./tests/run.sh static
DVCS_BRIDGE=/path/to/partons_bridge ./tests/run.sh native
DVCS_BRIDGE=/path/to/partons_bridge ./tests/run.sh quick
./tests/run.sh offline
```

`static` parses tracked Python/JSON, checks distribution and documentation
contracts, runs focused Python corpus/result tests, validates JLab wrapper
generation with local fixtures, syntax-checks shell entry points, and verifies
installer dry runs do not mutate state. It does not build PARTONS, start a
container, use a GPU, contact SWIF2, or run a scientific campaign.

`native` requires an executable bridge and runs its capabilities and self-test
only. `quick` also requires the Python scientific dependencies and an installed
native stack; it initializes a temporary synthetic project and generates the
default quick corpus, so it can be long-running and CPU-intensive. It does not
exercise CUDA, Apptainer, SWIF2, or the named-model holdout. `offline` requires
an existing `.dvcs/install.env` and smoke-tests the installed launcher help;
use the network-disabled container commands in [Containers](CONTAINERS.md) for
a stronger runtime network-isolation check.

The C++ tests are built and run by the native build/container process. Their
source files are in `cpp/partons_bridge/tests/`; do not report them as passed
until they run against the bridge and pinned dependency stack being accepted.
For complete local, CUDA, and JLab acceptance sequences, including checks that
require hardware or site credentials, use [Acceptance procedures](ACCEPTANCE.md).

## Documentation maintenance

When a public command, project field, artifact, dependency pin, native
capability, or workflow edge changes, update the corresponding reference page
and this map where applicable. Then run `./tests/run.sh static` in a supported
Python environment. It checks local Markdown targets and that the generated
schema-8 experiment has a documented leaf for every field. Review
`git diff --check`, `git diff -- docs README.md user`, and a case-insensitive
repository search for experiment-specific names before publishing a change.

Do not use a development checkout, a cluster path, a personal corpus name, or
an experimental branch as a general documentation example. Use relative paths
and placeholders such as `PROJECT`, `CORPUS`, and `/path/to/...`; document a
feature only when it is implemented on `main`.
