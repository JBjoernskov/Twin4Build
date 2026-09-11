
[![docs](https://app.readthedocs.org/projects/twin4build/badge/?version=latest)](https://twin4build.readthedocs.io/en/latest/)
[![docs-dev](https://app.readthedocs.org/projects/twin4build/badge/?version=dev)](https://twin4build.readthedocs.io/en/dev/)


# twin4build: A python package for Data-driven and Ontology-based modeling and simulation of buildings

Dynamic modeling and simulation of buildings, featuring fully differentiable models for parameter estimation and optimal control. Supports integration of semantic models for automatic model generation and rapid implementation.


## Core Classes and Functionality

Twin4Build provides several top-level classes for building, simulating, translating, calibrating, and optimizing building energy models:

- **Model**:  
  The main container for your building system, components, and their connections. Use this class to assemble your digital twin from reusable components. 

- **Simulator**:  
  Runs time-based simulations of your Model, producing time series outputs for all components. Handles the simulation loop and time stepping.

- **Translator**:  
  Automatically generates a Model from a semantic model (ontology-based building description) and maintains a link between these. Enables ontology-driven, automated model creation.

- **Estimator**:  
  Performs parameter estimation (calibration) for your Model using measured data. Supports gradient-based optimization with automatic differentiation (SciPy and CasADi/IPOPT backends, single-shooting or collocation).

- **Optimizer**:  
  Optimizes building operation by adjusting setpoints or control variables to minimize objectives or satisfy constraints, using gradient-based methods.



All classes are accessible via the main package import:
```python
import twin4build as tb
```

A typical workflow is Model → Simulator → Estimator or Optimizer. Execution
policy is selected on `Simulator`; solver settings are passed through
`estimate(..., options=...)` or `optimize(..., options=...)`.


## Examples and Tutorials
Notebooks live in [`twin4build/examples/`](twin4build/examples/) on **this branch** (relative links always open the files from the branch you are viewing on GitHub).

GitHub READMEs cannot parameterize Colab URLs by viewing branch — absolute Colab links are fixed in the file. Prefer the version-matched Colab badges on the docs site:

- [Examples (latest / `main`)](https://twin4build.readthedocs.io/en/latest/manual/examples_and_tutorials.html)
- [Examples (`dev`)](https://twin4build.readthedocs.io/en/dev/manual/examples_and_tutorials.html)

### Basics of Twin4Build
[minimal_example.ipynb](twin4build/examples/minimal_example.ipynb) — Part 1: Connecting components, simulating a model, and visualization

[space_co2_controller_example.ipynb](twin4build/examples/space_co2_controller_example.ipynb) — Part 2: Modeling and control of indoor CO2 concentration

[bems_example_lecture.ipynb](twin4build/examples/bems_example_lecture.ipynb) — Part 3: Adding a custom System component - RC modeling from scratch of 2 rooms with parameter estimation and heat optimization

### Translator

[translator_example.ipynb](twin4build/examples/translator_example.ipynb) — Part 1: How to use the translator to generate simulation models from semantic models.

### Estimator

[estimator_example.ipynb](twin4build/examples/estimator_example.ipynb) — Part 1: Basic parameter estimation and calibration

### Optimizer

[optimizer_example.ipynb](twin4build/examples/optimizer_example.ipynb) — Part 1: Optimization of space heater power consumption, constrained by heating and cooling setpoints.

Bi-objective fronts use AUGMECON through `Optimizer.pareto_front`. The
authoritative method matrix compares SLSQP direct shooting,
`("scipy", "SLSQP", "ad")`, with the IPOPT solver using collocation
transcription,
`("casadi", "ipopt", "ad", "collocation")`. IPOPT promotes every augmented
one-step boundary state to the decision vector, enforces dynamics as hard
continuity defects, and keeps comfort limits as soft penalties; the epsilon
row is the only non-dynamics hard inequality. Same-class components can be
grouped with `model.batch_components()`; inspect the layout with
`get_batched_component_info()` and `get_batch_id_for_component()`.

Simulation has two independent policy dimensions. `execution_mode="object"`
runs component `do_step` methods, while `execution_mode="functional"` uses
the functional one-step model. `execution_backend="eager"` is the default;
`execution_backend="cuda_graph"` captures and replays fixed-shape functional
execution on CUDA. CUDA Graph is a backend, not a mode. Functional workflows
can also use `build_functional_model`, `record_exogenous_inputs`, and
`rollout_functional` directly.

Model layout is independent of execution policy: a model is either
`standard` or `batched`, and either layout may use supported execution modes.
Solver options select Hessian strategy with
`hessian="exact"|"gauss_newton"|"limited_memory"`; the default is `exact`.

## Documentation
- **Latest (`main`)**: https://twin4build.readthedocs.io/en/latest/
- **Dev**: https://twin4build.readthedocs.io/en/dev/
- **Canonical benchmarks on this branch**: [`benchmarks/`](benchmarks/)
- **Benchmark methodology (`main`)**: https://twin4build.readthedocs.io/en/latest/manual/benchmarks.html
- **Benchmark methodology (`dev`)**: https://twin4build.readthedocs.io/en/dev/manual/benchmarks.html

The three canonical scaling notebooks use the complete translated 23-component
`full_workflow` graph at exactly 1, 10, 50, and 100 zones. The one-zone
rows replace the former standalone baselines. Simulation, estimation, and the
combined optimization/Pareto notebook retain explicit device/method
applicability, quality, safety preflights, and incremental raw checkpoints.

Below is a code snippet showing the basic functionality of the package.
```python
import datetime
import pytz
import twin4build as tb

# Create a model
model = tb.Model(id="example_model")

# Define components
damper = tb.DamperSystem(id="damper")
space = tb.BuildingSpaceSystem(id="space")

# Add connections to the model
model.add_connection(damper, space,
                     "airFlowRate", "supplyAirFlowRate")

# Load the model
model.load()

# Create a simulator instance
simulator = tb.Simulator(model)

# Simulate the model (timezone-aware datetimes are required)
step_size = 600  # Seconds
start_time = datetime.datetime(year=2025, month=1, day=10, tzinfo=pytz.UTC)
end_time = datetime.datetime(year=2025, month=1, day=12, tzinfo=pytz.UTC)
simulator.simulate(step_size=step_size,
                   start_time=start_time,
                   end_time=end_time)

# Plot the results
tb.plot.plot(
    simulator.date_time_steps,
    entries=[
        tb.plot.Entry(data=damper.output["airFlowRate"].history(), label="Air flow rate", axis=1),
        tb.plot.Entry(data=damper.output["damperPosition"].history(), label="Damper position", axis=2),
    ],
    ylabel_1axis="Air flow rate [kg/s]",
    ylabel_2axis="Damper position",
    show=True,
)
```

### GPU support

Models expose a torch-style `to(device, dtype)` API. After `model.load()`, a single call moves every component tensor (parameters and their bounds, states, state-space matrices, schedule tables) to the target device, and the Simulator, Estimator, and Optimizer compute there end to end - data only returns to the CPU at the scipy/IPOPT and plotting boundaries:

```python
model.to("cuda")                     # run on the GPU in float64 (the default dtype)
model.to("cuda", torch.float32)      # opt-in single precision
model.to("cpu", torch.float64)       # back to the defaults
```

Two things to know:

- **Precision is part of the experiment.** Consumer GPUs can have very different
  float32 and float64 throughput. The default remains `float64`; record dtype,
  device, and hardware with every performance result.
- **GPU speedups are workload-specific.** Kernel launch, host-side solver work,
  batch size, model size, and compilation policy can dominate different cases.
  Use the canonical [`benchmarks/`](benchmarks/) methodology instead of
  generalizing from one run.

## Installation

The package is installed with pip:

```bat
pip install twin4build
```

Optional extras:

```bat
pip install twin4build[database]     # PostgreSQL connectivity
pip install twin4build[gpu] --extra-index-url https://download.pytorch.org/whl/cu128
pip install twin4build[all]          # Dev + database (not CUDA torch)
```

### GPU / CUDA installation

`pip install twin4build` installs whatever `torch` wheel PyPI serves for the
platform. On Windows that is the **CPU-only** build, so
`torch.cuda.is_available()` is False even on a machine with a supported GPU,
and GPU execution modes run on CPU or fail with an opaque torch error.

Install a CUDA 12.8+ build (Blackwell needs cu128 or newer):

```bat
pip install twin4build[gpu] --extra-index-url https://download.pytorch.org/whl/cu128
```

If CPU torch is already installed, reinstall it from that index:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install twin4build[gpu]
```

`[gpu]` pulls Triton on Linux. Triton ships no Windows wheels, so the compiled
step (`compile_step`) and CUDA-graph paths need **Linux or WSL** — native
Windows can use CUDA eager execution after the CUDA torch install, but not
Inductor/Triton.

`model.to("cuda")` now raises with this install line when the process has no
CUDA, instead of torch's "not compiled with CUDA enabled".

The following python versions are supported (Twin4Build 2.0 requires Python 3.10+; 3.9 is no longer supported):

[![CI](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ci.yml)

One matrix workflow runs the suite on Windows and Ubuntu against Python 3.10, 3.11 and 3.12.




### Graphviz (included)

Graph drawing uses [Graphviz](https://graphviz.org) through [pygraphviz](https://pygraphviz.github.io) 2.0+. The pygraphviz wheel bundles the Graphviz libraries, so `pip install twin4build` is enough — you do not need apt, winget, choco, or brew.

The bundled Graphviz is licensed under EPL-2.0 (see pygraphviz's `LICENSE.graphviz`). Matplotlib's optional LaTeX text rendering is a separate system binary and is not included. Skip drawing with `draw_semantic_model=False` / `draw_simulation_model=False`.

### psycopg2 binaries (Linux-only)
You might need to install the tools to build psycopg2 from source, here is an example for Ubuntu:

```bat
sudo apt-get update
sudo apt-get install -y python3-dev libpq-dev build-essential
```

## Publications
<a id="1">[1]</a> 
[Bjørnskov, J. & Thomsen, A. & Jradi, M. (2025). Large-scale field demonstration of an interoperable and ontology-based energy modeling framework for building digital twins. Applied Energy, 387, [125597]](https://doi.org/10.1016/j.apenergy.2025.125597)

<a id="2">[2]</a> 
[Bjørnskov, J. & Jradi, M. & Wetter, M. (2025). Automated Model Generation and Parameter Estimation of Building Energy Models Using an Ontology-Based Framework. Energy and Buildings 329, [115228]](https://doi.org/10.1016/j.enbuild.2024.115228)

<a id="3">[3]</a> 
[Bjørnskov, J. & Jradi, M. (2023). An Ontology-Based Innovative Energy Modeling Framework for Scalable and Adaptable Building Digital Twins. Energy and Buildings, 292, [113146].](https://doi.org/10.1016/j.enbuild.2023.113146)

<a id="3">[4]</a> 
[Bjørnskov, J., Badhwar, A., Singh, D., Sehgal, M., Åkesson, R., & Jradi, M. (2025). Development and demonstration of a digital twin platform leveraging ontologies and data-driven simulation models. Journal of Building Performance Simulation, 1–13.](https://doi.org/10.1080/19401493.2025.2504005)

<a id="4">[5]</a> 
[Bjørnskov, J. & Jradi, M. (2023). Implementation and demonstration of an automated energy modeling framework for scalable and adaptable building digital twins based on the SAREF ontology. Building Simulation.](https://portal.findresearcher.sdu.dk/en/publications/implementation-and-demonstration-of-an-automated-energy-modeling-)

<a id="5">[6]</a> 
[Andersen, A. H. & Bjørnskov, J. & Jradi, M. (2023). Adaptable and Scalable Energy Modeling of Ventilation Systems as Part of Building Digital Twins. In Proceedings of the 18th International IBPSA Building Simulation Conference: BS2023 International Building Performance Simulation Association.](https://portal.findresearcher.sdu.dk/en/publications/adaptable-and-scalable-energy-modeling-of-ventilation-systems-as-)






## Cite as
```bibtex
@article{OntologyBasedBuildingModelingFramework,
    title = {An ontology-based innovative energy modeling framework for scalable and adaptable building digital twins},
    journal = {Energy and Buildings},
    volume = {292},
    pages = {113146},
    year = {2023},
    issn = {0378-7788},
    doi = {https://doi.org/10.1016/j.enbuild.2023.113146},
    url = {https://www.sciencedirect.com/science/article/pii/S0378778823003766},
    author = {Jakob Bjørnskov and Muhyiddine Jradi},
    keywords = {Digital twin, Data-driven, Building energy model, Building simulation, Ontology, SAREF},
}
```

