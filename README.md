
<p align="center">
    <img src="https://raw.githubusercontent.com/JBjoernskov/Twin4Build/main/Twin4build_logo.jpg" width="400">
</p>

[![docs](https://app.readthedocs.org/projects/twin4build/badge/?version=latest)](https://twin4build.readthedocs.io/en/latest/)
[![docs-dev](https://app.readthedocs.org/projects/twin4build/badge/?version=dev)](https://twin4build.readthedocs.io/en/dev/)


# twin4build: A python package for Data-driven and Ontology-based modeling and simulation of buildings

Dynamic modeling and simulation of buildings, featuring fully differentiable models for parameter estimation and optimal control. Supports integration of semantic models for automatic model generation and fast implementation. 


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

A typical workflow would look like this:
<p align="center">
    <img src="https://raw.githubusercontent.com/JBjoernskov/Twin4Build/main/docs/source/_static/t4b_workflow.png" width="800">
</p>


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

## Documentation
- **Latest (`main`)**: https://twin4build.readthedocs.io/en/latest/
- **Dev**: https://twin4build.readthedocs.io/en/dev/

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
model.to("cuda", torch.float32)      # opt-in single precision (fast on consumer GPUs)
model.to("cpu", torch.float64)       # back to the defaults
```

Two things to know:

- **`float32` is the mode that pays off on consumer GPUs** (e.g. a Colab T4 runs float64 at 1/32 of its float32 throughput). The default stays `float64` everywhere; `dtype` is a process-wide setting, so use one dtype per process.
- **A single small-model run is not faster on the GPU.** Simulation steps of a few-zone model are microsecond-scale operations, so kernel-launch latency dominates at batch size 1. GPU execution is the enabler for *batched* workflows - multi-start estimation, scenario/ensemble studies, portfolios of buildings (via the component batch dimension `n_c`) - and for large many-zone models. See `twin4build/examples/gpu_benchmark_estimation.ipynb` and `twin4build/examples/gpu_benchmark_optimizer.ipynb` for measurements.

## Installation

The package is installed with pip:

```bat
pip install twin4build
```

Optional extras:

```bat
pip install twin4build[database]     # PostgreSQL connectivity
pip install twin4build[all]          # Everything
```

The following python versions are supported:

| Python version  | Windows  | Ubuntu |
| :------------ |---------------:| -----:|
| 3.9 | [![windows-python3.9](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-9.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-9.yml)        |    [![ubuntu-python3.9](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-9.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-9.yml) |
| 3.10 | [![windows-python3.10](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-10.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-10.yml)        |    [![ubuntu-python3.10](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-10.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-10.yml) |
| 3.11 | [![windows-python3.11](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-11.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-11.yml)        |    [![ubuntu-python3.11](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-11.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-11.yml) |
| 3.12 | [![windows-python3.12](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-12.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-12.yml)        |    [![ubuntu-python3.12](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-12.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-12.yml) |




### Graphviz (recomended)

To utilize the graph-drawing capabilities of twin4build, the drawing engine [Graphviz](https://graphviz.org/download) must be installed.
It can be installed by downloading the install-file from the official website or by using your favorite package manager: 

#### Ubuntu
```bat
sudo add-apt-repository universe
sudo apt update
sudo apt install graphviz
```

#### Windows
On windows, the winget or choco package managers can be used:
```bat
winget install graphviz
```
```bat
choco install graphviz
```

#### MacOS
```bat
brew install graphviz
```
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

