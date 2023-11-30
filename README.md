#  twin4build: A python package for Data-driven and Ontology-based modeling and simulation of buildings

twin4build is a python package which aims to provide a flexible and automated framework for dynamic modelling of indoor climate and energy consumption in buildings. It leverages the [SAREF core](https://saref.etsi.org/core/) ontology and its extensions [SAREF4BLDG](https://saref.etsi.org/saref4bldg/) and [SAREF4SYST](https://saref.etsi.org/saref4syst/).

This is a work-in-progress library and the functionality is therefore updated regularly.
More information on the use of the framework and code examples are coming in the near future!

<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/231081820-0049b8ab-2d28-4eb9-98dc-f7d7ef039ef8.png" width="800">
</p>


## Installation

| Python version  | Windows  | Ubuntu |
| :------------ |:---------------:| -----:|
| 3.8      | [![windows-python3.8](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-8.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-8.yml)        |   [![ubuntu-python3.8](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-8.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-8.yml) |
| 3.9 | [![windows-python3.9](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-9.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-9.yml)        |    [![ubuntu-python3.9](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-9.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-9.yml) |
| 3.10 | [![windows-python3.10](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-10.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-10.yml)        |    [![ubuntu-python3.10](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-10.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-10.yml) |
| 3.11 | [![windows-python3.11](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-11.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-11.yml)        |    [![ubuntu-python3.11](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-11.yml/badge.svg?branch=main)](https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-11.yml) |


The package can be installed with pip and git using one of the above python versions:
```bat
python -m pip install git+https://github.com/JBjoernskov/Twin4Build
```

[Graphviz](https://graphviz.org/download) must be installed separately:

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

## Getting started

https://github.com/JBjoernskov/Twin4Build/blob/e13d93008a212c866f9bbdc734411c38a64a19ba/twin4build/examples/space_co2_controller_example.py#L109-L122


## Documentation

The core modules of this package are currently:

[model.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/model.py): Contains the Model class, which represents the simulation model of the building.

[simulator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/simulator/simulator.py): Contains the Simulator class, which can simulate a Model instance for a given period.

[monitor.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/monitor/monitor.py): Contains the Monitor class, which can monitor and evaluate the performance of a building for a certain period by comparing readings from virtual measuring devices with readings from physical measuring devices.

[evaluator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/evaluator/evaluator.py): Contains the Evaluator class, which can evaluate and compare Model instances on different metrics, e.g. energy consumption and indoor comfort.


### Model and Simulator
An example scipt showing the use of the Model class and how to simulate a Model instance is given in [test_model.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/tests/test_model.py).

### Monitor
[This example script](https://github.com/JBjoernskov/Twin4Build/blob/HEAD/twin4build/monitor/tests/test_monitor.py) demonstrates the use of the Monitor class. 



Running this example generates the following figures, which compares physical with virtual sensor and meter readings on different components. The red line indicates the timestamp where operation of the physical system was drastically changed. A binary classification signal is also generated for each component which informs whether a component performs as expected (0) or not (1). 
<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/229446212-8e2a4ebf-75d0-4ef7-86a2-08d3cb1987ae.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446232-00b53fba-8046-4b88-80dd-1a474cd8cfd5.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446234-dfd107a4-07a5-4e69-9110-2eff9b2735e4.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446238-636ed18f-c700-4285-bbe9-947ddade8ca2.png" width="400">
</p>

### Evaluator

[This example script](https://github.com/JBjoernskov/Twin4Build/blob/HEAD/twin4build/evaluator/tests/test_evaluator.py) demonstrates the use of the Evaluator class. 
Running this example generates the following figures, which compares two different scenarios. 

<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/229446225-b7b4ebf4-943d-43e3-88f6-e16f34046fca.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446228-1f668c00-43f8-4632-a1fa-b0935e7518b9.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446222-00e7acf4-d291-425c-8dd8-9b6f59345bc8.png" width="400">
</p>


## Accessing time series data for running examples

[This folder](https://syddanskuni-my.sharepoint.com/:f:/g/personal/jabj_mmmi_sdu_dk/EutVYojScvhBgVBtglvkD3MB8L4GigGOB5ZR5qN6QAFGMA?e=sSCAI1) contains the necessary files for running some of the examples. It is password protected - contact JBjoernskov for password. Download the folder and paste the content into twin4build/test/data/time_series_data.

## Publications

+ [Bjørnskov, J., & Jradi, M. (2023). An Ontology-Based Innovative Energy Modeling Framework for Scalable and Adaptable Building Digital Twins. Energy and Buildings, 292, [113146].](https://doi.org/10.1016/j.enbuild.2023.113146)

+ [Bjørnskov, J., & Jradi, M. (Accepted/In press). Implementation and demonstration of an automated energy modeling framework for scalable and adaptable building digital twins based on the SAREF ontology. Building Simulation.](https://portal.findresearcher.sdu.dk/en/publications/implementation-and-demonstration-of-an-automated-energy-modeling-)

+ [Andersen, A. H., Bjørnskov, J., & Jradi, M. (2023). Adaptable and Scalable Energy Modeling of Ventilation Systems as Part of Building Digital Twins. In Proceedings of the 18th International IBPSA Building Simulation Conference: BS2023 International Building Performance Simulation Association.](https://portal.findresearcher.sdu.dk/en/publications/adaptable-and-scalable-energy-modeling-of-ventilation-systems-as-)






## Cite as
```yaml
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

