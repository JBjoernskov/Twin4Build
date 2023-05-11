# Ontology-based Building Modeling Framework (Beta)

This project aims to provide a flexible framework for dynamic modelling of indoor climate and energy consumption in buildings.
It is based on the [SAREF core](https://saref.etsi.org/core/) ontology and its extensions [SAREF4BLDG](https://saref.etsi.org/saref4bldg/) and [SAREF4SYST](https://saref.etsi.org/saref4syst/).

This is a work-in-progress beta version and the functionality is therefore updated regularly.
More information on the use of the framework and code examples are coming in the near future!

<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/231081820-0049b8ab-2d28-4eb9-98dc-f7d7ef039ef8.png" width="800">
</p>


## Installation

The package can be install with pip and git as follows:
```bat
python -m pip install git+https://github.com/JBjoernskov/Twin4Build
```
The package has been tested for Python 3.7.12, but should also work for other 3.7.X versions. 
To generate a graph of the simulation model, [Graphviz](https://graphviz.org/download) must be installed separately (Remember to add the directory to system path).



## Documentation

The core modules of this package are currently:

[model.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/model.py): Contains the Model class, which represents the simulation model of the building.

[simulator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/simulator/simulator.py): Contains the Simulator class, which can simulate a Model instance for a given period.

[monitor.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/monitor/monitor.py): Contains the Monitor class, which can monitor and evaluate the performance of a building for a certain period by comparing readings from virtual measuring devices with readings from physical measuring devices.

[evaluator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/evaluator/evaluator.py): Contains the Evaluator class, which can evaluate and compare Model instances on different metrics, e.g. energy consumption and indoor comfort.


### Model and Simulator
An example scipt showing the use of the Model class and how to simulate a Model instance is given in [test_model.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/tests/test_model.py).

### Monitor
[This example script](https://github.com/JBjoernskov/Twin4Build/blob/HEAD/twin4build/monitor/tests/test_monitor.py) shown below demonstrates the use of the Monitor class. 


```python
import requests
```
<!-- https://github.com/JBjoernskov/Twin4Build/blob/7e96bdae6f01f1417a69ab0462d9b9187daa39ba/twin4build/monitor/tests/test_monitor.py#L1-L106 -->



Running this example generates the following figures, which compares physical with virtual sensor and meter readings on different components. The red line indicates the timestamp where operation of the physical system was drastically changed. A binary classification signal is also generated for each component which informs whether a component performs as expected (0) or not (1). 
<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/229446212-8e2a4ebf-75d0-4ef7-86a2-08d3cb1987ae.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446232-00b53fba-8046-4b88-80dd-1a474cd8cfd5.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446234-dfd107a4-07a5-4e69-9110-2eff9b2735e4.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446238-636ed18f-c700-4285-bbe9-947ddade8ca2.png" width="400">
</p>

### Evaluator

The example script shown below demonstrates the use of the Evaluator class. 

https://github.com/JBjoernskov/Twin4Build/blob/6cde0a90c7cf1e46085ebd42b49e9a083841bec3/twin4build/evaluator/tests/test_evaluator.py#L1-L136

Running this example generates the following figures, which compares two different scenarios. 

<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/229446225-b7b4ebf4-943d-43e3-88f6-e16f34046fca.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446228-1f668c00-43f8-4632-a1fa-b0935e7518b9.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446222-00e7acf4-d291-425c-8dd8-9b6f59345bc8.png" width="400">
</p>


## Accessing time series data for running examples

[This folder](https://syddanskuni-my.sharepoint.com/:f:/g/personal/jabj_mmmi_sdu_dk/EutVYojScvhBgVBtglvkD3MB8L4GigGOB5ZR5qN6QAFGMA?e=HDNuGv) contains the necessary files for running some of the examples. It is password protected - contact JBjoernskov for password. 

## Cite as
```yaml
@article{OntologyBasedBuildingModelingFramework,
  title={{An Ontology-Based Innovative Energy Modeling Framework for Scalable and Adaptable Building Digital Twins}},
  author={Bjørnskov, Jakob and Jradi, Muhyiddine},
  Url= {https://ssrn.com/abstract=4341936 or http://dx.doi.org/10.2139/ssrn.4341936}, 
  year={2023}}
```

