# Ontology-based Building Modeling Framework (Beta)

This project aims to provide a flexible framework for dynamic modelling of indoor climate and energy consumption in buildings.
It is based on the ontologies [SAREF4BLDG](https://saref.etsi.org/saref4bldg/) and [SAREF4SYST](https://saref.etsi.org/saref4syst/). 

This is a work-in-progress beta version and the functionality is therefore updated regularly.  




## Installation

The package can be install with pip and git as follows:
```bat
python -m pip install git+https://github.com/JBjoernskov/Twin4Build
```
The package has been tested for Python 3.7.12, but should also work for other 3.7.X versions. 
To generate a graph of the simulation model, [Graphviz](https://graphviz.org/download) must be installed separately (Remember to add the directory to system path).



## Documentation

The core modules of this package are currently:

[model.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/model.py): Contains the Model class, which represents the simulation model of the building.\
[simulator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/simulator/simulator.py): Contains the Simulator class, which can simulate a Model instance for a given period.\
[monitor.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/monitor/monitor.py): Contains the Monitor class, which can monitor and evaluate the performance of a building for a certain period by comparing readings from virtual measuring devices with readings from physical measuring devices.\
<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/212349194-958f9284-3411-4240-84a5-5acb80b2a8f6.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349204-15d13023-ab8a-4976-bbdd-3e8f2f1fbec7.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349206-abfa41c7-045e-4bb1-8ed4-f1fc5d739f7a.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349209-e5eaa4e4-bbfb-458a-9cdd-154b5f8050fa.png" width="400">
</p>


[evaluator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/evaluator/evaluator.py): Contains the Evaluator class, which can evaluate and compare Model instances on different metrics, e.g. energy consumption and indoor comfort.


### Model and Simulator
An example scipt showing the use of the Model class and how to simulate a Model instance is given in [test_model.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/tests/test_model.py).

### Monitor
An example scipt showing the use of the Monitor class and how to use a Monitor instance is given in [test_monitor.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/monitor/tests/test_monitor.py).

<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/229446212-8e2a4ebf-75d0-4ef7-86a2-08d3cb1987ae.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446232-00b53fba-8046-4b88-80dd-1a474cd8cfd5.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446234-dfd107a4-07a5-4e69-9110-2eff9b2735e4.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446238-636ed18f-c700-4285-bbe9-947ddade8ca2.png" width="400">
</p>

### Evaluator

The evaluator class is a wrapper of the Simulator 
The example scipt shown below demonstrates the use of the Evaluator class. 

https://github.com/JBjoernskov/Twin4Build/blob/cecc71406ba5adbea7bea91ee355ca529ac092e9/twin4build/evaluator/tests/test_evaluator.py#L1-L136




<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/229446225-b7b4ebf4-943d-43e3-88f6-e16f34046fca.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446228-1f668c00-43f8-4632-a1fa-b0935e7518b9.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/229446222-00e7acf4-d291-425c-8dd8-9b6f59345bc8.png" width="400">
</p>