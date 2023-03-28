# Twin4Build (Beta)

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

[model.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/model.py): Contains the Model class, which represents the simulation model of the building. 
[simulator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/simulator/simulator.py): Contains the Simulator class, which can simulate a Model instance for a given period. 
[monitor.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/monitor/monitor.py): Contains the Monitor class, which can monitor and evaluate the performance of a building for a certain period by comparing readings from virtual measuring devices with readings from physical measuring devices.
[evaluator.py](https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/evaluator/evaluator.py): Contains the Evaluator class, which can evaluate and compare Model instances on different metrics, e.g. energy consumption and indoor comfort.

<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/212348894-bb581b90-6824-4ada-a1d9-311c113ab174.png" width="800">
</p>

<p float="left">
    <img src="https://user-images.githubusercontent.com/74002963/212349194-958f9284-3411-4240-84a5-5acb80b2a8f6.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349204-15d13023-ab8a-4976-bbdd-3e8f2f1fbec7.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349206-abfa41c7-045e-4bb1-8ed4-f1fc5d739f7a.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349209-e5eaa4e4-bbfb-458a-9cdd-154b5f8050fa.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349210-a7e5ac4b-cb29-403f-b763-df319d1cc999.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349211-f8de7f96-bd3b-4725-bae6-76b0d40408f5.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349215-c7bdfe84-9b4e-4ac9-9970-e6ea130e7248.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349218-ddc97b0a-277b-492a-b1d6-6ddeba06304a.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349221-911abdce-616e-4511-b530-251f5f2eebd3.png" width="400">
    <img src="https://user-images.githubusercontent.com/74002963/212349222-7b1cbc95-a0a5-495f-9b44-7b5805c62396.png" width="400">
</p>