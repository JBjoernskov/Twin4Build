twin4build: A python package for Data-driven and Ontology-based modeling and simulation of buildings
====================================================================================================

twin4build is a python package which aims to provide a flexible and
automated framework for dynamic modelling of indoor climate and energy
consumption in buildings. It leverages the `SAREF
core <https://saref.etsi.org/core/>`__ ontology and its extensions
`SAREF4BLDG <https://saref.etsi.org/saref4bldg/>`__ and
`SAREF4SYST <https://saref.etsi.org/saref4syst/>`__.

This is a work-in-progress library and the functionality is therefore
updated regularly. More information on the use of the framework and code
examples are coming in the near future!

.. image:: https://user-images.githubusercontent.com/74002963/231081820-0049b8ab-2d28-4eb9-98dc-f7d7ef039ef8.png
  :width: 800
  :alt: Example model


Installation
------------
|windows-python3.8| |ubuntu-python3.8|
|windows-python3.9| |ubuntu-python3.9|
|windows-python3.10| |ubuntu-python3.10|
|windows-python3.11| |ubuntu-python3.11|

The package can be installed with pip and git using one of the above
python versions:

.. code:: bat

   python -m pip install git+https://github.com/JBjoernskov/Twin4Build

Graphviz
~~~~~~~~

`Graphviz <https://graphviz.org/download>`__ must be installed
separately:

Ubuntu
^^^^^^

.. code:: bat

   sudo add-apt-repository universe
   sudo apt update
   sudo apt install graphviz

Windows
^^^^^^^

On windows, the winget or choco package managers can be used:

.. code:: bat

   winget install graphviz

.. code:: bat

   choco install graphviz

MacOS
^^^^^

.. code:: bat

   brew install graphviz

Getting started
---------------

Below is a simple example

.. code:: python


   import twin4build as tb
   import twin4build.utils.plot.plot as plot


   def fcn(self):
       ##############################################################
       ################## First, define components ##################
       ##############################################################

       #Define a schedule for the damper position
       position_schedule = tb.ScheduleSystem(
               weekDayRulesetDict = {
                   "ruleset_default_value": 0,
                   "ruleset_start_minute": [0,0,0,0,0,0,0],
                   "ruleset_end_minute": [0,0,0,0,0,0,0],
                   "ruleset_start_hour": [6,7,8,12,14,16,18],
                   "ruleset_end_hour": [7,8,12,14,16,18,22],
                   "ruleset_value": [0,0.1,1,0,0,0.5,0.7]}, #35
               add_noise=False,
               saveSimulationResult = self.saveSimulationResult,
               id="Position schedule")

       # Define damper component
       damper = tb.DamperSystem(
           nominalAirFlowRate = Measurement(hasValue=1.6),
           a=5,
           saveSimulationResult=self.saveSimulationResult,
           id="Damper")

       #################################################################
       ################## Add connections to the model #################
       #################################################################
       self.add_connection(position_schedule, damper, 
                           "scheduleValue", "damperPosition")

       # Cycles are not allowed (with the exeption of controllers - see the controller example). If the following line is commented in, 
       # a cycle is introduced and the model will generate an error when "model.get_execution_order()" is run". 
       # You can see the generated graph with the cycle in the "system_graph.png" file.
       # self.add_connection(damper, damper, "airFlowRate", "damperPosition") #<------------------- comment in to create a cycle


   model = tb.Model(id="example_model", saveSimulationResult=True)
   model.load_model(infer_connections=False, fcn=fcn)

   # Create a simulator instance
   simulator = tb.Simulator()

   # Simulate the model
   stepSize = 600 #Seconds
   startTime = datetime.datetime(year=2021, month=1, day=10, hour=0, minute=0, second=0)
   endTime = datetime.datetime(year=2021, month=1, day=12, hour=0, minute=0, second=0)
   simulator.simulate(model,
                       stepSize=stepSize,
                       startTime=startTime,
                       endTime=endTime)

   plot.plot_damper(model, simulator, "Damper", show=False) #Set show=True to plot

Documentation
-------------

The core modules of this package are currently:

`model.py <https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/model.py>`__:
Contains the Model class, which represents the simulation model of the
building.

`simulator.py <https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/simulator/simulator.py>`__:
Contains the Simulator class, which can simulate a Model instance for a
given period.

`monitor.py <https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/monitor/monitor.py>`__:
Contains the Monitor class, which can monitor and evaluate the
performance of a building for a certain period by comparing readings
from virtual measuring devices with readings from physical measuring
devices.

`evaluator.py <https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/evaluator/evaluator.py>`__:
Contains the Evaluator class, which can evaluate and compare Model
instances on different metrics, e.g. energy consumption and indoor
comfort.

Model and Simulator
~~~~~~~~~~~~~~~~~~~

An example scipt showing the use of the Model class and how to simulate
a Model instance is given in
`test_model.py <https://github.com/JBjoernskov/Twin4Build/blob/main/twin4build/model/tests/test_model.py>`__.

Monitor
~~~~~~~

`This example
script <https://github.com/JBjoernskov/Twin4Build/blob/HEAD/twin4build/monitor/tests/test_monitor.py>`__
demonstrates the use of the Monitor class.

Running this example generates the following figures, which compares
physical with virtual sensor and meter readings on different components.
The red line indicates the timestamp where operation of the physical
system was drastically changed. A binary classification signal is also
generated for each component which informs whether a component performs
as expected (0) or not (1).


.. image:: https://user-images.githubusercontent.com/74002963/229446212-8e2a4ebf-75d0-4ef7-86a2-08d3cb1987ae.png
  :width: 400

.. image:: https://user-images.githubusercontent.com/74002963/229446232-00b53fba-8046-4b88-80dd-1a474cd8cfd5.png
  :width: 400

.. image:: https://user-images.githubusercontent.com/74002963/229446234-dfd107a4-07a5-4e69-9110-2eff9b2735e4.png
  :width: 400

.. image:: https://user-images.githubusercontent.com/74002963/229446238-636ed18f-c700-4285-bbe9-947ddade8ca2.png
  :width: 400


Evaluator
~~~~~~~~~

`This example
script <https://github.com/JBjoernskov/Twin4Build/blob/HEAD/twin4build/evaluator/tests/test_evaluator.py>`__
demonstrates the use of the Evaluator class. Running this example
generates the following figures, which compares two different scenarios.

.. image:: https://user-images.githubusercontent.com/74002963/229446225-b7b4ebf4-943d-43e3-88f6-e16f34046fca.png
  :width: 400

.. image:: https://user-images.githubusercontent.com/74002963/229446228-1f668c00-43f8-4632-a1fa-b0935e7518b9.png
  :width: 400

.. image:: https://user-images.githubusercontent.com/74002963/229446222-00e7acf4-d291-425c-8dd8-9b6f59345bc8.png
  :width: 400

Accessing time series data for running examples
-----------------------------------------------

`This
folder <https://syddanskuni-my.sharepoint.com/:f:/g/personal/jabj_mmmi_sdu_dk/EutVYojScvhBgVBtglvkD3MB8L4GigGOB5ZR5qN6QAFGMA?e=sSCAI1>`__
contains the necessary files for running some of the examples. It is
password protected - contact JBjoernskov for password. Download the
folder and paste the content into twin4build/test/data/time_series_data.

Publications
------------

-  `Bjørnskov, J., & Jradi, M. (2023). An Ontology-Based Innovative
   Energy Modeling Framework for Scalable and Adaptable Building Digital
   Twins. Energy and Buildings, 292,
   [113146]. <https://doi.org/10.1016/j.enbuild.2023.113146>`__

-  `Bjørnskov, J., & Jradi, M. (Accepted/In press). Implementation and
   demonstration of an automated energy modeling framework for scalable
   and adaptable building digital twins based on the SAREF ontology.
   Building
   Simulation. <https://portal.findresearcher.sdu.dk/en/publications/implementation-and-demonstration-of-an-automated-energy-modeling->`__

-  `Andersen, A. H., Bjørnskov, J., & Jradi, M. (2023). Adaptable and
   Scalable Energy Modeling of Ventilation Systems as Part of Building
   Digital Twins. In Proceedings of the 18th International IBPSA
   Building Simulation Conference: BS2023 International Building
   Performance Simulation
   Association. <https://portal.findresearcher.sdu.dk/en/publications/adaptable-and-scalable-energy-modeling-of-ventilation-systems-as->`__

Cite as
-------

.. code:: yaml

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

.. |windows-python3.8| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-8.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-8.yml
.. |ubuntu-python3.8| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-8.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-8.yml
.. |windows-python3.9| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-9.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-9.yml
.. |ubuntu-python3.9| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-9.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-9.yml
.. |windows-python3.10| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-10.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-10.yml
.. |ubuntu-python3.10| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-10.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-10.yml
.. |windows-python3.11| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-11.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/win-py3-11.yml
.. |ubuntu-python3.11| image:: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-11.yml/badge.svg?branch=main
   :target: https://github.com/JBjoernskov/Twin4Build/actions/workflows/ub-py3-11.yml
