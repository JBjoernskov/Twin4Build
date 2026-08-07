Examples and Tutorials
======================

All hands-on guides live in the example notebooks under
``twin4build/examples/`` -- each is a self-contained, runnable tutorial that
combines explanation with executable code. The theory and API reference live
in the class docstrings (see the `API Documentation <../auto/twin4build.html>`_,
in particular :class:`~twin4build.simulator.simulator.Simulator`,
:class:`~twin4build.estimator.estimator.Estimator`,
:class:`~twin4build.optimizer.optimizer.Optimizer`, and
:class:`~twin4build.translator.translator.Translator`).

Every notebook can be opened directly in Google Colab (badge links below) or
run locally in Jupyter. Colab badges are rewritten at docs build time to the
git branch/tag for this documentation version (e.g. ``dev`` docs open
``blob/dev/...``, ``latest`` opens ``blob/main/...``).

Basics of Twin4Build
--------------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/minimal_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Minimal example</b>: Connecting components, simulating a model, and visualizing results &mdash; the core mechanics every other example builds on</p>

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/building_space_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Building space</b>: The combined thermal + CO2 building space model driven by weather data, occupancy schedules, and ventilation, plus two zones coupled by an energy-conserving partition wall component</p>

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/space_heater_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Space heater</b>: A discretized space heater (radiator) model coupled to a building space and a PID temperature controller</p>

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/space_co2_controller_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>CO2 control</b>: Modeling and closed-loop control of indoor CO2 concentration with dampers and a PID controller</p>

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/bems_example_lecture.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Custom components (lecture)</b>: Adding a custom System component &mdash; RC modeling from scratch of 2 rooms with parameter estimation and heat optimization</p>

Translator
----------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/translator_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Translator</b>: Generating simulation models automatically from semantic (ontology-based) building descriptions via signature pattern matching</p>

Estimator
---------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/estimator_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Estimator</b>: Parameter estimation and calibration against measured data &mdash; single-shooting and collocation transcriptions, SciPy and CasADi/IPOPT backends, and result interpretation</p>

Optimizer
---------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/optimizer_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Optimizer</b>: Optimization of space heater power consumption, constrained by heating and cooling setpoints</p>

Full Workflow
-------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/full_workflow_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Full workflow</b>: End-to-end pipeline &mdash; translate a semantic model, calibrate parameters with the Estimator, and optimize a control schedule with the Optimizer</p>

Running Examples
----------------

Prerequisites
~~~~~~~~~~~~~

Before running examples, ensure you have:

1. **Twin4Build installed**: See the :doc:`installation` guide
2. **Jupyter Notebook**: ``pip install jupyter``

Running in Jupyter
~~~~~~~~~~~~~~~~~~

1. **Start Jupyter**:

   .. code-block:: bash

       jupyter notebook

2. **Navigate** to the ``twin4build/examples`` directory

3. **Open** the desired notebook and run cells sequentially

Troubleshooting Examples
------------------------

**Import Errors**

- Ensure Twin4Build is installed correctly
- Verify the Python environment

**Data File Errors**

- The notebooks download or ship their data files alongside the notebook;
  run them from the ``twin4build/examples`` directory so relative paths resolve

**Memory Issues**

- Reduce the simulation duration or step count
- Simplify model complexity

Getting Help
------------

If you encounter issues with examples:

1. **Check the documentation**: Review relevant sections in the :doc:`developer_reference`
2. **Examine the code**: Look at the example source code for implementation details
3. **Search issues**: Check `GitHub Issues <https://github.com/JBjoernskov/Twin4Build/issues>`_ for similar problems
4. **Ask questions**: Create a new issue with specific error information

Additional Resources
--------------------

- `API Documentation <../auto/twin4build.html>`_
- :doc:`developer_reference`
- `GitHub Repository <https://github.com/JBjoernskov/Twin4Build/>`_
