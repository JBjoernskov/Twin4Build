Examples and Tutorials
=====================

.. .. include:: ../../../README.md
..    :parser: myst_parser.sphinx_
..    :start-after: ## Examples and Tutorials
..    :end-before: ## Documentation


This guide provides an overview of the available examples and tutorials for Twin4Build.

Basics of Twin4Build
--------------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/minimal_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Connecting components, simulating a model, and visualization</p>

.. code-block:: python

    import twin4build as tb
    import twin4build.utils.plot.plot as plot
    
    # Create a model
    model = tb.Model(id="example_model")
    
    # Add components and connections
    # Run simulation
    # Visualize results

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/space_co2_controller_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 2: Modeling and control of indoor CO2 concentration</p>

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/bems_example_lecture.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 3: Adding a custom System component - RC modeling from scratch of 2 rooms with parameter estimation and heat optimization</p>

Translator
----------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/translator_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: How to use the translator to generate simulation models from semantic models</p>

.. code-block:: python

    from twin4build import Translator
    
    # Create translator
    translator = Translator()
    
    # Load semantic model
    # Generate simulation model
    # Validate translation

Estimator
---------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/estimator_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Basic parameter estimation and calibration</p>

.. code-block:: python

    from twin4build import Estimator
    
    # Create estimator
    estimator = Estimator()
    
    # Load measured data
    # Define parameters to estimate
    # Run calibration
    # Analyze results

Optimizer
---------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/optimizer_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Part 1: Optimization of space heater power consumption, constrained by heating and cooling setpoints</p>

.. code-block:: python

    from twin4build import Optimizer
    
    # Create optimizer
    optimizer = Optimizer()
    
    # Define objective function
    # Set constraints
    # Run optimization
    # Analyze optimal solutions

Full Workflow
-------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/full_workflow_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> End-to-end pipeline: translate a semantic model, calibrate parameters with the Estimator, and optimize a control schedule with the Optimizer</p>

.. code-block:: python

    import twin4build as tb

    # 1. Translate a semantic model into a simulation model
    translator = tb.Translator()
    model = translator.translate(...)

    # 2. Calibrate parameters against measurements
    estimator = tb.Estimator(tb.Simulator(model))
    estimator.estimate(parameters=...)

    # 3. Optimize a control schedule on the calibrated model
    optimizer = tb.Optimizer(tb.Simulator(model))
    optimizer.optimize(variables=...)

Running Examples
---------------

Prerequisites
~~~~~~~~~~~~~

Before running examples, ensure you have:

1. **Twin4Build installed**: See [Installation Guide](installation.rst)
2. **Jupyter Notebook**: `pip install jupyter`
3. **Required data files**: Some examples require specific data files

Running in Jupyter
~~~~~~~~~~~~~~~~~~

1. **Start Jupyter**:

   .. code-block:: bash

       jupyter notebook

2. **Navigate** to the examples directory:

   .. code-block:: bash

       cd twin4build/examples

3. **Open** the desired notebook and run cells sequentially

Example Structure
-----------------

Each example typically follows this structure:

1. **Setup and Imports**
   - Import required modules
   - Configure logging and settings

2. **Model Creation**
   - Define building components
   - Establish connections
   - Set initial conditions

3. **Simulation/Processing**
   - Run simulations or analysis
   - Handle data processing

4. **Results and Visualization**
   - Plot results
   - Generate reports
   - Export data

5. **Analysis and Discussion**
   - Interpret results
   - Compare with expectations

Troubleshooting Examples
-----------------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**
- Ensure Twin4Build is installed correctly
- Check that all dependencies are available
- Verify Python environment

**Data File Errors**
- Download required data files
- Check file paths and permissions
- Verify data format compatibility

**Memory Issues**
- Reduce simulation duration
- Simplify model complexity
- Increase system memory

Getting Help
-----------

If you encounter issues with examples:

1. **Check the documentation**: Review relevant sections in the developer reference
2. **Examine the code**: Look at the example source code for implementation details
3. **Search issues**: Check GitHub Issues for similar problems
4. **Ask questions**: Create a new issue with specific error information

Additional Resources
-------------------

- `API Documentation <../auto/twin4build>`_
- `Developer Reference <developer_reference>`_
- `GitHub Repository <https://github.com/JBjoernskov/Twin4Build/>`_