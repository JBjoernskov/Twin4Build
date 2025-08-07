Examples and Tutorials
=====================

.. .. include:: ../../../README.md
..    :parser: myst_parser.sphinx_
..    :start-after: ## Examples and Tutorials
..    :end-before: ## Documentation


This guide provides an overview of the available examples and tutorials for Twin4Build.

Basics of Twin4Build
--------------------

**Minimal Example** (`minimal_example.ipynb`)

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/minimal_example.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

- **Purpose**: Connecting components, simulating a model, and visualization

.. code-block:: python

    import twin4build as tb
    import twin4build.utils.plot.plot as plot
    
    # Create a model
    model = tb.Model(id="example_model")
    
    # Add components and connections
    # Run simulation
    # Visualize results

**Space CO2 Controller Example** (`space_co2_controller_example.ipynb`)

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/space_co2_controller_example.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

- **Purpose**: Modeling and control of indoor CO2 concentration

Translator Examples
-------------------

**Translator Example** (`translator_example.ipynb`)

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/translator_example.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

- **Purpose**: How to use the translator to generate simulation models from semantic models
- **Topics**: Ontology-driven modeling, automated model creation

.. code-block:: python

    from twin4build import Translator
    
    # Create translator
    translator = Translator()
    
    # Load semantic model
    # Generate simulation model
    # Validate translation

Estimator Examples
------------------

**Estimator Example** (`estimator_example.ipynb`)

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/estimator_example.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

- **Purpose**: Basic parameter estimation and calibration
- **Topics**: Least-squares optimization, PyTorch-based calibration

.. code-block:: python

    from twin4build import Estimator
    
    # Create estimator
    estimator = Estimator()
    
    # Load measured data
    # Define parameters to estimate
    # Run calibration
    # Analyze results

Optimizer Examples
------------------

**Optimizer Example** (`optimizer_example.ipynb`)

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/main/twin4build/examples/optimizer_example.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

- **Purpose**: Optimization of space heater power consumption, constrained by heating and cooling setpoints
- **Topics**: Gradient-based optimization, constraint handling

.. code-block:: python

    from twin4Build import Optimizer
    
    # Create optimizer
    optimizer = Optimizer()
    
    # Define objective function
    # Set constraints
    # Run optimization
    # Analyze optimal solutions

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

Running as Python Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~

Some examples are also available as Python scripts:

.. code-block:: bash

    python twin4build/examples/translator_example.py
    python twin4build/examples/optimizer_doc.py

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

- **API Documentation**: [Auto-generated API docs](../auto/index.html)
- **Developer Reference**: [Comprehensive developer guide](developer_reference.rst)
- **GitHub Repository**: [Source code and issues](https://github.com/JBjoernskov/Twin4Build/)
- **Online Documentation**: [Read the Docs](https://twin4build.readthedocs.io/)
