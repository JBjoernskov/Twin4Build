Examples and Tutorials
=====================

.. .. include:: ../../../README.md
..    :parser: myst_parser.sphinx_
..    :start-after: ## Examples and Tutorials
..    :end-before: ## Documentation


This guide provides an overview of the available examples and tutorials for Twin4Build.

Getting Started Examples
-----------------------

Basic Twin4Build Usage
~~~~~~~~~~~~~~~~~~~~~~

**Minimal Example** (`minimal_example.ipynb`)
- **Purpose**: Introduction to basic Twin4Build concepts
- **Topics**: Connecting components, simulating a model, and visualization
- **Difficulty**: Beginner
- **Duration**: 15-20 minutes

.. code-block:: python

    import twin4build as tb
    import twin4build.utils.plot.plot as plot
    
    # Create a model
    model = tb.Model(id="example_model")
    
    # Add components and connections
    # Run simulation
    # Visualize results

**Space CO2 Controller Example** (`space_co2_controller_example.ipynb`)
- **Purpose**: Modeling and control of indoor CO2 concentration
- **Topics**: Control systems, sensor integration, feedback loops
- **Difficulty**: Intermediate
- **Duration**: 30-45 minutes

Core Component Examples
----------------------

Translator Examples
~~~~~~~~~~~~~~~~~~

**Translator Example** (`translator_example.ipynb`)
- **Purpose**: Generate simulation models from semantic models
- **Topics**: Ontology-driven modeling, automated model creation
- **Difficulty**: Intermediate
- **Duration**: 45-60 minutes

.. code-block:: python

    from twin4build import Translator
    
    # Create translator
    translator = Translator()
    
    # Load semantic model
    # Generate simulation model
    # Validate translation

Estimator Examples
~~~~~~~~~~~~~~~~~

**Estimator Example** (`estimator_example.ipynb`)
- **Purpose**: Parameter estimation and model calibration
- **Topics**: Least-squares optimization, PyTorch-based calibration
- **Difficulty**: Advanced
- **Duration**: 60-90 minutes

.. code-block:: python

    from twin4build import Estimator
    
    # Create estimator
    estimator = Estimator()
    
    # Load measured data
    # Define parameters to estimate
    # Run calibration
    # Analyze results

Optimizer Examples
~~~~~~~~~~~~~~~~~

**Optimizer Example** (`optimizer_example.ipynb`)
- **Purpose**: Optimization of building operation
- **Topics**: Gradient-based optimization, constraint handling
- **Difficulty**: Advanced
- **Duration**: 60-90 minutes

.. code-block:: python

    from twin4Build import Optimizer
    
    # Create optimizer
    optimizer = Optimizer()
    
    # Define objective function
    # Set constraints
    # Run optimization
    # Analyze optimal solutions

Advanced Examples
----------------

Building System Examples
~~~~~~~~~~~~~~~~~~~~~~~

**Building Space Example** (`building_space_example.ipynb`)
- **Purpose**: Complete building space modeling
- **Topics**: Multi-zone buildings, HVAC systems, thermal dynamics
- **Difficulty**: Advanced
- **Duration**: 90-120 minutes

**Space Heater Example** (`space_heater_example.ipynb`)
- **Purpose**: Heating system modeling and control
- **Topics**: Thermal systems, control strategies, energy optimization
- **Difficulty**: Intermediate
- **Duration**: 45-60 minutes

Neural Policy Controller
~~~~~~~~~~~~~~~~~~~~~~~

**Neural Policy Controller Example** (`neural_policy_controller_example/`)
- **Purpose**: Machine learning-based control strategies
- **Topics**: Neural networks, reinforcement learning, adaptive control
- **Difficulty**: Expert
- **Duration**: 120+ minutes

Running Examples
---------------

Prerequisites
~~~~~~~~~~~~

Before running examples, ensure you have:

1. **Twin4Build installed**: See [Installation Guide](installation.rst)
2. **Jupyter Notebook**: `pip install jupyter`
3. **Required data files**: Some examples require specific data files
4. **Sufficient computational resources**: Advanced examples may require more memory

Running in Jupyter
~~~~~~~~~~~~~~~~~

1. **Start Jupyter**:
   .. code-block:: bash

       jupyter notebook

2. **Navigate** to the examples directory:
   .. code-block:: bash

       cd twin4build/examples

3. **Open** the desired notebook and run cells sequentially

Running as Python Scripts
~~~~~~~~~~~~~~~~~~~~~~~~

Some examples are also available as Python scripts:

.. code-block:: bash

    python twin4build/examples/translator_example.py
    python twin4build/examples/optimizer_doc.py

Example Structure
----------------

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
   - Manage computational resources

4. **Results and Visualization**
   - Plot results
   - Generate reports
   - Export data

5. **Analysis and Discussion**
   - Interpret results
   - Compare with expectations
   - Suggest improvements

Customizing Examples
-------------------

Modifying Parameters
~~~~~~~~~~~~~~~~~~~

Most examples use configurable parameters that you can modify:

.. code-block:: python

    # Example: Modify simulation duration
    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 1, 7)  # Change to 7 days
    
    # Example: Adjust component parameters
    space_heater.nominalPower = 5000  # Change from default

Adding New Components
~~~~~~~~~~~~~~~~~~~~

Examples can be extended with custom components:

.. code-block:: python

    class CustomComponent(tb.System):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Add custom initialization
            
        def do_step(self, secondTime, dateTime, stepSize):
            # Implement custom behavior
            pass

Troubleshooting Examples
-----------------------

Common Issues
~~~~~~~~~~~~

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

**Performance Issues**
- Use smaller time steps
- Optimize component connections
- Consider parallel processing

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

Creating Your Own Examples
-------------------------

Guidelines for creating new examples:

1. **Start with a clear purpose**: Define what the example demonstrates
2. **Keep it focused**: Cover one main concept or workflow
3. **Include documentation**: Add comments and explanations
4. **Test thoroughly**: Ensure the example runs without errors
5. **Follow naming conventions**: Use descriptive file names
6. **Add to documentation**: Update this guide when adding new examples

Example template:

.. code-block:: python

    """
    Example Title
    
    Description: Brief description of what this example demonstrates
    
    Topics: List of main topics covered
    
    Difficulty: Beginner/Intermediate/Advanced/Expert
    
    Duration: Estimated time to complete
    """
    
    import twin4build as tb
    # Additional imports
    
    def main():
        """Main function demonstrating the example."""
        # Setup
        # Implementation
        # Results
        # Analysis
    
    if __name__ == "__main__":
        main()
