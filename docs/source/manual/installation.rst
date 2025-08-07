Installation Guide
=================

This guide covers how to install Twin4Build for both users and developers.

For Users
---------

Install from PyPI
~~~~~~~~~~~~~~~~~

The easiest way to install Twin4Build is using pip:

.. code-block:: bash

    pip install twin4build

Install with Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For additional functionality, you can install optional dependencies:

.. code-block:: bash

    # For database connectivity
    pip install twin4build[database]
    
    # For all optional dependencies
    pip install twin4build[all]

For Developers
--------------

Install from Source
~~~~~~~~~~~~~~~~~~~

1. **Clone the repository**:
   .. code-block:: bash

       git clone https://github.com/JBjoernskov/Twin4Build.git
       cd Twin4Build

2. **Create a virtual environment**:
   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install in development mode**:
   .. code-block:: bash

       pip install -e .

4. **Install development dependencies**:
   .. code-block:: bash

       pip install -e .[dev]

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux

Python Dependencies
~~~~~~~~~~~~~~~~~~~

Core dependencies (automatically installed):
- matplotlib
- seaborn
- pandas
- torch
- pydot
- tqdm
- fmpy
- scipy
- numpy
- prettytable
- jupyter
- nbformat
- rdflib
- brickschema
- click_spinner
- xlrd
- rdflib_sqlalchemy
- pydotplus
- typer
- openpyxl
- psycopg2
- pyarrow
- beautifulsoup4
- python-dateutil
- pathlib

Development dependencies:
- coverage
- black
- flake8
- mypy
- sphinx
- sphinx-rtd-theme

Verifying Installation
---------------------

To ensure everything is working properly, run the test suite:

.. code-block:: bash

    python -m unittest discover twin4build/tests/ -v

For developers, you can also run tests with coverage:

.. code-block:: bash

    coverage run -m unittest discover twin4build/tests/
    coverage report

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**Import Error: No module named 'twin4build'**
- Ensure you're in the correct virtual environment
- Verify the package was installed correctly: `pip list | grep twin4build`

**Database Connection Issues**
- Ensure PostgreSQL is installed and running
- Check database configuration in `database_config_example.ini`

**Test Failures**
- Ensure all dependencies are installed: `pip install -e .[dev]`
- Check that the virtual environment is activated
- Verify Python version compatibility (3.9+)

Getting Help
-----------

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/JBjoernskov/Twin4Build/issues) for similar problems
2. Review the [troubleshooting section](developer_reference.rst#debugging-tips) in the developer reference
3. Create a new issue with detailed error information

For more detailed development setup, see the [Developer Reference](developer_reference.rst).




.. .. include:: ../../../README.md
..    :parser: myst_parser.sphinx_
..    :start-after: ## Installation
..    :end-before: ## Publications