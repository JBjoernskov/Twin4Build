Developer reference
===================

This guide describes the Twin4Build 2.0 source tree, development workflow,
public execution model, documentation build, and release process. For the
tensor execution contract, also read :doc:`differentiable_system_models`.

Architecture
------------

The preferred public workflow is:

1. Construct a :class:`~twin4build.model.model.Model`, or translate a
   :class:`~twin4build.model.semantic_model.semantic_model.SemanticModel`.
2. Call :meth:`~twin4build.model.model.Model.load`.
3. Optionally move the model with
   :meth:`~twin4build.model.model.Model.to`.
4. Create a :class:`~twin4build.simulator.simulator.Simulator` and call
   :meth:`~twin4build.simulator.simulator.Simulator.simulate`.
5. Pass that simulator to an
   :class:`~twin4build.estimator.estimator.Estimator` or
   :class:`~twin4build.optimizer.optimizer.Optimizer`.

Execution mode and backend belong to ``Simulator``. Model layout is configured
separately with ``Model.batch_components()``. Solver-specific settings belong
in the ``options`` argument of ``estimate`` or ``optimize``.

Repository layout
~~~~~~~~~~~~~~~~~

::

   Twin4Build/
   ├── twin4build/
   │   ├── core/                 # Stable convenience imports and ontologies
   │   ├── model/                # Model, SimulationModel, and SemanticModel
   │   ├── simulator/            # Object and functional execution
   │   ├── estimator/            # Calibration and transcription backends
   │   ├── optimizer/            # Optimization and Pareto-front support
   │   ├── translator/           # Semantic-to-simulation translation
   │   ├── systems/              # Reusable component models
   │   ├── utils/                # Ports, results, logging, plotting, and helpers
   │   ├── examples/             # Tutorials and their data
   │   └── tests/                # Unit and integration tests
   ├── benchmarks/               # Canonical benchmark notebooks and shared code
   ├── docs/
   │   ├── source/manual/        # Hand-written documentation
   │   └── source/auto/          # Generated API pages
   ├── scripts/                  # Development setup and validation
   ├── .github/workflows/        # CI and tagged PyPI publication
   └── pyproject.toml            # Package metadata, version, and tool settings

Do not treat ``generated_files/``, test fixtures, examples, or benchmark
helpers as public API. Generated results must not be added to API navigation.

Development setup
-----------------

- Python 3.10 or higher (3.12 recommended)
- Git
- A code editor (VS Code, PyCharm, etc.)
- **Conda** (recommended) or any Python environment manager

Graph drawing uses the pygraphviz 2.0 wheel (no system Graphviz install).

**Quick Start**: Use the automated setup script ``python scripts/setup_dev.py`` after cloning the repository for the fastest setup experience.

From the repository root:

.. code-block:: console

   python scripts/setup_dev.py
   python scripts/setup_dev.py --python 3.12 --env t4bdev

The script creates a Conda environment, installs ``.[dev]``, and runs the
discovered unittest suite. Use ``--help`` for its current options. It requires
Conda; it is not a generic virtual-environment bootstrapper.

    # Clone the repository
    git clone https://github.com/JBjoernskov/Twin4Build.git
    cd Twin4Build

    # Run the automated setup script
    python scripts/setup_dev.py

    # Or with custom options
    python scripts/setup_dev.py --python 3.12 --env t4bdev

**What the setup script does:**

- Creates a conda environment with your specified Python version (default: 3.12)
- Installs Twin4Build in development mode with all dependencies
- Runs the test suite to verify installation
- Provides clear next steps and available tools

**Script options:**

- ``--python VERSION``: Specify Python version (e.g., 3.10, 3.11, 3.12)
- ``--env NAME``: Specify conda environment name (default: t4bdev)
- ``--help``: Show all available options

**Manual Setup (Alternative)**

If you prefer to set up manually or need a different environment manager:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/JBjoernskov/Twin4Build.git
    cd Twin4Build

    # Create conda environment
    conda create -n t4bdev python=3.12
    conda activate t4bdev

    # Install in development mode with dependencies
    pip install -e .[dev]

**Alternative environment managers**: You can also use venv, virtualenv, poetry, or pipenv - just ensure you have an isolated Python 3.10+ environment.

Code Style and Conventions
--------------------------

Python Style Guide
~~~~~~~~~~~~~~~~~~

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Keep line length under 88 characters (Black formatter default)
- Use meaningful variable and function names

Naming Conventions
~~~~~~~~~~~~~~~~~~

- **Classes**: PascalCase (e.g., `Model`, `SpaceHeaterSystem`)
- **Functions and variables**: snake_case (e.g., `run_simulation`, `temperature_data`)
- **Module-level constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TIMESTEP`)
- **Private methods**: prefix with underscore (e.g., `_internal_calculation`)
- **Private attributes**: prefix with underscore (e.g., `_components`)
- **Keys used in System.input and System.output dictionaries**: camelCase (e.g., `indoorTemperature`, `co2Concentration`)

Docstring Standards
~~~~~~~~~~~~~~~~~~~

Use Google-style docstrings and type hints:

.. code-block:: python

    def calculate_energy_consumption(self, temperature: float, duration: float) -> float:
        """Calculate energy consumption for a given temperature and duration.
        
        Args:
            temperature: The target temperature in Celsius
            duration: The duration in hours
            
        Returns:
            Energy consumption in kWh
            
        Raises:
            ValueError: If temperature is outside valid range
        """
        pass

For public class properties (acessed from outside the class), use the @property decorator:

.. code-block:: python

    class MyClass:
        @property
        def property_name(self) -> Any:
            """Description of the property."""
            return self._property_name
    
Avoid defining setter methods for public class properties unless necessary.
This way, we avoid accidently changing the value of a property.
If necessary, define a setter method for the property.

.. code-block:: python

    class MyClass:
        @property_name.setter
        def property_name(self, value: Any) -> None:
            """Description of the property."""
            self._property_name = value

Development Workflow
--------------------

Branching Strategy
~~~~~~~~~~~~~~~~~~

Twin4Build follows a disciplined branching model to keep development organized and reversible:

- **Main branch**: Stable releases only, updated through approved merges from dev branch
- **Dev branch**: Integration branch for completed features with tests and documentation
- **Feature branches**: Each feature lives in its own branch containing only logically related changes

  - Feature branches may have one level of sub-branching when needed
  - Once complete, features merge into dev (never directly into main)
  - Main contributors can make exemptions to this rule

Git Workflow
~~~~~~~~~~~~

.. code-block:: console

   conda create -n t4bdev python=3.12
   conda activate t4bdev
   python -m pip install -e ".[dev]"

With ``venv``, replace the first two commands with the platform-appropriate
environment creation and activation commands. Keep the editable install so
source changes are imported immediately.

Style and compatibility
-----------------------

Follow PEP 8, use type hints on public interfaces, and format Python with the
repository's Black and isort settings. Public Python identifiers use
``snake_case`` for parameters and methods and ``PascalCase`` for classes.

Port names are model schema keys and many remain camelCase, for example
``indoorTemperature``. That does not make camelCase Python parameters
preferred. Use ``start_time``, ``end_time``, ``step_size``,
``weekday_ruleset``, and other snake_case parameters.

Use the 2.0 system class names such as ``BuildingSpaceSystem``,
``DamperSystem``, ``WallSystem``, and ``FmuSystem``. Names ending in
``TorchSystem`` are migration aliases scheduled for removal and must not be
introduced in new examples or documentation.

Docstrings
~~~~~~~~~~

Public APIs use type hints and Google-style docstrings, which Sphinx parses
through Napoleon:

.. code-block:: python

   def energy_used(power: float, duration: float) -> float:
       """Return energy use.

       Args:
           power: Power in kW.
           duration: Duration in hours.

       Returns:
           Energy in kWh.

       Raises:
           ValueError: If duration is negative.
       """
       if duration < 0:
           raise ValueError("duration must be non-negative")
       return power * duration

Document user-visible behavior, units, tensor shapes, and exceptions. Avoid
restating implementation details that can change without affecting the API.

Model and simulator usage
-------------------------

Construct components and connect sender output ports to receiver input ports:

.. code-block:: python

   import datetime as dt
   from dateutil import tz
   import twin4build as tb

   model = tb.Model(id="developer_example")
   source = tb.ScheduleSystem(
       weekday_ruleset={"ruleset_default_value": 0.5},
       id="source",
   )
   damper = tb.DamperSystem(id="damper")
   model.add_connection(
       source,
       damper,
       output_port="scheduleValue",
       input_port="damperPosition",
   )
   model.load(
       draw_semantic_model=False,
       draw_simulation_model=False,
   )

   start_time = dt.datetime(2025, 1, 1, tzinfo=tz.UTC)
   simulator = tb.Simulator(
       model,
       execution_mode="object",
       execution_backend="eager",
   )
   simulator.simulate(
       start_time=start_time,
       end_time=start_time + dt.timedelta(hours=1),
       step_size=600,
       show_progress_bar=False,
   )
   values = damper.output["airFlowRate"].history()

``Model.add_connection`` adds both components when necessary. Call
``Model.add_component`` explicitly for an unconnected component. ``Model``
does not have a ``simulate`` method; simulation is owned by ``Simulator`` and
results remain on component port histories.

Use timezone-aware datetimes. A single period accepts scalar datetime and
integer arguments; batched periods accept equally sized lists. The interval is
half-open: timesteps begin at ``start_time`` and stop before ``end_time``.

Semantic translation
~~~~~~~~~~~~~~~~~~~~

Translate explicitly in new code:

.. code-block:: python

   semantic_model = tb.SemanticModel(rdf_file="building.ttl", id="building")
   model = tb.Translator().translate(semantic_model)
   model.load()

Passing ``semantic_model_filename`` to ``Model.load`` is deprecated. Restoring
a serialized simulation model uses ``model.load(filename=...)``.

Model layout, execution, and devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are independent dimensions:

* ``model_layout="standard"`` is the ordinary model returned by ``Model.load``;
  ``model_layout="batched"`` is produced by ``model.batch_components()``.
* ``execution_mode="object"`` is the general port/history engine;
  ``execution_mode="functional"`` executes the reusable functional model.
* ``execution_backend="eager"`` is the default. The CUDA-only
  ``execution_backend="cuda_graph"`` captures and replays fixed-shape
  functional execution.

CUDA Graph is a backend, not a mode. Inspect batching with
``get_batched_component_info`` and ``get_batch_id_for_component``. Functional
tooling is also available directly through ``build_functional_model``,
``record_exogenous_inputs``, and ``rollout_functional``.

Move a loaded model with ``model.to(device, dtype)``. The default precision is
float64. Device and precision materially affect numerical results and
performance and must be recorded in experiments.

Testing and validation
----------------------

The CI test command is:

.. code-block:: console

   python -m unittest discover twin4build/tests/ -v

Run one importable module:

.. code-block:: console

   python -m unittest twin4build.tests.simulator.test_simulator -v

Pytest is included in ``.[dev]`` and can run the same suite:

.. code-block:: console

   python -m pytest twin4build/tests/ -v
   python -m pytest twin4build/tests/systems/junction/test_junction_systems.py
   python -m pytest twin4build/tests/ --cov=twin4build --cov-report=term-missing

For formatting and static checks:

.. code-block:: console

   python scripts/validate_code.py
   python scripts/validate_code.py --fix
   python scripts/validate_code.py --test

The validation script runs Black, isort, flake8, file checks, and optionally
tests. Review auto-fixes before committing.

Tests should use ``test_*.py`` files and ``unittest.TestCase`` classes unless a
focused test has a reason to use another supported pytest idiom. Public
behavior needs unit coverage; interactions among Model, Simulator, Estimator,
Optimizer, and Translator need integration coverage.

Writing components
------------------

Custom components inherit :class:`~twin4build.systems.saref4syst.system.System`.
Declare :class:`~twin4build.utils.types.Scalar` or
:class:`~twin4build.utils.types.Vector` ports, initialize their histories, and
implement ``do_step``.

.. code-block:: python

   import torch
   import twin4build as tb

   class GainSystem(tb.System):
       def __init__(self, gain: float = 1.0, **kwargs):
           super().__init__(**kwargs)
           self.gain = gain
           self.input = {"value": tb.Scalar()}
           self.output = {"value": tb.Scalar()}

       def initialize(self, start_time, end_time, step_size):
           _, _, n_t, _ = tb.Simulator.get_simulation_timesteps(
               start_time, end_time, step_size
           )
           n_s = len(start_time)
           self.input["value"].initialize(n_t=n_t, n_s=n_s)
           self.output["value"].initialize(n_t=n_t, n_s=n_s)

       def forward(self, state, inputs, parameters, **kwargs):
           return state, {"value": inputs["value"] * self.gain}

       def do_step(self, second_time, date_time, step_size, step_index):
           _, outputs = self.forward(
               None,
               {"value": self.input["value"].get()},
               {},
           )
           self.output["value"]._set(outputs["value"], i_t=step_index)

The exact ``forward`` signature and state semantics depend on the component
family. Before adding functional support, follow
:doc:`differentiable_system_models` and add value, Jacobian, Hessian, replay,
device, and dtype tests as applicable. ``do_step`` must delegate its
mathematics to the pure implementation so execution paths cannot drift.

Documentation
-------------

Hand-written pages live in ``docs/source/manual/``. API pages in
``docs/source/auto/`` are generated and should not be edited by hand.

From ``docs/`` on Linux or macOS:

.. code-block:: console

   make buildapi
   make html

From ``docs\`` on Windows:

.. code-block:: doscon

   make.bat buildapi
   make.bat html

``buildapi`` replaces ``docs/source/auto/`` and excludes tests, examples,
generated modules, benchmark code, and private implementation modules.
``html`` writes local output to ``docs/build/html/``. Open
``docs/build/html/index.html`` to inspect it. On Read the Docs,
``READTHEDOCS_OUTPUT`` controls the output directory.

For a clean local rebuild, remove ``docs/build/`` and
``docs/source/auto/``, then run both commands again. Do not remove hand-written
manual pages.

Notebook links
~~~~~~~~~~~~~~

Example and benchmark badges contain ``GITHUB_NOTEBOOK_BRANCH`` in source.
``docs/source/conf.py`` substitutes a resolvable Git ref:

* ``latest`` uses ``main``;
* named Read the Docs branch and tag builds use that branch or tag;
* external pull-request previews use the commit SHA;
* local builds use the checked-out branch, falling back to ``dev``.

Canonical performance notebooks live in ``benchmarks/``; tutorials live in
``twin4build/examples/``. See :doc:`benchmarks` and
:doc:`examples_and_tutorials`.

API generation policy
~~~~~~~~~~~~~~~~~~~~~

The API reference documents supported package modules. Exclude:

* ``twin4build.tests`` and generated test fixtures;
* ``twin4build.examples``;
* ``twin4build.generated_files``;
* repository-level ``benchmarks``;
* private modules whose names begin with an underscore.

When adding a public package, regenerate the API pages and verify that it is
reachable from ``auto/twin4build``. Do not solve unwanted API pages only with
navigation hiding; prevent their generation.

Contribution workflow
---------------------

Create focused branches from the repository's current integration branch,
link work to an issue when required by the maintainers, and use descriptive
imperative commit messages. Before requesting review:

* run relevant tests and static checks;
* update public API documentation and migration notes;
* add architecture notes for changes with non-obvious boundaries;
* ensure examples use preferred 2.0 names and parameters;
* avoid committing generated runtime data.

Changes to ``main`` and ``dev`` are reviewed through pull requests. Confirm the
current branch policy with maintainers rather than relying on a hard-coded
exception in this guide.

Release process
---------------

``pyproject.toml`` is the source of truth for the package version. Twin4Build
uses semantic versioning: increment MAJOR for incompatible API changes, MINOR
for backward-compatible features, and PATCH for backward-compatible fixes.

For a release:

1. Set ``[project].version`` in ``pyproject.toml``.
2. Add release notes to ``CHANGELOG.md`` and verify migration guidance.
3. Run the full test matrix locally where practical.
4. Regenerate and build the documentation with warnings treated as errors.
5. Build and validate distributions:

   .. code-block:: console

      python -m build
      python -m twine check dist/*

6. Merge the reviewed release changes.
7. Create and push an annotated ``vMAJOR.MINOR.PATCH`` tag that exactly matches
   ``pyproject.toml``.

The ``Publish to PyPI`` workflow builds and publishes only pushed tags whose
names start with ``v`` and only for the configured maintainer actor. PyPI uses
trusted publishing; contributors should not run ``twine upload`` as part of
the normal release path. Branch merges alone do not publish a release, and
the repository does not automatically bump the version on every ``dev``
merge.

Getting help
------------

Use `GitHub Issues <https://github.com/JBjoernskov/Twin4Build/issues>`_ for
reproducible bug reports and feature discussions. Include the Twin4Build and
Python versions, operating system, device and dtype where relevant, a minimal
example, and the complete error message.
