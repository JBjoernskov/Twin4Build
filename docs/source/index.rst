.. Twin4Build documentation master file, created by
   sphinx-quickstart on Tue Oct 29 13:01:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Twin4Build documentation
========================

Overview
--------

This documentation is organized into three main sections:

**Getting Started**
   Contains tutorials and installation instructions to help you begin using Twin4Build:
    
   * Installation - Instructions for installing Twin4Build and its dependencies
   * Examples and Tutorials - Step-by-step guides showing basic usage
   * Benchmarks - Reproducible performance methodology and canonical notebooks

**API Reference**
   Detailed documentation of all Twin4Build modules and their components.

**Developer Reference**
   Guide for developers who want to contribute to Twin4Build.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   manual/installation
   manual/examples_and_tutorials
   manual/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   auto/twin4build

.. toctree::
   :maxdepth: 2
   :caption: Developer Reference
   :hidden:

   manual/developer_reference
   manual/differentiable_system_models
   


Core workflow
-------------

Twin4Build has one primary workflow:

1. Construct a :class:`~twin4build.model.model.Model` directly, or translate a
   :class:`~twin4build.model.semantic_model.semantic_model.SemanticModel`.
2. Call ``model.load()`` and, when needed, ``model.to(device, dtype)``.
3. Create a :class:`~twin4build.simulator.simulator.Simulator` and run it.
4. Pass that simulator to an
   :class:`~twin4build.estimator.estimator.Estimator` or
   :class:`~twin4build.optimizer.optimizer.Optimizer`.

Backend-specific solver settings belong in the ``options`` argument of
``estimate`` or ``optimize``. Execution policy belongs to ``Simulator``.

See the `project README <https://github.com/JBjoernskov/Twin4Build/>`_ for
installation badges, publications, and citation information.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`