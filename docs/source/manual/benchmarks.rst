Benchmarks
==========

The canonical performance suite consists of exactly three authoritative
notebooks in ``benchmarks/``. All scale the complete translated full-workflow
topology over exactly ``1, 10, 50, 100`` zones. The ``n_zones=1`` rows
replace the former standalone simulation, estimation, and optimization
baselines; those notebooks and the separate Pareto notebook are superseded.

.. important::

   **Published results:** No reviewed timing dataset is published yet.
   Smoke output validates structure and plumbing only. Full mode performs five
   repetitions, retains every raw row, and reports median and spread.

Every runner writes an atomic in-progress JSON checkpoint after each case.
Interrupted runs therefore retain prior rows. CUDA timing is synchronized,
component-batching time is separate, and failed, nonconverged, infeasible, unavailable,
or preflight-unsafe rows retain explicit status and reasons.

Local structural and smoke checks
---------------------------------

.. code-block:: console

   pytest twin4build/tests/examples/test_canonical_benchmarks.py
   python -m benchmarks.run_smoke

Do not run or publish full timings as part of routine tests.

Simulation scaling
------------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/benchmarks/simulation_scaling_benchmark.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open simulation scaling benchmark in Colab"/></a></p>

``simulation_scaling_benchmark.ipynb`` records ``model_layout`` (``standard``
or ``batched``), ``execution_mode`` (``object`` or ``functional``), and
``execution_backend`` (``eager`` or ``cuda_graph``). CUDA Graph is a backend
for functional execution, never a mode. Each zone is a complete,
prefixed deep copy of the exact 23-component, 32-connection, 13-state
full-workflow graph. Every size has a batching-mapping audit and
standard/batched numerical parity result.

Estimation scaling
------------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/benchmarks/estimation_scaling_benchmark.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open estimation scaling benchmark in Colab"/></a></p>

``estimation_scaling_benchmark.ipynb`` estimates all 28 theta and uses all
four measurements per zone. Its exact applicability matrix is:

* CPU: SLSQP single shooting and custom batched-SQP;
* CUDA: SLSQP single shooting, custom batched-SQP, and IPOPT collocation.

Collocation is never run on CPU. CUDA collocation explicitly passes
``options={"hessian": "exact"}``. The fixed full scaling
budget is five solver iterations, matching the canonical SLSQP stage in
``full_workflow_example`` and preventing an accidental 100-iteration by
8,400-theta run. Convergence, objective, and parameter-recovery quality remain
visible; fixed-budget nonconvergence is not a successful speedup.

Before constructing collocation problems, preflight records state/NLP
dimensions and dense-Hessian-equivalent bytes. Cases beyond the conservative
safety cap remain in output as skipped rows with dimensions and reasons.

Optimization and Pareto scaling
-------------------------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/benchmarks/optimization_scaling_benchmark.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open optimization and Pareto scaling benchmark in Colab"/></a></p>

``optimization_scaling_benchmark.ipynb`` contains two studies. Standard
scaling uses only constrained SciPy SLSQP+AD on CPU and CUDA with the actual
full-workflow electricity-cost objective and all heating/cooling comfort
limits represented by soft objective penalties (they are not hard nonlinear
solver constraints). It uses the example's 300-iteration budget and labels
convergence, objective quality, maximum comfort-limit violation, and speedup
eligibility.

Pareto scaling runs both supported epsilon-subproblem solvers: SLSQP+AD with
direct shooting, and the IPOPT solver with collocation transcription and an
exact sparse segment-local Lagrangian Hessian. One shared valve schedule is
broadcast to every batched
zone while the flattened augmented ``n_c`` state is promoted at every
boundary. Dynamics continuity is enforced with hard equality defects; comfort
limits remain soft penalties and ``f2_norm <= eps`` is the only non-dynamics
hard inequality. The host solvers remain sequential; fixed-shape device
derivative evaluation is captured and replayed. Preflight records controls,
boundary states, dynamics rows, the epsilon row, and sparse Jacobian/Hessian
nonzero counts.

Reproducing and reporting
-------------------------

Use full mode only in an intentional measurement session. Record commit,
Python, Twin4Build, Torch, CUDA runtime, CPU/GPU, OS, precision, horizon,
timestep, solver options, and all raw rows. Compare only equivalent numerical
work and successful quality outcomes. Component batching, fallback, preflight skips,
and solver failures are part of the record and must not be hidden.
