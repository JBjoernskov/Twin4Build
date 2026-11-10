Settings
========

The library's knobs are in one place, the module ``twin4build.settings``. Each
setting takes its value from, first found:

1. ``tb.settings.set(...)`` (or ``tb.settings.override(...)``) in this process;
2. its environment variable;
3. its default.

.. code-block:: python

   import twin4build as tb

   tb.settings.value("zoh_expm")  # 'phi1'
   tb.settings.set(zoh_expm="block")  # for the rest of this process
   with tb.settings.override(jac_check=True):  # for this block only
       result = estimator.estimate(...)
   tb.settings.reset("zoh_expm")  # the environment variable's or the default
   tb.settings.snapshot()  # every setting with its value and its source

The library reads a setting where it uses it, never when a module is imported,
so a value given after ``import twin4build`` holds from its next use. A step that
``torch.compile`` or a CUDA graph has already traced may keep the values it was
traced with (``zoh_expm`` and ``expm_checkpoint`` are read inside the step):
give those settings their values before the run.

A boolean variable takes ``1``, ``true``, ``yes`` or ``on``, or ``0``,
``false``, ``no`` or ``off``, in any case. An empty variable counts as unset;
any other text is an error that names the variable. ``set`` and ``override``
check their values the same way, and an unknown name is an error.

The settings that are off their defaults are logged, with where each value came
from, when an estimation, a simulation, an optimization or a Pareto front
starts. ``tb.settings.snapshot()`` gives every setting as
``{name: {"value": ..., "source": ...}}``, the source being ``"set"``, the
environment variable's name or ``"default"``. The results record how they were
computed: the result of ``Estimator.estimate`` (and the file it is saved to)
and of ``Optimizer.optimize`` carry the snapshot as ``result["settings"]``
(``result.settings``), and a front from ``Optimizer.pareto_front`` as
``front.settings``.

.. code-block:: python

   result = estimator.estimate(...)
   result.settings["zoh_expm"]  # {'value': 'phi1', 'source': 'default'}

Numerics
--------

``zoh_expm``
   ``T4B_ZOH_EXPM`` -- ``"phi1"`` (the default) or ``"block"``.

   How a state-space step is discretised in transform mode (compiled, captured,
   vmapped). ``phi1`` exponentiates the n x n matrix ``A T`` and carries
   ``phi1(A T) = sum (A T)^k / (k+1)!`` through the same scaling and squaring,
   ``Bd = phi1(A T) B T``. ``block`` exponentiates the (n + m) x (n + m) block
   ``[[A T, B T], [0, 0]]``, a matrix as wide as the inputs are many (a zone of
   6 states and 11 inputs: 17 x 17), whose saved products (about 26 per
   component per step) are the tape of an unrolled rollout. Both are the same
   zero-order-hold step; their results differ at rounding level.

Memory
------

``expm_checkpoint``
   ``T4B_EXPM_CHECKPOINT`` -- boolean, off by default.

   Recompute each matrix exponential in the backward pass instead of saving its
   chain (``torch.utils.checkpoint``): one matrix per component per step stays
   on the autograd tape, for one more forward of every exponential. For a fit
   that runs out of memory in the backward pass of large state-space blocks.

``rollout_checkpoint``
   ``T4B_ROLLOUT_CHECKPOINT`` -- boolean, off by default.

   Recompute every time step of a functional rollout in the backward pass
   instead of saving its intermediates: the tape holds the state and exogenous
   rows per step and nothing the step computed from them, and the backward pass
   runs every step's forward again. For a long rollout whose tape does not fit
   in memory.

Runs
----

``testing``
   ``TWIN4BUILD_TESTING`` -- boolean, off by default.

   Estimators and optimizers stop after one iteration (``maxiter=1``, in every
   phase of a schedule too) and a Pareto front computes at most three points.
   ``twin4build.utils.test_notebook`` turns it on in the kernels it starts, so
   that every cell of the example notebooks runs, quickly. The results are not
   converged.

Diagnostics
-----------

``jac_check``
   ``TWIN4BUILD_JAC_CHECK`` -- boolean, off by default.

   Collocation compares the composed constraint Jacobian with finite
   differences on a few columns and logs each column's error and the worst.
   For a collocation fit that does not converge.

``bench_timestep``
   ``TWIN4BUILD_BENCH_TIMESTEP`` -- boolean, off by default.

   Collocation times one step of every segment both ways, the functional map
   and the object-graph simulation, and logs the times and their agreement.

``match_diag_file``
   ``TWIN4BUILD_MATCH_DIAG_FILE`` -- a file name, unset by default.

   The translator writes its pattern-matching trace (per-node walker decisions,
   merge attempts) to this file. For a pattern that does not match where it
   should.

``match_diag_pattern``
   ``TWIN4BUILD_MATCH_DIAG_PATTERN`` -- text, unset by default.

   With ``match_diag_file``: trace only the patterns whose id contains this
   text, which keeps the file small.

``cuda_graph_sync_phases``
   ``T4B_CUDA_GRAPH_SYNC_PHASES`` -- boolean, off by default.

   A CUDA-graph capture synchronizes the device after every phase (warmup,
   record, replay, parity check), so that an asynchronous CUDA fault is
   reported in the phase that launched it. Costs one synchronization per phase.

``logger_strict``
   ``TWIN4BUILD_LOGGER_STRICT`` -- boolean, off by default.

   The logger raises when a status change names no logged line, instead of
   ignoring it. For finding a mismatched ``LOGGER.task`` and
   ``LOGGER.ok(..., change_status=True)`` pair.
