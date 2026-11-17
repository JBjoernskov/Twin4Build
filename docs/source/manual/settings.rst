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
traced with (``zoh_expm``, ``small_matmul_max`` and ``expm_checkpoint`` are
read inside the step, ``combo_kernels`` when it is compiled): give those
settings their values before the run.

A boolean variable takes ``1``, ``true``, ``yes`` or ``on``, or ``0``,
``false``, ``no`` or ``off``, in any case; a whole-number variable takes a
whole number. An empty variable counts as unset; any other text is an error
that names the variable. ``set`` and ``override``
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
   ``T4B_ZOH_EXPM`` -- ``"phi1"`` (the default), ``"block"`` or ``"aug"``.

   How a state-space step is discretised in transform mode (compiled, captured,
   vmapped). ``phi1`` exponentiates the n x n matrix ``A T`` and carries
   ``phi1(A T) = sum (A T)^k / (k+1)!`` through the same scaling and squaring,
   ``Bd = phi1(A T) B T``. ``block`` exponentiates the (n + m) x (n + m) block
   ``[[A T, B T], [0, 0]]``, a matrix as wide as the inputs are many (a zone of
   6 states and 11 inputs: 17 x 17), whose saved products (four for the series
   and one per doubling, ``expm_squarings``, per component per step) are the
   tape of an unrolled rollout. ``aug`` needs the
   next state only: it folds the input into a vector, ``b = B T u``, and reads
   ``x_next = exp(A T) x + phi1(A T) b`` off one exponential of the (n + 1) x
   (n + 1) matrix ``[[A T, b], [0, 0]]`` -- about half the products of
   ``phi1`` and no n x m product, and the most accurate of the three on stiff
   blocks (``|A T|`` around 1e5: about 1e-15 of the state against
   ``torch.matrix_exp``, ``phi1`` about 1e-6). A model with large fused blocks
   steps faster with it (a 159-state block: 5.8 against 11.8 ms forward and
   backward, 16 windows, float32, on a laptop GPU). ``aug`` changes the step
   only; ``discretize_onestep``, which returns the discrete matrices, uses
   ``block`` with it. All three are the same zero-order-hold step; their
   results differ at rounding level.

``expm_squarings``
   ``T4B_EXPM_SQUARINGS`` -- a whole number, 18 by default.

   The doublings of a state-space step's matrix exponential in transform mode.
   The exponential is computed by scaling and squaring: the matrix is balanced
   (a diagonal similarity that brings its norm towards its spectral radius),
   divided by ``2^expm_squarings``, exponentiated by a degree-8 series, and
   squared ``expm_squarings`` times. The count is the same for every block and
   every step, since a traced step cannot choose it from the data, so it must
   cover the stiffest block a run reaches: the balanced matrix's 1-norm over
   ``2^expm_squarings`` should stay below about 0.5 for float32 accuracy and
   0.05 for float64. Each doubling is one matrix product per block per step,
   and three in a reverse-mode gradient, so a fit on a model with large blocks
   pays for every doubling it does not need (a 214-state block, float32, 8 per
   call on a laptop GPU: 18 doublings 7.2 ms forward and backward, 10
   doublings 3.8 ms). 18 covers norms up to about 1e5 without balancing. A
   matrix beyond the series' range (scaled norm above 2) comes out as NaN, never
   as a wrong finite value: raise the count.

``small_matmul_max``
   ``T4B_SMALL_MATMUL_MAX`` -- a whole number, 16 by default.

   The largest matrix whose products are written as a broadcast multiply and
   sum while ``torch.compile`` traces a step: Inductor fuses them with the
   surrounding arithmetic into few kernels, where a library call ends the
   kernel around it. It holds for the exponential's own products and for
   every product of the step whose left factor has at most this many rows and
   columns (a small system's next state and outputs, a fused block's rows): a
   step of many small systems is otherwise mostly launches. A larger matrix
   uses ``bmm``, since the broadcast form moves n times a product's memory and
   pays only while the work per call is small. The crossover depends on the
   matrices per call (blocks times shooting windows) more than on the GPU or
   the dtype: measured on a laptop GPU, 960 matrices per call cross between
   13 x 13 and 17 x 17, in float32 and float64; 16 per call between 33 x 33
   and 65 x 65. 16 suits batched models (hundreds of matrices per call); runs
   that exponentiate a few blocks at a time may gain about 1 ms per step with
   32. A large block (a fused storey) must stay above it. The setting has no
   effect outside ``torch.compile``. In a source checkout,
   ``python benchmarks/small_matmul_crossover.py --matrices <blocks x windows>``
   measures the crossover on a GPU (``benchmarks/README.md``).

``combo_kernels``
   ``T4B_COMBO_KERNELS`` -- a boolean, off by default.

   Compile a step with Inductor's combo kernels: kernels that do not depend on
   each other are combined into one launch. A step of many small systems is
   thousands of kernels of a few numbers each, and a launch costs more than
   its arithmetic. Measured on a building of a few hundred rooms, float32,
   together with the small products above: about 11,600 kernel launches and
   2,700 library calls per step became about 1,600 and 190, and a week's
   simulation took a third of the time. It lengthens the compile, and it is an experimental
   Inductor option: compare a compiled run with the eager one before relying
   on it. Read when a step is compiled; no effect on a CPU.

``fused_sparse``
   ``T4B_FUSED_SPARSE`` -- boolean, on by default.

   A fused block (connected state-space components joined into one, see
   ``FusedStateSpaceSystem``) holds its bilinear terms, state x input and
   input x input, as their entries: a few thousand of the tens of millions a
   dense ``(n_c, m, n, n)`` or ``(n_c, m, n, m)`` tensor has for a storey-wide
   block (about 158 states and 588 inputs), whose broadcast over the inputs
   took most of a fused step's time. They are summed in a fixed order, so a
   step gives the same bits on every call and in a CUDA graph. Off, the terms
   are dense tensors; the matrices are the same and the trajectories agree to
   rounding. For comparing the two forms.

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
