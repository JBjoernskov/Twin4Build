Benchmarks
==========

Reproducible performance studies, in the same runnable-notebook format as the
:doc:`examples_and_tutorials`. Each notebook states what it measures, prints a
results table, and ends with a "how to read the results" section so the numbers
can be interpreted rather than just quoted.

All of them detect the available hardware and degrade gracefully: the GPU arms
are skipped automatically when CUDA is unavailable, so every notebook runs
(CPU-only) on any machine.

Like the examples, each opens directly in Google Colab via the badge below.
Select a GPU runtime first (**Runtime → Change runtime type → GPU**), otherwise
only the CPU results are produced.

Estimation and optimization
---------------------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/gpu_benchmark_collocation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Solver comparison, CPU vs GPU</b>: SLSQP single-shooting against IPOPT collocation (Gauss-Newton and exact-Hessian variants) on the full-workflow calibration problem &mdash; wall-clock, fit quality, the torch-vs-IPOPT cost split that bounds any GPU speedup, and whether float32 can satisfy the collocation defect tolerance</p>

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/gpu_benchmark_estimation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Estimation kernels, CPU vs GPU</b>: The real estimation pipeline plus isolated kernel benchmarks (bilinear ZOH rollout, cached-discretization rollout, collocation defect evaluation) swept over batch size</p>

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/gpu_benchmark_optimizer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Optimizer, CPU vs GPU</b>: Control-schedule optimization throughput across devices and batch sizes</p>

Scaling
-------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/gpu_benchmark_scaling.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Model-size scaling</b>: Forward simulation, estimation and collocation swept over the number of zones, showing how per-candidate cost falls with the component-batch dimension</p>

Device support
--------------

.. raw:: html

   <p><a target="_blank" href="https://colab.research.google.com/github/JBjoernskov/Twin4Build/blob/GITHUB_NOTEBOOK_BRANCH/twin4build/examples/gpu_verify_device_support.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> <b>Device-support verification</b>: Checks that a model moved with ``Model.to(device, dtype)`` produces the same results on GPU as on CPU &mdash; run this first if a GPU result looks wrong</p>

Interpreting benchmark results
------------------------------

A few things are worth keeping in mind when reading any of these numbers.

**A GPU does not make a small model faster.** For a single simulation or a
single-start estimation of a handful of zones, each step is a microsecond-scale
operation and kernel-launch latency dominates; the CPU usually wins. Device
support pays off for *batched* work &mdash; multi-start estimation, scenario and
ensemble studies, portfolios of buildings via the ``n_c`` component-batch
dimension &mdash; and for large many-zone models, where per-candidate cost falls
almost linearly with batch size.

**Single-shooting and collocation parallelise differently.** A single-shooting
rollout is sequential in time (step ``t+1`` needs step ``t``) and cannot be
parallelised over the horizon on any hardware. Collocation evaluates every
segment's defect and derivative independently, which is exactly the shape a GPU
rewards. Comparisons between the two therefore shift with the hardware, and a
result measured on one device does not transfer to the other.

**Only part of a solve can move to the GPU.** The torch rollouts, Jacobians and
Hessians run on the model's device, but the IPOPT/SciPy solver &mdash; including
the sparse KKT factorization &mdash; stays on the CPU. The fraction of wall-clock
spent in torch is a hard ceiling on any speedup (Amdahl), and the remaining
per-iteration cost multiplied by the iteration count is a floor no hardware
removes. The collocation notebook measures both explicitly.

**Precision matters for collocation specifically.** ``Model.to(device,
torch.float32)`` is a large speedup on consumer GPUs, whose float64 throughput
is typically 1/32 to 1/64 of float32. But collocation enforces the dynamics as
hard equality constraints with a tight violation tolerance, which single
precision may be unable to satisfy. Forward simulation and optimization tolerate
float32 far more readily.

**Compare quality at equal wall-clock, not at equal iterations.** Collocation
promotes every timestep-boundary state to a decision variable, so its iterations
are not comparable in count or cost to a single-shooting iteration. The
meaningful question is which method reaches a given fit quality first.
