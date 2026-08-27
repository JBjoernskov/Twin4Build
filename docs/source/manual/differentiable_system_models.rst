Building differentiable System models
=====================================

This guide defines the tensor-execution contract for Twin4Build component
models. It applies to every :class:`~twin4build.systems.saref4syst.system.System`
that implements tensor mathematics and to every ``torch.nn.Module`` used inside
such a system. Follow it even when a component currently runs only on the CPU:
the same rules make the model correct under batching, automatic
differentiation, ``torch.func`` transforms, compilation, and CUDA Graph replay.

The central design rule is:

   **Topology and tensor mathematics are separate.** Topology is static Python
   metadata established during initialization. Numerical behavior is a pure
   tensor function whose outputs depend only on its tensor inputs.

Why this contract exists
------------------------

Twin4Build has two component execution paths:

* ``do_step`` integrates a component into the object-graph simulator. It reads
  and writes ports and histories.
* ``forward(state, inputs, parameters, sample_time, ...)`` is composed by
  ``Simulator(..., execution_mode="composed")`` for estimation and
  collocation maps. PyTorch may call it under ``vmap``,
  ``jacrev``, higher-order differentiation, or CUDA Graph capture.

``do_step`` must be a thin I/O wrapper around ``forward``. There must not be a
second implementation of the physics. Otherwise normal simulation and the
derivative supplied to an optimizer can silently describe different models.

Execution-mode ownership
------------------------

Execution policy belongs to :class:`~twin4build.simulator.simulator.Simulator`.
Use ``Simulator(model, execution_mode="composed")`` to select reusable pure
rollouts, or override one public run with
``simulator.simulate(..., execution_mode="object_graph")``. Estimator no
longer accepts ``fast`` or ``fast_validate`` options. The experimental
``("custom", "batched-bfgs", "ad")``, ``batched-lm``, and
``batched-newton`` methods require composed execution and fail rather than
silently dropping a derivative path.

The object-graph materialization pass remains responsible for complete port
and history population, including data-driven and FMU components. Pure
rollouts reuse the captured exogenous tensors and must keep shapes static;
this is required for start-axis ``vmap`` and direct CUDA Graph replay.

Pure ``forward`` contract
-------------------------

``forward`` and every helper it calls must:

* accept all changing state, input, and estimated-parameter values as tensors;
* return tensors with stable shapes and a stable dictionary structure;
* avoid reading or writing ports, histories, ``tps.Parameter`` objects, global
  state, or mutable module attributes;
* avoid logging, file I/O, data loading, lazy initialization, and randomness;
* preserve the input device and dtype; and
* use the same equations as ``do_step``.

The plain parameter dictionary is intentional. Differentiated parameters must
be explicit tensor arguments rather than values discovered through module or
``tps.Parameter`` state. Declare their stable order with ``PARAM_NAMES``.

.. code-block:: python

   class MySystem(core.System):
       SUPPORTS_TRANSFORM_MODE = True
       PARAM_NAMES = ("capacity", "conductance")

       def forward(
           self, state, inputs, parameters, sample_time, transform_mode=None
       ):
           # Tensor-only mathematics; no port or module mutation.
           next_state = ...
           return next_state, {"temperature": next_state[..., 0]}

       def do_step(self, second_time, date_time, step_size, step_index):
           state = self.get_state()
           inputs = {name: port.get() for name, port in self.input.items()}
           next_state, outputs = self.forward(
               state, inputs, self._forward_params(), step_size
           )
           self.set_state(next_state)
           for name, value in outputs.items():
               self.output[name]._set(value, i_t=step_index)

Tensor values must not control Python
-------------------------------------

Never convert a tensor value to a Python value inside the pure path. Operations
such as these synchronize CUDA with the host and make the graph depend on one
particular parameter iterate:

.. code-block:: python

   # Incorrect
   if bool((matrix != 0).any()):
       ...
   if tensor.item() > 0:
       ...
   indices = tensor.tolist()

This also includes ``float(tensor)``, ``int(tensor)``, NumPy conversion, and
exceptions or dictionary keys selected from tensor data.

Use tensor control flow for numerical choices:

.. code-block:: python

   output = torch.where(condition, value_if_true, value_if_false)

Use Python control flow only for genuinely static structure, such as the number
of ports or states determined during ``initialize``. If a matrix entry may be
nonzero anywhere in the admissible parameter domain, declare that possibility
as metadata; do not inspect its value at runtime.

Device- and dtype-safe tensor construction
------------------------------------------

Derive device and dtype from a runtime tensor. Persistent tensors in an
``nn.Module`` must be parameters or registered buffers so that ``module.to()``
moves them. Temporary tensors should use ``*_like``, ``new_*``, or explicit
device-native constructors.

.. code-block:: python

   reference = parameters["capacity"]
   zero = torch.zeros_like(reference)
   identity = torch.eye(
       n_states, dtype=reference.dtype, device=reference.device
   )
   basis = identity[0]

Do not construct a CPU tensor and copy it to CUDA inside ``forward``:

.. code-block:: python

   # Incorrect during CUDA Graph capture: this performs a host-to-device copy.
   basis = torch.tensor([1.0, 0.0], device=reference.device)

Constructors such as ``torch.eye`` and ``torch.zeros`` with the final CUDA
device allocate and fill on-device. This distinction matters during capture,
where an unpinned host-to-device copy is illegal.

Use functional tensor assembly
------------------------------

Prefer ``stack``, ``cat``, ``pad``, broadcasting, and out-of-place arithmetic.
Writing tensor arguments into newly allocated tensors by slice assignment can
fail under ``vmap`` because a transformed ``BatchedTensor`` carries a hidden
batch dimension that the destination does not have.

.. code-block:: python

   # Preferred: transformed batch dimensions propagate through every operand.
   top = torch.cat([A * sample_time, B * sample_time], dim=-1)
   block = torch.nn.functional.pad(top, (0, 0, 0, B.shape[-1]))

In-place mutation of inputs, parameters, buffers, cached tensors, or returned
static graph outputs is forbidden. Use in-place operations only on local
tensors when their transform behavior is explicitly tested.

Caching and ``transform_mode``
------------------------------

Ordinary sequential simulation may cache expensive parameter-only work using
the identity of the parameter dictionary and static values such as
``sample_time``. A transformed call must bypass those caches:

.. code-block:: python

   if transform_mode:
       matrices = self._build_matrices(parameters)
       discretization_cache = None
   else:
       cache = getattr(self, "_forward_cache", None)
       if cache is None or cache[0] is not parameters:
           cache = (parameters, self._build_matrices(parameters))
           self._forward_cache = cache
       matrices = cache[1]

Set ``SUPPORTS_TRANSFORM_MODE = True`` only when ``forward`` accepts the
argument and the complete call tree respects it. Composite systems must pass
``transform_mode`` to supporting children and resolve child parameter
dictionaries without identity caches. Do not detect active transforms through
private PyTorch state inside a captured call; the composer supplies this mode
explicitly.

A transformed call must not populate, invalidate, or otherwise mutate an eager
cache. Cached tensors that depend on differentiated inputs can retain stale
values or old autograd graphs and must never be shared between optimizer
iterations.

State-space systems and fusion
------------------------------

A state-space leaf that opts into fusion supplies three static contracts:

``_build_matrices(parameters)``
   A pure function returning ``(A, B, C, D, E, F)``. An explicit parameter
   dictionary is the differentiated path; ``None`` may read the component's own
   values for normal simulation.

``_ss_layout()``
   The stable mapping from named ports to matrix columns/rows. Any
   topology-dependent dimensions must be finalized by ``initialize`` first.

``_ss_support()``
   The conservative structural support of ``D``, ``E``, and ``F`` over the
   entire admissible parameter domain:

   * ``D`` entries are ``(output_row, input_column)``;
   * ``E`` entries are ``(modulating_input, state_row, state_column)``; and
   * ``F`` entries are ``(modulating_input, state_row, forced_input)``.

.. code-block:: python

   def _ss_support(self):
       return {
           "D": frozenset(),
           "E": frozenset({(2, 0, 0)}),
           "F": frozenset({(1, 0, 3)}),
       }

Support is about what *may* be nonzero, not what happens to be nonzero at the
current parameters. Under-declaration can produce an incorrect fused model or
miss an algebraic cycle. Over-declaration preserves numerical correctness but
may reject a valid fusion or perform unnecessary work. Never discover support
by evaluating one default, lower-bound, or random parameter point.

Persistent tensors in ``nn.Module``
-----------------------------------

The same pure-path rules apply to nested ``nn.Module`` classes:

* use ``nn.Parameter`` for trainable module-owned values;
* use ``register_buffer`` for persistent non-trainable tensors;
* create shape/device-dependent temporary tensors from runtime inputs;
* do not create parameters, buffers, or submodules lazily in ``forward``;
* do not mutate module state while differentiating or capturing; and
* when Twin4Build supplies an explicit parameter dictionary, use that
  dictionary rather than closing over a second module-owned value.

CUDA Graph replay uses fixed memory addresses and reuses output storage.
Callers that retain a replay output across another replay must clone it.
Component code should simply return its tensors and must not retain references
to intermediate or output tensors.

Required validation
-------------------

A new or modified tensor component should test all applicable levels:

#. ``do_step`` and ``forward`` produce the same one-step result.
#. A multi-step composed rollout matches the object-graph simulator.
#. CPU and CUDA values agree in float64.
#. ``vmap`` output agrees with an explicit loop.
#. First derivatives and, when used by exact-Hessian estimation, second
   derivatives agree with finite differences or a trusted formulation.
#. Eager and ``transform_mode=True`` values and derivatives agree.
#. A transformed call does not mutate eager caches.
#. State-space support contains every observed nonzero at zero-valued,
   boundary, and randomized interior parameter points.
#. The full composed function can be traced with ``torch.compile(...,
   fullgraph=True)`` as a diagnostic for hidden Python behavior.
#. CUDA Graph capture and replay track perturbed state, parameter, multiplier,
   and objective-scale inputs—not only the values used during capture.

Value parity at the capture point alone is insufficient: a parameter can be
accidentally baked into a graph and still pass that comparison. Always perturb
every dynamic input family and compare replay with a fresh eager evaluation.

Review checklist
----------------

Before merging a component model, verify:

* one implementation of the equations, shared by ``do_step`` and ``forward``;
* no tensor-to-Python conversion or data-dependent Python structure;
* no side effects or mutable caches in transformed execution;
* device-native, dtype-preserving tensor creation;
* functional assembly compatible with hidden ``vmap`` batch dimensions;
* conservative static support for every fusable state-space unit;
* explicit transform-mode propagation through composites; and
* value, rollout, Jacobian, Hessian, and replay tests as applicable.

See :class:`~twin4build.systems.building_space.building_space_thermal_system.BuildingSpaceThermalSystem`
for a state-space implementation and
:class:`~twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system.CascadeControllerSystem`
for transform propagation through a composite.
