# Standard library imports
import datetime
from typing import Callable, Dict, List

# Third party imports
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps


class FunctionSystem(core.System):
    r"""Applies a user-supplied transformation to named scalar inputs.

    Computes ``output = fn(inputs)`` at each time step, where ``inputs`` is a
    dict mapping each declared input port name to its current value (a torch
    tensor).  The generic building block for derived signals that should be
    usable as optimization objectives or constraints -- e.g. the comfort
    residual between a zone temperature and its setpoint::

        discomfort = tb.FunctionSystem(
            inputs=["setpoint", "measured"],
            fn=lambda d: torch.relu(d["setpoint"] - d["measured"]),
            id="Discomfort",
        )
        model.add_connection(heating_setpoint, discomfort, "scheduleValue", "setpoint")
        model.add_connection(building_space, discomfort, "indoorTemperature", "measured")
        # objective: (discomfort, "output", "min")

    The transformation must be built from torch operations and be stateless
    (a pure function of the current inputs).  Under those conditions the
    component is composable: the fast estimation/optimization objectives and
    the GPU-batched Pareto prepass thread ``fn`` directly into the composed
    one-step map (including under ``torch.func.vmap``), and gradients flow
    through it.  Use differentiable primitives (``torch.relu``,
    ``torch.clamp``, ``torch.abs``, ...) rather than Python branches on
    tensor values.

    .. note::
        A Python callable cannot be serialized into the semantic model, so a
        model containing a ``FunctionSystem`` will not round-trip through
        semantic-model save/load.  Simulation, estimation, and optimization
        are unaffected.

    Args:
        inputs: Names of the scalar input ports, in any order. Each must be
            connected before ``model.load()``.
        fn: Callable mapping ``{input_name: tensor}`` to the output tensor.
            Must preserve the batch dimensions of its inputs (any elementwise
            torch expression does).
        **kwargs: Additional keyword arguments passed to the parent
            :class:`core.System` (e.g. ``id``).

    Inputs:
        One :class:`tps.Scalar` port per name in ``inputs``.

    Outputs:
        output (Scalar): Result of ``fn(inputs)``.
    """

    def __init__(
        self,
        inputs: List[str],
        fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert inputs, "FunctionSystem requires at least one input port name"
        assert callable(fn), "fn must be a callable"

        self.input = {name: tps.Scalar() for name in inputs}
        self.output = {"output": tps.Scalar()}
        self._fn = fn

        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for inp in self.input.values():
            inp.initialize(n_t=max_timesteps, n_s=batch_size)
        for out in self.output.values():
            out.initialize(n_t=max_timesteps, n_s=batch_size)

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the math)."""
        _, outs = self.forward(
            None,
            {name: port.get() for name, port in self.input.items()},
            {},
            step_size,
        )
        self.output["output"]._set(outs["output"], i_t=step_index)

    #: No estimated physical parameters (the ``forward`` theta contract);
    #: the transformation is a fixed user-supplied callable.
    PARAM_NAMES = ()

    def forward(self, x, inputs, params, sample_time):
        """Pure algebraic map ``(inputs,) -> outputs`` (stateless).

        ``output = fn(inputs)``.  Functorch-compatible as long as ``fn`` is
        built from torch ops; keeps whatever batch dims the inputs carry
        (``(1,)`` under the composer, ``(n_s, n_c)`` under ``do_step``).
        Returns ``(x, {"output"})``.
        """
        return x, {"output": self._fn(inputs)}
