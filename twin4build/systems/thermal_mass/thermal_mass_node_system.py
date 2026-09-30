# Standard library imports
import datetime
from typing import Dict, List, Optional

# Third party imports
import torch
import torch.nn as nn

# Local application imports
import twin4build.utils.types as tps
from twin4build import core
from twin4build.systems.utils.discrete_statespace_system import (
    DiscreteStatespaceSystem,
    bilinear_onestep,
)
from twin4build.utils.slots import wired_width


class ThermalMassNodeSystem(core.System, nn.Module):
    r"""
    Lumped Thermal Mass Node (1C) Shared by Several Walls.

    One thermal capacitance with one temperature state, fed by the heat flows
    of any number of :class:`~twin4build.systems.wall.wall_system.WallSystem`
    components (or anything else delivering a heat flow).  It is the far side
    of a star of walls: the interior mass of a building that every room is
    coupled to (partitions, slabs, corridors) but that no sensor reads.  A
    :class:`WallSystem` between each zone and this node keeps the interzonal
    energy balance exact, the node stores what the walls deliver.

    Args:
        C: Thermal capacitance of the node [J/K]
        T_init: Initial node temperature [degC]

    Mathematical Formulation
    ------------------------

    .. math::

       C \frac{dT_n}{dt} = \sum_j \dot{Q}_j

    where :math:`\dot{Q}_j` is the heat flow delivered INTO the node on slot
    ``j`` of the ``heatFlowRate`` vector input (a wall's ``heatFlowRateB``
    when the node is its side B).

    *State:* :math:`\mathbf{x} = [T_n]`.  *Inputs:*
    :math:`\mathbf{u} = [\dot{Q}_1, \dots, \dot{Q}_{n}]`.  *Output:* ``temperature``
    :math:`= T_n`.

    .. math::

       \mathbf{A} = [0], \qquad \mathbf{B} = \frac{1}{C}[1, \dots, 1], \qquad
       \mathbf{C} = [1], \qquad \mathbf{D} = [0, \dots, 0]

    The node's temperature is a start-of-step signal by declaration
    (:attr:`LAGGED_OUTPUT_PORTS`): the walls read the previous step's node
    temperature, the loader cuts those edges before its cycle search
    instead of enumerating the cycles of a star of a hundred walls around
    every other feedback loop of the building.  The node is deliberately
    not a fusable component, so the walls stay small blocks and the node
    one block instead of one monolithic block.

    Example:
        >>> node = tb.ThermalMassNodeSystem(C=3e9, T_init=21.0, id="core")
        >>> for i, (zone, wall) in enumerate(zip(zones, walls)):
        ...     model.add_connection(zone, wall, "indoorTemperature", "temperatureA")
        ...     model.add_connection(node, wall, "temperature", "temperatureB")
        ...     model.add_connection(wall, zone, "heatFlowRateA", "wallHeatGain")
        ...     model.add_connection(wall, node, "heatFlowRateB", "heatFlowRate", input_port_index=i)
    """

    def __init__(
        self,
        C: float = 1e8,
        T_init: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.C = tps.Parameter(
            torch.tensor(C, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.T_init = T_init

        self._input = {
            "heatFlowRate": tps.Vector(),  # Heat flows into the node [W], one slot per wall
        }
        self._output = {
            "temperature": tps.Scalar(T_init),  # Node temperature [degC]
        }
        self.parameter = {
            "C": {"lb": 1e4, "ub": 1e11},
        }
        self._config = {"parameters": list(self.parameter.keys())}
        self.n_flows = 0
        self.INITIALIZED = False

    @property
    def config(self) -> Dict[str, List[str]]:
        return self._config

    @property
    def input(self) -> dict:
        """``heatFlowRate``: heat flows into the node [W], a vector with one
        slot per connected wall."""
        return self._input

    @property
    def output(self) -> dict:
        """``temperature``: the node temperature [degC]."""
        return self._output

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Size the heat-flow vector from the wired slots, expand the
        parameter, build the state-space model and set the initial state."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_batched") and self._n_c_batched > 1:
            self.n_c = self._n_c_batched
        else:
            self.n_c = 1

        # Count logical vector slots, not connection objects (see the
        # thermal zone's ``wallHeatGain``).  A node with nothing wired keeps
        # one zero slot so the matrices are well formed.
        self.n_flows = max(wired_width(self, "heatFlowRate"), 1)
        self.input["heatFlowRate"].initialize(
            n_t=max_timesteps, n_s=batch_size, n_c=self.n_c, n_v=self.n_flows
        )
        self.output["temperature"].init_value = float(self.T_init)  # see WallSystem.initialize
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size, n_c=self.n_c)

        self.C = self.C.expand_to_n_c(self.n_c)

        self._create_state_space_model()
        self.ss_model.initialize(start_time, end_time, step_size)
        self.ss_model.set_state(self._get_initial_state_tensor())

        self._fwd_mat_cache = None
        self._forward_params_cache = None

        self.INITIALIZED = True

    def _get_initial_state_tensor(self):
        t = self.output["temperature"].get()  # (n_s, n_c)
        n_s, n_c = t.shape
        x0 = torch.zeros((n_s, n_c, 1), dtype=t.dtype, device=t.device)
        x0[:, :, 0] = t
        return x0

    #: Physical parameters, in a fixed order (the ``forward`` theta contract).
    SUPPORTS_TRANSFORM_MODE = True
    PARAM_NAMES = ("C",)
    #: The temperature is delivered from the start of the step: its edges are
    #: cut by the loader before cycle detection (a declared one-step lag).
    LAGGED_OUTPUT_PORTS = frozenset({"temperature"})

    def _ss_layout(self):
        """``u = [heatFlowRate x n_flows]``; output row ``temperature``."""
        return {
            "u": [("heatFlowRate", self.n_flows)],
            "y": {"temperature": 0},
        }

    def _ss_support(self):
        """Structural support of ``D``, ``E`` and ``F``: none."""
        return {"D": frozenset(), "E": frozenset(), "F": frozenset()}

    def _build_matrices(self, p=None):
        """``(A, B, C, D, E, F)`` from the physical parameters, a pure
        function of ``p`` (a dict for :attr:`PARAM_NAMES`; defaults to the
        component's own values)."""
        if p is None:
            p = {name: getattr(self, name).get() for name in self.PARAM_NAMES}
        C = p["C"]  # (n_c,)
        n_c = C.shape[0]
        dev, dt = C.device, C.dtype
        n_u = self.n_flows

        A = torch.zeros((n_c, 1, 1), dtype=dt, device=dev)
        B = (1 / C).reshape(n_c, 1, 1).expand(n_c, 1, n_u)
        C_out = torch.ones((n_c, 1, 1), dtype=dt, device=dev)
        D = torch.zeros((n_c, 1, n_u), dtype=dt, device=dev)
        # No bilinear terms: E is (n_c, n_u, n_x, n_x), F is (n_c, n_u, n_x, n_u).
        E = torch.zeros((n_c, n_u, 1, 1), dtype=dt, device=dev)
        F = torch.zeros((n_c, n_u, 1, n_u), dtype=dt, device=dev)
        return A, B, C_out, D, E, F

    def _create_state_space_model(self):
        A, B, C_out, D, E, F = self._build_matrices()
        x0 = self._get_initial_state_tensor()[0, :, :]  # (n_c, 1)
        self.ss_model = DiscreteStatespaceSystem(
            A=A,
            B=B,
            C=C_out,
            D=D,
            x0=x0,
            state_names=["T_node"],
            E=E,
            F=F,
            add_noise=False,
            id=f"ss_model_{self.id}",
        )

    def step_constants(self, params):
        """The theta-only matrices for one rollout (see ``StepParams``)."""
        return self._build_matrices(params)

    def forward(self, x, inputs, params, sample_time, transform_mode=None):
        """Pure one-step node dynamics ``(state, inputs, params) -> (new_state, outputs)``.
        ``inputs["heatFlowRate"]`` has shape ``(n_c, n_flows)``."""
        if transform_mode:
            matrices = getattr(params, "matrices", None)
            if matrices is None:
                matrices = self._build_matrices(params)
            disc_cache = None
        else:
            cache = getattr(self, "_fwd_mat_cache", None)
            if cache is None or cache[0] is not params or cache[2] != sample_time:
                cache = (params, self._build_matrices(params), sample_time, {})
                self._fwd_mat_cache = cache
            matrices = cache[1]
            disc_cache = cache[3]
        A, B, C_out, D, E, F = matrices
        u = inputs["heatFlowRate"]
        x_next, y = bilinear_onestep(
            A, B, C_out, D, E, F, x, u, sample_time, disc_cache=disc_cache, transform_mode=transform_mode
        )
        return x_next, {"temperature": y[..., 0]}

    def do_step(
        self,
        second_time=None,
        date_time=None,
        step_size=None,
        step_index: Optional[int] = None,
    ) -> None:
        """One simulation step: a port-I/O wrapper around :meth:`forward`."""
        inputs = {"heatFlowRate": self.input["heatFlowRate"].get()}  # (n_s, n_c, n_v)
        x = self.ss_model.get_state()
        x_next, outs = self.forward(
            x, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.ss_model.set_state(x_next)
        self.output["temperature"]._set(outs["temperature"], i_t=step_index)
