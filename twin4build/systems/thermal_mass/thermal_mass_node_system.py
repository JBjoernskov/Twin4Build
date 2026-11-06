# Third party imports
import torch

# Local application imports
from twin4build import core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter, Setting, State
from twin4build.systems.utils.discrete_statespace_system import bilinear_onestep


class ThermalMassNodeSystem(core.System):
    r"""
    Lumped Thermal Mass Node (1C) Shared by Several Walls.

    One thermal capacitance with one temperature state, fed by the heat flows
    of any number of :class:`~twin4build.systems.wall.wall_system.WallSystem`
    components (or anything else delivering a heat flow).  It is the far side
    of a star of walls: the interior mass of a building that every room is
    coupled to (partitions, slabs, corridors) but that no sensor reads.  A
    :class:`WallSystem` between each zone and this node keeps the interzonal
    energy balance exact, the node stores what the walls deliver.

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

    T_init = Setting(20.0, doc="Initial node temperature [degC]")
    heatFlowRate = Input(size="wired", doc="Heat flows into the node [W], one slot per wall")
    temperature = Output(init=T_init, doc="Node temperature [degC]")
    T_node = State(init=T_init, names=["T_node"], doc="Node temperature [degC]")
    C = Parameter(1e8, lb=1e4, ub=1e11, relative=(0.1, 10.0), scaling="log", doc="Thermal capacitance [J/K]")

    @property
    def n_flows(self) -> int:
        """The heat flows the node sums: its wired slots (none wired: one,
        held at zero, so the matrices are well formed)."""
        return max(int(self.input["heatFlowRate"]._n_v or 0), 1)

    def _state_heat_capacities(self):
        """``T_node`` holds ``C`` [J/K]."""
        return self.C.get().reshape(-1, 1)

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

    def derive(self, p, step_size):
        """The state-space matrices, which depend on the parameters alone."""
        return {"matrices": self._build_matrices(p)}

    def forward(self, x, inputs, params, step_size):
        """One step of the node: the summed heat flows into its
        capacitance."""
        A, B, C_out, D, E, F = params["matrices"]
        u = inputs["heatFlowRate"]
        if u.shape[-1] == 0:  # nothing wired: no heat flows in
            u = u.new_zeros(*u.shape[:-1], 1)
        x_next, y = bilinear_onestep(
            A, B, C_out, D, E, F, x, u, step_size
        )
        return x_next, {"temperature": y[..., 0]}
