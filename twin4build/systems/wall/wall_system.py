# Third party imports
import torch

# Local application imports
from twin4build import core
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter, Setting, State
from twin4build.systems.utils.discrete_statespace_system import bilinear_onestep


class WallSystem(core.System):
    r"""
    Wall Model (2R1C) for Energy-Consistent Heat Transfer Between Two Zones.

    This model represents a wall separating two temperature zones (two building
    spaces, or a building space and a known boundary temperature such as a
    corridor, the ground, or an unmodeled neighbouring building) as a single
    lumped thermal mass with a resistance on each side. Both sides' heat flows
    are computed from the SAME wall state, so the interzonal energy balance
    holds by construction -- unlike modeling the partition inside each zone,
    which duplicates the wall and breaks conservation.

    Mathematical Formulation
    ------------------------

    **Continuous-Time Differential Equation:**

    The wall temperature :math:`T_w` follows the energy balance

    .. math::

       C \frac{dT_w}{dt} = \frac{T_a - T_w}{R_a} + \frac{T_b - T_w}{R_b}

    where:

       - :math:`T_w`: Wall temperature [degC] (state)
       - :math:`T_a`: Side-A temperature [degC] (input, e.g. zone A air)
       - :math:`T_b`: Side-B temperature [degC] (input, e.g. zone B air or a
         boundary-temperature schedule)
       - :math:`C`: Wall thermal capacitance [J/K]
       - :math:`R_a, R_b`: Side resistances [K/W]

    **Heat Flow Outputs:**

    The heat flows delivered INTO each side (positive = heating that side) are

    .. math::

       \dot{Q}_a = \frac{T_w - T_a}{R_a}, \qquad
       \dot{Q}_b = \frac{T_w - T_b}{R_b}

    Energy conservation follows directly: the wall stores exactly what the two
    sides exchange,

    .. math::

       C \frac{dT_w}{dt} = -(\dot{Q}_a + \dot{Q}_b)

    **State-Space Representation:**

    The system is implemented using the DiscreteStatespaceSystem with matrices:

    *State vector:* :math:`\mathbf{x} = \begin{bmatrix}T_w\end{bmatrix}`

    *Input vector:* :math:`\mathbf{u} = \begin{bmatrix}T_a \\ T_b\end{bmatrix}`

    .. math::

       \mathbf{A} = \begin{bmatrix}-\frac{1}{R_a C} - \frac{1}{R_b C}\end{bmatrix},
       \quad
       \mathbf{B} = \begin{bmatrix}\frac{1}{R_a C} & \frac{1}{R_b C}\end{bmatrix}

    .. math::

       \mathbf{C} = \begin{bmatrix}\frac{1}{R_a} \\ \frac{1}{R_b} \\ 1\end{bmatrix},
       \quad
       \mathbf{D} = \begin{bmatrix}-\frac{1}{R_a} & 0 \\ 0 & -\frac{1}{R_b} \\ 0 & 0\end{bmatrix}

    yielding outputs :math:`[\dot{Q}_a, \dot{Q}_b, T_w]^T`. The step update is
    the exact zero-order-hold discretization; outputs are evaluated at the
    end-of-step state (the DiscreteStatespaceSystem convention).

    Physical Interpretation
    -----------------------

    **Zone-zone partitions:**
       - Each zone connects its ``indoorTemperature`` to one side and receives
         the corresponding heat flow back on its ``wallHeatGain`` port.
       - Because one component owns the wall state, the heat leaving zone A
         through the wall equals the heat absorbed by the wall plus the heat
         entering zone B -- no double-counted wall mass.

    **Known boundary temperatures:**
       - Side B may instead be fed by a schedule or sensor (corridor, ground,
         unmodeled neighbour), replacing the deprecated in-zone
         ``boundaryTemperature`` / ``R_boundary`` / ``C_boundary`` path.

    Numerical Coupling
    ------------------

    The zone <-> wall connections form an algebraic loop that plain
    Gauss-Seidel co-simulation would resolve with a one-step lag -- which for
    stiff couplings (loop gain :math:`\Delta t/(R\,C) > 1`) diverges for any
    execution order.  The framework therefore FUSES the connected zone(s) and
    wall(s) into one monolithic state-space block at ``model.load()`` (see
    :class:`~twin4build.systems.utils.fused_statespace_system.FusedStateSpaceSystem`):
    the coupling is eliminated exactly at the matrix level, so the pair is
    integrated jointly with zero lag and is unconditionally stable for any
    positive :math:`R`, :math:`C` and step size.

    A wall set ``fusable=False`` is kept out of fusion and couples to its
    zones one step late: walls between rooms then leave every room its own
    block (a room's ``wallHeatGain`` may take slots from fused and unfused
    walls alike).

    Computational Features
    ----------------------

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Parameter Estimation:** C, R_a and R_b available for calibration
       - **Composability:** the wall is one pure ``forward``, so it
         takes part in the fast estimation and optimization paths

    Examples
    --------
    Partition wall between two zones:

    >>> import twin4build as tb
    >>>
    >>> wall = tb.WallSystem(C=2e5, R_a=0.05, R_b=0.05, id="wall_AB")
    >>> # zone_a.indoorTemperature -> wall.temperatureA
    >>> # zone_b.indoorTemperature -> wall.temperatureB
    >>> # wall.heatFlowRateA -> zone_a.wallHeatGain (slot 0)
    >>> # wall.heatFlowRateB -> zone_b.wallHeatGain (slot 0)

    Wall toward a known boundary temperature (e.g. corridor schedule):

    >>> wall = tb.WallSystem(C=5e5, R_a=0.03, R_b=0.03, id="wall_corridor")
    >>> # corridor_schedule.scheduleValue -> wall.temperatureB
    """

    T_init = Setting(20.0, doc="Initial wall temperature [degC]")
    fusable = Setting(
        True, structural=True,
        doc="Fuse with the zones it is connected to; False: couple to them one step late",
    )
    temperatureA = Input(doc="Side-A temperature [degC]")
    temperatureB = Input(doc="Side-B temperature [degC]")
    heatFlowRateA = Output(doc="Heat flow into side A [W]")
    heatFlowRateB = Output(doc="Heat flow into side B [W]")
    wallTemperature = Output(init=T_init, doc="Wall temperature [degC]")
    T_wall = State(init=T_init, names=["T_wall"], doc="Wall temperature [degC]")
    C = Parameter(1e5, lb=1e3, ub=1e8, relative=(0.1, 10.0), scaling="log", doc="Thermal capacitance [J/K]")
    R_a = Parameter(0.05, lb=1e-4, ub=10.0, relative=(0.1, 10.0), scaling="log",
                    doc="Resistance between side A and the wall [K/W]")
    R_b = Parameter(0.05, lb=1e-4, ub=10.0, relative=(0.1, 10.0), scaling="log",
                    doc="Resistance between side B and the wall [K/W]")

    def _state_heat_capacities(self):
        """``T_wall`` holds ``C`` [J/K]."""
        return self.C.get().reshape(-1, 1)

    #: Fusable coupling ports: both temperature inputs enter the linear B
    #: matrix, and all outputs are exact linear functions of (state, inputs),
    #: so a zone<->wall connection can be eliminated into one monolithic
    #: state-space block (see FusedStateSpaceSystem).
    FUSABLE_INPUT_PORTS = frozenset({"temperatureA", "temperatureB"})
    FUSABLE_OUTPUT_PORTS = frozenset(
        {"heatFlowRateA", "heatFlowRateB", "wallTemperature"}
    )

    def _ss_layout(self):
        """Port <-> matrix index map, mirroring :meth:`forward` exactly:
        ``u = [temperatureA, temperatureB]``; output rows
        ``[heatFlowRateA, heatFlowRateB, wallTemperature]``."""
        return {
            "u": [("temperatureA", 1), ("temperatureB", 1)],
            "y": {"heatFlowRateA": 0, "heatFlowRateB": 1, "wallTemperature": 2},
        }

    def _ss_support(self):
        """Conservative structural support of the ``D``, ``E`` and ``F`` matrices."""
        return {
            "D": frozenset({(0, 0), (1, 1)}),
            "E": frozenset(),
            "F": frozenset(),
        }

    def _build_matrices(self, p=None):
        """Build the wall state-space matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a pure function of ``p`` (a dict of physical
        values for :attr:`PARAM_NAMES`; defaults to ``self.<name>.get()``).
        Passing ``p`` is the functorch fast path; see the thermal system for
        the rationale.
        """
        if p is None:
            p = {name: getattr(self, name).get() for name in self.PARAM_NAMES}

        C = p["C"]  # (n_c,)
        R_a = p["R_a"]
        R_b = p["R_b"]
        n_c = C.shape[0]
        # Parameters' device/dtype: _build_matrices re-runs on cache miss
        # during stepping, outside initialize()'s device context.
        dev, dt = C.device, C.dtype

        inv_ra = 1 / R_a
        inv_rb = 1 / R_b
        A = (-(inv_ra + inv_rb) / C).reshape(n_c, 1, 1)
        B = torch.stack([inv_ra / C, inv_rb / C], dim=-1).unsqueeze(1)

        # Outputs: [heatFlowRateA, heatFlowRateB, wallTemperature]
        one = torch.ones_like(C)
        zero = torch.zeros_like(C)
        C_out = torch.stack([inv_ra, inv_rb, one], dim=1).unsqueeze(-1)
        D = torch.stack(
            [
                torch.stack([-inv_ra, zero], dim=-1),
                torch.stack([zero, -inv_rb], dim=-1),
                torch.stack([zero, zero], dim=-1),
            ],
            dim=1,
        )

        # No bilinear terms.
        E = torch.zeros((n_c, 2, 1, 1), dtype=dt, device=dev)
        F = torch.zeros((n_c, 2, 1, 2), dtype=dt, device=dev)

        return A, B, C_out, D, E, F

    def derive(self, p, step_size):
        """The state-space matrices, which depend on the parameters alone."""
        return {"matrices": self._build_matrices(p)}

    def forward(self, x, inputs, params, step_size):
        """One step of the wall: the next wall temperature, and the heat flows
        INTO each side, from the end-of-step wall state."""
        A, B, C_out, D, E, F = params["matrices"]
        u = torch.stack([inputs["temperatureA"], inputs["temperatureB"]], dim=-1)
        x_next, y = bilinear_onestep(
            A,
            B,
            C_out,
            D,
            E,
            F,
            x,
            u,
            step_size,
        )
        return x_next, {
            "heatFlowRateA": y[..., 0],
            "heatFlowRateB": y[..., 1],
            "wallTemperature": y[..., 2],
        }


# Deprecated aliases (removed in twin4build 2.1)
WallTorchSystem = WallSystem
