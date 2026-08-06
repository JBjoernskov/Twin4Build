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


class WallSystem(core.System, nn.Module):
    r"""
    Wall Model (2R1C) for Energy-Consistent Heat Transfer Between Two Zones.

    This model represents a wall separating two temperature zones (two building
    spaces, or a building space and a known boundary temperature such as a
    corridor, the ground, or an unmodeled neighbouring building) as a single
    lumped thermal mass with a resistance on each side. Both sides' heat flows
    are computed from the SAME wall state, so the interzonal energy balance
    holds by construction -- unlike modeling the partition inside each zone,
    which duplicates the wall and breaks conservation.

    Args:
        C: Thermal capacitance of the wall [J/K]
        R_a: Thermal resistance between side-A temperature and the wall node [K/W]
        R_b: Thermal resistance between side-B temperature and the wall node [K/W]
        T_init: Initial wall temperature [degC]

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

    Computational Features
    ----------------------

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Parameter Estimation:** C, R_a and R_b available for calibration
       - **Composability:** ``do_step`` delegates to a pure ``forward``, so the
         component participates in the fast estimation/optimization paths

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

    def __init__(
        self,
        C: float = 1e5,
        R_a: float = 0.05,
        R_b: float = 0.05,
        T_init: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.C = tps.Parameter(
            torch.tensor(C, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.R_a = tps.Parameter(
            torch.tensor(R_a, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.R_b = tps.Parameter(
            torch.tensor(R_b, dtype=tps.float_dtype()), requires_grad=False, scaling="log"
        )
        self.T_init = T_init

        self._input = {
            "temperatureA": tps.Scalar(),  # Side-A temperature [degC]
            "temperatureB": tps.Scalar(),  # Side-B temperature [degC]
        }
        self._output = {
            "heatFlowRateA": tps.Scalar(0),  # Heat flow into side A [W]
            "heatFlowRateB": tps.Scalar(0),  # Heat flow into side B [W]
            "wallTemperature": tps.Scalar(T_init),  # Wall temperature [degC]
        }
        self.parameter = {
            "C": {"lb": 1e3, "ub": 1e8},
            "R_a": {"lb": 1e-4, "ub": 10.0},
            "R_b": {"lb": 1e-4, "ub": 10.0},
        }
        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    @property
    def config(self) -> Dict[str, List[str]]:
        """Get the configuration parameters.

        Returns:
            Dict[str, List[str]]: Dictionary containing configuration parameter names.
        """
        return self._config

    @property
    def input(self) -> dict:
        """
        Get the input ports of the wall system.

        Returns:
            dict: Dictionary containing input ports:
                - "temperatureA": Side-A temperature [degC]
                - "temperatureB": Side-B temperature [degC]
        """
        return self._input

    @property
    def output(self) -> dict:
        """
        Get the output ports of the wall system.

        Returns:
            dict: Dictionary containing output ports:
                - "heatFlowRateA": Heat flow into side A [W]
                - "heatFlowRateB": Heat flow into side B [W]
                - "wallTemperature": Wall temperature [degC]
        """
        return self._output

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the wall system for simulation.

        This method performs the following initialization steps:

        1. Initializes input/output data structures
        2. Creates or reinitializes the internal state-space model
        3. Sets the initial wall temperature

        Args:
            start_time (datetime.datetime): Start time of the simulation period.
            end_time (datetime.datetime): End time of the simulation period.
            step_size (int): Time step size in seconds.
        """
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_compiled") and self._n_c_compiled > 1:
            self.n_c = self._n_c_compiled
        else:
            self.n_c = 1

        for input in self.input.values():
            input.initialize(n_t=max_timesteps, n_s=batch_size, n_c=self.n_c)
        for output in self.output.values():
            output.initialize(n_t=max_timesteps, n_s=batch_size, n_c=self.n_c)

        # Expand parameters to n_c dimension for vectorization
        self.C = self.C.expand_to_n_c(self.n_c)
        self.R_a = self.R_a.expand_to_n_c(self.n_c)
        self.R_b = self.R_b.expand_to_n_c(self.n_c)

        self._create_state_space_model()
        self.ss_model.initialize(start_time, end_time, step_size)
        self.ss_model.set_state(self._get_initial_state_tensor())

        # Drop per-params forward caches: a fresh simulation must not reuse
        # matrices (or their autograd graph) from a previous run.
        self._fwd_mat_cache = None
        self._forward_params_cache = None

        self.INITIALIZED = True

    def _get_initial_state_tensor(self):
        # Scalar.get() returns shape (n_s, n_c)
        t_wall = self.output["wallTemperature"].get()
        n_s, n_c = t_wall.shape
        x0 = torch.zeros(
            (n_s, n_c, 1), dtype=t_wall.dtype, device=t_wall.device
        )
        x0[:, :, 0] = t_wall
        return x0

    #: Physical parameters, in a fixed order (the ``forward`` theta contract).
    PARAM_NAMES = ("C", "R_a", "R_b")

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

        A = torch.zeros((n_c, 1, 1), dtype=dt, device=dev)
        A[:, 0, 0] = -1 / (R_a * C) - 1 / (R_b * C)

        B = torch.zeros((n_c, 1, 2), dtype=dt, device=dev)
        B[:, 0, 0] = 1 / (R_a * C)
        B[:, 0, 1] = 1 / (R_b * C)

        # Outputs: [heatFlowRateA, heatFlowRateB, wallTemperature]
        C_out = torch.zeros((n_c, 3, 1), dtype=dt, device=dev)
        C_out[:, 0, 0] = 1 / R_a
        C_out[:, 1, 0] = 1 / R_b
        C_out[:, 2, 0] = 1.0

        D = torch.zeros((n_c, 3, 2), dtype=dt, device=dev)
        D[:, 0, 0] = -1 / R_a
        D[:, 1, 1] = -1 / R_b

        # No bilinear terms.
        E = torch.zeros((n_c, 2, 1, 1), dtype=dt, device=dev)
        F = torch.zeros((n_c, 2, 1, 2), dtype=dt, device=dev)

        return A, B, C_out, D, E, F

    def _create_state_space_model(self):
        """Create the internal :class:`DiscreteStatespaceSystem` used by
        ``do_step`` from the matrices built by :meth:`_build_matrices`."""
        A, B, C_out, D, E, F = self._build_matrices()
        x0_tensor = self._get_initial_state_tensor()  # (n_s, n_c, 1)
        x0 = x0_tensor[0, :, :]  # (n_c, 1)
        self.ss_model = DiscreteStatespaceSystem(
            A=A,
            B=B,
            C=C_out,
            D=D,
            x0=x0,
            state_names=["T_wall"],
            E=E,
            F=F,
            add_noise=False,
            id=f"ss_model_{self.id}",
        )

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step wall dynamics ``(state, inputs, params) -> (new_state, outputs)``.

        Functorch-compatible re-expression of :meth:`do_step`. ``inputs`` is a
        dict with ``temperatureA`` and ``temperatureB``; ``params`` a dict for
        :attr:`PARAM_NAMES`. Returns the next wall temperature and the named
        outputs ``{heatFlowRateA, heatFlowRateB, wallTemperature}`` (heat
        flows INTO each side, computed from the end-of-step wall state).
        """
        # Params-only matrices, cached per params-dict identity (rebuilt once
        # per theta in a sequential rollout, not once per step).  sample_time
        # is part of the key: the attached disc_cache holds (Ad, Bd)
        # discretized at a specific T.
        cache = getattr(self, "_fwd_mat_cache", None)
        if cache is None or cache[0] is not params or cache[2] != sample_time:
            cache = (params, self._build_matrices(params), sample_time, {})
            self._fwd_mat_cache = cache
        A, B, C_out, D, E, F = cache[1]
        u = torch.stack(
            [inputs["temperatureA"], inputs["temperatureB"]], dim=-1
        )
        x_next, y = bilinear_onestep(
            A, B, C_out, D, E, F, x, u, sample_time, disc_cache=cache[3]
        )
        return x_next, {
            "heatFlowRateA": y[..., 0],
            "heatFlowRateB": y[..., 1],
            "wallTemperature": y[..., 2],
        }

    def do_step(
        self,
        second_time=None,
        date_time=None,
        step_size=None,
        step_index: Optional[int] = None,
    ) -> None:
        """Perform one simulation step.

        Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the dynamics); the inner ``ss_model`` only carries the
        state between steps.

        Args:
            second_time (float, optional): Current simulation time in seconds.
            date_time (datetime.datetime, optional): Current simulation date and time.
            step_size (float, optional): Time step size in seconds.
            step_index (int, optional): Current simulation step index.
        """
        inputs = {
            "temperatureA": self.input["temperatureA"].get(),
            "temperatureB": self.input["temperatureB"].get(),
        }
        x = self.ss_model.get_state()  # (n_s, n_c, 1)
        x_next, outs = self.forward(
            x, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.ss_model.set_state(x_next)
        self.output["heatFlowRateA"]._set(outs["heatFlowRateA"], i_t=step_index)
        self.output["heatFlowRateB"]._set(outs["heatFlowRateB"], i_t=step_index)
        self.output["wallTemperature"]._set(outs["wallTemperature"], i_t=step_index)

# Deprecated aliases (removed in twin4build 2.1)
WallTorchSystem = WallSystem
