# Standard library imports
import datetime
from typing import Any, Dict, List, Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.constants as constants
import twin4build.utils.types as tps
from twin4build.systems.utils.discrete_statespace_system import (
    DiscreteStatespaceSystem,
    bilinear_onestep,
)


class BuildingSpaceMassTorchSystem(core.System, nn.Module):
    r"""
    Building Space CO2 Concentration Model using Mass Balance Dynamics.

    This model represents the CO2 concentration dynamics in a building space considering
    supply and exhaust air flows, occupant CO2 generation, infiltration, and outdoor 
    CO2 concentration using bilinear state-space dynamics.

    Args:
        V: Volume of the space [m³]
        G_occ: CO2 generation rate per occupant [kg_CO2/s]
        m_inf: Infiltration rate [kg/s]

    Mathematical Formulation
    ------------------------

    **Continuous-Time Differential Equation:**

    The CO2 concentration dynamics are governed by a mass balance on the room air.
    With the room air mass :math:`m_{air} = \rho_{air} V` [kg], the implemented
    equation is:

    .. math::

       m_{air}\frac{dC}{dt} = \dot{m}_{sup}C_{out} - \dot{m}_{exh}C + \dot{m}_{inf}(C_{out} - C) + G_{occ} N_{occ} \frac{M_{air}}{M_{CO2}} \cdot 10^6

    where:

       - :math:`m_{air} = \rho_{air} V`: Mass of air in the space [kg]
         (:math:`\rho_{air}` is the constant air density from
         ``twin4build.utils.constants``)
       - :math:`V`: Volume of the space [m³] (parameter)
       - :math:`C`: Indoor CO2 concentration [ppmv] (state variable)
       - :math:`\dot{m}_{sup}`: Supply air mass flow rate [kg/s] (input)
       - :math:`\dot{m}_{exh}`: Exhaust air mass flow rate [kg/s] (input)
       - :math:`\dot{m}_{inf}`: Infiltration mass flow rate [kg/s] (parameter)
       - :math:`C_{out}`: Outdoor CO2 concentration [ppmv] (input)
       - :math:`G_{occ}`: CO2 generation rate per occupant [kg_CO2/s] (parameter)
       - :math:`N_{occ}`: Number of occupants (input)
       - :math:`M_{air}`, :math:`M_{CO2}`: Molar masses of air and CO2

    The factor :math:`\frac{M_{air}}{M_{CO2}} \cdot 10^6` converts the occupant
    CO2 mass generation [kg_CO2/s] per kg of room air into a rate of change of
    the volumetric (molar) concentration [ppmv/s].

    .. note::
       Concentrations are expressed in **ppmv** (parts per million by volume), 
       which is equivalent to **ppm-moles** (molar fraction × 10⁶) for ideal gases.

    Note: Supply air CO2 concentration is assumed equal to outdoor CO2 concentration.

    **State-Space Representation:**

    The system is implemented using the DiscreteStatespaceSystem (the continuous
    dynamics above are discretized each step by ``bilinear_onestep`` using the
    bilinear/Tustin one-step map) with matrices:

    *State vector:* :math:`\mathbf{x} = \begin{bmatrix}C\end{bmatrix}`

    *Input vector:* :math:`\mathbf{u} = \begin{bmatrix}\dot{m}_{sup} \\ \dot{m}_{exh} \\ C_{out} \\ N_{occ}\end{bmatrix}`

    *Base System Matrices:*

    .. math::

       \mathbf{A} = \begin{bmatrix} -\frac{\dot{m}_{inf}}{m_{air}} \end{bmatrix}

       \mathbf{B} = \begin{bmatrix} 0 & 0 & \frac{\dot{m}_{inf}}{m_{air}} & \frac{G_{occ}}{m_{air}}\frac{M_{air}}{M_{CO2}} \cdot 10^6 \end{bmatrix}

       \mathbf{C} = \begin{bmatrix} 1 \end{bmatrix}

       \mathbf{D} = \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix}

    **Bilinear Coupling Matrices:**

    *State-Input Coupling (E matrices):*

    .. math::

       \mathbf{E} \in \mathbb{R}^{4 \times 1 \times 1} = \begin{bmatrix}
       \begin{bmatrix} 0 \end{bmatrix} & \text{(supply flow)} \\
       \begin{bmatrix} -\frac{1}{m_{air}} \end{bmatrix} & \text{(exhaust flow)} \\
       \begin{bmatrix} 0 \end{bmatrix} & \text{(outdoor CO2)} \\
       \begin{bmatrix} 0 \end{bmatrix} & \text{(occupants)}
       \end{bmatrix}

    *Input-Input Coupling (F matrices):*

    .. math::

       \mathbf{F} \in \mathbb{R}^{4 \times 1 \times 4} = \begin{bmatrix}
       \begin{bmatrix} 0 & 0 & \frac{1}{m_{air}} & 0 \end{bmatrix} & \text{(supply flow)} \\
       \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix} & \text{(exhaust flow)} \\
       \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix} & \text{(outdoor CO2)} \\
       \begin{bmatrix} 0 & 0 & 0 & 0 \end{bmatrix} & \text{(occupants)}
       \end{bmatrix}


    *Bilinear Effects*

    The bilinear terms handle specific flow-dependent mass transfer effects:
       - :math:`\mathbf{E}[1,0,0] \cdot u_1 \cdot x_0 = -\frac{1}{m_{air}} \dot{m}_{exh} C`: Exhaust flow removing CO2
       - :math:`\mathbf{F}[0,0,2] \cdot u_0 \cdot u_2 = \frac{1}{m_{air}} \dot{m}_{sup} C_{out}`: Supply flow bringing outdoor air

    Physical Interpretation
    -----------------------

    **Mass Balance System:**
       - Single state represents indoor CO2 concentration
       - Inputs represent ventilation flows, outdoor conditions, and occupancy
       - Bilinear terms model flow-dependent mass transfer accurately

    **Flow-Dependent Effects:**
       - Supply air flow brings outdoor CO2 at outdoor concentration (F matrix coupling)
       - Exhaust air flow removes CO2 at indoor concentration (E matrix coupling)

    Computational Features
    ----------------------

       - **Automatic Differentiation:** PyTorch tensors enable gradient computation
       - **Adaptive Discretization:** Matrices updated when flows change significantly
       - **Parameter Estimation:** All mass balance parameters available for calibration

    Examples
    --------
    Basic CO2 model:

    >>> import twin4build as tb
    >>>
    >>> # Create CO2 model with default parameters
    >>> co2_model = tb.BuildingSpaceMassTorchSystem(
    ...     V=150,          # Room volume [m³]
    ...     G_occ=6e-6,     # Higher CO2 generation per person
    ...     m_inf=0.002,    # Higher infiltration rate
    ...     id="zone_1_co2"
    ... )

    Large space CO2 model:

    >>> # Model for large space with higher occupancy
    >>> co2_model = tb.BuildingSpaceMassTorchSystem(
    ...     V=500,          # Large space volume
    ...     G_occ=4e-6,     # Lower per-person generation
    ...     m_inf=0.005,    # Higher infiltration for large space
    ...     id="large_space_co2"
    ... )
    """

    def __init__(
        self, V: float = 100, G_occ: float = 5e-6, m_inf: float = 0.001, **kwargs
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        # Store parameters as tps.Parameters
        self.V = tps.Parameter(
            torch.tensor(V, dtype=tps.float_dtype()), requires_grad=False
        )
        self.G_occ = tps.Parameter(
            torch.tensor(G_occ, dtype=tps.float_dtype()), requires_grad=False
        )
        self.m_inf = tps.Parameter(
            torch.tensor(m_inf, dtype=tps.float_dtype()), requires_grad=False
        )

        # Define inputs and outputs
        self.input = {
            "supplyAirFlowRate": tps.Scalar(),  # Supply air flow rate [kg/s]
            "exhaustAirFlowRate": tps.Scalar(),  # Exhaust air flow rate [kg/s]
            "outdoorCO2": tps.Scalar(),  # Outdoor CO2 concentration [ppmv]
            "numberOfPeople": tps.Scalar(),  # Number of occupants
        }

        # Define outputs
        self.output = {
            "indoorCO2": tps.Scalar(400),  # Indoor CO2 concentration [ppmv]
        }

        # Define parameters for calibration
        self.parameter = {
            "V": {"lb": 10.0, "ub": 1000.0},
            "G_occ": {"lb": 0.000001, "ub": 0.00001},
            "m_inf": {"lb": 0.0001, "ub": 0.01},
        }

        self._config = {"parameters": list(self.parameter.keys())}
        self.INITIALIZED = False

    def initialize(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
    ) -> None:
        """Initialize the mass balance model by setting up the state-space representation."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)

        if hasattr(self, "_n_c_compiled") and self._n_c_compiled > 1:
            self.n_c = self._n_c_compiled
        else:
            self.n_c = 1

        # Initialize I/O
        for input in self.input.values():
            input.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )
        for output in self.output.values():
            output.initialize(
                n_t=max_timesteps,
                n_s=batch_size,
                n_c=self.n_c,
            )

        # Expand parameters to n_c dimension for vectorization
        self.V = self.V.expand_to_n_c(self.n_c)
        self.G_occ = self.G_occ.expand_to_n_c(self.n_c)
        self.m_inf = self.m_inf.expand_to_n_c(self.n_c)

        if not self.INITIALIZED:
            # First initialization
            self._create_state_space_model()
            self.ss_model.initialize(start_time, end_time, step_size)

            # FIX: Set correct initial state for batch
            x0_tensor = self._get_initial_state_tensor()
            self.ss_model.set_state(x0_tensor)

            self.INITIALIZED = True
        else:
            # Re-initialize the state space
            self._create_state_space_model()  # We need to re-create the model because the parameters have changed to create a new computation graph
            self.ss_model.initialize(start_time, end_time, step_size)

            # FIX: Set correct initial state for batch
            x0_tensor = self._get_initial_state_tensor()
            self.ss_model.set_state(x0_tensor)

        # Drop per-params forward caches: a fresh simulation must not reuse
        # matrices (or their autograd graph) from a previous run.
        self._fwd_mat_cache = None
        self._forward_params_cache = None

    def _get_initial_state_tensor(self):
        # Get dimensions from indoorCO2
        # Scalar.get() returns shape (n_s, n_c)
        co2_indoor = self.output["indoorCO2"].get()
        n_s = co2_indoor.shape[0]
        n_c = co2_indoor.shape[1]

        # x0 shape: (n_s, n_c, n_states) where n_states = 1
        x0 = torch.zeros(
            (n_s, n_c, 1), dtype=co2_indoor.dtype, device=co2_indoor.device
        )

        x0[:, :, 0] = co2_indoor

        return x0

    #: Physical parameters, in a fixed order (the ``forward`` theta contract).
    PARAM_NAMES = ("V", "G_occ", "m_inf")

    def _ss_layout(self):
        """Port <-> matrix index map, mirroring :meth:`forward` exactly:
        ``u = [supplyAirFlowRate, exhaustAirFlowRate, outdoorCO2,
        numberOfPeople]``; single output row ``indoorCO2``."""
        return {
            "u": [
                ("supplyAirFlowRate", 1), ("exhaustAirFlowRate", 1),
                ("outdoorCO2", 1), ("numberOfPeople", 1),
            ],
            "y": {"indoorCO2": 0},
        }

    def _build_matrices(self, p=None):
        """Build the CO2 mass-balance matrices ``(A, B, C, D, E, F)`` from the
        physical parameters -- a pure function of ``p`` (a dict of physical values
        for :attr:`PARAM_NAMES`; defaults to ``self.<name>.get()``).  Passing ``p``
        explicitly is the functorch fast path (plain-tensor args, so
        ``vmap(jacrev)`` is clean); see the thermal system for the rationale.
        """
        if p is None:
            p = {name: getattr(self, name).get() for name in self.PARAM_NAMES}

        # Single state for CO2 concentration
        n_states = 1
        n_inputs = len(self.input)

        # Get parameter values - shape (n_c,)
        V = p["V"]
        G_occ = p["G_occ"]
        m_inf = p["m_inf"]
        n_c = self.n_c
        # Parameters' device/dtype: _build_matrices re-runs on cache miss
        # during stepping, outside initialize()'s device context.
        dev, dt = V.device, V.dtype

        # Calculate air mass from volume and density
        density_air = constants.RHO_AIR
        air_mass = V * density_air  # (n_c,)

        # Initialize A and B matrices with zeros - shape (n_c, n_states, n_states/n_inputs)
        A = torch.zeros((n_c, n_states, n_states), dtype=dt, device=dev)
        B = torch.zeros((n_c, n_states, n_inputs), dtype=dt, device=dev)

        # State matrix A: -sum of all flow rates / air_mass
        A[:, 0, 0] = -(m_inf / air_mass)  # Base coefficient from infiltration

        # Input matrix B coefficients
        # Outdoor CO2 (from infiltration)
        B[:, 0, 2] = m_inf / air_mass  # outdoorCO2 coefficient

        # Number of people
        B[:, 0, 3] = (
            (G_occ / air_mass) * (constants.M_AIR / constants.M_CO2) * 1e6
        )  # numberOfPeople coefficient

        # Output matrix C - Identity matrix for direct observation
        # Shape: (n_c, n_states, n_states)
        C = (
            torch.eye(n_states, dtype=dt, device=dev)
            .unsqueeze(0)
            .expand(n_c, -1, -1)
            .clone()
        )

        # Feedthrough matrix D (no direct feedthrough) - Shape: (n_c, n_states, n_inputs)
        D = torch.zeros((n_c, n_states, n_inputs), dtype=dt, device=dev)

        # E matrix for input-state coupling: shape (n_c, n_inputs, n_states, n_states)
        E = torch.zeros((n_c, n_inputs, n_states, n_states), dtype=dt, device=dev)
        # -m_ex*C (input 1, state 0)
        E[:, 1, 0, 0] = -1 / air_mass  # exhaustAirFlowRate * C

        # F matrix for input-input coupling: shape (n_c, n_inputs, n_states, n_inputs)
        F = torch.zeros((n_c, n_inputs, n_states, n_inputs), dtype=dt, device=dev)
        # m_sup*C_sup (inputs 0 and 2)
        F[:, 0, 0, 2] = 1 / air_mass  # supplyAirFlowRate * supplyAirCO2

        return A, B, C, D, E, F

    def _create_state_space_model(self):
        """Create the internal :class:`DiscreteStatespaceSystem` used by
        ``do_step`` from the matrices built by :meth:`_build_matrices`."""
        A, B, C, D, E, F = self._build_matrices()

        # Initial state - shape (n_c, n_states)
        x0_tensor = self._get_initial_state_tensor()  # (n_s, n_c, n_states)
        x0 = x0_tensor[0, :, :]  # first simulation, all components: (n_c, n_states)

        self.ss_model = DiscreteStatespaceSystem(
            A=A,
            B=B,
            C=C,
            D=D,
            x0=x0,
            state_names=None,
            add_noise=False,
            id=f"ss_mass_model_{self.id}",
            E=E,
            F=F,
        )

    def forward(self, x, inputs, params, sample_time):
        """Pure one-step CO2 dynamics ``(state, inputs, params) -> (new_state, outputs)``.

        Functorch-compatible re-expression of :meth:`do_step`; ``inputs`` is a dict
        of resolved input-port values assembled here in do_step order
        ``[supplyAirFlowRate, exhaustAirFlowRate, outdoorCO2, numberOfPeople]``,
        ``params`` a dict for :attr:`PARAM_NAMES`.  Returns ``(x_next, {"indoorCO2"})``.
        """
        # Params-only matrices, cached per params-dict identity (rebuilt once
        # per theta in a sequential rollout, not once per step).  sample_time
        # is part of the key: the attached disc_cache holds (Ad, Bd)
        # discretized at a specific T.
        cache = getattr(self, "_fwd_mat_cache", None)
        if cache is None or cache[0] is not params or cache[2] != sample_time:
            cache = (params, self._build_matrices(params), sample_time, {})
            self._fwd_mat_cache = cache
        A, B, C, D, E, F = cache[1]
        u = torch.stack(
            [inputs["supplyAirFlowRate"], inputs["exhaustAirFlowRate"],
             inputs["outdoorCO2"], inputs["numberOfPeople"]], dim=-1,
        )
        x_next, y = bilinear_onestep(
            A, B, C, D, E, F, x, u, sample_time, disc_cache=cache[3]
        )
        return x_next, {"indoorCO2": y[..., 0]}

    @property
    def config(self):
        """Get the system configuration."""
        return self._config

    def do_step(
        self,
        second_time: Optional[float] = None,
        date_time: Optional[datetime.datetime] = None,
        step_size: Optional[float] = None,
        step_index: Optional[int] = None,
    ) -> None:
        """Execute a single simulation step.

        Thin port-I/O wrapper around :meth:`forward` (the single source of
        truth for the dynamics); the inner ``ss_model`` only carries the
        state between steps.
        """
        inputs = {
            port: self.input[port].get()
            for port in (
                "supplyAirFlowRate", "exhaustAirFlowRate", "outdoorCO2",
                "numberOfPeople",
            )
        }
        x = self.ss_model.get_state()  # (n_s, n_c, n_states)
        x_next, outs = self.forward(
            x, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        self.ss_model.set_state(x_next)
        self.output["indoorCO2"]._set(outs["indoorCO2"], i_t=step_index)
