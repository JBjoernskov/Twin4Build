# Third party imports
import torch

# Local application imports
import twin4build.core as core
from twin4build.systems.building_space.building_space_mass_system import (
    BuildingSpaceMassSystem,
)
from twin4build.systems.building_space.building_space_thermal_system import (
    BOUNDARY_WALL_GONE,
    BuildingSpaceThermalSystem,
)
from twin4build.systems.saref4syst.declarations import Child, Input, Output
from twin4build.utils.rgetattr import rgetattr


class BuildingSpaceSystem(core.System):
    r"""
    Combined building space model for both thermal (RC) and CO2 (mass balance) dynamics.

    This class composes BuildingSpaceThermalSystem and BuildingSpaceMassSystem
    to provide a unified building space model that captures both thermal and air quality
    dynamics in a building zone.

    Args:
       thermal: The values of the BuildingSpaceThermalSystem, a dict
       mass: The values of the BuildingSpaceMassSystem, a dict

    Mathematical Formulation
    ------------------------

       See individual component documentation:
          - BuildingSpaceThermalSystem: RC network thermal dynamics
          - BuildingSpaceMassSystem: CO2 mass balance dynamics

       Both models use DiscreteStatespaceSystem for efficient computation and
       automatic differentiation support.

    System Composition:

       The combined model consists of two parallel subsystems:

       **Thermal Subsystem (BuildingSpaceThermalSystem):**
          - Models temperature dynamics using RC network
          - Handles heat transfer between indoor air, walls, and adjacent zones
          - Includes HVAC thermal effects, solar gains, and occupant heat gains

       **Mass Balance Subsystem (BuildingSpaceMassSystem):**
          - Models CO2 concentration dynamics using mass balance equations
          - Handles ventilation, infiltration, and occupant CO2 generation
          - Tracks indoor air quality changes

    Implementation Details:

       - Both subsystems run in parallel during each simulation step
       - Input signals are shared between both models where applicable
       - Each subsystem maintains its own state variables and outputs
       - The combined model provides unified input/output interfaces
       - All parameters from both subsystems are available for calibration

    Combined Input/Output Interface:

       **Shared Inputs:**
          - supplyAirFlowRate: Used by both thermal (heating/cooling) and mass (ventilation)
          - exhaustAirFlowRate: Used by both thermal (heat removal) and mass (CO2 removal)
          - numberOfPeople: Used by both thermal (heat gain) and mass (CO2 generation)
          - outdoorTemperature: Used by thermal model
          - outdoorCo2Concentration: Used by mass balance model (400 ppmv while unwired;
            ``outdoorCO2`` is its deprecated name)

       **Thermal-Only Inputs:**
          - supplyAirTemperature, globalIrradiation, heatGain
          - wallHeatGain (heat flows from connected WallSystem components)

       **Combined Outputs:**
          - indoorTemperature: From thermal subsystem
          - wallTemperature: From thermal subsystem
          - surplusAirFlowRate, makeUpAirFlowRate: the air-flow imbalance of
            the summed branches (see
            :mod:`~twin4build.systems.building_space.air_balance`)
          - indoorCO2: From mass balance subsystem

    Shared parameters:

       The infiltration flow is one parameter, ``mass.m_inf``: the thermal
       subsystem's ``m_inf`` reads it (:attr:`SHARED_UNIT_PARAMS`), so the
       heat and the CO2 the infiltration carries come from the same flow.
    """

    #: Unit parameters that read another unit's parameter, as ``{path:
    #: source}`` relative to the component: the thermal infiltration is the
    #: CO2 model's.  Only the source is a parameter of the component; every
    #: execution path routes the path to it (the object path by pointing the
    #: path at the source's parameter object in :meth:`initialize`, the
    #: functional ones by key in :meth:`_resolve_sub_params` and
    #: ``FusedStateSpaceSystem``).
    SHARED_UNIT_PARAMS = {"thermal.m_inf": "mass.m_inf"}

    #: The old names of the children's values (removed in 2.1).
    DEPRECATED_KEYWORDS = {"thermal_kwargs": "thermal", "mass_kwargs": "mass"}
    REMOVED_ATTRIBUTES = {"boundaryTemperature": BOUNDARY_WALL_GONE}

    outdoorTemperature = Input(doc="Outdoor temperature [degC]")
    supplyAirFlowRate = Input(size="wired", doc="Supply air flow of each branch into the room [kg/s]")
    exhaustAirFlowRate = Input(size="wired", doc="Exhaust air flow of each branch out of the room [kg/s]")
    supplyAirTemperature = Input(doc="Supply air temperature [degC]")
    globalIrradiation = Input(doc="Solar irradiation [W/m2]")
    numberOfPeople = Input(doc="Number of occupants")
    heatGain = Input(doc="Space heater heat input [W]")
    makeUpAirTemperature = Input(
        default=outdoorTemperature, doc="Temperature of the make-up air [degC]; unwired: outdoor air"
    )
    wallHeatGain = Input(size="wired", doc="Heat flows from connected WallSystem components [W], one slot per wall")
    infiltrationAirFlowRate = Input(default=0.0, doc="Envelope inflow on top of the make-up flow [kg/s]")
    outdoorCo2Concentration = Input(default=400.0, alias="outdoorCO2", doc="Outdoor CO2 concentration [ppmv]")
    makeUpAirCO2 = Input(default=outdoorCo2Concentration, doc="CO2 of the make-up air [ppmv]; unwired: outdoor air")
    supplyAirCO2 = Input(default=outdoorCo2Concentration, doc="CO2 of the supply air [ppmv]; unwired: outdoor air")
    exchangeCO2Gain = Input(size="wired", doc="CO2 exchanged with other zones [kg/s], one slot per opening")
    indoorTemperature = Output(init=20.0, doc="Indoor air temperature [degC]")
    wallTemperature = Output(init=20.0, doc="Exterior wall temperature [degC]")
    surplusAirFlowRate = Output(doc="Supply in excess of the exhaust [kg/s]")
    makeUpAirFlowRate = Output(doc="Exhaust in excess of the supply, made up through the envelope [kg/s]")
    indoorCO2 = Output(init=400.0, doc="Indoor CO2 concentration [ppmv]")
    thermal = Child(BuildingSpaceThermalSystem)
    mass = Child(BuildingSpaceMassSystem)

    #: The branch flows, summed into the children's scalar flows.
    _FLOW_PORTS = ("supplyAirFlowRate", "exhaustAirFlowRate")

    OUTPUT_DEPENDENCIES = {
        "surplusAirFlowRate": {"inputs": ("supplyAirFlowRate", "exhaustAirFlowRate")},
        "makeUpAirFlowRate": {"inputs": ("supplyAirFlowRate", "exhaustAirFlowRate")},
    }
    #: The class that computes those state-free outputs on its own, from the same ports: a world that needs them
    #: without the room gets it in the room's place (:mod:`twin4build.model.signal_dependencies`).
    OUTPUT_PROJECTIONS = {"surplusAirFlowRate": "AirImbalanceSystem", "makeUpAirFlowRate": "AirImbalanceSystem"}

    FUSABLE_INPUT_PORTS = frozenset({"wallHeatGain", "exchangeCO2Gain", "heatGain"})

    FUSABLE_OUTPUT_PORTS = frozenset({"indoorTemperature", "indoorCO2"})

    def _ss_units(self):
        """State-space leaf units in state order (``thermal`` then ``mass`` --
        the order :meth:`System.get_state` concatenates)."""
        return [("thermal", self.thermal), ("mass", self.mass)]

    def _resolve_sub_params(self, sub, prefix, params):
        """Full physical-parameter dict for a submodel: estimated values from
        ``params`` (keyed ``"<prefix>.<name>"``, a shared one by its source's
        key, :attr:`SHARED_UNIT_PARAMS`), the rest from the ``tps.Parameter``
        defaults; the submodel's value settings come along."""
        out = sub.value_setting_params()
        for name in sub.PARAM_NAMES:
            key = f"{prefix}.{name}"
            key = self.SHARED_UNIT_PARAMS.get(key, key)
            out[name] = params[key] if key in params else rgetattr(self, key).get()
        return out

    def derive(self, p, step_size):
        """Each unit's parameters (estimated values from ``p``, the rest the
        unit's own) together with what the unit derives from them."""
        p_th = self._resolve_sub_params(self.thermal, "thermal", p)
        p_ma = self._resolve_sub_params(self.mass, "mass", p)
        return {
            "thermal_params": self.thermal.full_params(p_th, step_size),
            "mass_params": self.mass.full_params(p_ma, step_size),
        }

    def forward(self, x, inputs, params, step_size):
        """One step of the room: thermal ++ mass.

        State is ``[thermal_state | mass_state]`` (the order
        :meth:`System.get_state` produces).  ``params`` is keyed by the composite
        attr path (``"thermal.C_air"``, ``"mass.V"``, ...); it is routed to the two
        submodels as :meth:`derive` resolved them.  Both
        submodels read the shared ``inputs`` dict (they pick the ports they need).

        Returns ``(x_next, {**thermal_outputs, **mass_outputs})`` -- i.e.
        ``indoorTemperature``, ``wallTemperature``, ``indoorCO2``.
        """
        n_th = self.thermal.state_size()
        x_th, x_ma = x[..., :n_th], x[..., n_th:]
        # Vector flow ports carry one trailing branch axis more than the
        # scalar ports; sum it away for the submodels.
        ref_dim = inputs["outdoorTemperature"].dim()
        inputs = dict(inputs)
        for k in self._FLOW_PORTS:
            v = inputs.get(k)
            if v is None:  # no branch wired (the functional engine passes no empty vector): no flow
                inputs[k] = torch.zeros_like(inputs["outdoorTemperature"])
            elif v.dim() > ref_dim:
                inputs[k] = v.sum(dim=-1)
        p_th, p_ma = params["thermal_params"], params["mass_params"]
        x_th_n, out_th = self.thermal.forward(x_th, inputs, p_th, step_size)
        x_ma_n, out_ma = self.mass.forward(x_ma, inputs, p_ma, step_size)
        return torch.cat([x_th_n, x_ma_n], dim=-1), {**out_th, **out_ma}


# Deprecated aliases (removed in twin4build 2.1)
BuildingSpaceTorchSystem = BuildingSpaceSystem
