"""The air handling unit without its terminals.

:class:`AirHandlingUnitCoreSystem` is the unit as a scalar device: heat
recovery, coil and the two fans, driven by the *totals* -- supply flow,
exhaust flow and the mixed return temperature -- on scalar ports.  The
terminals are their own components (:class:`DamperSystem`, one per VAV box,
gated by the fan and carrying its exhaust flow), and the duct network is the
two junctions (:class:`SupplyFlowJunctionSystem` sums the branch flows,
:class:`ReturnFlowJunctionSystem` mixes the branch exhausts)::

    command_i --> damper_i --airFlowRate--> room_i, supply junction[i]
                           --exhaustAirFlowRate--> return junction[i]
    room_i --indoorTemperature--> return junction[i]
    supply junction --> core.totalSupplyAirFlowRate
    return junction --> core.totalExhaustAirFlowRate, core.returnAirTemperature
    core --supplyAirTemperature--> every room

Modelled this way a room's block in the estimation problem is bounded by
measured signals (its commands, the weather, the unit's supply temperature)
and the coupling structure the estimator derives from the wiring is one
block per room, plus one for the unit.  :class:`AirHandlingUnitSystem` keeps
the composite form (dampers and junctions inside, vector ports per branch)
and shares this module's :func:`unit_chain` for the device itself, so the
two wirings give the same supply temperature and powers.
"""

from __future__ import annotations

# Local application imports
import twin4build.core as core
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.coil.coil_system import CoilSystem
from twin4build.systems.fan.fan_system import FanSystem
from twin4build.systems.saref4syst.declarations import Child, Input, Output


#: The device's submodels, in the order the chain steps them.
CORE_SUB_NAMES = ("coil", "heat_recovery", "supply_fan", "exhaust_fan")


def resolve_sub_params(sub, prefix, params):
    """Full physical-parameter dict for a submodel: estimated values from
    ``params`` (keyed ``"<prefix>.<name>"``), the rest from the submodel's
    own values; an absent (optional) parameter stays out.  The submodel's
    value settings come along (its ``forward`` reads them from ``p``)."""
    out = sub.value_setting_params() if hasattr(sub, "value_setting_params") else {}
    for name in sub.PARAM_NAMES:
        key = f"{prefix}.{name}"
        if key in params:
            out[name] = params[key]
        elif getattr(sub, name) is not None:  # an absent (optional) parameter stays out
            out[name] = getattr(sub, name).get()
    return out


def unit_chain(
    subs,
    P,
    supply_flow_total,
    secondary_flow,
    return_temp,
    outdoor_temperature,
    supply_setpoint,
    step_size,
):
    """One step of the device on its totals (functorch-safe, stateless).

    Exhaust fan on the return stream, heat recovery between the return and
    the outdoor air, coil trimming to the setpoint, supply fan after the coil.
    ``subs`` maps :data:`CORE_SUB_NAMES` to the submodels and ``P`` to their
    resolved parameter dicts.  Returns the device's nine outputs.
    """
    _, f_exh = subs["exhaust_fan"].forward(
        None,
        {"airFlowRate": secondary_flow, "inletAirTemperature": return_temp},
        P["exhaust_fan"],
        step_size,
    )
    _, hr = subs["heat_recovery"].forward(
        None,
        {
            "primaryAirFlowRate": supply_flow_total,
            "secondaryAirFlowRate": secondary_flow,
            "primaryTemperatureIn": outdoor_temperature,
            "secondaryTemperatureIn": f_exh["outletAirTemperature"],
            "primaryTemperatureOutSetpoint": supply_setpoint,
        },
        P["heat_recovery"],
        step_size,
    )
    _, coil = subs["coil"].forward(
        None,
        {
            "inletAirTemperature": hr["primaryTemperatureOut"],
            "outletAirTemperatureSetpoint": supply_setpoint,
            "airFlowRate": supply_flow_total,
        },
        P["coil"],
        step_size,
    )
    _, f_sup = subs["supply_fan"].forward(
        None,
        {
            "airFlowRate": supply_flow_total,
            "inletAirTemperature": coil["outletAirTemperature"],
        },
        P["supply_fan"],
        step_size,
    )
    return {
        "totalSupplyAirFlowRate": supply_flow_total,
        "totalExhaustAirFlowRate": secondary_flow,
        "supplyAirTemperature": f_sup["outletAirTemperature"],
        "preheatSupplyAirTemperature": hr["primaryTemperatureOut"],
        "exhaustAirTemperatureOut": hr["secondaryTemperatureOut"],
        "heatingPower": coil["heatingPower"],
        "coolingPower": coil["coolingPower"],
        "supplyFanPower": f_sup["Power"],
        "exhaustFanPower": f_exh["Power"],
    }


class AirHandlingUnitCoreSystem(core.System):
    r"""
    An air handling unit on its totals: the device without the terminals.

    The supply flow, the exhaust flow and the mixed return temperature come in
    on scalar ports (from a :class:`SupplyFlowJunctionSystem` and a
    :class:`ReturnFlowJunctionSystem` fed by the terminals' dampers); the unit
    runs the exhaust fan, the heat recovery, the coil and the supply fan and
    reports the supply temperature, the preheat temperature and the powers.
    The composite :class:`AirHandlingUnitSystem` computes the same device on
    the branch flows of its internal dampers; the two agree step for step.

    Its parameters are the coil's, the heat recovery's and the fans' own,
    prefixed (``"coil.<name>"``, ``"heat_recovery.<name>"``,
    ``"supply_fan.<name>"``, ``"exhaust_fan.<name>"``).  A child's values are
    given as a dict: ``AirHandlingUnitCoreSystem(id="unit",
    supply_fan=dict(beta3=2000.0))``.
    """

    #: The old names of the children's constructor values (removed in 2.1).
    DEPRECATED_KEYWORDS = {
        "coil_kwargs": "coil",
        "heat_recovery_kwargs": "heat_recovery",
        "supply_fan_kwargs": "supply_fan",
        "exhaust_fan_kwargs": "exhaust_fan",
    }

    totalSupplyAirFlowRate = Input(doc="The supply flow [kg/s]")
    totalExhaustAirFlowRate = Input(doc="The exhaust flow [kg/s]")
    returnAirTemperature = Input(doc="The flow-weighted mix of the exhausts [degC]")
    supplyAirTemperatureSetpoint = Input(doc="Supply air temperature setpoint [degC]")
    outdoorAirTemperature = Input(doc="Outdoor air temperature [degC]")
    supplyAirTemperature = Output(doc="Supply air temperature, after the supply fan [degC]")
    preheatSupplyAirTemperature = Output(doc="Supply air temperature after the heat recovery, before the coil [degC]")
    exhaustAirTemperatureOut = Output(doc="Exhaust air temperature after the heat recovery [degC]")
    totalSupplyAirFlowRate = Output(doc="The supply flow, passed through for the unit's flow meter [kg/s]")
    totalExhaustAirFlowRate = Output(doc="The exhaust flow, passed through for the unit's flow meter [kg/s]")
    heatingPower = Output(doc="Coil heating power [W]")
    coolingPower = Output(doc="Coil cooling power [W]")
    supplyFanPower = Output(doc="Supply fan power [W]")
    exhaustFanPower = Output(doc="Exhaust fan power [W]")
    coil = Child(CoilSystem)
    heat_recovery = Child(AirToAirHeatRecoverySystem)
    supply_fan = Child(FanSystem)
    exhaust_fan = Child(FanSystem)

    def derive(self, p, step_size):
        """The submodels' parameters (estimated values from ``p``, the rest the
        submodels' own), resolved once per run or rollout."""
        return {
            "sub_params": {
                n: getattr(self, n).full_params(resolve_sub_params(getattr(self, n), n, p), step_size)
                for n in CORE_SUB_NAMES
            }
        }

    def forward(self, x, inputs, params, step_size):
        """Stateless: the :func:`unit_chain` on the totals."""
        P = params["sub_params"]
        subs = {n: getattr(self, n) for n in CORE_SUB_NAMES}
        outs = unit_chain(
            subs,
            P,
            inputs["totalSupplyAirFlowRate"],
            inputs["totalExhaustAirFlowRate"],
            inputs["returnAirTemperature"],
            inputs["outdoorAirTemperature"],
            inputs["supplyAirTemperatureSetpoint"],
            step_size,
        )
        return x, outs
