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

# Standard library imports
import datetime

# Third party imports
import torch
import torch.nn as nn  # noqa: F401 - torch needed for tensor ops

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.coil.coil_system import CoilSystem
from twin4build.systems.fan.fan_system import FanSystem


#: The device's submodels, in the order the chain steps them.
CORE_SUB_NAMES = ("coil", "heat_recovery", "supply_fan", "exhaust_fan")


def resolve_sub_params(sub, prefix, params):
    """Full physical-parameter dict for a submodel: estimated values from
    ``params`` (keyed ``"<prefix>.<name>"``), the rest from the submodel's
    own ``tps.Parameter`` defaults."""
    out = {}
    for name in sub.PARAM_NAMES:
        key = f"{prefix}.{name}"
        out[name] = params[key] if key in params else getattr(sub, name).get()
    return out


def unit_chain(
    subs,
    P,
    supply_flow_total,
    secondary_flow,
    return_temp,
    outdoor_temperature,
    supply_setpoint,
    sample_time,
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
        sample_time,
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
        sample_time,
    )
    _, coil = subs["coil"].forward(
        None,
        {
            "inletAirTemperature": hr["primaryTemperatureOut"],
            "outletAirTemperatureSetpoint": supply_setpoint,
            "airFlowRate": supply_flow_total,
        },
        P["coil"],
        sample_time,
    )
    _, f_sup = subs["supply_fan"].forward(
        None,
        {
            "airFlowRate": supply_flow_total,
            "inletAirTemperature": coil["outletAirTemperature"],
        },
        P["supply_fan"],
        sample_time,
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


class AirHandlingUnitCoreSystem(core.System, nn.Module):
    r"""
    An air handling unit on its totals: the device without the terminals.

    The supply flow, the exhaust flow and the mixed return temperature come in
    on scalar ports (from a :class:`SupplyFlowJunctionSystem` and a
    :class:`ReturnFlowJunctionSystem` fed by the terminals' dampers); the unit
    runs the exhaust fan, the heat recovery, the coil and the supply fan and
    reports the supply temperature, the preheat temperature and the powers.
    The composite :class:`AirHandlingUnitSystem` computes the same device on
    the branch flows of its internal dampers; the two agree step for step.

    Inputs
        - ``totalSupplyAirFlowRate`` [kg/s]
        - ``totalExhaustAirFlowRate`` [kg/s]
        - ``returnAirTemperature`` [°C], the flow-weighted mix of the exhausts
        - ``supplyAirTemperatureSetpoint`` [°C]
        - ``outdoorAirTemperature`` [°C]

    Outputs
        - ``supplyAirTemperature``, ``preheatSupplyAirTemperature``,
          ``exhaustAirTemperatureOut`` [°C]
        - ``totalSupplyAirFlowRate``, ``totalExhaustAirFlowRate`` [kg/s]
          (the inputs, passed through for the unit's flow meters)
        - ``heatingPower``, ``coolingPower``, ``supplyFanPower``,
          ``exhaustFanPower`` [W]

    Parameters
        The coil's, the heat recovery's and the fans' own, prefixed
        (``"coil.<name>"``, ``"heat_recovery.<name>"``, ``"supply_fan.<name>"``,
        ``"exhaust_fan.<name>"``).
    """

    SUPPORTS_TRANSFORM_MODE = True
    PARAM_NAMES = ()  # all parameters live on the owned submodels (prefixed)
    _SUB_NAMES = CORE_SUB_NAMES

    def __init__(
        self,
        coil_kwargs: dict | None = None,
        heat_recovery_kwargs: dict | None = None,
        supply_fan_kwargs: dict | None = None,
        exhaust_fan_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Args:
            coil_kwargs: Keyword arguments for :class:`CoilSystem`.
            heat_recovery_kwargs: Keyword arguments for
                :class:`AirToAirHeatRecoverySystem`.
            supply_fan_kwargs: Keyword arguments for the supply :class:`FanSystem`.
            exhaust_fan_kwargs: Keyword arguments for the exhaust :class:`FanSystem`.
            **kwargs: Passed to the System base class (``id`` is required).
        """
        coil_kwargs = dict(coil_kwargs or {})
        heat_recovery_kwargs = dict(heat_recovery_kwargs or {})
        supply_fan_kwargs = dict(supply_fan_kwargs or {})
        exhaust_fan_kwargs = dict(exhaust_fan_kwargs or {})
        assert "id" in kwargs, "id is required for AirHandlingUnitCoreSystem"
        unit_id = kwargs["id"]
        coil_kwargs.setdefault("id", f"{unit_id}_coil")
        heat_recovery_kwargs.setdefault("id", f"{unit_id}_heat_recovery")
        supply_fan_kwargs.setdefault("id", f"{unit_id}_supply_fan")
        exhaust_fan_kwargs.setdefault("id", f"{unit_id}_exhaust_fan")
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.coil = CoilSystem(**coil_kwargs)
        self.heat_recovery = AirToAirHeatRecoverySystem(**heat_recovery_kwargs)
        self.supply_fan = FanSystem(**supply_fan_kwargs)
        self.exhaust_fan = FanSystem(**exhaust_fan_kwargs)

        self._input = {
            "totalSupplyAirFlowRate": tps.Scalar(),
            "totalExhaustAirFlowRate": tps.Scalar(),
            "returnAirTemperature": tps.Scalar(),
            "supplyAirTemperatureSetpoint": tps.Scalar(),
            "outdoorAirTemperature": tps.Scalar(),
        }
        self._output = {
            "supplyAirTemperature": tps.Scalar(),
            "preheatSupplyAirTemperature": tps.Scalar(),
            "exhaustAirTemperatureOut": tps.Scalar(),
            "totalSupplyAirFlowRate": tps.Scalar(),
            "totalExhaustAirFlowRate": tps.Scalar(),
            "heatingPower": tps.Scalar(),
            "coolingPower": tps.Scalar(),
            "supplyFanPower": tps.Scalar(),
            "exhaustFanPower": tps.Scalar(),
        }
        self._config = {
            "parameters": [
                f"{sub_name}.{p}"
                for sub_name in self._SUB_NAMES
                for p in getattr(self, sub_name)._config["parameters"]
            ]
        }
        self.PARAM_NAMES = tuple(
            f"{sub_name}.{param_name}"
            for sub_name in self._SUB_NAMES
            for param_name in getattr(getattr(self, sub_name), "PARAM_NAMES", ())
        )
        self.INITIALIZED = False

    @property
    def input(self) -> dict:
        return self._input

    @property
    def output(self) -> dict:
        return self._output

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        start_time: list[datetime.datetime],
        end_time: list[datetime.datetime],
        step_size: int,
    ) -> None:
        """Initialize the ports and the submodels."""
        _, _, max_timesteps, _ = core.Simulator.get_simulation_timesteps(
            start_time, end_time, step_size
        )
        batch_size = len(start_time)
        for port in self.input.values():
            port.initialize(n_t=max_timesteps, n_s=batch_size)
        for port in self.output.values():
            port.initialize(n_t=max_timesteps, n_s=batch_size)
        for sub_name in self._SUB_NAMES:
            getattr(self, sub_name).initialize(start_time, end_time, step_size)
        self.INITIALIZED = True

    def _sub_params(self, params, transform_mode=None):
        """The submodels' resolved parameter dicts, identity-cached on
        ``params`` (a sequential rollout re-calls ``forward`` with the same
        dict every step)."""
        if transform_mode:
            return {
                n: resolve_sub_params(getattr(self, n), n, params)
                for n in self._SUB_NAMES
            }
        cache = getattr(self, "_fwd_param_cache", None)
        if cache is None or cache[0] is not params:
            cache = (
                params,
                {
                    n: resolve_sub_params(getattr(self, n), n, params)
                    for n in self._SUB_NAMES
                },
            )
            self._fwd_param_cache = cache
        return cache[1]

    def forward(self, x, inputs, params, sample_time, transform_mode=None):
        """Pure one-step of the device (functorch-safe, stateless): the
        :func:`unit_chain` on the scalar inputs.  ``params`` is keyed by the
        composite attr path (``"coil.<name>"``, ...); non-estimated entries
        fall back to the submodels' defaults."""
        P = self._sub_params(params, transform_mode)
        subs = {n: getattr(self, n) for n in self._SUB_NAMES}
        outs = unit_chain(
            subs,
            P,
            inputs["totalSupplyAirFlowRate"],
            inputs["totalExhaustAirFlowRate"],
            inputs["returnAirTemperature"],
            inputs["outdoorAirTemperature"],
            inputs["supplyAirTemperatureSetpoint"],
            sample_time,
        )
        return x, outs

    def do_step(
        self,
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """Thin port-I/O wrapper around :meth:`forward`."""
        inputs = {name: port.get() for name, port in self.input.items()}
        _, outs = self.forward(
            None, inputs, self._forward_params(), self._scalar_sample_time(step_size)
        )
        for name, value in outs.items():
            self.output[name]._set(value, i_t=step_index)
