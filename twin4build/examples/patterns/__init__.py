"""The public example set of signature patterns (``twin4build.examples.patterns``).

Signature patterns say how a graph shape maps onto a component; they are
*examples of how to express a graph shape*, not a standard.  How a BMS
names and links its points differs per vendor, per integrator and per
building, and a real deployment writes its own patterns and passes them::

    model = tb.Translator().translate(
        semantic_model,
        patterns=example_patterns.default_patterns() + my_patterns,
    )

Every pattern is bound to the ``System`` class it models
(``SignaturePattern(..., system=cls)`` or ``sp.bind(cls)``); the
translator groups them by that binding.  The factories live in the per-area
modules of this package; this module assembles and binds them.

The set is minimal on purpose: exactly the patterns the library's own
translator example (``twin4build/examples/translator_example.py``) matches on
its one-room model.  Anything a real building needs beyond that is
deployment-specific and belongs next to the deployment.
"""

from typing import List

import twin4build.systems as _s

from twin4build.examples.patterns import building_space as _building_space_system
from twin4build.examples.patterns import pid_controller as _pid_controller_system
from twin4build.examples.patterns import damper as _damper_system
from twin4build.examples.patterns import outdoor_environment as _outdoor_environment_system
from twin4build.examples.patterns import schedule as _schedule_system
from twin4build.examples.patterns import sensor as _sensor_system
from twin4build.examples.patterns import space_heater as _space_heater_system
from twin4build.examples.patterns import valve as _valve_system


def _bind(system, sp):
    sp.bind(system)
    return sp


def building_space_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.BuildingSpaceSystem`."""
    out = [
        _bind(_s.BuildingSpaceSystem, _building_space_system.saref_signature_pattern()),
        _bind(_s.BuildingSpaceSystem, _building_space_system.saref_signature_pattern_sensor()),
    ]
    return out


def damper_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.DamperSystem`."""
    out = [
        _bind(_s.DamperSystem, _damper_system.saref_signature_pattern()),
    ]
    return out


def outdoor_environment_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.OutdoorEnvironmentSystem`."""
    out = [
        _bind(_s.OutdoorEnvironmentSystem, _outdoor_environment_system.saref_signature_pattern()),
    ]
    return out


def pid_controller_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.PIDControllerSystem`."""
    out = [
        _bind(_s.PIDControllerSystem, _pid_controller_system.saref_signature_pattern()),
    ]
    return out


def schedule_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.ScheduleSystem`."""
    out = [
        _bind(_s.ScheduleSystem, _schedule_system.saref_signature_pattern()),
    ]
    return out


def sensor_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.SensorSystem`."""
    out = [
        _bind(_s.SensorSystem, _sensor_system.get_temperature_before_air_to_air_supply_side()),
        _bind(_s.SensorSystem, _sensor_system.get_temperature_before_air_to_air_exhaust_side()),
        _bind(_s.SensorSystem, _sensor_system.get_temperature_after_air_to_air_supply_side()),
        _bind(_s.SensorSystem, _sensor_system.get_temperature_after_air_to_air_exhaust_side()),
        _bind(_s.SensorSystem, _sensor_system.get_signature_pattern_input()),
        _bind(_s.SensorSystem, _sensor_system.get_space_temperature_signature_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_space_co2_signature_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_position_signature_pattern()),
    ]
    return out


def space_heater_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.SpaceHeaterSystem`."""
    out = [
        _bind(_s.SpaceHeaterSystem, _space_heater_system.saref_signature_pattern()),
    ]
    return out


def valve_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.ValveSystem`."""
    out = [
        _bind(_s.ValveSystem, _valve_system.saref_signature_pattern()),
    ]
    return out


def default_patterns() -> List:
    """Every example pattern, bound: the set the library's own translator
    example matches.  It is deliberately small; compose your own list from
    the per-class helpers above and your own patterns."""
    out = []
    out += building_space_patterns()
    out += damper_patterns()
    out += outdoor_environment_patterns()
    out += pid_controller_patterns()
    out += schedule_patterns()
    out += sensor_patterns()
    out += space_heater_patterns()
    out += valve_patterns()
    return out


__all__ = [
    "default_patterns",
    "building_space_patterns",
    "damper_patterns",
    "outdoor_environment_patterns",
    "pid_controller_patterns",
    "schedule_patterns",
    "sensor_patterns",
    "space_heater_patterns",
    "valve_patterns",
]
