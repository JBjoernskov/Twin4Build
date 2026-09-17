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
translator groups them by that binding.  The factories live in the per-area modules of this package; this module
assembles and binds them.
"""

from typing import List

import twin4build.systems as _s

from twin4build.examples.patterns import air_handling_unit as _air_handling_unit_system
from twin4build.examples.patterns import air_to_air_heat_recovery as _air_to_air_heat_recovery_system
from twin4build.examples.patterns import building_space as _building_space_system
from twin4build.examples.patterns import building_space_thermal as _building_space_thermal_system
from twin4build.examples.patterns import controller_identification_pi as _controller_identification_pi_system
from twin4build.examples.patterns import on_off_controller as _on_off_controller_system
from twin4build.examples.patterns import pid_controller as _pid_controller_system
from twin4build.examples.patterns import damper as _damper_system
from twin4build.examples.patterns import fan_coil_unit as _fan_coil_unit_system
from twin4build.examples.patterns import return_flow_junction as _return_flow_junction_system
from twin4build.examples.patterns import supply_flow_junction as _supply_flow_junction_system
from twin4build.examples.patterns import outdoor_environment as _outdoor_environment_system
from twin4build.examples.patterns import schedule as _schedule_system
from twin4build.examples.patterns import sensor as _sensor_system
from twin4build.examples.patterns import space_heater as _space_heater_system
from twin4build.examples.patterns import valve as _valve_system


def _bind(system, sp):
    sp.bind(system)
    return sp


def air_handling_unit_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.AirHandlingUnitSystem`."""
    out = [
        _bind(_s.AirHandlingUnitSystem, _air_handling_unit_system.brick_signature_pattern_vav_dampers()),
        _bind(_s.AirHandlingUnitSystem, _air_handling_unit_system.brick_signature_pattern_vav_damper_commands()),
    ]
    return out


def air_to_air_heat_recovery_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.AirToAirHeatRecoverySystem`."""
    out = [
        _bind(_s.AirToAirHeatRecoverySystem, _air_to_air_heat_recovery_system.saref_signature_pattern()),
    ]
    return out


def building_space_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.BuildingSpaceSystem`."""
    out = [
        _bind(_s.BuildingSpaceSystem, _building_space_system.saref_signature_pattern()),
        _bind(_s.BuildingSpaceSystem, _building_space_system.saref_signature_pattern_sensor()),
    ]
    for with_volume in (True, False):
        for heat_source in ("command", "equipment"):
            out.append(_bind(_s.BuildingSpaceSystem, _building_space_system.brick_signature_pattern_vav_no_reheat(with_volume, heat_source)))
            out.append(_bind(_s.BuildingSpaceSystem, _building_space_system.brick_signature_pattern_vav(with_volume, heat_source)))
            out.append(_bind(_s.BuildingSpaceSystem, _building_space_system.brick_signature_pattern(with_volume, heat_source)))
    return out


def building_space_thermal_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.BuildingSpaceThermalSystem`."""
    out = [
        _bind(_s.BuildingSpaceThermalSystem, _building_space_thermal_system.brick_signature_pattern()),
    ]
    return out


def controller_identification_pi_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.ControllerIdentificationPISystem`."""
    out = [
        _bind(_s.ControllerIdentificationPISystem, _controller_identification_pi_system.brick_signature_pattern_vav()),
        _bind(_s.ControllerIdentificationPISystem, _controller_identification_pi_system.brick_signature_pattern_vav_room()),
        _bind(_s.ControllerIdentificationPISystem, _controller_identification_pi_system.brick_signature_pattern_space_heater_room()),
        _bind(_s.ControllerIdentificationPISystem, _controller_identification_pi_system.brick_signature_pattern_vav_damper()),
    ]
    return out


def damper_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.DamperSystem`."""
    out = [
        _bind(_s.DamperSystem, _damper_system.brick_signature_pattern()),
        _bind(_s.DamperSystem, _damper_system.saref_signature_pattern()),
    ]
    return out


def fan_coil_unit_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.FanCoilUnitSystem`."""
    out = [
        _bind(_s.FanCoilUnitSystem, _fan_coil_unit_system.brick_signature_pattern()),
        _bind(_s.FanCoilUnitSystem, _fan_coil_unit_system.brick_signature_pattern_vav_ahu()),
    ]
    return out


def on_off_controller_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.OnOffControllerSystem`."""
    out = [
        _bind(_s.OnOffControllerSystem, _on_off_controller_system.brick_signature_pattern()),
        _bind(_s.OnOffControllerSystem, _on_off_controller_system.saref_signature_pattern()),
    ]
    return out


def outdoor_environment_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.OutdoorEnvironmentSystem`."""
    out = [
        _bind(_s.OutdoorEnvironmentSystem, _outdoor_environment_system.brick_signature_pattern()),
        _bind(_s.OutdoorEnvironmentSystem, _outdoor_environment_system.brick_signature_pattern_standalone()),
        _bind(_s.OutdoorEnvironmentSystem, _outdoor_environment_system.saref_signature_pattern()),
    ]
    return out


def pid_controller_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.PIDControllerSystem`."""
    out = [
        _bind(_s.PIDControllerSystem, _pid_controller_system.saref_signature_pattern()),
    ]
    return out


def return_flow_junction_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.ReturnFlowJunctionSystem`."""
    out = [
        _bind(_s.ReturnFlowJunctionSystem, _return_flow_junction_system.brick_signature_pattern()),
        _bind(_s.ReturnFlowJunctionSystem, _return_flow_junction_system.saref_signature_pattern()),
    ]
    return out


def schedule_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.ScheduleSystem`."""
    out = [
        _bind(_s.ScheduleSystem, _schedule_system.brick_signature_pattern()),
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
        _bind(_s.SensorSystem, _sensor_system.get_flow_signature_pattern_after_coil_air_side()),
        _bind(_s.SensorSystem, _sensor_system.get_flow_signature_pattern_after_coil_water_side()),
        _bind(_s.SensorSystem, _sensor_system.get_flow_signature_pattern_before_coil_water_side()),
        _bind(_s.SensorSystem, _sensor_system.get_space_temperature_signature_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_space_co2_signature_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_position_signature_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_command_sensor_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_damper_command_sensor_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_zone_air_temp_sensor_with_ref_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_zone_air_temp_sensor_virtual_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_room_zone_air_temp_sensor_with_ref_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_room_zone_air_temp_sensor_virtual_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_room_zone_co2_sensor_with_ref_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_ahu_supply_air_temp_sensor_with_ref_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_ahu_supply_air_temp_sensor_virtual_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_supply_air_flow_sensor_with_ref_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_supply_air_flow_sensor_virtual_pattern()),
        _bind(_s.SensorSystem, _sensor_system.get_brick_sensor_leaf_pattern()),
    ]
    return out


def space_heater_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.SpaceHeaterSystem`."""
    out = [
        _bind(_s.SpaceHeaterSystem, _space_heater_system.brick_signature_pattern_space_heater_valve()),
        _bind(_s.SpaceHeaterSystem, _space_heater_system.brick_signature_pattern_room_heating_command()),
        _bind(_s.SpaceHeaterSystem, _space_heater_system.brick_signature_pattern()),
        _bind(_s.SpaceHeaterSystem, _space_heater_system.saref_signature_pattern()),
    ]
    return out


def supply_flow_junction_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.SupplyFlowJunctionSystem`."""
    out = [
        _bind(_s.SupplyFlowJunctionSystem, _supply_flow_junction_system.brick_signature_pattern()),
        _bind(_s.SupplyFlowJunctionSystem, _supply_flow_junction_system.saref_signature_pattern()),
    ]
    return out


def valve_patterns() -> List:
    """Patterns bound to :class:`~twin4build.systems.ValveSystem`."""
    out = [
        _bind(_s.ValveSystem, _valve_system.brick_signature_pattern()),
        _bind(_s.ValveSystem, _valve_system.saref_signature_pattern()),
    ]
    return out


def default_patterns() -> List:
    """Every example pattern, bound: what ``translate(patterns=None)`` used
    to do implicitly.  Compose your own list from the per-class helpers
    above and your own patterns instead of relying on this."""
    out = []
    out += air_handling_unit_patterns()
    out += air_to_air_heat_recovery_patterns()
    out += building_space_patterns()
    out += building_space_thermal_patterns()
    out += controller_identification_pi_patterns()
    out += damper_patterns()
    out += fan_coil_unit_patterns()
    out += on_off_controller_patterns()
    out += outdoor_environment_patterns()
    out += pid_controller_patterns()
    out += return_flow_junction_patterns()
    out += schedule_patterns()
    out += sensor_patterns()
    out += space_heater_patterns()
    out += supply_flow_junction_patterns()
    out += valve_patterns()
    return out


__all__ = ["default_patterns"] + [
    "air_handling_unit_patterns", "air_to_air_heat_recovery_patterns", "building_space_patterns", "building_space_thermal_patterns", "controller_identification_pi_patterns", "damper_patterns", "fan_coil_unit_patterns", "on_off_controller_patterns", "outdoor_environment_patterns", "pid_controller_patterns", "return_flow_junction_patterns", "schedule_patterns", "sensor_patterns", "space_heater_patterns", "supply_flow_junction_patterns", "valve_patterns"
]
