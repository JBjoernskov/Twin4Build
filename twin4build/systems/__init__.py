"""Systems module for Twin4Build package.

This module provides a comprehensive collection of building system components that can be used to model
and simulate building systems. Each system is implemented as a PyTorch-based component for efficient
simulation and optimization.

Key Components:
    Building Spaces:
        - BuildingSpaceSystem: Combined thermal + CO2 (mass balance) building space model
        - BuildingSpaceMassSystem: Building space CO2 mass balance model
        - BuildingSpaceThermalSystem: Building space thermal (RC) model

    Building Envelope:
        - WallSystem: 2R1C wall between two zones (or zone and boundary)

    HVAC Components:
        - SpaceHeaterSystem: Space heating system
        - FanCoilUnitSystem: Fan coil unit (heating/cooling terminal unit)
        - ValveSystem: Control valve system
        - CoilSystem: Heating/cooling coil system
        - DamperSystem: Air flow control damper
        - FanSystem: Air handling fan system
        - AirToAirHeatRecoverySystem: Heat recovery system
        - AirHandlingUnitSystem: Air handling unit system

    Control Systems:
        - PIDControllerSystem: Proportional-Integral-Derivative controller
        - CascadeControllerSystem: Cascade (outer/inner loop) controller
        - OnOffControllerSystem: Threshold-based on/off controller
        - SmoothOnOffControllerSystem: Torch-based on/off controller
        - ScheduleSwitchControllerSystem: Schedule-based switching controller
        - SATLinearRuleSystem / SATCompensatedControllerSystem: Supply air
          temperature compensation rules
        - ClassificationAnnControllerSystem: ANN-based classification control
        - NeuralPolicyControllerSystem: Neural network policy control
        - ControllerIdentificationSystem / ControllerIdentificationPISystem:
          Controller identification models

    Monitoring & Measurement:
        - SensorSystem: Generic sensor system
        - ScheduleSystem: Time-based scheduling system
        - PiecewiseLinearScheduleSystem: Schedule with piecewise linear interpolation

    Environmental:
        - OutdoorEnvironmentSystem: External environmental conditions
        - ShadingDeviceSystem: Solar shading control

    Utility Systems:
        - FmuSystem: FMU-based system
        - SupplyFlowJunctionSystem: Supply flow distribution
        - ReturnFlowJunctionSystem: Return flow collection
        - PiecewiseLinearSystem: Piecewise linear interpolation
        - TimeSeriesInputSystem: Time series data input
        - MaxSystem: Maximum value selection
        - OnOffSystem: Binary state system
        - DiscreteStatespaceSystem: General-purpose discrete state-space model
        - ScalarProductSystem: Elementwise scalar product
        - FunctionSystem: User-supplied transformation of named inputs
        - OccupancySystem / OccupancyDetectorSystem: Occupancy modeling and detection
        - SigmoidGate: Smooth gating function

Note:
    Most systems are implemented using PyTorch for efficient computation and
    optimization. FMUs can still be wrapped via FmuSystem.
"""

import importlib

# Define what gets exported with wildcard imports
__all__ = [
    # Building Spaces
    "BuildingSpaceSystem",
    "BuildingSpaceMassSystem",
    "BuildingSpaceThermalSystem",
    # Wall
    "WallSystem",
    # Space Heater
    "SpaceHeaterSystem",
    # Valves
    "ValveSystem",
    # Coils
    "CoilSystem",
    # Fan Coil Unit
    "FanCoilUnitSystem",
    # Controllers
    "PIDControllerSystem",
    "CascadeControllerSystem",
    "OnOffControllerSystem",
    "SmoothOnOffControllerSystem",
    "ScheduleSwitchControllerSystem",
    "SATLinearRuleSystem",
    "SATCompensatedControllerSystem",
    "ClassificationAnnControllerSystem",
    "NeuralPolicyControllerSystem",
    "ControllerIdentificationSystem",
    "ControllerIdentificationPISystem",
    "RewireReport",
    "LoopScore",
    "ActuatorSeeds",
    "score_pair",
    "derive_actuator_seeds",
    "confidence_label",
    # Sensors
    "SensorSystem",
    # Schedules
    "ScheduleSystem",
    "PiecewiseLinearScheduleSystem",
    # Outdoor Environment
    "OutdoorEnvironmentSystem",
    # Junction
    "SupplyFlowJunctionSystem",
    "ReturnFlowJunctionSystem",
    # Air to Air Heat Recovery
    "AirToAirHeatRecoverySystem",
    # Air Handling Unit
    "AirHandlingUnitSystem",
    # Damper
    "DamperSystem",
    # Fan
    "FanSystem",
    # Shading
    "ShadingDeviceSystem",
    # Utils
    "FmuSystem",
    "fmuSystem",  # deprecated alias until 2.1
    "PiecewiseLinearSystem",
    "TimeSeriesInputSystem",
    "MaxSystem",
    "OnOffSystem",
    "DiscreteStatespaceSystem",
    "ScalarProductSystem",
    "FunctionSystem",
    "OccupancySystem",
    "OccupancyDetectorSystem",
    "SigmoidGate",
]

_MODULES = {
    "AirHandlingUnitSystem": "air_handling_unit.air_handling_unit_system",
    "AirToAirHeatRecoverySystem": "air_to_air_heat_recovery.air_to_air_heat_recovery_system",
    "BuildingSpaceMassSystem": "building_space.building_space_mass_system",
    "BuildingSpaceThermalSystem": "building_space.building_space_thermal_system",
    "BuildingSpaceSystem": "building_space.building_space_system",
    "CoilSystem": "coil.coil_system",
    "FanCoilUnitSystem": "fan_coil_unit.fan_coil_unit_system",
    "ClassificationAnnControllerSystem": "controller.classification_ann_controller.classification_ann_controller_system",
    "ControllerIdentificationSystem": "controller.controller_identification.controller_identification_system",
    "ControllerIdentificationPISystem": "controller.controller_identification.controller_identification_pi_system",
    "ActuatorSeeds": "controller.controller_identification.loop_classifier",
    "LoopScore": "controller.controller_identification.loop_classifier",
    "confidence_label": "controller.controller_identification.loop_classifier",
    "derive_actuator_seeds": "controller.controller_identification.loop_classifier",
    "score_pair": "controller.controller_identification.loop_classifier",
    "RewireReport": "controller.controller_identification.pi_loop_rewire",
    "NeuralPolicyControllerSystem": "controller.neural_policy_controller.neural_policy_controller_system",
    "OnOffControllerSystem": "controller.rulebased_controller.on_off_controller.on_off_controller_system",
    "SmoothOnOffControllerSystem": "controller.rulebased_controller.on_off_controller.smooth_on_off_controller_system",
    "SATCompensatedControllerSystem": "controller.rulebased_controller.sat_compensated_controller.sat_compensated_controller_system",
    "SATLinearRuleSystem": "controller.rulebased_controller.sat_compensated_controller.sat_compensated_controller_system",
    "ScheduleSwitchControllerSystem": "controller.rulebased_controller.schedule_switch_controller.schedule_switch_controller_system",
    "CascadeControllerSystem": "controller.setpoint_controller.cascade_controller.cascade_controller_system",
    "PIDControllerSystem": "controller.setpoint_controller.pid_controller.pid_controller_system",
    "DamperSystem": "damper.damper_system",
    "FanSystem": "fan.fan_system",
    "ReturnFlowJunctionSystem": "junction.return_flow_junction_system",
    "SupplyFlowJunctionSystem": "junction.supply_flow_junction_system",
    "OutdoorEnvironmentSystem": "outdoor_environment.outdoor_environment_system",
    "PiecewiseLinearScheduleSystem": "schedule.piecewise_linear_schedule_system",
    "ScheduleSystem": "schedule.schedule_system",
    "SensorSystem": "sensor.sensor_system",
    "ShadingDeviceSystem": "shading_device.shading_device_system",
    "SpaceHeaterSystem": "space_heater.space_heater_system",
    "DiscreteStatespaceSystem": "utils.discrete_statespace_system",
    "FmuSystem": "utils.fmu_system",
    "MaxSystem": "utils.max_system",
    "OccupancyDetectorSystem": "utils.occupancy_detector_system",
    "SigmoidGate": "utils.sigmoid_gate",
    "OccupancySystem": "utils.occupancy_system",
    "OnOffSystem": "utils.on_off_system",
    "PiecewiseLinearSystem": "utils.piecewise_linear_system",
    "FunctionSystem": "utils.function_system",
    "ScalarProductSystem": "utils.scalar_product_system",
    "TimeSeriesInputSystem": "utils.time_series_input_system",
    "ValveSystem": "valve.valve_system",
    "WallSystem": "wall.wall_system",
}

_ALIASES = {
    "BuildingSpaceTorchSystem": "BuildingSpaceSystem",
    "BuildingSpaceMassTorchSystem": "BuildingSpaceMassSystem",
    "BuildingSpaceThermalTorchSystem": "BuildingSpaceThermalSystem",
    "WallTorchSystem": "WallSystem",
    "DamperTorchSystem": "DamperSystem",
    "ValveTorchSystem": "ValveSystem",
    "CoilTorchSystem": "CoilSystem",
    "FanTorchSystem": "FanSystem",
    "SpaceHeaterTorchSystem": "SpaceHeaterSystem",
    "FanCoilUnitTorchSystem": "FanCoilUnitSystem",
    "AirHandlingUnitTorchSystem": "AirHandlingUnitSystem",
    "OnOffControllerTorchSystem": "SmoothOnOffControllerSystem",
    "ScheduleSwitchControllerTorchSystem": "ScheduleSwitchControllerSystem",
    "SATCompensatedControllerTorchSystem": "SATCompensatedControllerSystem",
    "ControllerIdentificationTorchSystem": "ControllerIdentificationSystem",
    "ControllerIdentificationPITorchSystem": "ControllerIdentificationPISystem",
    "fmuSystem": "FmuSystem",
}


def __getattr__(name):
    canonical_name = _ALIASES.get(name, name)
    module_name = _MODULES.get(canonical_name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(
        importlib.import_module(f"{__name__}.{module_name}"), canonical_name
    )
    globals()[name] = value
    return value


def _load_system_classes():
    """Load canonical system exports for translator pattern discovery."""
    return tuple(getattr(importlib.import_module(f"{__name__}.{module}"), name)
                 for name, module in _MODULES.items()
                 if name.endswith("System"))
