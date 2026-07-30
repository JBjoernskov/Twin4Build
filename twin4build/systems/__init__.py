"""Systems module for Twin4Build package.

This module provides a comprehensive collection of building system components that can be used to model
and simulate building systems. Each system is implemented as a PyTorch-based component for efficient
simulation and optimization.

Key Components:
    Building Spaces:
        - BuildingSpaceTorchSystem: Combined thermal + CO2 (mass balance) building space model
        - BuildingSpaceMassTorchSystem: Building space CO2 mass balance model
        - BuildingSpaceThermalTorchSystem: Building space thermal (RC) model

    Building Envelope:
        - WallTorchSystem: 2R1C wall between two zones (or zone and boundary)

    HVAC Components:
        - SpaceHeaterTorchSystem: Space heating system
        - FanCoilUnitTorchSystem: Fan coil unit (heating/cooling terminal unit)
        - ValveTorchSystem: Control valve system
        - CoilTorchSystem: Heating/cooling coil system
        - DamperTorchSystem: Air flow control damper
        - FanTorchSystem: Air handling fan system
        - AirToAirHeatRecoverySystem: Heat recovery system
        - AirHandlingUnitTorchSystem: Air handling unit system

    Control Systems:
        - PIDControllerSystem: Proportional-Integral-Derivative controller
        - CascadeControllerSystem: Cascade (outer/inner loop) controller
        - OnOffControllerSystem: Threshold-based on/off controller
        - OnOffControllerTorchSystem: Torch-based on/off controller
        - ScheduleSwitchControllerTorchSystem: Schedule-based switching controller
        - SATLinearRuleSystem / SATCompensatedControllerTorchSystem: Supply air
          temperature compensation rules
        - ClassificationAnnControllerSystem: ANN-based classification control
        - NeuralPolicyControllerSystem: Neural network policy control
        - ControllerIdentificationTorchSystem / ControllerIdentificationPITorchSystem:
          Controller identification models

    Monitoring & Measurement:
        - SensorSystem: Generic sensor system
        - ScheduleSystem: Time-based scheduling system
        - PiecewiseLinearScheduleSystem: Schedule with piecewise linear interpolation

    Environmental:
        - OutdoorEnvironmentSystem: External environmental conditions
        - ShadingDeviceSystem: Solar shading control

    Utility Systems:
        - fmuSystem: FMU-based system
        - SupplyFlowJunctionSystem: Supply flow distribution
        - ReturnFlowJunctionSystem: Return flow collection
        - PiecewiseLinearSystem: Piecewise linear interpolation
        - TimeSeriesInputSystem: Time series data input
        - MaxSystem: Maximum value selection
        - OnOffSystem: Binary state system
        - DiscreteStatespaceSystem: General-purpose discrete state-space model
        - ScalarProductSystem: Elementwise scalar product
        - OccupancySystem / OccupancyDetectorSystem: Occupancy modeling and detection
        - SigmoidGate: Smooth gating function

Note:
    Most systems are implemented using PyTorch for efficient computation and
    optimization. FMUs can still be wrapped via fmuSystem.
"""

# Define what gets exported with wildcard imports
__all__ = [
    # Building Spaces
    "BuildingSpaceTorchSystem",
    "BuildingSpaceMassTorchSystem",
    "BuildingSpaceThermalTorchSystem",
    # Wall
    "WallTorchSystem",
    # Space Heater
    "SpaceHeaterTorchSystem",
    # Valves
    "ValveTorchSystem",
    # Coils
    "CoilTorchSystem",
    # Fan Coil Unit
    "FanCoilUnitTorchSystem",
    # Controllers # TODO: Convert to Torch
    "PIDControllerSystem",
    "CascadeControllerSystem",
    "CascadePIDControllerSystem",  # backward-compatible alias
    "OnOffControllerSystem",
    "OnOffControllerTorchSystem",
    "ScheduleSwitchControllerTorchSystem",
    "SATLinearRuleSystem",
    "SATCompensatedControllerTorchSystem",
    "ClassificationAnnControllerSystem",
    "NeuralPolicyControllerSystem",
    "ControllerIdentificationTorchSystem",
    "ControllerIdentificationPITorchSystem",
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
    "AirHandlingUnitTorchSystem",
    # Damper
    "DamperTorchSystem",
    # Fan
    "FanTorchSystem",
    # Shading
    "ShadingDeviceSystem",
    # Uncomment these if you want to include utility classes # TODO: Convert to Torch
    "fmuSystem",
    "PiecewiseLinearSystem",
    "TimeSeriesInputSystem",
    "MaxSystem",
    "OnOffSystem",
    "DiscreteStatespaceSystem",
    "ScalarProductSystem",
    "OccupancySystem",
    "OccupancyDetectorSystem",
    "SigmoidGate",
]

# Local application imports
from twin4build.systems.air_handling_unit.air_handling_unit_torch_system import (
    AirHandlingUnitTorchSystem,
)

# Air to Air Heat Recovery
from twin4build.systems.air_to_air_heat_recovery.air_to_air_heat_recovery_system import (
    AirToAirHeatRecoverySystem,
)
from twin4build.systems.building_space.building_space_mass_torch_system import (
    BuildingSpaceMassTorchSystem,
)
from twin4build.systems.building_space.building_space_thermal_torch_system import (
    BuildingSpaceThermalTorchSystem,
)

# Building Spaces
from twin4build.systems.building_space.building_space_torch_system import (
    BuildingSpaceTorchSystem,
)

# Coils
from twin4build.systems.coil.coil_torch_system import CoilTorchSystem

# Fan Coil Unit
from twin4build.systems.fan_coil_unit.fan_coil_unit_torch_system import (
    FanCoilUnitTorchSystem,
)
from twin4build.systems.controller.classification_ann_controller.classification_ann_controller_system import (
    ClassificationAnnControllerSystem,
)
from twin4build.systems.controller.controller_identification.controller_identification_torch_system import (
    ControllerIdentificationTorchSystem,
)
from twin4build.systems.controller.controller_identification.controller_identification_pi_torch_system import (
    ControllerIdentificationPITorchSystem,
)
from twin4build.systems.controller.controller_identification.loop_classifier import (
    ActuatorSeeds,
    LoopScore,
    confidence_label,
    derive_actuator_seeds,
    score_pair,
)
from twin4build.systems.controller.controller_identification.pi_loop_rewire import (
    RewireReport,
)
from twin4build.systems.controller.neural_policy_controller.neural_policy_controller_system import (
    NeuralPolicyControllerSystem,
)
from twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_system import (
    OnOffControllerSystem,
)
from twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_torch_system import (
    OnOffControllerTorchSystem,
)
from twin4build.systems.controller.rulebased_controller.sat_compensated_controller.sat_compensated_controller_torch_system import (
    SATCompensatedControllerTorchSystem,
    SATLinearRuleSystem,
)
from twin4build.systems.controller.rulebased_controller.schedule_switch_controller.schedule_switch_controller_torch_system import (
    ScheduleSwitchControllerTorchSystem,
)
from twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system import (
    CascadePIDControllerSystem,  # backward-compatible alias
)
from twin4build.systems.controller.setpoint_controller.cascade_controller.cascade_controller_system import (
    CascadeControllerSystem,
)

# Controllers
from twin4build.systems.controller.setpoint_controller.pid_controller.pid_controller_system import (
    PIDControllerSystem,
)

# Damper
from twin4build.systems.damper.damper_torch_system import DamperTorchSystem

# Fan
from twin4build.systems.fan.fan_torch_system import FanTorchSystem
from twin4build.systems.junction.return_flow_junction_system import (
    ReturnFlowJunctionSystem,
)

# Junction
from twin4build.systems.junction.supply_flow_junction_system import (
    SupplyFlowJunctionSystem,
)

# Outdoor Environment
from twin4build.systems.outdoor_environment.outdoor_environment_system import (
    OutdoorEnvironmentSystem,
)
from twin4build.systems.schedule.piecewise_linear_schedule_system import (
    PiecewiseLinearScheduleSystem,
)

# Schedules
from twin4build.systems.schedule.schedule_system import ScheduleSystem

# Sensors
from twin4build.systems.sensor.sensor_system import SensorSystem

# Shading
from twin4build.systems.shading_device.shading_device_system import ShadingDeviceSystem

# Space Heater
from twin4build.systems.space_heater.space_heater_torch_system import (
    SpaceHeaterTorchSystem,
)
from twin4build.systems.utils.discrete_statespace_system import DiscreteStatespaceSystem

# Utils
from twin4build.systems.utils.fmu_system import fmuSystem
from twin4build.systems.utils.max_system import MaxSystem
from twin4build.systems.utils.occupancy_detector_system import OccupancyDetectorSystem
from twin4build.systems.utils.sigmoid_gate import SigmoidGate
from twin4build.systems.utils.occupancy_system import OccupancySystem
from twin4build.systems.utils.on_off_system import OnOffSystem
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.systems.utils.scalar_product_system import ScalarProductSystem
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem

# Valves
from twin4build.systems.valve.valve_torch_system import ValveTorchSystem

# Wall
from twin4build.systems.wall.wall_torch_system import WallTorchSystem

# Time series input
