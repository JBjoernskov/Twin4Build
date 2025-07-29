"""Systems module for Twin4Build package.

This module provides a comprehensive collection of building system components that can be used to model
and simulate building systems. Each system is implemented as a PyTorch-based component for efficient
simulation and optimization.

Key Components:
    Building Spaces:
        - BuildingSpaceTorchSystem: Basic building space model
        - BuildingSpaceMassTorchSystem: Building space with mass effects
        - BuildingSpaceThermalTorchSystem: Building space with thermal dynamics

    HVAC Components:
        - SpaceHeaterTorchSystem: Space heating system
        - ValveTorchSystem: Control valve system
        - CoilTorchSystem: Heating/cooling coil system
        - DamperTorchSystem: Air flow control damper
        - FanTorchSystem: Air handling fan system
        - AirToAirHeatRecoverySystem: Heat recovery system

    Control Systems:
        - PIControllerFMUSystem: Proportional-Integral controller
        - RulebasedSetpointInputControllerSystem: Rule-based setpoint control
        - OnOffControllerSystem: Binary control system
        - SequenceControllerSystem: Sequential control logic
        - ClassificationAnnControllerSystem: ANN-based classification control
        - NeuralPolicyControllerSystem: Neural network policy control

    Monitoring & Measurement:
        - SensorSystem: Generic sensor system
        - ScheduleSystem: Time-based scheduling system

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

Note:
    Most systems are implemented using PyTorch for efficient computation and optimization.
    Some legacy systems (marked with TODO) are still using FMU-based implementations.
"""

# Define what gets exported with wildcard imports
__all__ = [
    # Building Spaces
    "BuildingSpaceTorchSystem",
    "BuildingSpaceMassTorchSystem",
    "BuildingSpaceThermalTorchSystem",
    # Space Heater
    "SpaceHeaterTorchSystem",
    # Valves
    "ValveTorchSystem",
    # Coils
    "CoilTorchSystem",
    # Controllers # TODO: Convert to Torch
    "PIDControllerSystem",
    "RulebasedSetpointInputControllerSystem",
    "OnOffControllerSystem",
    "SequenceControllerSystem",
    "ClassificationAnnControllerSystem",
    "NeuralPolicyControllerSystem",
    # Sensors
    "SensorSystem",
    # Schedules
    "ScheduleSystem",
    # Outdoor Environment
    "OutdoorEnvironmentSystem",
    # Junction
    "SupplyFlowJunctionSystem",
    "ReturnFlowJunctionSystem",
    # Air to Air Heat Recovery
    "AirToAirHeatRecoverySystem",
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
]

# Local application imports
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
from twin4build.systems.controller.classification_ann_controller.classification_ann_controller_system import (
    ClassificationAnnControllerSystem,
)
from twin4build.systems.controller.neural_policy_controller.neural_policy_controller_system import (
    NeuralPolicyControllerSystem,
)
from twin4build.systems.controller.rulebased_controller.on_off_controller.on_off_controller_system import (
    OnOffControllerSystem,
)
from twin4build.systems.controller.rulebased_controller.rulebased_setpoint_input_controller.rulebased_setpoint_input_controller_system import (
    RulebasedSetpointInputControllerSystem,
)
from twin4build.systems.controller.sequence_controller.sequence_controller_system import (
    SequenceControllerSystem,
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
from twin4build.systems.utils.on_off_system import OnOffSystem
from twin4build.systems.utils.piecewise_linear_system import PiecewiseLinearSystem
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem

# Valves
from twin4build.systems.valve.valve_torch_system import ValveTorchSystem

# Time series input
