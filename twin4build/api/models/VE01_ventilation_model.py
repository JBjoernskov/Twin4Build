"""
VE01 - First Ventilation System Model
The ventilation system supplies air for 3 floors
Requires: 
C02 timeseries data for each one of the 20 demand control ventilated rooms [ppm]
Outputs:
- Damper position for each one of the 20 rooms [%] (0-1)
- Approximated airflow for each one of the 20 rooms [kg/s]
- Total mass air flow rate for the entire system [kg/s]
"""

import json
import pandas as pd
import requests
import twin4build as tb
from twin4build.saref.property_.flow.flow import Flow
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_duct_device.junction_duct.junction_duct_system import JunctionDuctSystem
#from twin4build.saref.property_value.property_value import PropertyValue

def model_definition(self):

    """
    Defining a model for U044 first ventilation system VE01
    The ventilation system supplies air for 3 floors with:
    - 20 Rooms with CO2-based demand controlled ventilation.
    - 18 Auxiliary rooms with constant ventilation. (To be verified)
    The rooms with controlled ventilation have supply and return dampers with controlled damper position.
    Each one of these rooms will be consisting of a space model, two damper models and a controller.
    As part of the simulation, an occupancy schedule or occupancy data should be provided for the room to estimate the rising CO2 levels.
    """

    ######################################
    ############ DCV Rooms ###############
    ######################################
    
    total_airflow_property = Flow()
    total_airflow_sensor = tb.SensorSystem(
        observes=total_airflow_property,
        saveSimulationResult=True,
        id="Total_AirFlow_sensor")

    ######### Cellar ###############

    # Ø22-601b-00

    co2_property_22_601b_00 = tb.Co2()

    """
    co2_controller_22_601b_00 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_601b_00,
        room_identifier=0,
        saveSimulationResult=True,
        id="CO2_controller_22_601b_00")
    """
    position_controller_property_22_601b_00 = tb.OpeningPosition()
    co2_controller_sensor_22_601b_00 = tb.SensorSystem(
        observes=position_controller_property_22_601b_00,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_controller_sensor_22_601b_00")
    
    position_property_22_601b_00 = tb.OpeningPosition()
    damper_position_sensor_22_601b_00 = tb.SensorSystem(
        observes=position_property_22_601b_00,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_601b_00")
    
    supply_damper_22_601b_00 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(4800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_601b_00")
    
    return_damper_22_601b_00 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(4800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_601b_00")
    
    
    space_22_601b_00_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_601b_00,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_601b_00")
    
    co2_property_22_601b_00.isPropertyOf = space_22_601b_00_CO2_sensor

    ######### Ground floor ###############

    # Ø22-604-0

    co2_property_22_604_0 = tb.Co2()
    
    space_22_604_0_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_604_0,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_604_0")

    co2_controller_22_604_0 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_604_0,
        room_identifier=1,
        saveSimulationResult=True,
        id="CO2_controller_22_604_0") 

    supply_damper_22_604_0 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(3000/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_604_0")

    return_damper_22_604_0 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(3000/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_604_0")
    
    position_property_22_604_0 = tb.OpeningPosition()
    damper_position_sensor_22_604_0 = tb.SensorSystem(
        observes=position_property_22_604_0,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_604_0")

    co2_property_22_604_0.isPropertyOf = space_22_604_0_CO2_sensor

    # Ø22-603-0

    co2_property_22_603_0 = tb.Co2()
    co2_controller_22_603_0 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_603_0,
        room_identifier=2,
        saveSimulationResult=True,
        id="CO2_controller_22_603_0")

    supply_damper_22_603_0 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(2200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_603_0")

    return_damper_22_603_0 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(2200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_603_0")

    space_22_603_0_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_603_0,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_603_0")
    
    position_property_22_603_0 = tb.OpeningPosition()
    damper_position_sensor_22_603_0 = tb.SensorSystem(
        observes=position_property_22_603_0,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_603_0")

    co2_property_22_603_0.isPropertyOf = space_22_603_0_CO2_sensor

    # Ø22-601b-0

    co2_property_22_601b_0 = tb.Co2()
    co2_controller_22_601b_0 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_601b_0,
        room_identifier=3,
        saveSimulationResult=True,
        id="CO2_controller_22_601b_0")

    supply_damper_22_601b_0 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(4800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_601b_0")

    return_damper_22_601b_0 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(4800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_601b_0")

    space_22_601b_0_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_601b_0,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_601b_0")
    
    position_property_22_601b_0 = tb.OpeningPosition()
    damper_position_sensor_22_601b_0 = tb.SensorSystem(
        observes=position_property_22_601b_0,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_601b_0")
    
    co2_property_22_601b_0.isPropertyOf = space_22_601b_0_CO2_sensor

    ######### First floor ###############

    # Ø22-601b-1 

    co2_property_22_601b_1 = tb.Co2()
    co2_controller_22_601b_1 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_601b_1,
        room_identifier=4,
        saveSimulationResult=True,
        id="CO2_controller_22_601b_1")

    supply_damper_22_601b_1 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(6300/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_601b_1")

    return_damper_22_601b_1 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(6300/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_601b_1")

    space_22_601b_1_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_601b_1,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_601b_1")
    
    position_property_22_601b_1 = tb.OpeningPosition()
    damper_position_sensor_22_601b_1 = tb.SensorSystem(
        observes=position_property_22_601b_1,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_601b_1")
    
    co2_property_22_601b_1.isPropertyOf = space_22_601b_1_CO2_sensor

    # Ø22-603-1 

    co2_property_22_603_1 = tb.Co2()
    co2_controller_22_603_1 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_603_1,
        room_identifier=5,
        saveSimulationResult=True,
        id="CO2_controller_22_603_1")

    supply_damper_22_603_1 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(2200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_603_1")

    return_damper_22_603_1 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(2200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_603_1")

    space_22_603_1_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_603_1,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_603_1")
    
    position_property_22_603_1 = tb.OpeningPosition()
    damper_position_sensor_22_603_1 = tb.SensorSystem(
        observes=position_property_22_603_1,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_603_1")
    
    co2_property_22_603_1.isPropertyOf = space_22_603_1_CO2_sensor

    # Ø22-604-1 

    co2_property_22_604_1 = tb.Co2()
    co2_controller_22_604_1 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_604_1,
        room_identifier=6,
        saveSimulationResult=True,
        id="CO2_controller_22_604_1")

    supply_damper_22_604_1 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(3000/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_604_1")

    return_damper_22_604_1 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(3000/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_604_1")

    space_22_604_1_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_604_1,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_604_1")
    
    position_property_22_604_1 = tb.OpeningPosition()
    damper_position_sensor_22_604_1 = tb.SensorSystem(
        observes=position_property_22_604_1,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_604_1")
    
    co2_property_22_604_1.isPropertyOf = space_22_604_1_CO2_sensor


    ######### Second floor ###############

    # Ø22-601b-2

    co2_property_22_601b_2 = tb.Co2()
    co2_controller_22_601b_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_601b_2,
        room_identifier=7,
        saveSimulationResult=True,
        id="CO2_controller_22_601b_2")

    supply_damper_22_601b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(4800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_601b_2")

    return_damper_22_601b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(4800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_601b_2")

    space_22_601b_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_601b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_601b_2")
    
    position_property_22_601b_2 = tb.OpeningPosition()
    damper_position_sensor_22_601b_2 = tb.SensorSystem(
        observes=position_property_22_601b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_601b_2")
    co2_property_22_601b_2.isPropertyOf = space_22_601b_2_CO2_sensor

    # Ø22-603b-2

    co2_property_22_603b_2 = tb.Co2()
    co2_controller_22_603b_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_603b_2,
        room_identifier=8,
        saveSimulationResult=True,
        id="CO2_controller_22_603b_2")

    supply_damper_22_603b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_603b_2")

    return_damper_22_603b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(800/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_603b_2")

    space_22_603b_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_603b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_603b_2")
    
    position_property_22_603b_2 = tb.OpeningPosition()
    damper_position_sensor_22_603b_2 = tb.SensorSystem(
        observes=position_property_22_603b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_603b_2")
    
    co2_property_22_603b_2.isPropertyOf = space_22_603b_2_CO2_sensor

    # Ø22-603a-2 (Office)
    co2_property_22_603a_2 = tb.Co2()
    co2_controller_22_603a_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_603a_2,
        room_identifier=9,
        saveSimulationResult=True,
        id="CO2_controller_22_603a_2")

    supply_damper_22_603a_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_603a_2")

    return_damper_22_603a_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_603a_2")

    space_22_603a_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_603a_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_603a_2")
    
    position_property_22_603a_2 = tb.OpeningPosition()
    
    damper_position_sensor_22_603a_2 = tb.SensorSystem(
        observes=position_property_22_603a_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_603a_2")
    
    co2_property_22_603a_2.isPropertyOf = space_22_603a_2_CO2_sensor

    # Ø22-604a-2 (Office)
    co2_property_22_604a_2 = tb.Co2()
    co2_controller_22_604a_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_604a_2,
        room_identifier=10,
        saveSimulationResult=True,
        id="CO2_controller_22_604a_2")

    supply_damper_22_604a_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_604a_2")

    return_damper_22_604a_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_604a_2")

    space_22_604a_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_604a_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_604a_2")
    
    position_property_22_604a_2 = tb.OpeningPosition()
    damper_position_sensor_22_604a_2 = tb.SensorSystem(
        observes=position_property_22_604a_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_604a_2")
    
    co2_property_22_604a_2.isPropertyOf = space_22_604a_2_CO2_sensor
 
    # Ø22-604b-2 (Office)
    co2_property_22_604b_2 = tb.Co2()
    co2_controller_22_604b_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_604b_2,
        room_identifier=11,
        saveSimulationResult=True,
        id="CO2_controller_22_604b_2")

    supply_damper_22_604b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_604b_2")

    return_damper_22_604b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_604b_2")

    space_22_604b_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_604b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_604b_2")
    
    position_property_22_604b_2 = tb.OpeningPosition()
    damper_position_sensor_22_604b_2 = tb.SensorSystem(
        observes=position_property_22_604b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_604b_2")
    
    co2_property_22_604b_2.isPropertyOf = space_22_604b_2_CO2_sensor

    # Ø22-605a-2 (Office)
    co2_property_22_605a_2 = tb.Co2()
    co2_controller_22_605a_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_605a_2,
        room_identifier=12,
        saveSimulationResult=True,
        id="CO2_controller_22_605a_2")

    supply_damper_22_605a_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_605a_2")

    return_damper_22_605a_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_605a_2")

    space_22_605a_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_605a_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_605a_2")
    
    position_property_22_605a_2 = tb.OpeningPosition()
    damper_position_sensor_22_605a_2 = tb.SensorSystem(
        observes=position_property_22_605a_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_605a_2")
    
    co2_property_22_605a_2.isPropertyOf = space_22_605a_2_CO2_sensor

    # Ø22-605b-2 (Office)
    co2_property_22_605b_2 = tb.Co2()
    co2_controller_22_605b_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_605b_2,
        room_identifier=13,
        saveSimulationResult=True,
        id="CO2_controller_22_605b_2")

    supply_damper_22_605b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_605b_2")

    return_damper_22_605b_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_605b_2")

    space_22_605b_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_605b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_605b_2")
    
    position_property_22_605b_2 = tb.OpeningPosition()
    damper_position_sensor_22_605b_2 = tb.SensorSystem(
        observes=position_property_22_605b_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_605b_2")
    
    co2_property_22_605b_2.isPropertyOf = space_22_605b_2_CO2_sensor

    # Ø22-604e-2 (Office)
    co2_property_22_604e_2 = tb.Co2()
    co2_controller_22_604e_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_604e_2,
        room_identifier=14,
        saveSimulationResult=True,
        id="CO2_controller_22_604e_2")

    supply_damper_22_604e_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_604e_2")

    return_damper_22_604e_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_604e_2")

    space_22_604e_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_604e_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_604e_2")
    
    position_property_22_604e_2 = tb.OpeningPosition()
    damper_position_sensor_22_604e_2 = tb.SensorSystem(
        observes=position_property_22_604e_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_604e_2")
    
    co2_property_22_604e_2.isPropertyOf = space_22_604e_2_CO2_sensor

    # Ø22-604d-2 (Office)
    co2_property_22_604d_2 = tb.Co2()
    co2_controller_22_604d_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_604d_2,
        room_identifier=15,
        saveSimulationResult=True,
        id="CO2_controller_22_604d_2")

    supply_damper_22_604d_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_604d_2")

    return_damper_22_604d_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_604d_2")

    space_22_604d_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_604d_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_604d_2")
    
    position_property_22_604d_2 = tb.OpeningPosition()
    damper_position_sensor_22_604d_2 = tb.SensorSystem(
        observes=position_property_22_604d_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_604d_2")
    
    co2_property_22_604d_2.isPropertyOf = space_22_604d_2_CO2_sensor

    # Ø22-604c-2 (Office)
    co2_property_22_604c_2 = tb.Co2()
    co2_controller_22_604c_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_604c_2,
        room_identifier=16,
        saveSimulationResult=True,
        id="CO2_controller_22_604c_2")

    supply_damper_22_604c_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_604c_2")

    return_damper_22_604c_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_604c_2")

    space_22_604c_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_604c_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_604c_2")
    
    position_property_22_604c_2 = tb.OpeningPosition()
    damper_position_sensor_22_604c_2 = tb.SensorSystem(
        observes=position_property_22_604c_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_604c_2")
    
    co2_property_22_604c_2.isPropertyOf = space_22_604c_2_CO2_sensor

    # Ø22-605e-2 (Office)
    co2_property_22_605e_2 = tb.Co2()
    co2_controller_22_605e_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_605e_2,
        room_identifier=17,
        saveSimulationResult=True,
        id="CO2_controller_22_605e_2")

    supply_damper_22_605e_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_605e_2")

    return_damper_22_605e_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_605e_2")

    space_22_605e_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_605e_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_605e_2")
    
    position_property_22_605e_2 = tb.OpeningPosition()
    damper_position_sensor_22_605e_2 = tb.SensorSystem(
        observes=position_property_22_605e_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_605e_2")
    
    co2_property_22_605e_2.isPropertyOf = space_22_605e_2_CO2_sensor

    # Ø22-605d-2 (Office)
    co2_property_22_605d_2 = tb.Co2()
    co2_controller_22_605d_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_605d_2,
        room_identifier=18,
        saveSimulationResult=True,
        id="CO2_controller_22_605d_2")

    supply_damper_22_605d_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_605d_2")

    return_damper_22_605d_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_605d_2")

    space_22_605d_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_605d_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_605d_2")
    
    position_property_22_605d_2 = tb.OpeningPosition()
    damper_position_sensor_22_605d_2 = tb.SensorSystem(
        observes=position_property_22_605d_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_605d_2")
    
    co2_property_22_605d_2.isPropertyOf = space_22_605d_2_CO2_sensor

    # Ø22-605c-2 (Office)
    co2_property_22_605c_2 = tb.Co2()
    co2_controller_22_605c_2 = tb.ClassificationAnnControllerSystem(
        observes=co2_property_22_605c_2,
        room_identifier=19,
        saveSimulationResult=True,
        id="CO2_controller_22_605c_2")

    supply_damper_22_605c_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Supply_damper_22_605c_2")

    return_damper_22_605c_2 = tb.DamperSystem(
        nominalAirFlowRate=tb.PropertyValue(hasValue=(200/3600)*1.225),
        a=5,
        saveSimulationResult=True,
        id="Return_damper_22_605c_2")

    space_22_605c_2_CO2_sensor = tb.SensorSystem(
        observes=co2_property_22_605c_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="CO2_sensor_22_605c_2")
    
    position_property_22_605c_2 = tb.OpeningPosition()
    damper_position_sensor_22_605c_2 = tb.SensorSystem(
        observes=position_property_22_605c_2,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="Damper_position_sensor_22_605c_2")
    co2_property_22_605c_2.isPropertyOf = space_22_605c_2_CO2_sensor

    # Fan model
    # Parameters estimated using BMS data and Least Squares method
    flow_property_01_main_fan = Flow()
    ve01_main_fan = tb.FanSystem(nominalAirFlowRate=tb.Measurement(hasValue=1.31919033e+01),
                   nominalPowerRate=tb.Measurement(hasValue=8.61452510e+03),
                   c1=5.51225583e-02,
                   c2=1.61440204e-01, 
                   c3=5.58449279e-01, 
                   c4=5.19023563e-01,
                   saveSimulationResult=True,
                   id="main_fan")

    main_fan_power_sensor = tb.SensorSystem(
        observes=flow_property_01_main_fan,
        saveSimulationResult=True,
        doUncertaintyAnalysis=False,
        id="main_fan_power_sensor")
    
    junction_duct = JunctionDuctSystem(
        airFlowRateBias = 1.56,
        saveSimulationResult=True,
        id="main_junction_air_duct")
    
    
    """
    Component connections
    """

    # Ø22_601b_00
    self.add_connection(co2_controller_sensor_22_601b_00, supply_damper_22_601b_00,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_sensor_22_601b_00, return_damper_22_601b_00,
                            "inputSignal", "damperPosition")
    #self.add_connection(space_22_601b_00_CO2_sensor, co2_controller_22_601b_00,
    #                        "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_601b_00, damper_position_sensor_22_601b_00,
                            "damperPosition", "damperPosition_22_601b_00")    

    # Ø22_604_0

    self.add_connection(co2_controller_22_604_0, supply_damper_22_604_0,
                         "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_604_0, return_damper_22_604_0,
                         "inputSignal", "damperPosition")
    self.add_connection(space_22_604_0_CO2_sensor, co2_controller_22_604_0,
                         "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_604_0, damper_position_sensor_22_604_0, 
                            "damperPosition", "damperPosition_22_604_0")
    
    # Ø22_603_0

    self.add_connection(co2_controller_22_603_0, supply_damper_22_603_0,
                         "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_603_0, return_damper_22_603_0,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_603_0_CO2_sensor, co2_controller_22_603_0,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_603_0, damper_position_sensor_22_603_0,
                            "damperPosition", "damperPosition_22_603_0")
    
    # Ø22_601b_0

    self.add_connection(co2_controller_22_601b_0, supply_damper_22_601b_0,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_601b_0, return_damper_22_601b_0,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_601b_0_CO2_sensor, co2_controller_22_601b_0,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_601b_0, damper_position_sensor_22_601b_0,
                            "damperPosition", "damperPosition_22_601b_0")
    
    
    # Ø22_601b_1

    self.add_connection(co2_controller_22_601b_1, supply_damper_22_601b_1,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_601b_1, return_damper_22_601b_1,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_601b_1_CO2_sensor, co2_controller_22_601b_1,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_601b_1, damper_position_sensor_22_601b_1,
                            "damperPosition", "damperPosition_22_601b_1")
    
    # Ø22_603_1

    self.add_connection(co2_controller_22_603_1, supply_damper_22_603_1,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_603_1, return_damper_22_603_1,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_603_1_CO2_sensor, co2_controller_22_603_1,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_603_1, damper_position_sensor_22_603_1,
                            "damperPosition", "damperPosition_22_603_1")
    
    # Ø22_604_1

    self.add_connection(co2_controller_22_604_1, supply_damper_22_604_1,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_604_1, return_damper_22_604_1,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_604_1_CO2_sensor, co2_controller_22_604_1,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_604_1, damper_position_sensor_22_604_1,
                            "damperPosition", "damperPosition_22_604_1")
    
    # Ø22_601b_2

    self.add_connection(co2_controller_22_601b_2, supply_damper_22_601b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_601b_2, return_damper_22_601b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_601b_2_CO2_sensor, co2_controller_22_601b_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_601b_2, damper_position_sensor_22_601b_2,
                            "damperPosition", "damperPosition_22_601b_2")
    
    # Ø22_603b_2

    self.add_connection(co2_controller_22_603b_2, supply_damper_22_603b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_603b_2, return_damper_22_603b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_603b_2_CO2_sensor, co2_controller_22_603b_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_603b_2, damper_position_sensor_22_603b_2,
                            "damperPosition", "damperPosition_22_603b_2")
    
    # Ø22_603a_2

    self.add_connection(co2_controller_22_603a_2, supply_damper_22_603a_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_603a_2, return_damper_22_603a_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_603a_2_CO2_sensor, co2_controller_22_603a_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_603a_2, damper_position_sensor_22_603a_2,
                            "damperPosition", "damperPosition_22_603a_2")
    
    
    # Ø22_604a_2

    self.add_connection(co2_controller_22_604a_2, supply_damper_22_604a_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_604a_2, return_damper_22_604a_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_604a_2_CO2_sensor, co2_controller_22_604a_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_604a_2, damper_position_sensor_22_604a_2,
                            "damperPosition", "damperPosition_22_604a_2")
    
    # Ø22_604b_2

    self.add_connection(co2_controller_22_604b_2, supply_damper_22_604b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_604b_2, return_damper_22_604b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_604b_2_CO2_sensor, co2_controller_22_604b_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_604b_2, damper_position_sensor_22_604b_2,
                            "damperPosition", "damperPosition_22_604b_2")
    
    # Ø22_605a_2

    self.add_connection(co2_controller_22_605a_2, supply_damper_22_605a_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_605a_2, return_damper_22_605a_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_605a_2_CO2_sensor, co2_controller_22_605a_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_605a_2, damper_position_sensor_22_605a_2,
                            "damperPosition", "damperPosition_22_605a_2")
    
    
    # Ø22_605b_2

    self.add_connection(co2_controller_22_605b_2, supply_damper_22_605b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_605b_2, return_damper_22_605b_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_605b_2_CO2_sensor, co2_controller_22_605b_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_605b_2, damper_position_sensor_22_605b_2,
                            "damperPosition", "damperPosition_22_605b_2")
    
    # Ø22_604e_2

    self.add_connection(co2_controller_22_604e_2, supply_damper_22_604e_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_604e_2, return_damper_22_604e_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_604e_2_CO2_sensor, co2_controller_22_604e_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_604e_2, damper_position_sensor_22_604e_2,
                            "damperPosition", "damperPosition_22_604e_2")
    
    # Ø22_604d_2

    self.add_connection(co2_controller_22_604d_2, supply_damper_22_604d_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_604d_2, return_damper_22_604d_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_604d_2_CO2_sensor, co2_controller_22_604d_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_604d_2, damper_position_sensor_22_604d_2,
                            "damperPosition", "damperPosition_22_604d_2")
    
    # Ø22_604c_2

    self.add_connection(co2_controller_22_604c_2, supply_damper_22_604c_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_604c_2, return_damper_22_604c_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_604c_2_CO2_sensor, co2_controller_22_604c_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_604c_2, damper_position_sensor_22_604c_2,
                            "damperPosition", "damperPosition_22_604c_2")
    
    # Ø22_605e_2

    self.add_connection(co2_controller_22_605e_2, supply_damper_22_605e_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_605e_2, return_damper_22_605e_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_605e_2_CO2_sensor, co2_controller_22_605e_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_605e_2, damper_position_sensor_22_605e_2,
                            "damperPosition", "damperPosition_22_605e_2")
    
    # Ø22_605d_2

    self.add_connection(co2_controller_22_605d_2, supply_damper_22_605d_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_605d_2, return_damper_22_605d_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_605d_2_CO2_sensor, co2_controller_22_605d_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_605d_2, damper_position_sensor_22_605d_2,
                            "damperPosition", "damperPosition_22_605d_2")
    
    # Ø22_605c_2

    self.add_connection(co2_controller_22_605c_2, supply_damper_22_605c_2,
                            "inputSignal", "damperPosition")
    self.add_connection(co2_controller_22_605c_2, return_damper_22_605c_2,
                            "inputSignal", "damperPosition")
    self.add_connection(space_22_605c_2_CO2_sensor, co2_controller_22_605c_2,
                            "indoorCo2Concentration", "actualValue")
    self.add_connection(supply_damper_22_605c_2, damper_position_sensor_22_605c_2,
                            "damperPosition", "damperPosition_22_605c_2")

    # Total air flow rate sensor
    self.add_connection(supply_damper_22_601b_00, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_601b_00")
    self.add_connection(supply_damper_22_604_0, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_604_0")
    self.add_connection(supply_damper_22_603_0, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_603_0")
    self.add_connection(supply_damper_22_601b_0, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_601b_0")
    self.add_connection(supply_damper_22_601b_1, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_601b_1")
    self.add_connection(supply_damper_22_603_1, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_603_1")
    self.add_connection(supply_damper_22_604_1, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_604_1")
    self.add_connection(supply_damper_22_601b_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_601b_2")
    self.add_connection(supply_damper_22_603b_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_603b_2")
    self.add_connection(supply_damper_22_603a_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_603a_2")
    self.add_connection(supply_damper_22_604a_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_604a_2")
    self.add_connection(supply_damper_22_604b_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_604b_2")
    self.add_connection(supply_damper_22_605a_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_605a_2")
    self.add_connection(supply_damper_22_605b_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_605b_2")
    self.add_connection(supply_damper_22_604e_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_604e_2")
    self.add_connection(supply_damper_22_604d_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_604d_2")
    self.add_connection(supply_damper_22_604c_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_604c_2")
    self.add_connection(supply_damper_22_605e_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_605e_2")
    self.add_connection(supply_damper_22_605d_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_605d_2")
    self.add_connection(supply_damper_22_605c_2, total_airflow_sensor,
                            "airFlowRate", "airFlowRate_22_605c_2")
    

    # Junction duct connections

        # Total air flow rate sensor
    self.add_connection(supply_damper_22_601b_00, junction_duct,
                            "airFlowRate", "airFlowRate_22_601b_00")
    self.add_connection(supply_damper_22_604_0, junction_duct,
                            "airFlowRate", "airFlowRate_22_604_0")
    self.add_connection(supply_damper_22_603_0, junction_duct,
                            "airFlowRate", "airFlowRate_22_603_0")
    self.add_connection(supply_damper_22_601b_0, junction_duct,
                            "airFlowRate", "airFlowRate_22_601b_0")
    self.add_connection(supply_damper_22_601b_1, junction_duct,
                            "airFlowRate", "airFlowRate_22_601b_1")
    self.add_connection(supply_damper_22_603_1, junction_duct,
                            "airFlowRate", "airFlowRate_22_603_1")
    self.add_connection(supply_damper_22_604_1, junction_duct,
                            "airFlowRate", "airFlowRate_22_604_1")
    self.add_connection(supply_damper_22_601b_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_601b_2")
    self.add_connection(supply_damper_22_603b_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_603b_2")
    self.add_connection(supply_damper_22_603a_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_603a_2")
    self.add_connection(supply_damper_22_604a_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_604a_2")
    self.add_connection(supply_damper_22_604b_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_604b_2")
    self.add_connection(supply_damper_22_605a_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_605a_2")
    self.add_connection(supply_damper_22_605b_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_605b_2")
    self.add_connection(supply_damper_22_604e_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_604e_2")
    self.add_connection(supply_damper_22_604d_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_604d_2")
    self.add_connection(supply_damper_22_604c_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_604c_2")
    self.add_connection(supply_damper_22_605e_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_605e_2")
    self.add_connection(supply_damper_22_605d_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_605d_2")
    self.add_connection(supply_damper_22_605c_2, junction_duct,
                            "airFlowRate", "airFlowRate_22_605c_2")

    


    # Fan model
    self.add_connection(junction_duct, ve01_main_fan,
                            "totalAirFlowRate", "airFlowRate")
    self.add_connection(ve01_main_fan, main_fan_power_sensor,
                            "Power", "Power")
    

def get_total_airflow_rate(model, offset=1.56):
    total_airflow_sensor = model.component_dict["Total_AirFlow_sensor"]
    # Assuming total_airflow_sensor.savedOutput[first_key] is a list
    first_key = next(iter(total_airflow_sensor.savedOutput))
    sum_series = pd.Series(0, index=range(len(total_airflow_sensor.savedOutput[first_key])))
    for key in total_airflow_sensor.savedOutput:
        sum_series = sum_series.add(pd.Series(total_airflow_sensor.savedOutput[key], index=range(len(total_airflow_sensor.savedOutput[key]))), fill_value=0)
    sum_series = sum_series + offset
    return sum_series

def request_to_ventilation_api_test():
        try :
            #fetch input data from the file C:\Project\t4b_fork\Twin4Build\twin4build\api\models\ventilation_input_data.json
            with open(r"C:\Project\t4b_fork\Twin4Build\twin4build\api\models\what_if_PID_controller_test.json") as file:
                input_data = json.load(file)
            

            url = "http://127.0.0.1:8070/simulate_ventilation"

            #we will send a request to API and store its response here
            response = requests.post(url,json=input_data)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                model_output_data = response.json()

                #Save the output data to a file
                with open(r"C:\Project\t4b_fork\Twin4Build\twin4build\api\models\ventilation_output_data.json","w") as file:
                    json.dump(model_output_data,file)

            else:
                print("Got a reponse from api other than 200 response is: %s"%str(response.status_code))


        except Exception as e :
            print("Error: %s" %e)


if __name__ == "__main__":
    request_to_ventilation_api_test()