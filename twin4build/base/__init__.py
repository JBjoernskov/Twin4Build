import rdflib
from twin4build.model.semantic_model.semantic_model import SemanticModel
from twin4build.saref.measurement.measurement import Measurement
from twin4build.saref.property_value.property_value import PropertyValue
from twin4build.saref.property_.s4bldg_property.s4bldg_property import *

from twin4build.saref4syst.system import System
from twin4build.saref4syst.connection import Connection
from twin4build.saref4syst.connection_point import ConnectionPoint

from twin4build.saref.property_.property_ import Property
from twin4build.saref.property_.temperature.temperature import Temperature

from twin4build.saref.property_.temperature.outlet_temperature.outlet_temperature import OutletTemperature
from twin4build.saref.property_.Co2.Co2 import Co2
from twin4build.saref.property_.opening_position.opening_position import OpeningPosition
from twin4build.saref.property_.energy.energy import Energy
from twin4build.saref.property_.pressure.pressure import Pressure
from twin4build.saref.property_.flow.flow import Flow
from twin4build.saref.property_.power.power import Power
from twin4build.saref.property_.motion.motion import Motion
# from types import NoneType
NoneType = type(None)

from twin4build.saref.profile.schedule.schedule import Schedule
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_device import DistributionDevice
from twin4build.saref4bldg.building_space.building_space import BuildingSpace
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.coil.coil import Coil
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller import Controller
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.rulebased_controller.rulebased_controller import RulebasedController
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.classification_ann_controller.classification_ann_controller import ClassificationAnnController
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.neural_policy_controller.neural_policy_controller import NeuralPolicyController
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.setpoint_controller.setpoint_controller import SetpointController
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery import AirToAirHeatRecovery
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.damper.damper import Damper
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_controller.valve.valve import Valve
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.fan.fan import Fan
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_moving_device.pump.pump import Pump
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater import SpaceHeater
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.saref4bldg.physical_object.building_object.building_device.shading_device.shading_device import ShadingDevice
from twin4build.utils.outdoor_environment.outdoor_environment import OutdoorEnvironment
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_junction.flow_junction import FlowJunction


FSO = rdflib.Namespace("https://w3id.org/fso#")
SAREF = rdflib.Namespace("https://saref.etsi.org/core/")
S4BLDG = rdflib.Namespace("https://saref.etsi.org/saref4bldg/")
S4SYST = rdflib.Namespace("https://saref.etsi.org/saref4syst/")
XSD = rdflib.Namespace("http://www.w3.org/2001/XMLSchema#")

def get_ontologies():
    sm = SemanticModel()
    FSO = "https://alikucukavci.github.io/FSO/fso.ttl"
    SAREF = "https://saref.etsi.org/core/"
    S4BLDG = "https://saref.etsi.org/saref4bldg/"
    S4SYST = "https://saref.etsi.org/saref4syst/"
    namespaces = [FSO, SAREF, S4BLDG, S4SYST]
    sm.parse_namespaces(sm.graph, namespaces)
    return sm

ontologies = get_ontologies()
