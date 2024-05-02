# import twin4build.saref.device.sensor.sensor as sensor
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.utils.time_series_input import TimeSeriesInputSystem
from twin4build.utils.pass_input_to_output import PassInputToOutput
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes
import twin4build.base as base
import numpy as np
import copy

def get_signature_pattern_input():
    node0 = Node(cls=(base.Sensor,), id="<n<SUB>1</SUB>(Sensor)>")
    sp = SignaturePattern(ownedBy="SensorSystem", priority=-1)
    sp.add_modeled_node(node0)
    return sp

# def get_flow_signature_pattern_measure_before():
#     node0 = Node(cls=(base.Sensor,))
#     node1 = Node(cls=(base.Temperature,))
#     node2 = Node(cls=(base.Coil))
#     sp = SignaturePattern(ownedBy="SensorSystem")
#     sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
#     sp.add_edge(Exact(object=node0, subject=node2, predicate="connectedBefore") | IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedBefore"))
#     sp.add_input("measuredValue", node2, ("inletWaterTemperature"))
#     sp.add_modeled_node(node0)
#     return sp

# Temperature sensor placed in air flow stream after coil
def get_flow_signature_pattern_after_coil_air_side1():
    node0 = Node(cls=(base.Sensor,), id="get_flow_signature_pattern_after_coil_air_side1")
    node1 = Node(cls=(base.Temperature,))
    node2 = Node(cls=(base.Coil), id="Coil_cas1")
    node3 = Node(cls=(base.Fan, base.AirToAirHeatRecovery, base.Damper), id="FanHRDamper") #Placed on air-side
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedAfter"))
    # sp.add_edge((Exact(object=node0, subject=node3, predicate="connectedBefore") | IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedBefore")) | (Exact(object=node0, subject=node3, predicate="connectedAfter") | IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedAfter")))
    sp.add_input("measuredValue", node2, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp

def get_flow_signature_pattern_after_coil_air_side2():
    node0 = Node(cls=(base.Sensor,), id="get_flow_signature_pattern_after_coil_air_side2")
    node1 = Node(cls=(base.Temperature,))
    node2 = Node(cls=(base.Coil))
    node3 = Node(cls=(base.Fan, base.AirToAirHeatRecovery, base.Damper)) #Placed on air-side
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedBefore"))
    sp.add_input("measuredValue", node2, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp

# Temperature sensor placed in air flow stream after coil - check that the sensor in not placed on the water side
def get_flow_signature_pattern_after_coil_air_side3():
    node0 = Node(cls=(base.Sensor,), id="get_flow_signature_pattern_after_coil_air_side3")
    node1 = Node(cls=(base.Temperature,))
    node2 = Node(cls=(base.Coil), id="Coil_cas3")
    node3 = Node(cls=(base.Sensor), id="sensor_waterside_outlet") #We can find the sensor on the water side 
    node4 = Node(cls=(base.Valve, base.Pump), id="VP") #Placed on water-side
    node5 = Node(cls=(base.Sensor), id="sensor_waterside_inlet") #We can find the sensor on the water side
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node3, subject=node2, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node3, subject=node4, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node5, subject=node4, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node5, subject=node2, predicate="connectedBefore"))
    # sp.add_edge((Exact(object=node0, subject=node3, predicate="connectedBefore") | IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedBefore")) | (Exact(object=node0, subject=node3, predicate="connectedAfter") | IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedAfter")))
    sp.add_input("measuredValue", node2, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp

# Temperature sensor placed in water flow stream after coil
def get_flow_signature_pattern_after_coil_water_side1():
    node0 = Node(cls=(base.Sensor,), id="get_flow_signature_pattern_after_coil_water_side1")
    node1 = Node(cls=(base.Temperature,))
    node2 = Node(cls=(base.Coil), id="Coil_cws1")
    node3 = Node(cls=(base.Valve, base.Pump), id="VP") #Placed on water-side
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedAfter"))
    sp.add_input("measuredValue", node2, ("outletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp

def get_flow_signature_pattern_after_coil_water_side2():
    node0 = Node(cls=(base.Sensor,), id="get_flow_signature_pattern_after_coil_water_side2")   
    node1 = Node(cls=(base.Temperature,))
    node2 = Node(cls=(base.Coil))
    node3 = Node(cls=(base.Valve, base.Pump)) #Placed on water-side
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedBefore"))
    sp.add_input("measuredValue", node2, ("outletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp


###################
def get_flow_signature_pattern_before_coil_water_side1():
    node0 = Node(cls=(base.Sensor,), id="get_flow_signature_pattern_before_coil_water_side1")
    node1 = Node(cls=(base.Temperature,))
    node2 = Node(cls=(base.Coil), id="Coil_cws1")
    node3 = Node(cls=(base.Valve, base.Pump), id="VP") #Placed on water-side
    node4 = Node(cls=(base.Sensor,))


    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedBefore"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedAfter"))
    sp.add_edge(IgnoreIntermediateNodes(object=node4, subject=node3, predicate="connectedBefore"))
    sp.add_edge(IgnoreIntermediateNodes(object=node4, subject=node0, predicate="connectedBefore"))
    sp.add_input("measuredValue", node2, ("inletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp

def get_flow_signature_pattern_before_coil_water_side2():
    node0 = Node(cls=(base.Sensor,), id="get_flow_signature_pattern_before_coil_water_side2")   
    node1 = Node(cls=(base.Temperature,))
    node2 = Node(cls=(base.Coil))
    node3 = Node(cls=(base.Valve, base.Pump)) #Placed on water-side
    node4 = Node(cls=(base.Sensor,))
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node2, predicate="connectedBefore"))
    sp.add_edge(IgnoreIntermediateNodes(object=node0, subject=node3, predicate="connectedBefore"))
    sp.add_edge(IgnoreIntermediateNodes(object=node4, subject=node3, predicate="connectedBefore"))
    sp.add_edge(IgnoreIntermediateNodes(object=node4, subject=node0, predicate="connectedBefore"))
    sp.add_input("measuredValue", node2, ("inletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp


# Properties of spaces
def get_space_temperature_signature_pattern():
    node0 = Node(cls=(base.Sensor,), id="space_temperature_sensor")
    node1 = Node(cls=(base.Temperature))
    node2 = Node(cls=(base.BuildingSpace))
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, ("indoorTemperature"))
    sp.add_modeled_node(node0)
    return sp

# Properties of spaces
def get_space_co2_signature_pattern():
    node0 = Node(cls=(base.Sensor,), id="space_co2_sensor")
    node1 = Node(cls=(base.Co2))
    node2 = Node(cls=(base.BuildingSpace))
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, ("indoorCo2Concentration"))
    sp.add_modeled_node(node0)
    return sp

def get_position_signature_pattern():
    node0 = Node(cls=(base.Sensor,))
    node1 = Node(cls=(base.OpeningPosition))
    node2 = Node(cls=(base.Valve, base.Damper))
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, ("valvePosition", "damperPosition"))
    sp.add_modeled_node(node0)
    return sp

class SensorSystem(Sensor):
    sp = [get_signature_pattern_input(), 
          get_space_temperature_signature_pattern(),
          get_space_co2_signature_pattern(), 
          get_position_signature_pattern(), 
          get_flow_signature_pattern_after_coil_air_side1(), 
          get_flow_signature_pattern_after_coil_air_side2(), 
          get_flow_signature_pattern_after_coil_air_side3(),
          get_flow_signature_pattern_after_coil_water_side1(), 
          get_flow_signature_pattern_after_coil_water_side2(),
          get_flow_signature_pattern_before_coil_water_side1(),
          get_flow_signature_pattern_before_coil_water_side2()]
    def __init__(self,
                 filename=None,
                 df_input=None,
                 addUncertainty=False,
                **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.df_input = df_input
        self.datecolumn = 0
        self.valuecolumn = 1
        self.addUncertainty = addUncertainty
        self._config = {"parameters": {},
                        "readings": {"filename": self.filename,
                                     "datecolumn": self.datecolumn,
                                     "valuecolumn": self.valuecolumn}
                        }

    @property
    def config(self):
        return self._config

    def set_is_physical_system(self):
        assert (len(self.connectsAt)==0 and self.filename is None and self.df_input is None)==False, f"Sensor object \"{self.id}\" has no inputs and the argument \"filename\" or \"df_input\" in the constructor was not provided."
        if len(self.connectsAt)==0:
            self.isPhysicalSystem = True
        else:
            self.isPhysicalSystem = False

    def set_do_step_instance(self):
        if self.isPhysicalSystem:
            self.do_step_instance = self.physicalSystem
        else:
            self.do_step_instance = PassInputToOutput(id="pass input to output")

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        if self.filename is not None:
            self.physicalSystem = TimeSeriesInputSystem(id=f"time series input - {self.id}", filename=self.filename, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
        elif self.df_input is not None:
            self.physicalSystem = TimeSeriesInputSystem(id=f"time series input - {self.id}", df_input=self.df_input, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
        else:
            self.physicalSystem = None
        self.set_is_physical_system()
        self.set_do_step_instance()
        self.do_step_instance.input = self.input
        self.do_step_instance.output = self.output
        self.do_step_instance.initialize(startTime,
                                        endTime,
                                        stepSize)

        self.inputUncertainty = copy.deepcopy(self.input)
        percentile = 2
        self.standardDeviation = self.observes.MEASURING_UNCERTAINTY/percentile
        # property_ = self.observes
        # if property_.MEASURING_TYPE=="P":
        #     key = list(self.inputUncertainty.keys())[0]
        #     self.inputUncertainty[key] = property_.MEASURING_UNCERTAINTY/100
        # else:
        #     key = list(self.inputUncertainty.keys())[0]
        #     self.inputUncertainty[key] = property_.MEASURING_UNCERTAINTY

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.do_step_instance.input = self.input
        self.do_step_instance.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        if self.addUncertainty:
            for key in self.do_step_instance.output:
                self.output[key] = self.do_step_instance.output[key] + np.random.normal(0, self.standardDeviation)
        else:
            self.output = self.do_step_instance.output

    def get_subset_gradient(self, x_key, y_keys=None, as_dict=False):
        if as_dict==False:
            return np.array([1])
        else:
            return {key: 1 for key in y_keys}
        
    def get_physical_readings(self,
                            startTime=None,
                            endTime=None,
                            stepSize=None):
        assert self.physicalSystem is not None, f"Cannot return physical readings for Sensor with id \"{self.id}\" as the argument \"filename\" was not provided when the object was initialized."
        self.physicalSystem.initialize(startTime,
                                        endTime,
                                        stepSize)
        return self.physicalSystem.physicalSystemReadings

