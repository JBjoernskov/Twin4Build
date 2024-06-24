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

def get_flow_signature_pattern_before_coil_air_side():
    node0 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Temperature,), id="<Temperature\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Coil), id="<Coil\nn<SUB>3</SUB>>") #waterside
    node3 = Node(cls=(base.Coil), id="<Coil\nn<SUB>4</SUB>>") #airside
    node4 = Node(cls=(base.Coil), id="<Coil\nn<SUB>5</SUB>>") #supersystem
    node5 = Node(cls=base.System, id="<System\nn<SUB>6</SUB>>") #before waterside
    node6 = Node(cls=base.System, id="<System\nn<SUB>7</SUB>>") #after waterside
    node7 = Node(cls=base.System, id="<System\nn<SUB>8</SUB>>") #before airside
    node8 = Node(cls=base.System, id="<System\nn<SUB>9</SUB>>") #after airside
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node5, subject=node2, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node2, subject=node6, predicate="returnsFluidTo"))
    sp.add_edge(Exact(object=node7, subject=node3, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node3, subject=node8, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node2, subject=node4, predicate="subSystemOf"))
    sp.add_edge(Exact(object=node3, subject=node4, predicate="subSystemOf"))
    sp.add_edge(IgnoreIntermediateNodes(object=node3, subject=node0, predicate="suppliesFluidTo"))
    sp.add_input("measuredValue", node4, ("outletAirTemperature"))
    sp.add_modeled_node(node0)
    return sp

def get_flow_signature_pattern_after_coil_water_side():
    node0 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Temperature,), id="<Temperature\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Coil), id="<Coil\nn<SUB>3</SUB>>") #waterside
    node3 = Node(cls=(base.Coil), id="<Coil\nn<SUB>4</SUB>>") #airside
    node4 = Node(cls=(base.Coil), id="<Coil\nn<SUB>5</SUB>>") #supersystem
    node5 = Node(cls=base.System, id="<System\nn<SUB>6</SUB>>") #before waterside
    node6 = Node(cls=base.System, id="<System\nn<SUB>7</SUB>>") #after waterside
    node7 = Node(cls=base.System, id="<System\nn<SUB>8</SUB>>") #before airside
    node8 = Node(cls=base.System, id="<System\nn<SUB>9</SUB>>") #after airside
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node5, subject=node2, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node2, subject=node6, predicate="returnsFluidTo"))
    sp.add_edge(Exact(object=node7, subject=node3, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node3, subject=node8, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node2, subject=node4, predicate="subSystemOf"))
    sp.add_edge(Exact(object=node3, subject=node4, predicate="subSystemOf"))
    sp.add_edge(IgnoreIntermediateNodes(object=node2, subject=node0, predicate="returnsFluidTo"))
    sp.add_input("measuredValue", node4, ("outletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp

def get_flow_signature_pattern_before_coil_water_side():
    node0 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Temperature,), id="<Temperature\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Coil), id="<Coil\nn<SUB>3</SUB>>") #waterside
    node3 = Node(cls=(base.Coil), id="<Coil\nn<SUB>4</SUB>>") #airside
    node4 = Node(cls=(base.Coil), id="<Coil\nn<SUB>5</SUB>>") #supersystem
    node5 = Node(cls=base.System, id="<System\nn<SUB>6</SUB>>") #before waterside
    node6 = Node(cls=base.System, id="<System\nn<SUB>7</SUB>>") #after waterside
    node7 = Node(cls=base.System, id="<System\nn<SUB>8</SUB>>") #before airside
    node8 = Node(cls=base.System, id="<System\nn<SUB>9</SUB>>") #after airside
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    # sp.add_edge(Exact(object=node5, subject=node2, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node2, subject=node6, predicate="returnsFluidTo"))
    sp.add_edge(Exact(object=node7, subject=node3, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node3, subject=node8, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node2, subject=node4, predicate="subSystemOf"))
    sp.add_edge(Exact(object=node3, subject=node4, predicate="subSystemOf"))
    sp.add_edge(IgnoreIntermediateNodes(object=node2, subject=node0, predicate="hasFluidSuppliedBy"))
    sp.add_input("measuredValue", node4, ("inletWaterTemperature"))
    sp.add_modeled_node(node0)
    return sp


# Properties of spaces
def get_space_temperature_signature_pattern():
    node0 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Temperature), id="<Temperature\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.BuildingSpace), id="<BuildingSpace\nn<SUB>3</SUB>>")
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, ("indoorTemperature"))
    sp.add_modeled_node(node0)
    return sp

# Properties of spaces
def get_space_co2_signature_pattern():
    node0 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Co2), id="<Co2\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.BuildingSpace), id="<BuildingSpace\nn<SUB>3</SUB>>")
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, ("indoorCo2Concentration"))
    sp.add_modeled_node(node0)
    return sp

def get_position_signature_pattern():
    node0 = Node(cls=(base.Sensor,), id="<Sensor\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.OpeningPosition), id="<OpeningPosition\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Valve, base.Damper), id="<Valve, Damper\nn<SUB>3</SUB>>")
    sp = SignaturePattern(ownedBy="SensorSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="observes"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="isPropertyOf"))
    sp.add_input("measuredValue", node2, ("valvePosition", "damperPosition"))
    sp.add_modeled_node(node0)
    return sp

class SensorSystem(Sensor):
    sp = [get_signature_pattern_input(), 
            get_flow_signature_pattern_before_coil_air_side(),
            get_flow_signature_pattern_after_coil_water_side(),
            get_flow_signature_pattern_before_coil_water_side(),
          get_space_temperature_signature_pattern(),
          get_space_co2_signature_pattern(), 
          get_position_signature_pattern()]
    def __init__(self,
                 filename=None,
                 df_input=None,
                #  addUncertainty=False,
                **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.df_input = df_input
        self.datecolumn = 0
        self.valuecolumn = 1
        # self.addUncertainty = addUncertainty
        self._config = {"parameters": [],
                        "readings": {"filename": self.filename,
                                     "df_input": self.df_input,
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
                    stepSize=None,
                    model=None):
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

        # self.inputUncertainty = copy.deepcopy(self.input)
        # percentile = 2
        # self.standardDeviation = self.observes.MEASURING_UNCERTAINTY/percentile
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
        # if self.addUncertainty:
        #     for key in self.do_step_instance.output:
        #         self.output[key] = self.do_step_instance.output[key] + np.random.normal(0, self.standardDeviation)
        # else:
        self.output = self.do_step_instance.output

    # def get_subset_gradient(self, x_key, y_keys=None, as_dict=False):
    #     if as_dict==False:
    #         return np.array([1])
    #     else:
    #         return {key: 1 for key in y_keys}
        
    def get_physical_readings(self,
                            startTime=None,
                            endTime=None,
                            stepSize=None):
        assert self.physicalSystem is not None, f"Cannot return physical readings for Sensor with id \"{self.id}\" as the argument \"filename\" was not provided when the object was initialized."
        self.physicalSystem.initialize(startTime,
                                        endTime,
                                        stepSize)
        return self.physicalSystem.physicalSystemReadings

