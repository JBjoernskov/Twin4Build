from .fan import Fan
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=(base.FlowJunction,), id="<Meter\nn<SUB>1</SUB>>")
    node1 = Node(cls=(base.Fan, base.AirToAirHeatRecovery, base.Coil), id="<Fan, AirToAirHeatRecovery, Coil\nn<SUB>2</SUB>>")
    node2 = Node(cls=(base.Fan,), id="<Fan\nn<SUB>3</SUB>>")
    node3 = Node(cls=(base.PropertyValue), id="<PropertyValue\nn<SUB>4</SUB>>")
    node4 = Node(cls=(float, int), id="<Float, Int\nn<SUB>5</SUB>>")
    node5 = Node(cls=base.NominalPowerRate, id="<nominalPowerRate\nn<SUB>6</SUB>>")
    node6 = Node(cls=base.PropertyValue, id="<PropertyValue\nn<SUB>7</SUB>>")
    node7 = Node(cls=(float, int), id="<Float, Int\nn<SUB>8</SUB>>")
    node8 = Node(cls=base.NominalAirFlowRate, id="<nominalAirFlowRate\nn<SUB>9</SUB>>")



    sp = SignaturePattern(ownedBy="FanFMUSystem")
    sp.add_edge(IgnoreIntermediateNodes(object=node2, subject=node0, predicate="feedsFluidTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node1, subject=node2, predicate="feedsFluidTo"))
    sp.add_edge(Optional(object=node3, subject=node4, predicate="hasValue"))
    sp.add_edge(Optional(object=node3, subject=node5, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node2, subject=node3, predicate="hasPropertyValue"))
    sp.add_edge(Optional(object=node6, subject=node7, predicate="hasValue"))
    sp.add_edge(Optional(object=node6, subject=node8, predicate="isValueOfProperty"))
    sp.add_edge(Optional(object=node2, subject=node6, predicate="hasPropertyValue"))
    sp.add_input("airFlowRate", node0, "airFlowRateIn")
    sp.add_input("inletAirTemperature", node1, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_parameter("nominalPowerRate.hasValue", node4)
    sp.add_parameter("nominalAirFlowRate.hasValue", node7)
    sp.add_modeled_node(node2)
    return sp

class FanFMUSystem(FMUComponent, Fan):
    sp = [get_signature_pattern()]
    def __init__(self,
                c1=None,
                c2=None,
                c3=None,
                c4=None,
                f_total=None,
                **kwargs):
        # Fan.__init__(self, **kwargs)
        super().__init__(**kwargs)
        # self.c1 = 0.09206979
        # self.c2 = -0.06898674
        # self.c3 = 0.91641847
        # self.c4 = -0.1151978
    
        self.c1=c1
        self.c2=c2
        self.c3=c3
        self.c4=c4
        self.f_total=f_total
        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "EPlusFan_0FMU_0test2port.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.input = {"airFlowRate": tps.Scalar(),
                      "inletAirTemperature": tps.Scalar()}
        self.output = {"outletAirTemperature": tps.Scalar(),
                       "Power": tps.Scalar()}
        
        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"airFlowRate": 0,
                                "inletAirTemperature": -np.inf}
        self.inputUpperBounds = {"airFlowRate": np.inf,
                                "inletAirTemperature": np.inf}
        
        # self.FMUinputMap = {"airFlowRate": "airFlowRate",
        #                 "inletAirTemperature": "inletAirTemperature"}
        
        # self.FMUoutputMap = {"outletAirTemperature": "outletAirTemperature",
        #                   "Power": "Power"}

        self.FMUinputMap = {"airFlowRate": "inlet.m_flow",
                        "inletAirTemperature": "inlet.forward.T"}
        
        self.FMUoutputMap = {"outletAirTemperature": "outlet.forward.T",
                          "Power": "Power"}

        self.FMUparameterMap = {"nominalPowerRate.hasValue": "nominalPowerRate",
                                "nominalAirFlowRate.hasValue": "nominalAirFlowRate",
                                "c1": "c1",
                                "c2": "c2",
                                "c3": "c3",
                                "c4": "c4",
                                "f_total": "f_total"}

        self.input_conversion = {"airFlowRate": do_nothing,
                                      "inletAirTemperature": to_degK_from_degC}
        
        self.output_conversion = {"outletAirTemperature": to_degC_from_degK,
                                      "Power": do_nothing}
        self.INITIALIZED = False
        self._config = {"parameters": list(self.FMUparameterMap.keys())}

    @property
    def config(self):
        return self._config

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
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
 
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = True ###



        


        