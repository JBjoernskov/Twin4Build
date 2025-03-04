from .fan import Fan
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.base as base
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.SAREF.FlowJunction)
    node1 = Node(cls=(base.S4BLDG.Fan, base.S4BLDG.AirToAirHeatRecovery, base.S4BLDG.Coil))
    node2 = Node(cls=base.S4BLDG.Fan)
    node3 = Node(cls=base.S4BLDG.PropertyValue)
    node4 = Node(cls=base.XSD.float)
    node5 = Node(cls=base.S4BLDG.NominalPowerRate)
    node6 = Node(cls=base.S4BLDG.PropertyValue)
    node7 = Node(cls=base.XSD.float)
    node8 = Node(cls=base.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="FanFMUSystem")

    sp.add_triple(SinglePath(subject=node2, object=node0, predicate=base.FSO.feedsFluidTo))
    sp.add_triple(SinglePath(subject=node1, object=node2, predicate=base.FSO.feedsFluidTo))
    sp.add_triple(Optional_(subject=node3, object=node4, predicate=base.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node3, object=node5, predicate=base.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node2, object=node3, predicate=base.SAREF.hasPropertyValue))
    sp.add_triple(Optional_(subject=node6, object=node7, predicate=base.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node6, object=node8, predicate=base.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node2, object=node6, predicate=base.SAREF.hasPropertyValue))
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



        


        