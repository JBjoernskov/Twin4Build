import twin4build.systems.utils.fmu.fmu_component as fmu_component
from twin4build.utils.uppath import uppath
import numpy as np
import os
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_, MultiPath
import twin4build.utils.input_output_types as tps

# Supply fan
def get_signature_pattern_supply():
    node0 = Node(cls=core.S4BLDG.FlowJunction)
    node1 = Node(cls=(core.S4BLDG.Fan, core.S4BLDG.AirToAirHeatRecovery, core.S4BLDG.Coil))
    node2 = Node(cls=core.S4BLDG.Fan)
    node3 = Node(cls=core.S4BLDG.PropertyValue)
    node4 = Node(cls=core.XSD.float)
    node5 = Node(cls=core.S4BLDG.NominalPowerRate)
    node6 = Node(cls=core.S4BLDG.PropertyValue)
    node7 = Node(cls=core.XSD.float)
    node8 = Node(cls=core.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="FanFMUSystem")

    sp.add_triple(SinglePath(subject=node0, object=node2, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(SinglePath(subject=node2, object=node1, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Optional_(subject=node3, object=node4, predicate=core.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node3, object=node5, predicate=core.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node2, object=node3, predicate=core.SAREF.hasPropertyValue))
    sp.add_triple(Optional_(subject=node6, object=node7, predicate=core.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node6, object=node8, predicate=core.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node2, object=node6, predicate=core.SAREF.hasPropertyValue))
    sp.add_input("airFlowRate", node0, "airFlowRateIn")
    sp.add_input("inletAirTemperature", node1, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_parameter("nominalPowerRate", node4)
    sp.add_parameter("nominalAirFlowRate", node7)
    sp.add_modeled_node(node2)
    return sp

# Return fan
def get_signature_pattern_return():
    node1 = Node(cls=(core.S4BLDG.FlowJunction, core.S4BLDG.AirToAirHeatRecovery))
    node2 = Node(cls=core.S4BLDG.Fan)
    node3 = Node(cls=core.S4BLDG.PropertyValue)
    node4 = Node(cls=core.XSD.float)
    node5 = Node(cls=core.S4BLDG.NominalPowerRate)
    node6 = Node(cls=core.S4BLDG.PropertyValue)
    node7 = Node(cls=core.XSD.float)
    node8 = Node(cls=core.S4BLDG.NominalAirFlowRate)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="FanFMUSystem")

    sp.add_triple(SinglePath(subject=node2, object=node1, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Optional_(subject=node3, object=node4, predicate=core.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node3, object=node5, predicate=core.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node2, object=node3, predicate=core.SAREF.hasPropertyValue))
    sp.add_triple(Optional_(subject=node6, object=node7, predicate=core.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node6, object=node8, predicate=core.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node2, object=node6, predicate=core.SAREF.hasPropertyValue))
    sp.add_input("airFlowRate", node1, ("airFlowRateIn", "secondaryAirFlowRate"))
    sp.add_input("inletAirTemperature", node1, ("airTemperatureOut", "secondaryTemperatureOut"))
    sp.add_parameter("nominalPowerRate", node4)
    sp.add_parameter("nominalAirFlowRate", node7)
    sp.add_modeled_node(node2)
    return sp

class FanFMUSystem(fmu_component.FMUComponent):
    sp = [get_signature_pattern_supply(), get_signature_pattern_return()]
    def __init__(self,
                nominalPowerRate=None,
                nominalAirFlowRate=None,
                c1=None,
                c2=None,
                c3=None,
                c4=None,
                f_total=None,
                **kwargs):
        super().__init__(**kwargs)
        self.start_time = 0
        fmu_filename = "EPlusFan_0FMU_0test2port.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = fmu_component.unzip_fmu(self.fmu_path)
        self.nominalPowerRate = nominalPowerRate
        self.nominalAirFlowRate = nominalAirFlowRate
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        f_total = f_total

        self.input = {"airFlowRate": tps.Scalar(),
                      "inletAirTemperature": tps.Scalar()}
        self.output = {"outletAirTemperature": tps.Scalar(),
                       "Power": tps.Scalar()}
        
        self.inputLowerBounds = {"airFlowRate": 0,
                                "inletAirTemperature": -np.inf}
        self.inputUpperBounds = {"airFlowRate": np.inf,
                                "inletAirTemperature": np.inf}
        
        self.FMUinputMap = {"airFlowRate": "inlet.m_flow",
                        "inletAirTemperature": "inlet.forward.T"}
        
        self.FMUoutputMap = {"outletAirTemperature": "outlet.forward.T",
                          "Power": "Power"}

        self.FMUparameterMap = {"nominalPowerRate": "nominalPowerRate",
                                "nominalAirFlowRate": "nominalAirFlowRate",
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
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = True



        


        