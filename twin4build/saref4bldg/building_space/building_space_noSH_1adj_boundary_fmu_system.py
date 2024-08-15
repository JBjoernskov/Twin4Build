import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, change_sign, add_attr, integrate, multiply
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional

def get_signature_pattern():
    node0 = Node(cls=base.Damper, id="<n<SUB>1</SUB>(Damper)>") #supply damper
    node1 = Node(cls=base.Damper, id="<n<SUB>2</SUB>(Damper)>") #return damper
    node2 = Node(cls=base.BuildingSpace, id="<n<SUB>3</SUB>(BuildingSpace)>")
    node5 = Node(cls=base.Schedule, id="<n<SUB>6</SUB>(Schedule)>") #return valve
    node7 = Node(cls=base.Sensor, id="<n<SUB>8</SUB>(Sensor)>")
    node8 = Node(cls=base.Temperature, id="<n<SUB>9</SUB>(Temperature)>")
    node9 = Node(cls=base.BuildingSpace, id="<n<SUB>10</SUB>(BuildingSpace)>")
    sp = SignaturePattern(ownedBy="BuildingSpaceNoSH1AdjBoundaryFMUSystem", priority=150)

    sp.add_edge(Exact(object=node0, subject=node2, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="hasFluidReturnedBy"))
    sp.add_edge(Exact(object=node2, subject=node5, predicate="hasProfile"))
    sp.add_edge(IgnoreIntermediateNodes(object=node7, subject=node0, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node7, subject=node8, predicate="observes"))
    sp.add_edge(Exact(object=node9, subject=node2, predicate="connectedTo"))


    sp.add_input("airFlowRate", node0)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    sp.add_input("indoorTemperature_adj1", node9, "indoorTemperature")

    sp.add_modeled_node(node2)

    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")

    return sp

class BuildingSpaceNoSH1AdjBoundaryFMUSystem(FMUComponent, base.BuildingSpace):
    sp = [get_signature_pattern()]
    def __init__(self,
                C_supply=None,
                C_air=None,
                C_int=None,
                C_boundary=None,
                R_int=None,
                R_boundary=None,
                Q_occ_gain=None,
                CO2_occ_gain=None,
                CO2_start=None,
                T_boundary=22,
                infiltration=0.005,
                airVolume=None,
                **kwargs):
        building_space.BuildingSpace.__init__(self, **kwargs)


        self.C_supply = C_supply#400
        self.C_air = C_air#1
        self.C_int = C_int#1
        self.C_boundary = C_boundary#1
        self.R_int = R_int#1
        self.R_boundary = R_boundary#1
        self.Q_occ_gain = Q_occ_gain#80
        self.CO2_occ_gain = CO2_occ_gain#8.18E-6
        self.CO2_start = CO2_start#400      
        self.T_boundary = T_boundary
        self.infiltration = infiltration
        self.airVolume = airVolume




        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "R2C2_0noSH_01adj_0boundary_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.input = {'airFlowRate': None,
                    'supplyAirTemperature': None,
                    'numberOfPeople': None,
                    "outdoorCo2Concentration": None,
                    "indoorTemperature_adj1": None,
                    "T_boundary": None}
        self.output = {"indoorTemperature": None, 
                       "indoorCo2Concentration": None}
        
        self.FMUinputMap = {'airFlowRate': "m_a_flow",
                    'supplyAirTemperature': "T_a_supply",
                    'numberOfPeople': "N_occ",
                    "outdoorCo2Concentration": "CO2_supply",
                    "indoorTemperature_adj1": "T_adj1",
                    "T_boundary": "T_boundary"}
        self.FMUoutputMap = {"indoorTemperature": "T_air", 
                             "indoorCo2Concentration": "CO2_concentration"}

        self.FMUparameterMap = {"C_supply": "C_supply",
                                "C_air": "C_air",
                                "C_int": "C_int",
                                "C_boundary": "C_boundary",
                                "R_int": "R_int",
                                "R_boundary": "R_boundary",
                                "Q_occ_gain": "Q_occ_gain", 
                                "CO2_occ_gain": "CO2_occ_gain", 
                                "CO2_start": "CO2_start", 
                                "airVolume": "airVolume",}
        

        self.input_conversion = {'airFlowRate': add_attr(self, "infiltration"),
                                    'supplyAirTemperature': to_degK_from_degC,
                                    'numberOfPeople': do_nothing,
                                    "outdoorCo2Concentration": do_nothing,
                                    "indoorTemperature_adj1": to_degK_from_degC,
                                    "T_boundary": to_degK_from_degC}
        self.output_conversion = {"indoorTemperature": to_degC_from_degK, 
                                  "indoorCo2Concentration": do_nothing}

        self.INITIALIZED = False
        self._config = {"parameters": list(self.FMUparameterMap.keys()) + ["T_boundary", "infiltration"]}

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
        self.input["T_boundary"] = self.T_boundary
        self.input["outdoorCo2Concentration"] = 400


        


        