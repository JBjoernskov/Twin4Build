import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes, Optional

def get_signature_pattern():
    node0 = Node(cls=base.Damper, id="<n<SUB>1</SUB>(Damper)>") #supply damper
    node1 = Node(cls=base.Damper, id="<n<SUB>2</SUB>(Damper)>") #return damper
    node2 = Node(cls=base.BuildingSpace, id="<n<SUB>3</SUB>(BuildingSpace)>")
    node3 = Node(cls=base.Valve, id="<n<SUB>4</SUB>(Valve)>") #supply valve
    node4 = Node(cls=base.SpaceHeater, id="<n<SUB>5</SUB>(SpaceHeater)>")
    node5 = Node(cls=base.Schedule, id="<n<SUB>6</SUB>(Schedule)>") #return valve
    node6 = Node(cls=base.OutdoorEnvironment, id="<n<SUB>7</SUB>(OutdoorEnvironment)>")
    node7 = Node(cls=base.Sensor, id="<n<SUB>8</SUB>(Sensor)>")
    node8 = Node(cls=base.Temperature, id="<n<SUB>9</SUB>(Temperature)>") 
    sp = SignaturePattern(ownedBy="BuildingSpaceFMUSystem")

    sp.add_edge(Exact(object=node0, subject=node2, predicate="connectedBefore"))
    sp.add_edge(Exact(object=node1, subject=node2, predicate="connectedAfter"))
    sp.add_edge(Exact(object=node3, subject=node2, predicate="isContainedIn"))
    sp.add_edge(Exact(object=node4, subject=node2, predicate="isContainedIn"))
    sp.add_edge(Exact(object=node3, subject=node4, predicate="connectedBefore"))
    sp.add_edge(Exact(object=node2, subject=node5, predicate="hasProfile"))
    sp.add_edge(Exact(object=node2, subject=node6, predicate="connectedTo"))
    sp.add_edge(IgnoreIntermediateNodes(object=node7, subject=node0, predicate="connectedBefore"))
    sp.add_edge(Exact(object=node7, subject=node8, predicate="observes"))


    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    # cs.add_input("supplyAirTemperature", x)
    # cs.add_input("supplyWaterTemperature", x)
    # cs.add_input("globalIrradiation", x)
    # cs.add_input("outdoorTemperature", x)
    # cs.add_input("numberOfPeople", x)
    # cs.add_input("outdoorCo2Concentration", x)
    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)

    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")

    return sp

class BuildingSpaceFMUSystem(FMUComponent, base.BuildingSpace, base.SpaceHeater):
    sp = [get_signature_pattern()]
    def __init__(self,
                C_supply=None,
                C_wall=None,
                C_air=None,
                R_out=None,
                R_in=None,
                f_wall=None,
                f_air=None,
                Q_occ_gain=None,
                CO2_occ_gain=None,
                CO2_start=None,
                m_flow_nominal_sh=None,
                fraRad_sh=None,
                Q_flow_nominal_sh=None,
                T_a_nominal_sh=None,
                T_b_nominal_sh=None,
                TAir_nominal_sh=None,
                n_sh=None,
                **kwargs):
        building_space.BuildingSpace.__init__(self, **kwargs)


        self.C_supply = C_supply#400
        self.C_wall = C_wall#1
        self.C_air = C_air#1
        self.R_out = R_out#1
        self.R_in = R_in#1
        self.f_wall = f_wall#1
        self.f_air = f_air#1
        self.Q_occ_gain = Q_occ_gain#80
        self.CO2_occ_gain = CO2_occ_gain#8.18E-6
        self.CO2_start = CO2_start#400      
        self.m_flow_nominal_sh = m_flow_nominal_sh#1
        self.fraRad_sh = fraRad_sh#0.35
        self.Q_flow_nominal_sh = Q_flow_nominal_sh#1000
        self.T_a_nominal_sh = T_a_nominal_sh
        self.T_b_nominal_sh = T_b_nominal_sh
        self.TAir_nominal_sh = TAir_nominal_sh
        self.n_sh = n_sh#1.24




        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "R2C2_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.input = {'airFlowRate': None,
                    'waterFlowRate': None,
                    'supplyAirTemperature': None,
                    'supplyWaterTemperature': None,
                    'globalIrradiation': None,
                    'outdoorTemperature': None,
                    'numberOfPeople': None,
                    "outdoorCo2Concentration": None}
        self.output = {"indoorTemperature": None, "indoorCo2Concentration": None}
        
        self.FMUinputMap = {'airFlowRate': "m_a_flow",
                    'waterFlowRate': "m_w_flow",
                    'supplyAirTemperature': "T_a_supply",
                    'supplyWaterTemperature': "T_w_supply",
                    'globalIrradiation': "Rad_outdoor",
                    'outdoorTemperature': "T_outdoor",
                    'numberOfPeople': "N_occ",
                    "outdoorCo2Concentration": "CO2_supply"}
        self.FMUoutputMap = {"indoorTemperature": "T_air", 
                             "indoorCo2Concentration": "CO2_concentration"}

        self.FMUparameterMap = {"C_supply": "C_supply",
                                "C_wall": "C_wall", 
                                "C_air": "C_air", 
                                "R_out": "R_out", 
                                "R_in": "R_in", 
                                "f_wall": "f_wall", 
                                "f_air": "f_air", 
                                "Q_occ_gain": "Q_occ_gain", 
                                "CO2_occ_gain": "CO2_occ_gain", 
                                "CO2_start": "CO2_start", 
                                "m_flow_nominal_sh": "m_flow_nominal_sh", 
                                "fraRad_sh": "fraRad_sh", 
                                "Q_flow_nominal_sh": "Q_flow_nominal_sh", 
                                "T_a_nominal_sh": "T_a_nominal_sh", 
                                "T_b_nominal_sh": "T_b_nominal_sh", 
                                "TAir_nominal_sh": "TAir_nominal_sh", 
                                "n_sh": "n_sh", }
        


        self.parameter = {"C_supply": {"lb": 200, "ub": 600},
                          "C_wall": {"lb": 5000, "ub": 1e+6},
                            "C_air": {"lb": 5000, "ub": 1e+6},
                            "R_out": {"lb": 0.0001, "ub": 1},
                            "R_in": {"lb": 0.0001, "ub": 1},
                            "f_wall": {"lb": 0, "ub": 1},
                            "f_air": {"lb": 0, "ub": 1},
                            "Q_occ_gain": {"lb": 0, "ub": 200},
                            "CO2_occ_gain": {"lb": 0, "ub": 1e-5},
                            "CO2_start": {"lb": 200, "ub": 1000},
                            "m_flow_nominal_sh": {"lb": 0, "ub": 2},
                            "fraRad_sh": {"lb": 0, "ub": 1},
                            "Q_flow_nominal_sh": {"lb": 0, "ub": 5000},
                            "T_a_nominal_sh": {"lb": 273, "ub": 323},
                            "T_b_nominal_sh": {"lb": 273, "ub": 323},
                            "TAir_nominal_sh": {"lb": 273, "ub": 323},
                            "n_sh": {"lb": 1, "ub": 1.99}
        }
        
        


        self.input_unit_conversion = {'airFlowRate': do_nothing,
                                    'waterFlowRate': do_nothing,
                                    'supplyAirTemperature': to_degK_from_degC,
                                    'supplyWaterTemperature': to_degK_from_degC,
                                    'globalIrradiation': do_nothing,
                                    'outdoorTemperature': to_degK_from_degC,
                                    'numberOfPeople': do_nothing,
                                    "outdoorCo2Concentration": do_nothing}
        self.output_unit_conversion = {"indoorTemperature": to_degC_from_degK, "indoorCo2Concentration": do_nothing}

        self.INITIALIZED = False
        self._config = {"parameters": list(self.parameter.keys())}

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
                    stepSize=None):
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = True ###


        parameters = {"T_start_wall": 23.24,
                      "T_start_air": 23.24,}
        self.set_parameters(parameters=parameters)


        


        