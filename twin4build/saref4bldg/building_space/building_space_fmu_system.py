import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node 

def get_signature_pattern():
    node0 = Node(cls=(base.Fan, base.Coil, base.AirToAirHeatRecovery, base.Sensor, base.Meter))
    node1 = Node(cls=(base.Fan,))
    cs = SignaturePattern(ownedBy=BuildingSpaceFMUSystem)
    cs.add_edge(node0, node1, "connectedBefore")
    cs.add_input("airFlow", node0)
    return cs

class BuildingSpaceFMUSystem(FMUComponent, building_space.BuildingSpace):
    # cs = get_signature_pattern()
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
                n_sh=None,
                T_a_nominal_sh=None,
                T_b_nominal_sh=None,
                TAir_nominal_sh=None,
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
        self.n_sh = n_sh#1.24


        if self.temperatureClassification is not None:
            self.T_a_nominal_sh=273.15+int(self.temperatureClassification[0:2])
            self.T_b_nominal_sh=273.15+int(self.temperatureClassification[3:5])
            self.TAir_nominal_sh=273.15+int(self.temperatureClassification[6:])
        else:
            self.T_a_nominal_sh=273.15+45
            self.T_b_nominal_sh=273.15+30
            self.TAir_nominal_sh=273.15+20

        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "R2C2_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.input = {'AirFlowRate': None,
                    'waterFlowRate': None,
                    'supplyAirTemperature': None,
                    'supplyWaterTemperature': None,
                    'globalIrradiation': None,
                    'outdoorTemperature': None,
                    'numberOfPeople': None,
                    "outdoorCo2Concentration": None}
        self.output = {"indoorTemperature": None, "indoorCo2Concentration": None}
        
        self.FMUinputMap = {'AirFlowRate': "m_a_flow",
                    'waterFlowRate': "m_w_flow",
                    'supplyAirTemperature': "T_a_supply",
                    'supplyWaterTemperature': "T_w_supply",
                    'globalIrradiation': "Rad_outdoor",
                    'outdoorTemperature': "T_outdoor",
                    'numberOfPeople': "N_occ",
                    "outdoorCo2Concentration": "CO2_supply"}
        self.FMUoutputMap = {"indoorTemperature": "T_air", 
                             "indoorCo2Concentration": "CO2_concetration"}

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
        





        self.input_unit_conversion = {'AirFlowRate': do_nothing,
                    'waterFlowRate': do_nothing,
                    'supplyAirTemperature': to_degK_from_degC,
                    'supplyWaterTemperature': to_degK_from_degC,
                    'globalIrradiation': do_nothing,
                    'outdoorTemperature': to_degK_from_degC,
                    'numberOfPeople': do_nothing,
                    "outdoorCo2Concentration": do_nothing}
        self.output_unit_conversion = {"indoorTemperature": to_degC_from_degK, "indoorCo2Concentration": do_nothing}

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
                    stepSize=None):
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
 
        if self.INITIALIZED:
            self.reset()
        else:
            FMUComponent.__init__(self, fmu_path=self.fmu_path, unzipdir=self.unzipdir)
            self.INITIALIZED = True ###


        


        