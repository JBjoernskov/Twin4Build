import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, change_sign, add, get, integrate, multiply_const
import twin4build.base as base
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=base.S4BLDG.Damper) #supply damper
    node1 = Node(cls=base.S4BLDG.Damper) #return damper
    node2 = Node(cls=base.S4BLDG.BuildingSpace)
    node3 = Node(cls=base.S4BLDG.Valve) #supply valve
    node4 = Node(cls=base.S4BLDG.SpaceHeater)
    node5 = Node(cls=base.S4BLDG.Schedule) #return valve
    node7 = Node(cls=base.SAREF.Sensor)
    node8 = Node(cls=base.SAREF.Temperature)
    node9 = Node(cls=base.S4BLDG.BuildingSpace)
    node10 = Node(cls=base.S4BLDG.BuildingSpace)
    sp = SignaturePattern(ownedBy="BuildingSpace2AdjBoundaryFMUSystem", priority=200)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=base.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node3, object=node2, predicate=base.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node4, object=node2, predicate=base.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=base.SAREF.hasProfile))
    sp.add_triple(SinglePath(subject=node7, object=node0, predicate=base.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node7, object=node8, predicate=base.SAREF.observes))
    sp.add_triple(Exact(subject=node9, object=node2, predicate=base.S4SYST.connectedTo))
    sp.add_triple(Exact(subject=node10, object=node2, predicate=base.S4SYST.connectedTo))

    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    sp.add_input("indoorTemperature_adj1", node9, "indoorTemperature")
    sp.add_input("indoorTemperature_adj2", node10, "indoorTemperature")

    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)

    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")

    return sp

class BuildingSpace2AdjBoundaryFMUSystem(FMUComponent, base.BuildingSpace, base.SpaceHeater):
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
                fraRad_sh=None,
                Q_flow_nominal_sh=None,
                T_a_nominal_sh=None,
                T_b_nominal_sh=None,
                TAir_nominal_sh=None,
                n_sh=None,
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
        self.fraRad_sh = fraRad_sh#0.35
        self.Q_flow_nominal_sh = Q_flow_nominal_sh#1000
        self.T_a_nominal_sh = T_a_nominal_sh
        self.T_b_nominal_sh = T_b_nominal_sh
        self.TAir_nominal_sh = TAir_nominal_sh
        self.n_sh = n_sh#1.24
        self.T_boundary = T_boundary
        self.infiltration = infiltration
        self.airVolume = airVolume




        self.start_time = 0
        fmu_filename = "R2C2_02adj_0boundary_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.input = {'airFlowRate': tps.Scalar(),
                    'waterFlowRate': tps.Scalar(),
                    'supplyAirTemperature': tps.Scalar(),
                    'supplyWaterTemperature': tps.Scalar(),
                    'numberOfPeople': tps.Scalar(),
                    "outdoorCo2Concentration": tps.Scalar(),
                    "indoorTemperature_adj1": tps.Scalar(),
                    "indoorTemperature_adj2": tps.Scalar(),
                    "T_boundary": tps.Scalar(),
                    "m_infiltration": tps.Scalar(),
                    "T_infiltration": tps.Scalar()}
        self.output = {"indoorTemperature": tps.Scalar(), 
                       "indoorCo2Concentration": tps.Scalar(), 
                       "spaceHeaterPower": tps.Scalar(),
                        "spaceHeaterEnergy": tps.Scalar()}
        
        self.FMUinputMap = {'airFlowRate': "m_a_flow",
                    'waterFlowRate': "m_w_flow",
                    'supplyAirTemperature': "T_a_supply",
                    'supplyWaterTemperature': "T_w_supply",
                    'numberOfPeople': "N_occ",
                    "outdoorCo2Concentration": "CO2_supply",
                    "indoorTemperature_adj1": "T_adj1",
                    "indoorTemperature_adj2": "T_adj2",
                    "T_boundary": "T_boundary",
                    "m_infiltration": "m_infiltration",
                    "T_infiltration": "T_infiltration"}
        self.FMUoutputMap = {"indoorTemperature": "T_air", 
                             "indoorCo2Concentration": "CO2_concentration",
                             "spaceHeaterPower": "r2C2_1.rad.Q_flow"}

        self.FMUparameterMap = {"C_supply": "C_supply",
                                "C_air": "C_air",
                                "C_int": "C_int",
                                "C_boundary": "C_boundary",
                                "R_int": "R_int",
                                "R_boundary": "R_boundary",
                                "Q_occ_gain": "Q_occ_gain",
                                "CO2_occ_gain": "CO2_occ_gain",
                                "CO2_start": "CO2_start", 
                                "airVolume": "airVolume",
                                "fraRad_sh": "fraRad_sh",
                                "Q_flow_nominal_sh": "Q_flow_nominal_sh",
                                "T_a_nominal_sh": "T_a_nominal_sh",
                                "T_b_nominal_sh": "T_b_nominal_sh",
                                "TAir_nominal_sh": "TAir_nominal_sh",
                                "n_sh": "n_sh"}
        

        self.input_conversion = {'airFlowRate': do_nothing,
                                    'waterFlowRate': do_nothing,
                                    'supplyAirTemperature': to_degK_from_degC,
                                    'supplyWaterTemperature': to_degK_from_degC,
                                    'numberOfPeople': do_nothing,
                                    "outdoorCo2Concentration": do_nothing,
                                    "indoorTemperature_adj1": to_degK_from_degC,
                                    "indoorTemperature_adj2": to_degK_from_degC,
                                    "T_boundary": to_degK_from_degC,
                                    "m_infiltration": do_nothing,
                                    "T_infiltration": get(self.output, "indoorTemperature", conversion=to_degK_from_degC)}
        


        self.output_conversion = {"indoorTemperature": to_degC_from_degK, 
                                  "indoorCo2Concentration": do_nothing,
                                  "spaceHeaterPower": change_sign,
                                  "spaceHeaterEnergy": integrate(self.output, "spaceHeaterPower", conversion=multiply_const(1/3600/1000))}

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
        self.input["m_infiltration"] = self.infiltration
        self.output_conversion["spaceHeaterEnergy"].v = 0

        


        