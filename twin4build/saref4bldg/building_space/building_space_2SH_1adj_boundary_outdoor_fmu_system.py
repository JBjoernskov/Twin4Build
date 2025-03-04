import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, change_sign, add, get, integrate, multiply_const, multiply
import twin4build.base as base
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern of the FMU component.
    """
    node0 = Node(cls=base.S4BLDG.Damper) #supply damper
    node1 = Node(cls=base.S4BLDG.Damper) #return damper
    node2 = Node(cls=base.S4BLDG.BuildingSpace)
    node3 = Node(cls=base.S4BLDG.Valve) #supply valve
    node4 = Node(cls=base.S4BLDG.SpaceHeater)
    node5 = Node(cls=base.S4BLDG.Schedule) #return valve
    node6 = Node(cls=base.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=(base.S4BLDG.Coil, base.S4BLDG.AirToAirHeatRecovery, base.S4BLDG.Fan))
    node8 = Node(cls=base.SAREF.Temperature)
    node9 = Node(cls=base.S4BLDG.BuildingSpace)
    node10 = Node(cls=base.S4BLDG.Valve) #supply valve
    node11 = Node(cls=base.S4BLDG.SpaceHeater)
    sp = SignaturePattern(semantic_model_=base.ontologies, ownedBy="BuildingSpace2SH1AdjBoundaryOutdoorFMUSystem", priority=200)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=base.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node3, object=node2, predicate=base.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node4, object=node2, predicate=base.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node10, object=node2, predicate=base.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node11, object=node2, predicate=base.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node10, object=node11, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=base.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=base.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node2, object=node6, predicate=base.S4SYST.connectedTo))
    sp.add_triple(SinglePath(subject=node0, object=node7, predicate=base.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node9, object=node2, predicate=base.S4SYST.connectedTo))

    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate1", node3, "waterFlowRate")
    sp.add_input("waterFlowRate2", node10, "waterFlowRate")
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_input("indoorTemperature_adj1", node9, "indoorTemperature")

    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)
    sp.add_modeled_node(node11)

    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")

    return sp


class BuildingSpace2SH1AdjBoundaryOutdoorFMUSystem(FMUComponent, base.BuildingSpace, base.SpaceHeater):
    """
    A class representing an FMU of a building space with 11 adjacent spaces, 2 space heaters, balanced supply and return ventilation, and an outdoor boundary.
    """
    sp = [get_signature_pattern()]
    def __init__(self,
                C_supply=None,
                C_wall=None,
                C_air=None,
                C_int=None,
                C_boundary=None,
                R_out=None,
                R_in=None,
                R_int=None,
                R_boundary=None,
                f_wall=None,
                f_air=None,
                Q_occ_gain=None,
                CO2_occ_gain=None,
                CO2_start=None,
                fraRad_sh1=None,
                Q_flow_nominal_sh1=None,
                T_a_nominal_sh1=None,
                T_b_nominal_sh1=None,
                TAir_nominal_sh1=None,
                n_sh1=None,
                fraRad_sh2=None,
                Q_flow_nominal_sh2=None,
                T_a_nominal_sh2=None,
                T_b_nominal_sh2=None,
                TAir_nominal_sh2=None,
                n_sh2=None,
                T_boundary=22,
                infiltration=0.005,
                airVolume=None,
                **kwargs):
        """
        Initialize a BuildingSpace2SH1AdjBoundaryOutdoorFMUSystem object.

        Args:
            C_supply (float, optional): The CO2 concentration of the supply air. Defaults to None.
            C_wall (float, optional): The thermal capacitance of the wall. Defaults to None.
            C_air (float, optional): The thermal capacitance of the air. Defaults to None.
            C_int (float, optional): The thermal capacitance of the interior walls. Defaults to None.
            C_boundary (float, optional): The thermal capacitance of the boundary. Defaults to None.
            R_out (float, optional): The exterior wall outer thermal resistance. Defaults to None.
            R_in (float, optional): The exterior wall inner thermal resistance. Defaults to None.
            R_int (float, optional): The thermal resistance of the interior walls. Defaults to None.
            R_boundary (float, optional): The boundary thermal resistance. Defaults to None.
            f_wall (float, optional): The fraction of solar radiation that is absorbed by the wall. Defaults to None.
            f_air (float, optional): The fraction of solar radiation that is absorbed by the air. Defaults to None.
            Q_occ_gain (float, optional): The occupancy thermal gain of the building space. Defaults to None.
            CO2_occ_gain (float, optional): The occupancy CO2 generation rate of the building space. Defaults to None.
            CO2_start (float, optional): The occupancy CO2 concentration of the building space. Defaults to None.
            fraRad_sh1 (float, optional): The fraction of radiation of space heater 1. Defaults to None.
            Q_flow_nominal_sh1 (float, optional): The nominal heat flow rate of space heater 1. Defaults to None.
            T_a_nominal_sh1 (float, optional): The nominal supply air temperature of space heater 1. Defaults to None.
            T_b_nominal_sh1 (float, optional): The nominal return air temperature of space heater 1. Defaults to None.
            TAir_nominal_sh1 (float, optional): The nominal air temperature of space heater 1. Defaults to None.
            n_sh1 (float, optional): The nominal heat transfer coefficient of space heater 1. Defaults to None.
            fraRad_sh2 (float, optional): The fraction of radiation of space heater 2. Defaults to None.
            Q_flow_nominal_sh2 (float, optional): The nominal heat flow rate of space heater 2. Defaults to None.
            T_a_nominal_sh2 (float, optional): The nominal supply air temperature of space heater 2. Defaults to None.
            T_b_nominal_sh2 (float, optional): The nominal return air temperature of the space heater. Defaults to None.
            TAir_nominal_sh2 (float, optional): The nominal air temperature of space heater 2. Defaults to None.
            n_sh2 (float, optional): The nominal heat transfer coefficient of space heater 2. Defaults to None.
            T_boundary (float, optional): The boundary temperature of the building space. Defaults to None.
            infiltration (float, optional): The infiltration rate of the building space. Defaults to None.
            airVolume (float, optional): The air volume of the building space. Defaults to None.
        """
        building_space.BuildingSpace.__init__(self, **kwargs)


        self.C_supply = C_supply#400
        self.C_wall = C_wall#1
        self.C_air = C_air#1
        self.C_int = C_int#1
        self.C_boundary = C_boundary#1
        self.R_out = R_out#1
        self.R_in = R_in#1
        self.R_int = R_int#1
        self.R_boundary = R_boundary#1
        self.f_wall = f_wall#1
        self.f_air = f_air#1
        self.Q_occ_gain = Q_occ_gain#80
        self.CO2_occ_gain = CO2_occ_gain#8.18E-6
        self.CO2_start = CO2_start#400      
        self.fraRad_sh1 = fraRad_sh1#0.35
        self.Q_flow_nominal_sh1 = Q_flow_nominal_sh1#1000
        self.T_a_nominal_sh1 = T_a_nominal_sh1
        self.T_b_nominal_sh1 = T_b_nominal_sh1
        self.TAir_nominal_sh1 = TAir_nominal_sh1
        self.n_sh1 = n_sh1#1.24

        self.fraRad_sh2 = fraRad_sh2#0.35
        self.Q_flow_nominal_sh2 = Q_flow_nominal_sh2#1000
        self.T_a_nominal_sh2 = T_a_nominal_sh2
        self.T_b_nominal_sh2 = T_b_nominal_sh2
        self.TAir_nominal_sh2 = TAir_nominal_sh2
        self.n_sh2 = n_sh2#1.24

        self.T_boundary = T_boundary
        self.infiltration = infiltration
        self.airVolume = airVolume




        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "R2C2_02SH_01adj_0boundary_0outdoor_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.input = {'airFlowRate': tps.Scalar(),
                    'waterFlowRate1': tps.Scalar(),
                    'waterFlowRate2': tps.Scalar(),
                    'supplyAirTemperature': tps.Scalar(),
                    'supplyWaterTemperature': tps.Scalar(),
                    'globalIrradiation': tps.Scalar(),
                    'outdoorTemperature': tps.Scalar(),
                    'numberOfPeople': tps.Scalar(),
                    "outdoorCo2Concentration": tps.Scalar(),
                    "indoorTemperature_adj1": tps.Scalar(),
                    "T_boundary": tps.Scalar(),
                    "m_infiltration": tps.Scalar(),
                    "T_infiltration": tps.Scalar()}
        self.output = {"indoorTemperature": tps.Scalar(), 
                       "indoorCo2Concentration": tps.Scalar(), 
                       "spaceHeaterPower1": tps.Scalar(),
                        "spaceHeaterEnergy1": tps.Scalar(), 
                       "spaceHeaterPower2": tps.Scalar(),
                        "spaceHeaterEnergy2": tps.Scalar(),
                        "spaceHeaterEnergy": tps.Scalar()}
        
        self.FMUinputMap = {'airFlowRate': "m_a_flow",
                    'waterFlowRate1': "m_w_flow1",
                    'waterFlowRate2': "m_w_flow2",
                    'supplyAirTemperature': "T_a_supply",
                    'supplyWaterTemperature': "T_w_supply",
                    'globalIrradiation': "Rad_outdoor",
                    'outdoorTemperature': "T_outdoor",
                    'numberOfPeople': "N_occ",
                    "outdoorCo2Concentration": "CO2_supply",
                    "indoorTemperature_adj1": "T_adj1",
                    "T_boundary": "T_boundary",
                    "m_infiltration": "m_infiltration",
                    "T_infiltration": "T_infiltration"}
        self.FMUoutputMap = {"indoorTemperature": "T_air", 
                             "indoorCo2Concentration": "CO2_concentration",
                             "spaceHeaterPower1": "r2C2_1.rad.Q_flow",
                             "spaceHeaterPower2": "r2C2_1.rad1.Q_flow"}

        self.FMUparameterMap = {"C_supply": "C_supply",
                                "C_wall": "C_wall", 
                                "C_air": "C_air",
                                "C_int": "C_int",
                                "C_boundary": "C_boundary",
                                "R_out": "R_out", 
                                "R_in": "R_in", 
                                "R_int": "R_int",
                                "R_boundary": "R_boundary",
                                "f_wall": "f_wall", 
                                "f_air": "f_air", 
                                "Q_occ_gain": "Q_occ_gain", 
                                "CO2_occ_gain": "CO2_occ_gain", 
                                "CO2_start": "CO2_start",  
                                "airVolume": "airVolume",
                                "fraRad_sh1": "fraRad_sh1", 
                                "Q_flow_nominal_sh1": "Q_flow_nominal_sh1", 
                                "T_a_nominal_sh1": "T_a_nominal_sh1", 
                                "T_b_nominal_sh1": "T_b_nominal_sh1", 
                                "TAir_nominal_sh1": "TAir_nominal_sh1", 
                                "n_sh1": "n_sh1",
                                "fraRad_sh2": "fraRad_sh2", 
                                "Q_flow_nominal_sh2": "Q_flow_nominal_sh2", 
                                "T_a_nominal_sh2": "T_a_nominal_sh2", 
                                "T_b_nominal_sh2": "T_b_nominal_sh2", 
                                "TAir_nominal_sh2": "TAir_nominal_sh2", 
                                "n_sh2": "n_sh2"}


        self.input_conversion = {'airFlowRate': do_nothing,
                                    'waterFlowRate1': do_nothing,
                                    'waterFlowRate2': do_nothing,
                                    'supplyAirTemperature': to_degK_from_degC,
                                    'supplyWaterTemperature': to_degK_from_degC,
                                    'globalIrradiation': do_nothing,
                                    'outdoorTemperature': to_degK_from_degC,
                                    'numberOfPeople': do_nothing,
                                    "outdoorCo2Concentration": do_nothing,
                                    "indoorTemperature_adj1": to_degK_from_degC,
                                    "T_boundary": to_degK_from_degC,
                                    "m_infiltration": do_nothing,
                                    "T_infiltration": get(self.output, "indoorTemperature", conversion=to_degK_from_degC)}
        self.output_conversion = {"indoorTemperature": to_degC_from_degK, 
                                  "indoorCo2Concentration": do_nothing,
                                  "spaceHeaterPower1": change_sign,
                                  "spaceHeaterPower2": change_sign,
                                  "spaceHeaterEnergy1": integrate(self.output, "spaceHeaterPower1", conversion=multiply_const(1/3600/1000)),
                                  "spaceHeaterEnergy2": integrate(self.output, "spaceHeaterPower2", conversion=multiply_const(1/3600/1000)),
                                  "spaceHeaterEnergy": add(self.output, ("spaceHeaterEnergy1", "spaceHeaterEnergy2"))}

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
        Initialize the FMU component.

        Args:
            startTime (float, optional): The start time of the simulation. Defaults to None.
            endTime (float, optional): The end time of the simulation. Defaults to None.
            stepSize (float, optional): The step size of the simulation. Defaults to None.
            model (Model, optional): The model of the simulation. Defaults to None.
        '''
        if self.INITIALIZED:
            self.reset()
        else:
            self.initialize_fmu()
            self.INITIALIZED = True ###
        self.input["T_boundary"] = tps.Scalar(self.T_boundary)
        self.input["m_infiltration"] = tps.Scalar(self.infiltration)
        self.output_conversion["spaceHeaterEnergy1"].v = 0
        self.output_conversion["spaceHeaterEnergy2"].v = 0

        


        