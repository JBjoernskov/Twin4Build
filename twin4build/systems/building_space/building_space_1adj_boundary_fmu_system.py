import twin4build.systems.utils.fmu.fmu_component as fmu_component
from twin4build.utils.uppath import uppath
import os
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, change_sign, add_attr, integrate, multiply_const, get
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    node0 = Node(cls=core.S4BLDG.Damper) #supply damper
    node1 = Node(cls=core.S4BLDG.Damper) #return damper
    node2 = Node(cls=core.S4BLDG.BuildingSpace)
    node3 = Node(cls=core.S4BLDG.Valve) #supply valve
    node4 = Node(cls=core.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.S4BLDG.Schedule) #return valve
    node7 = Node(cls=core.SAREF.Sensor)
    node8 = Node(cls=core.SAREF.Temperature)
    node9 = Node(cls=core.S4BLDG.BuildingSpace)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="BuildingSpace1AdjBoundaryFMUSystem", priority=150)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node3, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node4, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=core.SAREF.hasProfile))
    sp.add_triple(SinglePath(subject=node7, object=node0, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node7, object=node8, predicate=core.SAREF.observes))
    sp.add_triple(Exact(subject=node9, object=node2, predicate=core.S4SYST.connectedTo))

    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    sp.add_input("indoorTemperature_adj1", node9, "indoorTemperature")

    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)

    # cs.add_parameter("globalIrradiation", node2, "globalIrradiation")

    return sp

class BuildingSpace1AdjBoundaryFMUSystem(fmu_component.FMUComponent):
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
                T_boundary=None,
                infiltration=None,
                airVolume=None,
                **kwargs):
        super().__init__(**kwargs)
        
        # Keep the rest of the initialization
        self.C_supply = C_supply
        self.C_air = C_air
        self.C_int = C_int
        self.C_boundary = C_boundary
        self.R_int = R_int
        self.R_boundary = R_boundary
        self.Q_occ_gain = Q_occ_gain
        self.CO2_occ_gain = CO2_occ_gain
        self.CO2_start = CO2_start
        self.fraRad_sh = fraRad_sh
        self.Q_flow_nominal_sh = Q_flow_nominal_sh
        self.T_a_nominal_sh = T_a_nominal_sh
        self.T_b_nominal_sh = T_b_nominal_sh
        self.TAir_nominal_sh = TAir_nominal_sh
        self.n_sh = n_sh
        self.T_boundary = T_boundary
        self.infiltration = infiltration
        self.airVolume = airVolume

        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "R2C2_01adj_0boundary_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = fmu_component.unzip_fmu(self.fmu_path)

        self.input = {'airFlowRate': tps.Scalar(),
                    'waterFlowRate': tps.Scalar(),
                    'supplyAirTemperature': tps.Scalar(),
                    'supplyWaterTemperature': tps.Scalar(),
                    'numberOfPeople': tps.Scalar(),
                    "outdoorCo2Concentration": tps.Scalar(),
                    "indoorTemperature_adj1": tps.Scalar(),
                    "T_boundary": tps.Scalar()}
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
                    "T_boundary": "T_boundary"}
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
        
        self.input_conversion = {'airFlowRate': add_attr(self, "infiltration"),
                                    'waterFlowRate': do_nothing,
                                    'supplyAirTemperature': to_degK_from_degC,
                                    'supplyWaterTemperature': to_degK_from_degC,
                                    'numberOfPeople': do_nothing,
                                    "outdoorCo2Concentration": do_nothing,
                                    "indoorTemperature_adj1": to_degK_from_degC,
                                    "T_boundary": to_degK_from_degC}
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
        self.output_conversion["spaceHeaterEnergy"].v = 0

        


        