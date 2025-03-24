import twin4build.systems.utils.fmu.fmu_component as fmu_component
from twin4build.utils.uppath import uppath
import os
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, change_sign, add_attr, integrate, multiply_const
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern for the building space 0 adjacent boundary outdoor FMU system.
    """

    node0 = Node(cls=core.S4BLDG.Damper) #supply damper
    node1 = Node(cls=core.S4BLDG.Damper) #return damper
    node2 = Node(cls=core.S4BLDG.BuildingSpace)
    node3 = Node(cls=core.S4BLDG.Valve) #supply valve
    node4 = Node(cls=core.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.S4BLDG.Schedule) #return valve
    node6 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=(core.S4BLDG.Coil, core.S4BLDG.AirToAirHeatRecovery, core.S4BLDG.Fan))
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="BuildingSpace0AdjBoundaryOutdoorFMUSystem", priority=60)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node3, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node4, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=core.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node2, object=node6, predicate=core.S4SYST.connectedTo))
    sp.add_triple(SinglePath(subject=node0, object=node7, predicate=core.FSO.hasFluidSuppliedBy))

    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue") ##############################
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))

    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)

    return sp

def get_signature_pattern_sensor():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern of the FMU component.
    """

    node0 = Node(cls=core.S4BLDG.Damper) #supply damper
    node1 = Node(cls=core.S4BLDG.Damper) #return damper
    node2 = Node(cls=core.S4BLDG.BuildingSpace)
    node3 = Node(cls=core.S4BLDG.Valve) #supply valve
    node4 = Node(cls=core.S4BLDG.SpaceHeater)
    node5 = Node(cls=core.S4BLDG.Schedule) #return valve
    node6 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=core.SAREF.Sensor)
    node8 = Node(cls=core.SAREF.Temperature)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="BuildingSpace0AdjBoundaryOutdoorFMUSystem", priority=59)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node3, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node4, object=node2, predicate=core.S4BLDG.isContainedIn))
    sp.add_triple(Exact(subject=node3, object=node4, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=core.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node2, object=node6, predicate=core.S4SYST.connectedTo))
    sp.add_triple(SinglePath(subject=node0, object=node7, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node7, object=node8, predicate=core.SAREF.observes))

    sp.add_input("airFlowRate", node0)
    sp.add_input("waterFlowRate", node3)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")

    sp.add_modeled_node(node4)
    sp.add_modeled_node(node2)
    return sp

class BuildingSpace0AdjBoundaryOutdoorFMUSystem(fmu_component.FMUComponent):
    sp = [get_signature_pattern()]
    def __init__(self,
                C_supply=None,
                C_air=None,
                C_int=None,
                C_boundary=None,
                R_out=None,
                R_in=None,
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
        self.R_out = R_out#1
        self.R_in = R_in#1
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
        fmu_filename = "R2C2_00adj_0boundary_0outdoor_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = fmu_component.unzip_fmu(self.fmu_path)

        self.input = {'airFlowRate': tps.Scalar(),
                    'waterFlowRate': tps.Scalar(),
                    'supplyAirTemperature': tps.Scalar(),
                    'supplyWaterTemperature': tps.Scalar(),
                    'globalIrradiation': tps.Scalar(),
                    'outdoorTemperature': tps.Scalar(),
                    'numberOfPeople': tps.Scalar(),
                    "outdoorCo2Concentration": tps.Scalar(),
                    "T_boundary": tps.Scalar()}
        self.output = {"indoorTemperature": tps.Scalar(), 
                       "indoorCo2Concentration": tps.Scalar(), 
                       "spaceHeaterPower": tps.Scalar(),
                        "spaceHeaterEnergy": tps.Scalar()}
        
        self.FMUinputMap = {'airFlowRate': "m_a_flow",
                            'waterFlowRate': "m_w_flow",
                            'supplyAirTemperature': "T_a_supply",
                            'supplyWaterTemperature': "T_w_supply",
                            'globalIrradiation': "Rad_outdoor",
                            'outdoorTemperature': "T_outdoor",
                            'numberOfPeople': "N_occ",
                            "outdoorCo2Concentration": "CO2_supply",
                            "T_boundary": "T_boundary"}
        self.FMUoutputMap = {"indoorTemperature": "T_air", 
                             "indoorCo2Concentration": "CO2_concentration",
                             "spaceHeaterPower": "r2C2_1.rad.Q_flow"}

        self.FMUparameterMap = {"C_supply": "C_supply",
                                "C_air": "C_air",
                                "C_boundary": "C_boundary",
                                "R_out": "R_out",
                                "R_in": "R_in",
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
                                    'globalIrradiation': do_nothing,
                                    'outdoorTemperature': to_degK_from_degC,
                                    'numberOfPeople': do_nothing,
                                    "outdoorCo2Concentration": do_nothing,
                                    "T_boundary": to_degK_from_degC}
        self.output_conversion = {"indoorTemperature": to_degC_from_degK, 
                                  "indoorCo2Concentration": do_nothing,
                                  "spaceHeaterPower": change_sign,
                                  "spaceHeaterEnergy": integrate(self.output, "spaceHeaterPower", conversion=multiply_const(1/3600/1000))}

        self.INITIALIZED = False
        self._config = {"parameters": list(self.FMUparameterMap.keys()) + ["T_boundary", "infiltration"]}

        self.optional_inputs = ["T_boundary"]

    @property
    def config(self):
        """
        Get the configuration of the FMU component.

        Returns:
            dict: The configuration of the FMU component.
        """
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
        self.output_conversion["spaceHeaterEnergy"].v = 0


        


        