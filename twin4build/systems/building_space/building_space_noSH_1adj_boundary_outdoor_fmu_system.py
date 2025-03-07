import twin4build.systems.utils.fmu.fmu_component as fmu_component
from twin4build.utils.uppath import uppath
import os
from twin4build.utils.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing, change_sign, add_attr, integrate, multiply_const, get
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, SinglePath, Optional_
import twin4build.utils.input_output_types as tps

def get_signature_pattern():
    """
    Get the signature pattern of the FMU component.

    Returns:
        SignaturePattern: The signature pattern of the FMU component.
    """

    node0 = Node(cls=core.S4BLDG.Damper) #supply damper
    node1 = Node(cls=core.S4BLDG.Damper) #return damper
    node2 = Node(cls=core.S4BLDG.BuildingSpace)
    node5 = Node(cls=core.S4BLDG.Schedule)
    node6 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=(core.S4BLDG.Coil, core.S4BLDG.AirToAirHeatRecovery, core.S4BLDG.Fan))
    node9 = Node(cls=core.S4BLDG.BuildingSpace)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem", priority=159)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=core.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node2, object=node6, predicate=core.S4SYST.connectedTo))
    sp.add_triple(SinglePath(subject=node0, object=node7, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node9, object=node2, predicate=core.S4SYST.connectedTo))

    sp.add_input("airFlowRate", node0)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, ("outletAirTemperature", "primaryTemperatureOut", "outletAirTemperature"))
    sp.add_input("indoorTemperature_adj1", node9, "indoorTemperature")
    # sp.add_input("supplyWaterTemperature", node10, "measuredValue") ## for Applied energy paper

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
    node5 = Node(cls=core.S4BLDG.Schedule) #return valve
    node6 = Node(cls=core.S4BLDG.OutdoorEnvironment)
    node7 = Node(cls=core.SAREF.Sensor)
    node8 = Node(cls=core.SAREF.Temperature)
    node9 = Node(cls=core.S4BLDG.BuildingSpace)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem", priority=158)

    sp.add_triple(Exact(subject=node0, object=node2, predicate=core.FSO.suppliesFluidTo))
    sp.add_triple(Exact(subject=node1, object=node2, predicate=core.FSO.hasFluidReturnedBy))
    sp.add_triple(Exact(subject=node2, object=node5, predicate=core.SAREF.hasProfile))
    sp.add_triple(Exact(subject=node2, object=node6, predicate=core.S4SYST.connectedTo))
    sp.add_triple(SinglePath(subject=node0, object=node7, predicate=core.FSO.hasFluidSuppliedBy))
    sp.add_triple(Exact(subject=node7, object=node8, predicate=core.SAREF.observes))
    sp.add_triple(Exact(subject=node9, object=node2, predicate=core.S4SYST.connectedTo))

    sp.add_input("airFlowRate", node0)
    sp.add_input("numberOfPeople", node5, "scheduleValue")
    sp.add_input("outdoorTemperature", node6, "outdoorTemperature")
    sp.add_input("outdoorCo2Concentration", node6, "outdoorCo2Concentration")
    sp.add_input("globalIrradiation", node6, "globalIrradiation")
    sp.add_input("supplyAirTemperature", node7, "measuredValue")
    sp.add_input("indoorTemperature_adj1", node9, "indoorTemperature")

    sp.add_modeled_node(node2)
    return sp

class BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem(fmu_component.FMUComponent):
    """
    A class representing an FMU of a building space with 1 adjacent spaces, balanced supply and return ventilation, and an outdoor boundary. (no space heater)
    """
    sp = [get_signature_pattern(), get_signature_pattern_sensor()]
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
                T_boundary=None,
                infiltration=None,
                airVolume=None,
                **kwargs):
        super().__init__(**kwargs)

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
        self.T_boundary = T_boundary
        self.infiltration = infiltration
        self.airVolume = airVolume

        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "R2C2_0noSH_01adj_0boundary_0outdoor_0FMU.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = fmu_component.unzip_fmu(self.fmu_path)

        self.input = {'airFlowRate': tps.Scalar(),
                    'supplyAirTemperature': tps.Scalar(),
                    'globalIrradiation': tps.Scalar(),
                    'outdoorTemperature': tps.Scalar(),
                    'numberOfPeople': tps.Scalar(),
                    "outdoorCo2Concentration": tps.Scalar(),
                    "indoorTemperature_adj1": tps.Scalar(),
                    "T_boundary": tps.Scalar(),
                    "m_infiltration": tps.Scalar(),
                    "T_infiltration": tps.Scalar()}
        self.output = {"indoorTemperature": tps.Scalar(), 
                       "indoorCo2Concentration": tps.Scalar()}
        
        self.FMUinputMap = {'airFlowRate': "m_a_flow",
                    'supplyAirTemperature': "T_a_supply",
                    'globalIrradiation': "Rad_outdoor",
                    'outdoorTemperature': "T_outdoor",
                    'numberOfPeople': "N_occ",
                    "outdoorCo2Concentration": "CO2_supply",
                    "indoorTemperature_adj1": "T_adj1",
                    "T_boundary": "T_boundary",
                    "m_infiltration": "m_infiltration",
                    "T_infiltration": "T_infiltration"}
        self.FMUoutputMap = {"indoorTemperature": "T_air", 
                             "indoorCo2Concentration": "CO2_concentration"}

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
                                "airVolume": "airVolume",}
        
    
        self.input_conversion = {'airFlowRate': do_nothing,
                                    'supplyAirTemperature': to_degK_from_degC,
                                    'globalIrradiation': do_nothing,
                                    'outdoorTemperature': to_degK_from_degC,
                                    'numberOfPeople': do_nothing,
                                    "outdoorCo2Concentration": do_nothing,
                                    "indoorTemperature_adj1": to_degK_from_degC,
                                    "T_boundary": to_degK_from_degC,
                                    "m_infiltration": do_nothing,
                                    "T_infiltration": get(self.output, "indoorTemperature", conversion=to_degK_from_degC)}
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

        


        