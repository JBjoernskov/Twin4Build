from twin4build.saref4syst.system import System
from twin4build.logger.Logging import Logging
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes
import twin4build.base as base
import twin4build.components as components
logger = Logging.get_logger("ai_logfile")


def get_signature_pattern():
    node0 = Node(cls=(base.Schedule,), id="<Schedule<SUB>1</SUB>>")
    node1 = Node(cls=(base.BuildingSpace,), id="<BuildingSpace<SUB>2</SUB>>")
    node2 = Node(cls=(base.Damper,), id="<Damper<SUB>3</SUB>>")
    node3 = Node(cls=(base.Damper,), id="<Damper<SUB>4</SUB>>")
    node4 = Node(cls=(base.Co2), id="<Co2<SUB>5</SUB>>")
    node5 = Node(cls=(base.Sensor,), id="<Sensor<SUB>6</SUB>>")
    node6 = Node(cls=(base.Sensor,), id="<Sensor<SUB>7</SUB>>")
    node7 = Node(cls=(base.OpeningPosition), id="<OpeningPosition<SUB>8</SUB>>")
    sp = SignaturePattern(ownedBy="ScheduleSystem", priority=100)
    sp.add_edge(Exact(object=node1, subject=node0, predicate="hasProfile"))
    sp.add_edge(Exact(object=node1, subject=node4, predicate="hasProperty"))
    sp.add_edge(Exact(object=node5, subject=node4, predicate="observes"))
    sp.add_edge(Exact(object=node2, subject=node1, predicate="suppliesFluidTo"))
    sp.add_edge(Exact(object=node3, subject=node1, predicate="hasFluidReturnedBy"))
    sp.add_edge(Exact(object=node6, subject=node7, predicate="observes"))
    sp.add_edge(Exact(object=node2, subject=node7, predicate="hasProperty"))

    sp.add_modeled_node(node0)
    return sp


class OccupancySystem(base.Schedule, System):
    # sp = [get_signature_pattern()]
    def __init__(self,
                airVolume=None,
                outdoorCo2Concentration=400,
                infiltration=0.005,
                generationCo2Concentration=0.0042/1000*1.225,
                filename_supplyAirFlowRate=None,
                filename_exhaustAirFlowRate=None,
                filename_indoorCO2Concentration=None,
                df_input=None,
                datecolumn=None,
                valuecolumn=None,
                **kwargs):
        super().__init__(**kwargs)
        self.airVolume = airVolume
        self.outdoorCo2Concentration = outdoorCo2Concentration
        self.infiltration = infiltration
        self.generationCo2Concentration = generationCo2Concentration
        self.filename = [filename_supplyAirFlowRate, filename_exhaustAirFlowRate, filename_indoorCO2Concentration]
        self.df_input = df_input
        self.datecolumn = datecolumn
        self.valuecolumn = valuecolumn

        
        self.input = {"supplyAirFlowRate": None,
                      "exhaustAirFlowRate": None,
                      "indoorCO2Concentration": None,}
        self.output = {"scheduleValue": None}

        self._config = {"parameters": ["airVolume", "outdoorCo2Concentration", "infiltration", "generationCo2Concentration"],
                        "readings": {"filename": self.filename,
                                     "df_input": self.df_input,
                                     "datecolumn": self.datecolumn,
                                     "valuecolumn": self.valuecolumn}}



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


        (self.filename_supplyAirFlowRate, self.filename_exhaustAirFlowRate, self.filename_indoorCO2Concentration) = self.filename

        self.do_step_instance_supplyDamperPosition = components.TimeSeriesInputSystem(id=f"supplyDamperPosition", filename=self.filename_supplyAirFlowRate, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
        self.do_step_instance_supplyDamperPosition.output = {"supplyDamperPosition": None}
        self.do_step_instance_supplyDamperPosition.initialize(startTime,
                                        endTime,
                                        stepSize)

        self.do_step_instance_exhaustDamperPosition = components.TimeSeriesInputSystem(id=f"exhaustDamperPosition", filename=self.filename_exhaustAirFlowRate, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
        self.do_step_instance_exhaustDamperPosition.output = {"exhaustDamperPosition": None}
        self.do_step_instance_exhaustDamperPosition.initialize(startTime,
                                        endTime,
                                        stepSize)
        
        self.do_step_instance_indoorCO2Concentration = components.TimeSeriesInputSystem(id=f"indoorCO2Concentration", filename=self.filename_indoorCO2Concentration, datecolumn=self.datecolumn, valuecolumn=self.valuecolumn)
        self.do_step_instance_indoorCO2Concentration.output = {"indoorCO2Concentration": None}
        self.do_step_instance_indoorCO2Concentration.initialize(startTime,
                                        endTime,
                                        stepSize)
        

        (modeled_match_nodes, (component_cls, sp, group)) = model.instance_to_group_map[self]

        space_node = sp.get_node_by_id("<BuildingSpace<SUB>2</SUB>>")
        modeled_space = next(iter(group[space_node]))
        modeled_space = model.instance_map_reversed[modeled_space]
        self.airVolume = modeled_space.airVolume
        self.outdoorCo2Concentration = modeled_space.C_supply
        self.infiltration = modeled_space.infiltration
        self.generationCo2Concentration = modeled_space.CO2_occ_gain
        self.previous_indoorCO2Concentration = self.outdoorCo2Concentration
        self.airMass = self.airVolume*1.225

        supply_damper_node = sp.get_node_by_id("<Damper<SUB>3</SUB>>")
        exhaust_damper_node = sp.get_node_by_id("<Damper<SUB>4</SUB>>")
        modeled_supply_damper = next(iter(group[supply_damper_node]))
        modeled_exhaust_damper = next(iter(group[exhaust_damper_node]))
        
        modeled_supply_damper = model.instance_map_reversed[modeled_supply_damper]
        modeled_exhaust_damper = model.instance_map_reversed[modeled_exhaust_damper]

        self.do_step_instance_supplyDamper = components.DamperSystem(**model.get_object_properties(modeled_supply_damper))
        self.do_step_instance_supplyDamper.initialize(startTime,
                                        endTime,
                                        stepSize)
        
        self.do_step_instance_exhaustDamper = components.DamperSystem(**model.get_object_properties(modeled_exhaust_damper))
        self.do_step_instance_exhaustDamper.initialize(startTime,
                                        endTime,
                                        stepSize)

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        # diff equation
        # self.output["occupancy"] = (self.airMass*(self.input["indoorCO2Concentration"]-self.previous_indoorCO2Concentration)/stepSize - self.outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + self.infiltration) + self.input["indoorCO2Concentration"]*self.input["exhaustAirFlowRate"])/(self.generationCo2Concentration*1e+6)
        # Steady state. Behaves much better and is more stable

        self.do_step_instance_supplyDamperPosition.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        self.do_step_instance_exhaustDamperPosition.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        self.do_step_instance_indoorCO2Concentration.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)

        self.do_step_instance_supplyDamper.input["damperPosition"] = self.do_step_instance_supplyDamperPosition.output["supplyDamperPosition"]
        self.do_step_instance_exhaustDamper.input["damperPosition"] = self.do_step_instance_supplyDamperPosition.output["supplyDamperPosition"]
        
        self.do_step_instance_supplyDamper.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        self.do_step_instance_exhaustDamper.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)

        self.input["supplyAirFlowRate"] = self.do_step_instance_supplyDamper.output["airFlowRate"]
        self.input["exhaustAirFlowRate"] = self.do_step_instance_exhaustDamper.output["airFlowRate"]
        self.input["indoorCO2Concentration"] = self.do_step_instance_indoorCO2Concentration.output["indoorCO2Concentration"]
        self.output["scheduleValue"] = (-self.outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + self.infiltration) + self.input["indoorCO2Concentration"]*(self.input["exhaustAirFlowRate"]+self.infiltration))/(self.generationCo2Concentration*1e+6)
        self.previous_indoorCO2Concentration = self.input["indoorCO2Concentration"]