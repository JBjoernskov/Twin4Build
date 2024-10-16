from twin4build.saref4syst.system import System
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes
import twin4build.base as base
import twin4build.components as components
import twin4build.utils.input_output_types as tps


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
    sp = [get_signature_pattern()]
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        
        self.input = {"supplyAirFlowRate": tps.Scalar(),
                      "exhaustAirFlowRate": tps.Scalar(),
                      "indoorCO2Concentration": tps.Scalar(),}
        self.output = {"scheduleValue": tps.Scalar()}
        self.optional_inputs = ["supplyAirFlowRate", "exhaustAirFlowRate", "indoorCO2Concentration"]

        self._config = {"parameters": []}



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

        (modeled_match_nodes, (component_cls, sp, groups)) = model.instance_to_group_map[self]

        space_node = sp.get_node_by_id("<BuildingSpace<SUB>2</SUB>>")
        modeled_space = next(iter(groups[0][space_node]))
        modeled_space = model.instance_map_reversed[modeled_space]
        self.airVolume = modeled_space.airVolume
        self.outdoorCo2Concentration = modeled_space.C_supply
        self.infiltration = modeled_space.infiltration
        self.generationCo2Concentration = modeled_space.CO2_occ_gain
        self.previous_indoorCO2Concentration = modeled_space.CO2_start
        self.airMass = self.airVolume*1.225

        supply_damper_node = sp.get_node_by_id("<Damper<SUB>3</SUB>>")
        exhaust_damper_node = sp.get_node_by_id("<Damper<SUB>4</SUB>>")
        modeled_supply_damper = next(iter(groups[0][supply_damper_node]))
        modeled_exhaust_damper = next(iter(groups[0][exhaust_damper_node]))
        
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
        

        damper_position_sensor_node = sp.get_node_by_id("<Sensor<SUB>7</SUB>>")
        modeled_damper_position_sensor = next(iter(groups[0][damper_position_sensor_node]))
        modeled_damper_position_sensor = model.instance_map_reversed[modeled_damper_position_sensor]
        filename_damper_position = modeled_damper_position_sensor.filename
        datecolumn_damper_position=self.datecolumn = modeled_damper_position_sensor.datecolumn
        valuecolumn_damper_position=self.valuecolumn = modeled_damper_position_sensor.valuecolumn

        co2_sensor_node = sp.get_node_by_id("<Sensor<SUB>6</SUB>>")
        modeled_co2_sensor = next(iter(groups[0][co2_sensor_node]))
        modeled_co2_sensor = model.instance_map_reversed[modeled_co2_sensor]
        filename_co2 = modeled_co2_sensor.filename
        datecolumn_co2=self.datecolumn = modeled_co2_sensor.datecolumn
        valuecolumn_co2=self.valuecolumn = modeled_co2_sensor.valuecolumn
        
        self.do_step_instance_supplyDamperPosition = components.TimeSeriesInputSystem(id=f"supplyDamperPosition", filename=filename_damper_position, datecolumn=datecolumn_damper_position, valuecolumn=valuecolumn_damper_position)
        self.do_step_instance_supplyDamperPosition.output = {"supplyDamperPosition": tps.Scalar()}
        self.do_step_instance_supplyDamperPosition.initialize(startTime,
                                        endTime,
                                        stepSize)

        self.do_step_instance_exhaustDamperPosition = components.TimeSeriesInputSystem(id=f"exhaustDamperPosition", filename=filename_damper_position, datecolumn=datecolumn_damper_position, valuecolumn=valuecolumn_damper_position)
        self.do_step_instance_exhaustDamperPosition.output = {"exhaustDamperPosition": tps.Scalar()}
        self.do_step_instance_exhaustDamperPosition.initialize(startTime,
                                        endTime,
                                        stepSize)
        
        self.do_step_instance_indoorCO2Concentration = components.TimeSeriesInputSystem(id=f"indoorCO2Concentration", filename=filename_co2, datecolumn=datecolumn_co2, valuecolumn=valuecolumn_co2)
        self.do_step_instance_indoorCO2Concentration.output = {"indoorCO2Concentration": tps.Scalar()}
        self.do_step_instance_indoorCO2Concentration.initialize(startTime,
                                        endTime,
                                        stepSize)

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):


        self.do_step_instance_supplyDamperPosition.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        self.do_step_instance_exhaustDamperPosition.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        self.do_step_instance_indoorCO2Concentration.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)

        self.do_step_instance_supplyDamper.input["damperPosition"].set(self.do_step_instance_supplyDamperPosition.output["supplyDamperPosition"])
        self.do_step_instance_exhaustDamper.input["damperPosition"].set(self.do_step_instance_supplyDamperPosition.output["supplyDamperPosition"])
        
        self.do_step_instance_supplyDamper.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)
        self.do_step_instance_exhaustDamper.do_step(secondTime=secondTime, dateTime=dateTime, stepSize=stepSize)

        self.input["supplyAirFlowRate"].set(self.do_step_instance_supplyDamper.output["airFlowRate"])
        self.input["exhaustAirFlowRate"].set(self.do_step_instance_exhaustDamper.output["airFlowRate"])
        self.input["indoorCO2Concentration"].set(self.do_step_instance_indoorCO2Concentration.output["indoorCO2Concentration"])
        
        # Steady state.
        # self.output["scheduleValue"] = (-self.outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + self.infiltration) + self.input["indoorCO2Concentration"]*(self.input["exhaustAirFlowRate"]+self.infiltration))/(self.generationCo2Concentration*1e+6)
        
        # diff equation
        self.output["scheduleValue"].set((self.airMass*(self.input["indoorCO2Concentration"]-self.previous_indoorCO2Concentration)/stepSize - self.outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + self.infiltration) + self.input["indoorCO2Concentration"]*(self.input["exhaustAirFlowRate"]+self.infiltration))/(self.generationCo2Concentration*1e+6))
        if self.output["scheduleValue"] < 0: self.output["scheduleValue"].set(0)
        self.previous_indoorCO2Concentration = self.input["indoorCO2Concentration"]