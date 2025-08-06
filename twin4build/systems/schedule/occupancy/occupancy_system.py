# Standard library imports
import datetime
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.systems as systems
import twin4build.utils.types as tps
from twin4build.translator.translator import Exact, Node, SignaturePattern, SinglePath


def get_signature_pattern():
    node0 = Node(cls=(core.namespace.S4BLDG.Schedule))
    node1 = Node(cls=(core.namespace.S4BLDG.BuildingSpace))
    node2 = Node(cls=(core.namespace.S4BLDG.Damper))
    node3 = Node(cls=(core.namespace.S4BLDG.Damper))
    node4 = Node(cls=(core.namespace.SAREF.Co2))
    node5 = Node(cls=(core.namespace.SAREF.Sensor))
    node6 = Node(cls=(core.namespace.SAREF.Sensor))
    node7 = Node(cls=(core.namespace.SAREF.OpeningPosition))
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="ScheduleSystem", priority=100
    )
    sp.add_triple(
        Exact(subject=node1, object=node0, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_triple(
        Exact(subject=node1, object=node4, predicate=core.namespace.SAREF.hasProperty)
    )
    sp.add_triple(
        Exact(subject=node5, object=node4, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node2, object=node1, predicate=core.namespace.FSO.suppliesFluidTo)
    )
    sp.add_triple(
        Exact(
            subject=node3, object=node1, predicate=core.namespace.FSO.hasFluidReturnedBy
        )
    )
    sp.add_triple(
        Exact(subject=node6, object=node7, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node2, object=node7, predicate=core.namespace.SAREF.hasProperty)
    )

    sp.add_modeled_node(node0)
    return sp


class OccupancySystem(core.System):
    # sp = [get_signature_pattern()]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input = {
            "supplyAirFlowRate": tps.Scalar(),
            "exhaustAirFlowRate": tps.Scalar(),
            "indoorCO2Concentration": tps.Scalar(),
        }
        self.output = {"scheduleValue": tps.Scalar()}
        self.optional_inputs = [
            "supplyAirFlowRate",
            "exhaustAirFlowRate",
            "indoorCO2Concentration",
        ]

        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        model = simulator.model
        (modeled_match_nodes, (component_cls, sp, groups)) = (
            model.instance_to_group_map[self]
        )

        space_node = sp.get_node_by_id("<BuildingSpace<SUB>2</SUB>>")  # TODO
        modeled_space = groups[0][space_node]
        modeled_space = model.sem2sim_map[modeled_space]
        self.airVolume = modeled_space.airVolume
        self.outdoorCo2Concentration = modeled_space.C_supply
        self.infiltration = modeled_space.infiltration
        self.generationCo2Concentration = modeled_space.CO2_occ_gain
        self.previous_indoorCO2Concentration = modeled_space.CO2_start
        self.airMass = self.airVolume * 1.225

        supply_damper_node = sp.get_node_by_id("<Damper<SUB>3</SUB>>")  # TODO
        exhaust_damper_node = sp.get_node_by_id("<Damper<SUB>4</SUB>>")  # TODO
        modeled_supply_damper = groups[0][supply_damper_node]
        modeled_exhaust_damper = groups[0][exhaust_damper_node]

        modeled_supply_damper = model.sem2sim_map[modeled_supply_damper]
        modeled_exhaust_damper = model.sem2sim_map[modeled_exhaust_damper]

        self.do_step_instance_supplyDamper = systems.DamperSystem(
            **model.get_object_properties(modeled_supply_damper)
        )
        self.do_step_instance_supplyDamper.initialize(startTime, endTime, stepSize)

        self.do_step_instance_exhaustDamper = systems.DamperSystem(
            **model.get_object_properties(modeled_exhaust_damper)
        )
        self.do_step_instance_exhaustDamper.initialize(startTime, endTime, stepSize)

        damper_position_sensor_node = sp.get_node_by_id("<Sensor<SUB>7</SUB>>")  # TODO
        modeled_damper_position_sensor = groups[0][damper_position_sensor_node]
        modeled_damper_position_sensor = model.sem2sim_map[
            modeled_damper_position_sensor
        ]
        filename_damper_position = modeled_damper_position_sensor.filename
        datecolumn_damper_position = self.datecolumn = (
            modeled_damper_position_sensor.datecolumn
        )
        valuecolumn_damper_position = self.valuecolumn = (
            modeled_damper_position_sensor.valuecolumn
        )

        co2_sensor_node = sp.get_node_by_id("<Sensor<SUB>6</SUB>>")  # TODO
        modeled_co2_sensor = groups[0][co2_sensor_node]
        modeled_co2_sensor = model.sem2sim_map[modeled_co2_sensor]
        filename_co2 = modeled_co2_sensor.filename
        datecolumn_co2 = self.datecolumn = modeled_co2_sensor.datecolumn
        valuecolumn_co2 = self.valuecolumn = modeled_co2_sensor.valuecolumn

        self.do_step_instance_supplyDamperPosition = systems.TimeSeriesInputSystem(
            id="supplyDamperPosition",
            filename=filename_damper_position,
            datecolumn=datecolumn_damper_position,
            valuecolumn=valuecolumn_damper_position,
        )
        self.do_step_instance_supplyDamperPosition.output = {"value": tps.Scalar()}
        self.do_step_instance_supplyDamperPosition.initialize(
            startTime, endTime, stepSize, simulator
        )

        self.do_step_instance_exhaustDamperPosition = systems.TimeSeriesInputSystem(
            id="exhaustDamperPosition",
            filename=filename_damper_position,
            datecolumn=datecolumn_damper_position,
            valuecolumn=valuecolumn_damper_position,
        )
        self.do_step_instance_exhaustDamperPosition.output = {"value": tps.Scalar()}
        self.do_step_instance_exhaustDamperPosition.initialize(
            startTime, endTime, stepSize, simulator
        )

        self.do_step_instance_indoorCO2Concentration = systems.TimeSeriesInputSystem(
            id="indoorCO2Concentration",
            filename=filename_co2,
            datecolumn=datecolumn_co2,
            valuecolumn=valuecolumn_co2,
        )
        self.do_step_instance_indoorCO2Concentration.output = {"value": tps.Scalar()}
        self.do_step_instance_indoorCO2Concentration.initialize(
            startTime, endTime, stepSize, simulator
        )

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:

        self.do_step_instance_supplyDamperPosition.do_step(
            secondTime=secondTime,
            dateTime=dateTime,
            stepSize=stepSize,
            stepIndex=stepIndex,
        )
        self.do_step_instance_exhaustDamperPosition.do_step(
            secondTime=secondTime,
            dateTime=dateTime,
            stepSize=stepSize,
            stepIndex=stepIndex,
        )
        self.do_step_instance_indoorCO2Concentration.do_step(
            secondTime=secondTime,
            dateTime=dateTime,
            stepSize=stepSize,
            stepIndex=stepIndex,
        )

        self.do_step_instance_supplyDamper.input["damperPosition"].set(
            self.do_step_instance_supplyDamperPosition.output["value"], stepIndex
        )
        self.do_step_instance_exhaustDamper.input["damperPosition"].set(
            self.do_step_instance_exhaustDamperPosition.output["value"], stepIndex
        )

        self.do_step_instance_supplyDamper.do_step(
            secondTime=secondTime,
            dateTime=dateTime,
            stepSize=stepSize,
            stepIndex=stepIndex,
        )
        self.do_step_instance_exhaustDamper.do_step(
            secondTime=secondTime,
            dateTime=dateTime,
            stepSize=stepSize,
            stepIndex=stepIndex,
        )

        self.input["supplyAirFlowRate"].set(
            self.do_step_instance_supplyDamper.output["value"], stepIndex
        )
        self.input["exhaustAirFlowRate"].set(
            self.do_step_instance_exhaustDamper.output["value"], stepIndex
        )
        self.input["indoorCO2Concentration"].set(
            self.do_step_instance_indoorCO2Concentration.output["value"], stepIndex
        )

        # Steady state.
        # self.output["scheduleValue"] = (-self.outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + self.infiltration) + self.input["indoorCO2Concentration"]*(self.input["exhaustAirFlowRate"]+self.infiltration))/(self.generationCo2Concentration*1e+6)

        # diff equation
        self.output["scheduleValue"].set(
            (
                self.airMass
                * (
                    self.input["indoorCO2Concentration"]
                    - self.previous_indoorCO2Concentration
                )
                / stepSize
                - self.outdoorCo2Concentration
                * (self.input["supplyAirFlowRate"] + self.infiltration)
                + self.input["indoorCO2Concentration"]
                * (self.input["exhaustAirFlowRate"] + self.infiltration)
            )
            / (self.generationCo2Concentration * 1e6),
            stepIndex,
        )
        if self.output["scheduleValue"] < 0:
            self.output["scheduleValue"].set(0, stepIndex)
        self.previous_indoorCO2Concentration = self.input["indoorCO2Concentration"]
