# Standard library imports
import datetime
import random
from random import randrange
from typing import Optional

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import Exact, Node, SignaturePattern, SinglePath


def get_signature_pattern():
    node0 = Node(cls=(core.namespace.S4BLDG.Schedule))
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="ScheduleSystem", priority=10
    )
    sp.add_modeled_node(node0)
    return sp


class ScheduleSystem(core.System):
    sp = [get_signature_pattern()]

    def __init__(
        self,
        weekDayRulesetDict=None,
        weekendRulesetDict=None,
        mondayRulesetDict=None,
        tuesdayRulesetDict=None,
        wednesdayRulesetDict=None,
        thursdayRulesetDict=None,
        fridayRulesetDict=None,
        saturdayRulesetDict=None,
        sundayRulesetDict=None,
        add_noise=False,
        useSpreadsheet=False,
        useDatabase=False,
        filename=None,
        datecolumn=0,
        valuecolumn=1,
        uuid=None,
        name=None,
        dbconfig=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            useSpreadsheet == False or useDatabase == False
        ), "useSpreadsheet and useDatabase cannot both be True."
        self.weekDayRulesetDict = weekDayRulesetDict
        self.weekendRulesetDict = weekendRulesetDict
        self.mondayRulesetDict = mondayRulesetDict
        self.tuesdayRulesetDict = tuesdayRulesetDict
        self.wednesdayRulesetDict = wednesdayRulesetDict
        self.thursdayRulesetDict = thursdayRulesetDict
        self.fridayRulesetDict = fridayRulesetDict
        self.saturdayRulesetDict = saturdayRulesetDict
        self.sundayRulesetDict = sundayRulesetDict
        self.add_noise = add_noise
        self.useSpreadsheet = useSpreadsheet
        self.useDatabase = useDatabase
        self.filename = filename
        self.datecolumn = datecolumn
        self.valuecolumn = valuecolumn
        self.uuid = uuid
        self.name = name
        self.dbconfig = dbconfig
        random.seed(0)
        self.input = {}
        self.output = {"scheduleValue": tps.Scalar(is_leaf=True)}
        self._config = {
            "parameters": [
                "weekDayRulesetDict",
                "weekendRulesetDict",
                "mondayRulesetDict",
                "tuesdayRulesetDict",
                "wednesdayRulesetDict",
                "thursdayRulesetDict",
                "fridayRulesetDict",
                "saturdayRulesetDict",
                "sundayRulesetDict",
                "add_noise",
                "useSpreadsheet",
                "useDatabase",
            ],
            "spreadsheet": ["filename", "datecolumn", "valuecolumn"],
            "database": ["uuid", "name", "dbconfig"],
        }

    @property
    def config(self):
        return self._config

    def validate(self, p):
        validated_for_simulator = True
        validated_for_estimator = True
        validated_for_optimizer = True

        if self.useSpreadsheet and self.filename is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename must be provided if useSpreadsheet is True to enable use of Simulator, Estimator, and Optimizer."
            p(message, plain=True, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif self.useDatabase and (self.uuid is None and self.name is None):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: uuid or name must be provided if useDatabase is True to enable use of Simulator, Estimator, and Optimizer."
            p(message, plain=True, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif (
            self.useSpreadsheet == False
            and self.useDatabase == False
            and self.weekDayRulesetDict is None
        ):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: weekDayRulesetDict must be provided if useSpreadsheet and useDatabase are False to enable use of Simulator, Estimator, and Optimizer."
            p(message, plain=True, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        return (
            validated_for_simulator,
            validated_for_estimator,
            validated_for_optimizer,
        )

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        self.noise = 0
        self.bias = 0
        assert (
            self.useSpreadsheet and self.filename is None
        ) == False, "filename must be provided if useSpreadsheet is True."
        assert (
            self.useDatabase and (self.uuid is None and self.name is None)
        ) == False, "uuid or name must be provided if useDatabase is True."
        assert (
            self.useSpreadsheet == False
            and self.useDatabase == False
            and self.weekDayRulesetDict is None
        ) == False, "weekDayRulesetDict must be provided if useSpreadsheet and useDatabase are False."

        if self.mondayRulesetDict is None:
            self.mondayRulesetDict = self.weekDayRulesetDict
        if self.tuesdayRulesetDict is None:
            self.tuesdayRulesetDict = self.weekDayRulesetDict
        if self.wednesdayRulesetDict is None:
            self.wednesdayRulesetDict = self.weekDayRulesetDict
        if self.thursdayRulesetDict is None:
            self.thursdayRulesetDict = self.weekDayRulesetDict
        if self.fridayRulesetDict is None:
            self.fridayRulesetDict = self.weekDayRulesetDict
        if self.saturdayRulesetDict is None:
            if self.weekendRulesetDict is None:
                self.saturdayRulesetDict = self.weekDayRulesetDict
            else:
                self.saturdayRulesetDict = self.weekendRulesetDict
        if self.sundayRulesetDict is None:
            if self.weekendRulesetDict is None:
                self.sundayRulesetDict = self.weekDayRulesetDict
            else:
                self.sundayRulesetDict = self.weekendRulesetDict
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.weekDayRulesetDict is not None, (
            "weekDayRulesetDict must be provided as argument."
        )
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.mondayRulesetDict is not None, (
            "mondayRulesetDict must be provided as argument."
        )
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.tuesdayRulesetDict is not None, (
            "tuesdayRulesetDict must be provided as argument."
        )
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.wednesdayRulesetDict is not None, (
            "wednesdayRulesetDict must be provided as argument."
        )
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.thursdayRulesetDict is not None, (
            "thursdayRulesetDict must be provided as argument."
        )
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.fridayRulesetDict is not None, (
            "fridayRulesetDict must be provided as argument."
        )
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.saturdayRulesetDict is not None, (
            "saturdayRulesetDict must be provided as argument."
        )
        assert (
            self.useSpreadsheet or self.useDatabase
        ) or self.sundayRulesetDict is not None, (
            "sundayRulesetDict must be provided as argument."
        )

        if self.useSpreadsheet or self.useDatabase:
            time_series_input = TimeSeriesInputSystem(
                id=f"time series input - {self.id}",
                filename=self.filename,
                datecolumn=self.datecolumn,
                valuecolumn=self.valuecolumn,
                useSpreadsheet=self.useSpreadsheet,
                useDatabase=self.useDatabase,
                uuid=self.uuid,
                name=self.name,
                dbconfig=self.dbconfig,
            )
            time_series_input.initialize(startTime, endTime, stepSize, simulator)
            self.output["scheduleValue"].initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
                values=time_series_input.df.values,
            )
        else:
            required_dicts = [
                self.mondayRulesetDict,
                self.tuesdayRulesetDict,
                self.wednesdayRulesetDict,
                self.thursdayRulesetDict,
                self.fridayRulesetDict,
                self.saturdayRulesetDict,
                self.sundayRulesetDict,
            ]
            required_keys = [
                "ruleset_start_minute",
                "ruleset_end_minute",
                "ruleset_start_hour",
                "ruleset_end_hour",
                "ruleset_value",
            ]
            for rulesetDict in required_dicts:
                has_key = False
                len_key = None
                for key in required_keys:
                    if key in rulesetDict:
                        if len_key is not None:
                            assert (
                                len(rulesetDict[key]) == len_key
                            ), "All keys in rulesetDict must have the same length."
                        len_key = len(rulesetDict[key])
                        has_key = True
                if has_key == False:
                    for key in required_keys:
                        rulesetDict[key] = []
                else:
                    for key in required_keys:
                        if key not in rulesetDict:
                            rulesetDict[key] = [0] * len_key

            self.output["scheduleValue"].initialize(
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                simulator=simulator,
                values=[
                    self.get_schedule_value(dateTime)
                    for dateTime in simulator.dateTimeSteps
                ],
            )

    def get_schedule_value(self, dateTime):
        if (
            dateTime.minute == 0
        ):  # Compute a new noise value if a new hour is entered in the simulation
            self.noise = randrange(-4, 4)

        if (
            dateTime.hour == 0 and dateTime.minute == 0
        ):  # Compute a new bias value if a new day is entered in the simulation
            self.bias = randrange(-10, 10)

        if dateTime.weekday() == 0:
            rulesetDict = self.mondayRulesetDict
        elif dateTime.weekday() == 1:
            rulesetDict = self.tuesdayRulesetDict
        elif dateTime.weekday() == 2:
            rulesetDict = self.wednesdayRulesetDict
        elif dateTime.weekday() == 3:
            rulesetDict = self.thursdayRulesetDict
        elif dateTime.weekday() == 4:
            rulesetDict = self.fridayRulesetDict
        elif dateTime.weekday() == 5:
            rulesetDict = self.saturdayRulesetDict
        elif dateTime.weekday() == 6:
            rulesetDict = self.sundayRulesetDict

        n = len(rulesetDict["ruleset_start_hour"])
        found_match = False
        for i_rule in range(n):
            if (
                rulesetDict["ruleset_start_hour"][i_rule] == dateTime.hour
                and dateTime.minute >= rulesetDict["ruleset_start_minute"][i_rule]
            ):
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif (
                rulesetDict["ruleset_start_hour"][i_rule] < dateTime.hour
                and dateTime.hour < rulesetDict["ruleset_end_hour"][i_rule]
            ):
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif (
                rulesetDict["ruleset_end_hour"][i_rule] == dateTime.hour
                and dateTime.minute <= rulesetDict["ruleset_end_minute"][i_rule]
            ):
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break

        if found_match == False:
            schedule_value = rulesetDict["ruleset_default_value"]
        elif self.add_noise and schedule_value > 0:
            schedule_value += self.noise + self.bias
            if schedule_value < 0:
                schedule_value = 0
        return schedule_value

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        """
        simulates a schedule and calculates the schedule value based on rulesets defined for different weekdays and times.
        It also adds noise and bias to the calculated value.
        """
        self.output["scheduleValue"].set(stepIndex=stepIndex)
