# Standard library imports
import datetime
import random
from random import randrange
from typing import Optional

# Third party imports
import numpy as np

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import Exact, Node, SignaturePattern, SinglePath


class ScheduleSystem(core.System):
    r"""
    A system that either 1) generates a schedule value based on rulesets defined for different weekdays and times or 2) reads a schedule value from a spreadsheet or database.

    This system provides a flexible way to create and apply different schedules for various days of the week.
    It supports both spreadsheet-based and database-based input methods.

    Args:
        weekDayRulesetDict: A dictionary of rulesets for weekdays.
        weekendRulesetDict: A dictionary of rulesets for weekends.
        mondayRulesetDict: A dictionary of rulesets for Mondays.
        tuesdayRulesetDict: A dictionary of rulesets for Tuesdays.
        wednesdayRulesetDict: A dictionary of rulesets for Wednesdays.
        thursdayRulesetDict: A dictionary of rulesets for Thursdays.
        fridayRulesetDict: A dictionary of rulesets for Fridays.
        saturdayRulesetDict: A dictionary of rulesets for Saturdays.
        sundayRulesetDict: A dictionary of rulesets for Sundays.
        add_noise: A boolean to add noise to the schedule value.
        useSpreadsheet: A boolean to use a spreadsheet to read the schedule value.
        useDatabase: A boolean to use a database to read the schedule value.
        filename: The filename of the spreadsheet to read the schedule value.
        datecolumn: The column index of the date in the spreadsheet.
        valuecolumn: The column index of the value in the spreadsheet.
        uuid: The uuid of the database to read the schedule value.
        name: The name of the database to read the schedule value.
        dbconfig: The configuration of the database to read the schedule value.

    """

    def __init__(
        self,
        weekDayRulesetDict: dict = None,
        weekendRulesetDict: dict = None,
        mondayRulesetDict: dict = None,
        tuesdayRulesetDict: dict = None,
        wednesdayRulesetDict: dict = None,
        thursdayRulesetDict: dict = None,
        fridayRulesetDict: dict = None,
        saturdayRulesetDict: dict = None,
        sundayRulesetDict: dict = None,
        add_noise: bool = False,
        useSpreadsheet: bool = False,
        useDatabase: bool = False,
        usedict: bool = False,
        filename: str = None,
        datecolumn: int = 0,
        valuecolumn: int = 1,
        uuid: str = None,
        name: str = None,
        dbconfig: dict = None,
        **kwargs,
    ):
        # Count how many data sources are provided
        has_dict = (
            weekDayRulesetDict is not None
            or weekendRulesetDict is not None
            or mondayRulesetDict is not None
            or tuesdayRulesetDict is not None
            or wednesdayRulesetDict is not None
            or thursdayRulesetDict is not None
            or fridayRulesetDict is not None
            or saturdayRulesetDict is not None
            or sundayRulesetDict is not None
        )
        has_filename = filename is not None
        has_database = dbconfig is not None or uuid is not None or name is not None
        n_sources = sum([has_dict, has_filename, has_database])
        n_flags = sum([useSpreadsheet, useDatabase, usedict])

        # If multiple sources provided, user must explicitly set a flag
        assert not (n_sources > 1 and n_flags == 0), (
            "Multiple data sources provided (weekDayRulesetDict, filename, database). "
            "You must explicitly set one of usedict=True, useSpreadsheet=True, or useDatabase=True "
            "to specify which source to use."
        )

        # Auto-detect data source if no flags are explicitly set
        if not useSpreadsheet and not useDatabase and not usedict:
            if has_dict:
                usedict = True
            elif has_filename:
                useSpreadsheet = True
            elif has_database:
                useDatabase = True

        super().__init__(**kwargs)
        assert (
            sum([useSpreadsheet, useDatabase, usedict]) <= 1
        ), "Only one of useSpreadsheet, useDatabase, or usedict can be True."
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
        self.usedict = usedict
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
                "usedict",
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
            p(message, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif self.useDatabase and (self.uuid is None and self.name is None):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: uuid or name must be provided if useDatabase is True to enable use of Simulator, Estimator, and Optimizer."
            p(message, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif self.usedict:
            # Check that all days can be covered (either directly or via fallback dicts)
            missing_days = []
            if self.mondayRulesetDict is None and self.weekDayRulesetDict is None:
                missing_days.append("mondayRulesetDict")
            if self.tuesdayRulesetDict is None and self.weekDayRulesetDict is None:
                missing_days.append("tuesdayRulesetDict")
            if self.wednesdayRulesetDict is None and self.weekDayRulesetDict is None:
                missing_days.append("wednesdayRulesetDict")
            if self.thursdayRulesetDict is None and self.weekDayRulesetDict is None:
                missing_days.append("thursdayRulesetDict")
            if self.fridayRulesetDict is None and self.weekDayRulesetDict is None:
                missing_days.append("fridayRulesetDict")
            if self.saturdayRulesetDict is None and self.weekendRulesetDict is None and self.weekDayRulesetDict is None:
                missing_days.append("saturdayRulesetDict")
            if self.sundayRulesetDict is None and self.weekendRulesetDict is None and self.weekDayRulesetDict is None:
                missing_days.append("sundayRulesetDict")
            if missing_days:
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: The following ruleset dicts are missing (provide directly or via weekDayRulesetDict/weekendRulesetDict): {', '.join(missing_days)}"
                p(message, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False

        if not self.useSpreadsheet and not self.useDatabase and not self.usedict:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Either weekDayRulesetDict with usedict=True, useSpreadsheet=True, or useDatabase=True must be provided to enable use of Simulator, Estimator, and Optimizer."
            p(message, status="WARNING")
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
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        step_size: int,
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
            self.useSpreadsheet or self.useDatabase or self.usedict
        ), "One of useSpreadsheet, useDatabase, or usedict must be True."

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
        if self.usedict:
            assert self.mondayRulesetDict is not None, (
                "mondayRulesetDict must be provided (directly or via weekDayRulesetDict) when usedict is True."
            )
            assert self.tuesdayRulesetDict is not None, (
                "tuesdayRulesetDict must be provided (directly or via weekDayRulesetDict) when usedict is True."
            )
            assert self.wednesdayRulesetDict is not None, (
                "wednesdayRulesetDict must be provided (directly or via weekDayRulesetDict) when usedict is True."
            )
            assert self.thursdayRulesetDict is not None, (
                "thursdayRulesetDict must be provided (directly or via weekDayRulesetDict) when usedict is True."
            )
            assert self.fridayRulesetDict is not None, (
                "fridayRulesetDict must be provided (directly or via weekDayRulesetDict) when usedict is True."
            )
            assert self.saturdayRulesetDict is not None, (
                "saturdayRulesetDict must be provided (directly or via weekDayRulesetDict/weekendRulesetDict) when usedict is True."
            )
            assert self.sundayRulesetDict is not None, (
                "sundayRulesetDict must be provided (directly or via weekDayRulesetDict/weekendRulesetDict) when usedict is True."
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
            time_series_input.initialize(start_time, end_time, step_size)

            # The batch initialization args are calculated in the time_series_input.initialize() method.
            # They are stored in the time_series_input object and reused here.
            self.output["scheduleValue"].initialize(
                time_series_input.n_timesteps,
                batch_size=time_series_input.batch_size,
                values=time_series_input.values,
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

            second_time_steps, date_time_steps, max_timesteps, n_timesteps = (
                core.Simulator.get_simulation_timesteps(start_time, end_time, step_size)
            )
            values = np.empty((len(start_time), max_timesteps))
            values.fill(
                0
            )  # Before we used nan, but this caused issues with the optimizer when the optimizer tried to compute the gradient of the loss function.
            for batch_index, (date_time_steps_, n_timesteps_) in enumerate(
                zip(date_time_steps, n_timesteps)
            ):
                # OLD: Only compute schedule values for actual timesteps
                values[batch_index, :n_timesteps_] = [
                    self.get_schedule_value(date_time)
                    for date_time in date_time_steps_[:n_timesteps_]
                ]
                values[batch_index, n_timesteps_:] = values[
                    batch_index, n_timesteps_ - 1
                ]
                # NEW: Compute schedule values for ALL timesteps (including extended dates for shorter periods)
                # values[batch_index,:] = [self.get_schedule_value(date_time) for date_time in date_time_steps_]

            assert not np.isnan(values).any(), "Values contain NaN."

            self.output["scheduleValue"].initialize(
                max_timesteps,
                batch_size=len(start_time),
                values=values,
            )

    def get_schedule_value(self, date_time):
        if (
            date_time.minute == 0
        ):  # Compute a new noise value if a new hour is entered in the simulation
            self.noise = randrange(-4, 4)

        if (
            date_time.hour == 0 and date_time.minute == 0
        ):  # Compute a new bias value if a new day is entered in the simulation
            self.bias = randrange(-10, 10)

        if date_time.weekday() == 0:
            rulesetDict = self.mondayRulesetDict
        elif date_time.weekday() == 1:
            rulesetDict = self.tuesdayRulesetDict
        elif date_time.weekday() == 2:
            rulesetDict = self.wednesdayRulesetDict
        elif date_time.weekday() == 3:
            rulesetDict = self.thursdayRulesetDict
        elif date_time.weekday() == 4:
            rulesetDict = self.fridayRulesetDict
        elif date_time.weekday() == 5:
            rulesetDict = self.saturdayRulesetDict
        elif date_time.weekday() == 6:
            rulesetDict = self.sundayRulesetDict

        n = len(rulesetDict["ruleset_start_hour"])
        found_match = False
        for i_rule in range(n):
            if (
                rulesetDict["ruleset_start_hour"][i_rule] == date_time.hour
                and date_time.minute >= rulesetDict["ruleset_start_minute"][i_rule]
            ):
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif (
                rulesetDict["ruleset_start_hour"][i_rule] < date_time.hour
                and date_time.hour < rulesetDict["ruleset_end_hour"][i_rule]
            ):
                schedule_value = rulesetDict["ruleset_value"][i_rule]
                found_match = True
                break
            elif (
                rulesetDict["ruleset_end_hour"][i_rule] == date_time.hour
                and date_time.minute <= rulesetDict["ruleset_end_minute"][i_rule]
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
        second_time: float,
        date_time: datetime.datetime,
        step_size: int,
        step_index: int,
    ) -> None:
        """
        simulates a schedule and calculates the schedule value based on rulesets defined for different weekdays and times.
        It also adds noise and bias to the calculated value.
        """
        self.output["scheduleValue"].set(step_index=step_index)


def saref_signature_pattern():
    """
    Get the SAREF signature pattern of the schedule component.

    Returns:
        SignaturePattern: The SAREF signature pattern of the schedule component.
    """
    node0 = Node(cls=(core.namespace.S4BLDG.Schedule))
    sp = SignaturePattern(id="schedule_signature_pattern")
    sp.add_modeled_node(node0)
    return sp


def brick_signature_pattern():
    """
    Get the BRICK signature pattern of the schedule component.

    Returns:
        SignaturePattern: The BRICK signature pattern of the schedule component.
    """
    node0 = Node(cls=core.namespace.BRICK.Schedule)
    sp = SignaturePattern(id="schedule_signature_pattern_brick")
    sp.add_modeled_node(node0)
    return sp


ScheduleSystem.add_signature_pattern(brick_signature_pattern())
ScheduleSystem.add_signature_pattern(saref_signature_pattern())
