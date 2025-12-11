# Standard library imports
import datetime
import random
import warnings
from typing import Optional

# Third party imports
import numpy as np

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import Exact, Node, SignaturePattern, SinglePath
from twin4build.utils.deprecation import deprecate_args


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
        noise_hour_range: float = 4.0,
        noise_day_range: float = 10.0,
        use_spreadsheet: bool = False,
        use_database: bool = False,
        use_dict: bool = False,
        filename: str = None,
        datecolumn: int = 0,
        valuecolumn: int = 1,
        uuid: str = None,
        name: str = None,
        dbconfig: dict = None,
        **kwargs,
    ):
        # Handle deprecated camelCase arguments
        deprecated_args = ["useSpreadsheet", "useDatabase", "usedict"]
        new_args = ["use_spreadsheet", "use_database", "use_dict"]
        positions = [None, None, None]
        value_map = deprecate_args(deprecated_args, new_args, positions, kwargs)
        use_spreadsheet = value_map.get("use_spreadsheet", use_spreadsheet)
        use_database = value_map.get("use_database", use_database)
        use_dict = value_map.get("use_dict", use_dict)

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
        n_flags = sum([use_spreadsheet, use_database, use_dict])

        # If multiple sources provided, user must explicitly set a flag
        assert not (n_sources > 1 and n_flags == 0), (
            f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Multiple data sources provided (weekDayRulesetDict, filename, database). "
            "You must explicitly set one of use_dict=True, use_spreadsheet=True, or use_database=True "
            "to specify which source to use."
        )

        # Auto-detect data source if no flags are explicitly set
        if not use_spreadsheet and not use_database and not use_dict:
            if has_dict:
                use_dict = True
            elif has_filename:
                use_spreadsheet = True
            elif has_database:
                use_database = True

        super().__init__(**kwargs)
        assert (
            sum([use_spreadsheet, use_database, use_dict]) <= 1
        ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Only one of use_spreadsheet, use_database, or use_dict can be True."

        # Store as private variables for property access
        self._weekDayRulesetDict = weekDayRulesetDict
        self._weekendRulesetDict = weekendRulesetDict
        self._mondayRulesetDict = mondayRulesetDict
        self._tuesdayRulesetDict = tuesdayRulesetDict
        self._wednesdayRulesetDict = wednesdayRulesetDict
        self._thursdayRulesetDict = thursdayRulesetDict
        self._fridayRulesetDict = fridayRulesetDict
        self._saturdayRulesetDict = saturdayRulesetDict
        self._sundayRulesetDict = sundayRulesetDict
        self.add_noise = add_noise
        self.noise_hour_range = float(noise_hour_range)
        self.noise_day_range = float(noise_day_range)
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._use_dict = use_dict
        self._filename = filename
        self.datecolumn = datecolumn
        self.valuecolumn = valuecolumn
        self._uuid = uuid
        self._name = name
        self._dbconfig = dbconfig
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
                "noise_hour_range",
                "noise_day_range",
                "use_spreadsheet",
                "use_database",
                "use_dict",
            ],
            "spreadsheet": ["filename", "datecolumn", "valuecolumn"],
            "database": ["uuid", "name", "dbconfig"],
        }

    @property
    def config(self):
        return self._config

    # ==================== Ruleset Dict Properties ====================

    @property
    def weekDayRulesetDict(self):
        return self._weekDayRulesetDict

    @weekDayRulesetDict.setter
    def weekDayRulesetDict(self, value):
        self._weekDayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def weekendRulesetDict(self):
        return self._weekendRulesetDict

    @weekendRulesetDict.setter
    def weekendRulesetDict(self, value):
        self._weekendRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def mondayRulesetDict(self):
        return self._mondayRulesetDict

    @mondayRulesetDict.setter
    def mondayRulesetDict(self, value):
        self._mondayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def tuesdayRulesetDict(self):
        return self._tuesdayRulesetDict

    @tuesdayRulesetDict.setter
    def tuesdayRulesetDict(self, value):
        self._tuesdayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def wednesdayRulesetDict(self):
        return self._wednesdayRulesetDict

    @wednesdayRulesetDict.setter
    def wednesdayRulesetDict(self, value):
        self._wednesdayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def thursdayRulesetDict(self):
        return self._thursdayRulesetDict

    @thursdayRulesetDict.setter
    def thursdayRulesetDict(self, value):
        self._thursdayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def fridayRulesetDict(self):
        return self._fridayRulesetDict

    @fridayRulesetDict.setter
    def fridayRulesetDict(self, value):
        self._fridayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def saturdayRulesetDict(self):
        return self._saturdayRulesetDict

    @saturdayRulesetDict.setter
    def saturdayRulesetDict(self, value):
        self._saturdayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def sundayRulesetDict(self):
        return self._sundayRulesetDict

    @sundayRulesetDict.setter
    def sundayRulesetDict(self, value):
        self._sundayRulesetDict = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    # ==================== Data Source Flags ====================

    @property
    def use_spreadsheet(self):
        return self._use_spreadsheet

    @use_spreadsheet.setter
    def use_spreadsheet(self, value):
        self._use_spreadsheet = value

    @property
    def use_database(self):
        return self._use_database

    @use_database.setter
    def use_database(self, value):
        self._use_database = value

    @property
    def use_dict(self):
        return self._use_dict

    @use_dict.setter
    def use_dict(self, value):
        self._use_dict = value

    # ==================== Deprecated Properties (camelCase) ====================

    @property
    def useSpreadsheet(self):
        """Deprecated: Use use_spreadsheet instead."""
        warnings.warn(
            "Property 'useSpreadsheet' is deprecated. Use 'use_spreadsheet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_spreadsheet

    @useSpreadsheet.setter
    def useSpreadsheet(self, value):
        warnings.warn(
            "Property 'useSpreadsheet' is deprecated. Use 'use_spreadsheet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_spreadsheet = value

    @property
    def useDatabase(self):
        """Deprecated: Use use_database instead."""
        warnings.warn(
            "Property 'useDatabase' is deprecated. Use 'use_database' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_database

    @useDatabase.setter
    def useDatabase(self, value):
        warnings.warn(
            "Property 'useDatabase' is deprecated. Use 'use_database' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_database = value

    @property
    def usedict(self):
        """Deprecated: Use use_dict instead."""
        warnings.warn(
            "Property 'usedict' is deprecated. Use 'use_dict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._use_dict

    @usedict.setter
    def usedict(self, value):
        warnings.warn(
            "Property 'usedict' is deprecated. Use 'use_dict' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._use_dict = value

    # ==================== Spreadsheet Properties ====================

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value
        if value is not None:
            self._use_spreadsheet = True
            self._use_database = False
            self._use_dict = False

    # ==================== Database Properties ====================

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, value):
        self._uuid = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_dict = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_dict = False

    @property
    def dbconfig(self):
        return self._dbconfig

    @dbconfig.setter
    def dbconfig(self, value):
        self._dbconfig = value
        if value is not None:
            self._use_database = True
            self._use_spreadsheet = False
            self._use_dict = False

    def validate(self, p):
        validated_for_simulator = True
        validated_for_estimator = True
        validated_for_optimizer = True

        if self.use_spreadsheet and self.filename is None:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename must be provided if use_spreadsheet is True to enable use of Simulator, Estimator, and Optimizer."
            p(message, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif self.use_database and (self.uuid is None and self.name is None):
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: uuid or name must be provided if use_database is True to enable use of Simulator, Estimator, and Optimizer."
            p(message, status="WARNING")
            validated_for_simulator = False
            validated_for_estimator = False
            validated_for_optimizer = False

        elif self.use_dict:
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
            if (
                self.saturdayRulesetDict is None
                and self.weekendRulesetDict is None
                and self.weekDayRulesetDict is None
            ):
                missing_days.append("saturdayRulesetDict")
            if (
                self.sundayRulesetDict is None
                and self.weekendRulesetDict is None
                and self.weekDayRulesetDict is None
            ):
                missing_days.append("sundayRulesetDict")
            if missing_days:
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: The following ruleset dicts are missing (provide directly or via weekDayRulesetDict/weekendRulesetDict): {', '.join(missing_days)}"
                p(message, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False

        if not self.use_spreadsheet and not self.use_database and not self.use_dict:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Either weekDayRulesetDict with use_dict=True, use_spreadsheet=True, or use_database=True must be provided to enable use of Simulator, Estimator, and Optimizer."
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
            self.use_spreadsheet and self.filename is None
        ) == False, f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: filename must be provided if use_spreadsheet is True."
        assert (
            self.use_database and (self.uuid is None and self.name is None)
        ) == False, f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: uuid or name must be provided if use_database is True."
        assert (
            self.use_spreadsheet or self.use_database or self.use_dict
        ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: One of use_spreadsheet, use_database, or use_dict must be True."

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
        if self.use_dict:
            assert (
                self.mondayRulesetDict is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: mondayRulesetDict must be provided (directly or via weekDayRulesetDict) when use_dict is True."
            assert (
                self.tuesdayRulesetDict is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: tuesdayRulesetDict must be provided (directly or via weekDayRulesetDict) when use_dict is True."
            assert (
                self.wednesdayRulesetDict is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: wednesdayRulesetDict must be provided (directly or via weekDayRulesetDict) when use_dict is True."
            assert (
                self.thursdayRulesetDict is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: thursdayRulesetDict must be provided (directly or via weekDayRulesetDict) when use_dict is True."
            assert (
                self.fridayRulesetDict is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: fridayRulesetDict must be provided (directly or via weekDayRulesetDict) when use_dict is True."
            assert (
                self.saturdayRulesetDict is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: saturdayRulesetDict must be provided (directly or via weekDayRulesetDict/weekendRulesetDict) when use_dict is True."
            assert (
                self.sundayRulesetDict is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: sundayRulesetDict must be provided (directly or via weekDayRulesetDict/weekendRulesetDict) when use_dict is True."

        if self.use_spreadsheet or self.use_database:
            time_series_input = TimeSeriesInputSystem(
                id=f"time series input - {self.id}",
                filename=self.filename,
                datecolumn=self.datecolumn,
                valuecolumn=self.valuecolumn,
                use_spreadsheet=self.use_spreadsheet,
                use_database=self.use_database,
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
                            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: All keys in rulesetDict must have the same length."
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

            assert not np.isnan(
                values
            ).any(), (
                f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Values contain NaN."
            )

            self.output["scheduleValue"].initialize(
                max_timesteps,
                batch_size=len(start_time),
                values=values,
            )

    def get_schedule_value(self, date_time):

        if self.add_noise:
            if (
                date_time.minute == 0
            ):  # Compute a new noise value if a new hour is entered in the simulation
                
                self.noise = random.uniform(
                    -self.noise_hour_range, self.noise_hour_range
                )

            if (
                date_time.hour == 0 and date_time.minute == 0
            ):  # Compute a new bias value if a new day is entered in the simulation
                
                self.bias = random.uniform(
                    -self.noise_day_range, self.noise_day_range
                )

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
