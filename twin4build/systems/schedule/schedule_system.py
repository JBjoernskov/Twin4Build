# Standard library imports
import datetime
import random
from typing import Optional

# Third party imports
import numpy as np
import torch

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.translator.translator import StepRule, Node, SignaturePattern, PathRule
from twin4build.utils.deprecation import deprecate_args, deprecate_name


class ScheduleSystem(core.System):
    r"""
    A system that either 1) generates a schedule value based on rulesets defined for different weekdays and times or 2) reads a schedule value from a spreadsheet or database.

    This system provides a flexible way to create and apply different schedules for various days of the week.
    It supports both spreadsheet-based and database-based input methods.

    Args:
        weekday_ruleset: A dictionary of rulesets for weekdays.
        weekend_ruleset: A dictionary of rulesets for weekends.
        monday_ruleset ... sunday_ruleset: Per-day ruleset dictionaries.
        add_noise: A boolean to add random noise to the ruleset-based schedule value.
        noise_hour_range: Half-width of the uniform noise component redrawn each
            hour (sampled from [-noise_hour_range, +noise_hour_range]) when
            ``add_noise`` is True.
        noise_day_range: Half-width of the uniform noise component redrawn each
            day (sampled from [-noise_day_range, +noise_day_range]) when
            ``add_noise`` is True.
        source: Preferred data-source selector: ``"dict"``, ``"spreadsheet"``,
            ``"database"``, or ``None`` for auto-detect.
        use_spreadsheet / use_database / use_dict: Legacy boolean source flags
            (deprecated; use ``source=``; removed in 2.1).
        filename: The filename of the spreadsheet to read the schedule value from.
        date_column: The column index of the date in the spreadsheet.
        value_column: The column index of the value in the spreadsheet.
        uuid: The uuid identifying the time series in the database.
        name: The name identifying the time series in the database.
        dbconfig: The configuration of the database to read the schedule value from.

    """

    def __init__(
        self,
        weekday_ruleset: dict = None,
        weekend_ruleset: dict = None,
        monday_ruleset: dict = None,
        tuesday_ruleset: dict = None,
        wednesday_ruleset: dict = None,
        thursday_ruleset: dict = None,
        friday_ruleset: dict = None,
        saturday_ruleset: dict = None,
        sunday_ruleset: dict = None,
        add_noise: bool = False,
        noise_hour_range: float = 4.0,
        noise_day_range: float = 10.0,
        source: Optional[str] = None,
        use_spreadsheet: bool = False,
        use_database: bool = False,
        use_dict: bool = False,
        filename: str = None,
        date_column: int = 0,
        value_column: int = 1,
        uuid: str = None,
        name: str = None,
        dbconfig: dict = None,
        **kwargs,
    ):
        for legacy_key, new_key in (
            ("useSpreadsheet", "use_spreadsheet"),
            ("useDatabase", "use_database"),
            ("usedict", "use_dict"),
        ):
            if legacy_key in kwargs:
                raise TypeError(
                    f"`{legacy_key}` has been removed. Use `{new_key}` instead."
                )

        legacy_map = deprecate_args(
            [
                "weekDayRulesetDict",
                "weekendRulesetDict",
                "mondayRulesetDict",
                "tuesdayRulesetDict",
                "wednesdayRulesetDict",
                "thursdayRulesetDict",
                "fridayRulesetDict",
                "saturdayRulesetDict",
                "sundayRulesetDict",
                "datecolumn",
                "valuecolumn",
            ],
            [
                "weekday_ruleset",
                "weekend_ruleset",
                "monday_ruleset",
                "tuesday_ruleset",
                "wednesday_ruleset",
                "thursday_ruleset",
                "friday_ruleset",
                "saturday_ruleset",
                "sunday_ruleset",
                "date_column",
                "value_column",
            ],
            [None] * 11,
            kwargs,
        )
        weekday_ruleset = legacy_map.get("weekday_ruleset", weekday_ruleset)
        weekend_ruleset = legacy_map.get("weekend_ruleset", weekend_ruleset)
        monday_ruleset = legacy_map.get("monday_ruleset", monday_ruleset)
        tuesday_ruleset = legacy_map.get("tuesday_ruleset", tuesday_ruleset)
        wednesday_ruleset = legacy_map.get("wednesday_ruleset", wednesday_ruleset)
        thursday_ruleset = legacy_map.get("thursday_ruleset", thursday_ruleset)
        friday_ruleset = legacy_map.get("friday_ruleset", friday_ruleset)
        saturday_ruleset = legacy_map.get("saturday_ruleset", saturday_ruleset)
        sunday_ruleset = legacy_map.get("sunday_ruleset", sunday_ruleset)
        date_column = legacy_map.get("date_column", date_column)
        value_column = legacy_map.get("value_column", value_column)

        if source is not None:
            source = str(source).lower()
            assert source in {
                "dict",
                "spreadsheet",
                "database",
            }, f"|CLASS: ScheduleSystem|: source must be 'dict', 'spreadsheet', or 'database', got {source!r}"
            use_dict = source == "dict"
            use_spreadsheet = source == "spreadsheet"
            use_database = source == "database"
        elif use_spreadsheet or use_database or use_dict:
            deprecate_name(
                "use_spreadsheet/use_database/use_dict",
                "source='dict'|'spreadsheet'|'database'",
            )

        # Count how many data sources are provided
        has_dict = (
            weekday_ruleset is not None
            or weekend_ruleset is not None
            or monday_ruleset is not None
            or tuesday_ruleset is not None
            or wednesday_ruleset is not None
            or thursday_ruleset is not None
            or friday_ruleset is not None
            or saturday_ruleset is not None
            or sunday_ruleset is not None
        )
        has_filename = filename is not None
        has_database = dbconfig is not None or uuid is not None or name is not None
        n_sources = sum([has_dict, has_filename, has_database])
        n_flags = sum([use_spreadsheet, use_database, use_dict])

        # If multiple sources provided, user must explicitly set a flag
        assert not (n_sources > 1 and n_flags == 0), (
            f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Multiple data sources provided (weekday_ruleset, filename, database). "
            "You must explicitly set source='dict'|'spreadsheet'|'database' "
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
        self._weekday_ruleset = weekday_ruleset
        self._weekend_ruleset = weekend_ruleset
        self._monday_ruleset = monday_ruleset
        self._tuesday_ruleset = tuesday_ruleset
        self._wednesday_ruleset = wednesday_ruleset
        self._thursday_ruleset = thursday_ruleset
        self._friday_ruleset = friday_ruleset
        self._saturday_ruleset = saturday_ruleset
        self._sunday_ruleset = sunday_ruleset
        self.add_noise = add_noise
        self.noise_cache = {}
        self.noise_hour_range = float(noise_hour_range)
        self.noise_day_range = float(noise_day_range)
        self._use_spreadsheet = use_spreadsheet
        self._use_database = use_database
        self._use_dict = use_dict
        self._filename = filename
        self.date_column = date_column
        self.value_column = value_column
        self._uuid = uuid
        self._name = name
        self._dbconfig = dbconfig

        self.input = {}
        self.output = {"scheduleValue": tps.Scalar(is_leaf=True)}
        self._config = {
            "parameters": [
                "weekday_ruleset",
                "weekend_ruleset",
                "monday_ruleset",
                "tuesday_ruleset",
                "wednesday_ruleset",
                "thursday_ruleset",
                "friday_ruleset",
                "saturday_ruleset",
                "sunday_ruleset",
                "add_noise",
                "noise_hour_range",
                "noise_day_range",
                "use_spreadsheet",
                "use_database",
                "use_dict",
            ],
            "spreadsheet": ["filename", "date_column", "value_column"],
            "database": ["uuid", "name", "dbconfig"],
        }

    @property
    def config(self):
        return self._config

    # ==================== Ruleset Dict Properties ====================

    @property
    def weekday_ruleset(self):
        return self._weekday_ruleset

    @weekday_ruleset.setter
    def weekday_ruleset(self, value):
        self._weekday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def weekend_ruleset(self):
        return self._weekend_ruleset

    @weekend_ruleset.setter
    def weekend_ruleset(self, value):
        self._weekend_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def monday_ruleset(self):
        return self._monday_ruleset

    @monday_ruleset.setter
    def monday_ruleset(self, value):
        self._monday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def tuesday_ruleset(self):
        return self._tuesday_ruleset

    @tuesday_ruleset.setter
    def tuesday_ruleset(self, value):
        self._tuesday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def wednesday_ruleset(self):
        return self._wednesday_ruleset

    @wednesday_ruleset.setter
    def wednesday_ruleset(self, value):
        self._wednesday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def thursday_ruleset(self):
        return self._thursday_ruleset

    @thursday_ruleset.setter
    def thursday_ruleset(self, value):
        self._thursday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def friday_ruleset(self):
        return self._friday_ruleset

    @friday_ruleset.setter
    def friday_ruleset(self, value):
        self._friday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def saturday_ruleset(self):
        return self._saturday_ruleset

    @saturday_ruleset.setter
    def saturday_ruleset(self, value):
        self._saturday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    @property
    def sunday_ruleset(self):
        return self._sunday_ruleset

    @sunday_ruleset.setter
    def sunday_ruleset(self, value):
        self._sunday_ruleset = value
        if value is not None:
            self._use_dict = True
            self._use_spreadsheet = False
            self._use_database = False

    # Deprecated camelCase property aliases (removed in 2.1)
    @property
    def weekDayRulesetDict(self):
        deprecate_name("weekDayRulesetDict", "weekday_ruleset")
        return self.weekday_ruleset

    @weekDayRulesetDict.setter
    def weekDayRulesetDict(self, value):
        deprecate_name("weekDayRulesetDict", "weekday_ruleset")
        self.weekday_ruleset = value

    @property
    def weekendRulesetDict(self):
        deprecate_name("weekendRulesetDict", "weekend_ruleset")
        return self.weekend_ruleset

    @weekendRulesetDict.setter
    def weekendRulesetDict(self, value):
        deprecate_name("weekendRulesetDict", "weekend_ruleset")
        self.weekend_ruleset = value

    @property
    def datecolumn(self):
        deprecate_name("datecolumn", "date_column")
        return self.date_column

    @datecolumn.setter
    def datecolumn(self, value):
        deprecate_name("datecolumn", "date_column")
        self.date_column = value

    @property
    def valuecolumn(self):
        deprecate_name("valuecolumn", "value_column")
        return self.value_column

    @valuecolumn.setter
    def valuecolumn(self, value):
        deprecate_name("valuecolumn", "value_column")
        self.value_column = value

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
            if self.monday_ruleset is None and self.weekday_ruleset is None:
                missing_days.append("monday_ruleset")
            if self.tuesday_ruleset is None and self.weekday_ruleset is None:
                missing_days.append("tuesday_ruleset")
            if self.wednesday_ruleset is None and self.weekday_ruleset is None:
                missing_days.append("wednesday_ruleset")
            if self.thursday_ruleset is None and self.weekday_ruleset is None:
                missing_days.append("thursday_ruleset")
            if self.friday_ruleset is None and self.weekday_ruleset is None:
                missing_days.append("friday_ruleset")
            if (
                self.saturday_ruleset is None
                and self.weekend_ruleset is None
                and self.weekday_ruleset is None
            ):
                missing_days.append("saturday_ruleset")
            if (
                self.sunday_ruleset is None
                and self.weekend_ruleset is None
                and self.weekday_ruleset is None
            ):
                missing_days.append("sunday_ruleset")
            if missing_days:
                message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: The following ruleset dicts are missing (provide directly or via weekday_ruleset/weekend_ruleset): {', '.join(missing_days)}"
                p(message, status="WARNING")
                validated_for_simulator = False
                validated_for_estimator = False
                validated_for_optimizer = False

        if not self.use_spreadsheet and not self.use_database and not self.use_dict:
            message = f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Either weekday_ruleset with use_dict=True, use_spreadsheet=True, or use_database=True must be provided to enable use of Simulator, Estimator, and Optimizer."
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
        random.seed(0)
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

        if self._monday_ruleset is None:
            self._monday_ruleset = self._weekday_ruleset
        if self._tuesday_ruleset is None:
            self._tuesday_ruleset = self._weekday_ruleset
        if self._wednesday_ruleset is None:
            self._wednesday_ruleset = self._weekday_ruleset
        if self._thursday_ruleset is None:
            self._thursday_ruleset = self._weekday_ruleset
        if self._friday_ruleset is None:
            self._friday_ruleset = self._weekday_ruleset
        if self._saturday_ruleset is None:
            if self._weekend_ruleset is None:
                self._saturday_ruleset = self._weekday_ruleset
            else:
                self._saturday_ruleset = self._weekend_ruleset
        if self._sunday_ruleset is None:
            if self._weekend_ruleset is None:
                self._sunday_ruleset = self._weekday_ruleset
            else:
                self._sunday_ruleset = self._weekend_ruleset
        if self.use_dict:
            assert (
                self.monday_ruleset is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: monday_ruleset must be provided (directly or via weekday_ruleset) when use_dict is True."
            assert (
                self.tuesday_ruleset is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: tuesday_ruleset must be provided (directly or via weekday_ruleset) when use_dict is True."
            assert (
                self.wednesday_ruleset is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: wednesday_ruleset must be provided (directly or via weekday_ruleset) when use_dict is True."
            assert (
                self.thursday_ruleset is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: thursday_ruleset must be provided (directly or via weekday_ruleset) when use_dict is True."
            assert (
                self.friday_ruleset is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: friday_ruleset must be provided (directly or via weekday_ruleset) when use_dict is True."
            assert (
                self.saturday_ruleset is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: saturday_ruleset must be provided (directly or via weekday_ruleset/weekend_ruleset) when use_dict is True."
            assert (
                self.sunday_ruleset is not None
            ), f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: sunday_ruleset must be provided (directly or via weekday_ruleset/weekend_ruleset) when use_dict is True."

        if self.use_spreadsheet or self.use_database:
            time_series_input = TimeSeriesInputSystem(
                id=f"time series input - {self.id}",
                filename=self.filename,
                date_column=self.date_column,
                value_column=self.value_column,
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
                n_t=time_series_input.n_timesteps,
                n_s=time_series_input.batch_size,
                n_c=1,
                values=time_series_input.values,
            )
        else:
            required_dicts = [
                self.monday_ruleset,
                self.tuesday_ruleset,
                self.wednesday_ruleset,
                self.thursday_ruleset,
                self.friday_ruleset,
                self.saturday_ruleset,
                self.sunday_ruleset,
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
                # ``n_timesteps_ == 0`` means the requested period has zero
                # length (start_time >= end_time, or step_size larger than
                # the period).  The pad-with-last-value step below would
                # then index ``values[batch, -1]`` on an empty axis -- skip
                # the pad and let the upstream simulator raise the real
                # validation error instead of an opaque ``IndexError``.
                if n_timesteps_ > 0:
                    values[batch_index, n_timesteps_:] = values[
                        batch_index, n_timesteps_ - 1
                    ]
                # NEW: Compute schedule values for ALL timesteps (including extended dates for shorter periods)
                # values[batch_index,:] = [self.get_schedule_value(date_time) for date_time in date_time_steps_]

                if self.add_noise:
                    # cache noise
                    index = (
                        start_time[batch_index],
                        end_time[batch_index],
                        step_size[batch_index],
                    )
                    if index not in self.noise_cache:
                        self.noise_cache[index] = self.get_noise(
                            date_time_steps_[:n_timesteps_]
                        )
                    values[batch_index, :n_timesteps_] += self.noise_cache[index]

            assert not np.isnan(
                values
            ).any(), (
                f"|CLASS: {self.__class__.__name__}|ID: {self.id}|: Values contain NaN."
            )

            # Convert values from (n_s, n_t) to time-first (n_t, n_s, n_c) where n_c=1
            # First transpose to (n_t, n_s), then unsqueeze to (n_t, n_s, 1)
            values = torch.tensor(values, dtype=tps.float_dtype()).T.unsqueeze(
                -1
            )  # (n_t, n_s, 1)
            self.output["scheduleValue"].initialize(
                n_t=max_timesteps,
                n_s=len(start_time),
                n_c=1,
                values=values,
            )

    def get_noise(self, date_time_steps):
        noise = []
        for date_time in date_time_steps:
            if (
                date_time.minute == 0
            ):  # Compute a new noise value if a new hour is entered in the simulation
                noise_hour = random.uniform(
                    -self.noise_hour_range, self.noise_hour_range
                )
            if (
                date_time.hour == 0 and date_time.minute == 0
            ):  # Compute a new bias value if a new day is entered in the simulation
                noise_day = random.uniform(-self.noise_day_range, self.noise_day_range)
            noise.append(noise_hour + noise_day)
        return np.array(noise)

    def get_schedule_value(self, date_time):

        # if self.add_noise:
        #     if (
        #         date_time.minute == 0
        #     ):  # Compute a new noise value if a new hour is entered in the simulation

        #         self.noise = random.uniform(
        #             -self.noise_hour_range, self.noise_hour_range
        #         )

        #     if (
        #         date_time.hour == 0 and date_time.minute == 0
        #     ):  # Compute a new bias value if a new day is entered in the simulation

        #         self.bias = random.uniform(
        #             -self.noise_day_range, self.noise_day_range
        #         )

        if date_time.weekday() == 0:
            rulesetDict = self.monday_ruleset
        elif date_time.weekday() == 1:
            rulesetDict = self.tuesday_ruleset
        elif date_time.weekday() == 2:
            rulesetDict = self.wednesday_ruleset
        elif date_time.weekday() == 3:
            rulesetDict = self.thursday_ruleset
        elif date_time.weekday() == 4:
            rulesetDict = self.friday_ruleset
        elif date_time.weekday() == 5:
            rulesetDict = self.saturday_ruleset
        elif date_time.weekday() == 6:
            rulesetDict = self.sunday_ruleset

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
        # elif self.add_noise and schedule_value > 0:
        #     schedule_value += self.noise + self.bias
        #     if schedule_value < 0:
        #         schedule_value = 0
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
        self.output["scheduleValue"]._set(i_t=step_index)


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
