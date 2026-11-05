# Standard library imports
import copy
import random

# Third party imports
import numpy as np

# Local application imports
import twin4build.core as core
from twin4build.systems.saref4syst.declarations import Output, Setting
from twin4build.utils.data_loaders.sources import as_source


class ScheduleSystem(core.System):
    r"""A schedule: a data source (ADR 0001, decision 6) whose value at each
    step comes from rulesets for the days of the week, or from a reader.

    A ruleset is a dict::

        {
            "ruleset_default_value": 20.0,
            "ruleset_start_hour": [7],   "ruleset_start_minute": [0],
            "ruleset_end_hour": [17],    "ruleset_end_minute": [0],
            "ruleset_value": [22.0],
        }

    The first rule that covers a step's local time gives its value; no rule,
    the default.  ``weekday_ruleset`` stands for Monday to Friday and
    ``weekend_ruleset`` for Saturday and Sunday, unless a day has its own;
    without a weekend ruleset the weekend follows ``weekday_ruleset``.

    Args:
        weekday_ruleset, weekend_ruleset: The rulesets of the weekdays and the
            weekend.
        monday_ruleset ... sunday_ruleset: A day's own ruleset.
        add_noise: Add random noise to a ruleset schedule: a value drawn from
            ``[-noise_hour_range, noise_hour_range]`` each hour plus one from
            ``[-noise_day_range, noise_day_range]`` each day.  The draws are
            the same in every run.
        noise_hour_range, noise_day_range: The half-widths of the noise.
        source: A reader (:class:`~twin4build.utils.data_loaders.sources.CsvSource`,
            :class:`~twin4build.utils.data_loaders.sources.DatabaseSource`, ...)
            to read the schedule from instead of rulesets.
    """

    #: Deprecated keyword names and the ones that replaced them.
    DEPRECATED_KEYWORDS = {
        "weekDayRulesetDict": "weekday_ruleset",
        "weekendRulesetDict": "weekend_ruleset",
        "mondayRulesetDict": "monday_ruleset",
        "tuesdayRulesetDict": "tuesday_ruleset",
        "wednesdayRulesetDict": "wednesday_ruleset",
        "thursdayRulesetDict": "thursday_ruleset",
        "fridayRulesetDict": "friday_ruleset",
        "saturdayRulesetDict": "saturday_ruleset",
        "sundayRulesetDict": "sunday_ruleset",
    }

    #: The data-source keywords of 1.x and what replaced them.
    REMOVED_ATTRIBUTES = {
        **{name: "give the schedule a reader, source=CsvSource(path, date_column, value_column)"
           for name in ("filename", "date_column", "value_column", "datecolumn", "valuecolumn", "use_spreadsheet")},
        **{name: "give the schedule a reader, source=DatabaseSource(uuid)"
           for name in ("uuid", "name", "dbconfig", "use_database")},
        "use_dict": "a schedule with rulesets and no source reads the rulesets",
        "fill": "the fill is an option of the reader (CsvSource(..., fill=...))",
    }

    DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    RULE_KEYS = ("ruleset_start_minute", "ruleset_end_minute", "ruleset_start_hour", "ruleset_end_hour", "ruleset_value")

    weekday_ruleset = Setting(None, doc="The ruleset of Monday to Friday")
    weekend_ruleset = Setting(None, doc="The ruleset of Saturday and Sunday")
    monday_ruleset = Setting(None)
    tuesday_ruleset = Setting(None)
    wednesday_ruleset = Setting(None)
    thursday_ruleset = Setting(None)
    friday_ruleset = Setting(None)
    saturday_ruleset = Setting(None)
    sunday_ruleset = Setting(None)
    add_noise = Setting(False, doc="Add the hourly and daily noise to a ruleset schedule")
    noise_hour_range = Setting(4.0, doc="Half-width of the noise drawn each hour")
    noise_day_range = Setting(10.0, doc="Half-width of the noise drawn each day")
    source = Setting(None, convert=as_source, doc="A reader to read the schedule from instead of rulesets")
    scheduleValue = Output(doc="The schedule's value")

    @property
    def has_data(self) -> bool:
        """Whether the schedule has something to give: a reader or a ruleset."""
        return self.source is not None or any(
            getattr(self, f"{day}_ruleset") is not None for day in ("weekday", "weekend", *self.DAYS)
        )

    def set_dbconfig(self, dbconfig) -> None:
        """The database connection of a database reader."""
        if hasattr(self.source, "dbconfig"):
            self.source.dbconfig = dbconfig

    def day_rulesets(self) -> list:
        """The ruleset of each day, Monday first, each with every rule key (a
        copy: the settings are never changed)."""
        days = []
        for i, day in enumerate(self.DAYS):
            ruleset = getattr(self, f"{day}_ruleset")
            if ruleset is None and i >= 5:
                ruleset = self.weekend_ruleset
            if ruleset is None:
                ruleset = self.weekday_ruleset
            if ruleset is None:
                raise ValueError(
                    f"ScheduleSystem '{self.id}': no ruleset for {day} (give weekday_ruleset, weekend_ruleset "
                    f"or {day}_ruleset), and no source"
                )
            ruleset = copy.deepcopy(ruleset)
            lengths = {len(ruleset[k]) for k in self.RULE_KEYS if k in ruleset}
            if len(lengths) > 1:
                raise ValueError(f"ScheduleSystem '{self.id}': the rules of {day} differ in length")
            n = lengths.pop() if lengths else 0
            for key in self.RULE_KEYS:
                ruleset.setdefault(key, [0] * n)
            days.append(ruleset)
        return days

    @staticmethod
    def value_at(date_time, days) -> float:
        """The value the rulesets give at a local time."""
        ruleset = days[date_time.weekday()]
        for i in range(len(ruleset["ruleset_start_hour"])):
            start_hour, end_hour = ruleset["ruleset_start_hour"][i], ruleset["ruleset_end_hour"][i]
            if start_hour == date_time.hour and date_time.minute >= ruleset["ruleset_start_minute"][i]:
                return ruleset["ruleset_value"][i]
            if start_hour < date_time.hour < end_hour:
                return ruleset["ruleset_value"][i]
            if end_hour == date_time.hour and date_time.minute <= ruleset["ruleset_end_minute"][i]:
                return ruleset["ruleset_value"][i]
        return ruleset["ruleset_default_value"]

    def _noise(self, timestamps) -> np.ndarray:
        """A new hourly value at each full hour, a new daily value at
        midnight."""
        hour, day = 0.0, 0.0
        noise = []
        for t in timestamps:
            if t.minute == 0:
                hour = random.uniform(-self.noise_hour_range, self.noise_hour_range)
            if t.hour == 0 and t.minute == 0:
                day = random.uniform(-self.noise_day_range, self.noise_day_range)
            noise.append(hour + day)
        return np.array(noise)

    def series(self, timestamps, step_size):
        """The schedule's value at every step of the run."""
        if self.source is not None:
            start_time, end_time, step_sizes = self.periods(timestamps, step_size)
            return {"scheduleValue": self.source.read(start_time, end_time, step_sizes, owner=self.id).values}
        days = self.day_rulesets()
        values = np.zeros(timestamps.shape, dtype=float)
        noise_cache = self.__dict__.setdefault("noise_cache", {})
        random.seed(0)
        for b, row in enumerate(timestamps):
            own = [t for t in row if t is not None]
            values[b, : len(own)] = [self.value_at(t, days) for t in own]
            if self.add_noise and own:
                key = (own[0], len(own), step_size)
                if key not in noise_cache:
                    noise_cache[key] = self._noise(own)
                values[b, : len(own)] += noise_cache[key]
        return {"scheduleValue": values}
