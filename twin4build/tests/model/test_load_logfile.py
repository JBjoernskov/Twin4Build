"""``load()`` leaves a configured ``LOGGER.logfile`` alone.

``Model.load`` and ``SimulationModel.load`` assigned their ``logfile``
argument to ``LOGGER.logfile`` unconditionally, so the default
``logfile=None`` undid ``LOGGER.logfile = "run.log"``, the documented way
to configure the log.  That went unseen while ``None`` fell back to
``progress.log``; with ``None`` meaning "no file" (#138) the log of
everything after the first ``load()`` would move to stdout.  ``load()``
sets the logfile only when it is given one.
"""

# Standard library imports
import os
import shutil
import tempfile
import unittest

# Local application imports
import twin4build
from twin4build.model.model import Model
from twin4build.systems.damper.damper_system import DamperSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.utils.logger import LOGGER

twin4build._IS_TESTING = True


class TestLoadLogfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.logfile = LOGGER.logfile
        schedule = ScheduleSystem(
            weekday_ruleset={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [1],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule",
        )
        damper = DamperSystem(id="damper")
        self.model = Model(id="test_load_logfile")
        self.model.add_component(schedule)
        self.model.add_component(damper)
        self.model.add_connection(schedule, damper, "scheduleValue", "damperPosition")

    def tearDown(self):
        LOGGER.logfile = self.logfile
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self, model, **kwargs):
        if isinstance(model, Model):
            kwargs.update(draw_semantic_model=False, draw_simulation_model=False)
        model.load(**kwargs)

    def test_the_default_keeps_the_configured_logfile(self):
        configured = os.path.join(self.tmp, "configured.log")
        for model in (self.model, self.model.simulation_model):
            LOGGER.logfile = configured
            self._load(model)
            self.assertEqual(LOGGER.logfile, configured)

    def test_a_given_logfile_is_set(self):
        given = os.path.join(self.tmp, "given.log")
        for model in (self.model, self.model.simulation_model):
            LOGGER.logfile = None
            self._load(model, logfile=given)
            self.assertEqual(LOGGER.logfile, given)


if __name__ == "__main__":
    unittest.main()
