"""``twin4build.settings``: every knob in one place.

A value comes from ``set``, else the environment variable, else the default;
values are checked; ``override`` restores what was there; ``snapshot`` records
every setting with its source; and the library reads the settings where it uses
them, never when a module is imported; estimation and optimization results
carry the settings they ran with.
"""

# Standard library imports
import datetime
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from dateutil import tz

# Local application imports
import twin4build as tb
from twin4build import settings
from twin4build.simulator import _functional
from twin4build.systems.damper.damper_system import DamperSystem
from twin4build.systems.schedule.schedule_system import ScheduleSystem
from twin4build.systems.sensor.sensor_system import SensorSystem
from twin4build.systems.utils import discrete_statespace_system as dss
from twin4build.systems.utils.time_series_input_system import TimeSeriesInputSystem
from twin4build.utils.data_loaders.sources import DataFrameSource
from twin4build.utils.logger import LOGGER

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "source" / "manual" / "settings.rst"
VARIABLES = [s.env for s in settings._REGISTRY.values()]
START = datetime.datetime(2023, 1, 1, tzinfo=tz.UTC)
END = START + datetime.timedelta(hours=1)
STEP = 600


class _NoSettingsInTheEnvironment(unittest.TestCase):
    """Each test starts with no setting set, nor any in the environment."""

    def setUp(self):
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for variable in VARIABLES:
            os.environ.pop(variable, None)
        settings.reset()
        self.addCleanup(settings.reset)


class TestSettings(_NoSettingsInTheEnvironment):
    def test_set_then_environment_then_default(self):
        def entry():
            return settings.snapshot()["jac_check"]

        self.assertEqual(entry(), {"value": False, "source": "default"})
        os.environ["TWIN4BUILD_JAC_CHECK"] = "yes"
        self.assertEqual(entry(), {"value": True, "source": "TWIN4BUILD_JAC_CHECK"})
        settings.set(jac_check=False)
        self.assertEqual(entry(), {"value": False, "source": "set"})
        self.assertFalse(settings.value("jac_check"))
        settings.reset("jac_check")
        self.assertTrue(settings.value("jac_check"))
        del os.environ["TWIN4BUILD_JAC_CHECK"]
        self.assertEqual(entry(), {"value": False, "source": "default"})

    def test_override_restores_what_was_there(self):
        settings.set(zoh_expm="block")
        with settings.override(zoh_expm="phi1", jac_check=True):
            self.assertEqual(settings.value("zoh_expm"), "phi1")
            self.assertTrue(settings.value("jac_check"))
            with settings.override(jac_check=False):
                self.assertFalse(settings.value("jac_check"))
            self.assertTrue(settings.value("jac_check"))
        self.assertEqual(
            settings.snapshot()["zoh_expm"], {"value": "block", "source": "set"}
        )
        self.assertEqual(
            settings.snapshot()["jac_check"], {"value": False, "source": "default"}
        )
        with self.assertRaises(RuntimeError):
            with settings.override(jac_check=True):
                raise RuntimeError("the block fails")
        self.assertFalse(settings.value("jac_check"))
        with self.assertRaises(ValueError):  # a bad value: nothing is applied
            with settings.override(jac_check=True, zoh_expm="taylor"):
                pass
        self.assertFalse(settings.value("jac_check"))

    def test_snapshot_records_every_setting_and_its_source(self):
        os.environ["TWIN4BUILD_TESTING"] = "1"
        settings.set(zoh_expm="block")
        snapshot = settings.snapshot()
        self.assertEqual(sorted(snapshot), sorted(settings._REGISTRY))
        # Among them the settings that change a run's numbers.
        for name in ("zoh_expm", "testing"):
            self.assertIn(name, snapshot)
        self.assertEqual(snapshot["zoh_expm"], {"value": "block", "source": "set"})
        self.assertEqual(
            snapshot["testing"], {"value": True, "source": "TWIN4BUILD_TESTING"}
        )
        self.assertEqual(
            snapshot["jac_check"], {"value": False, "source": "default"}
        )
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)

    def test_unknown_names(self):
        with self.assertRaisesRegex(KeyError, "no setting 'no_such_knob'.*jac_check"):
            settings.value("no_such_knob")
        with self.assertRaises(KeyError):  # nothing is applied
            settings.set(jac_check=True, no_such_knob=1)
        self.assertFalse(settings.value("jac_check"))
        with self.assertRaises(KeyError):
            with settings.override(no_such_knob=1):
                pass
        with self.assertRaises(KeyError):
            settings.reset("no_such_knob")

    def test_set_checks_the_values(self):
        with self.assertRaises(TypeError):
            settings.set(jac_check=1)
        with self.assertRaises(TypeError):
            settings.set(jac_check=None)
        with self.assertRaises(TypeError):
            settings.set(zoh_expm=1)
        with self.assertRaises(ValueError):
            settings.set(zoh_expm="taylor")
        with self.assertRaises(ValueError):  # nothing is applied
            settings.set(jac_check=True, zoh_expm="taylor")
        self.assertFalse(settings.value("jac_check"))
        settings.set(zoh_expm=" BLOCK ")
        self.assertEqual(settings.value("zoh_expm"), "block")
        # None turns off a setting that is off by default, whatever the
        # environment says.
        os.environ["TWIN4BUILD_MATCH_DIAG_FILE"] = "match.log"
        settings.set(match_diag_file=None)
        self.assertEqual(
            settings.snapshot()["match_diag_file"], {"value": None, "source": "set"}
        )

    def test_values_from_the_environment(self):
        spellings = {"1": True, "true": True, "Yes": True, " on ": True}
        spellings.update({"0": False, "false": False, "NO": False, "off": False})
        for text, expected in spellings.items():
            with self.subTest(text=text):
                os.environ["TWIN4BUILD_TESTING"] = text
                self.assertIs(settings.value("testing"), expected)
        for text in ("", "  "):  # an empty variable counts as unset
            os.environ["TWIN4BUILD_TESTING"] = text
            self.assertEqual(
                settings.snapshot()["testing"], {"value": False, "source": "default"}
            )
        os.environ["TWIN4BUILD_TESTING"] = "maybe"
        with self.assertRaisesRegex(ValueError, "TWIN4BUILD_TESTING='maybe'"):
            settings.value("testing")
        os.environ["T4B_ZOH_EXPM"] = "Block"
        self.assertEqual(settings.value("zoh_expm"), "block")
        os.environ["T4B_ZOH_EXPM"] = "taylor"
        with self.assertRaisesRegex(ValueError, "T4B_ZOH_EXPM='taylor'"):
            settings.value("zoh_expm")
        os.environ["TWIN4BUILD_MATCH_DIAG_FILE"] = " Logs/Match.log "
        self.assertEqual(settings.value("match_diag_file"), "Logs/Match.log")

    def test_a_setting_is_declared_once(self):
        with self.assertRaises(ValueError):
            settings._register(settings._REGISTRY["testing"])
        with self.assertRaises(TypeError):
            settings._register(
                settings._Setting(
                    name="steps", env="T4B_TEST_STEPS", kind=int, default=1, doc=""
                )
            )
        self.assertNotIn("steps", settings.snapshot())

    def test_the_log_line_names_values_and_sources_once_per_change(self):
        settings.log_active("test")  # whatever came before: the defaults now
        with mock.patch.object(LOGGER, "info") as info:
            settings.log_active("estimate")
            self.assertFalse(info.called)
            with settings.override(jac_check=True):
                settings.log_active("estimate")
                settings.log_active("simulate")  # unchanged: no second line
            self.assertEqual(info.call_count, 1)
            message, *arguments = info.call_args[0]
            self.assertEqual(
                message % tuple(arguments), "Settings (estimate): jac_check=True (set)"
            )
            settings.log_active("estimate")  # back at the defaults: one line
            self.assertEqual(info.call_count, 2)

    def test_the_package_exports_the_settings(self):
        self.assertIs(tb.settings, settings)
        self.assertIn("settings", tb.__all__)
        # Only the library declares settings: the declaration is private.
        self.assertEqual(
            sorted(settings.__all__),
            ["log_active", "override", "reset", "set", "snapshot", "value"],
        )
        self.assertFalse(hasattr(settings, "Setting"))
        self.assertFalse(hasattr(settings, "register"))

    @unittest.skipUnless(PAGE.exists(), "a source checkout's docs")
    def test_the_manual_describes_every_setting(self):
        page = PAGE.read_text(encoding="utf-8")
        for setting in settings._REGISTRY.values():
            with self.subTest(setting=setting.name):
                self.assertIn(f"``{setting.name}``", page)
                self.assertIn(f"``{setting.env}``", page)


class TestSettingsAreReadWhereTheyAreUsed(_NoSettingsInTheEnvironment):
    """A setting given after the import takes effect at its next use."""

    def test_rollout_checkpoint(self):
        def step(y, theta, u, constants):
            return y * 2.0

        y = torch.ones(2, requires_grad=True)
        checkpoint = torch.utils.checkpoint.checkpoint
        with mock.patch.object(
            torch.utils.checkpoint, "checkpoint", wraps=checkpoint
        ) as recorded:
            _functional._step_call(step, y, None, None, None)
            self.assertFalse(recorded.called)
            with settings.override(rollout_checkpoint=True):
                out = _functional._step_call(step, y, None, None, None)
            self.assertTrue(recorded.called)
        torch.testing.assert_close(out, y * 2.0)

    def test_zoh_expm_and_expm_checkpoint(self):
        A = torch.tensor([[-1.0, 0.2], [0.1, -0.5]], dtype=torch.float64)
        B = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
        u = torch.ones(1, dtype=torch.float64)

        def step(transform_mode=True):
            return dss._discretize_onestep(A, B, None, None, u, 0.5, transform_mode)

        reference = step(transform_mode=False)  # torch.matrix_exp
        checkpoint = torch.utils.checkpoint.checkpoint
        for choice, expm in (("phi1", "_expm_phi1_ss"), ("block", "_expm_ss")):
            with self.subTest(zoh_expm=choice):
                with settings.override(zoh_expm=choice), mock.patch.object(
                    dss, expm, wraps=getattr(dss, expm)
                ) as recorded:
                    Ad, Bd = step()
                self.assertTrue(recorded.called)
                torch.testing.assert_close(Ad, reference[0])
                torch.testing.assert_close(Bd, reference[1])
                with settings.override(zoh_expm=choice, expm_checkpoint=True):
                    with mock.patch.object(
                        torch.utils.checkpoint, "checkpoint", wraps=checkpoint
                    ) as recorded:
                        step()
                self.assertTrue(recorded.called)


class TestResultsCarryTheSettings(_NoSettingsInTheEnvironment):
    """estimate, optimize and pareto_front return the settings they ran with.

    The runs have ``testing`` on, which also stops them after one iteration.
    """

    def _model(self, name):
        model_id = f"test_settings_{name}"
        self.addCleanup(
            shutil.rmtree,
            os.path.join("generated_files", "models", model_id),
            ignore_errors=True,
        )
        return tb.Model(id=model_id)

    def _damper_with_a_setpoint(self, name):
        dates = pd.date_range(start=START, periods=7, freq=f"{STEP}s")
        frame = pd.DataFrame({"value": np.full(len(dates), 0.5)}, index=dates)
        setpoint = TimeSeriesInputSystem(id="setpoint", source=DataFrameSource(frame))
        damper = DamperSystem(id="damper", nominalAirFlowRate=1.0)
        model = self._model(name)
        model.add_connection(setpoint, damper, "value", "damperPosition")
        model.load()
        return model, setpoint, damper

    def test_estimate(self):
        schedule = ScheduleSystem(
            weekday_ruleset={
                "ruleset_start_minute": [0],
                "ruleset_end_minute": [0],
                "ruleset_start_hour": [0],
                "ruleset_end_hour": [24],
                "ruleset_value": [0.5],
                "ruleset_default_value": 0,
            },
            id="schedule",
        )
        damper = DamperSystem(id="damper", a=0.5, nominalAirFlowRate=0.5)
        dates = pd.date_range(start=START, periods=6, freq=f"{STEP}s")
        sensor = SensorSystem(
            id="airflow_sensor",
            source=DataFrameSource(
                pd.DataFrame({"value": np.full(len(dates), 0.3)}, index=dates)
            ),
        )
        model = self._model("estimate")
        model.add_connection(schedule, damper, "scheduleValue", "damperPosition")
        model.add_connection(damper, sensor, "airFlowRate", "measuredValue")
        model.load()
        estimator = tb.Estimator(tb.Simulator(model))
        with settings.override(testing=True):
            result = estimator.estimate(
                parameters=[(damper, "a", 0.5, 0.1, 2.0)],
                measurements=[(sensor, 0.01)],
                start_time=START,
                end_time=END,
                step_size=STEP,
                method=("scipy", "SLSQP", "ad"),
                n_warmup=0,
            )
            expected = settings.snapshot()
        self.assertEqual(result.settings, expected)
        self.assertEqual(result.settings["testing"], {"value": True, "source": "set"})
        self.assertLessEqual(result.iterations, 1)

    def test_optimize(self):
        model, setpoint, damper = self._damper_with_a_setpoint("optimize")
        optimizer = tb.Optimizer(tb.Simulator(model))
        with settings.override(testing=True):
            result = optimizer.optimize(
                variables=[(setpoint, "value", 0.0, 1.0)],
                objectives=[(damper, "airFlowRate", "min")],
                start_time=START,
                end_time=END,
                step_size=STEP,
                method=("scipy", "SLSQP", "ad"),
                options={"maxiter": 50},
            )
            expected = settings.snapshot()
        self.assertEqual(result.settings, expected)
        self.assertEqual(result.settings["testing"], {"value": True, "source": "set"})
        self.assertLessEqual(result.nit, 1)

    def test_pareto_front(self):
        model, setpoint, damper = self._damper_with_a_setpoint("pareto")
        optimizer = tb.Optimizer(tb.Simulator(model, execution_mode="functional"))
        with settings.override(testing=True):
            front = optimizer.pareto_front(
                start_time=START,
                end_time=END,
                step_size=STEP,
                variables=[(setpoint, "value", 0.0, 1.0)],
                objective1=(setpoint, "value", "min"),
                objective2=(damper, "airFlowRate", "max"),
                n_points=5,
                batched_prepass=False,
            )
            expected = settings.snapshot()
        self.assertEqual(front.settings, expected)
        self.assertEqual(front.settings["testing"], {"value": True, "source": "set"})
        self.assertLessEqual(len(front.eps), 3)


# Imports every module of the library (not its tests or examples) while
# recording the environment variables read, then reads one setting.
_RECORD_IMPORTS = """
import importlib, json, os, pkgutil
read = set()
getitem = os._Environ.__getitem__
def recording_getitem(self, key):
    read.add(key)
    return getitem(self, key)
os._Environ.__getitem__ = recording_getitem
import twin4build
for module in pkgutil.walk_packages(twin4build.__path__, "twin4build."):
    if {"tests", "examples", "generated_files"}.isdisjoint(module.name.split(".")):
        try:
            importlib.import_module(module.name)
        except ImportError:  # an optional dependency that is not installed
            pass
variables = {s.env for s in twin4build.settings._REGISTRY.values()}
at_import = sorted(read & variables)
twin4build.settings.value("testing")
print(json.dumps({"at_import": at_import, "at_use": sorted(read & variables)}))
"""


class TestNoModuleReadsTheSettingsAtImport(unittest.TestCase):
    def test_no_module_reads_a_settings_variable_when_it_is_imported(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT)] + [p for p in sys.path if p]
        )
        completed = subprocess.run(
            [sys.executable, "-c", _RECORD_IMPORTS],
            env=environment,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        reads = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(reads["at_import"], [], "read when imported")
        self.assertEqual(reads["at_use"], ["TWIN4BUILD_TESTING"])  # it records


if __name__ == "__main__":
    unittest.main()
