"""Every core System passes ``twin4build.testing.check_system`` (ADR 0001,
decision 11), with the example below; a core System class without an
example fails the table test.

Covered:

1. every System class of the package has an example here, or a reason it
   has none;
2. every example passes ``check_system``;
3. ``check_system`` fails a System whose gradient is not finite, and one
   given while it is part of a model;
4. no System class of the package writes its own ``do_step`` or
   ``initialize`` (decision 1), except the ones not converted yet, a list
   that only shrinks.
"""

# Standard library imports
import datetime
import importlib
import os
import pkgutil
import shutil
import tempfile
import unittest

# Third party imports
import pandas as pd
import torch

# Local application imports
import twin4build as tb
import twin4build.core as core
import twin4build.systems
import twin4build.utils
from twin4build.systems.saref4syst.declarations import Input, Output, Parameter
from twin4build.testing import check_system
from twin4build.systems.utils.pass_input_to_output import PassInputToOutput
from twin4build.systems.utils.sigmoid_gate import BandGate
from twin4build.utils.data_loaders.sources import CsvSource
from twin4build.utils.unit_converters.to_degC_from_degK import ToDegCFromDegK
from twin4build.utils.unit_converters.to_degK_from_degC import ToDegKFromDegC

tb._IS_TESTING = True

START = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


def _ruleset(default, value=None):
    return {
        "ruleset_default_value": default,
        "ruleset_start_minute": [0] if value is not None else [],
        "ruleset_end_minute": [0] if value is not None else [],
        "ruleset_start_hour": [0] if value is not None else [],
        "ruleset_end_hour": [1] if value is not None else [],
        "ruleset_value": [value] if value is not None else [],
    }


def _csv(directory, name, values):
    """A recorded series at 10-minute steps from START."""
    path = os.path.join(directory, f"{name}.csv")
    index = pd.date_range(START, periods=len(values), freq="10min")
    pd.DataFrame({"time": index.strftime("%Y-%m-%d %H:%M:%S%z"), "value": values}).to_csv(path, index=False)
    return path


def examples(directory):
    """``{System class: (the example System, its inputs)}``: one per core
    System class.  ``directory`` holds the files the data sources read."""
    series = [5.0, 6.0, 7.5, 7.0, 6.0, 5.5, 5.0, 4.5]
    return {
        tb.AirHandlingUnitSystem: (
            tb.AirHandlingUnitSystem(id="ahu"),
            {
                "supplyDamperPosition": [0.5, 0.8],
                "exhaustDamperPosition": [0.5, 0.8],
                "exhaustTemperature": [22.0, 23.0],
                "supplyAirTemperatureSetpoint": 18.0,
                "outdoorAirTemperature": 5.0,
            },
        ),
        tb.AirHandlingUnitCoreSystem: (
            tb.AirHandlingUnitCoreSystem(id="ahu_core"),
            {
                "totalSupplyAirFlowRate": 1.0,
                "totalExhaustAirFlowRate": 0.9,
                "returnAirTemperature": 22.0,
                "supplyAirTemperatureSetpoint": 18.0,
                "outdoorAirTemperature": 5.0,
            },
        ),
        tb.AirImbalanceSystem: (
            tb.AirImbalanceSystem(id="imbalance"),
            {"supplyAirFlowRate": [0.2, 0.1], "exhaustAirFlowRate": [0.25]},
        ),
        tb.AirToAirHeatRecoverySystem: (
            tb.AirToAirHeatRecoverySystem(id="recovery"),
            {
                "primaryAirFlowRate": 1.0,
                "secondaryAirFlowRate": 0.9,
                "primaryTemperatureIn": 5.0,
                "secondaryTemperatureIn": 22.0,
                "primaryTemperatureOutSetpoint": 18.0,
            },
        ),
        BandGate: (BandGate(id="band", threshold=0.2, band=0.3), {"inputSignal": 0.4}),
        tb.BuildingSpaceMassSystem: (
            tb.BuildingSpaceMassSystem(id="mass", V=100.0),
            {"supplyAirFlowRate": 0.1, "exhaustAirFlowRate": 0.1, "numberOfPeople": 3.0},
        ),
        tb.BuildingSpaceSystem: (
            tb.BuildingSpaceSystem(id="room", thermal=dict(C_wall=1e6, C_air=1e5, R_out=0.01, R_in=0.02), mass=dict(V=100.0)),
            {
                "outdoorTemperature": 5.0,
                "supplyAirFlowRate": [0.1],
                "exhaustAirFlowRate": [0.1],
                "supplyAirTemperature": 18.0,
                "globalIrradiation": 100.0,
                "numberOfPeople": 2.0,
                "heatGain": 500.0,
                "wallHeatGain": [50.0],
            },
        ),
        tb.BuildingSpaceThermalSystem: (
            tb.BuildingSpaceThermalSystem(id="thermal", C_wall=1e6, C_air=1e5, R_out=0.01, R_in=0.02),
            {
                "outdoorTemperature": 5.0,
                "supplyAirFlowRate": 0.1,
                "exhaustAirFlowRate": 0.1,
                "supplyAirTemperature": 18.0,
                "globalIrradiation": 100.0,
                "numberOfPeople": 2.0,
                "heatGain": 500.0,
            },
        ),
        tb.CascadeControllerSystem: (
            tb.CascadeControllerSystem(id="cascade"),
            {"setpointValue_a": 21.0, "actualValue_a": 20.0, "actualValue_b": 18.0},
        ),
        tb.CoilSystem: (
            tb.CoilSystem(id="coil"),
            {"inletAirTemperature": 10.0, "outletAirTemperatureSetpoint": 18.0, "airFlowRate": 1.0},
        ),
        tb.DamperSystem: (tb.DamperSystem(id="damper"), {"damperPosition": 0.4}),
        tb.FanCoilUnitSystem: (
            tb.FanCoilUnitSystem(id="fcu"),
            {"supplyWaterTemperature": 50.0, "valvePosition": 0.5, "airFlowRate": 0.2, "inletAirTemperature": 20.0},
        ),
        tb.FanSystem: (tb.FanSystem(id="fan"), {"airFlowRate": 1.0, "inletAirTemperature": 18.0}),
        tb.MaxSystem: (tb.MaxSystem(id="max"), {"inputs": [1.0, 3.0, 2.0]}),
        tb.OccupancyDetectorSystem: (tb.OccupancyDetectorSystem(id="detector"), {"occupancy": 2.0}),
        tb.OnOffSystem: (tb.OnOffSystem(id="on_off", threshold=0.5, is_off_value=0.0), {"value": 1.0, "criteriaValue": 0.5}),
        tb.OutdoorEnvironmentSystem: (
            tb.OutdoorEnvironmentSystem(
                id="outdoor",
                outdoorTemperature_source=CsvSource(_csv(directory, "temperature", series)),
                globalIrradiation_source=CsvSource(_csv(directory, "irradiation", [0.0, 50.0, 120.0, 200.0, 150.0, 80.0, 20.0, 0.0])),
            ),
            {},
        ),
        PassInputToOutput: (PassInputToOutput(id="pass"), {"value": 1.5}),
        tb.PIDControllerSystem: (tb.PIDControllerSystem(id="pid", kp=0.1, Ti=600.0), {"actualValue": 20.0, "setpointValue": 21.0}),
        tb.PiecewiseLinearSystem: (tb.PiecewiseLinearSystem(id="curve", X=[0.0, 1.0, 2.0], Y=[0.0, 2.0, 3.0]), {"x": 1.5}),
        tb.ReturnFlowJunctionSystem: (
            tb.ReturnFlowJunctionSystem(id="return"),
            {"airFlowRateIn": [0.2, 0.3], "airTemperatureIn": [21.0, 23.0]},
        ),
        tb.SATCompensatedControllerSystem: (
            tb.SATCompensatedControllerSystem(id="sat_cascade"),
            {"setpointValue_a": 21.0, "actualValue_a": 20.0, "actualValue_b": 18.0},
        ),
        tb.SATLinearRuleSystem: (tb.SATLinearRuleSystem(id="sat_rule"), {"supplyAirTemp": 18.0}),
        tb.ScalarProductSystem: (tb.ScalarProductSystem(id="product"), {"input_1": 2.0, "input_2": 3.0}),
        tb.ScheduleSystem: (tb.ScheduleSystem(id="schedule", weekday_ruleset=_ruleset(1.0, 3.0)), {}),
        tb.SensorSystem: (tb.SensorSystem(id="sensor", source=CsvSource(_csv(directory, "sensor", series))), {}),
        tb.SigmoidGate: (tb.SigmoidGate(id="gate", threshold=0.5), {"inputSignal": 0.4}),
        tb.SpaceHeaterSystem: (
            tb.SpaceHeaterSystem(id="heater", thermalMassHeatCapacity=2e5, UA=45.0, nelements=3),
            {"supplyWaterTemperature": 60.0, "waterFlowRate": 0.02, "indoorTemperature": 20.0},
        ),
        tb.SupplyFlowJunctionSystem: (tb.SupplyFlowJunctionSystem(id="supply"), {"airFlowRateOut": [0.2, 0.3]}),
        tb.ThermalMassNodeSystem: (tb.ThermalMassNodeSystem(id="node"), {"heatFlowRate": [100.0, -50.0]}),
        tb.TimeSeriesInputSystem: (tb.TimeSeriesInputSystem(id="series", source=CsvSource(_csv(directory, "series", series))), {}),
        ToDegCFromDegK: (ToDegCFromDegK(id="to_c"), {"K": 300.0}),
        ToDegKFromDegC: (ToDegKFromDegC(id="to_k"), {"C": 20.0}),
        tb.ValveSystem: (tb.ValveSystem(id="valve"), {"valvePosition": 0.4}),
        tb.WallSystem: (tb.WallSystem(id="wall"), {"temperatureA": 20.0, "temperatureB": 5.0}),
        tb.WeightedSumSystem: (tb.WeightedSumSystem(id="sum"), {"inputs": [1.0, 2.0]}),
    }


#: The System classes of the package without an example, and why.
WITHOUT_EXAMPLE = {
    "ThresholdGate": "a base without ports; SigmoidGate, BandGate and OccupancyDetectorSystem declare them",
    "FusedStateSpaceSystem": "the loader builds it from the state-space Systems it fuses; never written by a user",
    # Written the old way, with their own do_step and initialize; each gets an
    # example when it is converted (see STEPPING_THEMSELVES).
    "DiscreteStatespaceSystem": "not converted yet",
    "ControllerIdentificationSystem": "converted in 2.1",
    "ControllerIdentificationPISystem": "converted in 2.1, with its parent",
    "OccupancySystem": "to be deleted (ADR 0001, decision 7)",
}


#: The System classes of the package that still write their own do_step and
#: initialize, and why (ADR 0001, decision 1).  The list only shrinks.
STEPPING_THEMSELVES = {
    "ControllerIdentificationSystem": "exempt for 2.0, converted in 2.1: it builds its candidate controllers and gates from its wiring at the run start, one set per actuator, which the declarations do not express",
    "FusedStateSpaceSystem": "exempt: the loader builds it from the state-space Systems it fuses and steps it "
    "as one block; nobody writes one",
    "DiscreteStatespaceSystem": "converted to derive + forward once the numerics branches land",
    "OccupancySystem": "waiting on the decision about what replaces its CO2 inversion (decision 7)",
}


def _core_system_classes():
    for package in (twin4build.systems, twin4build.utils):
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    return {
        cls
        for cls in subclasses(core.System)
        if cls.__module__.startswith(("twin4build.systems.", "twin4build.utils.")) and not cls.__name__.startswith("_")
    }


class TestCoreExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        cls.examples = examples(cls.directory)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_every_core_system_has_an_example(self):
        named = {cls.__name__ for cls in self.examples}
        missing = sorted(c.__name__ for c in _core_system_classes() if c.__name__ not in named | set(WITHOUT_EXAMPLE))
        self.assertEqual(missing, [], "core System classes without an example in this table")
        self.assertEqual(sorted(named & set(WITHOUT_EXAMPLE)), [])

    def test_only_the_systems_not_converted_yet_step_themselves(self):
        own = sorted(c.__name__ for c in _core_system_classes() if {"do_step", "initialize"} & set(vars(c)))
        self.assertEqual(own, sorted(STEPPING_THEMSELVES), "do_step and initialize belong to the base (ADR 0001)")
        self.assertTrue(all(STEPPING_THEMSELVES.values()), "every exemption states its reason")

    def test_every_example_passes_check_system(self):
        for cls, (system, inputs) in self.examples.items():
            with self.subTest(system=cls.__name__):
                check_system(system, inputs)


class FragileSystem(core.System):
    """``y = sqrt(|u - a|)``: at ``u == a`` its gradient is not finite."""

    u = Input()
    y = Output()
    a = Parameter(1.0, lb=0.0, ub=10.0)

    def forward(self, x, inputs, p, step_size):
        return x, {"y": torch.sqrt(torch.abs(inputs["u"] - p["a"]))}


class TestCheckSystemFails(unittest.TestCase):
    def test_a_gradient_that_is_not_finite_fails(self):
        with self.assertRaisesRegex(AssertionError, "not finite"):
            check_system(FragileSystem(id="fragile"), {"u": 1.0})

    def test_a_system_of_a_model_is_refused(self):
        model = tb.Model(id="check_system_wired")
        damper = tb.DamperSystem(id="damper")
        model.add_connection(tb.ScheduleSystem(id="s", weekday_ruleset=_ruleset(0.5)), damper, "scheduleValue", "damperPosition")
        with self.assertRaisesRegex(ValueError, "not part of a model"):
            check_system(damper, {"damperPosition": 0.5})


if __name__ == "__main__":
    unittest.main()
