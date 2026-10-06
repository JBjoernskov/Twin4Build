"""The main directory, and the other small accessors a workflow needs.

* ``set_main_dir``: where models keep their files, set by the project
  instead of following the script that happens to run;
* ``Model.serialize`` returns the path of the saved instance graph and
  ``Model.instance_graph_path`` gives it without serializing;
* ``discretize_onestep``: the one-step discretization the state-space
  components step with, under a public name;
* ``SpaceHeaterSystem.solve_UA``: the nominal-sizing ``UA`` before the model
  is initialized.
"""
import datetime
import math
import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import torch
from dateutil import tz

import twin4build as tb
import twin4build.utils.get_main_dir as main_dir_module
from twin4build.systems.utils import discrete_statespace_system
from twin4build.systems.utils.discrete_statespace_system import discretize_onestep
from twin4build.utils.get_main_dir import get_main_dir, set_main_dir

tb._IS_TESTING = True

from twin4build.tests.simulator.test_fusion_batched import build

START = datetime.datetime(2024, 1, 4, tzinfo=tz.UTC)
STEP = 600


class MainDirCase(unittest.TestCase):
    """The main directory is a temporary one for the test and what it was
    afterwards."""

    def setUp(self):
        self.before = main_dir_module._main_dir
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        set_main_dir(self.tmp)

    def tearDown(self):
        main_dir_module._main_dir = self.before
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestSetMainDir(MainDirCase):
    def test_the_directory_set_is_the_directory_got(self):
        self.assertEqual(get_main_dir(), self.tmp)
        other = os.path.join(self.tmp, "project")
        self.assertEqual(set_main_dir(pathlib.Path(other)), other)
        self.assertEqual(get_main_dir(), other)
        self.assertIsInstance(get_main_dir(), str)

    def test_a_relative_path_is_made_absolute(self):
        self.assertEqual(set_main_dir("some_project"), os.path.abspath("some_project"))
        self.assertTrue(os.path.isabs(get_main_dir()))

    def test_models_keep_their_files_there(self):
        model = tb.Model(id="main_dir_files")
        path, exists = model.get_dir(folder_list=["results"], filename="a.txt")
        self.assertFalse(exists)
        self.assertEqual(
            path, os.path.join(self.tmp, "generated_files", "models", "main_dir_files", "results", "a.txt")
        )
        self.assertTrue(os.path.isdir(os.path.dirname(path)))

    def test_none_has_it_determined_again(self):
        self.assertIsNone(set_main_dir(None))
        found = get_main_dir()
        self.assertIsInstance(found, str)
        self.assertNotEqual(found, self.tmp)


class TestInstanceGraphPath(MainDirCase):
    def test_serialize_returns_the_path_of_the_instance_graph(self):
        model = build(n_pairs=2, model_id="main_dir_serialize")
        expected = os.path.join(
            self.tmp, "generated_files", "models", "main_dir_serialize", "simulation_model",
            "semantic_model", "instance_graph.ttl",
        )
        # the path is known before anything is written, and asking for it writes nothing
        self.assertEqual(model.instance_graph_path, expected)
        self.assertFalse(os.path.exists(os.path.dirname(expected)))
        path = model.serialize()
        self.assertEqual(path, expected)
        self.assertEqual(model.instance_graph_path, expected)
        self.assertTrue(os.path.isfile(path))
        # what the case code asked the private semantic model for
        self.assertEqual(path, model._simulation_model._semantic_model.get_dir(filename="instance_graph.ttl")[0])

        reloaded = tb.Model(id="main_dir_reloaded")
        reloaded.load(filename=path, draw_semantic_model=False, draw_simulation_model=False)
        self.assertEqual(set(reloaded.components), set(model.components))
        self.assertAlmostEqual(float(reloaded.components["Radiator1"].UA.get().reshape(-1)[0]), 45.0)


class TestDiscretizeOnestep(unittest.TestCase):
    def test_a_first_order_lag(self):
        """``dx/dt = -a x + a u`` held over ``T``: ``x_next = e^(-aT) x + (1 - e^(-aT)) u``."""
        a, T = 1e-3, 600.0
        A = torch.tensor([[[-a]]], dtype=torch.float64)
        B = torch.tensor([[[a]]], dtype=torch.float64)
        u = torch.tensor([[20.0]], dtype=torch.float64)
        for transform_mode in (None, False, True):
            Ad, Bd = discretize_onestep(A, B, None, None, u, T, transform_mode=transform_mode)
            self.assertEqual(tuple(Ad.shape), (1, 1, 1))
            self.assertEqual(tuple(Bd.shape), (1, 1, 1))
            self.assertAlmostEqual(float(Ad), math.exp(-a * T), places=9)
            self.assertAlmostEqual(float(Bd), 1.0 - math.exp(-a * T), places=9)

    def test_the_bilinear_terms_and_the_private_name(self):
        """With a flow that carries heat (``E``, ``F``) the public function
        is the one the components step with."""
        torch.manual_seed(0)
        n, m = 3, 2
        A = -torch.diag(torch.rand(n, dtype=torch.float64) + 0.5).unsqueeze(0) * 1e-3
        B = torch.rand((1, n, m), dtype=torch.float64) * 1e-3
        E = -torch.rand((1, m, n, n), dtype=torch.float64) * 1e-4
        F = torch.rand((1, m, n, m), dtype=torch.float64) * 1e-4
        u = torch.tensor([[0.5, 20.0]], dtype=torch.float64)
        Ad, Bd = discretize_onestep(A, B, E, F, u, 600.0, transform_mode=False)
        Ad_p, Bd_p = discrete_statespace_system._discretize_onestep(A, B, E, F, u, 600.0, transform_mode=False)
        torch.testing.assert_close(Ad, Ad_p, rtol=0, atol=0)
        torch.testing.assert_close(Bd, Bd_p, rtol=0, atol=0)
        # against the matrix exponential of the effective matrices
        A_eff = A[0] + E[0, 0] * u[0, 0] + E[0, 1] * u[0, 1]
        B_eff = B[0] + F[0, 0] * u[0, 0] + F[0, 1] * u[0, 1]
        block = torch.zeros((n + m, n + m), dtype=torch.float64)
        block[:n, :n], block[:n, n:] = A_eff * 600.0, B_eff * 600.0
        expected = torch.matrix_exp(block)
        torch.testing.assert_close(Ad[0], expected[:n, :n], rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(Bd[0], expected[:n, n:], rtol=1e-12, atol=1e-14)


class TestSolveUA(unittest.TestCase):
    def _heater(self, **kwargs):
        defaults = dict(
            Q_flow_nominal_sh=2400.0, T_a_nominal_sh=60.0, T_b_nominal_sh=45.0, TAir_nominal_sh=21.0,
            thermalMassHeatCapacity=1.2e5, nelements=3, id="heater",
        )
        defaults.update(kwargs)
        return tb.SpaceHeaterSystem(**defaults)

    def _initialize(self, heater):
        heater.initialize(start_time=[START], end_time=[START + datetime.timedelta(hours=1)], step_size=[STEP])

    def test_the_value_meets_the_nominal_sizing(self):
        heater = self._heater()
        ua = heater.solve_UA()
        self.assertIsInstance(ua, float)
        self.assertGreater(ua, 0.0)
        self.assertAlmostEqual(heater._ua_residual(np.asarray([ua])), 0.0, places=6)
        # between the UA of the outlet's and of the inlet's temperature difference
        self.assertLess(ua, 2400.0 / (45.0 - 21.0))
        self.assertGreater(ua, 2400.0 / (60.0 - 21.0))
        # a larger radiator needs a larger UA
        self.assertAlmostEqual(self._heater(Q_flow_nominal_sh=4800.0).solve_UA() / ua, 2.0, places=6)

    def test_the_component_is_not_changed(self):
        heater = self._heater()
        heater.solve_UA()
        self.assertAlmostEqual(float(heater.UA.get().reshape(-1)[0]), 10.0)  # the placeholder
        self.assertTrue(heater.initialize_UA)
        self.assertFalse(heater.INITIALIZED)

    def test_it_is_the_value_initialize_sets(self):
        heater = self._heater()
        ua = heater.solve_UA()
        self._initialize(heater)
        self.assertAlmostEqual(float(heater.UA.get().reshape(-1)[0]), ua, places=9)

    def test_a_radiator_sized_before_the_model_is_initialized(self):
        heater = self._heater()
        heater.Q_flow_nominal_sh = 3600.0  # the room's design heat loss
        ua = heater.solve_UA()
        heater.UA.set(ua, normalized=False)
        heater.initialize_UA = False
        self.assertAlmostEqual(float(heater.UA.get().reshape(-1)[0]), ua, places=9)
        self._initialize(heater)
        self.assertAlmostEqual(float(heater.UA.get().reshape(-1)[0]), ua, places=9)
        self.assertAlmostEqual(ua / self._heater().solve_UA(), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
