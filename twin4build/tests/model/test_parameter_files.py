"""A load writes a component's parameter file only when its content changes: loading a model again leaves every file
untouched, and a file whose content differs from the model's is written back (26 Oct 2026: every load rewrote every
component's file, 2 778 on the HTR model)."""
import json
import os
import unittest

import twin4build as tb
from twin4build.tests.estimator.example_fixture import load_model

tb._IS_TESTING = True


class TestParameterFiles(unittest.TestCase):
    def setUp(self):
        self.model = load_model()
        self.sim = self.model.simulation_model
        # the fixture sets its sensors' data files after the load: one load writes those (a genuine change)
        self.sim._load_parameters()
        self.files = {}
        for component in self.sim.components.values():
            path, exists = self.sim.get_dir(folder_list=["model_parameters", component.__class__.__name__],
                                            filename=f"{component.id}.json")
            if exists:
                self.files[component.id] = path
        self.assertTrue(self.files)

    def _stamps(self):
        return {cid: os.stat(path).st_mtime_ns for cid, path in self.files.items()}

    def test_loading_again_writes_nothing(self):
        before = self._stamps()
        self.sim._load_parameters()
        self.assertEqual(self._stamps(), before)

    def test_a_file_that_differs_is_written_back(self):
        cid, path = next(iter(self.files.items()))
        with open(path) as f:
            text = f.read()
        config = json.loads(text)
        config["_stale_entry_for_test"] = 1  # a file differing from what the model writes
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        others = {c: s for c, s in self._stamps().items() if c != cid}
        self.sim._load_parameters()
        with open(path) as f:
            self.assertNotIn("_stale_entry_for_test", json.load(f))
        self.assertEqual({c: s for c, s in self._stamps().items() if c != cid}, others)


if __name__ == "__main__":
    unittest.main()
