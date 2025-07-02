import unittest
import os
from twin4build.utils.test_notebook import test_notebook
from twin4build.utils.uppath import uppath

class TestExamples(unittest.TestCase):
    def test_minimal_example(self):
        notebook_path = os.path.join(uppath(os.path.abspath(__file__), 2), "examples", "minimal_example.ipynb")
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_space_co2_controller_example(self):
        notebook_path = os.path.join(uppath(os.path.abspath(__file__), 2), "examples", "space_co2_controller_example.ipynb")
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_building_space_example(self):
        notebook_path = os.path.join(uppath(os.path.abspath(__file__), 2), "examples", "building_space_example.ipynb")
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_optimizer_example(self):
        notebook_path = os.path.join(uppath(os.path.abspath(__file__), 2), "examples", "optimizer_example.ipynb")
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

if __name__ == '__main__':
    unittest.main()
