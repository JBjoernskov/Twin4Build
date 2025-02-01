import unittest
import os

import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")

from twin4build.utils.test_notebook import test_notebook
from twin4build.utils.uppath import uppath



class TestExamples(unittest.TestCase):
    def test_minimal_example(self):
        notebook_path = os.path.join(uppath(os.path.abspath(__file__), 1), "minimal_example.ipynb")
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_space_co2_controller_example(self):
        notebook_path = os.path.join(uppath(os.path.abspath(__file__), 1), "space_co2_controller_example.ipynb")
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_parameter_estimation_example(self):
        notebook_path = os.path.join(uppath(os.path.abspath(__file__), 1), "parameter_estimation_example", "parameter_estimation_example.ipynb")
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

if __name__ == '__main__':
    unittest.main()
