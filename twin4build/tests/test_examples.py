# Standard library imports
import os
import unittest

# Local application imports
from twin4build.utils.test_notebook import test_notebook
from twin4build.utils.uppath import uppath


class TestExamples(unittest.TestCase):
    def test_minimal_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 2), "examples", "minimal_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_space_co2_controller_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 2),
            "examples",
            "space_co2_controller_example.ipynb",
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_estimator_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 2), "examples", "estimator_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_optimizer_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 2), "examples", "optimizer_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    def test_translator_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 2), "examples", "translator_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")


if __name__ == "__main__":
    unittest.main()
