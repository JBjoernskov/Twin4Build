# Standard library imports
import os
import unittest

# Local application imports
# Set test flag
import twin4build
from twin4build.utils.test_notebook import test_notebook
from twin4build.utils.uppath import uppath

twin4build._IS_TESTING = True


# @unittest.skip("Temporarily disabled - skipping all example tests")
class TestExamples(unittest.TestCase):
    # @unittest.skip("Temporarily disabled - notebook test failing")
    def test_minimal_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 3), "examples", "minimal_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    # @unittest.skip("Temporarily disabled - notebook test failing")
    def test_space_co2_controller_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 3),
            "examples",
            "space_co2_controller_example.ipynb",
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    # @unittest.skip("Temporarily disabled - notebook test failing")
    def test_estimator_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 3), "examples", "estimator_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    # @unittest.skip("Temporarily disabled - notebook test failing")
    def test_optimizer_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 3), "examples", "optimizer_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    # @unittest.skip("Temporarily disabled - notebook test failing")
    def test_translator_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 3), "examples", "translator_example.ipynb"
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    # @unittest.skip("Temporarily disabled - notebook test failing")
    def test_bems_lecture_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 3),
            "examples",
            "bems_example_lecture.ipynb",
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")

    # The full-workflow notebook is wired against an external case-study
    # dataset (see ``DATA_ROOT`` in ``full_workflow_example.py``).  When
    # that dataset is not present (e.g. on GitHub CI runners) the notebook
    # cannot execute its sensor-loading cells, so we skip the test rather
    # than report a false negative.  Locally, where the case-study repo
    # is checked out alongside Twin4Build, the test still runs end-to-end.
    _FULL_WORKFLOW_DATA_ROOT = (
        r"C:\Users\jabj\Documents\python\Twin4build-Case-Studies\DP37\model\data"
    )

    @unittest.skipUnless(
        os.path.isdir(_FULL_WORKFLOW_DATA_ROOT),
        "full_workflow_example.ipynb requires the Twin4build-Case-Studies "
        "DP37 dataset; skipping on environments where it is unavailable "
        "(e.g. CI).",
    )
    def test_full_workflow_example(self):
        notebook_path = os.path.join(
            uppath(os.path.abspath(__file__), 3),
            "examples",
            "full_workflow_example.ipynb",
        )
        result = test_notebook(notebook_path)
        self.assertTrue(result, f"Test failed for {notebook_path}")


if __name__ == "__main__":
    unittest.main()
