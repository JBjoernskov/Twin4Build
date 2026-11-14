"""jupyter and nbformat are the [notebooks] extra: importing twin4build, its notebook helper included, needs
neither, and running a notebook without them names the extra."""

# Standard library imports
import os
import subprocess
import sys
import unittest
from unittest import mock

# Local application imports
import twin4build
from twin4build.utils import test_notebook as notebook_helper

twin4build._IS_TESTING = True

#: A fresh interpreter in which the notebook packages cannot be imported.
WITHOUT_NOTEBOOKS = """
import sys
for name in ("jupyter", "nbformat", "nbconvert"):
    sys.modules[name] = None
import twin4build
import twin4build.utils.test_notebook
print("imported")
"""


class TestNotebooksExtra(unittest.TestCase):
    def test_importing_twin4build_needs_no_notebook_package(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        done = subprocess.run([sys.executable, "-c", WITHOUT_NOTEBOOKS], capture_output=True, text=True, env=env,
                              timeout=600)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn("imported", done.stdout)

    def test_running_a_notebook_without_the_extra_names_it(self):
        with mock.patch.dict(sys.modules, {"nbformat": None}):
            with self.assertRaisesRegex(ImportError, r"pip install 'twin4build\[notebooks\]'"):
                notebook_helper.test_notebook("any.ipynb")


if __name__ == "__main__":
    unittest.main()
