# Standard library imports
import json
import os
import shutil
import sys
import tempfile
import traceback

# Third party imports
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


def test_notebook(notebook_path):
    """
    Execute a notebook and return True if it runs without errors, False otherwise.
    Prints detailed error information if an exception occurs.

    :param notebook_path: str, path to the notebook file
    :return: bool, True if notebook executes without errors, False otherwise
    """
    # Create a temporary kernel spec using the current Python executable so
    # the notebook runs in the same environment as the test runner.
    tmpdir = tempfile.mkdtemp(prefix="t4b_kernel_")
    kernel_name = "python_current"
    kernel_dir = os.path.join(tmpdir, "kernels", kernel_name)
    os.makedirs(kernel_dir)
    kernel_spec = {
        "argv": [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Python (current)",
        "language": "python",
    }
    with open(os.path.join(kernel_dir, "kernel.json"), "w") as f:
        json.dump(kernel_spec, f)

    old_jupyter_path = os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_PATH"] = tmpdir + os.pathsep + old_jupyter_path

    # Notebooks run in a subprocess kernel, so module-level flags like
    # ``twin4build._IS_TESTING`` don't propagate.  Set an env var that
    # ``Estimator.estimate(...)`` and ``Optimizer.optimize(...)`` honor
    # by capping ``maxiter`` to 1, so the example notebooks complete in
    # seconds instead of minutes.  The notebooks still execute every
    # cell end-to-end (so we still catch import / API errors), they
    # just don't burn cycles on real convergence.
    old_fast_flag = os.environ.get("TWIN4BUILD_TESTING", None)
    os.environ["TWIN4BUILD_TESTING"] = "1"
    try:
        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(
            timeout=3600 * 5, kernel_name=kernel_name
        )  # 5 hour timeout

        # Execute the notebook
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})

        return True
    except CellExecutionError as e:
        print(f"Error executing notebook {notebook_path}:")
        if hasattr(e, "traceback"):
            print("Traceback:")
            print(e.traceback)
        else:
            print("Error message:")
            print(str(e))

        if hasattr(e, "source"):
            print("Cell source:")
            print(e.source)

        return False
    except Exception:
        print(f"Error executing notebook {notebook_path}:")
        print(traceback.format_exc())
        return False
    finally:
        if old_jupyter_path:
            os.environ["JUPYTER_PATH"] = old_jupyter_path
        else:
            os.environ.pop("JUPYTER_PATH", None)
        if old_fast_flag is None:
            os.environ.pop("TWIN4BUILD_TESTING", None)
        else:
            os.environ["TWIN4BUILD_TESTING"] = old_fast_flag
        shutil.rmtree(tmpdir, ignore_errors=True)


# Prevent pytest from collecting this function as a test
test_notebook.__test__ = False
