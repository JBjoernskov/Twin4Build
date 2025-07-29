# Standard library imports
import os
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
    try:
        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Create an ExecutePreprocessor
        ep = ExecutePreprocessor(
            timeout=3600 * 5, kernel_name="python3"
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
