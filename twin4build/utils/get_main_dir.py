# Standard library imports
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Local application imports
from twin4build.utils.uppath import uppath


class MainPathNotFound(Warning):
    pass


_main_dir = None
_warning_issued = False


def get_main_dir():
    """
    Get the main directory of the project.
    Cannot be used with multiprocessing.
    """
    global _main_dir, _warning_issued

    if _main_dir is not None:
        return _main_dir

    # Check if we're running under Sphinx
    if "sphinx" in sys.modules:
        _main_dir = str(Path(__file__).parent.parent.parent)
        return _main_dir

    main = sys.modules["__main__"]
    if hasattr(main, "__file__"):
        _main_dir = uppath(os.path.abspath(str(main.__file__)), 1)
    else:
        _main_dir = tempfile.gettempdir()
        if not _warning_issued:
            warnings.warn(
                "Could not determine main module path, using temp dir: %s" % _main_dir,
                MainPathNotFound,
                stacklevel=2,
            )
            _warning_issued = True

    return _main_dir
