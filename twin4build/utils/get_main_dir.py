import sys
import os
import tempfile
import warnings
from twin4build.utils.uppath import uppath
def get_main_dir():
    """
    Get the main directory of the project.
    Cannot be used with multiprocessing.
    """
    main = sys.modules['__main__']
    if hasattr(main, '__file__'):
        d = uppath(os.path.abspath(str(main.__file__)), 1)
    else:
        d = tempfile.gettempdir()
        warnings.warn("Could not determine main module path, using temp dir: %s" % d)
    return d

