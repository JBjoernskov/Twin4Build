# Standard library imports
import os


def uppath(_path, n):
    """
    Returns the absolute path of "_path" where n levels are removed.
    E.g.
    uppath("C:/Example/test/path/file", 2)
    gives  "C:/Example/test"
    """
    return os.sep.join(_path.split(os.sep)[:-n])
