# Standard library imports
import functools

# Local application imports
from twin4build.utils.rgetattr import rgetattr


def rdelattr(obj, attr):
    """
    Recursively delete an attribute using dot notation.

    Args:
        obj: The object to delete the attribute from
        attr: The attribute path using dot notation (e.g., 'a.b.c')

    Example:
        rdelattr(obj, 'layer.weight.data')  # deletes obj.layer.weight.data
    """
    pre, _, post = attr.rpartition(".")
    return delattr(rgetattr(obj, pre) if pre else obj, post)
