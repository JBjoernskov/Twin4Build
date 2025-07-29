# Standard library imports
import functools

# Local application imports
from twin4build.utils.rgetattr import rgetattr


def rhasattr(obj, attr, *args):
    try:
        rgetattr(obj, attr)
    except AttributeError:
        return False
    return True
