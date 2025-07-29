# Third party imports
import torch.nn as nn
from torch import Tensor

# Local application imports
from twin4build.utils.rgetattr import rgetattr


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# def _set_nested_attr(obj: nn.Module, names: list[str], value: Tensor) -> None:
#     """
#     Set the attribute specified by the given list of names to value.
#     For example, to set the attribute obj.conv.weight,
#     use _del_nested_attr(obj, ['conv', 'weight'], value)
#     """
#     if len(names) == 1:
#         setattr(obj, names[0], value)
#     else:
#         _set_nested_attr(getattr(obj, names[0]), names[1:], value)
