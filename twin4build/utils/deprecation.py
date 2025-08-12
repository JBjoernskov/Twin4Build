# Standard library imports
import warnings


def deprecate_args(deprecated_args, new_args, positions, kwargs):
    value_map = {}
    for old_arg, new_arg, pos in zip(deprecated_args, new_args, positions):
        if old_arg in kwargs:
            if isinstance(pos, int):
                warnings.warn(
                    f"Keyword argument '{old_arg}' is deprecated. Use positional argument '{new_arg}' instead at position {pos}.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            elif pos is None:
                warnings.warn(
                    f"Keyword argument '{old_arg}' is deprecated. Use '{new_arg}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            else:
                raise ValueError(f"Invalid position: {pos}")

            value_map[new_arg] = kwargs[old_arg]
            kwargs.pop(old_arg)
    return value_map
