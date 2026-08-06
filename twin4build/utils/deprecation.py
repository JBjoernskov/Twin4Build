# Standard library imports
import warnings
from typing import Any, Dict, List, Optional


_REMOVED_IN = "2.1"


def deprecate_args(
    deprecated_args: List[str],
    new_args: List[Optional[str]],
    positions: List[Optional[int]],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Map deprecated keyword arguments to preferred names and warn.

    Warnings state that the old name will be removed in twin4build 2.1.
    """
    value_map: Dict[str, Any] = {}
    for old_arg, new_arg, pos in zip(deprecated_args, new_args, positions):
        if old_arg not in kwargs:
            continue
        if isinstance(pos, int):
            warnings.warn(
                f"Keyword argument '{old_arg}' is deprecated and will be removed "
                f"in twin4build {_REMOVED_IN}. Use positional argument '{new_arg}' "
                f"instead at position {pos}.",
                DeprecationWarning,
                stacklevel=3,
            )
        elif pos is None:
            if new_arg is None:
                warnings.warn(
                    f"Keyword argument '{old_arg}' is deprecated and will be removed "
                    f"in twin4build {_REMOVED_IN}.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"Keyword argument '{old_arg}' is deprecated and will be removed "
                    f"in twin4build {_REMOVED_IN}. Use '{new_arg}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        else:
            raise ValueError(f"Invalid position: {pos}")

        if new_arg is not None:
            value_map[new_arg] = kwargs[old_arg]
        kwargs.pop(old_arg)
    return value_map


def deprecate_name(old: str, new: str, stacklevel: int = 3) -> None:
    """Emit a DeprecationWarning for a renamed public symbol."""
    warnings.warn(
        f"'{old}' is deprecated and will be removed in twin4build {_REMOVED_IN}. "
        f"Use '{new}' instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
