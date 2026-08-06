"""Deprecated shim — use ``twin4build.utils.logger`` instead (removed in 2.1)."""

# Standard library imports
import warnings

warnings.warn(
    "'twin4build.utils.print_progress' is deprecated and will be removed in "
    "twin4build 2.1. Use 'twin4build.utils.logger' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Local application imports
from twin4build.utils.logger import (  # noqa: F401,E402
    CURSES_AVAILABLE,
    LOGGER,
    Logger,
    autoreset_print,
    print_color_palette,
)

__all__ = [
    "CURSES_AVAILABLE",
    "LOGGER",
    "Logger",
    "autoreset_print",
    "print_color_palette",
]
