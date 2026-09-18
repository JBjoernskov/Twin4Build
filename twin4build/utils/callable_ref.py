"""Name a module-level callable by import path, and back.

A unit transformation is a callable, not a literal: ``Model.serialize()``
stores it as ``module:qualname`` and ``Model.load(filename=...)`` imports it
again.  Lambdas and closures cannot be named and are dropped with a warning.
"""

import importlib
import warnings
from typing import Callable, Optional


def callable_to_ref(fn: Optional[Callable], owner: str = "") -> Optional[str]:
    """``module:qualname`` of ``fn``; ``None`` for ``None`` or an unnamable callable."""
    if fn is None:
        return None
    qualname = getattr(fn, "__qualname__", "")
    if not qualname or "<" in qualname or getattr(fn, "__module__", None) is None:
        warnings.warn(
            f"{owner}: the transformation {fn!r} is not importable by name and will "
            "not survive serialization; use a module-level function.",
            stacklevel=3,
        )
        return None
    return f"{fn.__module__}:{qualname}"


def ref_to_callable(ref: Optional[str]) -> Optional[Callable]:
    """Import the callable named by ``module:qualname`` (``None`` for an empty ref)."""
    if not ref or ref == "None":
        return None
    module_name, _, qualname = ref.partition(":")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
