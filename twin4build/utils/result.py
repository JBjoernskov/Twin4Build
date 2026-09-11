"""Shared result container for Twin4Build workflows."""

from __future__ import annotations

from typing import Any


class ResultDict(dict):
    """Dictionary with synchronized attribute access.

    Estimation and optimization results use the same access convention:
    ``result["success"]`` and ``result.success`` are interchangeable, including
    for backend-specific metadata added after construction.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def copy(self):
        """Return a shallow copy preserving the concrete result type."""
        copied = type(self).__new__(type(self))
        dict.__init__(copied, self)
        return copied
