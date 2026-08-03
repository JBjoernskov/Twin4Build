from __future__ import annotations  # This allows using string literals in type hints

# Standard library imports
import datetime
import functools
import os
import sys
import warnings
from collections import OrderedDict
from typing import List, Optional, Union

# Third party imports
import numpy as np
import torch
import torch.nn as nn
from dateutil import tz

# Local application imports
import twin4build.core as core
from twin4build.utils.deprecation import deprecate_args

# ---------------------------------------------------------------------------
# Framework-wide floating-point dtype.
#
# Twin4Build allocates simulation tensors in many places (ports, states,
# state-space matrices, data tables).  All of them resolve their dtype through
# float_dtype() so the whole framework can switch precision in one place --
# Model.to(dtype=...) calls set_float_dtype().  The default (float64) matches
# the historical behavior.  Note this is a process-wide setting, not
# per-model: running two models with different dtypes in one process is not
# supported.
# ---------------------------------------------------------------------------
_FLOAT_DTYPE: torch.dtype = torch.float64


def float_dtype() -> torch.dtype:
    """The framework-wide floating-point dtype (default ``torch.float64``)."""
    return _FLOAT_DTYPE


def set_float_dtype(dtype: torch.dtype) -> None:
    """Set the framework-wide floating-point dtype.

    Called by ``Model.to(dtype=...)``.  Only floating-point dtypes are
    accepted; ``torch.float32`` trades accuracy for large speedups on
    consumer GPUs whose float64 throughput is fractional.
    """
    global _FLOAT_DTYPE
    dtype = torch.empty(0, dtype=dtype).dtype  # normalize/validate
    if not dtype.is_floating_point:
        raise ValueError(f"float dtype required, got {dtype}")
    _FLOAT_DTYPE = dtype


class Vector:
    """A custom vector implementation.

    This class implements a vector (1D array) wrapper around PyTorch tensors with
    support for history logging, batching, and normalization.

    Attributes:
        tensor (torch.Tensor): The underlying tensor storing vector values with shape (n_s, n_c, n_v).
        n_s (int): Number of simulations.
        n_c (int): Number of parallel components.
        n_v (int): The size of the vector (number of elements).
        log_history (bool): Whether to log the history of values.
        history (torch.Tensor): The history of values over time with shape (n_s, n_c, n_t, n_v).
        is_leaf (bool): Whether this vector is a leaf node in the graph (input).
        do_normalization (bool): Whether to normalize the history.
        optional (bool): Whether the vector is optional.
    """

    def __init__(
        self,
        tensor: Optional[Union[float, int]] = None,
        n_v: Optional[int] = None,
        log_history: bool = True,
        is_leaf: bool = False,
        do_normalization: bool = False,
        optional: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize a Vector instance.

        Args:
            tensor (Optional[Union[float, int]]): Initial value to broadcast. None means zeros.
            n_v (Optional[int]): The size of the vector.
            log_history (bool): Whether to log history. Defaults to True.
            is_leaf (bool): Whether this vector is a leaf node. Defaults to False.
            do_normalization (bool): Whether to normalize history. Defaults to False.
            optional (bool): Whether this vector is optional. Defaults to False.
        """
        # Handle deprecated arguments
        deprecated_args = ["size", "n_timesteps", "n_s", "n_c", "n_t"]
        new_args = ["n_v", None, None, None, None]
        positions = [None, None, None, None, None]
        value_map = deprecate_args(deprecated_args, new_args, positions, kwargs)
        n_v = value_map.get("n_v", n_v)

        # Only accept float/int/None for tensor arg
        assert isinstance(
            tensor, (float, int, type(None))
        ), "tensor must be a float, int, or None"

        self._tensor = None  # Will be created in initialize()
        self._n_s = None  # Will be set in initialize()
        self._n_c = None  # Will be set in initialize()
        self._n_v = n_v  # Can be set here or in initialize()
        self._n_t = None  # Will be set in initialize()
        self._log_history = log_history
        self._is_leaf = is_leaf
        self._do_normalization = do_normalization
        self._optional = optional

        self._init_value = tensor  # Store raw value for initialize()

        self._history = None
        self._normalized_history = None
        self._initialized = False
        self._requires_reinittialization = True
        self._min_history = None  # Will be set to float when first calculated
        self._max_history = None  # Will be set to float when first calculated
        self._history_is_populated = False
        self._is_normalized = False

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value

    @property
    def init_value(self):
        """Initial value used in initialize() to create tensor."""
        return self._init_value

    @init_value.setter
    def init_value(self, value: Union[float, int]) -> None:
        """Set initial value (float/int to broadcast)."""
        assert isinstance(value, (float, int)), "init_value must be float or int"
        self._init_value = value

    @property
    def n_v(self):
        """Size of the vector."""
        return self._n_v

    @property
    def size(self):
        """Size of the vector. Alias for n_v for backward compatibility."""
        return self._n_v

    @property
    def n_s(self):
        """Number of simulations."""
        return self._n_s

    @property
    def n_c(self):
        """Number of parallel components."""
        return self._n_c

    @property
    def batch_size(self):
        """Total batch size (n_s * n_c). Provided for backward compatibility."""
        return self._n_s * self._n_c

    @property
    def n_t(self):
        """Number of timesteps."""
        return self._n_t

    @property
    def n_timesteps(self):
        """Number of timesteps. Alias for n_t for backward compatibility."""
        return self._n_t

    @property
    def log_history(self):
        return self._log_history

    @log_history.setter
    def log_history(self, value: bool):
        self._log_history = value

    def history(
        self,
        i_t: Union[int, slice] = slice(None),
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
        i_v: Union[int, slice] = slice(None),
    ) -> torch.Tensor:
        """Get the history tensor with optional slicing.

        Args:
            i_t (Union[int, slice]): Time index. Defaults to slice(None) for all.
            i_s (Union[int, slice]): Simulation index (n_s dimension). Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index (n_c dimension). Defaults to slice(None) for all.
            i_v (Union[int, slice]): Vector index (n_v dimension). Defaults to slice(None) for all.

        Returns:
            torch.Tensor: History tensor with shape (n_t, n_s, n_c, n_v) or reduced shape if indices specified.
        """
        assert (
            self._history_is_populated
        ), "History is not populated. Set log_history to True to populate history during simulation."
        # Internal storage is (n_t, n_s, n_c, n_v)
        return self._history[i_t, i_s, i_c, i_v]

    @property
    def normalized_history(self):
        return self._normalized_history

    @property
    def is_leaf(self):
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, value: bool):
        assert isinstance(value, bool), "is_leaf must be a boolean"
        self._is_leaf = value

    @property
    def do_normalization(self):
        return self._do_normalization

    @do_normalization.setter
    def do_normalization(self, value: bool):
        assert isinstance(value, bool), "do_normalization must be a boolean"
        self._do_normalization = value

    @property
    def optional(self):
        return self._optional

    def __str__(self) -> str:
        """Get string representation of the vector.

        Returns:
            str: String representation of the tensor value.
        """
        return str(self._tensor)

    def set_requires_grad(self, requires_grad: bool):
        """Set requires_grad for the history tensor (leaf vectors only)."""
        assert (
            self._is_leaf or not requires_grad
        ), "Only leaf vectors can have their requires_grad attribute set to True"
        # If history not yet finalized and setting to False, nothing to do
        if self._history is None:
            if not requires_grad:
                self._requires_reinittialization = True
                return
            else:
                raise ValueError("Cannot set requires_grad=True on unfinalized history")
        if self._do_normalization:
            self._normalized_history.requires_grad = requires_grad
        else:
            self._history.requires_grad = requires_grad
        self._requires_reinittialization = not requires_grad

    def make_pickable(self):
        if self.tensor is not None:
            if self.n_v > 0:
                self.tensor = torch.tensor(
                    [self.tensor.item()], dtype=float_dtype(), requires_grad=False
                )
            else:
                self.tensor = torch.tensor([], dtype=float_dtype(), requires_grad=False)

        # _init_value is already a simple float/int, nothing to do

    def __getitem__(self, key: int) -> float:
        """Get value at specified index.

        Args:
            key (int): Index to access.

        Returns:
            float: Value at specified index.
        """
        return self.tensor[key]

    def __setitem__(self, key: int, value: float) -> None:
        """Set value at specified index.

        Args:
            key (int): Index to set.
            value (float): Value to set.
        """
        self.tensor[key] = value

    def initialize(
        self,
        n_t: int = None,
        n_s: Optional[int] = 1,
        n_c: Optional[int] = 1,
        n_v: Optional[int] = None,
        values: Optional[List[float]] = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the vector tensor and history.

        Creates the underlying torch tensor with shape (n_s, n_c, n_v).
        History has shape (n_s, n_c, n_t, n_v).

        Args:
            n_t (int): Number of timesteps.
            n_s (Optional[int]): Number of simulations.
            n_c (Optional[int]): Number of parallel components.
            n_v (Optional[int]): Size of the vector.
            values (Optional[List[float]]): Initial values for leaf vectors.
            force (bool): Force reinitialization.
        """
        # Handle deprecated arguments
        deprecated_args = ["n_timesteps", "size", "batch_size"]
        new_args = ["n_t", "n_v", "n_s"]
        positions = [None, None, None]
        value_map = deprecate_args(deprecated_args, new_args, positions, kwargs)
        n_t = value_map.get("n_t", n_t)
        n_v = value_map.get("n_v", n_v)
        n_s = value_map.get("n_s", n_s)

        assert isinstance(n_t, int), "n_t must be an integer"

        self._n_t = n_t
        self._n_s = n_s
        self._n_c = n_c
        if n_v is not None:
            self._n_v = n_v

        # Create tensor with shape (n_s, n_c, n_v) from _init_value
        if self._init_value is None:
            self._tensor = torch.zeros(
                (self.n_s, self.n_c, self.n_v), dtype=float_dtype()
            )
        else:
            # Broadcast init_value to full tensor
            self._tensor = torch.full(
                (self.n_s, self.n_c, self.n_v), self._init_value, dtype=float_dtype()
            )

        if values is not None:
            values = _expand_to_4D_tensor(values, self.n_s, self.n_c)

        # We return early if this vector has requires_grad=True.
        # This is the case when used in the optimizer.
        # Here we dont want to reinitialize the history as the torch.optim.Optimizer changes this in-place.
        if (
            self._initialized
            and self._requires_reinittialization == False
            and force == False
        ):
            # For non-leaf vectors, reset history for new simulation run
            if not self._is_leaf:
                self._history.zero_()
                self._history_is_populated = False
            return

        if self._is_leaf:
            assert values is not None, "Values must be provided for leaf vectors"
            # Values expected in time-first format: (n_t, n_s, n_c, n_v)
            assert (
                values.shape[0] == self.n_t
            ), f"Values first dim ({values.shape[0]}) must match n_t ({self.n_t})"
            assert (
                values.shape[1] == self.n_s
            ), f"Values second dim ({values.shape[1]}) must match n_s ({self.n_s})"
            assert (
                values.shape[2] == self.n_c
            ), f"Values third dim ({values.shape[2]}) must match n_c ({self.n_c})"
            assert (
                values.shape[3] == self.n_v
            ), f"Values fourth dim ({values.shape[3]}) must match n_v ({self.n_v})"
            # Values already in internal time-first format (n_t, n_s, n_c, n_v)
            self._history = values
            self._history_is_populated = True
            if self._do_normalization:
                self._normalized_history = self.normalize()

        else:
            # Pre-allocate history with time-first layout for efficient writes
            # Internal shape: (n_t, n_s, n_c, n_v)
            self._history = torch.zeros(
                self.n_t,
                self.n_s,
                self.n_c,
                self.n_v,
                dtype=float_dtype(),
                requires_grad=False,
            )
            self._history_is_populated = False

        self._initialized = True
        return self

    def _set(
        self,
        v: torch.Tensor = None,
        i_t: Optional[int] = None,
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
        i_v: Union[int, slice] = slice(None),
        transformation: callable = None,
    ) -> None:
        """Private efficient setter - v must be a correctly shaped tensor (or None for leaf).

        This is the hot path used during simulation. No type conversion or
        shape validation is performed.

        Args:
            v (torch.Tensor): Pre-shaped tensor matching target slice. None for leaf vectors.
            i_t (Optional[int]): Step index for history logging (required if log_history=True or is_leaf=True).
            i_s (Union[int, slice]): Simulation index (n_s dimension). Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index (n_c dimension). Defaults to slice(None) for all.
            i_v (Union[int, slice]): Vector index (n_v dimension). Defaults to slice(None) for all.
            transformation (callable): Optional function to transform to the value.
        """
        # Handle leaf vectors (get value from history)
        if self._is_leaf:
            assert v is None, "Values cannot be set for leaf vectors"
            assert i_t is not None, "i_t must be provided for leaf vectors"
            if self._do_normalization:
                v = self.denormalize(
                    self._normalized_history[i_t]
                )  # Time-first: (n_t, n_s, n_c, n_v)
            else:
                v = self._history[i_t]  # Time-first layout: shape (n_s, n_c, n_v)

        # Apply transformation if provided
        if transformation is not None:
            v = transformation(v)

        # Direct assignment - slice(None) acts as ':'
        self._tensor[i_s, i_c, i_v] = v

        # Log history - direct write to time-first tensor
        if self._log_history:
            assert i_t is not None, "i_t must be provided when log_history=True"
            is_leaf = self._is_leaf
            if not is_leaf or (is_leaf and self._do_normalization):
                # Direct write to pre-allocated tensor: _history[i_t, i_s, i_c, i_v]
                # i_s defaults to slice(None), i_c and i_v can be int or slice
                self._history[i_t, i_s, i_c, i_v] = v

            if i_t == self.n_t - 1:
                self._history_is_populated = True

    def set(
        self,
        v: Union[float, int, torch.Tensor] = None,
        i_t: Optional[int] = None,
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
        i_v: Union[int, slice] = slice(None),
        transformation: callable = None,
    ) -> None:
        """Public convenient setter - handles type conversion and broadcasting.

        Args:
            v (Union[float, int, torch.Tensor]): Value to set. Can be scalar,
                1D, 2D, or 3D tensor. Will be broadcast to match target shape. None for leaf vectors.
            i_t (Optional[int]): Step index for history logging.
            i_s (Union[int, slice]): Simulation index (n_s dimension). Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index (n_c dimension). Defaults to slice(None) for all.
            i_v (Union[int, slice]): Vector index (n_v dimension). Defaults to slice(None) for all.
            transformation (callable): Optional function to transform the value.
        """
        # For non-leaf, prepare value with unified conversion logic
        if not self._is_leaf:
            v = _prepare_value_for_set(
                v,
                target_shape=(self.n_s, self.n_c, self.n_v),
                indices={"i_s": i_s, "i_c": i_c, "i_v": i_v},
                device=self._tensor.device if self._tensor is not None else None,
            )

        # Delegate to efficient private method
        self._set(v, i_t, i_s, i_c, i_v, transformation)

    def get(
        self,
        i_t: Optional[int] = None,
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
        i_v: Union[int, slice, torch.Tensor] = slice(None),
        **kwargs,
    ) -> torch.Tensor:
        """Get vector values.

        Args:
            i_t (Optional[int]): Time index (for accessing history).
            i_s (Union[int, slice]): Simulation index within n_s dimension. Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index within n_c dimension. Defaults to slice(None) for all.
            i_v (Union[int, slice, torch.Tensor]): Index within the vector (n_v dimension). Defaults to slice(None) for all.

        Returns:
            torch.Tensor: Tensor of values with shape depending on indices:
                - (n_s, n_c, n_v) if no indices specified
                - Various reduced shapes when indices are specified
                - None if not yet initialized
        """
        if self._tensor is None:
            return None
        # clone() decouples the returned value from _tensor's storage so
        # that later inplace writes (by _set at the next timestep) do not
        # invalidate saved tensors in torch.func.jacrev's backward pass.
        return self._tensor[i_s, i_c, i_v].clone()

    def copy(self):
        """Create a copy of the vector.

        Returns:
            Vector: A new Vector instance with the same data.
        """
        copy = Vector(
            tensor=self._init_value,
            n_v=self.n_v,
            log_history=self.log_history,
            is_leaf=self.is_leaf,
            do_normalization=self.do_normalization,
            optional=self.optional,
        )
        return copy

    def normalize(self, v: torch.Tensor = None):
        """Normalize values using min-max scaling.

        Args:
            v (torch.Tensor, optional): Values to normalize. If None, normalizes history.

        Returns:
            torch.Tensor: Normalized values.
        """
        assert (
            self._history_is_populated
        ), "History must be populated before normalizing"
        if v is None:
            v = self._history
        v = _expand_to_4D_tensor(v, self.n_s, self.n_c)
        assert isinstance(v, torch.Tensor), "v must be a torch.Tensor"

        # Cache min/max as Python floats to avoid GradTrackingTensor issues
        if self._min_history is None:
            no_nan_history = self._history.detach()
            no_nan_history = no_nan_history[~torch.isnan(no_nan_history)]
            self._min_history = torch.min(no_nan_history).item()
        if self._max_history is None:
            no_nan_history = self._history.detach()
            no_nan_history = no_nan_history[~torch.isnan(no_nan_history)]
            self._max_history = torch.max(no_nan_history).item()

        # Convert cached floats to tensors when needed
        min_val = torch.tensor(self._min_history, dtype=float_dtype())
        max_val = torch.tensor(self._max_history, dtype=float_dtype())

        if torch.allclose(min_val, max_val):
            min_val = torch.tensor(0, dtype=float_dtype())
            if torch.allclose(max_val, torch.tensor(0, dtype=float_dtype())):
                max_val = torch.tensor(1, dtype=float_dtype())
            else:
                max_val = torch.tensor(1, dtype=float_dtype())

        self._is_normalized = True
        return (v - min_val) / (max_val - min_val)

    def denormalize(self, v: torch.Tensor):
        """Denormalize values from min-max scaling.

        Args:
            v (torch.Tensor): Normalized values to denormalize.

        Returns:
            torch.Tensor: Denormalized values.
        """
        assert self._is_normalized, ".normalize() must be called before denormalizing"
        # Use cached float values and convert to tensors
        min_val = torch.tensor(self._min_history, dtype=float_dtype())
        max_val = torch.tensor(self._max_history, dtype=float_dtype())
        return v * (max_val - min_val) + min_val


class Scalar:
    """A custom scalar implementation with operator overloading.

    This class wraps a single scalar value and provides arithmetic operations
    compatibility with other Scalar instances, numeric types, and numpy arrays.

    Attributes:
        tensor (torch.Tensor): The wrapped scalar value with shape (n_s, n_c).
        n_s (int): Number of simulations.
        n_c (int): Number of parallel components.
        log_history (bool): Whether to log the history of values.
        history (torch.Tensor): The history of values over time with shape (n_s, n_c, n_t).
        is_leaf (bool): Whether this scalar is a leaf node in the graph (input).
        do_normalization (bool): Whether to normalize the history.
        optional (bool): Whether the scalar is optional.
    """

    def __init__(
        self,
        tensor: Optional[Union[float, int]] = None,
        log_history: bool = True,
        is_leaf: bool = False,
        do_normalization: bool = False,
        optional: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize a Scalar instance.

        Args:
            tensor (Optional[Union[float, int]]): Initial value to broadcast. None means zeros.
            log_history (bool): Whether to log history. Defaults to True.
            is_leaf (bool): Whether this scalar is a leaf node. Defaults to False.
            do_normalization (bool): Whether to normalize history. Defaults to False.
            optional (bool): Whether this scalar is optional. Defaults to False.
        """
        # Handle deprecated arguments
        deprecated_args = ["scalar", "n_timesteps", "n_s", "n_c", "n_t"]
        new_args = ["tensor", None, None, None, None]
        positions = [None, None, None, None, None]
        value_map = deprecate_args(deprecated_args, new_args, positions, kwargs)
        tensor = value_map.get("tensor", tensor)

        # Only accept float/int/None for tensor arg
        assert isinstance(
            tensor, (float, int, type(None))
        ), "tensor must be a float, int, or None"

        self._tensor = None  # Will be created in initialize()
        self._n_s = None  # Will be set in initialize()
        self._n_c = None  # Will be set in initialize()
        self._n_t = None  # Will be set in initialize()
        self._init_value = tensor  # Store raw value for initialize()
        self._history = None
        self._normalized_history = None
        self._log_history = log_history
        self._is_leaf = is_leaf
        self._do_normalization = do_normalization
        self._initialized = False
        self._requires_reinittialization = True
        self._min_history = None  # Will be set to float when first calculated
        self._max_history = None  # Will be set to float when first calculated
        self._history_is_populated = False
        self._is_normalized = False
        self._optional = optional

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value

    @property
    def scalar(self):
        """Deprecated. Use 'tensor' instead."""
        warnings.warn(
            "Property 'scalar' is deprecated. Use 'tensor' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._tensor

    @property
    def init_value(self):
        """Initial value used in initialize() to create tensor."""
        return self._init_value

    @init_value.setter
    def init_value(self, value: Union[float, int]) -> None:
        """Set initial value (float/int to broadcast)."""
        assert isinstance(value, (float, int)), "init_value must be float or int"
        self._init_value = value

    @property
    def n_s(self):
        """Number of simulations."""
        return self._n_s

    @property
    def n_c(self):
        """Number of parallel components."""
        return self._n_c

    @property
    def batch_size(self):
        """Total batch size (n_s * n_c). Provided for backward compatibility."""
        return self._n_s * self._n_c

    @property
    def n_t(self):
        """Number of timesteps."""
        return self._n_t

    @property
    def n_timesteps(self):
        """Number of timesteps. Alias for n_t for backward compatibility."""
        return self._n_t

    @property
    def log_history(self):
        return self._log_history

    @log_history.setter
    def log_history(self, value: bool):
        self._log_history = value

    def history(
        self,
        i_t: Union[int, slice] = slice(None),
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
    ) -> torch.Tensor:
        """Get the history tensor with optional slicing.

        Args:
            i_t (Union[int, slice]): Time index. Defaults to slice(None) for all.
            i_s (Union[int, slice]): Simulation index (n_s dimension). Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index (n_c dimension). Defaults to slice(None) for all.

        Returns:
            torch.Tensor: History tensor with shape (n_t, n_s, n_c) or reduced shape if indices specified.
        """
        assert (
            self._history_is_populated
        ), "History is not populated. Set log_history to True to populate history during simulation."
        # Internal storage is (n_t, n_s, n_c)
        return self._history[i_t, i_s, i_c]

    @property
    def normalized_history(self):
        return self._normalized_history

    @property
    def is_leaf(self):
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, value: bool):
        assert isinstance(value, bool), "is_leaf must be a boolean"
        self._is_leaf = value

    @property
    def do_normalization(self):
        return self._do_normalization

    @do_normalization.setter
    def do_normalization(self, value: bool):
        assert isinstance(value, bool), "do_normalization must be a boolean"
        self._do_normalization = value

    @property
    def optional(self):
        return self._optional

    def __str__(self) -> str:
        """Get string representation of the scalar.

        Returns:
            str: String representation of the scalar value.
        """
        return str(self._tensor)

    def set_requires_grad(self, requires_grad: bool):
        """Set requires_grad for the history tensor (leaf scalars only)."""
        assert (
            self._is_leaf or not requires_grad
        ), "Only leaf scalars can have their requires_grad attribute set to True"
        # If history not yet finalized and setting to False, nothing to do
        if self._history is None:
            if not requires_grad:
                self._requires_reinittialization = True
                return
            else:
                raise ValueError("Cannot set requires_grad=True on unfinalized history")
        if self._do_normalization:
            self._normalized_history.requires_grad = requires_grad
        else:
            self._history.requires_grad = requires_grad
        self._requires_reinittialization = not requires_grad

    def initialize(
        self,
        n_t: int = None,
        n_s: Optional[int] = 1,
        n_c: Optional[int] = 1,
        values: Optional[List[float]] = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the scalar tensor and history.

        Creates the underlying torch tensor with shape (n_s, n_c).
        History has shape (n_s, n_c, n_t).

        Args:
            n_t (int): Number of timesteps.
            n_s (Optional[int]): Number of simulations.
            n_c (Optional[int]): Number of parallel components.
            values (Optional[List[float]]): Initial values for leaf scalars.
            force (bool): Force reinitialization.
        """
        # Handle deprecated arguments
        deprecated_args = ["n_timesteps", "batch_size"]
        new_args = ["n_t", "n_s"]
        positions = [None, None]
        value_map = deprecate_args(deprecated_args, new_args, positions, kwargs)
        n_t = value_map.get("n_t", n_t)
        n_s = value_map.get("n_s", n_s)

        assert isinstance(n_t, int), "n_t must be an integer"

        self._n_s = n_s
        self._n_c = n_c
        self._n_t = n_t

        # Create tensor with shape (n_s, n_c) from _init_value
        if self._init_value is None:
            self._tensor = torch.zeros((self.n_s, self.n_c), dtype=float_dtype())
        else:
            # Broadcast init_value to full tensor
            self._tensor = torch.full(
                (self.n_s, self.n_c), self._init_value, dtype=float_dtype()
            )

        if values is not None:
            values = _expand_to_3D_scalar_tensor(values, self.n_s, self.n_c)

        # We return early if this scalar has requires_grad=True.
        # This is the case when used in the optimizer.
        # Here we dont want to reinitialize the history as the torch.optim.Optimizer changes this in-place.
        if (
            self._initialized
            and self._requires_reinittialization == False
            and force == False
        ):
            # For non-leaf scalars, reset history for new simulation run
            if not self._is_leaf:
                self._history.zero_()
                self._history_is_populated = False
            return

        if self._is_leaf:
            assert values is not None, "Values must be provided for leaf scalars"
            # Values expected in time-first format: (n_t, n_s, n_c)
            assert (
                values.shape[0] == self.n_t
            ), f"First dimension of values ({values.shape[0]}) must match n_t ({self.n_t})."
            assert (
                values.shape[1] == self.n_s
            ), f"Second dimension of values ({values.shape[1]}) must match n_s ({self.n_s})."
            assert (
                values.shape[2] == self.n_c
            ), f"Third dimension of values ({values.shape[2]}) must match n_c ({self.n_c})."
            # Values already in internal time-first format (n_t, n_s, n_c)
            self._history = values
            self._history_is_populated = True
            if self._do_normalization:
                self._normalized_history = self.normalize()

        else:
            # Pre-allocate history with time-first layout for efficient writes
            # Internal shape: (n_t, n_s, n_c)
            self._history = torch.zeros(
                self.n_t, self.n_s, self.n_c, dtype=float_dtype(), requires_grad=False
            )
            self._history_is_populated = False

        self._initialized = True

    def _set(
        self,
        v: torch.Tensor = None,
        i_t: Optional[int] = None,
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
        transformation: callable = None,
        **kwargs,
    ) -> None:
        """Private efficient setter - v must be a correctly shaped tensor (or None for leaf).

        This is the hot path used during simulation. No type conversion or
        shape validation is performed.

        Args:
            v (torch.Tensor): Pre-shaped tensor matching target slice. None for leaf scalars.
            i_t (Optional[int]): Step index for history logging (required if log_history=True or is_leaf=True).
            i_s (Union[int, slice]): Simulation index (n_s dimension). Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index (n_c dimension). Defaults to slice(None) for all.
            transformation (callable): Optional function to transform the value.
        """
        # Handle leaf scalars (get value from history)
        if self._is_leaf:
            assert v is None, "Values cannot be set for leaf scalars"
            assert i_t is not None, "i_t must be provided for leaf scalars"
            if self._do_normalization:
                v = self.denormalize(
                    self._normalized_history[i_t]
                )  # Time-first: (n_t, n_s, n_c)
            else:
                v = self._history[i_t]  # Time-first layout: shape (n_s, n_c)

        # Apply transformation if provided
        if transformation is not None:
            v = transformation(v)

        # Direct assignment - slice(None) acts as ':'
        self._tensor[i_s, i_c] = v

        # Log history - direct write to time-first tensor
        if self._log_history:
            assert i_t is not None, "i_t must be provided when log_history=True"
            is_leaf = self._is_leaf
            if not is_leaf or (is_leaf and self._do_normalization):
                # Direct write to pre-allocated tensor: _history[i_t, i_s, i_c]
                # i_s defaults to slice(None), i_c can be int or slice
                self._history[i_t, i_s, i_c] = v

            if i_t == self.n_t - 1:
                self._history_is_populated = True

    def set(
        self,
        v: Union[Scalar, float, int, torch.Tensor] = None,
        i_t: Optional[int] = None,
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
        transformation: callable = None,
        **kwargs,
    ) -> None:
        """Public convenient setter - handles type conversion and broadcasting.

        Args:
            v (Union[Scalar, float, torch.Tensor]): Value to set. Can be scalar,
                1D, or 2D tensor. Will be broadcast to match target shape. None for leaf scalars.
            i_t (Optional[int]): Step index for history logging.
            i_s (Union[int, slice]): Simulation index (n_s dimension). Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index (n_c dimension). Defaults to slice(None) for all.
            transformation (callable): Optional function to transform the value.
        """
        # For non-leaf, prepare value with unified conversion logic
        if not self._is_leaf:
            v = _prepare_value_for_set(
                v,
                target_shape=(self.n_s, self.n_c),
                indices={"i_s": i_s, "i_c": i_c},
                device=self._tensor.device if self._tensor is not None else None,
            )

        # Delegate to efficient private method
        self._set(v, i_t, i_s, i_c, transformation)

    def get(
        self,
        i_t: Optional[int] = None,
        i_s: Union[int, slice] = slice(None),
        i_c: Union[int, slice] = slice(None),
        i_v: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Get the scalar value.

        Args:
            i_t (Optional[int]): Time index (for accessing history).
            i_s (Union[int, slice]): Simulation index within n_s dimension. Defaults to slice(None) for all.
            i_c (Union[int, slice]): Component index within n_c dimension. Defaults to slice(None) for all.
            i_v (Optional[int]): Unused for Scalar, included for API compatibility.

        Returns:
            torch.Tensor: Scalar value with shape depending on indices:
                - (n_s, n_c) if no indices specified
                - (n_c,) if i_s specified
                - (n_s,) if i_c specified
                - scalar if both i_s and i_c specified
                - None if not yet initialized
        """
        if self._tensor is None:
            return None
        # clone() decouples the returned value from _tensor's storage so
        # that later inplace writes (by _set at the next timestep) do not
        # invalidate saved tensors in torch.func.jacrev's backward pass.
        return self._tensor[i_s, i_c].clone()

    def normalize(self, v: torch.Tensor = None):
        assert (
            self._history_is_populated == True
        ), "History must be populated before normalizing"
        if v is None:
            v = self._history

        # Handle different input shapes - don't force 3D for scalar inputs
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=float_dtype())

        # Cache min/max as Python floats to avoid GradTrackingTensor issues
        if self._min_history is None:
            no_nan_history = self._history.detach()
            no_nan_history = no_nan_history[~torch.isnan(no_nan_history)]
            self._min_history = torch.min(no_nan_history).item()
        if self._max_history is None:
            no_nan_history = self._history.detach()
            no_nan_history = no_nan_history[~torch.isnan(no_nan_history)]
            self._max_history = torch.max(no_nan_history).item()

        # Convert cached floats to tensors when needed
        min_val = torch.tensor(self._min_history, dtype=float_dtype())
        max_val = torch.tensor(self._max_history, dtype=float_dtype())

        if torch.allclose(min_val, max_val):
            min_val = torch.tensor(0, dtype=float_dtype())
            if torch.allclose(max_val, torch.tensor(0, dtype=float_dtype())):
                max_val = torch.tensor(1, dtype=float_dtype())
            else:
                max_val = torch.tensor(1, dtype=float_dtype())

        self._is_normalized = True
        return (v - min_val) / (max_val - min_val)

    def denormalize(self, v: torch.Tensor):
        assert (
            self._is_normalized == True
        ), ".normalize() must be called before denormalizing"
        # Use cached float values and convert to tensors
        min_val = torch.tensor(self._min_history, dtype=float_dtype())
        max_val = torch.tensor(self._max_history, dtype=float_dtype())
        return v * (max_val - min_val) + min_val

    def get_float(self) -> float:
        """Get the scalar value as a float.

        Returns:
            float: Scalar value.
        """
        return self._tensor.item()

    def copy(self):
        """Create a copy of the scalar.

        Returns:
            Scalar: A new Scalar instance with the same data.
        """
        copy = Scalar(
            tensor=self._init_value,
            log_history=self.log_history,
            is_leaf=self.is_leaf,
            do_normalization=self.do_normalization,
            optional=self.optional,
        )
        return copy


def normalize_unit(
    v: torch.Tensor,
    min_value: torch.Tensor,
    max_value: torch.Tensor,
    log_scaling=False,
) -> torch.Tensor:
    """THE physical -> normalized-[0, 1] map (single source of truth).

    Pure tensor math, torch.func-safe (no Parameter access, no ``.item()``),
    shared by :class:`Parameter` / :class:`TensorParameter` and every
    estimator code path.  ``log_scaling`` is a bool (one parameter) or a
    boolean tensor mask (a mixed theta vector, elementwise).
    """
    if isinstance(log_scaling, torch.Tensor):
        lin = (v - min_value) / (max_value - min_value)
        safe_lb = min_value.clamp(min=1e-30)
        safe_ub = max_value.clamp(min=1e-30)
        logv = (torch.log(v.clamp(min=1e-30)) - torch.log(safe_lb)) / (
            torch.log(safe_ub) - torch.log(safe_lb)
        )
        return torch.where(log_scaling, logv, lin)
    if log_scaling:
        return (torch.log(v) - torch.log(min_value)) / (
            torch.log(max_value) - torch.log(min_value)
        )
    return (v - min_value) / (max_value - min_value)


def denormalize_unit(
    z: torch.Tensor,
    min_value: torch.Tensor,
    max_value: torch.Tensor,
    log_scaling=False,
) -> torch.Tensor:
    """THE normalized-[0, 1] -> physical map (single source of truth).

    Inverse of :func:`normalize_unit`; same contract (torch.func-safe, bool or
    boolean-mask ``log_scaling``).
    """
    if isinstance(log_scaling, torch.Tensor):
        lin = min_value + z * (max_value - min_value)
        safe_lb = min_value.clamp(min=1e-30)
        safe_ub = max_value.clamp(min=1e-30)
        logv = torch.exp(
            torch.log(safe_lb) + z * (torch.log(safe_ub) - torch.log(safe_lb))
        )
        return torch.where(log_scaling, logv, lin)
    if log_scaling:
        log_min = torch.log(min_value)
        return torch.exp(z * (torch.log(max_value) - log_min) + log_min)
    return z * (max_value - min_value) + min_value


def theta_bound_tensors(parameters, device=None):
    """Plain ``(lb, ub, log_mask)`` tensors for a flat parameter list.

    The vectorized companion to :func:`denormalize_unit`: the estimator's fast
    paths denormalize a whole theta vector at once and need the physical
    bounds and scaling as plain tensors (``Parameter`` itself is a Tensor
    subclass and breaks under functorch).  Scalar bounds only (``n_c == 1``).
    ``device`` places the bounds where the rollout runs (the model's device).
    """
    lb = torch.tensor(
        [float(np.asarray(p.min_value.detach().cpu()).flatten()[0]) for p in parameters],
        dtype=float_dtype(),
        device=device,
    )
    ub = torch.tensor(
        [float(np.asarray(p.max_value.detach().cpu()).flatten()[0]) for p in parameters],
        dtype=float_dtype(),
        device=device,
    )
    log_mask = torch.tensor(
        [getattr(p, "scaling", "linear") == "log" for p in parameters],
        device=device,
    )
    return lb, ub, log_mask


class Parameter(nn.Parameter):
    """
    A custom nn.Parameter implementation that normalizes the data between 0 and 1 to stabilize gradients in physical systems where the parameters scales can be different.
    This makes it possible to use torch.optim.Optimizer to optimize the parameters.

    Supports an optional `n_c` dimension for parallel components, allowing the same
    parameter to have different values for multiple parallel instances.

    Args:
        data: The parameter value (scalar or 1D tensor). Created as scalar initially,
              use expand_to_n_c() in initialize() to expand to multiple components.
        min_value: Minimum value for normalization.
        max_value: Maximum value for normalization.
        requires_grad: Whether to track gradients for this parameter.
        n_c: Optional number of parallel components. When given, ``data`` is
            broadcast to shape ``(n_c,)`` at construction. When ``None``
            (default), ``n_c`` is inferred from ``data`` and can be expanded
            later with ``expand_to_n_c()`` during ``initialize()``.
        scaling: Normalization scaling mode. ``"linear"`` (default) uses standard
            min-max normalization. ``"log"`` uses logarithmic normalization so that
            equal steps in normalized [0, 1] space correspond to equal multiplicative
            changes in the physical value. Log scaling requires ``min_value > 0``.
    """

    def __new__(
        cls,
        data,
        min_value=None,
        max_value=None,
        requires_grad=True,
        n_c=None,
        scaling="linear",
    ):
        assert scaling in (
            "linear",
            "log",
        ), f"scaling must be 'linear' or 'log', got '{scaling}'"

        if n_c is not None:
            data = _broadcast_for_n_c(data, n_c)
        # Prepare data - convert to tensor with shape (n_c,) where n_c is inferred
        data, n_c = _prepare_parameter_data(data)

        # Set min and max values with defaults - all should have shape (n_c,)
        if min_value is None:
            if torch.all(data < 0):
                min_value = data.detach().clone()
            elif scaling == "log":
                # Log scaling requires min > 0; default to data/10
                min_value = (data.detach().clone().abs() * 0.1).clamp(min=1e-10)
            else:
                min_value = torch.zeros(n_c, dtype=float_dtype())
        else:
            min_value = _prepare_bound_value(min_value, data.shape, n_c)

        if max_value is None:
            if torch.all(data < 0):
                max_value = torch.zeros(n_c, dtype=float_dtype())
            elif torch.allclose(data, torch.zeros_like(data)):
                max_value = torch.ones(n_c, dtype=float_dtype())
            else:
                max_value = data.detach().clone()
        else:
            max_value = _prepare_bound_value(max_value, data.shape, n_c)

        assert torch.all(
            max_value > min_value
        ), "max_value must be greater than min_value"

        if scaling == "log":
            assert torch.all(min_value > 0), "min_value must be > 0 for log scaling"
            # Normalize in log-space: (log(v) - log(min)) / (log(max) - log(min))
            normalized_data = (torch.log(data) - torch.log(min_value)) / (
                torch.log(max_value) - torch.log(min_value)
            )
        else:
            # Normalize the data (linear)
            normalized_data = (data - min_value) / (max_value - min_value)

        # Create the parameter using the parent's __new__ method
        instance = super().__new__(cls, normalized_data, requires_grad)

        # Store min and max values as properties
        instance._min_value = min_value
        instance._max_value = max_value
        instance._n_c = n_c
        instance._scaling = scaling

        return instance

    def __reduce_ex__(self, proto):
        """Custom serialization method that reuses PyTorch's logic but returns our own rebuild function."""
        # Get the state using our own logic (equivalent to PyTorch's)
        state = _get_tps_obj_state(self)

        # Add our custom attributes to the state
        if state is None:
            state = {}
        elif isinstance(state, dict):
            state = state.copy()
        else:
            # If state is not a dict (e.g., tuple from slots), convert to dict
            state = {"__dict__": state} if not isinstance(state, dict) else state.copy()

        # Add our custom attributes
        state["_min_value"] = self._min_value
        state["_max_value"] = self._max_value
        state["_n_c"] = self._n_c
        state["_scaling"] = self._scaling
        state["_is_tps_parameter"] = True

        # Use our own rebuild functions
        hooks = OrderedDict()
        if not state:
            return (
                _rebuild_tps_parameter,
                (self.data, self.requires_grad, hooks),
            )
        else:
            return (
                _rebuild_tps_parameter_with_state,
                (self.data, self.requires_grad, hooks, state),
            )

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def n_c(self):
        """Number of parallel components."""
        return self._n_c

    @property
    def scaling(self):
        """Normalization scaling mode ('linear' or 'log')."""
        return self._scaling

    @min_value.setter
    def min_value(self, value):
        # Bounds follow the parameter's device/dtype so they stay valid after
        # Model.to (bounds are often assigned after the move, e.g. by
        # Estimator._set_bounds).
        self._min_value = _match_tensor(
            _broadcast_for_n_c(value, self._n_c), self.data
        )

    @max_value.setter
    def max_value(self, value):
        self._max_value = _match_tensor(
            _broadcast_for_n_c(value, self._n_c), self.data
        )

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        v = _match_tensor(_broadcast_for_n_c(v, self._n_c), self.data)

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _broadcast_for_n_c(min_value, self._n_c)

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _broadcast_for_n_c(max_value, self._n_c)

        min_value = _match_tensor(min_value, self.data)
        max_value = _match_tensor(max_value, self.data)
        self._min_value = min_value
        self._max_value = max_value
        assert (
            torch.allclose(min_value, max_value) == False
        ), "min_value and max_value must be different"

        return normalize_unit(
            v, self._min_value, self._max_value, self._scaling == "log"
        )

    def denormalize(self, v: torch.Tensor):
        return denormalize_unit(
            v, self._min_value, self._max_value, self._scaling == "log"
        )

    def get(self):
        """Get the denormalized value."""
        return self.denormalize(self)

    def set(self, value, normalized: bool = True):
        """Set the parameter value (will be normalized internally)."""
        value = _broadcast_for_n_c(value, self._n_c)

        if normalized:
            normalized_value = value
        else:
            normalized_value = self.normalize(value)

        self.data.copy_(normalized_value)

    def expand_to_n_c(self, n_c: int):
        """
        Expand this parameter to support n_c parallel components.

        If the parameter is currently scalar (n_c=1), it will be broadcast to shape (n_c,).
        If already has n_c components, this is a no-op.

        Args:
            n_c: Number of parallel components to expand to.

        Returns:
            A new Parameter with n_c components.
        """
        if self._n_c == n_c:
            return self
        if self._n_c != 1:
            raise ValueError(
                f"Cannot expand parameter with n_c={self._n_c} to n_c={n_c}"
            )

        # Get denormalized value and expand to shape (n_c,)
        denorm_value = self.get()
        if denorm_value.dim() == 1 and denorm_value.shape[0] == 1:
            denorm_value = denorm_value.expand(n_c).clone()
        elif denorm_value.dim() == 0:
            denorm_value = denorm_value.expand(n_c).clone()

        # Expand min/max values to shape (n_c,)
        min_val = self._min_value
        max_val = self._max_value
        if min_val.dim() == 1 and min_val.shape[0] == 1:
            min_val = min_val.expand(n_c).clone()
        if max_val.dim() == 1 and max_val.shape[0] == 1:
            max_val = max_val.expand(n_c).clone()

        # n_c is inferred from denorm_value shape in constructor
        return Parameter(
            denorm_value,
            min_value=min_val,
            max_value=max_val,
            requires_grad=self.requires_grad,
            scaling=self._scaling,
        )


class State:
    """A System's continuous internal state -- first-class, alongside :class:`Parameter` / :class:`Scalar` / :class:`Vector`.

    A ``System`` holds three kinds of things: I/O ports (:class:`Scalar` /
    :class:`Vector`), parameters (:class:`Parameter`), and **state**.  State was
    historically stored as ad-hoc plain attributes (``self.x``, ``self.err_prev``,
    ``ss_model.x``); ``State`` makes it a declared, typed member so it can be
    enumerated generically (which components carry state, how big, get/set) --
    the foundation for multiple-shooting / collocation, where each ``State``
    becomes a per-segment boundary decision variable.

    Semantics vs. the neighbouring types:

    * Unlike ports, ``State`` keeps **no time history** -- it is a single current
      snapshot of shape ``(n_s, n_c, n_v)`` (n_v = the state dimension).
    * Unlike :class:`Parameter`, it is **not optimized** as ``theta`` and is **not**
      an ``nn.Parameter`` -- in the functional one-step (``forward``) it is passed
      as an explicit *argument*, not substituted via ``functional_call``.
    * It carries an **initial-condition hook** (``init_value`` constant or
      ``init_fn`` callable evaluated at :meth:`initialize`), the one thing ports /
      parameters don't model -- e.g. seeding air/wall temperatures from a
      component's start values or output ports.

    Args:
        n_v: State dimension (number of state variables). May be set at
            :meth:`initialize` instead.
        init_value: Constant initial value broadcast to ``(n_s, n_c, n_v)`` when
            no ``init_fn`` is given.
        init_fn: Optional ``callable(component) -> tensor`` evaluated at
            :meth:`initialize`, returning the initial state (broadcastable to
            ``(n_s, n_c, n_v)``).  Lets a component derive its initial condition
            from its own parameters / output ports.
        lb, ub: Optional physical bounds (each broadcastable to ``(n_v,)``) used to
            box the boundary decision variables in collocation.
        names: Optional per-dimension names for diagnostics.
    """

    def __init__(
        self,
        n_v: Optional[int] = None,
        init_value: Union[float, int] = 0.0,
        init_fn: Optional[callable] = None,
        lb=None,
        ub=None,
        names: Optional[List[str]] = None,
    ) -> None:
        self._n_v = n_v
        self._init_value = init_value
        self._init_fn = init_fn
        self._lb = lb
        self._ub = ub
        self._names = names
        self._tensor = None
        self._n_s = None
        self._n_c = None
        self._initialized = False

    @property
    def tensor(self):
        return self._tensor

    @property
    def n_v(self):
        return self._n_v

    @property
    def n_s(self):
        return self._n_s

    @property
    def n_c(self):
        return self._n_c

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    def names(self) -> List[str]:
        if self._names is not None:
            return list(self._names)
        return [f"x{i}" for i in range(self._n_v or 0)]

    def initialize(
        self,
        n_s: int = 1,
        n_c: int = 1,
        n_v: Optional[int] = None,
        component=None,
        force: bool = False,
    ) -> "State":
        """Allocate the state tensor and apply the initial condition.

        Args:
            n_s: Number of parallel simulations (segments in collocation).
            n_c: Number of parallel components.
            n_v: State dimension (overrides the constructor value if given).
            component: The owning System, forwarded to ``init_fn`` so a component
                can seed its state from its own attributes / ports.
            force: Reinitialize even if already initialized (re-seeds the value).
        """
        if n_v is not None:
            self._n_v = n_v
        assert self._n_v is not None, "State.initialize requires n_v"
        self._n_s = n_s
        self._n_c = n_c

        if self._initialized and not force:
            # Keep the current value; just (re)validate the batch shape.  A new
            # simulation run re-seeds explicitly via force=True or set().
            if self._tensor is not None and self._tensor.shape == (n_s, n_c, self._n_v):
                return self

        if self._init_fn is not None and component is not None:
            val = self._init_fn(component)
            self._tensor = self._to_shape(val, n_s, n_c, self._n_v)
        else:
            self._tensor = torch.full(
                (n_s, n_c, self._n_v), float(self._init_value), dtype=float_dtype()
            )
        self._initialized = True
        return self

    @staticmethod
    def _to_shape(v, n_s, n_c, n_v):
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=float_dtype())
        v = v.to(float_dtype())
        if v.dim() == 1 and v.shape[0] == n_v:  # (n_v,)
            v = v.reshape(1, 1, n_v).expand(n_s, n_c, n_v).clone()
        elif v.dim() == 2 and v.shape == (n_c, n_v):  # (n_c, n_v)
            v = v.reshape(1, n_c, n_v).expand(n_s, n_c, n_v).clone()
        elif v.dim() == 3 and v.shape == (n_s, n_c, n_v):
            v = v.clone()
        elif v.dim() == 0:
            v = v.reshape(1, 1, 1).expand(n_s, n_c, n_v).clone()
        else:
            raise ValueError(
                f"State value shape {tuple(v.shape)} not broadcastable to "
                f"({n_s}, {n_c}, {n_v})"
            )
        return v

    def get(self) -> torch.Tensor:
        """Current state, shape ``(n_s, n_c, n_v)`` (cloned so later in-place
        writes can't invalidate tensors captured by autograd / ``jacrev``)."""
        return None if self._tensor is None else self._tensor.clone()

    def set(self, v: torch.Tensor) -> None:
        """Set the state from a tensor broadcastable to ``(n_s, n_c, n_v)``.

        Preserves gradients (does not detach) so a decision-variable slice can be
        written straight in during multiple-shooting / collocation.  A tensor
        already at the exact ``(n_s, n_c, n_v)`` shape is stored as-is (no clone),
        matching the simulation hot path (``self.x = x_new``).
        """
        if (
            isinstance(v, torch.Tensor)
            and v.dim() == 3
            and v.shape == (self._n_s, self._n_c, self._n_v)
        ):
            self._tensor = v
        else:
            self._tensor = self._to_shape(v, self._n_s, self._n_c, self._n_v)

    def reset(self, tensor: torch.Tensor) -> None:
        """Adopt ``tensor`` as the current state, syncing ``(n_s, n_c, n_v)`` from
        its shape.  Used when a component re-assigns its whole state (e.g. on a new
        simulation run whose batch size ``n_s`` differs)."""
        assert (
            isinstance(tensor, torch.Tensor) and tensor.dim() == 3
        ), f"State.reset expects a 3D (n_s, n_c, n_v) tensor, got {getattr(tensor, 'shape', type(tensor))}"
        self._n_s, self._n_c, self._n_v = tensor.shape
        self._tensor = tensor
        self._initialized = True

    def copy(self) -> "State":
        return State(
            n_v=self._n_v,
            init_value=self._init_value,
            init_fn=self._init_fn,
            lb=self._lb,
            ub=self._ub,
            names=self._names,
        )

    def __str__(self) -> str:
        return str(self._tensor)


class TensorParameter:
    """
    A custom nn.Parameter implementation that normalizes the data between 0 and 1 to stabilize gradients in physical systems where the parameters scales can be different.

    This class is used to represent model parameters as a Tensor when we calculate the Jacobian analytically as the jac = torch.nn.functional.Jacobian() has the signature jac(f: callable, input: Tensor) -> Tensor.

    Supports an `n_c` dimension for parallel components via expand_to_n_c() method.

    Args:
        tensor: The parameter value (scalar or 1D tensor). Created as scalar initially,
                use expand_to_n_c() in initialize() to expand to multiple components.
        min_value: Minimum value for normalization.
        max_value: Maximum value for normalization.
        normalized: Whether the input tensor is already normalized.
        scaling: Normalization scaling mode. ``"linear"`` (default) uses standard
            min-max normalization. ``"log"`` uses logarithmic normalization so that
            equal steps in normalized [0, 1] space correspond to equal multiplicative
            changes in the physical value. Log scaling requires ``min_value > 0``.

    Note:
        n_c (number of parallel components) is not specified at construction time.
        Use expand_to_n_c() method during initialize() when n_c is known.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        min_value=None,
        max_value=None,
        normalized: bool = True,
        scaling: str = "linear",
    ):
        assert scaling in (
            "linear",
            "log",
        ), f"scaling must be 'linear' or 'log', got '{scaling}'"

        # Prepare tensor (converts numpy/list to tensor), infer n_c from shape
        tensor, n_c = _prepare_parameter_data(tensor)
        self._n_c = n_c
        self._scaling = scaling

        # Process min/max values with broadcasting
        self._min_value = _prepare_bound_value(min_value, tensor.shape, n_c)
        self._max_value = _prepare_bound_value(max_value, tensor.shape, n_c)

        if scaling == "log" and self._min_value is not None:
            assert torch.all(
                self._min_value > 0
            ), "min_value must be > 0 for log scaling"

        self.set(tensor, normalized=normalized)

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def n_c(self):
        """Number of parallel components."""
        return self._n_c

    @property
    def scaling(self):
        """Normalization scaling mode ('linear' or 'log')."""
        return self._scaling

    @min_value.setter
    def min_value(self, value):
        # Bounds follow the owned tensor's device/dtype (see Parameter).
        self._min_value = _match_tensor(
            _broadcast_for_n_c(value, self._n_c), getattr(self, "tensor", None)
        )

    @max_value.setter
    def max_value(self, value):
        self._max_value = _match_tensor(
            _broadcast_for_n_c(value, self._n_c), getattr(self, "tensor", None)
        )

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        anchor = getattr(self, "tensor", None)
        v = _match_tensor(_broadcast_for_n_c(v, self._n_c), anchor)

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _broadcast_for_n_c(min_value, self._n_c)

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _broadcast_for_n_c(max_value, self._n_c)

        min_value = _match_tensor(min_value, anchor)
        max_value = _match_tensor(max_value, anchor)
        self._min_value = min_value
        self._max_value = max_value
        assert (
            torch.allclose(min_value, max_value) == False
        ), "min_value and max_value must be different"

        return normalize_unit(
            v, self._min_value, self._max_value, self._scaling == "log"
        )

    def denormalize(self, v: torch.Tensor):
        return denormalize_unit(
            v, self._min_value, self._max_value, self._scaling == "log"
        )

    def get(self):
        """Get the denormalized value."""
        # Handle the case where this object has been converted during multiprocessing
        # (when _min_value and _max_value are not available)
        if hasattr(self, "_min_value") and hasattr(self, "_max_value"):
            return self.tensor
        else:
            # Fallback for objects that don't have the custom attributes
            return self.tensor

    def set(self, value, normalized: bool = True):
        """Set the parameter value (will be normalized internally)."""
        value = _broadcast_for_n_c(value, self._n_c)

        if normalized:
            value = self.denormalize(value)
        self.tensor = value

    def expand_to_n_c(self, n_c: int):
        """
        Expand this parameter to support n_c parallel components.

        If the parameter is currently scalar (n_c=1), it will be broadcast to shape (n_c,).
        If already has n_c components, this is a no-op.

        Args:
            n_c: Number of parallel components to expand to.

        Returns:
            A new TensorParameter with n_c components.
        """
        if self._n_c == n_c:
            return self
        if self._n_c != 1:
            raise ValueError(
                f"Cannot expand parameter with n_c={self._n_c} to n_c={n_c}"
            )

        # Get current value and expand to shape (n_c,)
        current_value = self.tensor
        if current_value.dim() == 1 and current_value.shape[0] == 1:
            current_value = current_value.expand(n_c).clone()
        elif current_value.dim() == 0:
            current_value = current_value.expand(n_c).clone()

        # Expand min/max values to shape (n_c,)
        min_val = self._min_value
        max_val = self._max_value
        if min_val is not None:
            if min_val.dim() == 1 and min_val.shape[0] == 1:
                min_val = min_val.expand(n_c).clone()
            elif min_val.dim() == 0:
                min_val = min_val.expand(n_c).clone()
        if max_val is not None:
            if max_val.dim() == 1 and max_val.shape[0] == 1:
                max_val = max_val.expand(n_c).clone()
            elif max_val.dim() == 0:
                max_val = max_val.expand(n_c).clone()

        # n_c is inferred from current_value shape in constructor
        return TensorParameter(
            current_value,
            min_value=min_val,
            max_value=max_val,
            normalized=False,  # Value is already denormalized
            scaling=self._scaling,
        )


def _expand_to_3D_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    """
    Convert a Scalar, float, int, or torch.Tensor to torch.Tensor with shape (batch_size, 1)

    3 cases of tensor dimensions are handled:
    1. 2D tensor -> do nothing
    2. 1D tensor -> reshape to batch dimension of 1 (1, n)
    3. 0D tensor -> reshape to batch dimension of 1 (1, 1)

    """
    if isinstance(v, list):
        v = torch.Tensor(v)

    if isinstance(v, Scalar):
        v = v.get()
    elif isinstance(v, (float, int)):
        v = torch.tensor([[[v]]], dtype=float_dtype())
    elif isinstance(v, torch.Tensor):
        assert v.dim() == 3, f"Value must have 3 dimensions, got {v.dim()} dimensions"
        # if v.dim() == 2:
        #     v = v.reshape((1, v.shape[0], v.shape[1]))
        # elif v.dim() == 1:
        #     v = v.reshape((1, 1, v.shape[0]))
        # elif v.dim() == 0:
        #     v = v.reshape((1, 1, 1))
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    return v


def _convert_to_2D_scalar_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    """
    Convert a Scalar, float, int, or torch.Tensor to torch.Tensor with shape (batch_size, 1)

    Interpret
    """
    if isinstance(v, Scalar):
        v = _convert_to_2D_scalar_tensor(v.get())
    elif isinstance(v, (float, int)):
        v = torch.tensor([[v]], dtype=float_dtype())
    elif isinstance(v, torch.Tensor):
        assert (
            v.dim() == 2
        ), f"Value must have less that or equal to 2 dimensions, got {v.dim()} dimensions"
        # if v.dim() == 1:
        #     assert v.shape[0] == 1, f"Value must be a single value, got {v.shape[0]} values and shape {v.shape}"
        #     v = v.reshape((1, 1))
        # elif v.dim() == 0:
        #     v = v.reshape((1, 1))

    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    return v


def _expand_to_2D_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    """
    Convert a Scalar, float, int, or torch.Tensor to torch.Tensor with shape (batch_size, n_timesteps)

    3 cases of tensor dimensions are handled:
    1. 2D tensor -> do nothing
    2. 1D tensor -> reshape to batch dimension of 1 (1, n)
    3. 0D tensor -> reshape to batch dimension of 1 (1, 1)

    """
    if isinstance(v, (list, np.ndarray)):
        v = torch.Tensor(v)

    if isinstance(v, Scalar):
        v = _expand_to_2D_tensor(v.get())
    elif isinstance(v, (float, int)):
        v = torch.tensor([[v]], dtype=float_dtype())
    elif isinstance(v, torch.Tensor):
        assert (
            v.dim() <= 2
        ), f"Value must have less that or equal to 2 dimensions, got {v.dim()} dimensions"
        if v.dim() == 1:
            v = v.reshape((1, v.shape[0]))
        elif v.dim() == 0:
            v = v.reshape((1, 1))
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    return v


def _expand_to_2D_scalar_tensor(
    v: Union[Scalar, float, int, torch.Tensor], n_s: int, n_c: int
):
    """
    Convert a Scalar, float, int, or torch.Tensor to torch.Tensor with shape (n_s, n_c).

    Args:
        v: Input value to convert.
           - Scalar: extracts tensor via .get()
           - float/int: broadcasts to full (n_s, n_c) tensor
           - Tensor: must be exactly (n_s, n_c) or broadcastable from (1, n_c) or (n_s, 1)
        n_s: Number of samples/scenarios.
        n_c: Number of parallel components.

    Returns:
        torch.Tensor with shape (n_s, n_c).

    Raises:
        ValueError: If tensor shape is incompatible with (n_s, n_c).
    """
    if isinstance(v, Scalar):
        v = v.get()  # Already (n_s, n_c)
    elif isinstance(v, (float, int)):
        v = torch.full((n_s, n_c), v, dtype=float_dtype())
    elif isinstance(v, torch.Tensor):
        # Handle scalar tensor
        if v.dim() == 0:
            v = v.expand(n_s, n_c).clone()
        elif v.dim() == 2:
            if v.shape == (n_s, n_c):
                pass  # Already correct shape
            elif v.shape == (1, n_c):
                v = v.expand(n_s, n_c).clone()
            elif v.shape == (n_s, 1):
                v = v.expand(n_s, n_c).clone()
            else:
                raise ValueError(
                    f"Tensor shape {v.shape} incompatible with target ({n_s}, {n_c}). "
                    f"Ensure connected components have matching n_c dimensions."
                )
        else:
            raise ValueError(
                f"Expected 0D or 2D tensor, got {v.dim()}D with shape {v.shape}. "
                f"Target shape is ({n_s}, {n_c})."
            )
    else:
        raise TypeError(f"Unsupported type: {type(v)}")

    return v


def _expand_to_3D_scalar_tensor(
    v: Union[Scalar, float, int, torch.Tensor], n_s: int, n_c: int
):
    """
    Validate a torch.Tensor has shape (n_t, n_s, n_c) for Scalar history (time-first layout).

    Args:
        v: Input value to convert. If Tensor, must have shape (n_t, n_s, n_c).
        n_s: Number of samples/scenarios.
        n_c: Number of parallel components.

    Returns:
        torch.Tensor with shape (n_t, n_s, n_c).
    """
    if isinstance(v, (list, np.ndarray)):
        v = torch.tensor(v, dtype=float_dtype())

    if isinstance(v, torch.Tensor):
        assert (
            v.dim() == 3
        ), f"Expected tensor with 3 dimensions (n_t, n_s, n_c), got {v.dim()}"
        assert (
            v.shape[1] == n_s and v.shape[2] == n_c
        ), f"Expected shape (n_t, {n_s}, {n_c}), got {v.shape}"
    else:
        raise TypeError(f"Unsupported type: {type(v)}")

    return v


def _expand_to_4D_tensor(
    v: Union[Vector, float, int, torch.Tensor], n_s: int, n_c: int
):
    """
    Validate a torch.Tensor has shape (n_t, n_s, n_c, n_v) for Vector history (time-first layout).

    Args:
        v: Input value to convert. If Tensor, must have shape (n_t, n_s, n_c, n_v).
        n_s: Number of samples/scenarios.
        n_c: Number of parallel components.

    Returns:
        torch.Tensor with shape (n_t, n_s, n_c, n_v).
    """
    if isinstance(v, (list, np.ndarray)):
        v = torch.tensor(v, dtype=float_dtype())

    if isinstance(v, torch.Tensor):
        assert (
            v.dim() == 4
        ), f"Expected tensor with 4 dimensions (n_t, n_s, n_c, n_v), got {v.dim()}"
        assert (
            v.shape[1] == n_s and v.shape[2] == n_c
        ), f"Expected shape (n_t, {n_s}, {n_c}, n_v), got {v.shape}"
    else:
        raise TypeError(f"Unsupported type: {type(v)}")

    return v


def _expand_to_1D_scalar_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    if isinstance(v, Scalar):
        v = _expand_to_1D_scalar_tensor(v.get())
    elif isinstance(v, (float, int)):
        v = torch.tensor([v], dtype=float_dtype())
    elif isinstance(v, torch.Tensor):
        assert (
            v.dim() <= 1
        ), f"Value must have less that or equal to 1 dimensions, got {v.dim()} dimensions"
        if v.dim() == 0:
            v = v.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    return v


def _convert_to_1D_scalar_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    if isinstance(v, Scalar):
        v = _convert_to_1D_scalar_tensor(v.get())
    elif isinstance(v, (float, int)):
        v = torch.tensor([v], dtype=float_dtype())
    elif isinstance(v, torch.Tensor):
        assert (
            v.dim() <= 1
        ), f"Value must have less that or equal to 1 dimensions, got {v.dim()} dimensions"
        if v.dim() == 0:
            v = v.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    return v


def _to_float64_tensor(v):
    """Convert input to a float64 tensor."""
    if isinstance(v, (float, int)):
        return torch.tensor(v, dtype=float_dtype())
    elif isinstance(v, torch.Tensor):
        return v.to(dtype=float_dtype())
    elif isinstance(v, np.ndarray):
        return torch.tensor(v, dtype=float_dtype())
    elif isinstance(v, (list, tuple)):
        return torch.tensor(v, dtype=float_dtype())
    else:
        raise TypeError(f"Unsupported type: {type(v)}")


def _prepare_parameter_data(data):
    """
    Prepare parameter data, converting to tensor with shape (n_c,).

    Args:
        data: Input data (scalar, int, float, or tensor)

    Returns:
        Tuple of (prepared_data with shape (n_c,), n_c inferred from data)
    """
    data = _to_float64_tensor(data)

    # Infer n_c from data shape
    if data.dim() == 0:
        n_c = 1
        data = data.unsqueeze(0)  # (1,)
    elif data.dim() == 1:
        n_c = data.shape[0]
    else:
        raise ValueError(f"Data must be 0D or 1D, got {data.dim()}D")

    return data, n_c


def _prepare_bound_value(value, data_shape, n_c):
    """
    Prepare min/max bound value with broadcasting.

    Args:
        value: Bound value (scalar, int, float, or tensor), or None
        data_shape: Shape of the data tensor (should be (n_c,))
        n_c: Number of parallel components

    Returns:
        Prepared bound tensor with shape (n_c,), or None if input was None
    """
    if value is None:
        return None

    value = _to_float64_tensor(value)

    # Always expand to match n_c
    if value.dim() == 0:
        value = value.expand(n_c).clone()
    elif value.dim() == 1 and value.shape[0] == 1 and n_c > 1:
        value = value.expand(n_c).clone()
    elif value.dim() == 1 and value.shape[0] != n_c:
        raise ValueError(f"Bound value shape {value.shape} does not match n_c={n_c}")

    return value


def _prepare_value_for_set(
    v: Union[float, int, torch.Tensor, "Scalar"],
    target_shape: tuple,
    indices: dict = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Prepare a value for setting into a Scalar or Vector tensor.

    This unified function handles all the conversion and broadcasting logic:
    1. Converts Python scalars (int, float) to tensors
    2. Handles Scalar objects by calling .get()
    3. Expands/broadcasts to match the target shape

    Args:
        v: The value to prepare (float, int, tensor, or Scalar)
        target_shape: The full target shape, e.g. (n_s, n_c) for Scalar or (n_s, n_c, n_v) for Vector
        indices: Dict of which indices are being set, e.g. {'i_s': 0, 'i_c': slice(None)} or {'i_c': 0, 'i_v': 1}
                 If an index is slice(None), that dimension is expected in v.
                 If an index is an int, that dimension is not expected in v.

    Returns:
        torch.Tensor with appropriate shape for assignment
    """
    indices = indices or {}

    # Handle Scalar objects
    if hasattr(v, "get") and callable(v.get):
        v = v.get()

    # Convert Python scalars to tensors
    if isinstance(v, (int, float)):
        v = torch.tensor(v, dtype=float_dtype(), device=device)

    if not isinstance(v, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(v)}")

    # Match the target tensor's device (no-op when already there)
    if device is not None and v.device != device:
        v = v.to(device)

    # Determine expected shape based on which indices are set
    # If index is slice(None), we need that dimension in v
    # If index is an int, we don't need that dimension in v
    expected_dims = []
    dim_names = ["n_s", "n_c", "n_v"][: len(target_shape)]
    idx_map = {"n_s": "i_s", "n_c": "i_c", "n_v": "i_v"}

    for dim_name, dim_size in zip(dim_names, target_shape):
        idx_name = idx_map.get(dim_name)
        idx_value = indices.get(idx_name, slice(None))
        # Include dimension if index is slice(None) (selecting all)
        if isinstance(idx_value, slice):
            expected_dims.append(dim_size)

    # Handle 0D tensor (scalar)
    if v.dim() == 0:
        v = v.expand(*expected_dims).clone()
    # Handle dimension mismatch - try to expand
    elif v.dim() < len(expected_dims):
        # Try to unsqueeze missing dimensions
        while v.dim() < len(expected_dims):
            # Add dimension at end if it makes sense
            if v.shape[-1] == expected_dims[v.dim() - 1] if v.dim() > 0 else True:
                v = v.unsqueeze(-1)
            else:
                v = v.unsqueeze(0)
        # Expand to target shape if needed
        if v.shape != tuple(expected_dims):
            try:
                v = v.expand(*expected_dims).clone()
            except RuntimeError:
                pass  # Let it fail later with a clearer error

    return v


def _match_tensor(t: torch.Tensor, anchor) -> torch.Tensor:
    """Move ``t`` to ``anchor``'s device/dtype (no-op when already there, or
    when ``anchor`` is not a tensor -- e.g. during ``TensorParameter.__init__``
    before the data tensor exists)."""
    if not isinstance(anchor, torch.Tensor):
        return t
    return t.to(device=anchor.device, dtype=anchor.dtype)


def _broadcast_for_n_c(value, n_c):
    """
    Broadcast a value to match n_c if needed.

    Args:
        value: Input value (scalar, int, float, or tensor)
        n_c: Number of parallel components

    Returns:
        Broadcasted tensor with shape (n_c,)
    """
    value = _to_float64_tensor(value)

    if value.dim() == 0:
        # Always expand to 1D with shape (n_c,)
        value = value.expand(n_c).clone()
    elif value.shape[0] == 1 and n_c > 1:
        # Expand from (1,) to (n_c,)
        value = value.expand(n_c).clone()

    return value


# Add get() method to nn.Parameter for compatibility
if not hasattr(torch.nn.Parameter, "get"):

    def parameter_get(self):
        """Get the parameter value (fallback for regular nn.Parameter objects)."""
        return self

    torch.nn.Parameter.get = parameter_get


# Our own rebuild functions for tps.Parameter
def _rebuild_tps_parameter(data, requires_grad, backward_hooks):
    """Rebuild a tps.Parameter instance (equivalent to torch._utils._rebuild_parameter)."""
    param = Parameter(data, requires_grad=requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks
    return param


def _rebuild_tps_parameter_with_state(data, requires_grad, backward_hooks, state):
    """Rebuild a tps.Parameter instance with state (equivalent to torch._utils._rebuild_parameter_with_state)."""
    param = Parameter(data, requires_grad=requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    # Restore state on Parameter like python attr.
    param = _set_tps_obj_state(param, state)
    return param


def _get_tps_obj_state(obj):
    """Get the state of a tps.Parameter object (equivalent to torch._utils._get_obj_state)."""
    # Get the state of the python subclass
    # This loosely mimicks the function on the object class but since Tensor do not inherit
    # from it, we cannot call that function directly
    getstate_fn = getattr(obj, "__getstate__", None)
    if getstate_fn:
        state = getstate_fn()
    else:
        # Standard library imports
        import copyreg

        slots_to_save = copyreg._slotnames(obj.__class__)  # type: ignore[attr-defined]
        if slots_to_save:
            state = (
                obj.__dict__,
                {
                    name: getattr(obj, name)
                    for name in slots_to_save
                    if hasattr(obj, name)
                },
            )
        else:
            state = obj.__dict__
    return state


def _set_tps_obj_state(obj, state):
    """Set the state on a tps.Parameter object (equivalent to torch._utils._set_obj_state)."""
    if isinstance(state, dict):
        obj.__dict__.update(state)
    elif isinstance(state, tuple):
        obj.__dict__, slots = state
        for name, value in slots.items():
            setattr(obj, name, value)
    return obj
