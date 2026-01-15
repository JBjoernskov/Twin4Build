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
        tensor: Optional[torch.Tensor] = None,
        n_s: int = 1,
        n_c: int = 1,
        n_v: Optional[int] = None,
        n_t: Optional[int] = None,
        log_history: bool = True,
        is_leaf: bool = False,
        do_normalization: bool = False,
        optional: bool = False,
    ) -> None:
        """
        Initialize a Vector instance.

        Args:
            tensor (Optional[torch.Tensor]): Initial tensor value.
            n_s (int): Number of simulations. Defaults to 1.
            n_c (int): Number of parallel components. Defaults to 1.
            n_v (Optional[int]): The size of the vector.
            n_t (Optional[int]): Number of timesteps.
            log_history (bool): Whether to log history. Defaults to True.
            is_leaf (bool): Whether this vector is a leaf node. Defaults to False.
            do_normalization (bool): Whether to normalize history. Defaults to False.
            optional (bool): Whether this vector is optional. Defaults to False.
        """
        self._tensor = None
        self._n_s = n_s
        self._n_c = n_c
        self._n_v = n_v
        self._n_t = n_t
        self._log_history = log_history
        self._is_leaf = is_leaf
        self._do_normalization = do_normalization
        self._optional = optional

        self._init_tensor = tensor.clone() if tensor is not None else None

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

    @property
    def history(self):
        assert (
            self._history_is_populated
        ), "History is not populated. Set log_history to True to populate history."
        return self._history

    @property
    def normalized_history(self):
        return self._normalized_history

    @property
    def is_leaf(self):
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, value: bool):
        assert isinstance(value, bool), "is_leaf must be a boolean"
        if self._is_leaf:
            raise ("Leaf Vectors are currently not supported.")
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

    def make_pickable(self):
        if self.tensor is not None:
            if self.n_v > 0:
                self.tensor = torch.tensor(
                    [self.tensor.item()], dtype=torch.float64, requires_grad=False
                )
            else:
                self.tensor = torch.tensor([], dtype=torch.float64, requires_grad=False)

        if self._init_tensor is not None:
            self._init_tensor = torch.tensor(
                self._init_tensor.item(), dtype=torch.float64, requires_grad=False
            )

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
        n_t: int,
        n_v: Optional[int] = None,
        n_s: Optional[int] = None,
        n_c: Optional[int] = None,
        batch_size: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        size: Optional[int] = None,
        values: Optional[List[float]] = None,
        force: bool = False,
    ) -> None:
        """Initialize the vector tensor and sorting indices.

        Creates the underlying torch tensor with shape (n_s, n_c, n_v).
        History has shape (n_s, n_c, n_t, n_v).
        
        Args:
            n_t (int): Number of timesteps.
            n_v (Optional[int]): Size of the vector.
            n_s (Optional[int]): Number of simulations.
            n_c (Optional[int]): Number of parallel components.
            batch_size (Optional[int]): Backward compatible batch_size (sets n_s, keeps n_c=1).
            n_timesteps (Optional[int]): Backward compatible alias for n_t.
            size (Optional[int]): Backward compatible alias for n_v.
            values (Optional[List[float]]): Initial values for leaf vectors.
            force (bool): Force reinitialization.
        """
        # Handle backward compatibility aliases
        if n_timesteps is not None and n_t is None:
            n_t = n_timesteps
        if size is not None and n_v is None:
            n_v = size
            
        assert isinstance(n_t, int), "n_t must be an integer"
        if n_t is not None:
            self._n_t = n_t
        if n_v is not None:
            self._n_v = n_v
        if n_s is not None:
            self._n_s = n_s
        if n_c is not None:
            self._n_c = n_c
        # Backward compatibility: if batch_size is provided, use it as n_s
        if batch_size is not None and n_s is None:
            self._n_s = batch_size

        # Create tensor with shape (n_s, n_c, n_v)
        if self._init_tensor is None:
            self._tensor = torch.zeros(
                (self.n_s, self.n_c, self.n_v), dtype=torch.float64
            )
        else:
            self._tensor = self._init_tensor.clone()

        if values is not None:
            values = _expand_to_4D_tensor(values, self.n_s, self.n_c)

        # We return early if this scalar has requires_grad=True.
        # This is the case when used in the optimizer.
        # Here we dont want to reinitialize the history as the torch.optim.Optimizer changes this in-place.
        if (
            self._initialized
            and self._requires_reinittialization == False
            and force == False
        ):
            return

        if self._is_leaf:
            assert values is not None, "Values must be provided for leaf vectors"
            assert (
                values.shape[0] == self.n_s
            ), f"Values first dim ({values.shape[0]}) must match n_s ({self.n_s})"
            assert (
                values.shape[1] == self.n_c
            ), f"Values second dim ({values.shape[1]}) must match n_c ({self.n_c})"
            assert (
                values.shape[2] == self.n_t
            ), f"Values third dim ({values.shape[2]}) must match n_t ({self.n_t})"
            assert (
                values.shape[3] == self.n_v
            ), f"Values fourth dim ({values.shape[3]}) must match n_v ({self.n_v})"
            # Pre-allocate the history tensor with the correct size
            self._history = values
            self._history_is_populated = True
            if self._do_normalization:
                self._normalized_history = self.normalize()

        else:
            # History shape: (n_s, n_c, n_t, n_v)
            self._history = torch.zeros(
                (self.n_s, self.n_c, self.n_t, self.n_v),
                dtype=torch.float64,
                requires_grad=False,
            )
            self._history_is_populated = False

        self._initialized = True
        return self

    def set(
        self,
        v: Union[float, torch.Tensor],
        step_index: Optional[int] = None,
        index: Optional[int, torch.Tensor] = None,
    ) -> None:
        """Set the next value in the vector.

        Args:
            v (Union[float, torch.Tensor]): Value to set. Expected shape (n_s, n_c, n_v) or (n_s, n_c).
            step_index (Optional[int]): Step index to set value at.
            index (Optional[int, torch.Tensor]): Index within the vector to set value at.
        """
        if index is not None:
            self._tensor[:, :, index] = v
        else:
            self._tensor[:, :, :] = v

        if self._log_history:
            assert (
                step_index is not None
            ), "step_index must be provided when logging history"
            if self.is_leaf == False or (self.is_leaf and self._do_normalization):
                if v.dim() == 3:
                    # v has shape (n_s, n_c, n_v)
                    self._history[:, :, step_index, :] = v
                elif v.dim() == 2:
                    # v has shape (n_s, n_c) for a specific index
                    self._history[:, :, step_index, index] = v
                else:
                    raise ValueError(f"Unsupported dimension: {v.dim()}, expected 2 or 3")

            if step_index == self._history.shape[2] - 1:
                self._history_is_populated = True
            else:
                self._history_is_populated = False

    def get(self, index: Optional[int, torch.Tensor] = None) -> torch.Tensor:
        """Get vector values.

        Args:
            index (Optional[int, torch.Tensor]): Index within the vector to get.

        Returns:
            torch.Tensor: Tensor of values with shape (n_s, n_c, n_v) or (n_s, n_c) if index specified.
        """
        if index is not None:
            out = self.tensor[:, :, index]
        else:
            out = self.tensor
        return out

    def copy(self):
        """Create a copy of the vector.

        Returns:
            Vector: A new Vector instance with the same data.
        """
        tensor = self.tensor.clone() if self.tensor is not None else None
        copy = Vector(
            tensor=tensor,
            n_s=self.n_s,
            n_c=self.n_c,
            n_v=self.n_v,
            n_t=self.n_t,
            log_history=self.log_history,
            is_leaf=self.is_leaf,
            do_normalization=self.do_normalization,
            optional=self.optional,
        )
        return copy


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
        tensor: Optional[Union[float, int, torch.Tensor]] = None,
        n_s: int = 1,
        n_c: int = 1,
        n_t: Optional[int] = None,
        log_history: bool = True,
        is_leaf: bool = False,
        do_normalization: bool = False,
        optional: bool = False,
    ) -> None:
        """
        Initialize a Scalar instance.

        Args:
            tensor (Optional[Union[float, int, torch.Tensor]]): Initial tensor value.
            n_s (int): Number of simulations. Defaults to 1.
            n_c (int): Number of parallel components. Defaults to 1.
            n_t (Optional[int]): Number of timesteps.
            log_history (bool): Whether to log history. Defaults to True.
            is_leaf (bool): Whether this scalar is a leaf node. Defaults to False.
            do_normalization (bool): Whether to normalize history. Defaults to False.
            optional (bool): Whether this scalar is optional. Defaults to False.
        """
        assert isinstance(
            tensor, (float, int, torch.Tensor, type(None))
        ), "tensor must be a float, int, torch.Tensor, or None"

        if isinstance(tensor, torch.Tensor):
            assert (
                tensor.numel() == 1
            ), f"tensor must be a single value, got {tensor.numel()} values"
            assert (
                tensor.dim() == 0 or tensor.dim() == 1
            ), f"tensor must have 0 or 1 dimensions, got {tensor.dim()} dimensions"
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            tensor.requires_grad = False

        elif isinstance(tensor, (float, int)):
            tensor = torch.tensor([tensor], dtype=torch.float64, requires_grad=False)

        self._tensor = tensor
        self._n_s = n_s
        self._n_c = n_c
        self._n_t = n_t
        self._init_tensor = tensor
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

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value

    @property
    def scalar(self):
        warnings.warn(
            "Property 'scalar' is deprecated. Use 'tensor' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._tensor

    @property
    def history(self):
        assert (
            self._history_is_populated
        ), "History is not populated. Set log_history to True to populate history."
        return self._history

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

    def set_requires_grad(self, requires_grad: bool):  # TODO: Implement this for Vector
        assert self._is_leaf or (
            self._is_leaf == False and requires_grad == False
        ), "Only leaf scalars can have their requires_grad attribute set to True"
        if self._do_normalization:
            self._normalized_history.requires_grad = requires_grad
        else:
            self._history.requires_grad = requires_grad
        self._requires_reinittialization = not requires_grad

    def initialize(
        self,
        n_t: int,
        n_s: Optional[int] = None,
        n_c: Optional[int] = None,
        batch_size: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        values: Optional[List[float]] = None,
        force: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the scalar.

        Creates the underlying torch tensor with shape (n_s, n_c).
        History has shape (n_s, n_c, n_t).

        Args:
            n_t (int): The number of timesteps.
            n_s (Optional[int]): Number of simulations.
            n_c (Optional[int]): Number of parallel components.
            batch_size (Optional[int]): Backward compatible batch_size (sets n_s, keeps n_c=1).
            n_timesteps (Optional[int]): Backward compatible alias for n_t.
            values (Optional[List[float]]): The values to initialize the scalar with.
            force (bool): Whether to force the initialization.
        """
        # Handle backward compatibility alias
        if n_timesteps is not None and n_t is None:
            n_t = n_timesteps
            
        assert isinstance(n_t, int), "n_t must be an integer"

        if n_s is not None:
            self._n_s = n_s
        if n_c is not None:
            self._n_c = n_c
        # Backward compatibility: if batch_size is provided, use it as n_s
        if batch_size is not None and n_s is None:
            self._n_s = batch_size

        if n_t is not None:
            self._n_t = n_t

        # Create tensor with shape (n_s, n_c)
        if self._init_tensor is None:
            self._tensor = torch.zeros((self.n_s, self.n_c), dtype=torch.float64)
        else:
            self._tensor = self._init_tensor.clone()
            # Expand to (n_s, n_c) if needed
            if self._tensor.dim() == 0:
                self._tensor = self._tensor.expand(self.n_s, self.n_c).clone()
            elif self._tensor.dim() == 1 and self._tensor.shape[0] == 1:
                self._tensor = self._tensor.expand(self.n_s, self.n_c).clone()
            elif self._tensor.dim() == 1:
                # Assume it's (n_s,) and expand to (n_s, n_c)
                self._tensor = self._tensor.unsqueeze(1).expand(self.n_s, self.n_c).clone()

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
            return

        if self._is_leaf:
            assert values is not None, "Values must be provided for leaf scalars"
            assert (
                values.shape[0] == self.n_s
            ), f"First dimension of values ({values.shape[0]}) must match n_s ({self.n_s})."
            assert (
                values.shape[1] == self.n_c
            ), f"Second dimension of values ({values.shape[1]}) must match n_c ({self.n_c})."
            assert (
                values.shape[2] == self.n_t
            ), f"Third dimension of values ({values.shape[2]}) must match n_t ({self.n_t})."
            # Pre-allocate the history tensor with the correct size
            self._history = values
            self._history_is_populated = True
            if self._do_normalization:
                self._normalized_history = self.normalize()

        else:
            # History shape: (n_s, n_c, n_t)
            self._history = torch.zeros(
                self.n_s,
                self.n_c,
                self.n_t,
                dtype=torch.float64,
                requires_grad=False,
            )
            self._history_is_populated = False

        self._initialized = True

    def set(
        self,
        v: Union[Scalar, float, int, torch.Tensor] = None,
        step_index: Optional[int] = None,
        apply: callable = None,
        *args,
        **kwargs,
    ) -> None:
        """Set the scalar value.

        Args:
            v (Union[Scalar, float, torch.Tensor]): Value to set with shape (n_s, n_c).
            step_index (Optional[int]): Step index for history logging.
            apply (callable): Optional function to apply to the value.
        """
        if self._is_leaf:
            assert (
                v is None
            ), "Values cannot be set for leaf scalars. Use scalar.set(step_index=step_index) to set value based on history"
            assert (
                step_index is not None
            ), "step_index must be provided for leaf scalars"
            if self._do_normalization:
                v = self._normalized_history[:, :, step_index]
                v = self.denormalize(v)
            else:
                v = self._history[:, :, step_index]
        else:
            v = _expand_to_2D_scalar_tensor(v, self.n_s, self.n_c)

        if apply is not None:
            v = apply(v)

        self._tensor = v
        if self._log_history:
            assert (
                step_index is not None
            ), "step_index must be provided when logging history"
            if self.is_leaf == False or (self.is_leaf and self._do_normalization):
                self._history[:, :, step_index] = v

            if step_index == self._history.shape[2] - 1:
                self._history_is_populated = True

    def get(self, *args, **kwargs) -> torch.Tensor:
        """Get the scalar value.

        Returns:
            torch.Tensor: Scalar value with shape (n_s, n_c).
        """
        return self._tensor

    def normalize(self, v: torch.Tensor = None):
        assert (
            self._history_is_populated == True
        ), "History must be populated before normalizing"
        if v is None:
            v = self._history
        v = _expand_to_3D_scalar_tensor(v, self.n_s, self.n_c)
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
        min_val = torch.tensor(self._min_history, dtype=torch.float64)
        max_val = torch.tensor(self._max_history, dtype=torch.float64)

        if torch.allclose(min_val, max_val):
            min_val = torch.tensor(0, dtype=torch.float64)
            if torch.allclose(max_val, torch.tensor(0, dtype=torch.float64)):
                max_val = torch.tensor(1, dtype=torch.float64)
            else:
                max_val = torch.tensor(1, dtype=torch.float64)

        self._is_normalized = True
        return (v - min_val) / (max_val - min_val)

    def denormalize(self, v: torch.Tensor):
        assert (
            self._is_normalized == True
        ), ".normalize() must be called before denormalizing"
        # Use cached float values and convert to tensors
        min_val = torch.tensor(self._min_history, dtype=torch.float64)
        max_val = torch.tensor(self._max_history, dtype=torch.float64)
        return v * (max_val - min_val) + min_val

    def get_float(self) -> float:
        """Get the scalar value as a float.

        Returns:
            float: Scalar value.
        """
        return self._tensor.item()

    def copy(self):
        copy = Scalar()
        copy._tensor = self._tensor
        copy._init_tensor = self._init_tensor
        copy._n_s = self._n_s
        copy._n_c = self._n_c
        copy._n_t = self._n_t
        if self._history is None:
            copy._history = None
        else:
            copy._history = self._history.clone()
        copy._log_history = self._log_history
        copy._is_leaf = self._is_leaf
        return copy


class Parameter(nn.Parameter):
    """
    A custom nn.Parameter implementation that normalizes the data between 0 and 1 to stabilize gradients in physical systems where the parameters scales can be different.
    This makes it possible to use torch.optim.Optimizer to optimize the parameters.
    
    Supports an optional `n_c` dimension for parallel components, allowing the same
    parameter to have different values for multiple parallel instances.
    
    Args:
        data: The parameter value(s). Can be a scalar or 1D tensor of shape (n_c,).
        min_value: Minimum value for normalization. Can be scalar (broadcast) or per-component.
        max_value: Maximum value for normalization. Can be scalar (broadcast) or per-component.
        requires_grad: Whether to track gradients for this parameter.
        n_c: Number of parallel components. If None, inferred from data shape.
             If data is scalar and n_c > 1, the scalar is broadcast to all components.
    """

    def __new__(cls, data, min_value=None, max_value=None, requires_grad=True, n_c=None):
        # Prepare data with n_c handling
        data, n_c = _prepare_parameter_data(data, n_c)
        
        # Set min and max values with defaults
        if min_value is None:
            if torch.all(data < 0):
                min_value = data.detach().clone()
            else:
                min_value = torch.tensor(0, dtype=torch.float64)
        else:
            min_value = _prepare_bound_value(min_value, data.shape, n_c)

        if max_value is None:
            if torch.all(data < 0):
                max_value = torch.tensor(0, dtype=torch.float64)
            elif torch.allclose(data, torch.zeros_like(data)):
                max_value = torch.tensor(1, dtype=torch.float64)
            else:
                max_value = data.detach().clone()
        else:
            max_value = _prepare_bound_value(max_value, data.shape, n_c)

        assert torch.all(
            max_value > min_value
        ), "max_value must be greater than min_value"

        # Normalize the data
        normalized_data = (data - min_value) / (max_value - min_value)

        # Create the parameter using the parent's __new__ method
        instance = super().__new__(cls, normalized_data, requires_grad)

        # Store min and max values as properties
        instance._min_value = min_value
        instance._max_value = max_value
        instance._n_c = n_c

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

    @min_value.setter
    def min_value(self, value):
        self._min_value = _broadcast_for_n_c(value, self._n_c)

    @max_value.setter
    def max_value(self, value):
        self._max_value = _broadcast_for_n_c(value, self._n_c)

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        v = _broadcast_for_n_c(v, self._n_c)

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _broadcast_for_n_c(min_value, self._n_c)

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _broadcast_for_n_c(max_value, self._n_c)

        self._min_value = min_value
        self._max_value = max_value
        assert (
            torch.allclose(min_value, max_value) == False
        ), "min_value and max_value must be different"
        return (v - self._min_value) / (self._max_value - self._min_value)

    def denormalize(self, v: torch.Tensor):
        return v * (self._max_value - self._min_value) + self._min_value

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
        
        If the parameter is currently scalar, it will be broadcast to a tensor of shape (n_c,).
        If already has n_c components, this is a no-op.
        
        Args:
            n_c: Number of parallel components to expand to.
            
        Returns:
            A new Parameter with n_c components.
        """
        if self._n_c == n_c:
            return self
        if self._n_c != 1:
            raise ValueError(f"Cannot expand parameter with n_c={self._n_c} to n_c={n_c}")
        
        # Get denormalized value and expand
        denorm_value = self.get()
        if denorm_value.dim() == 0:
            denorm_value = denorm_value.expand(n_c).clone()
        
        # Expand min/max values
        min_val = self._min_value
        max_val = self._max_value
        if min_val.dim() == 0:
            min_val = min_val.expand(n_c).clone()
        if max_val.dim() == 0:
            max_val = max_val.expand(n_c).clone()
        
        return Parameter(
            denorm_value,
            min_value=min_val,
            max_value=max_val,
            requires_grad=self.requires_grad,
            n_c=n_c
        )


class TensorParameter:
    """
    A custom nn.Parameter implementation that normalizes the data between 0 and 1 to stabilize gradients in physical systems where the parameters scales can be different.

    This class is used to represent model parameters as a Tensor when we calculate the Jacobian analytically as the jac = torch.nn.functional.Jacobian() has the signature jac(f: callable, input: Tensor) -> Tensor.
    
    Supports an optional `n_c` dimension for parallel components, allowing the same
    parameter to have different values for multiple parallel instances.
    
    Args:
        tensor: The parameter value(s). Can be a scalar or 1D tensor of shape (n_c,).
        min_value: Minimum value for normalization. Can be scalar (broadcast) or per-component.
        max_value: Maximum value for normalization. Can be scalar (broadcast) or per-component.
        normalized: Whether the input tensor is already normalized.
        n_c: Number of parallel components. If None, inferred from tensor shape.
             If tensor is scalar and n_c > 1, the scalar is broadcast to all components.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        min_value=None,
        max_value=None,
        normalized: bool = True,
        n_c: int = None,
    ):
        # Prepare tensor with n_c handling
        tensor, n_c = _prepare_parameter_data(tensor, n_c)
        self._n_c = n_c
        
        # Process min/max values with broadcasting
        self._min_value = _prepare_bound_value(min_value, tensor.shape, n_c)
        self._max_value = _prepare_bound_value(max_value, tensor.shape, n_c)

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

    @min_value.setter
    def min_value(self, value):
        self._min_value = _broadcast_for_n_c(value, self._n_c)

    @max_value.setter
    def max_value(self, value):
        self._max_value = _broadcast_for_n_c(value, self._n_c)

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        v = _broadcast_for_n_c(v, self._n_c)

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _broadcast_for_n_c(min_value, self._n_c)

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _broadcast_for_n_c(max_value, self._n_c)

        self._min_value = min_value
        self._max_value = max_value
        assert (
            torch.allclose(min_value, max_value) == False
        ), "min_value and max_value must be different"
        return (v - self._min_value) / (self._max_value - self._min_value)

    def denormalize(self, v: torch.Tensor):
        return v * (self._max_value - self._min_value) + self._min_value

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
        
        If the parameter is currently scalar, it will be broadcast to a tensor of shape (n_c,).
        If already has n_c components, this is a no-op.
        
        Args:
            n_c: Number of parallel components to expand to.
            
        Returns:
            A new TensorParameter with n_c components.
        """
        if self._n_c == n_c:
            return self
        if self._n_c != 1:
            raise ValueError(f"Cannot expand parameter with n_c={self._n_c} to n_c={n_c}")
        
        # Get current value and expand
        current_value = self.tensor
        if current_value.dim() == 0:
            current_value = current_value.expand(n_c).clone()
        
        # Expand min/max values if they exist
        min_val = self._min_value
        max_val = self._max_value
        if min_val is not None and min_val.dim() == 0:
            min_val = min_val.expand(n_c).clone()
        if max_val is not None and max_val.dim() == 0:
            max_val = max_val.expand(n_c).clone()
        
        return TensorParameter(
            current_value,
            min_value=min_val,
            max_value=max_val,
            normalized=False,  # Value is already denormalized
            n_c=n_c
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
        v = torch.tensor([[[v]]], dtype=torch.float64)
    elif isinstance(v, torch.Tensor):
        assert (
            v.dim() == 3
        ), f"Value must have 3 dimensions, got {v.dim()} dimensions"
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
        v = torch.tensor([[v]], dtype=torch.float64)
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
        v = torch.tensor([[v]], dtype=torch.float64)
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


def _expand_to_2D_scalar_tensor(v: Union[Scalar, float, int, torch.Tensor], n_s: int, n_c: int):
    """
    Convert a Scalar, float, int, or torch.Tensor to torch.Tensor with shape (n_s, n_c).
    
    Args:
        v: Input value to convert.
        n_s: Number of samples/scenarios.
        n_c: Number of parallel components.
        
    Returns:
        torch.Tensor with shape (n_s, n_c).
    """
    if isinstance(v, (list, np.ndarray)):
        v = torch.tensor(v, dtype=torch.float64)
    
    if isinstance(v, Scalar):
        v = v.get()
    elif isinstance(v, (float, int)):
        v = torch.tensor(v, dtype=torch.float64)
    
    if isinstance(v, torch.Tensor):
        if v.dim() == 0:
            v = v.expand(n_s, n_c).clone()
        elif v.dim() == 1:
            # Assume it's (n_s,) or (n_c,) - try to broadcast
            if v.shape[0] == n_s:
                v = v.unsqueeze(1).expand(n_s, n_c).clone()
            elif v.shape[0] == n_c:
                v = v.unsqueeze(0).expand(n_s, n_c).clone()
            elif v.shape[0] == 1:
                v = v.expand(n_s, n_c).clone()
            else:
                raise ValueError(f"Cannot broadcast tensor of shape {v.shape} to ({n_s}, {n_c})")
        elif v.dim() == 2:
            if v.shape != (n_s, n_c):
                # Try broadcasting
                if v.shape[0] == 1:
                    v = v.expand(n_s, -1)
                if v.shape[1] == 1:
                    v = v.expand(-1, n_c)
                v = v.clone()
        else:
            raise ValueError(f"Expected tensor with <= 2 dimensions, got {v.dim()}")
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    
    return v


def _expand_to_3D_scalar_tensor(v: Union[Scalar, float, int, torch.Tensor], n_s: int, n_c: int):
    """
    Convert a Scalar, float, int, or torch.Tensor to torch.Tensor with shape (n_s, n_c, n_timesteps).
    
    Used for Scalar history tensors.
    
    Args:
        v: Input value to convert.
        n_s: Number of samples/scenarios.
        n_c: Number of parallel components.
        
    Returns:
        torch.Tensor with shape (n_s, n_c, n_timesteps).
    """
    if isinstance(v, (list, np.ndarray)):
        v = torch.tensor(v, dtype=torch.float64)
    
    if isinstance(v, Scalar):
        v = v.get()
    elif isinstance(v, (float, int)):
        v = torch.tensor([[[v]]], dtype=torch.float64)
    
    if isinstance(v, torch.Tensor):
        if v.dim() == 0:
            v = v.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif v.dim() == 1:
            # Assume it's (n_timesteps,)
            v = v.unsqueeze(0).unsqueeze(0).expand(n_s, n_c, -1).clone()
        elif v.dim() == 2:
            # Assume it's (n_s, n_timesteps) - old format, add n_c dimension
            v = v.unsqueeze(1).expand(-1, n_c, -1).clone()
        elif v.dim() == 3:
            # Already in correct format (n_s, n_c, n_timesteps)
            pass
        else:
            raise ValueError(f"Expected tensor with <= 3 dimensions, got {v.dim()}")
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    
    return v


def _expand_to_4D_tensor(v: Union[Vector, float, int, torch.Tensor], n_s: int, n_c: int):
    """
    Convert a Vector, float, int, or torch.Tensor to torch.Tensor with shape (n_s, n_c, n_timesteps, size).
    
    Used for Vector history tensors.
    
    Args:
        v: Input value to convert.
        n_s: Number of samples/scenarios.
        n_c: Number of parallel components.
        
    Returns:
        torch.Tensor with shape (n_s, n_c, n_timesteps, size).
    """
    if isinstance(v, (list, np.ndarray)):
        v = torch.tensor(v, dtype=torch.float64)
    
    if isinstance(v, Vector):
        v = v.get()
    elif isinstance(v, (float, int)):
        v = torch.tensor([[[[v]]]], dtype=torch.float64)
    
    if isinstance(v, torch.Tensor):
        if v.dim() == 0:
            v = v.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif v.dim() == 1:
            # Assume it's (size,)
            v = v.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n_s, n_c, 1, -1).clone()
        elif v.dim() == 2:
            # Assume it's (n_timesteps, size)
            v = v.unsqueeze(0).unsqueeze(0).expand(n_s, n_c, -1, -1).clone()
        elif v.dim() == 3:
            # Assume it's (n_s, n_timesteps, size) - old format, add n_c dimension
            v = v.unsqueeze(1).expand(-1, n_c, -1, -1).clone()
        elif v.dim() == 4:
            # Already in correct format (n_s, n_c, n_timesteps, size)
            pass
        else:
            raise ValueError(f"Expected tensor with <= 4 dimensions, got {v.dim()}")
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    
    return v


def _expand_to_1D_scalar_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    if isinstance(v, Scalar):
        v = _expand_to_1D_scalar_tensor(v.get())
    elif isinstance(v, (float, int)):
        v = torch.tensor([v], dtype=torch.float64)
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
        v = torch.tensor([v], dtype=torch.float64)
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
        return torch.tensor(v, dtype=torch.float64)
    elif isinstance(v, torch.Tensor):
        return v.to(dtype=torch.float64)
    else:
        raise TypeError(f"Unsupported type: {type(v)}")


def _prepare_parameter_data(data, n_c):
    """
    Prepare parameter data with n_c dimension handling.
    
    Args:
        data: Input data (scalar, int, float, or tensor)
        n_c: Number of parallel components (None to infer from data)
        
    Returns:
        Tuple of (prepared_data, n_c)
    """
    data = _to_float64_tensor(data)
    
    if n_c is not None and n_c > 1:
        # If data is scalar and n_c > 1, broadcast to all components
        if data.dim() == 0 or (data.dim() == 1 and data.shape[0] == 1):
            data = data.expand(n_c).clone()
        elif data.dim() == 1 and data.shape[0] != n_c:
            raise ValueError(f"Data shape {data.shape} does not match n_c={n_c}")
    else:
        # For scalar case (n_c=None or n_c=1), squeeze to scalar
        if data.dim() == 1 and data.shape[0] == 1:
            data = data.squeeze()
        n_c = data.shape[0] if data.dim() == 1 else 1
    
    return data, n_c


def _prepare_bound_value(value, data_shape, n_c):
    """
    Prepare min/max bound value with broadcasting.
    
    Args:
        value: Bound value (scalar, int, float, or tensor), or None
        data_shape: Shape of the data tensor
        n_c: Number of parallel components
        
    Returns:
        Prepared bound tensor, or None if input was None
    """
    if value is None:
        return None
    
    value = _to_float64_tensor(value)
    
    # Broadcast to match data shape if needed
    if len(data_shape) == 1 and data_shape[0] > 1:
        if value.dim() == 0:
            value = value.expand(data_shape[0]).clone()
        elif value.dim() == 1 and value.shape[0] == 1:
            value = value.expand(data_shape[0]).clone()
    
    return value


def _broadcast_for_n_c(value, n_c):
    """
    Broadcast a value to match n_c if needed.
    
    Args:
        value: Input value (scalar, int, float, or tensor)
        n_c: Number of parallel components
        
    Returns:
        Broadcasted tensor
    """
    value = _to_float64_tensor(value)
    
    if n_c > 1 and value.dim() == 0:
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
