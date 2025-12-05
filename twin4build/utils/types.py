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
        tensor (torch.Tensor): The underlying tensor storing vector values.
        batch_size (int): The batch size.
        size (int): The size of the vector (number of elements).
        log_history (bool): Whether to log the history of values.
        history (torch.Tensor): The history of values over time.
        is_leaf (bool): Whether this vector is a leaf node in the graph (input).
        do_normalization (bool): Whether to normalize the history.
        optional (bool): Whether the vector is optional.
    """

    def __init__(
        self,
        tensor: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        size: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        log_history: bool = True,
        is_leaf: bool = False,
        do_normalization: bool = False,
        optional: bool = False,
    ) -> None:
        """
        Initialize a Vector instance.

        Args:
            tensor (Optional[torch.Tensor]): Initial tensor value.
            batch_size (int): The batch size. Defaults to 1.
            size (Optional[int]): The size of the vector.
            log_history (bool): Whether to log history. Defaults to True.
            is_leaf (bool): Whether this vector is a leaf node. Defaults to False.
            do_normalization (bool): Whether to normalize history. Defaults to False.
            optional (bool): Whether this vector is optional. Defaults to False.
        """
        self._tensor = None
        self._batch_size = batch_size
        self._size = size
        self._n_timesteps = n_timesteps
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
    def size(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_timesteps(self):
        return self._n_timesteps

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
            if self.size > 0:
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

    # def reset(self) -> None:
    #     """Reset the vector to initial state."""
    #     self._size = self._init_size
    #     return self

    def initialize(
        self,
        n_timesteps: int,
        size: Optional[int] = None,
        batch_size: Optional[int] = None,
        values: Optional[List[float]] = None,
        force: bool = False,
    ) -> None:
        """Initialize the vector tensor and sorting indices.

        Creates the underlying torch tensor and computes indices for sorted access by group ID.
        """
        assert isinstance(n_timesteps, int), "n_timesteps must be an integer"
        if n_timesteps is not None:
            self._n_timesteps = n_timesteps
        if size is not None:
            self._size = size
        if batch_size is not None:
            self._batch_size = batch_size

        ### Vector stufff
        if self._init_tensor is None:
            self._tensor = torch.zeros(
                (self.batch_size, self.size), dtype=torch.float64
            )
        else:
            self._tensor = self._init_tensor.clone()
        # self.current_idx = 0

        if values is not None:
            values = _convert_to_3D_tensor(values)

        # We return early if this scalar has requires_grad=True.
        # This is the case when used in the optimizer.
        # Here we dont want to reinitialize the history as the torch.optim.Optimizer changes this in-place.
        if (
            self._initialized
            and self._requires_reinittialization == False
            and force == False
        ):
            # self._history_is_populated = False # When we reinitialize a leaf Scalar, a simulation must be run before the history is populated.
            return

        if self._is_leaf:
            assert values is not None, "Values must be provided for leaf scalars"
            assert (
                values.shape[0] == self.batch_size
            ), "Values must be the same length as the batch size"
            assert (
                values.shape[1] == self.n_timesteps
            ), "Values must be the same length as the number of date_time_steps"
            assert (
                values.shape[2] == self.size
            ), "Values must be the same length as the vector size"
            # Pre-allocate the history tensor with the correct size
            self._history = values
            self._history_is_populated = True
            if self._do_normalization:
                self._normalized_history = self.normalize()

        else:
            self._history = torch.zeros(
                (self.batch_size, self.n_timesteps, self.size),
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
            v (float): Value to set at current index.
            step_index (Optional[int]): Step index to set value at.
            index (Optional[int, torch.Tensor]): Index to set value at.
        """

        # v = _convert_to_2D_tensor(v)
        # print("v.shape", v.shape)
        # print("index", index)
        # print("self.tensor.shape", self.tensor.shape)

        if index is not None:
            self._tensor[:, index] = v
        else:
            self._tensor[:, :] = v

        if self._log_history:
            assert (
                step_index is not None
            ), "step_index must be provided when logging history"
            # if self._do_normalization:
            if self.is_leaf == False or (self.is_leaf and self._do_normalization):
                if v.dim() == 2:
                    self._history[:, step_index, :] = v
                elif v.dim() == 1:
                    self._history[:, step_index, index] = v
                else:
                    raise ValueError(f"Unsupported dimension: {v.dim()}")

            if step_index == self._history.shape[1] - 1:
                self._history_is_populated = True
            else:
                self._history_is_populated = False

    def get(self, index: Optional[int, torch.Tensor] = None) -> torch.Tensor:
        """Get vector values sorted by group ID.

        Returns:
            torch.Tensor: Tensor of values sorted by group ID.
        """
        if index is not None:
            out = self.tensor[:, index]
        else:
            out = self.tensor
        return out

    # def update(self, group_id: Optional[int] = None) -> None:
    #     """Update the vector with a new group ID.

    #     Args:
    #         group_id (Optional[int]): Group ID to add. If None, uses current size.
    #     """
    #     assert self._current_idx +1 <= self.size, "Vector size is not large enough to add a new group ID"
    #     if group_id is None:
    #         group_id = self._current_idx
    #     self.id_map_reverse[group_id] = self._current_idx
    #     self.id_map[self._current_idx] = group_id
    #     self._current_idx += 1

    def copy(self):
        """Create a copy of the vector.

        Returns:
            Vector: A new Vector instance with the same data.
        """
        tensor = self.tensor.clone() if self.tensor is not None else None
        copy = Vector(
            tensor=tensor,
            batch_size=self.batch_size,
            size=self.size,
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
        scalar (torch.Tensor): The wrapped scalar value.
        batch_size (int): The batch size.
        log_history (bool): Whether to log the history of values.
        history (torch.Tensor): The history of values over time.
        is_leaf (bool): Whether this scalar is a leaf node in the graph (input).
        do_normalization (bool): Whether to normalize the history.
        optional (bool): Whether the scalar is optional.
    """

    def __init__(
        self,
        scalar: Optional[Union[float, int, torch.Tensor]] = None,
        batch_size: int = 1,
        n_timesteps: Optional[int] = None,
        log_history: bool = True,
        is_leaf: bool = False,
        do_normalization: bool = False,
        optional: bool = False,
    ) -> None:
        """
        Initialize a Scalar instance.

        Args:
            scalar (Optional[Union[float, int, torch.Tensor]]): Initial scalar value.
            batch_size (int): The batch size. Defaults to 1.
            log_history (bool): Whether to log history. Defaults to True.
            is_leaf (bool): Whether this scalar is a leaf node. Defaults to False.
            do_normalization (bool): Whether to normalize history. Defaults to False.
            optional (bool): Whether this scalar is optional. Defaults to False.
        """
        assert isinstance(
            scalar, (float, int, torch.Tensor, type(None))
        ), "Scalar must be a float, int, np.ndarray, torch.Tensor, or None"

        if isinstance(scalar, torch.Tensor):
            assert (
                scalar.numel() == 1
            ), f"Scalar must be a single value, got {scalar.numel()} values"
            assert (
                scalar.dim() == 0 or scalar.dim() == 1
            ), f"Scalar must have 0 or 1 dimensions, got {scalar.dim()} dimensions"
            if scalar.dim() == 0:
                scalar = scalar.unsqueeze(0)
            scalar.requires_grad = False

        elif isinstance(scalar, (float, int)):
            scalar = torch.tensor([scalar], dtype=torch.float64, requires_grad=False)

        self._tensor = scalar
        self._batch_size = batch_size
        self._n_timesteps = n_timesteps
        self._init_scalar = scalar
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
    def batch_size(self):
        return self._batch_size

    @property
    def n_timesteps(self):
        return self._n_timesteps

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
        n_timesteps: int,
        batch_size: Optional[int] = None,
        values: Optional[List[float]] = None,
        force: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the scalar.

        Args:
            n_timesteps (int): The number of timesteps.
            batch_size (Optional[int]): The batch size.
            values (Optional[List[float]]): The values to initialize the scalar with.
            force (bool): Whether to force the initialization.
        """
        assert isinstance(n_timesteps, int), "n_timesteps must be an integer"

        if batch_size is not None:
            self._batch_size = batch_size

        if n_timesteps is not None:
            self._n_timesteps = n_timesteps

        if self._init_scalar is None:
            self._tensor = torch.zeros((self.batch_size), dtype=torch.float64)
        else:
            self._tensor = self._init_scalar.clone()
            if self.batch_size > 1:
                if self._tensor.dim() == 0:
                    self._tensor = self._tensor.expand(self.batch_size).clone()
                elif self._tensor.dim() == 1 and self._tensor.shape[0] == 1:
                    self._tensor = self._tensor.expand(self.batch_size).clone()

        if values is not None:
            values = _convert_to_2D_tensor(values)

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
                values.shape[0] == self.batch_size
            ), f"First dimension of values ({values.shape[0]}) must be the same as the batch size ({self.batch_size}). Did you forget to provide the batch_size argument in the initialize method?"
            assert (
                values.shape[1] == self.n_timesteps
            ), f"Second dimension of values ({values.shape[1]}) must be the same as the number of date_time_steps ({self.n_timesteps}). Did you forget to provide the n_timesteps argument in the initialize method?"
            # Pre-allocate the history tensor with the correct size
            self._history = values
            self._history_is_populated = True
            if self._do_normalization:
                self._normalized_history = self.normalize()

        else:
            self._history = torch.zeros(
                self.batch_size,
                self.n_timesteps,
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
            v (Union[Scalar, float]): Value to set.
        """
        if self._is_leaf:
            assert (
                v is None
            ), "Values cannot be set for leaf scalars. Use scalar.set(step_index=step_index) to set value based on history"
            assert (
                step_index is not None
            ), "step_index must be provided for leaf scalars"
            if self._do_normalization:
                v = self._normalized_history[:, step_index]
                v = self.denormalize(v)
            else:
                v = self._history[:, step_index]
        else:
            v = _convert_to_1D_scalar_tensor(v)

        if apply is not None:
            v = apply(v)

        self._tensor = v
        if self._log_history:
            assert (
                step_index is not None
            ), "step_index must be provided when logging history"
            # if self._do_normalization:
            if self.is_leaf == False or (self.is_leaf and self._do_normalization):
                self._history[:, step_index] = v

            if step_index == self._history.shape[1] - 1:
                self._history_is_populated = True
            # else:
            #     self._history_is_populated = False ################## TODO: Remove this once we have a way to handle the case where the history is not populated

    def get(self, *args, **kwargs) -> torch.Tensor:
        """Get the scalar value.

        Returns:
            float: Scalar value.
        """
        return self._tensor

    def normalize(self, v: torch.Tensor = None):
        assert (
            self._history_is_populated == True
        ), "History must be populated before normalizing"
        if v is None:
            v = self._history
        # else:
        # print(v)
        v = _convert_to_2D_tensor(v)
        # elif isinstance(v, torch.Tensor):
        #     v = torch.tensor(v, dtype=torch.float64)
        assert isinstance(v, torch.Tensor), "v must be a torch.Tensor"

        # Cache min/max as Python floats to avoid GradTrackingTensor issues
        if self._min_history is None:
            no_nan_history = self._history.detach()  # a[~a.isnan()])
            no_nan_history = no_nan_history[~torch.isnan(no_nan_history)]
            self._min_history = torch.min(
                no_nan_history  # , dim=1 # TODO: remove dim=1 if we want to normalize all periods together
            ).item()  # Store as numpy float
        if self._max_history is None:
            # with torch.no_grad():
            no_nan_history = self._history.detach()  # a[~a.isnan()])
            no_nan_history = no_nan_history[~torch.isnan(no_nan_history)]
            self._max_history = torch.max(
                no_nan_history  # , dim=1 # TODO: remove dim=1 if we want to normalize all periods together
            ).item()  # Store as Python float

        # Convert cached floats to tensors when needed
        min_val = _convert_to_1D_scalar_tensor(self._min_history)
        max_val = _convert_to_1D_scalar_tensor(self._max_history)

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
        min_val = _convert_to_1D_scalar_tensor(self._min_history)
        max_val = _convert_to_1D_scalar_tensor(self._max_history)
        return v * (max_val - min_val) + min_val

    def get_float(self) -> float:
        """Get the scalar value as a float.

        Returns:
            float: Scalar value.
        """
        return self._tensor.item()

    # def reset(self):
    #     if self._init_scalar is not None:
    #         self._tensor = self._init_scalar.clone()
    #     else:
    #         self._tensor = None

    def copy(self):
        copy = Scalar()
        copy._tensor = self._tensor
        copy._init_scalar = self._init_scalar
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
    """

    def __new__(cls, data, min_value=None, max_value=None, requires_grad=True):
        # Convert data to tensor if it's not already
        data = _convert_to_1D_scalar_tensor(data).squeeze()
        # validate = True
        # Set min and max values
        if min_value is None:
            if torch.all(data < 0):
                min_value = torch.tensor(data.detach().clone(), dtype=torch.float64)
            else:
                min_value = torch.tensor(0, dtype=torch.float64)
            # validate = False
        else:
            min_value = _convert_to_1D_scalar_tensor(min_value).squeeze()

        if max_value is None:
            if torch.all(data < 0):
                max_value = torch.tensor(0, dtype=torch.float64)
            elif torch.allclose(data, torch.zeros_like(data)):
                max_value = torch.tensor(1, dtype=torch.float64)
            else:
                max_value = torch.tensor(data.detach().clone(), dtype=torch.float64)

        else:
            max_value = _convert_to_1D_scalar_tensor(max_value).squeeze()

        # if validate:
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

    @min_value.setter
    def min_value(self, value):
        value = _convert_to_1D_scalar_tensor(value).squeeze()
        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        value = _convert_to_1D_scalar_tensor(value).squeeze()
        self._max_value = value

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        v = _convert_to_1D_scalar_tensor(v).squeeze()

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _convert_to_1D_scalar_tensor(min_value).squeeze()

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _convert_to_1D_scalar_tensor(max_value).squeeze()

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
        value = _convert_to_1D_scalar_tensor(value).squeeze()
        if normalized:
            normalized_value = value
        else:
            normalized_value = self.normalize(value)
        self.data.copy_(normalized_value)


class TensorParameter:
    """
    A custom nn.Parameter implementation that normalizes the data between 0 and 1 to stabilize gradients in physical systems where the parameters scales can be different.

    This class is used to represent model parameters as a Tensor when we calculate the Jacobian analytically as the jac = torch.nn.functional.Jacobian() has the signature jac(f: callable, input: Tensor) -> Tensor.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        min_value=None,
        max_value=None,
        normalized: bool = True,
    ):
        tensor = _convert_to_1D_scalar_tensor(tensor)
        self._min_value = min_value
        self._max_value = max_value

        self.set(tensor, normalized=normalized)

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @min_value.setter
    def min_value(self, value):
        value = _convert_to_1D_scalar_tensor(value).squeeze()
        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        value = _convert_to_1D_scalar_tensor(value).squeeze()
        self._max_value = value

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        v = _convert_to_1D_scalar_tensor(v).squeeze()

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _convert_to_1D_scalar_tensor(min_value).squeeze()

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _convert_to_1D_scalar_tensor(max_value).squeeze()

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
        value = _convert_to_1D_scalar_tensor(value).squeeze()
        if normalized:
            value = self.denormalize(value)
        self.tensor = value


def _convert_to_3D_tensor(v: Union[Scalar, float, int, torch.Tensor]):
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
        ), f"Value must have less that or equal to 2 dimensions, got {v.dim()} dimensions"
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


def _convert_to_2D_tensor(v: Union[Scalar, float, int, torch.Tensor]):
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
        v = _convert_to_2D_tensor(v.get())
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
