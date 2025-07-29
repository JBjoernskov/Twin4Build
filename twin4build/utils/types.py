from __future__ import annotations  # This allows using string literals in type hints

# Standard library imports
import datetime
import functools
import os
import sys
from collections import OrderedDict
from typing import List, Optional, Union

# Third party imports
import numpy as np
import torch
import torch.nn as nn
from dateutil import tz

# Local application imports
import twin4build.core as core

# ###Only for testing before distributing package
# if __name__ == '__main__':
#     uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
#     file_path = uppath(os.path.abspath(__file__), 3)
#     sys.path.append(file_path)


# class History(list):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def plain(self):
#         return [x.item() for x in self]


class Vector:
    """A custom vector implementation with mapping capabilities.

    This class implements a vector (1D array) with additional functionality to map between
    indices and group IDs. It maintains internal mappings and provides methods for
    initialization, updates, and value access.

    Attributes:
        size (int): Current size of the vector.
        id_map (Dict[int, int]): Maps vector indices to group IDs.
        id_map_reverse (Dict[int, int]): Maps group IDs to vector indices.
        tensor (torch.Tensor): The underlying tensor storing vector values.
        current_idx (int): Current index pointer for setting values.
        sorted_id_indices (torch.Tensor): Indices that sort the vector by group IDs.
    """

    def __init__(
        self, tensor: Optional[torch.Tensor] = None, size: Optional[int] = None
    ) -> None:
        """Initialize an empty Vector instance."""
        self.id_map = {}
        self.id_map_reverse = {}
        self.sorted_id_indices = None
        self.tensor = None
        self._init_tensor = tensor
        self._init_id_map_reverse = self.id_map_reverse
        self._init_id_map = self.id_map

        if tensor is None and size is None:
            self.size = 0
            self._init_size = 0

        else:
            assert isinstance(size, int), f"Size must be an integer. Got {type(size)}"
            self.size = size
            self._init_size = size
            for s in range(size):
                self.id_map_reverse[s] = s
                self.id_map[s] = s

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
        return self.tensor[key].item()

    def __setitem__(self, key: int, value: float) -> None:
        """Set value at specified index.

        Args:
            key (int): Index to set.
            value (float): Value to set.
        """
        self.tensor[key] = value

    def reset(self) -> None:
        """Reset the vector to initial state."""
        self.size = self._init_size
        self.id_map = self._init_id_map
        self.id_map_reverse = self._init_id_map_reverse
        return self

    def initialize(
        self,
        startTime: Optional[datetime.datetime] = None,
        endTime: Optional[datetime.datetime] = None,
        stepSize: Optional[int] = None,
        simulator: Optional[core.Simulator] = None,
    ) -> None:
        """Initialize the vector tensor and sorting indices.

        Creates the underlying torch tensor and computes indices for sorted access by group ID.
        """
        if self._init_tensor is None:
            self.tensor = torch.zeros(self.size, dtype=torch.float64)
        else:
            self.tensor = self._init_tensor.clone()
        self.current_idx = 0
        id_array = torch.empty(self.size, dtype=torch.int64)
        for idx, group_id in self.id_map.items():
            id_array[idx] = group_id
        self.sorted_id_indices = torch.argsort(id_array)
        return self

    def increment(self, v: int = 1) -> None:
        """Increment the vector size.

        Args:
            v (int, optional): Amount to increment by. Defaults to 1.
        """
        self.size += v
        return self

    def set(self, v: float, stepIndex: Optional[int] = None) -> None:
        """Set the next value in the vector.

        Args:
            v (float): Value to set at current index.
        """
        if isinstance(v, float):
            self[self.current_idx] = v
        elif isinstance(v, Scalar):
            self[self.current_idx] = v.get()
        elif isinstance(v, torch.Tensor):
            self[:] = v
        elif isinstance(v, Vector):
            self[:] = v[:]

        self.current_idx += 1
        if self.current_idx == self.size:
            self.current_idx = 0

    def get(self) -> torch.Tensor:
        """Get vector values sorted by group ID.

        Returns:
            torch.Tensor: Tensor of values sorted by group ID.
        """
        return self.tensor[self.sorted_id_indices]

    def update(self, group_id: Optional[int] = None) -> None:
        """Update the vector with a new group ID.

        Args:
            group_id (Optional[int]): Group ID to add. If None, uses current size.
        """
        if group_id is None:
            group_id = self.size
        self.id_map_reverse[group_id] = self.size
        self.id_map[self.size] = group_id
        self.increment()

    def copy(self):
        """Create a copy of the vector.

        Returns:
            Vector: A new Vector instance with the same data.
        """
        copy = Vector()
        copy.size = self.size
        copy.id_map = self.id_map.copy()
        copy.id_map_reverse = self.id_map_reverse.copy()
        copy.initialize()
        return copy


class Scalar:
    """A custom scalar implementation with operator overloading.

    This class wraps a single scalar value and provides arithmetic operations
    compatibility with other Scalar instances, numeric types, and numpy arrays.
    Implements total ordering through the @functools.total_ordering decorator.

    Attributes:
        scalar (Union[float, int, np.ndarray, None]): The wrapped scalar value.
    """

    def __init__(
        self,
        scalar: Optional[Union[float, int, torch.Tensor]] = None,
        log_history: bool = True,
        is_leaf: bool = False,
        do_normalization: bool = False,
    ) -> None:
        """Initialize a Scalar instance.

        Args:
            scalar (Optional[Union[float, int, np.ndarray]], optional): Initial scalar value.
                Defaults to None.
        """
        assert isinstance(
            scalar, (float, int, torch.Tensor, type(None))
        ), "Scalar must be a float, int, np.ndarray, torch.Tensor, or None"
        # if is_leaf: # If the Scalar is a leaf, we calculate the full history when initializing
        #     log_history = False

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

        self._scalar = scalar
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

    # def make_pickable(self):
    #     if self._scalar is not None:
    #         self._scalar = torch.tensor([self._scalar.item()], dtype=torch.float64, requires_grad=False)
    #     if self._init_scalar is not None:
    #         self._init_scalar = torch.tensor([self._init_scalar.item()], dtype=torch.float64, requires_grad=False)
    #     self._history = None
    #     self._normalized_history = None
    #     self._initialized = False
    #     self._requires_reinittialization = True
    #     self._min_history = None
    #     self._max_history = None
    #     self._history_is_populated = False
    #     self._is_normalized = False

    @property
    def log_history(self):
        return self._log_history

    @log_history.setter
    def log_history(self, value: bool):
        self._log_history = value

    @property
    def scalar(self):
        return self._scalar

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

    def __str__(self) -> str:
        """Get string representation of the scalar.

        Returns:
            str: String representation of the scalar value.
        """
        return str(self._scalar)

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
        startTime: Optional[datetime.datetime] = None,
        endTime: Optional[datetime.datetime] = None,
        stepSize: Optional[int] = None,
        simulator: Optional[core.Simulator] = None,
        values: Optional[List[float]] = None,
        force: bool = False,
    ):
        assert isinstance(
            values, (list, torch.Tensor, np.ndarray, type(None))
        ), "values must be a list or torch.Tensor"
        if isinstance(values, torch.Tensor):
            assert values.ndim == 1, "values must be a 1D torch.Tensor"

        elif isinstance(values, np.ndarray):
            assert values.ndim == 1, "values must be a 1D numpy array"
            values = torch.tensor(values, dtype=torch.float64)

        elif isinstance(values, list):
            values = torch.tensor(values, dtype=torch.float64)
            assert (
                values.ndim == 1
            ), "if a list is provided, it must convert to a 1D torch.Tensor"

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
            assert values.shape[0] == len(
                simulator.dateTimeSteps
            ), "Values must be the same length as the number of dateTimeSteps"
            # Pre-allocate the history tensor with the correct size
            self._history = values
            self._history_is_populated = True
            if self._do_normalization:
                self._normalized_history = self.normalize()

        else:
            self._history = torch.zeros(
                len(simulator.dateTimeSteps), dtype=torch.float64, requires_grad=False
            )
            self._history_is_populated = False

        self._initialized = True

    def set(
        self,
        v: Union[Scalar, float, int, torch.Tensor] = None,
        stepIndex: Optional[int] = None,
        apply: callable = None,
    ) -> None:
        """Set the scalar value.

        Args:
            v (Union[Scalar, float]): Value to set.
        """
        if self._is_leaf:
            assert (
                v is None
            ), "Values cannot be set for leaf scalars. Use scalar.set(stepIndex=step_index) to set value based on history"
            if self._do_normalization:
                v = self._normalized_history[stepIndex]
                v = self.denormalize(v)
            else:
                v = self._history[stepIndex]
        else:
            v = _convert_to_scalar_tensor(v)

        if apply is not None:
            v = apply(v)

        self._scalar = v
        if self._log_history:
            # if self._do_normalization:
            if self.is_leaf == False or (self.is_leaf and self._do_normalization):
                self._history[stepIndex] = v

            if stepIndex == self._history.shape[0] - 1:
                self._history_is_populated = True
            else:
                self._history_is_populated = False

    def get(self) -> torch.Tensor:
        """Get the scalar value.

        Returns:
            float: Scalar value.
        """
        return self._scalar

    def normalize(self, v: torch.Tensor = None):
        assert (
            self._history_is_populated == True
        ), "History must be populated before normalizing"
        if v is None:
            v = self._history
        # else:
        # print(v)
        v = _convert_to_1D_tensor(v)
        # elif isinstance(v, torch.Tensor):
        #     v = torch.tensor(v, dtype=torch.float64)
        assert isinstance(v, torch.Tensor), "v must be a torch.Tensor"

        # Cache min/max as Python floats to avoid GradTrackingTensor issues
        if self._min_history is None:
            # with torch.no_grad():
            self._min_history = torch.min(
                self._history.detach()
            ).item()  # Store as Python float
        if self._max_history is None:
            # with torch.no_grad():
            self._max_history = torch.max(
                self._history.detach()
            ).item()  # Store as Python float

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
        return self._scalar.item()

    def update(self, group_id: Optional[int] = None):
        pass

    def reset(self):
        if self._init_scalar is not None:
            self._scalar = self._init_scalar.clone()
        else:
            self._scalar = None

    def copy(self):
        copy = Scalar()
        copy._scalar = self._scalar
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
        data = _convert_to_scalar_tensor(data).squeeze()
        # validate = True
        # Set min and max values
        if min_value is None:
            if torch.all(data < 0):
                min_value = data.clone()
            else:
                min_value = torch.tensor(0, dtype=torch.float64)
            # validate = False
        else:
            min_value = _convert_to_scalar_tensor(min_value).squeeze()

        if max_value is None:
            if torch.all(data < 0):
                max_value = torch.tensor(0, dtype=torch.float64)
            elif torch.allclose(data, torch.zeros_like(data)):
                max_value = torch.tensor(1, dtype=torch.float64)
            else:
                max_value = data.clone()

        else:
            max_value = _convert_to_scalar_tensor(max_value).squeeze()

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
        value = _convert_to_scalar_tensor(value).squeeze()
        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        value = _convert_to_scalar_tensor(value).squeeze()
        self._max_value = value

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        v = _convert_to_scalar_tensor(v).squeeze()

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _convert_to_scalar_tensor(min_value).squeeze()

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _convert_to_scalar_tensor(max_value).squeeze()

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
        value = _convert_to_scalar_tensor(value).squeeze()
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
        tensor = _convert_to_scalar_tensor(tensor)
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
        value = _convert_to_scalar_tensor(value).squeeze()
        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        value = _convert_to_scalar_tensor(value).squeeze()
        self._max_value = value

    def normalize(
        self,
        v: torch.Tensor,
        min_value: torch.Tensor = None,
        max_value: torch.Tensor = None,
    ):
        v = _convert_to_scalar_tensor(v).squeeze()

        if min_value is None:
            min_value = self._min_value
        else:
            min_value = _convert_to_scalar_tensor(min_value).squeeze()

        if max_value is None:
            max_value = self._max_value
        else:
            max_value = _convert_to_scalar_tensor(max_value).squeeze()

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
        value = _convert_to_scalar_tensor(value).squeeze()
        if normalized:
            value = self.denormalize(value)
        self.tensor = value


def _convert_to_scalar_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    if isinstance(v, Scalar):
        v = v.get()
    elif isinstance(v, (float, int)):
        v = torch.tensor([v], dtype=torch.float64)
    elif isinstance(v, torch.Tensor):
        assert v.numel() == 1, f"Value must be a single value, got {v.numel()} values"
        assert (
            v.dim() == 0 or v.dim() == 1
        ), f"Value must have 0 or 1 dimensions, got {v.dim()} dimensions"
        if v.dim() == 0:
            v = v.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported type: {type(v)}")
    return v


def _convert_to_1D_tensor(v: Union[Scalar, float, int, torch.Tensor]):
    if isinstance(v, Scalar):
        v = v.get()
    elif isinstance(v, (float, int)):
        v = torch.tensor([v], dtype=torch.float64)
    elif isinstance(v, torch.Tensor):
        assert (
            v.dim() == 0 or v.dim() == 1
        ), f"Value must have 0 or 1 dimensions, got {v.dim()} dimensions"
        if v.dim() == 0:
            v = v.unsqueeze(0)
    elif isinstance(v, torch.Tensor) == False:
        raise TypeError(f"Unsupported type: {type(v)}")
    return v


def test():
    a = Vector()
    a.increment()
    a.increment()
    a.increment(2)
    a.initialize()

    print(a)

    for i in range(100):
        print("---")
        a.set(i)
        print(a)

    b = Scalar()
    b.set(5)
    c = Scalar()
    c.set(2.0)

    print(b + c)
    print(b - c)
    print(b * c)
    print(c - b)
    print(b / c)
    print(c / b)


if __name__ == "__main__":
    test()

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
