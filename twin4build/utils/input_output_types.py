from __future__ import annotations  # This allows using string literals in type hints
import os
import sys
import datetime
from dateutil import tz
import numpy as np
import functools
from typing import Optional, Union, List
import torch
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


class Vector():
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

    def __init__(self, tensor: Optional[torch.Tensor] = None, size: Optional[int] = None) -> None:
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

    def initialize(self, 
                   startTime: Optional[datetime.datetime] = None, 
                   endTime: Optional[datetime.datetime] = None,
                   stepSize: Optional[int] = None,
                   simulator: Optional[core.Simulator] = None) -> None:
        """Initialize the vector tensor and sorting indices.
        
        Creates the underlying torch tensor and computes indices for sorted access by group ID.
        """
        if self._init_tensor is None:
            self.tensor = torch.zeros(self.size, dtype=torch.float32)
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
    
@functools.total_ordering
class Scalar:
    """A custom scalar implementation with operator overloading.
    
    This class wraps a single scalar value and provides arithmetic operations 
    compatibility with other Scalar instances, numeric types, and numpy arrays.
    Implements total ordering through the @functools.total_ordering decorator.

    Attributes:
        scalar (Union[float, int, np.ndarray, None]): The wrapped scalar value.
    """

    def __init__(self, scalar: Optional[Union[float, int, torch.Tensor]] = None, save_history: bool = True, is_leaf: bool = False, normalize: bool = False) -> None:
        """Initialize a Scalar instance.

        Args:
            scalar (Optional[Union[float, int, np.ndarray]], optional): Initial scalar value. 
                Defaults to None.
        """
        assert isinstance(scalar, (float, int, torch.Tensor, type(None))), "Scalar must be a float, int, np.ndarray, torch.Tensor, or None"
        if is_leaf: # If the Scalar is a leaf, we calculate the full history when initializing
            save_history = False

        if isinstance(scalar, torch.Tensor):
            assert scalar.numel() == 1, f"Scalar must be a single value, got {scalar.numel()} values"
            assert scalar.dim() == 0 or scalar.dim() == 1, f"Scalar must have 0 or 1 dimensions, got {scalar.dim()} dimensions"
            if scalar.dim() == 0:
                scalar = scalar.unsqueeze(0)
            scalar.requires_grad = False

        elif isinstance(scalar, (float, int)):
            scalar = torch.tensor([scalar], dtype=torch.float32, requires_grad=False)

        self._scalar = scalar
        self._init_scalar = scalar
        self._history = None
        self._save_history = save_history
        self._is_leaf = is_leaf
        self._normalize = normalize
        self._initialized = False
        self._requires_reinittialization = True

    @property
    def save_history(self):
        return self._save_history

    @save_history.setter
    def save_history(self, value: bool):
        self._save_history = value

    @property
    def scalar(self):
        return self._scalar

    @property
    def history(self):
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
    def normalize(self):
        return self._normalize
    
    @normalize.setter
    def normalize(self, value: bool):
        assert isinstance(value, bool), "normalize must be a boolean"
        self._normalize = value

    def __add__(self, other: Union["Scalar", int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add another value to this scalar.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to add.

        Returns:
            Union[float, np.ndarray]: Result of addition.

        Raises:
            TypeError: If other is not a supported type.
        """
        if isinstance(other, Scalar):
            return self._scalar + other.scalar
        elif isinstance(other, (int, float)):
            return self._scalar + other
        elif isinstance(other, np.ndarray):
            return self._scalar + other
        else:
            raise TypeError(f"unsupported operand type(s) for +: 'Scalar' and '{type(other)}'")
        
    def __radd__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reverse add operation.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to add.

        Returns:
            Union[float, np.ndarray]: Result of addition.
        """
        return self.__add__(other)
    
    def __sub__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Subtract another value from this scalar.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to subtract.

        Returns:
            Union[float, np.ndarray]: Result of subtraction.

        Raises:
            TypeError: If other is not a supported type.
        """
        if isinstance(other, Scalar):
            return self._scalar - other.scalar
        elif isinstance(other, (int, float)):
            return self._scalar - other
        elif isinstance(other, np.ndarray):
            return self._scalar - other
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'Scalar' and '{type(other)}'")
        
    def __rsub__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reverse subtract operation.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to subtract from.

        Returns:
            Union[float, np.ndarray]: Result of subtraction.
        """
        return -self.__sub__(other)
    
    def __mul__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Multiply this scalar by another value.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to multiply by.

        Returns:
            Union[float, np.ndarray]: Result of multiplication.

        Raises:
            TypeError: If other is not a supported type.
        """
        if isinstance(other, Scalar):
            return self._scalar * other.scalar
        elif isinstance(other, (int, float)):
            return self._scalar * other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self._scalar * other
        else:
            raise TypeError(f"unsupported operand type(s) for *: 'Scalar' and '{type(other)}'")
        
    def __rmul__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reverse multiply operation.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to multiply by.

        Returns:
            Union[float, np.ndarray]: Result of multiplication.
        """
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Divide this scalar by another value.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to divide by.

        Returns:
            Union[float, np.ndarray]: Result of division.

        Raises:
            TypeError: If other is not a supported type.
        """
        if isinstance(other, Scalar):
            return self._scalar / other.scalar
        elif isinstance(other, (int, float)):
            return self._scalar / other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self._scalar / other
        else:
            raise TypeError(f"unsupported operand type(s) for /: 'Scalar' and '{type(other)}'")
    
    def __rtruediv__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reverse divide operation.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to divide by.

        Returns:
            Union[float, np.ndarray]: Result of division.
        """
        return self.__truediv__(other)
    
    def __eq__(self, other: Union[Scalar, int, float, np.ndarray]) -> bool:
        """Test equality with another value.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to compare with.

        Returns:
            bool: True if values are equal, False otherwise.

        Raises:
            TypeError: If other is not a supported type.
        """
        if isinstance(other, Scalar):
            return self._scalar == other.scalar
        elif isinstance(other, (int, float)):
            return self._scalar == other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self._scalar == other
        else:
            raise TypeError(f"unsupported operand type(s) for ==: 'Scalar' and '{type(other)}'")
        
    def __lt__(self, other: Union[Scalar, int, float, np.ndarray]) -> bool:
        """Test less than with another value.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to compare with.

        Returns:
            bool: True if self is less than other, False otherwise.

        Raises:
            TypeError: If other is not a supported type.
        """
        if isinstance(other, Scalar):
            return self._scalar < other.scalar
        elif isinstance(other, (int, float)):
            return self._scalar < other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self._scalar < other
        else:
            raise TypeError(f"unsupported operand type(s) for <: 'Scalar' and '{type(other)}'")

    def __str__(self) -> str:
        """Get string representation of the scalar.

        Returns:
            str: String representation of the scalar value.
        """
        return str(self._scalar)
    
    def set_requires_grad(self, requires_grad: bool):
        assert self._is_leaf, "Only leaf scalars can have their requires_grad attribute set"
        if self._normalize:
            self._normalized_history.requires_grad = requires_grad
        else:
            self._history.requires_grad = requires_grad
        self._requires_reinittialization = not requires_grad
    
    def initialize(self, 
                   startTime: Optional[datetime.datetime] = None, 
                   endTime: Optional[datetime.datetime] = None,
                   stepSize: Optional[int] = None,
                   simulator: Optional[core.Simulator] = None,
                   values: Optional[List[float]] = None):
        if self._initialized and self._requires_reinittialization==False:
            return
        
        if self._is_leaf:
            assert values is not None, "Values must be provided for leaf scalars"
            assert len(values) == len(simulator.dateTimeSteps), "Values must be the same length as the number of dateTimeSteps"
            # Pre-allocate the history tensor with the correct size
            self._history = torch.tensor(values, dtype=torch.float32, requires_grad=False)
            if self._normalize:
                self._min_history = 0#torch.min(self._history)
                self._max_history = torch.max(self._history)
                self._normalized_history = (self._history - self._min_history) / (self._max_history - self._min_history) # Min-max normalization
        else:
            self._history = torch.zeros(len(simulator.dateTimeSteps), dtype=torch.float32, requires_grad=False)
        self._initialized = True

    def set(self, v: Union[Scalar, float]=None, stepIndex: Optional[int] = None) -> None:
        """Set the scalar value.

        Args:
            v (Union[Scalar, float]): Value to set.
        """
        if self._is_leaf:
            if self._normalize:
                v = self._normalized_history[stepIndex]
                v = v * (self._max_history - self._min_history) + self._min_history
            else:
                v = self._history[stepIndex]
        elif isinstance(v, Scalar):
            v = v.get()
        elif isinstance(v, (float, int)):
            v = torch.tensor([v], dtype=torch.float32)
        elif isinstance(v, torch.Tensor):
            assert v.numel() == 1, f"Scalar must be a single value, got {v.numel()} values"
            assert v.dim() == 0 or v.dim() == 1, f"Scalar must have 0 or 1 dimensions, got {v.dim()} dimensions"
            if v.dim() == 0:
                v = v.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported type: {type(v)}")
        
        self._scalar = v
        if self._save_history:            
            # Use index_put_ to update a single value while maintaining the computational graph
            self._history[stepIndex] = v

    def get(self) -> torch.Tensor:
        """Get the scalar value.

        Returns:
            float: Scalar value.
        """
        return self._scalar
    
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
        copy._save_history = self._save_history
        copy._is_leaf = self._is_leaf
        return copy


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

    print(b+c)
    print(b-c)
    print(b*c)
    print(c-b)
    print(b/c)
    print(c/b)







if __name__ == '__main__':
    test()
