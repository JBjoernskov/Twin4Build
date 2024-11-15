from __future__ import annotations  # This allows using string literals in type hints
import os
import sys
import datetime
from dateutil import tz
import numpy as np
import functools
from typing import Optional, Union
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

class Vector():
    """A custom vector implementation with mapping capabilities.
    
    This class implements a vector (1D array) with additional functionality to map between
    indices and group IDs. It maintains internal mappings and provides methods for 
    initialization, updates, and value access.

    Attributes:
        size (int): Current size of the vector.
        id_map (Dict[int, int]): Maps vector indices to group IDs.
        id_map_reverse (Dict[int, int]): Maps group IDs to vector indices.
        array (np.ndarray): The underlying numpy array storing vector values.
        current_idx (int): Current index pointer for setting values.
        sorted_id_indices (np.ndarray): Indices that sort the vector by group IDs.
    """

    def __init__(self) -> None:
        """Initialize an empty Vector instance."""
        self.size = 0
        self.id_map = {}
        self.id_map_reverse = {}

    def __getitem__(self, key: int) -> float:
        """Get value at specified index.

        Args:
            key (int): Index to access.

        Returns:
            float: Value at specified index.
        """
        return self.array[key]
    
    def __setitem__(self, key: int, value: float) -> None:
        """Set value at specified index.

        Args:
            key (int): Index to set.
            value (float): Value to set.
        """
        self.array[key] = value

    def reset(self) -> None:
        """Reset the vector to initial empty state."""
        self.size = 0
        self.id_map = {}
        self.id_map_reverse = {}

    def initialize(self) -> None:
        """Initialize the vector array and sorting indices.
        
        Creates the underlying numpy array and computes indices for sorted access by group ID.
        """
        self.array = np.zeros(self.size)
        self.current_idx = 0
        id_array = np.empty(self.size, dtype=np.int64)
        for idx, group_id in self.id_map.items():
            id_array[idx] = group_id
        self.sorted_id_indices = np.argsort(id_array)

    def increment(self, v: int = 1) -> None:
        """Increment the vector size.

        Args:
            v (int, optional): Amount to increment by. Defaults to 1.
        """
        self.size += v

    def set(self, v: float) -> None:
        """Set the next value in the vector.

        Args:
            v (float): Value to set at current index.
        """
        self[self.current_idx] = v
        self.current_idx += 1
        if self.current_idx == self.array.size:
            self.current_idx = 0

    def get(self) -> np.ndarray:
        """Get vector values sorted by group ID.

        Returns:
            np.ndarray: Array of values sorted by group ID.
        """
        return self.array[self.sorted_id_indices]

    def update(self, group_id: Optional[int] = None) -> None:
        if group_id is None:
            group_id = self.size
        self.id_map_reverse[group_id] = self.size
        self.id_map[self.size] = group_id
        self.increment()

    def copy(self):
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

    def __init__(self, scalar: Optional[Union[float, int, np.ndarray]] = None) -> None:
        """Initialize a Scalar instance.

        Args:
            scalar (Optional[Union[float, int, np.ndarray]], optional): Initial scalar value. 
                Defaults to None.
        """
        self.scalar = scalar

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
            return self.scalar + other.scalar
        elif isinstance(other, (int, float)):
            return self.scalar + other
        elif isinstance(other, np.ndarray):
            return self.scalar + other
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
            return self.scalar - other.scalar
        elif isinstance(other, (int, float)):
            return self.scalar - other
        elif isinstance(other, np.ndarray):
            return self.scalar - other
        else:
            raise TypeError(f"unsupported operand type(s) for -: 'Scalar' and '{type(other)}'")
        
    def __rsub__(self, other: Union[Scalar, int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reverse subtract operation.

        Args:
            other (Union[Scalar, int, float, np.ndarray]): Value to subtract from.

        Returns:
            Union[float, np.ndarray]: Result of subtraction.
        """
        return self.__sub__(other)
    
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
            return self.scalar * other.scalar
        elif isinstance(other, (int, float)):
            return self.scalar * other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self.scalar * other
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
            return self.scalar / other.scalar
        elif isinstance(other, (int, float)):
            return self.scalar / other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self.scalar / other
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
            return self.scalar == other.scalar
        elif isinstance(other, (int, float)):
            return self.scalar == other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self.scalar == other
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
            return self.scalar < other.scalar
        elif isinstance(other, (int, float)):
            return self.scalar < other
        elif isinstance(other, (np.ndarray, np.generic)):
            return self.scalar < other
        else:
            raise TypeError(f"unsupported operand type(s) for <: 'Scalar' and '{type(other)}'")

    def __str__(self) -> str:
        """Get string representation of the scalar.

        Returns:
            str: String representation of the scalar value.
        """
        return str(self.scalar)
    
    def set(self, v: Union[Scalar, float]) -> None:
        """Set the scalar value.

        Args:
            v (Union[Scalar, float]): Value to set.
        """
        if isinstance(v, Scalar):
            self.scalar = v.get()
        else:
            self.scalar = v

    def get(self) -> float:
        """Get the scalar value.

        Returns:
            float: Scalar value.
        """
        return self.scalar

    def update(self, group_id: Optional[int] = None):
        pass

    def reset(self):
        pass
    
    def initialize(self):
        pass

    def copy(self):
        copy = Scalar()
        copy.scalar = self.scalar
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
