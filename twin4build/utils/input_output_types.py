import os
import sys
import datetime
from dateutil import tz
import numpy as np
import functools
from typing import Optional
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

class Vector():
    def __init__(self):
        self.size = 0
        self.id_map = {}
        self.id_map_reverse = {}

    def __getitem__(self, key):
        return self.array[key]
    
    def __setitem__(self, key, value):
        self.array[key] = value

    def initialize(self):
        self.array = np.empty(self.size)
        self.current_idx = 0
        id_array = np.empty(self.size)
        for idx, group_id in self.id_map.items():
            id_array[idx] = group_id
        self.sorted_id_indices = np.argsort(id_array)

    def increment(self, v=1):
        self.size += v

    def set(self, v): #should it test if v is an array?
        self[self.current_idx] = v
        self.current_idx += 1
        if self.current_idx == self.array.size:
            self.current_idx = 0

    def get(self):
        return self.array[self.sorted_id_indices]

    def update(self, group_id: Optional[int] = None):
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
class Scalar():
    def __init__(self, scalar=None):
        self.scalar = scalar

    def __add__(self, other):
        if isinstance(other, Scalar):
            return self.scalar + other.scalar
        elif isinstance(other, int) or isinstance(other, float):
            return self.scalar + other
        elif isinstance(other, np.ndarray):
            return self.scalar + other
        else:
            raise TypeError("unsupported operand type(s) for +: 'Scalar' and '{}'".format(type(other)))
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Scalar):
            return self.scalar - other.scalar
        elif isinstance(other, int) or isinstance(other, float):
            return self.scalar - other
        elif isinstance(other, np.ndarray):
            return self.scalar - other
        else:
            raise TypeError("unsupported operand type(s) for -: 'Scalar' and '{}'".format(type(other)))
        
    def __rsub__(self, other):
        return self.__sub__(other)
        
    # def __radd__(self, other):

    # def __sub__(self, other):
        
    # def __rsub__(self, other):
    
    # def __neg__(self):
    
    def __mul__(self, other):
        if isinstance(other, Scalar):
            return self.scalar * other.scalar
        elif isinstance(other, int) or isinstance(other, float):
            return self.scalar * other
        elif isinstance(other, np.ndarray) or isinstance(other, np.generic):
            return self.scalar * other
        else:
            raise TypeError("unsupported operand type(s) for *: 'Scalar' and '{}'".format(type(other)))
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Scalar):
            return self.scalar / other.scalar
        elif isinstance(other, int) or isinstance(other, float):
            return self.scalar / other
        elif isinstance(other, np.ndarray) or isinstance(other, np.generic):
            return self.scalar / other
        else:
            raise TypeError("unsupported operand type(s) for /: 'Scalar' and '{}'".format(type(other)))
    
    def __rtruediv__(self, other):
        return self.__div__(other)
    
    def __eq__(self, other):
        if isinstance(other, Scalar):
            return self.scalar == other.scalar
        elif isinstance(other, int) or isinstance(other, float):
            return self.scalar == other
        elif isinstance(other, np.ndarray) or isinstance(other, np.generic):
            return self.scalar == other
        else:
            raise TypeError("unsupported operand type(s) for ==: 'Scalar' and '{}'".format(type(other)))
        
    def __lt__(self, other):
        if isinstance(other, Scalar):
            return self.scalar < other.scalar
        elif isinstance(other, int) or isinstance(other, float):
            return self.scalar < other
        elif isinstance(other, np.ndarray) or isinstance(other, np.generic):
            return self.scalar < other
        else:
            raise TypeError("unsupported operand type(s) for <: 'Scalar' and '{}'".format(type(other)))
        


    def __str__(self):
        return str(self.v)
    
    def set(self, v):
        if isinstance(v, Scalar):
            self.scalar = v.get()
        else:
            self.scalar = v

    def get(self):
        return self.scalar

    def update(self, group_id: Optional[int] = None):
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
