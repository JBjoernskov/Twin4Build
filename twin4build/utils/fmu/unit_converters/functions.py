from twin4build.utils.rgetattr import rgetattr

def do_nothing(x):
    return x

def change_sign(x):
    return -x

def to_degC_from_degK(K):
    return K-273.15

def to_degK_from_degC(C):
    return C+273.15

class regularize():
    def __init__(self, limit):
        self.limit = limit
    def call(self, x):
        return self.limit if x<self.limit else x
    __call__ = call

class add():
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        
    def call(self, x):
        return x+rgetattr(self.obj, self.attr)
    __call__ = call

class get():
    def __init__(self, obj, idx, conversion=do_nothing):
        self.obj = obj
        self.idx = idx
        self.conversion = conversion
        
    def call(self, x):
        return self.conversion(self.obj[self.idx])
    __call__ = call