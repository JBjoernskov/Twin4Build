from twin4build.utils.rgetattr import rgetattr

def do_nothing(x, stepSize=None):
    return x

def change_sign(x, stepSize=None):
    return -x

def to_degC_from_degK(K, stepSize=None):
    return K-273.15

def to_degK_from_degC(C, stepSize=None):
    return C+273.15

class multiply():
    def __init__(self, factor):
        self.factor = factor
    def call(self, x, stepSize=None):
        return x*self.factor
    __call__ = call

class regularize():
    def __init__(self, limit):
        self.limit = limit
    def call(self, x, stepSize=None):
        return self.limit if x<self.limit else x
    __call__ = call

class add_attr():
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        
    def call(self, x, stepSize=None):
        return x+rgetattr(self.obj, self.attr)
    __call__ = call

class add():
    def __init__(self, obj, idx, conversion=do_nothing):
        if isinstance(idx, tuple)==False:
            idx = (idx,)
        if isinstance(obj, tuple)==False:
            obj = tuple([obj for i in idx])

        self.obj = obj
        self.idx = idx
        self.conversion = conversion
        
    def call(self, x, stepSize=None):
        y = 0
        for obj_, idx_ in zip(self.obj, self.idx):
            y += self.conversion(obj_[idx_])
        return y
    __call__ = call

class get():
    def __init__(self, obj, idx, conversion=do_nothing):
        self.obj = obj
        self.idx = idx
        self.conversion = conversion
        
    def call(self, x, stepSize=None):
        return self.conversion(self.obj[self.idx])
    __call__ = call

class integrate():
    def __init__(self, obj, idx, conversion=do_nothing):
        self.v = 0
        self.obj = obj
        self.idx = idx
        self.conversion = conversion
        
    def call(self, x, stepSize=None):
        self.v += self.conversion(self.obj[self.idx])*stepSize
        return self.v
    __call__ = call