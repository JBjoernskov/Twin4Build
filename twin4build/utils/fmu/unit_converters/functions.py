def do_nothing(x):
    return x

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