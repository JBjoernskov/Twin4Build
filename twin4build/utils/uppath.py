import os
def uppath(_path, n):
    return os.sep.join(_path.split(os.sep)[:-n])

