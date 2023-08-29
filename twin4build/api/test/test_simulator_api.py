
import sys
import os
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)

from twin4build.api.codes.ml_layer.simulator_api import execute_methods


em = execute_methods()

em.temp_run_simulation()