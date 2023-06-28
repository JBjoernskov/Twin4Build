import os
import sys

if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
from twin4build.utils.fmu.generate_fmu_from_modelica import generate_fmu_from_modelica
if __name__=="__main__":
    modelica_filename = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels/EPlusFan_FMU_test2port.mo"
    generate_fmu_from_modelica(modelica_filename)



