import unittest
from twin4build.utils.fmu.generate_fmu_from_modelica import generate_fmu_from_modelica

@unittest.skipIf(True, 'Currently not used')
def test_generate_fmu_from_modelica():
    modelica_filename = "C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels/EPlusFan_FMU_test2port.mo"
    generate_fmu_from_modelica(modelica_filename)



