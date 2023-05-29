
from OMPython import ModelicaSystem
import os
from twin4build.utils.uppath import uppath

def generate_fmu_from_modelica(modelica_filename):
    model_name = os.path.split(modelica_filename)[-1].replace(".mo","")
    fmu_foldername = os.path.join(uppath(__file__, 1), "tests", model_name)
    if not os.path.exists(fmu_foldername):
        os.makedirs(fmu_foldername)
    os.chdir(fmu_foldername)
    dependencies = ["Modelica",
                    r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\Modelica\BuildingsLibrary\Buildings 9.1.0\package.mo"]
    modelica_model = ModelicaSystem(fileName=modelica_filename, modelName=model_name, lmodel=dependencies, commandLineOptions="-d=aliasConflicts")#"--fmiFlags=s:cvode")# <- this does not work -- cannot load dll library
    modelica_model.convertMo2Fmu()




