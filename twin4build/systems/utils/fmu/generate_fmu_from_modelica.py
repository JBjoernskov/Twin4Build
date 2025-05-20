
# from OMPython import ModelicaSystem
import os
import shutil
from twin4build.utils.uppath import uppath
from fmpy.util import compile_platform_binary
def generate_fmu_from_modelica(modelica_filename):
    model_name = os.path.split(modelica_filename)[-1].replace(".mo","")
    fmu_foldername = os.path.join(uppath(__file__, 1), "tests", model_name)
    if not os.path.exists(fmu_foldername):
        os.makedirs(fmu_foldername)
    os.chdir(fmu_foldername)
    # r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\Modelica\BuildingsLibrary\Buildings_Master\modelica-buildings\Buildings\package.mo"
    dependencies = ["C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/Modelica/FMUPreparedModels/EPlusFan.mo",
                    "Buildings",
                    "Modelica"]
    modelica_model = None#ModelicaSystem(fileName=modelica_filename, modelName=model_name, lmodel=dependencies, commandLineOptions="-d=aliasConflicts")#"--fmiFlags=s:cvode")# <- this does not work -- cannot load dll library
    modelica_model.convertMo2Fmu(version="2.0", fmuType="cs", includeResources=False)
    dir_name = f"{fmu_foldername}/{model_name}.fmutmp"
    output_filename = f"{fmu_foldername}/{model_name}"
    shutil.make_archive(output_filename, 'zip', dir_name)
    output_filename_zip = f"{output_filename}.zip"
    os.rename(output_filename_zip, output_filename_zip.replace(".zip", ".FMU"))
    compile_platform_binary(filename=output_filename_zip.replace(".zip", ".FMU"))



