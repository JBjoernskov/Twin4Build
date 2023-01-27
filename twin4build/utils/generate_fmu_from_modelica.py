
from OMPython import OMCSessionZMQ
from OMPython import ModelicaSystem
import os
from uppath import uppath

def generate_fmu_from_modelica(modelica_filename):
    # omc = OMCSessionZMQ()

    # omc.sendExpression('loadModel(modelica_filename)')
    # fmu_filename = modelica_filename.replace(".mo", ".fmu")
    # model_path=omc.sendExpression("getInstallationDirectoryPath()")
    model_name = os.path.split(modelica_filename)[-1].replace(".mo","")
    fmu_foldername = os.path.join(uppath(__file__, 1), model_name)
    if not os.path.exists(fmu_foldername):
        os.makedirs(fmu_foldername)
    os.chdir(fmu_foldername)
    dependencies = ["Modelica",
                    r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\Modelica\BuildingsLibrary\Buildings 9.1.0\package.mo"]
    modelica_model = ModelicaSystem(fileName=modelica_filename, modelName=model_name, lmodel=dependencies)#,commandLineOptions="--fmiFlags=s:cvode") <- this does not work -- cannot load dll library
    # commandLineOptions = "--fmiFlags="
    # exp="".join(["setCommandLineOptions(","\"",commandLineOptions,"\"",")"])
    # modelica_model.getconn.sendExpression(exp)
    modelica_model.convertMo2Fmu()
    # exp="".join(["getCommandLineOptions()"])
    # aa = modelica_model.getconn.sendExpression(exp)
    # print(aa)


if __name__=="__main__":
    modelica_filename = r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\Modelica\FMUPreparedModels\Radiator.mo"
    generate_fmu_from_modelica(modelica_filename)

