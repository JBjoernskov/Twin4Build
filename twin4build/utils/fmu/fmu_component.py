from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave

import os
import sys

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class FMUComponent():
    def __init__(self, start_time=None, fmu_filename=None):

        logger.info("[FMU Component] : Entered in Initialise Function")

        self.model_description = read_model_description(fmu_filename)
        unzipdir = extract(fmu_filename)


        self.fmu = FMU2Slave(guid=self.model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                    instanceName='FMUComponent')

        self.inputs = dict()

        self.variables = {variable.name:variable for variable in self.model_description.modelVariables}
        self.fmu_inputs = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="input"}
        self.fmu_outputs = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="output"}
        self.parameters = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="parameter"}
        self.calculatedparameters = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="calculatedParameter"}

        self.component_stepSize = 600 #seconds
        self.fmu.instantiate()
        self.reset()

        self.results = dict()
        for key in self.variables.keys():
            self.results[key] = []
            
        logger.info("[FMU Component] : Exited from Initialise Function")


    def reset(self):
        # self.fmu.instantiate()
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def set_parameters(self, parameters):
        for key in parameters.keys():
            if key in self.parameters:
                self.fmu.setReal([self.parameters[key].valueReference], [parameters[key]])
            else:
                self.fmu.setReal([self.calculatedparameters[key].valueReference], [parameters[key]])

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        end_time = secondTime+stepSize
        
        for key in self.input.keys():
            self.fmu.setReal([self.variables[key].valueReference], [self.input[key]])
        while secondTime<end_time:
            self.fmu.doStep(currentCommunicationPoint=secondTime, communicationStepSize=self.component_stepSize)
            secondTime += self.component_stepSize

        # Currently only the values for the final timestep is saved.
        # Alternatively, the in-between values in the while loop could also be gathered.
        # However, this would need adjustments in the "SimulationResult" class and the "update_report" method.
        for key in self.fmu_outputs.keys():
            self.output[key] = self.fmu.getReal([self.variables[key].valueReference])[0]

    