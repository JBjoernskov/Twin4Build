from fmpy import read_model_description, extract
from fmpy.fmi1 import FMU1Slave
from fmpy.fmi2 import FMU2Slave
class FMUComponent():
    def __init__(self, start_time=None, fmu_filename=None):
        self.model_description = read_model_description(fmu_filename)
        self.unzipdir = extract(fmu_filename)


        self.fmu = FMU2Slave(guid=self.model_description.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                    instanceName='FMUComponent')

        self.inputs = dict()

        self.variables = {variable.name:variable for variable in self.model_description.modelVariables}
        self.fmu_inputs = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="input"}
        self.fmu_outputs = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="output"}
        self.parameters = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="parameter"}
        self.calculatedparameters = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="calculatedParameter"}

        self.component_stepSize = 60 #seconds
        self.fmu.instantiate()
        self.reset()

        self.results = dict()
        for key in self.variables.keys():
            self.results[key] = []

    def reset(self):
        # self.fmu = FMU2Slave(guid=self.model_description.guid,
        #             unzipDirectory=self.unzipdir,
        #             modelIdentifier=self.model_description.coSimulation.modelIdentifier,
        #             instanceName='FMUComponent')
        # self.fmu.instantiate()


        self.fmu.reset()
        self.set_parameters(self.initialParameters)
        print(self.initialParameters)

        self.fmu.setupExperiment(startTime=0)
        self.fmu.enterInitializationMode()
        # p = {"inlet1.m_flow": 1}
        # self.set_parameters(p)
        self.fmu.exitInitializationMode()

    def set_parameters(self, parameters):
        lookup_dict = self.parameters
        for key in parameters.keys():
            if key in lookup_dict:
                self.fmu.setReal([lookup_dict[key].valueReference], [parameters[key]])
            else:
                self.fmu.setReal([self.calculatedparameters[key].valueReference], [parameters[key]])

        

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        end_time = secondTime+stepSize
        for key in self.input.keys():
            FMUkey = self.FMUinput[key]
            self.fmu.setReal([self.variables[FMUkey].valueReference], [self.input[key]])

        while secondTime<end_time:
            self.fmu.doStep(currentCommunicationPoint=secondTime, communicationStepSize=self.component_stepSize)
            secondTime += self.component_stepSize

        # Currently only the values for the final timestep is saved.
        # Alternatively, the in-between values in the while loop could also be gathered.
        # However, this would need adjustments in the "SimulationResult" class and the "update_report" method.
        for key in self.output.keys():
            FMUkey = self.FMUoutput[key]
            self.output[key] = self.fmu.getReal([self.variables[FMUkey].valueReference])[0]

    