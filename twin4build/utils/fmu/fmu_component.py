from fmpy import read_model_description, extract
from fmpy.fmi1 import FMU1Slave
from fmpy.fmi2 import FMU2Slave
import numpy as np
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

        self.CALC_Y_RANGE = False

    def reset(self):
        self.fmu = FMU2Slave(guid=self.model_description.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                    instanceName='FMUComponent')
        self.fmu.instantiate()


        # self.fmu.reset()
        self.set_parameters(self.initialParameters)

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

    def get_gradient(self, x_key, i=None):
        if i is None:
            i = list(self.fmu_inputs.keys()).index(x_key)
        y_ref = [var.valueReference for var in self.fmu_outputs.values()]
        x_ref = [self.fmu_inputs[x_key].valueReference]
        dv = [0]*len(self.fmu_inputs)
        dv[i] = 1
        grad = self.fmu.getDirectionalDerivative(vUnknown_ref=y_ref, vKnown_ref=x_ref, dvKnown=dv)
        grad = np.array(grad)
        return grad

    def get_full_jacobian(self):
        # This code assumes that the order of <dict>.values() is always the same 
        n = len(self.fmu_outputs)
        m = len(self.fmu_inputs)
        jac = np.zeros((n, m))
        for i, x_key in enumerate(self.fmu_inputs.keys()):
            grad = self.get_gradient(x_key, i)
            jac[:,i] = grad
        return jac
    
    def get_subset_jacobian(self):
        # Only extract for self.input and self.output
        # This code assumes that the order of <dict>.values() is always the same 
        n = len(self.output)
        m = len(self.input)
        jac = np.zeros((n, m))
        for i, x_key in enumerate(self.input.keys()):
            FMUkey = self.FMUinput[x_key]
            grad = self.get_gradient(FMUkey)
            jac[:,i] = grad[self.subset_mask]
        return jac
    
    def do_error_analysis(self):
        inputs = list(self.input.values())
        inputs = np.array([inputs])
        jac = self.get_subset_jacobian()
        self.output_range.append(np.linalg.norm(jac*self.input_A_range+jac*inputs*self.input_P_range, axis=1))
        

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        end_time = secondTime+stepSize
        for key in self.input.keys():
            x = self.input_unit_conversion[key](self.input[key])
            FMUkey = self.FMUinput[key]
            self.fmu.setReal([self.variables[FMUkey].valueReference], [x])

        while secondTime<end_time:
            self.fmu.doStep(currentCommunicationPoint=secondTime, communicationStepSize=self.component_stepSize)
            secondTime += self.component_stepSize

        # Currently only the values for the final timestep is saved.
        # Alternatively, the in-between values in the while loop could also be gathered.
        # However, this would need adjustments in the "SimulationResult" class and the "update_report" method.
        for key in self.output.keys():
            FMUkey = self.FMUoutput[key]
            self.output[key] = self.output_unit_conversion[key](self.fmu.getReal([self.variables[FMUkey].valueReference])[0])

        if self.CALC_Y_RANGE:
            self.do_error_analysis()




