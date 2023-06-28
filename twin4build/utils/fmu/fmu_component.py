from fmpy import read_model_description, extract, instantiate_fmu
from fmpy.fmi1 import FMU1Slave
from fmpy.fmi2 import FMU2Slave
import copy
import numpy as np
from twin4build.saref.device.sensor.sensor import Sensor
from twin4build.saref.device.meter.meter import Meter
from twin4build.logger.Logging import Logging
from twin4build.utils.rgetattr import rgetattr
logger = Logging.get_logger("ai_logfile")


class FMUComponent():
    def __init__(self, start_time=None, fmu_filename=None):

        logger.info("[FMU Component] : Entered in Initialise Function")

        self.model_description = read_model_description(fmu_filename)
        self.unzipdir = extract(fmu_filename)


        self.fmu = FMU2Slave(guid=self.model_description.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                    instanceName='FMUComponent')
        # fmi_type = 'CoSimulation' if self.model_description.coSimulation is not None else 'ModelExchange'
        # self.fmu = instantiate_fmu(unzipdir=self.unzipdir, model_description=self.model_description, fmi_type=fmi_type)
        # from .sundials import CVodeSolver
        # # common solver constructor arguments
        # solver_args = {
        # 'nx': model_description.numberOfContinuousStates,
        # 'nz': model_description.numberOfEventIndicators,
        # 'get_x': fmu.getContinuousStates,
        # 'set_x': fmu.setContinuousStates,
        # 'get_dx': fmu.getContinuousStateDerivatives if is_fmi3 else fmu.getDerivatives,
        # 'get_z': fmu.getEventIndicators,
        # 'input': input
        # }
        # solver = CVodeSolver(set_time=fmu.setTime,
        #                      startTime=start_time,
        #                      maxStep=(stop_time - start_time) / 50.,
        #                      relativeTolerance=relative_tolerance,
        #                      **solver_args)
        
        self.inputs = dict()

        self.fmu_variables = {variable.name:variable for variable in self.model_description.modelVariables}
        self.fmu_inputs = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="input"}
        self.fmu_outputs = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="output"}
        self.fmu_parameters = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="parameter"}
        self.fmu_calculatedparameters = {variable.name:variable for variable in self.model_description.modelVariables if variable.causality=="calculatedParameter"}
        self.parameters = {key: rgetattr(self, attr) for attr,key in self.FMUparameterMap.items()}
        self.FMUmap = {}
        self.FMUmap.update(self.FMUinputMap)
        self.FMUmap.update(self.FMUoutputMap)
        self.FMUmap.update(self.FMUparameterMap)
        self.component_stepSize = 60 #seconds
        self.fmu.instantiate()
        self.fmu.setDebugLogging(loggingOn=True, categories="logDynamicStateSelection")
        self.reset()

        temp_joined = {key_input: None for key_input in self.FMUinputMap.values()}
        temp_joined.update({key_input: None for key_input in self.FMUparameterMap.values()})
        self.localGradients = {key_output: copy.deepcopy(temp_joined) for key_output in self.FMUoutputMap.values()}
        self.localGradientsSaved = []
            
        logger.info("[FMU Component] : Exited from Initialise Function")

    def reset(self):
        # self.fmu = FMU2Slave(guid=self.model_description.guid,
        #             unzipDirectory=self.unzipdir,
        #             modelIdentifier=self.model_description.coSimulation.modelIdentifier,
        #             instanceName='FMUComponent')
        # self.fmu.instantiate()

        self.fmu.reset()
        self.set_parameters(self.parameters)

        self.fmu.setupExperiment(startTime=0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        self.inputUncertainty = copy.deepcopy(self.input)
        self.outputUncertainty = copy.deepcopy(self.output)

        if self.doUncertaintyAnalysis:
            temp_dict = copy.deepcopy(self.inputUncertainty)
            for connection_point in self.connectsAt:
                reciever_property_name = connection_point.recieverPropertyName
                connection = connection_point.connectsSystemThrough
                sender_property_name = connection.senderPropertyName
                sender_component = connection.connectsSystem

                if isinstance(sender_component, Sensor) or isinstance(sender_component, Meter):
                    property_ = sender_component.measuresProperty
                    if property_.MEASURING_TYPE=="P":
                        temp_dict[reciever_property_name] = False
                    else:
                        temp_dict[reciever_property_name] = True
                else:
                    temp_dict[reciever_property_name] = True
                                                
            self.uncertainty_type_mask = np.array([el for el in temp_dict.values()])

    def set_parameters(self, parameters):
        lookup_dict = self.fmu_parameters
        for key in parameters.keys():
            if key in lookup_dict:
                self.fmu.setReal([lookup_dict[key].valueReference], [parameters[key]])
            # else:
            #     self.fmu.setReal([self.calculatedparameters[key].valueReference], [parameters[key]])

    def get_gradient(self, x_key):
        y_ref = [val.valueReference for val in self.fmu_outputs.values()]
        x_ref = [self.fmu_inputs[x_key].valueReference]
        dv = [1]
        grad = self.fmu.getDirectionalDerivative(vUnknown_ref=y_ref, vKnown_ref=x_ref, dvKnown=dv)
        grad = np.array(grad)
        return grad


    def get_jacobian(self):
        # This code assumes that the order of <dict>.values() is always the same
        n = len(self.fmu_outputs)
        m = len(self.fmu_inputs)
        jac = np.zeros((n, m))
        for i, x_key in enumerate(self.fmu_inputs.keys()):
            grad = self.get_gradient(x_key, i)
            jac[:,i] = grad
        return jac

    def get_subset_gradient(self, x_key, y_keys=None, as_dict=False):
        if y_keys:
            y_ref = [self.fmu_outputs[self.FMUoutputMap[key]].valueReference for key in y_keys]
        else:
            y_ref = [self.fmu_outputs[key].valueReference for key in self.FMUoutputMap.values()]
        x_ref = [self.fmu_variables[self.FMUmap[x_key]].valueReference]
        dv = [1]
        grad = self.fmu.getDirectionalDerivative(vUnknown_ref=y_ref, vKnown_ref=x_ref, dvKnown=dv)
        
        if as_dict==False:
            grad = np.array(grad)
        else:
            grad = {key: value for key, value in zip(y_keys, grad)}

        return grad

    def get_subset_jacobian(self):
        # Only extract for self.input and self.output
        # This code assumes that the order of <dict>.values() is always the same 
        n = len(self.output)
        m = len(self.input)
        jac = np.zeros((n, m))
        for i, x_key in enumerate(self.input.keys()):
            grad = self.get_subset_gradient(x_key)
            jac[:,i] = grad
        return jac
    
    def do_uncertainty_analysis(self):
        inputs = list(self.input.values())
        inputs = np.array([inputs])
        input_uncertainty = list(self.inputUncertainty.values())
        input_uncertainty = np.array([input_uncertainty])
        jac = self.get_subset_jacobian()
        output_uncertainty = np.linalg.norm(jac*input_uncertainty*(self.uncertainty_type_mask + inputs*(self.uncertainty_type_mask==False)), axis=1)
        for key, uncertainty_value in zip(self.outputUncertainty.keys(), output_uncertainty):
            self.outputUncertainty[key] = uncertainty_value

        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        end_time = secondTime+stepSize
        for key in self.input.keys():
            x = self.input_unit_conversion[key](self.input[key])
            FMUkey = self.FMUinputMap[key]
            self.fmu.setReal([self.fmu_variables[FMUkey].valueReference], [x])

        # if isinstance(self, coil_FMUmodel.CoilModel):


        while secondTime<end_time:
            self.fmu.doStep(currentCommunicationPoint=secondTime, communicationStepSize=self.component_stepSize)
            secondTime += self.component_stepSize


        

        # if isinstance(self, coil_FMUmodel.CoilModel):
        #     print("---DO STEP---")
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.fm_w"].valueReference]))
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.fm_a"].valueReference]))
        #     print("hA_x")
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.x_w"].valueReference]))
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.x_a"].valueReference]))
        #     print("hA_nominal")
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.hA_nominal_w"].valueReference]))
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.hA_nominal_a"].valueReference]))
        #     print("hA")
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.hA_1"].valueReference]))
        #     print(self.fmu.getReal([self.fmu_variables["com.hA.hA_2"].valueReference]))
        #     print("UA")
        #     print(self.fmu.getReal([self.fmu_variables["com.UA"].valueReference]))
        #     x_key="T_a2_nominal"
        #     y_keys=["outletAirTemperature"]
        #     y_ref = [self.fmu_outputs[self.FMUoutputMap[key]].valueReference for key in y_keys]
        #     x_ref = [self.fmu_variables[x_key].valueReference]
        #     dv = [1]
        #     grad = self.fmu.getDirectionalDerivative(vUnknown_ref=y_ref, vKnown_ref=x_ref, dvKnown=dv)
        #     grad = {key: value for key, value in zip(y_keys, grad)}
        #     print(grad)
            
        # Currently only the values for the final timestep is saved.
        # Alternatively, the in-between values in the while loop could also be saved.
        # However, this would need adjustments in the "SimulationResult" class and the "update_simulation_result" method.
        for key in self.output.keys():
            FMUkey = self.FMUoutputMap[key]
            self.output[key] = self.output_unit_conversion[key](self.fmu.getReal([self.fmu_variables[FMUkey].valueReference])[0])

        if self.doUncertaintyAnalysis:
            self.do_uncertainty_analysis()

        ############################################################################################
        for output_key, input_dict in self.localGradients.items():
            for input_key, value in input_dict.items():
                y_ref = [self.fmu_outputs[output_key].valueReference]
                x_ref = [self.fmu_variables[input_key].valueReference]
                dv = [1]
                value = self.fmu.getDirectionalDerivative(vUnknown_ref=y_ref, vKnown_ref=x_ref, dvKnown=dv)
                self.localGradients[output_key][input_key] = value[0]
        self.localGradientsSaved.append(copy.deepcopy(self.localGradients))
        ############################################################################################



