# Standard library imports
import datetime
import os
import time
from ctypes import byref
from typing import Optional

# Third party imports
import fmpy.fmi2 as fmi2
import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMICallException, FMU2Slave

# Local application imports
import twin4build.core as core
from twin4build.utils.do_nothing import do_nothing
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.rgetattr import rgetattr


def unzip_fmu(fmu_path=None, unzipdir=None):
    model_description = read_model_description(fmu_path)
    if unzipdir is None:
        filename = os.path.basename(fmu_path)
        filename, ext = os.path.splitext(filename)
        foldername, isfile = mkdir_in_root(folder_list=["generated_files", "fmu"])
        unzipdir = os.path.join(foldername, f"{filename}_temp_dir")

    if os.path.isdir(unzipdir):
        extracted_model_description = read_model_description(
            os.path.join(unzipdir, "modelDescription.xml")
        )
        # Validate guid. If the already extracted FMU guid does not match the FMU guid, extract again.
        if model_description.guid != extracted_model_description.guid:
            unzipdir = extract(fmu_path, unzipdir=unzipdir)
    else:
        unzipdir = extract(fmu_path, unzipdir=unzipdir)
    return unzipdir


class fmuSystem(core.System):
    # This init function is not safe for multiprocessing
    def __init__(self, fmu_path=None, unzipdir=None, **kwargs):
        self.fmu_path = fmu_path
        self.unzipdir = unzipdir
        self.model_description = None  # Because of memory leak
        super().__init__(**kwargs)

    def initialize_fmu(self):
        # if self.model_description is None: # Because of memory leak
        model_description = read_model_description(self.fmu_path)
        self.fmu = FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName=self.id,
        )

        self.inputs = dict()

        self.fmu_variables = {
            variable.name: variable for variable in model_description.modelVariables
        }
        self.fmu_inputs = {
            variable.name: variable
            for variable in model_description.modelVariables
            if variable.causality == "input"
        }
        self.fmu_outputs = {
            variable.name: variable
            for variable in model_description.modelVariables
            if variable.causality == "output"
        }
        self.fmu_parameters = {
            variable.name: variable
            for variable in model_description.modelVariables
            if variable.causality == "parameter"
        }

        self.FMUmap = {}
        self.FMUmap.update(self.FMUinputMap)
        self.FMUmap.update(self.FMUoutputMap)
        self.FMUmap.update(self.FMUparameterMap)
        self.component_stepSize = 600  # seconds

        debug_fmu_errors = False

        n_try = 5
        for i in range(n_try):  # Try 5 times to instantiate the FMU
            try:
                callbacks = fmi2.fmi2CallbackFunctions()
                if debug_fmu_errors:
                    callbacks.logger = fmi2.fmi2CallbackLoggerTYPE(fmi2.printLogMessage)
                else:
                    callbacks.logger = fmi2.fmi2CallbackLoggerTYPE(do_nothing)
                callbacks.allocateMemory = fmi2.fmi2CallbackAllocateMemoryTYPE(
                    fmi2.calloc
                )
                callbacks.freeMemory = fmi2.fmi2CallbackFreeMemoryTYPE(fmi2.free)
                if debug_fmu_errors:
                    try:
                        fmi2.addLoggerProxy(byref(callbacks))
                    except Exception as e:
                        print("Failed to add logger proxy function. %s" % e)
                self.fmu.instantiate(callbacks=callbacks)
                self.fmu.setupExperiment(startTime=0)
                self.fmu.enterInitializationMode()
                self.fmu.exitInitializationMode()
                break
            except:
                print(
                    f'Failed to instantiate "{self.id}" FMU. Trying again {str(i+1)}/{str(n_try)}...'
                )
                sleep_time = np.random.uniform(0.1, 1)
                time.sleep(sleep_time)
                if i == n_try - 1:
                    raise Exception("Failed to instantiate FMU.")

        self.fmu_initial_state = self.fmu.getFMUState()
        self.reset()

    def reset(self):
        self.fmu.setFMUState(self.fmu_initial_state)
        self.set_parameters()

    def set_parameters(self, parameters=None):
        lookup_dict = self.fmu_parameters
        if parameters is None:
            parameters = {
                key: rgetattr(self, attr) for attr, key in self.FMUparameterMap.items()
            }  # Update to newest parameters
        self.parameters = parameters
        for key in parameters.keys():
            if key in lookup_dict:
                assert (
                    parameters[key] is not None
                ), f'|CLASS: {self.__class__.__name__}|ID: {self.id}|: "{key}" is None.'
                self.fmu.setReal([lookup_dict[key].valueReference], [parameters[key]])

    def _do_step(self, secondTime=None, dateTime=None, stepSize=None, stepIndex=None):
        for key in self.FMUinputMap.keys():
            x = self.input_conversion[key](self.input[key].get(), stepSize=stepSize)
            FMUkey = self.FMUinputMap[key]
            self.fmu.setReal([self.fmu_variables[FMUkey].valueReference], [x])

        self.fmu.doStep(
            currentCommunicationPoint=secondTime, communicationStepSize=stepSize
        )

        # Currently only the values for the final timestep is saved.
        # Alternatively, the in-between values in the while loop could also be saved.
        # However, this would need adjustments in the "SimulationResult" class and the "update_simulation_result" method.
        for key in self.FMUoutputMap.keys():
            FMUkey = self.FMUmap[key]
            self.output[key].set(
                self.fmu.getReal([self.fmu_variables[FMUkey].valueReference])[0],
                stepIndex,
            )

        for key in self.output.keys():
            if key in self.output_conversion:
                self.output[key].set(
                    self.output_conversion[key](
                        self.output[key].get(), stepSize=stepSize
                    ),
                    stepIndex,
                )

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        try:
            self._do_step(
                secondTime=secondTime,
                dateTime=dateTime,
                stepSize=stepSize,
                stepIndex=stepIndex,
            )
        except FMICallException as inst:
            self.fmu.freeFMUState(self.fmu_initial_state)
            self.fmu.freeInstance()
            self.INITIALIZED = False
            raise (inst)
