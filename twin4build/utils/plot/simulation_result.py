import matplotlib.pyplot as plt
import math
import matplotlib.pylab as pylab
import matplotlib.dates as mdates

# from matplotlib.pyplot import cm
from itertools import cycle
import numpy as np
import seaborn as sns
import copy

global_colors = sns.color_palette("deep")
global_blue = global_colors[0]
global_orange = global_colors[1]
global_green = global_colors[2]
global_red = global_colors[3]
global_purple = global_colors[4]
global_brown = global_colors[5]
global_pink = global_colors[6]
global_grey = global_colors[7]
global_beis = global_colors[8]
global_sky_blue = global_colors[9]


class SimulationResult:
    def __init__(self,
                savedInput=None,
                savedOutput=None,
                savedInputUncertainty=None,
                savedOutputUncertainty=None,
                savedParameterGradient=None,
                saveSimulationResult=True,
                doUncertaintyAnalysis=False,
                trackGradient=False,
                **kwargs):
        if savedInput is None:
            savedInput = {}
        if savedOutput is None:
            savedOutput = {}
        if savedInputUncertainty is None:
            savedInputUncertainty = {}
        if savedOutputUncertainty is None:
            savedOutputUncertainty = {}
        if savedParameterGradient is None:
            savedParameterGradient = {}
        self.savedInput = savedInput 
        self.savedOutput = savedOutput 
        self.savedInputUncertainty = savedInputUncertainty 
        self.savedOutputUncertainty = savedOutputUncertainty 
        self.saveSimulationResult = saveSimulationResult
        self.doUncertaintyAnalysis = doUncertaintyAnalysis
        self.trackGradient = trackGradient
        
    def clear_results(self):
        self.savedInput = {}
        self.savedOutput = {}
        self.savedInputUncertainty = {}
        self.savedOutputUncertainty = {}
        self.savedOutputGradient = {}
        self.savedParameterGradient = {}
        
    def update_results(self):
        if self.saveSimulationResult:
            for key in self.input:
                if key not in self.savedInput:
                    self.savedInput[key] = [self.input[key].get()]
                else:
                    self.savedInput[key].append(self.input[key].get())
                
            for key in self.output:
                if key not in self.savedOutput:
                    self.savedOutput[key] = [self.output[key].get()]
                else:
                    self.savedOutput[key].append(copy.deepcopy(self.output[key].get()))

            if self.doUncertaintyAnalysis:
                for key in self.inputUncertainty:
                    if key not in self.savedInputUncertainty:
                        self.savedInputUncertainty[key] = [self.inputUncertainty[key]]
                    else:
                        self.savedInputUncertainty[key].append(self.inputUncertainty[key])
                    
                for key in self.outputUncertainty:
                    if key not in self.savedOutputUncertainty:
                        self.savedOutputUncertainty[key] = [self.outputUncertainty[key]]
                    else:
                        self.savedOutputUncertainty[key].append(self.outputUncertainty[key])
            
            if self.trackGradient:
                for measuring_device, attr_dict in self.parameterGradient.items():
                    if measuring_device not in self.savedParameterGradient:
                            self.savedParameterGradient[measuring_device] = {}
                    for attr, value in attr_dict.items():
                        if attr not in self.savedParameterGradient[measuring_device]:
                            self.savedParameterGradient[measuring_device][attr] = [value]
                        else:
                            self.savedParameterGradient[measuring_device][attr].append(value)
