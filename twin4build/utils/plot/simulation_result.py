import copy
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
