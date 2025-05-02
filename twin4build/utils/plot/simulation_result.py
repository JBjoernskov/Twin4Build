import copy
import twin4build.utils.input_output_types as tps
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
        
    def clear_results(self):
        self.savedInput = {}
        self.savedOutput = {}
        
    def update_results(self):
        if self.saveSimulationResult:
            for key in self.input:
                if isinstance(self.input[key], tps.Vector):
                    v = None
                elif isinstance(self.input[key], tps.Scalar):
                    v = self.input[key].get_float()
                else:
                    raise ValueError(f"Input {key} is not a Vector or Scalar")
                
                if key not in self.savedInput:
                    self.savedInput[key] = [v]
                else:
                    self.savedInput[key].append(v)
                
            for key in self.output:
                if isinstance(self.output[key], tps.Vector):
                    v = None
                elif isinstance(self.output[key], tps.Scalar):
                    v = self.output[key].get_float()
                else:
                    raise ValueError(f"Output {key} is not a Vector or Scalar")
                
                if key not in self.savedOutput:
                    self.savedOutput[key] = [v]
                else:
                    self.savedOutput[key].append(v)
