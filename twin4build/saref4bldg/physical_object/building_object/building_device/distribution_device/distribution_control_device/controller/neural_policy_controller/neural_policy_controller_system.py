
"""
Neural policy controller for RL-based building control

Features:
- The controller is based on a neural network model that takes as input the current state of the building and outputs the control signal
- The neural network model is trained using reinforcement learning techniques to optimize building energy performance
- The input and output of the controller is defined by a JSON schema that contains the keys and types of the input and output signals
- The neural policy is initialized at instantiation and the weights are updated manually by the user, typically through a training process

"""
from twin4build.base import NeuralPolicyController
import sys
import os
import numpy as np
import torch.nn as nn
import torch
import datetime
import calendar
from pathlib import Path
import twin4build.utils.input_output_types as tps

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 9)
sys.path.append(file_path)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Using device: {device}')


class NeuralPolicyControllerSystem(NeuralPolicyController):
    def __init__(self, 
                input_size = None,
                output_size = None,
                input_output_schema = None,
                policy_model = None,
                **kwargs):
        super().__init__(**kwargs)

        assert input_size is not None, "Input size must be defined"
        assert output_size is not None, "Output size must be defined"
        self.input_size = input_size
        self.output_size = output_size

        assert input_output_schema is not None, "Input and output schema must be defined"
        self.input_output_schema = input_output_schema

        if policy_model is not None:
            self.model = policy_model
        else:
            self.model = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_size),
                nn.Sigmoid()
            ).to(device)


        #Input and output can be any arbitrary vector
        self.input = {"actualValue": tps.Vector()}
        self.output = {"inputSignal": tps.Vector()}
        self.device =  device
        self._config = {"parameters": ["input_size", "output_size"]}
    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass    

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        pass

    def normalize_input_data(self, data):
        normalized_data = []
        for key in self.input_output_schema["input"]:
            min_val = self.input_output_schema["input"][key]["min"]
            max_val = self.input_output_schema["input"][key]["max"]
            normalized_data.append((data - min_val) / (max_val - min_val))
        return normalized_data
    
    def denormalize_output_data(self, data):
        denormalized_data = []
        for key in self.input_output_schema["output"]:
            min_val = self.input_output_schema["output"][key]["min"]
            max_val = self.input_output_schema["output"][key]["max"]
            denormalized_data.append(data * (max_val - min_val) + min_val)
        return denormalized_data
    
    def load_policy_model(self, model_path):
        #Load the neural network model from a file
        self.model.load_state_dict(torch.load(model_path))

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        normalized_input = self.normalize_input_data(self.input["actualValue"].get())
        input_tensor = torch.tensor(normalized_input).float().to(self.device)
        with torch.no_grad():
            predicted = self.model(input_tensor).cpu().numpy()
        denormalized_output = self.denormalize_output_data(predicted)

        output_vector = tps.Vector()
        output_vector.increment(len(denormalized_output))  # Ensure the vector is the right size
        output_vector.initialize()
        for value in denormalized_output:
            output_vector.set(value)
        self.output["inputSignal"].set(output_vector)



