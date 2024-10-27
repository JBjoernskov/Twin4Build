
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
import torch.nn as nn
import torch
import twin4build.utils.input_output_types as tps
from twin4build.model.model import Model
uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 9)
sys.path.append(file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TODO Add signature pattern

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
        try:
            self.validate_schema(input_output_schema)
        except (TypeError, ValueError) as e:
            print("Validation error:", e)

        self.input_output_schema = input_output_schema

        if policy_model is not None:
            self.model = policy_model
        else:
            self.model = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
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
        self.model.load_state_dict(torch.load(model_path))

    def validate_schema(data):
        if not isinstance(data, dict):
            raise TypeError("Data should be a dictionary.")
        for main_key in ["input", "output"]:
            if main_key not in data:
                raise ValueError(f"'{main_key}' key is required in the data.")
            if not isinstance(data[main_key], dict):
                raise TypeError(f"'{main_key}' should be a dictionary.")
            for param, param_data in data[main_key].items():
                if not isinstance(param_data, dict):
                    raise TypeError(f"Each parameter under '{main_key}' should be a dictionary.")
                required_keys = {"min": float, "max": float, "description": str}
                for key, expected_type in required_keys.items():
                    if key not in param_data:
                        raise ValueError(f"'{key}' key is required for '{param}' in '{main_key}'.")
                    
                    if not isinstance(param_data[key], expected_type):
                        raise TypeError(
                            f"'{key}' in '{param}' under '{main_key}' should be of type {expected_type.__name__}."
                        )
                if param_data["min"] > param_data["max"]:
                    raise ValueError(
                        f"'min' value should be <= 'max' for '{param}' in '{main_key}'."
                    )
        #print("Data is valid.")

    def insert_neural_policy(self, model:Model, input_output_dictionary, policy_path):
        """
        The input/output dictionary contains information on the input and output signals of the controller.
        These signals must match the component and signal keys to replace in the model
        The input dictionary will have items like this:
            "component_key": {
                "component_output_signal_key": {
                    "min": 0,
                    "max": 1,
                    "description": "Description of the signal"
                }
            }
        Whilst the output items will have a similar structure but for the output signals:
            "component_key": {
                "component_input_signal_key": {
                    "min": 0,
                    "max": 1,
                    "description": "Description of the signal"
                }
            }
        Note that the input signals must contain the key for the output compoenent signal and the output signals must contain the key for the input component signal

        This function instantiates the controller and adds it to the model.
        Then it goes through the input dictionary adding connection to the input signals
        Then it goes through the output dictionary finding the corresponding existing connections, deleting the existing connections and adding the new connections
        """
        try:
            self.validate_schema(input_output_dictionary)
        except (TypeError, ValueError) as e:
            print("Validation error:", e)
            return
        #Create the controller
        input_size = len(input_output_dictionary["input"])
        output_size = len(input_output_dictionary["output"])

        policy = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        ).to(device)

        #Load the policy model
        policy.load_state_dict(torch.load(policy_path))

        neural_policy_controller = NeuralPolicyControllerSystem(
            input_size = input_size,
            output_size = output_size,
            input_output_schema = input_output_dictionary,
            policy_model = policy
        )
        
        model._add_component(neural_policy_controller)

        #Add the input connections
        for component_key in input_output_dictionary["input"]:
            for signal_key in input_output_dictionary["input"][component_key]:
                model._add_connection(
                    component_key,
                    signal_key,
                    neural_policy_controller,
                    "actualValue"
                )

        #Find and remove the existing output connections
     
        for output_component_key in input_output_dictionary["output"]:
            receiving_component = model.component_dict[output_component_key]
            found = False  
            for connection in receiving_component.connectedThrough:
                for connection_point in connection.connectsSystemAt:
                    if connection_point.receiverPropertyName == input_output_dictionary["output"][output_component_key]["signal_key"]:
                        connected_component = connection_point.connectionPointOf
                        model.remove_connection(receiving_component, connected_component, connection.senderPropertyName, connection_point.receiverPropertyName)
                        found = True 
                        break
                if found:
                    break 
            
            if not found:
                print(f"Could not find connection for {output_component_key} and {input_output_dictionary['output'][output_component_key]['signal_key']}")
        
        
        #Add the output connections
        for component_key in input_output_dictionary["output"]:
            for signal_key in input_output_dictionary["output"][component_key]:
                model._add_connection(
                    neural_policy_controller,
                    "inputSignal",
                    component_key,
                    signal_key
                )
        
        return model


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



