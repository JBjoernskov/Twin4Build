# Standard library imports
import datetime

# Third party imports
import numpy as np
import torch
import torch.nn as nn

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO Add signature pattern


class NeuralPolicyControllerSystem(core.System):
    """
    Neural policy controller for RL-based building control.

    This class implements a neural network-based controller that uses reinforcement learning
    techniques to optimize building energy performance. The controller takes the current
    state of the building as input and outputs control signals based on a trained policy.

    Features:
        - The controller is based on a neural network model that takes as input the current
          state of the building and outputs the control signal
        - The neural network model is trained using reinforcement learning techniques to
          optimize building energy performance
        - The input and output of the controller is defined by a JSON schema that contains
          the keys and types of the input and output signals
        - The neural policy is initialized at instantiation and the weights are updated
          manually by the user, typically through a training process

    Args:
        input_size (int): Size of the input state vector
        output_size (int): Size of the output control vector
        input_output_schema (dict): JSON schema defining input/output structure and ranges
        policy_model (nn.Module, optional): Pre-trained neural network policy. If None,
            a default architecture is created.
        **kwargs: Additional keyword arguments passed to the parent System class
    """

    def __init__(
        self,
        input_size=None,
        output_size=None,
        input_output_schema=None,
        policy_model=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert input_size is not None, "Input size must be defined"
        assert output_size is not None, "Output size must be defined"
        self.input_size = input_size
        self.output_size = output_size

        assert (
            input_output_schema is not None
        ), "Input and output schema must be defined"
        try:
            self.validate_schema(input_output_schema)
        except (TypeError, ValueError) as e:
            print("Validation error:", e)

        self.input_output_schema = input_output_schema

        self.is_training = False

        if policy_model is not None:
            self.policy = policy_model
        else:
            self.policy = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_size),
                nn.Sigmoid(),
            ).to(device)

        # Initialize input
        self.input = {"actualValue": tps.Vector()}

        # Initialize output based on schema
        self.output = {}
        for output_key in self.input_output_schema["output"]:
            self.output[output_key] = tps.Scalar()

        self.device = device
        self._config = {"parameters": ["input_size", "output_size"]}

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        pass

    def normalize_input_data(self, data):
        normalized_data = []
        for key in self.input_output_schema["input"]:
            min_val = self.input_output_schema["input"][key]["min"]
            max_val = self.input_output_schema["input"][key]["max"]
            normalized_data.append((data - min_val) / (max_val - min_val))
        return normalized_data

    def denormalize_output_data(self, data):
        """
        Denormalize the output data using the schema.
        Inputs: data (numpy array or tensor of shape (output_size,))
        Outputs: denormalized data (numpy array)
        The min and max values are stored in the input_output_schema["output"] dictionary.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        keys = list(self.input_output_schema["output"].keys())
        min_vals = np.array(
            [self.input_output_schema["output"][key]["min"] for key in keys]
        )
        max_vals = np.array(
            [self.input_output_schema["output"][key]["max"] for key in keys]
        )
        denormalized_data = data * (max_vals - min_vals) + min_vals
        return denormalized_data

    def load_policy_model(self, policy_path):
        self.policy.load_state_dict(torch.load(policy_path))

    def validate_schema(self, data):
        if not isinstance(data, dict):
            raise TypeError("Data should be a dictionary.")
        for main_key in ["input", "output"]:
            if main_key not in data:
                raise ValueError(f"'{main_key}' key is required in the data.")
            if not isinstance(data[main_key], dict):
                raise TypeError(f"'{main_key}' should be a dictionary.")
            for param, param_data in data[main_key].items():
                if not isinstance(param_data, dict):
                    raise TypeError(
                        f"Each parameter under '{main_key}' should be a dictionary."
                    )
                required_keys = {
                    "min": (float, int),
                    "max": (float, int),
                    "description": str,
                }
                for key, expected_type in required_keys.items():
                    if key not in param_data:
                        raise ValueError(
                            f"'{key}' key is required for '{param}' in '{main_key}'."
                        )

                    if not isinstance(param_data[key], expected_type):
                        raise TypeError(
                            f"'{key}' in '{param}' under '{main_key}' should be of type {expected_type.__name__}."
                        )
                if param_data["min"] > param_data["max"]:
                    raise ValueError(
                        f"'min' value should be <= 'max' for '{param}' in '{main_key}'."
                    )
        # print("Data is valid.")

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        if self.is_training:
            action = dist.sample()
        else:
            action = mean
        action_logprob = dist.log_prob(action).sum()
        return action.numpy(), action_logprob.numpy()

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        normalized_input = self.normalize_input_data(self.input["actualValue"].get())
        state = torch.tensor(normalized_input).float().to(self.device)
        action, action_logprob = self.select_action(state)
        denormalized_output = self.denormalize_output_data(action)

        # The resulting denormalized output follows the same order as the input schema,
        for idx, key in enumerate(self.input_output_schema["output"]):
            output_key = key + "_input_signal"
            self.output[output_key].set(denormalized_output[idx], stepIndex)
