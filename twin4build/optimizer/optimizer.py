# import pygad
import twin4build.core as core
import torch
import datetime
from typing import Dict, List, Union
import torch.nn as nn
import twin4build.systems as systems

"""
An Optimizer class will be implemented here.

The optimizer optimizes operation of the building e.g. by changing setpoints.

Repeat 1-3 until convergence or stop criteria

1. Set inputs
2. Run simulation with inputs
3. Calculate loss (maybe operational costs or energy use)



Use of Genetic method seems promising 
https://github.com/ahmedfgad/GeneticAlgorithmPython

Monte Carlo discrete optimization could also be used. 
"Monte-Carlo Tree Search for Policy Optimization"
https://www.ijcai.org/proceedings/2019/0432.pdf



"""
def min_max_normalize(x, min_val=None, max_val=None):
    if min_val is None:
        min_val = torch.min(x)
    if max_val is None:
        max_val = torch.max(x)
    return (x - min_val) / (max_val - min_val)

class Optimizer():
    def __init__(self, simulator: core.Simulator):
        self.simulator = simulator

    def closure(self):
        self.optimizer.zero_grad()

        # Run simulation
        self.simulator.simulate(
            self.simulator.model,
            startTime=self.startTime,
            endTime=self.endTime,
            stepSize=self.stepSize,
            show_progress_bar=False
        )

        self.loss = 0
        k = 100

        # Handle equality constraints
        if self.equalityConstraints is not None:
            for component, (output_name, constraint_type, desired_value) in self.equalityConstraints.items():
                y = component.output[output_name].history
                desired_tensor = self.equality_constraint_values[component]

                min_val = torch.min(y)
                max_val = torch.max(y)
                y = min_max_normalize(y, min_val, max_val)
                desired_tensor = min_max_normalize(desired_tensor, min_val, max_val)
                
                self.loss += torch.mean((y - desired_tensor)**2)

        # Handle inequality constraints
        if self.inequalityConstraints is not None:
            for component, (output_name, constraint_type, desired_value) in self.inequalityConstraints.items():
                y = component.output[output_name].history
                desired_tensor = self.inequality_constraint_values[component]

                min_val = torch.min(y)
                max_val = torch.max(y)
                y = min_max_normalize(y, min_val, max_val)
                desired_tensor = min_max_normalize(desired_tensor, min_val, max_val)
                
                if constraint_type == "upper":
                    # Penalize when y > desired_value
                    self.loss += torch.mean(k*torch.relu(y - desired_tensor)**2)
                elif constraint_type == "lower":
                    # Penalize when y < desired_value
                    self.loss += torch.mean(k*torch.relu(desired_tensor - y)**2)

        # Handle minimization objectives
        if self.minimize is not None:
            for component, output_name in self.minimize.items():
                y = component.output[output_name].history
                min_val = torch.min(y)
                max_val = torch.max(y)
                y = min_max_normalize(y, min_val, max_val)
                self.loss += torch.mean(y**2)  # Minimize the mean value
        
        # Compute gradients
        self.loss.backward()
        return self.loss

    def optimize(self, 
                 decisionVariables: Dict[str, Dict] = None,
                 minimize: Dict[str, Dict] = None,
                 equalityConstraints: Dict[str, Dict] = None,
                 inequalityConstraints: Dict[str, Dict] = None,
                 startTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 endTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 stepSize: Union[float, List[float]] = None,):
        self.decisionVariables = decisionVariables
        self.minimize = minimize
        self.equalityConstraints = equalityConstraints
        self.inequalityConstraints = inequalityConstraints
        self.startTime = startTime
        self.endTime = endTime
        self.stepSize = stepSize

        # Disable gradients for all parameters since we're optimizing inputs.
        # It is VERY important to do this before initializing the model.
        # Otherwise, the model parameters and state space matrices will have requires_grad=True
        # and the backpropagate() call will fail.
        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

        self.simulator.get_simulation_timesteps(startTime, endTime, stepSize)
        self.simulator.model.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=self.simulator)

        # Enable gradients only for the inputs we want to optimize
        opt_list = []
        for component, (output_name, *bounds) in decisionVariables.items():
            component.output[output_name].set_requires_grad(True)
            opt_list.append(component.output[output_name].history)

        self.optimizer = torch.optim.Adadelta(opt_list, lr=0.1) # Adadelta works great

        def get_constraint_value(component_or_value):
            """Helper function to get constraint value, handling both ScheduleSystem and scalar values"""
            if isinstance(component_or_value, (int, float)):
                return torch.tensor(component_or_value)
            elif isinstance(component_or_value, systems.ScheduleSystem):
                component_or_value.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=self.simulator)
                return component_or_value.output["scheduleValue"].history
            elif isinstance(component_or_value, torch.Tensor):
                return component_or_value
            else:
                raise ValueError(f"Invalid constraint value type: {type(component_or_value)}")
            
    

        # Pre-compute all constraint values
        self.equality_constraint_values = {}
        if self.equalityConstraints is not None:
            for component, (output_name, constraint_type, desired_value) in self.equalityConstraints.items():
                self.equality_constraint_values[component] = get_constraint_value(desired_value)

        self.inequality_constraint_values = {}
        if self.inequalityConstraints is not None:
            for component, (output_name, constraint_type, desired_value) in self.inequalityConstraints.items():
                self.inequality_constraint_values[component] = get_constraint_value(desired_value)
                
        for i in range(100):  # 100 iterations
            self.optimizer.step(self.closure)

            # Apply bounds to decision variables
            with torch.no_grad():
                for component, (output_name, *bounds) in decisionVariables.items():
                    if len(bounds) > 0:
                        lower_bound = bounds[0] if len(bounds) > 0 else float('-inf')
                        upper_bound = bounds[1] if len(bounds) > 1 else float('inf')
                        component.output[output_name].history.clamp_(min=lower_bound, max=upper_bound)
            
            print(f"Loss at step {i}: {self.loss.detach().item()}")
            