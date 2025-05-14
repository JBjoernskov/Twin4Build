# import pygad
import twin4build.core as core
import torch
import datetime
from typing import Dict, List, Union
import torch.nn as nn
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

class Optimizer():
    def __init__(self, simulator: core.Simulator):
        self.simulator = simulator

    def optimize(self, 
                 targetInputs: Dict[str, Dict] = None,
                 targetOutputs: Dict[str, Dict] = None,
                 startTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 endTime: Union[datetime.datetime, List[datetime.datetime]] = None,
                 stepSize: Union[float, List[float]] = None,):
        # Optimize


        # Disable gradients for all parameters since we're optimizing inputs.
        # It is VERY important to do this before initializing the model.
        # Otherwise, the model parameters and state space matrices will have requires_grad=True
        # and the backpropagate() call will fail.
        for component in self.simulator.model.components.values():
            if isinstance(component, nn.Module):
                # print(f"\nDisabling gradients for component: {component.id}")
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

        self.simulator.get_simulation_timesteps(startTime, endTime, stepSize)
        self.simulator.model.initialize(startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=self.simulator)

        # Enable gradients only for the inputs we want to optimize
        opt_list = []
        for component in targetInputs.keys():
            component.output[targetInputs[component]].set_requires_grad(True)
            opt_list.append(component.output[targetInputs[component]].history)

        optimizer = torch.optim.Adam(opt_list, lr=0.001) # Adam 0.001 #SGD 0.0001
        desired_output = 21

        for i in range(100):  # 100 times
            optimizer.zero_grad()

            # Run simulation
            self.simulator.simulate(
                self.simulator.model,
                startTime=startTime,
                endTime=endTime,
                stepSize=stepSize,
                show_progress_bar=False
            )

            loss = 0
            for component in targetOutputs.keys():
                y = component.output[targetOutputs[component]].history
                loss += torch.mean((y - desired_output)**2)
            
            # Compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            with torch.no_grad():
                for component in targetInputs.keys():
                    component.output[targetInputs[component]].history.clamp_(min=0)
            
            print(f"Loss at step {i}: {loss.detach().item()}")
            