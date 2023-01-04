"""
An Optimizer class will be implemented here.

The optimizer optimizes operation of the building e.g. by changing setpoints.


Pytorch automatic differentiation could be used.
For some period:

0. Initialize inputs as torch.Tensors with "<variable>.requires_grad=true". 
All parameters are frozen with "<parameter>.requires_grad=false". 

Repeat 1-3 until convergence or stop criteria

1. Run simulation with inputs
2. Calculate loss (maybe operational costs or energy use)
3. Backpropagate and do step to update inputs




Monte Carlo discrete optimization could also be used. 
"Monte-Carlo Tree Search for Policy Optimization"
https://www.ijcai.org/proceedings/2019/0432.pdf

legal_actions: {variable1: [a1, a2, ... an]
                variable2: [a1, a2, ... an]}
lower_bounds: {variable1: 0,
                variable2: 0}
upper_bounds: {variable1: 999,
                variable2: 999}

model, decision_variables, legal_actions, lower_bound, upper_bound

"""


class Optimizer():
    def __init__(self,
                simulator=None):
        self.simulator = simulator

    def optimize(self):
        pass