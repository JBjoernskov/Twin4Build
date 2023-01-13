"""
An Estimator class will be implemented here.
The estimator estimates the parameters of the components models 
based on a period with measurements from the actual building. 

Pytorch automatic differentiation could be used.
For some period:

0. Initialize parameters as torch.Tensor with "<parameter>.requires_grad=true". 
    All inputs are provided as torch.Tensor with "<input>.requires_grad=false". 
    Selected model parameters can be "frozen" by setting "<parameter>.requires_grad=false"

Repeat 1-3 until convergence or stop criteria

1. Run simulation with inputs
2. Calculate loss based on predicted and measured values
3. Backpropagate and do step to update parameters

"""

class Estimator():
    def __init__(self,
                simulator=None):
        self.simulator = simulator

    def estimate(self):
        pass