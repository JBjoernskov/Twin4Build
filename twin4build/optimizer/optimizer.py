

"""
An Optimizer class will be implemented here.


legal_actions: {variable: [a1, a2, ... an]}
lower_bounds: {variable: 0}
upper_bounds: {variable: 999}

model, decision_variables, legal_actions, lower_bound, upper_bound

"Monte-Carlo Tree Search for Policy Optimization"
https://www.ijcai.org/proceedings/2019/0432.pdf

"""


class Optimizer():
    def __init__(self,
                simulator=None):
        self.simulator = simulator


    def MCTS(self):
        pass

    def optimize(self):
        self.MCTS()