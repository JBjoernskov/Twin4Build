# import pygad
import twin4build.core as core
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
    def __init__(self,
                model=None):
        self.model = model
        self.simulator = core.Simulator(model)

    def fitness_function(self):
        #set inputs 
        
        # self.model.set_parameters_from_array(theta, self.flat_component_list, self.flat_attr_list)
        # self.simulator.simulate(self.model,
        #                             stepSize=stepSize_,
        #                             startTime=startTime_,
        #                             endTime=endTime_,
        #                             show_progress_bar=False)
        # solution is a list of parameters
        # solution_idx is the index of the solution in the population
        # Run simulation with solution
        # Calculate loss
        # Return loss
        pass

    def optimize(self):
        pass
    #     # ga_instance = pygad.GA(num_generations=num_generations,
    #     #                num_parents_mating=num_parents_mating, 
    #     #                fitness_func=fitness_function,
    #     #                sol_per_pop=sol_per_pop, 
    #     #                num_genes=num_genes,
    #     #                on_generation=callback_generation)
        
    #     # Running the GA to optimize the parameters of the function.
    #     ga_instance.run()