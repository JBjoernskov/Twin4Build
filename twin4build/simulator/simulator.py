from tqdm import tqdm
import datetime
import math
import numpy as np
from twin4build.saref4bldg.building_space.building_space_model import BuildingSpaceModel
class Simulator:
    """
    The Simulator class simulates a model for a certain time period 
    using the <Simulator>.simulate(<Model>) method.
    """
    def __init__(self, 
                do_plot=False):
        self.do_plot = do_plot

    def do_component_timestep(self, component):
        #Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connectsAt:
            connection = connection_point.connectsSystemThrough
            connected_component = connection.connectsSystem
            if isinstance(component, BuildingSpaceModel):
                assert np.isnan(connected_component.output[connection.senderPropertyName])==False, f"Model output {connection.senderPropertyName} of component {connected_component.id} is NaN."
            component.input[connection_point.recieverPropertyName] = connected_component.output[connection.senderPropertyName]
        component.do_step(secondTime=self.secondTime, dateTime=self.dateTime, stepSize=self.stepSize)
        component.update_report()

    def do_system_time_step(self, model):
        """
        Do a system time step, i.e. execute the "do_step" for each component model. 

        
        Notes:
        The list model.execution_order currently consists of component groups that can be executed in parallel 
        because they dont require any inputs from each other. 
        However, in python neither threading or multiprocessing yields any performance gains.
        If the framework is implemented in another language, e.g. C++, parallel execution of components is expected to yield significant performance gains. 
        Another promising option for optimization is to group all components with identical classes/models as well as priority and perform a vectorized "do_step" on all such models at once.
        This can be done in python using numpy or torch.       
        """
  
        for component_group in model.execution_order:
            for component in component_group:
                self.do_component_timestep(component)

    def get_simulation_timesteps(self):
        n_timesteps = math.floor((self.endPeriod-self.startPeriod).total_seconds()/self.stepSize)
        self.secondTimeSteps = [i*self.stepSize for i in range(n_timesteps)]
        self.dateTimeSteps = [self.startPeriod+datetime.timedelta(seconds=i*self.stepSize) for i in range(n_timesteps)]
 
    def simulate(self, model, startPeriod, endPeriod, stepSize):
        """
        Simulate the "model" between the dates "startPeriod" and "endPeriod" with timestep equal to "stepSize" in seconds. 
        """
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod
        self.stepSize = stepSize
        model.initialize(startPeriod=startPeriod, endPeriod=endPeriod, stepSize=stepSize)
        self.get_simulation_timesteps()
        print("Running simulation") 
        for self.secondTime, self.dateTime in tqdm(zip(self.secondTimeSteps,self.dateTimeSteps), total=len(self.dateTimeSteps)):
            self.do_system_time_step(model)
        for component in model.flat_execution_order:
            if component.saveSimulationResult and self.do_plot:
                component.plot_report(self.dateTimeSteps)