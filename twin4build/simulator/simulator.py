from tqdm import tqdm
import datetime
import math
class Simulator:
    """
    The Simulator class simulates a model for a certain time period 
    using the <Simulator>.simulate(<Model>) method.
    """
    def __init__(self, 
                timeStep,
                startPeriod,
                endPeriod,
                do_plot):
        self.timeStep = timeStep
        self.startPeriod = startPeriod
        self.endPeriod = endPeriod
        self.do_plot = do_plot

    def do_component_timestep(self, component):
        #Gather all needed inputs for the component through all ingoing connections
        for connection_point in component.connectsAt:
            connection = connection_point.connectsSystemThrough
            connected_component = connection.connectsSystem
            component.input[connection_point.recieverPropertyName] = connected_component.output[connection.senderPropertyName]
            # print("aa")
            # print(connection_point.recieverPropertyName)
            # print(connection.senderPropertyName)
            # print(connected_component.output[connection.senderPropertyName])
        # print("------------------------")
        # print(component.id)
        # print("before")
        # print(component.input)
        # print(component.output)
        component.do_step()
        # print("after")
        # print(component.output)
        component.update_report()


    def do_system_time_step(self, model):
        # model.execution_order currently consists of component groups that can be executed in parallel 
        # because they dont require any inputs from each other. 
        # However, in python neither threading or multiprocessing yields any performance gains.
        # If the framework is implmented in another language, e.g. C++, parallel execution of components is expected to yield significant performance gains. 
        for component_group in model.execution_order:
            for component in component_group:
                self.do_component_timestep(component)

    def get_simulation_timesteps(self):
        n_timesteps = math.floor((self.endPeriod-self.startPeriod).total_seconds()/self.timeStep)
        self.timeSteps = [self.startPeriod+datetime.timedelta(seconds=i*self.timeStep) for i in range(n_timesteps)]
 
    def simulate(self, model):    
        print("Running simulation") 
        self.get_simulation_timesteps()
        for time in tqdm(self.timeSteps):
            self.do_system_time_step(model)
            # print(time)

        
        for component in model.flat_execution_order:
            if component.createReport and self.do_plot:
                component.plot_report(self.timeSteps)


        # for component in model.flat_execution_order:
        #     if isinstance(component, BuildingSpaceModel) and component.createReport:
        #         import numpy as np
        #         component.x_list = np.array(component.x_list)
        #         plt.figure()
        #         plt.title(component.id)
        #         plt.plot(self.timeSteps, component.x_list[:,0], color="black") ######################
        #         plt.plot(self.timeSteps, component.x_list[:,1], color="blue") ######################
        #         plt.plot(self.timeSteps, component.x_list[:,2], color="red") ######################
        #         plt.plot(self.timeSteps, component.x_list[:,3], color="green") ######################

                

                # plt.figure()
                # plt.title("input_OUTDOORTEMPERATURE")
                # plt.plot(self.timeSteps, np.array(component.input_OUTDOORTEMPERATURE)[:,:])

                # plt.figure()
                # plt.title("input_RADIATION")
                # plt.plot(self.timeSteps, np.array(component.input_RADIATION)[:,:])

                # plt.figure()
                # plt.title("input_SPACEHEATER")
                # plt.plot(self.timeSteps, np.array(component.input_SPACEHEATER)[:,:])

                # plt.figure()
                # plt.title("input_VENTILATION")
                # plt.plot(self.timeSteps, np.array(component.input_VENTILATION)[:,:])