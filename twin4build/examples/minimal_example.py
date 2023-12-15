import os
import sys
import datetime
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)
import twin4build.utils.plot.plot as plot
import twin4build as tb
def fcn(self):
    ##############################################################
    ################## First, define components ##################
    ##############################################################

    #Define a schedule for the damper position
    #Other arguments such as "mondayRulesetDict" can also be added to define a more detailed schedule.
    position_schedule = tb.ScheduleSystem(weekDayRulesetDict = {
                                            "ruleset_default_value": 0,
                                            "ruleset_start_minute": [0,0,0,0,0,0,0],
                                            "ruleset_end_minute": [0,0,0,0,0,0,0],
                                            "ruleset_start_hour": [6,7,8,12,14,16,18],
                                            "ruleset_end_hour": [7,8,12,14,16,18,22],
                                            "ruleset_value": [0,0.1,1,0,0,0.5,0.7]}, #35
                                        add_noise=False,
                                        id="Position schedule")
    
    # Define damper component
    damper = tb.DamperSystem(nominalAirFlowRate = tb.Measurement(hasValue=1.6),
                            a=5,
                            id="Damper")

    #################################################################
    ################## Add connections to the model #################
    #################################################################
    self.add_connection(position_schedule, damper, "scheduleValue", "damperPosition")

    # Cycles are not allowed (with the exeption of controllers - see the controller example). If the following line is commented in, 
    # a cycle is introduced and the model will generate an error when "model.get_execution_order()" is run". 
    # You can see the generated graph with the cycle in the "system_graph.png" file.
    # self.add_connection(damper, damper, "airFlowRate", "damperPosition") #<------------------- comment in to create a cycle

def minimal_example():
    '''
        This is a simple example of how to manually define components and add connections between components.
        In this example we define a damper and a schedule for the position of the damper.
        The system is then simulated and the results are plotted.
    '''
    stepSize = 600 #Seconds
    startTime = datetime.datetime(year=2021, month=1, day=10, hour=0, minute=0, second=0)
    endTime = datetime.datetime(year=2021, month=1, day=12, hour=0, minute=0, second=0)
    model = tb.Model(id="example_model", saveSimulationResult=True)
    model.load_model(infer_connections=False, fcn=fcn)
    
    # Create a simulator instance
    simulator = tb.Simulator()

    # Simulate the model
    simulator.simulate(model,
                        stepSize=stepSize,
                        startTime = startTime,
                        endTime = endTime)
    
    plot.plot_damper(model, simulator, "Damper", show=False) #Set show=True to plot

if __name__ == '__main__':
    minimal_example()

