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




Plot Euclidian Distance from final solution and amount of data and accuracy/error
"""
import math
import os
import sys
from twin4build.simulator.simulator import Simulator
from twin4build.logger.Logging import Logging
from twin4build.monitor.monitor import Monitor
from twin4build.utils.rsetattr import rsetattr
from twin4build.utils.rgetattr import rgetattr
from scipy.optimize import least_squares
import numpy as np

logger = Logging.get_logger("ai_logfile")

class Estimator():
    def __init__(self,
                model=None):
        self.model = model
        self.simulator = Simulator(model)
        logger.info("[Estimator : Initialise Function]")

    def estimate(self,
                x0=None,
                lb=None,
                ub=None,
                y_scale=None,
                trackGradients=True,
                targetParameters=None,
                targetMeasuringDevices=None,
                initialization_steps=None,
                startPeriod=None,
                endPeriod=None,
                startPeriod_test=None,
                endPeriod_test=None,
                stepSize=None):
        if startPeriod_test is None or endPeriod_test is None:
            test_period_supplied = False
            assert startPeriod_test is None and endPeriod_test is None, "Both startPeriod_test and endPeriod_test must be supplied"
        else:
            test_period_supplied = True
        
        self.simulator.get_simulation_timesteps(startPeriod, endPeriod, stepSize)
        self.n_initialization_steps = 60
        if test_period_supplied:
            self.n_train = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            self.n_init_train = self.n_train + self.n_initialization_steps
            self.startPeriod_train = startPeriod
            self.endPeriod_train = endPeriod
            self.startPeriod_test = startPeriod_test
            self.endPeriod_test = endPeriod_test
        else:
            split_train = 0.6
            self.n_estimate = len(self.simulator.dateTimeSteps)-self.n_initialization_steps
            self.n_train = math.ceil(self.n_estimate*split_train)
            self.n_init_train = self.n_train + self.n_initialization_steps
            self.n_test = self.n_estimate-self.n_train
            
            self.startPeriod_train = startPeriod
            self.endPeriod_train = self.simulator.dateTimeSteps[self.n_initialization_steps+self.n_train]
            self.startPeriod_test = self.simulator.dateTimeSteps[self.n_initialization_steps+self.n_train+1]
            self.endPeriod_test = endPeriod
        
        self.actual_readings = self.simulator.get_actual_readings(startPeriod=self.startPeriod_train, endPeriod=self.endPeriod_train, stepSize=stepSize).iloc[self.n_initialization_steps:,:]
        self.min_actual_readings = self.actual_readings.min(axis=0)
        self.max_actual_readings = self.actual_readings.max(axis=0)
        self.min_max_diff = self.max_actual_readings-self.min_actual_readings
        self.actual_readings_min_max_normalized = (self.actual_readings-self.min_actual_readings)/(self.max_actual_readings-self.min_actual_readings)
        x0 = [val for lst in x0.values() for val in lst]
        lb = [val for lst in lb.values() for val in lst]
        ub = [val for lst in ub.values() for val in lst]
        # x_scale = [val for lst in x_scale.values() for val in lst]
        self.y_scale = y_scale

        self.flat_component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        self.flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

        self.trackGradients = trackGradients
        self.targetParameters = targetParameters
        self.targetMeasuringDevices = targetMeasuringDevices
        bounds = (lb,ub)
        self.n_obj_eval = 0
        self.best_loss = math.inf

        if self.trackGradients:
            sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, jac=self.get_jacobian, verbose=2, x_scale="jac", loss="linear", max_nfev=10000000, xtol=1e-15, args=(stepSize, self.startPeriod_train, self.endPeriod_train)) #, x_scale="jac"
        else:
            sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, max_nfev=10000000, x_scale="jac", args=(stepSize, self.startPeriod_train, self.endPeriod_train))
        print(sol)
        # try:
        #     if self.trackGradients:
        #         sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, jac=self.get_jacobian, verbose=2, x_scale="jac", loss="linear", max_nfev=10000000, xtol=1e-15, args=(stepSize, self.startPeriod_train, self.endPeriod_train)) #, x_scale="jac"
        #     else:
        #         sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, max_nfev=10000000, x_scale="jac", args=(stepSize, self.startPeriod_train, self.endPeriod_train))
        #     print(sol)
        # except Exception as e:
        #     print(e)
        #     self.set_parameters(self.best_parameters)
        #     print(self.best_parameters)

        
        self.monitor = Monitor(self.model)
        self.monitor.monitor(startPeriod=self.startPeriod_test,
                            endPeriod=self.endPeriod_test,
                            stepSize=stepSize,
                            do_plot=False)

    def get_solution(self):
        sol_dict = {}
        sol_dict["MSE"] = self.monitor.get_MSE()
        sol_dict["RMSE"] = self.monitor.get_RMSE()
        sol_dict["n_obj_eval"] = self.n_obj_eval
        for component, attr_list in self.targetParameters.items():
            sol_dict[component.id] = []
            for attr in attr_list:
                sol_dict[component.id].append(rgetattr(component, attr))
        return sol_dict
            
    def get_jacobian(self, *args):
        jac = np.zeros((self.n_adjusted*len(self.targetMeasuringDevices), len(self.flat_attr_list)))
        k = 0
        # np.set_printoptions(threshold=sys.maxsize)
        for y_scale, measuring_device in zip(self.y_scale, self.targetMeasuringDevices):
            for i, (component, attr) in enumerate(zip(self.flat_component_list, self.flat_attr_list)):
                grad = np.array(component.savedParameterGradient[measuring_device][attr])[self.n_initialization_steps:]
                jac[k:k+self.n_adjusted, i] = grad[self.no_flow_mask]/y_scale
                # print("----")
                # print(measuring_device.id)
                # print(attr)
                # print(jac[k:k+self.n_adjusted, i])
                
            # jac[k:k+self.n_adjusted, :] = jac[k:k+self.n_adjusted, :]/(self.max_actual_readings[measuring_device.id]-self.min_actual_readings[measuring_device.id])
            k+=self.n_adjusted
        # print([el.id for el in self.targetMeasuringDevices])
        # np.set_printoptions(threshold=sys.maxsize)

        # print(self.targetMeasuringDevices[0].id)
        # print(jac[0:self.n_adjusted])

        # print(self.targetMeasuringDevices[1].id)
        # print(jac[self.n_adjusted:2*self.n_adjusted])

        # print(self.targetMeasuringDevices[2].id)
        # print(jac[2*self.n_adjusted:3*self.n_adjusted])
        # np.set_printoptions(precision=3)
        # np.set_printoptions(suppress=True)
        print(self.targetMeasuringDevices[0].id)
        print("Mean: ", np.mean(jac[0:self.n_adjusted], axis=0))
        print("Max: ", np.max(np.abs(jac[0:self.n_adjusted]), axis=0))

        print(self.targetMeasuringDevices[1].id)
        print("Mean: ", np.mean(jac[self.n_adjusted:2*self.n_adjusted], axis=0))
        print("Max: ", np.max(np.abs(jac[self.n_adjusted:2*self.n_adjusted]), axis=0))

        print(self.targetMeasuringDevices[2].id)
        print("Mean: ", np.mean(jac[2*self.n_adjusted:3*self.n_adjusted], axis=0))
        print("Max: ", np.max(np.abs(jac[2*self.n_adjusted:3*self.n_adjusted]), axis=0))

        
        return jac


    def set_parameters(self, x):
        for i, (obj, attr) in enumerate(zip(self.flat_component_list, self.flat_attr_list)):
            rsetattr(obj, attr, x[i])
        
    def obj_fun(self, x, stepSize, startPeriod, endPeriod):
        '''
            This function calculates the loss (residual) between the predicted and measured output using 
            the least_squares optimization method. It takes in an array x representing the parameters to be optimized, 
            sets these parameter values in the model and simulates the model to obtain the predictions. 
        '''
    

        # Set parameters for the model
        self.set_parameters(x)


        self.simulator.simulate(self.model,
                                stepSize=stepSize,
                                startPeriod=startPeriod,
                                endPeriod=endPeriod,
                                trackGradients=self.trackGradients,
                                targetParameters=self.targetParameters,
                                targetMeasuringDevices=self.targetMeasuringDevices)
        
        ############################################################################################
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        import copy

        coil = self.model.component_dict["coil"]
        temp_joined = {key_input: [] for key_input in coil.FMUinputMap.values()}
        temp_joined.update({key_input: [] for key_input in coil.FMUparameterMap.values()})
        localGradientsSaved_re = {key_output: copy.deepcopy(temp_joined) for key_output in coil.FMUoutputMap.values()}
        
        for i, localGradient in enumerate(coil.localGradientsSaved):
            for output_key, input_dict in localGradient.items():
                for input_key, value in input_dict.items():
                    localGradientsSaved_re[output_key][input_key].append(value)
          
        for output_key, input_dict in localGradient.items():
            fig, ax = plt.subplots()
            fig.suptitle(f"dy: {output_key}", fontsize=25)
            color = iter(cm.turbo(np.linspace(0, 1, len(localGradient[output_key]))))   
            for input_key, value in input_dict.items():
                label = f"dx: {input_key}"
                c = next(color)
                ax.plot(localGradientsSaved_re[output_key][input_key], label=label, color=c)
                print("-----------------------------")
                print(output_key)
                print(input_key)
                print(localGradientsSaved_re[output_key][input_key])
            ax.legend(prop={'size': 10})
            ax.set_ylim([-3, 7])
        plt.show()
        ############################################################################################


        # Filtering has to be constant size
        waterFlowRate = np.array(self.model.component_dict["valve"].savedInput["valvePosition"])
        airFlowRate = np.array(self.model.component_dict["fan flow meter"].savedOutput["airFlowRate"])
        tol = 1e-4
        self.no_flow_mask = np.logical_and(waterFlowRate>tol,airFlowRate>tol)[self.n_initialization_steps:]

        k = 0
        self.n_adjusted = np.sum(self.no_flow_mask==True)
        res = np.zeros((self.n_adjusted*len(self.targetMeasuringDevices)))
        for y_scale, measuring_device in zip(self.y_scale, self.targetMeasuringDevices):
            simulation_readings = np.array(next(iter(measuring_device.savedInput.values())))[self.n_initialization_steps:]
            actual_readings = self.actual_readings[measuring_device.id].to_numpy()
            simulation_readings_min_max_normalized = (simulation_readings-self.min_actual_readings[measuring_device.id])/(self.max_actual_readings[measuring_device.id]-self.min_actual_readings[measuring_device.id])
            # res+=np.abs(simulation_readings-actual_readings)

            # res[k:k+self.n_adjusted] = simulation_readings_min_max_normalized[self.no_flow_mask]-self.actual_readings_min_max_normalized[measuring_device.id][self.no_flow_mask]
            res[k:k+self.n_adjusted] = (simulation_readings[self.no_flow_mask]-actual_readings[self.no_flow_mask])/y_scale
            k+=self.n_adjusted
        self.n_obj_eval+=1
        self.loss = np.sum(res**2)
        if self.loss<self.best_loss:
            self.best_loss = self.loss
            self.best_parameters = x

        print("=================")
        print("Loss: {:0.2f}".format(self.loss))
        print("Best Loss: {:0.2f}".format(self.best_loss))
        print(x)
        print("=================")
        # print(self.targetMeasuringDevices[0].id)
        # print(res[0:self.n_adjusted])

        # print(self.targetMeasuringDevices[1].id)
        # print(res[self.n_adjusted:2*self.n_adjusted])

        # print(self.targetMeasuringDevices[2].id)
        # print(res[2*self.n_adjusted:3*self.n_adjusted])
        # 
        return res
