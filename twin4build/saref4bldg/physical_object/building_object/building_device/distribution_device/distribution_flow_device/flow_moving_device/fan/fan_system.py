from .fan import Fan
from typing import Union
import twin4build.saref.measurement.measurement as measurement
import numpy as np
from twin4build.logger.Logging import Logging
from scipy.optimize import least_squares

logger = Logging.get_logger("ai_logfile")

class FanSystem(Fan):
    def __init__(self,
                c1: Union[float, None]=None,
                c2: Union[float, None]=None,
                c3: Union[float, None]=None,
                c4: Union[float, None]=None,
                **kwargs):
        
        logger.info("[Fan Model Class] : Entered in Initialise Function")

        super().__init__(**kwargs)
        assert isinstance(c1, float) or c1 is None, "Attribute \"c1\" is of type \"" + str(type(c1)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(c2, float) or c2 is None, "Attribute \"c2\" is of type \"" + str(type(c2)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(c3, float) or c3 is None, "Attribute \"c3\" is of type \"" + str(type(c3)) + "\" but must be of type \"" + str(float) + "\""
        assert isinstance(c4, float) or c4 is None, "Attribute \"c4\" is of type \"" + str(type(c4)) + "\" but must be of type \"" + str(float) + "\""
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

        # self.input = {"airFlowRate": None,
        #               "inletAirTemperature": None}
        # self.output = {"outletAirTemperature": None,
        #                "Power": None,
        #                "Energy": 0}

        self.input = {"airFlowRate": None}
        self.output = {"Power": None,
                       "Energy": 0}

        logger.info("[Fan Model Class] : Exited from Initialise Function")
        self._config = {"parameters": ["c1", "c2", "c3", "c4", "nominalAirFlowRate.hasValue", "nominalPowerRate.hasValue"]}

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass
        
    def do_step(self, secondTime=None, dateTime=None, stepSize=None):

        logger.info("[Fan Model Class] : Entered in do step Function")

        # if self.input["airFlowRate"] < 1e-5:
        #     self.output["Power"] = 0
        # else:
        f_flow = self.input["airFlowRate"]/self.nominalAirFlowRate.hasValue
        f_pl = self.c1 + self.c2*f_flow + self.c3*f_flow**2 + self.c4*f_flow**3
        W_fan = f_pl*self.nominalPowerRate.hasValue

        self.output["Power"] = W_fan
        self.output["Energy"] =  self.output["Energy"] + W_fan*stepSize/3600/1000

        logger.info("[Fan Model Class] : Exited from do step Function")

    def do_period(self, input, stepSize=None, vectorize=False):
        '''
            Runs the simulation for the given input over the entire period and returns the predicted energy output.
        '''
        logger.info("[space heater model] : Entered in DoPeriod Function")
        self.clear_results()       
        if vectorize==True:
            for key in input:
                self.input[key] = input[key]
            self.do_step(stepSize=stepSize)
            output_predicted = self.output["Power"].to_numpy()
        else:
            start_time = input.index[0].to_pydatetime()
            # print("start")
            for time, row in input.iterrows():
                time_seconds = (time.to_pydatetime()-start_time).total_seconds()
                # print(time_seconds)

                for key in input:
                    self.input[key] = row[key]
                self.do_step(secondTime=time_seconds, stepSize=stepSize)
                self.update_results()
            output_predicted = np.array(self.savedOutput["Power"])

        logger.info("[space heater model] : Exited from DoPeriod Function")

        return output_predicted

    def obj_fun(self, x0, input, output, stepSize, vectorize):
        '''
            Calculates the residual between the predicted and actual energy output for the given 
            input and output data, given the current model parameters.
        '''

        logger.info("[space heater model] : Entered in Object Function")

        self.c1 = x0[0]
        self.c2 = x0[1]
        self.c3 = x0[2]
        self.c4 = x0[3]
        self.nominalAirFlowRate.hasValue = x0[4]
        self.nominalPowerRate.hasValue = x0[5]

        output_predicted = self.do_period(input, stepSize, vectorize)
        res = output_predicted-output #residual of predicted vs measured

        self.loss = np.sum(res**2)
        print("---")
        print(f"MAE: {np.mean(np.abs(res))}")
        print(f"RMSE: {np.mean(res**2)**(0.5)}")

        logger.info("[space heater model] : Exitedd from Object Function")

        return res

    def calibrate(self, x0 , lb, ub, input=None, output=None, stepSize=None, vectorize=False):
        '''
            Calibrates the model using the given input and output data, 
            optimizing the model parameters to minimize the residual between predicted and 
            actual energy output. Returns the optimized model parameters.
        '''

        assert input is not None
        assert output is not None
        assert stepSize is not None
        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output, stepSize, vectorize))
        return sol.x
