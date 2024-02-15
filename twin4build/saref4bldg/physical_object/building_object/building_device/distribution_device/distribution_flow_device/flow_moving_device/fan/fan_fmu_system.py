from .fan import Fan
from twin4build.utils.fmu.fmu_component import FMUComponent, unzip_fmu
from twin4build.utils.uppath import uppath
from scipy.optimize import least_squares
import numpy as np
import os
import sys
from twin4build.utils.fmu.unit_converters.functions import to_degC_from_degK, to_degK_from_degC, do_nothing
import twin4build.base as base
from twin4build.utils.signature_pattern.signature_pattern import SignaturePattern, Node, Exact, IgnoreIntermediateNodes

def get_signature_pattern():
    node0 = Node(cls=(base.Meter,))
    node1 = Node(cls=(base.Fan,))
    sp = SignaturePattern(ownedBy="FanFMUSystem")
    sp.add_edge(Exact(object=node0, subject=node1, predicate="connectedBefore") | IgnoreIntermediateNodes(object=node0, subject=node1, predicate="connectedBefore"))
    sp.add_input("airFlow", node0)
    sp.add_modeled_node(node1)
    return sp

class FanFMUSystem(FMUComponent, Fan):
    sp = get_signature_pattern()
    def __init__(self,
                c1=None,
                c2=None,
                c3=None,
                c4=None,
                f_total=None,
                **kwargs):
        Fan.__init__(self, **kwargs)
        # self.c1 = 0.09206979
        # self.c2 = -0.06898674
        # self.c3 = 0.91641847
        # self.c4 = -0.11519787

        self.c1=c1
        self.c2=c2
        self.c3=c3
        self.c4=c4
        self.f_total=f_total
        self.start_time = 0
        # fmu_filename = "EPlusFan_0FMU.fmu"#EPlusFan_0FMU_0test2port
        fmu_filename = "EPlusFan_0FMU_0test2port.fmu"
        self.fmu_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)
        self.unzipdir = unzip_fmu(self.fmu_path)

        self.input = {"airFlowRate": None,
                      "inletAirTemperature": None}
        self.output = {"outletAirTemperature": None,
                       "Power": None}
        
        #Used in finite difference jacobian approximation for uncertainty analysis.
        self.inputLowerBounds = {"airFlowRate": 0,
                                "inletAirTemperature": -np.inf}
        self.inputUpperBounds = {"airFlowRate": np.inf,
                                "inletAirTemperature": np.inf}
        
        # self.FMUinputMap = {"airFlowRate": "airFlowRate",
        #                 "inletAirTemperature": "inletAirTemperature"}
        
        # self.FMUoutputMap = {"outletAirTemperature": "outletAirTemperature",
        #                   "Power": "Power"}

        self.FMUinputMap = {"airFlowRate": "inlet.m_flow",
                        "inletAirTemperature": "inlet.forward.T"}
        
        self.FMUoutputMap = {"outletAirTemperature": "outlet.forward.T",
                          "Power": "Power"}

        self.FMUparameterMap = {"nominalPowerRate.hasValue": "nominalPowerRate",
                                "nominalAirFlowRate.hasValue": "nominalAirFlowRate",
                                "c1": "c1",
                                "c2": "c2",
                                "c3": "c3",
                                "c4": "c4",
                                "f_total": "f_total"}

        self.input_unit_conversion = {"airFlowRate": do_nothing,
                                      "inletAirTemperature": to_degK_from_degC}
        
        self.output_unit_conversion = {"outletAirTemperature": to_degC_from_degK,
                                      "Power": do_nothing}
        self.INITIALIZED = False
        self._config = {"parameters": list(self.FMUparameterMap.keys())}

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
        '''
            This function initializes the FMU component by setting the start_time and fmu_filename attributes, 
            and then sets the parameters for the FMU model.
        '''
 
        if self.INITIALIZED:
            self.reset()
        else:
            FMUComponent.__init__(self, fmu_path=self.fmu_path, unzipdir=self.unzipdir)
            self.INITIALIZED = True ###

    def do_period(self, input, stepSize=None, measuring_device_types=None):
        '''
            This function performs a simulation period for the FMU model with the given input dataframe and optional stepSize.
            It iterates through each row of the input dataframe and sets the input parameters for the FMU model accordingly. 
            It then runs the simulation with the given stepSize and saves the output to a list.
            Finally, it returns the predicted output of the simulation.
        '''
        if measuring_device_types:
            self.CALC_Y_RANGE = True
            measuring_device_types = [getattr(sys.modules[__name__], measuring_device_type) for measuring_device_type in measuring_device_types]
            A_list = [measuring_device_type.MEASURING_ERROR if measuring_device_type.MEASURING_TYPE=="A" else 0 for measuring_device_type in measuring_device_types]
            P_list = [measuring_device_type.MEASURING_ERROR/100 if measuring_device_type.MEASURING_TYPE=="P" else 0 for measuring_device_type in measuring_device_types]
            self.input_A_range = np.array([A_list])
            self.input_P_range = np.array([P_list])
            self.output_range = []
            self.subset_mask = np.array([True if key in self.FMUoutput.values() else False for key in self.fmu_outputs])

        self.clear_report()        
        start_time = input.index[0].to_pydatetime()
        # print("start")
        for time, row in input.iterrows():
            time_seconds = (time.to_pydatetime()-start_time).total_seconds()
            # print(time_seconds)

            for key in input:
                self.input[key] = row[key]
            self.do_step(secondTime=time_seconds, stepSize=stepSize)
            self.update_report()

        output_predicted = np.array(self.savedOutput["outletAirTemperature"])
        return output_predicted

    def obj_fun(self, x, input, output, stepSize):
        '''
            This function calculates the loss (residual) between the predicted and measured output using 
            the least_squares optimization method. It takes in an array x representing the parameter to be optimized, 
            input and output dataframes representing the input and measured output values, respectively. 
            It uses the do_period function to predict the output values with the given x parameter and calculates the 
            residual between the predicted and measured output. It returns the residual.
        '''
        parameters = {"m_flow_nominal": x[0],
                      "dp_nominal": x[1]
                    }
        # parameters = {"r_nominal": x[0],
        #               "Q_flow_nominal": x[1],
        #               "T_a1_nominal": x[2],
        #               "T_b1_nominal": x[3],
        #               "T_a2_nominal": x[4],
        #               "T_b2_nominal": x[5]
        #             }
        self.initialParameters = parameters#.update(parameters)
        self.reset()

        output_predicted = self.do_period(input, stepSize=stepSize)
        res = output_predicted-output #residual of predicted vs measured
        print(f"Loss: {np.sum(res**2)}")
        return res

    def calibrate(self, input=None, output=None, stepSize=None):
        '''
            This function performs calibration using the obj_fun function and the least_squares 
            optimization method with the given input and output. It initializes an array x0 representing the 
            initial parameter value, sets bounds for the parameter optimization, and then uses least_squares
            to find the optimal value for the Radiator.UAEle parameter. 
            Finally, it sets the optimal Radiator.UAEle parameter based on the calibration results.
        '''

        x0 = np.array([2/3, 2000])
        lb = [0, 0]
        ub = [1, 5000000]

        bounds = (lb,ub)
        sol = least_squares(self.obj_fun, x0=x0, bounds=bounds, args=(input, output, stepSize))
        self.reset()
        print(sol)

        


        