import os
import sys
import datetime
from dateutil.tz import tzutc
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    #change the number here according to your requirement
    #desired path looks like this "D:\Projects\Twin4Build
    file_path = uppath(os.path.abspath(__file__), 10)
    #file_path = uppath(os.path.abspath(__file__), 9)
    print(file_path)
    sys.path.append(file_path)

    calibrated_path = file_path+"/calibrated_folder"
    if not os.path.exists(calibrated_path):
         os.makedirs(calibrated_path)

from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_sampler import data_sampler
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.controller_system import ControllerSystem

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class dynamic_controller_calibration:
    def __init__(self,input_X,output_Y):

        logger.info("[Dynamic Controller Calibrator] : Entered in Initialise Function")

        self.input_data  = input_X
        self.output_data = output_Y
        self.model_set_parameters()
        #self.data_prep_method()
        self.save_plots()

    def model_set_parameters(self):
        self.controller = ControllerSystem(
                        controlsProperty = None,
                        K_p = 0.1,
                        K_i = 0.1,
                        K_d = 0.1,
                        input = {},
                        output = {},
                        savedInput = {},
                        savedOutput = {},
                        saveSimulationResult = True,
                        connectedThrough = [],
                        connectsAt = [],
                        id = "Controller")

    def save_plots(self):

        '''
            It uses the input_data and output_data attributes of the class instance
            and the controller attribute to create two plots: one with the predicted 
            output before calibration and one with the predicted output after calibration. 
            It also shows the input data in subplot form.
        '''
        logger.info("[Dynamic Controller Calibrator] : Entered in Save Plots Function")

        start_pred = self.controller.do_period(self.input_data)
        fig, ax = plt.subplots(2)
        ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
        ax[0].plot(self.output_data, color="blue", label="Measured")
        ax[0].set_title('Before calibration')
        fig.legend()
        self.input_data = self.input_data.set_index("time")
        self.input_data.plot(subplots=True)
        end_pred = self.controller.do_period(self.input_data)
        ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
        ax[1].plot(self.output_data, color="blue", label="Measured")
        ax[1].set_title('After calibration')
        fig.set_size_inches(15,8)
        plt.show()
        
        logger.info("[Dynamic Controller Calibrator] : Exited from Save Plots Function")


    def calibrate_results(self):
        return(self.controller.calibrate(self.input_data, self.output_data.to_numpy()))

def read_data():

    '''
        The read_data function loads time series data from a file, 
        samples it at a given time interval, and returns the data as a Pandas dataframe. 
        The data is then manipulated and processed, and two dataframes are returned - 
        one containing input data and another containing output data. 
        The function is used in a larger project to test a controller's performance.
    '''

    
    logger.info("[Dynamic Controller Calibrator] : Entered in Read Data Function")


    stepSize = 600 #seconds
    startPeriod = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2023, month=2, day=28, hour=0, minute=0, second=0, tzinfo=tzutc())


    response_filename = os.path.join(uppath(os.path.abspath(__file__), 9), "test", "data", "time_series_data", "OE20-601b-2_kafka_temperature.txt")
    data = [json.loads(line) for line in open(response_filename, 'rb')]
    data = data[1:] #remove header information
    data = np.array([row[0][0] for row in data])
    data = data[data[:, 0].argsort()]
    constructed_time_list,constructed_value_list,got_data = data_sampler(data=data, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 9)), "test", "data", "time_series_data", "OE20-601b-2.csv")
    format = "%m/%d/%Y %I:%M:%S %p"
    input = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=999)
    input = input.rename(columns={'Indoor air temperature setpoint': 'setpointValue',
                                    'Indoor air temperature (Celcius)': 'actualValue',
                                    'Space heater valve position (0-100%)': 'inputSignal'})

    input["actualValue"] = constructed_value_list
    input = input.drop(columns=["SDUBook numOutStatus Interval Trend-Log",
                                "CO2 (ppm)",
                                "Damper valve position (0-100%)"])
    data_collection = DataCollection(name="input", df=input)
    data_collection.interpolate_nans()
    input_data = data_collection.get_dataframe()

    input_data = input_data.iloc[321:3560,:].reset_index(drop=True)
    input_data = input_data.iloc[2300:,:].reset_index(drop=True)
    output_data = input_data["inputSignal"]/100
    input_data.drop(columns=["inputSignal"])

    
    logger.info("[Dynamic Controller Calibrator] : Exited from Read Data Function")


    return (input_data,output_data)


if __name__ == '__main__':
    #use id as used into id = "controller"
    controller_units = {"controller_1":
                                {"input_filename":"",
                                "output_filename" :""
                                },
                            }
    calibrated_variable_dict = {}

    for controller_unit in controller_units.keys():
        input_X,output_Y = read_data()
        controller_unit_cls_obj = dynamic_controller_calibration(input_X,output_Y)
        calibrated_variable_dict[controller_unit] = controller_unit_cls_obj.calibrate_results()

    calibrated_full_path = calibrated_path+"/calibrated_controller_parameters.json"
    with open(calibrated_full_path, "w") as outfile:
        json.dump(calibrated_variable_dict, outfile)
    
