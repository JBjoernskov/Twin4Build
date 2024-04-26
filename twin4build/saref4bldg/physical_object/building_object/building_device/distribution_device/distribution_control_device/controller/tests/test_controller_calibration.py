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


from twin4build.utils.data_loaders.load_spreadsheet import load_spreadsheet
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_sampler import data_sampler
import sys
import os

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 10)
sys.path.append(file_path)

from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_control_device.controller.setpoint_controller.pid_controller.pid_controller_system import ControllerSystem

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

def test():

    '''
        tests a ControllerSystem object by simulating its performance on a dataset, 
        calibrating it using the dataset, and then comparing the pre- and post-calibration 
        performance of the controller on the dataset. The function generates plots to visualize the results.
    '''

    logger.info("[Test Controller Calibration] : Test Function Entered ")

    controller = ControllerSystem(
                        observes = None,
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
    
    stepSize = 600 #seconds
    startTime = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endTime = datetime.datetime(year=2023, month=2, day=28, hour=0, minute=0, second=0, tzinfo=tzutc())

    response_filename = os.path.join(uppath(os.path.abspath(__file__), 9), "test", "data", "time_series_data", "OE20-601b-2_kafka_temperature.txt")
    data = [json.loads(line) for line in open(response_filename, 'rb')]
    data = data[1:] #remove header information
    data = np.array([row[0][0] for row in data])
    data = data[data[:, 0].argsort()]
    constructed_time_list,constructed_value_list,got_data = data_sampler(data=data, stepSize=stepSize, start_time=startTime, end_time=endTime, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 9)), "test", "data", "time_series_data", "OE20-601b-2.csv")
    format = "%m/%d/%Y %I:%M:%S %p"
    input = load_spreadsheet(filename=filename, stepSize=stepSize, start_time=startTime, end_time=endTime, format=format, dt_limit=999)
    input = input.rename(columns={'Indoor air temperature setpoint': 'setpointValue',
                                    'Indoor air temperature (Celcius)': 'actualValue',
                                    'Space heater valve position (0-100%)': 'inputSignal'})

    input["actualValue"] = constructed_value_list
    input = input.drop(columns=["SDUBook numOutStatus Interval Trend-Log",
                                "CO2 (ppm)",
                                "Damper valve position (0-100%)"])
    print(input)
    data_collection = DataCollection(name="input", df=input, nan_interpolation_gap_limit=9999)
    data_collection.interpolate_nans()
    input = data_collection.get_dataframe()

    input = input.iloc[321:3560,:].reset_index(drop=True)
    input = input.iloc[2300:,:].reset_index(drop=True)
    output = input["inputSignal"]/100
    input.drop(columns=["inputSignal"])

    print(input)

    start_pred = controller.do_period(input)
    fig, ax = plt.subplots(2)
    ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
    ax[0].plot(output, color="blue", label="Measured")
    ax[0].set_title('Before calibration')
    fig.legend()
    input = input.set_index("time")
    input.plot(subplots=True)
    controller.calibrate(input=input, output=output.to_numpy())
    end_pred = controller.do_period(input)
    ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
    ax[1].plot(output, color="blue", label="Measured")
    ax[1].set_title('After calibration')
    fig.set_size_inches(15,8)
    plt.show()

    logger.info("[Test Controller Calibration] : Exited form Test Function ")



if __name__ == '__main__':
    test()
