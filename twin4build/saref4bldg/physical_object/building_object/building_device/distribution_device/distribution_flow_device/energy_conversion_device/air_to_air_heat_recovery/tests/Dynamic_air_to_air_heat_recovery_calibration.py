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
    file_path = uppath(os.path.abspath(__file__), 11)
    #file_path = uppath(os.path.abspath(__file__), 9)
    print(file_path)
    sys.path.append(file_path)

    calibrated_path = file_path+"/calibrated_folder"
    if not os.path.exists(calibrated_path):
         os.makedirs(calibrated_path)



from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.energy_conversion_device.air_to_air_heat_recovery.air_to_air_heat_recovery_model import AirToAirHeatRecoveryModel
from twin4build.saref.measurement.measurement import Measurement
#import pwlf



class dynamic_calibration_heat_recovery:
    '''
        initializes the class with input and output data, 
        sets the parameters for an AirToAirHeatRecoveryModel, and calls the save_plots() method.
    '''
    def __init__(self,input_X,output_Y):
        self.input_data  = input_X
        self.output_data = output_Y
        self.model_set_parameters()
        #self.data_prep_method()
        self.save_plots()

    def model_set_parameters(self):
        '''
             creates an AirToAirHeatRecoveryModel object with specific parameter values.
        '''
        self.air_to_air_heat_recovery = AirToAirHeatRecoveryModel(
                specificHeatCapacityAir = Measurement(hasValue=1000),
                eps_75_h = 0.8,
                eps_75_c = 0.8,
                eps_100_h = 0.8,
                eps_100_c = 0.8,
                primaryAirFlowRateMax = Measurement(hasValue=25000/3600*1.225),
                secondaryAirFlowRateMax = Measurement(hasValue=25000/3600*1.225),
                subSystemOf = [],
                input = {},
                output = {},
                savedInput = {},
                savedOutput = {},
                saveSimulationResult = True,
                connectedThrough = [],
                connectsAt = [],
                id = "AirToAirHeatRecovery")

    def save_plots(self):
        # These lines are specific to this code. Please change if required
        #input_plot = self.input_data.iloc[20000:21000,:].reset_index(drop=True)
        #output_plot = self.input_plot["primaryTemperatureOut"].to_numpy()

        input_plot = self.input_data
        output_plot =self.output_data

        start_pred = self.air_to_air_heat_recovery.do_period(input_plot) ####
        fig, ax = plt.subplots(2)
        ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
        ax[0].plot(output_plot, color="blue", label="Measured")
        ax[0].set_title('Before calibration')
        fig.legend()
        self.input_data.set_index("time")
        self.input_data.plot(subplots=True)
        end_pred = self.air_to_air_heat_recovery.do_period(input_plot)
        ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
        ax[1].plot(output_plot, color="blue", label="Measured")
        ax[1].set_title('After calibration')
        for a in ax:
            a.set_ylim([18,22])

        plt.show()

    def calibrate_results(self):
        return(self.air_to_air_heat_recovery.calibrate(self.input_data, self.output_data))

def read_data():
    '''
        This is a Python function that reads data from several CSV files using a custom function 
        "load_from_file" with a defined file path, time range, and date format. 
        The data is loaded into pandas DataFrames and then processed, including conversions
        from imperial to metric units. Finally, the processed data is inserted into a pandas DataFrame called "input" 
        that is returned as output from the function. The primaryTemperatureIn column of "input" is calculated as a function of other columns.
    '''
    input = pd.DataFrame()

    stepSize = 600
    startPeriod = datetime.datetime(year=2021, month=10, day=1, hour=0, minute=0, second=0, tzinfo=tzutc()) 
    endPeriod = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    format = "%m/%d/%Y %I:%M:%S %p"

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "weather_BMS.csv")
    weather = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    temp = weather.copy()
    # weather["outdoorTemperature"] = (weather["outdoorTemperature"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_efficiency.csv")
    VE02_efficiency = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_airflowrate_supply_kg_s.csv")
    VE02_primaryAirFlowRate = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_primaryAirFlowRate["primaryAirFlowRate"] = VE02_primaryAirFlowRate["primaryAirFlowRate"]*0.0283168466/60*1.225 #convert from cubic feet per minute to kg/s

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_airflowrate_exhaust_kg_s.csv")
    VE02_secondaryAirFlowRate = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_secondaryAirFlowRate["secondaryAirFlowRate"] = VE02_secondaryAirFlowRate["secondaryAirFlowRate"]*0.0283168466/60*1.225 #convert from cubic feet per minute to kg/s

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_FTU1.csv")
    VE02_FTU1 = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_FTU1["FTU1"] = (VE02_FTU1["FTU1"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_FTG_MIDDEL.csv")
    VE02_FTG_MIDDEL = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_FTG_MIDDEL["FTG_MIDDEL"] = (VE02_FTG_MIDDEL["FTG_MIDDEL"]-32)*5/9 #convert from fahrenheit to celcius

    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 10)), "test", "data", "time_series_data", "VE02_FTI_KALK_SV.csv")
    VE02_FTI_KALK_SV = load_from_file(filename=filename, stepSize=stepSize, start_time=startPeriod, end_time=endPeriod, format=format, dt_limit=9999)
    # VE02_FTI_KALK_SV["FTI_KALK_SV"] = (VE02_FTI_KALK_SV["FTI_KALK_SV"]-32)*5/9 #convert from fahrenheit to celcius

    
    
    # primaryTemperatureIn can be calculated based on logged efficiency and temperature measurements.
    # However, primaryTemperatureIn should also be equal to outdoor temperature, which is available.
    # Outdoor temperature is therefore currently used. 
    primaryTemperatureIn = (VE02_efficiency["efficiency"]/100*VE02_FTU1["FTU1"]-VE02_FTG_MIDDEL["FTG_MIDDEL"])/(VE02_efficiency["efficiency"]/100-1)




    input.insert(0, "time", VE02_FTI_KALK_SV["Time stamp"])
    input.insert(0, "primaryAirFlowRate", VE02_primaryAirFlowRate["primaryAirFlowRate"])
    input.insert(0, "secondaryAirFlowRate", VE02_secondaryAirFlowRate["secondaryAirFlowRate"])
    # input.insert(0, "primaryTemperatureIn", primaryTemperatureIn)
    input.insert(0, "primaryTemperatureIn", weather["outdoorTemperature"])
    input.insert(0, "secondaryTemperatureIn", VE02_FTU1["FTU1"])
    input.insert(0, "primaryTemperatureOutSetpoint", VE02_FTI_KALK_SV["FTI_KALK_SV"])
    input.insert(0, "primaryTemperatureOut", VE02_FTG_MIDDEL["FTG_MIDDEL"])


    tol = 1e-5
    
    input.replace([np.inf, -np.inf], np.nan, inplace=True)
    input = (input.loc[(input["primaryAirFlowRate"]>tol) | (input["secondaryAirFlowRate"]>tol)]).dropna().reset_index(drop=True) # Filter data to remove 0 airflow data
    output = input["primaryTemperatureOut"].to_numpy()
    input.drop(columns=["primaryTemperatureOut"])
    return (input,output)

if __name__ == '__main__':
    #use id as used into id = "AirToAirHeatRecovery"
    AirToAirHeatRecovery_units = {"AirToAirHeatRecovery_1":
                                {"input_filename":"",
                                "output_filename" :""
                                },
                            }
    calibrated_variable_dict = {}

    for AirToAirHeatRecovery_unit in AirToAirHeatRecovery_units.keys():
        input_X,output_Y = read_data()
        air_to_heat_recovery_cls_obj = dynamic_calibration_heat_recovery(input_X,output_Y)
        calibrated_variable_dict[AirToAirHeatRecovery_unit] = air_to_heat_recovery_cls_obj.calibrate_results()

    calibrated_full_path = calibrated_path+"/calibrated_air_to_heat_recovery_parameters.json"
    with open(calibrated_full_path, "w") as outfile:
        json.dump(calibrated_variable_dict, outfile)