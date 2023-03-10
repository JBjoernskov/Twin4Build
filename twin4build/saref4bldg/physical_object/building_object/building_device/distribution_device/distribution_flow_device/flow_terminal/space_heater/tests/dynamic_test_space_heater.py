import os
import sys
import json
import datetime
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
    #print(file_path)
    sys.path.append(file_path)

    calibrated_path = file_path+"/calibrated_folder"
    if not os.path.exists(calibrated_path):
         os.makedirs(calibrated_path)


from twin4build.utils.data_loaders.load_from_file import load_from_file
from twin4build.utils.preprocessing.data_collection import DataCollection
from twin4build.utils.preprocessing.data_preparation import sample_data
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_model import SpaceHeaterModel
from twin4build.saref.measurement.measurement import Measurement
from twin4build.utils.constants import Constants

class dynamic_calibration:
    def __init__(self,input_data,output_data):
        self.input_data  = input_data
        self.output_data = output_data
        self.model_set_parameters()
        self.data_prep_method()
        self.save_plots()

    def model_set_parameters(self):
        """ This method set parameters of space heater model """
        self.stepSize = 600
        self.space_heater = SpaceHeaterModel(
                    outputCapacity = Measurement(hasValue=2689),
                    temperatureClassification = "45/30-21",
                    thermalMassHeatCapacity = Measurement(hasValue=50000),
                    stepSize = self.stepSize,
                    saveSimulationResult = True,
                    id = "space_heater")
        
        self.waterFlowRateMax = abs(self.space_heater.outputCapacity.hasValue/Constants.specificHeatCapacity["water"]/(self.space_heater.nominalSupplyTemperature-self.space_heater.nominalReturnTemperature))

    def data_prep_method(self):
        """We can converting data into desired format"""
        self.input_data["waterFlowRate"] = self.input_data["waterFlowRate"]*self.waterFlowRateMax
        self.input_data["supplyWaterTemperature"] = 40
        self.input_data = self.input_data.set_index("time")

        self.output_data = self.output_data["Power"].to_numpy()*1000
        self.output_data = np.cumsum(self.output_data*self.stepSize/3600/1000)


    def save_plots(self):
        self.space_heater.initialize()
        """This method is temp cause finally we might comment this method """
        start_pred = self.space_heater.do_period(self.input_data,stepSize=self.stepSize) ####sss
        fig, ax = plt.subplots(2)
        ax[0].plot(start_pred, color="black", linestyle="dashed", label="predicted")
        ax[0].plot(self.output_data, color="blue", label="Measured")
        ax[0].set_title('Before calibration')
        fig.legend()
        self.input_data.plot(subplots=True)
        end_pred = self.space_heater.do_period(self.input_data,stepSize=self.stepSize)
        ax[1].plot(end_pred, color="black", linestyle="dashed", label="predicted")
        ax[1].plot(self.output_data, color="blue", label="Measured")
        ax[1].set_title('After calibration')
        plt.show()
        #plt.savefig('plots_temp.png')

    def calibrate_results(self):
        return(self.space_heater.calibrate(self.input_data, self.output_data, stepSize=self.stepSize))

def read_data(input_filename,output_filename):
        """Currently we are using csv files to ingest data for further use we might have to change this method """
        filename_input = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "radiator_input.csv")
        filehandler_input = open(filename_input, 'rb')
        input_data = pd.read_csv(filehandler_input, low_memory=False)

        filename_output = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "radiator_output.csv")
        filehandler_output = open(filename_output, 'rb')
        output_data = pd.read_csv(filehandler_output, low_memory=False)
        return input_data,output_data



if __name__ == '__main__':

    #use id as used into id = "space_heater"
    total_room_heater_ids = {"room_heater_1":
                                {"input_filename":"radiator_input.csv",
                                "output_filename" :"radiator_output.csv"
                                },
                            "room_heater_2":
                                {"input_filename":"radiator_input.csv",
                                "output_filename":"radiator_output.csv"
                                }
                            }
    calibrated_variable_dict = {}

    for room_heater_id in total_room_heater_ids.keys():
        input_data,output_data = read_data(total_room_heater_ids[room_heater_id]['input_filename'],
                                        total_room_heater_ids[room_heater_id]['output_filename'])
        cls_obj = dynamic_calibration(input_data,output_data)
        calibrated_variable_dict[room_heater_id] = cls_obj.calibrate_results()


    calibrated_full_path = calibrated_path+"/calibrated_space_heater_parameters.json"
    with open(calibrated_full_path, "w") as outfile:
        json.dump(calibrated_variable_dict, outfile)