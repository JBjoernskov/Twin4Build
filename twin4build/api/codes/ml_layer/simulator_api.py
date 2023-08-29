import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.ticker as ticker
import json

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
    
from twin4build.utils.uppath import uppath
from twin4build.monitor.monitor import Monitor
from twin4build.model.model import Model
from twin4build.utils.plot.plot import bar_plot_line_format
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node


from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging

from fastapi import FastAPI
from fastapi import FastAPI, Request,Body, APIRouter
from fastapi import Depends, HTTPException

from fastapi.middleware.cors import CORSMiddleware
# import uvicorn

logger = Logging.get_logger("ai_logfile")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulatorAPI:
    "Using this class we are going to run all codes/methods of twin4build as an API  "
    def __init__(self):
        logger.info("[execute_methods] : Entered in Initialise Function")
        self.config = self.get_configuration()

    def get_configuration(self):
        config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
        conf=ConfigReader()
        config=conf.read_config_section(config_path)
        return (config)

    def run_performance_monitoring(self):
        "Using this method we are going to run performance code "
        pass

    def convert_simulation_result_to_json_response(self, simulation_result_dict):
        "Method to convert simulation results to json"
        ##############################################
        # This can be deleted once the code is ready #
        with open('simulation_result.json', 'w') as fp:
            json.dump(simulation_result_dict, fp)
        ##############################################
        simulation_result_json = json.dumps(simulation_result_dict)
        return simulation_result_json
        
    def get_simulation_result(self, simulator):
        model = simulator.model
        df_input = pd.DataFrame()
        df_output = pd.DataFrame()
        df_input.insert(0, "time", simulator.dateTimeSteps)
        df_output.insert(0, "time", simulator.dateTimeSteps)

        for component in model.component_dict.values():
            for property_, arr in component.savedInput.items():
                column_name = f"{component.id} ||| {property_}"
                df_input = df_input.join(pd.DataFrame({column_name: arr}))

            for property_, arr in component.savedOutput.items():
                column_name = f"{component.id} ||| {property_}"
                df_output = df_output.join(pd.DataFrame({column_name: arr}))

        df_measuring_devices = simulator.get_simulation_readings()

        df_input.set_index("time").to_dict(orient="list")
        simulation_result = df_output.set_index("time").to_dict(orient="list")
        df_measuring_devices.to_dict(orient="list")

        return simulation_result_dict

        

    def temp_run_simulation(self, input_dict=None):
        "Temporary method to run simulation and return json response. Eventually, this code can be copied into run_simulation method"

        ############### CONVERTING TO DICT ###############
        # This can be deleted once the code is ready #
        filename_input = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 1)), "input_data.json")
        with open(filename_input, 'r') as j:
            input_dict_loaded = json.loads(j.read())
        ##################################################

        filename_data_model = self.config['model']['filename']

        logger.info("[temp_run_simulation] : Entered in temp_run_simulation Function")
        model = Model(id="model", saveSimulationResult=True)
        model.load_model(datamodel_config_filename=filename_data_model, input_config=input_dict_loaded, infer_connections=True)

        startPeriod = datetime.datetime.strptime(input_dict_loaded["metadata"]["start_time"], '%Y-%m-%d %H:%M:%S')
        endPeriod = datetime.datetime.strptime(input_dict_loaded["metadata"]["end_time"], '%Y-%m-%d %H:%M:%S')
        stepSize = int(self.config['model']['stepSize'])

        simulator = Simulator(model=model,
                            do_plot=False)
        
        simulator.simulate(model=model,
                        startPeriod=startPeriod,
                        endPeriod=endPeriod,
                        stepSize=stepSize)

        simulation_result_dict = self.get_simulation_result(simulator)
        simulation_result_json = self.convert_simulation_result_to_json_response(simulation_result_dict)

        return simulation_result_json



  
    @app.post("/simulate")
    async def run_simulation(self,input_dict: dict):
        "Method to run simulation and return dict response"

        logger.info("[run_simulation] : Entered in run_simulation Function")




        #Model.extend_model = extend_model
        self.model = Model(id="model", saveSimulationResult=True)
        filename = self.config['model']['filename']
        #filename = "configuration_template_1space_BS2023.xlsx"
        self.model.load_BS2023_model(filename)

        stepSize = self.config['model']['stepSize'] #Seconds 

        #code here to convert period into proper datetime format
        startPeriod = self.config['model']['startPeriod']
        endPeriod = self.config['model']['endPeriod']

        # Parse date and time strings into datetime objects
        startPeriod = datetime.strptime(startPeriod, '%Y-%m-%d %H:%M:%S')
        endPeriod = datetime.strptime(endPeriod, '%Y-%m-%d %H:%M:%S')

        # Running simulation
        self.simulator.simulate(self.model,stepSize,startPeriod,endPeriod)
        
        #reading simulation results
        self.df_simulation_readings = self.simulator.get_simulation_readings()

        # code here to convert simulation results into json format
        #self.simution_json_response = self.convert_simulation_to_json_response()
        logger.info("[run_simulation] : Exited from run_simulation Function")

        # return Simulation results as API response
        result_dict = self.df_simulation_readings
        return result_dict

        

if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(app, host="localhost", port=8005)
    logger.info("[main]: app is running at 8005 port")