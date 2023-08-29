import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.ticker as ticker


###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
    
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
import uvicorn

logger = Logging.get_logger("ai_logfile")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class execute_methods:
    "Using this class we are going to run all codes/methods of twin4build as an API "
    def __init__(self):
        logger.info("[execute_methods] : Entered in Initialise Function")
        self.config = self.get_configuration()

    def get_configuration(self):
        # Read configuration using ConfigReader
        config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 5)), "config", "conf.ini")
        conf=ConfigReader()
        config=conf.read_config_section(config_path)
        return config

  
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