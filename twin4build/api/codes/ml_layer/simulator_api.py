"""

This code is main code to run simulations as an API 

"""


import os
import sys
import datetime
import pandas as pd

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
    
from twin4build.utils.uppath import uppath
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator

from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging

from fastapi import FastAPI
from fastapi import FastAPI, Request,Body, APIRouter
from fastapi import Depends, HTTPException

from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = Logging.get_logger("API_logfile")

class SimulatorAPI:
    "Using this class we are going to run all codes/methods of twin4build as an API  "
    def __init__(self):

        logger.info("[SimulatorAPI] : Entered in Initialise Function")
        self.config = self.get_configuration()
        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],)
        
        self.app.post("/simulate")(self.run_simulation)
        logger.info("[SimulatorAPI] : Exited from Initialise Function")

    def get_configuration(self):
        logger.info("[SimulatorAPI] : Entered in get_configuration Function")
        config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
        conf=ConfigReader()
        config=conf.read_config_section(config_path)
        logger.info("[SimulatorAPI] : Exited from get_configuration Function")
        return (config)
        
        
    def get_simulation_result(self, simulator):
        logger.info("[SimulatorAPI] : Entered in get_simulation_result Function")
        model = simulator.model
        df_input = pd.DataFrame()
        df_output = pd.DataFrame()
        df_input.insert(0, "time", simulator.dateTimeSteps)
        df_output.insert(0, "time", simulator.dateTimeSteps)

        for component in model.component_dict.values():
            for property_, arr in component.savedInput.items():
                column_name = f"{component.id}_{property_}"
                column_name = column_name.replace(" ","")
                column_name = column_name.replace("|||","_")
                column_name = column_name.split("|")[-1]
                df_input = df_input.join(pd.DataFrame({column_name: arr}))

            for property_, arr in component.savedOutput.items():
                column_name = f"{component.id}_{property_}"
                column_name = column_name.replace(" ","")
                column_name = column_name.replace("|||","_")
                column_name = column_name.split("|")[-1]
                df_output = df_output.join(pd.DataFrame({column_name: arr}))

        df_measuring_devices = simulator.get_simulation_readings()

        df_input.set_index("time").to_dict(orient="list")
        simulation_result_dict = df_output.to_dict(orient="list")
        df_measuring_devices.to_dict(orient="list")

        logger.info("[SimulatorAPI] : Exited from get_simulation_result Function")
        return simulation_result_dict


    async def run_simulation(self,input_dict: dict):
        "Method to run simulation and return dict response"
        try:
            logger.info("[run_simulation] : Entered in run_simulation Function")
            input_dict_loaded = input_dict
            filename_data_model = self.config['model']['filename']
            logger.info("[temp_run_simulation] : Entered in temp_run_simulation Function")
            model = Model(id="model", saveSimulationResult=True)
            model.load_model(datamodel_config_filename=filename_data_model, input_config=input_dict_loaded, infer_connections=True)

            startPeriod = datetime.datetime.strptime(input_dict_loaded["metadata"]["start_time"], '%Y-%m-%d %H:%M:%S')
            endPeriod = datetime.datetime.strptime(input_dict_loaded["metadata"]["end_time"], '%Y-%m-%d %H:%M:%S')
            stepSize = int(self.config['model']['stepsize'])
            

            simulator = Simulator(model=model,
                                do_plot=False)
            
            simulator.simulate(model=model,
                            startPeriod=startPeriod,
                            endPeriod=endPeriod,
                            stepSize=stepSize)

            simulation_result_dict = self.get_simulation_result(simulator)
            #simulation_result_json = self.convert_simulation_result_to_json_response(simulation_result_dict)
            logger.info("[run_simulation] : Sucessfull Execution of API ")
            return simulation_result_dict
        except Exception as api_error:
            print("Error during api calling, Error is %s: " %api_error)
            logger.error("Error during API call. Error is %s "%api_error)
            msg = "An error has been occured during API call please check. Error is %s"%api_error
            return(msg)

if __name__ == "__main__":
    app_instance = SimulatorAPI()
    #Start the FastAPI server
    uvicorn.run(app_instance.app, host="127.0.0.1", port=8070)
    logger.info("[main]: app is running at 80 port")