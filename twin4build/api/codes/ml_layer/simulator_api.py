"""

This code is main code to run simulations as an API 

"""

# importing modules
import json
import os
import sys
import datetime
import numpy as np
import pandas as pd

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 5)
    sys.path.append(file_path)
    
# importing custom modules
from twin4build.utils.uppath import uppath
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator


from twin4build.config.Config import ConfigReader
from twin4build.logger.Logging import Logging
from twin4build.api.models.VE01_ventilation_model import model_definition, get_total_airflow_rate

from fastapi import FastAPI
from fastapi import FastAPI, Request,Body, APIRouter
from fastapi import Depends, HTTPException


from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# logging module 
logger = Logging.get_logger("API_logfile")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],)

def get_configuration():
    logger.info("[SimulatorAPI] : Entered in get_configuration Function")
    config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
    conf=ConfigReader()
    config=conf.read_config_section(config_path)
    logger.info("[SimulatorAPI] : Exited from get_configuration Function")
    return (config)
            
def get_simulation_result(simulator):
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
    df_output = df_output.fillna('')
    simulation_result_dict = df_output.to_dict(orient="list")
    df_measuring_devices.to_dict(orient="list")

    logger.info("[SimulatorAPI] : Exited from get_simulation_result Function")
    return simulation_result_dict

def get_ventilation_simulation_result(simulator):
    logger.info("[SimulatorAPI] : Entered in get_ventilation_simulation_result Function")
    model = simulator.model
    df_output = pd.DataFrame()
    df_output.insert(0, "time", simulator.dateTimeSteps)

    for component in model.component_dict.values():
        
        if component.id == "Total_AirFlow_sensor":
            total_airflow_series = get_total_airflow_rate(model)
            df_output = df_output.join(pd.DataFrame({"Sum_of_damper_air_flow_rates": total_airflow_series}))
            continue

        for property_, arr in component.savedOutput.items():
            column_name = f"{component.id}_{property_}"
            column_name = column_name.replace(" ","")
            column_name = column_name.replace("|||","_")
            column_name = column_name.split("|")[-1]
            df_output = df_output.join(pd.DataFrame({column_name: arr}))
    
    df_output = df_output.fillna('')
    #Retrieve all the series data from the df_output dataframe which contains the "Supply_damper" substring in the name
    df_output_supply_dampers = df_output.filter(like="Supply_damper", axis=1)
    
    simulation_result_dict = {}
    #Two main sub-dicts are created in the simulation_result_dict, common_data and rooms_data.
    #common_data contains the values from the series Sum_of_damper_air_flow_rates and the simulation time steps
    #rooms_data contains the values from the series that contain the substring "Supply_damper" in the name.
    #The keys of the rooms_data dict are the room names, and the names are extracted from the column names of the df_output_supply_dampers the name of the room is the substring of the column name that is after the first underscore.
    simulation_result_dict["common_data"] = {"Sum_of_damper_air_flow_rates": df_output["Sum_of_damper_air_flow_rates"].to_list(), "Simulation_time": df_output["time"].to_list()}
    simulation_result_dict["rooms"] = {}
    room_names = []
    for column_name in df_output_supply_dampers.columns:
        #The column name looks like Supply_damper_22_601b_00_airflowrate, extract the substring between the second and fifth underscore to get the room name
        room_name = "_".join(column_name.split("_")[2:5])
        if room_name not in room_names:
            room_names.append(room_name)
            simulation_result_dict["rooms"][room_name] = {}
        # Add the values of all the columns that contain the substring of room_name in the name to the room_name key in the rooms_data dict
        sub_dict = df_output_supply_dampers[column_name].to_list()
        #make the sub_dict a list of floats
        # Check each item in sub_dict
        sub_dict = [float(x.item()) if isinstance(x, np.ndarray) else float(x) for x in sub_dict]
        
        simulation_result_dict["rooms"][room_name][column_name] = sub_dict

    #simulation_result_dict = df_output.to_dict(orient="list")

    logger.info("[SimulatorAPI] : Exited from get_ventilation_simulation_result Function")
    return simulation_result_dict

@app.post("/simulate")
async def run_simulation(input_dict: dict):
    "Method to run simulation and return dict response"
    # try:
    logger.info("[run_simulation] : Entered in run_simulation Function")
    input_dict_loaded = input_dict
    filename_data_model = config['model']['filename']
    filename = os.path.join(uppath(os.path.abspath(__file__), 4), "model", "tests", filename_data_model)
    logger.info("[temp_run_simulation] : Entered in temp_run_simulation Function")

    model = Model(id="model", saveSimulationResult=True)
    model.load_model(semantic_model_filename=filename, input_config=input_dict_loaded, infer_connections=True)

    startTime = datetime.datetime.strptime(input_dict_loaded["metadata"]["start_time"], time_format)
    endTime = datetime.datetime.strptime(input_dict_loaded["metadata"]["end_time"], time_format)
    

    stepSize = int(config['model']['stepsize'])
    sensor_inputs = input_dict_loaded["inputs_sensor"]
    weather_inputs = sensor_inputs["ml_inputs_dmi"]


    simulator = Simulator(model=model)
    
    simulator.simulate(model=model,
                    startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize)

    simulation_result_dict = get_simulation_result(simulator)
    logger.info("[run_simulation] : Sucessfull Execution of API ")
    return simulation_result_dict
    # except Exception as api_error:
    #     print("Error during api calling, Error is %s: " %api_error)
    #     logger.error("Error during API call. Error is %s "%api_error)
    #     msg = "An error has been occured during API call please check. Error is %s"%api_error
    #     return(msg)
    
@app.post("/simulate_ventilation")   
async def run_simulation_for_ventilation(input_dict: dict):
    "Method to run simulation for ventilation system and return dict response"
    # try:
    logger.info("[run_simulation] : Entered in run_simulation Function")

            #load Model
            model = Model(id="VE01_model", saveSimulationResult=True)
            model.load_model(fcn=model_definition,input_config=input_dict, infer_connections=False, do_load_parameters=False)

    startTime = datetime.datetime.strptime(input_dict["metadata"]["start_time"], time_format)
    endTime = datetime.datetime.strptime(input_dict["metadata"]["end_time"], time_format)

    #stepSize = int(input_dict['metadata']['stepsize'])
    stepSize = 600

    simulator = Simulator(model=model)

    simulator.simulate(model=model,startTime=startTime,endTime=endTime, stepSize=stepSize)

    simulation_result_dict = get_ventilation_simulation_result(simulator)

    #Make sure the dictionary is json serializable
    simulation_result_dict = json.loads(json.dumps(simulation_result_dict, default=str))
    

    logger.info("[run_ventilation_simulation] : Sucessfull Execution of API ")
    return simulation_result_dict
    # except Exception as api_error:
    #     print("Error during ventilation api calling, Error is %s: " %api_error)
    #     logger.error("Error during ventilation API call. Error is %s "%api_error)
    #     msg = "An error has been occured during ventilation API call please check. Error is %s"%api_error
    #     return(msg)

if __name__ == "__main__":
    config = get_configuration()
    time_format = '%Y-%m-%d %H:%M:%S%z'
    
    #Start the FastAPI server
    logger.info("[main]: app is running at 8070 port")
    uvicorn.run(app, host="127.0.0.1", port=8070)
    