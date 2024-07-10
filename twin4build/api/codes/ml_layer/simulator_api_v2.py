"""
This code runs simulations as an API.
"""

# Importing standard libraries
import json
import os
import sys
import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Body, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import traceback

# Importing custom modules
from twin4build.logger.Logging import Logging
from twin4build.utils.uppath import uppath
import twin4build as tb
from twin4build.config.Config import ConfigReader
from twin4build.api.models.VE01_ventilation_model import model_definition, get_total_airflow_rate
from twin4build.api.models.OE20_601b_2_model import fcn

# Setting up logging
logger = Logging.get_logger("API_logfile")

# Creating FastAPI app
app = FastAPI()

# Adding middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to get configuration
def get_configuration():
    logger.info("[SimulatorAPI] : Entered in get_configuration Function")
    config_path = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 4)), "config", "conf.ini")
    conf = ConfigReader()
    config = conf.read_config_section(config_path)
    logger.info("[SimulatorAPI] : Exited from get_configuration Function")
    return config

# Function to get simulation result
def get_simulation_result(simulator):
    logger.info("[SimulatorAPI] : Entered in get_simulation_result Function")
    model = simulator.model
    df_input = pd.DataFrame()
    df_output = pd.DataFrame()
    df_input.insert(0, "time", simulator.dateTimeSteps)
    df_output.insert(0, "time", simulator.dateTimeSteps)

    for component in model.component_dict.values():
        for property_, arr in component.savedInput.items():
            column_name = f"{component.id}_{property_}".replace(" ", "").replace("|||", "_").split("|")[-1]
            df_input = df_input.join(pd.DataFrame({column_name: arr}))

        for property_, arr in component.savedOutput.items():
            column_name = f"{component.id}_{property_}".replace(" ", "").replace("|||", "_").split("|")[-1]
            df_output = df_output.join(pd.DataFrame({column_name: arr}))

    df_measuring_devices = simulator.get_simulation_readings()
    df_input.set_index("time").to_dict(orient="list")
    df_output = df_output.fillna('')
    simulation_result_dict = df_output.to_dict(orient="list")
    df_measuring_devices.to_dict(orient="list")

    logger.info("[SimulatorAPI] : Exited from get_simulation_result Function")
    return simulation_result_dict

# Function to get ventilation simulation result
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
            column_name = f"{component.id}_{property_}".replace(" ", "").replace("|||", "_").split("|")[-1]
            df_output = df_output.join(pd.DataFrame({column_name: arr}))

    df_output = df_output.fillna('')
    df_output_supply_dampers = df_output.filter(like="Supply_damper", axis=1)

    simulation_result_dict = {
        "common_data": {
            "Sum_of_damper_air_flow_rates": df_output["Sum_of_damper_air_flow_rates"].to_list(),
            "Simulation_time": df_output["time"].to_list()
        },
        "rooms": {}
    }
    
    room_names = []
    for column_name in df_output_supply_dampers.columns:
        room_name = "_".join(column_name.split("_")[2:5])
        if room_name not in room_names:
            room_names.append(room_name)
            simulation_result_dict["rooms"][room_name] = {}

        sub_dict = df_output_supply_dampers[column_name].to_list()
        sub_dict = [float(x.item()) if isinstance(x, np.ndarray) else float(x) for x in sub_dict]
        
        simulation_result_dict["rooms"][room_name][column_name] = sub_dict

    logger.info("[SimulatorAPI] : Exited from get_ventilation_simulation_result Function")
    return simulation_result_dict

# API endpoint to run simulation
@app.post("/simulate")
async def run_simulation(input_dict: dict):
    "Method to run simulation and return dict response"
    try:
        logger.info("[run_simulation] : Entered in run_simulation Function")
        input_dict_loaded = input_dict
        filename_data_model = config['model']['filename']
        filename = os.path.join(uppath(os.path.abspath(__file__), 4), "model", "tests", filename_data_model)

        model = tb.Model(id="model", saveSimulationResult=True)
        model.load_model(input_config=input_dict_loaded, infer_connections=False, fcn=fcn, do_load_parameters=False)

        startTime = datetime.datetime.strptime(input_dict_loaded["metadata"]["start_time"], time_format)
        endTime = datetime.datetime.strptime(input_dict_loaded["metadata"]["end_time"], time_format)
        stepSize = int(config['model']['stepsize'])

        simulator = tb.Simulator(model=model)
        simulator.simulate(model=model, startTime=startTime, endTime=endTime, stepSize=stepSize)

        simulation_result_dict = get_simulation_result(simulator)
        logger.info("[run_simulation] : Successful Execution of API ")
        return simulation_result_dict
    except Exception as api_error:
        stack_trace = traceback.format_exc()
        print(stack_trace)  # Print stack trace to console
        logger.error("Error during API call. Error is %s \n%s" % (api_error, stack_trace))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=stack_trace)

# API endpoint to run ventilation simulation
@app.post("/simulate_ventilation")
async def run_simulation_for_ventilation(input_dict: dict):
    "Method to run simulation for ventilation system and return dict response"
    try:
        logger.info("[run_simulation] : Entered in run_simulation Function")

        model = tb.Model(id="VE01_model", saveSimulationResult=True)
        model.load_model(fcn=model_definition, input_config=input_dict, infer_connections=False, do_load_parameters=False)

        startTime = datetime.datetime.strptime(input_dict["metadata"]["start_time"], time_format)
        endTime = datetime.datetime.strptime(input_dict["metadata"]["end_time"], time_format)
        stepSize = 600

        simulator = tb.Simulator(model=model)
        simulator.simulate(model=model, startTime=startTime, endTime=endTime, stepSize=stepSize)

        simulation_result_dict = get_ventilation_simulation_result(simulator)
        simulation_result_dict = json.loads(json.dumps(simulation_result_dict, default=str))

        logger.info("[run_ventilation_simulation] : Successful Execution of API ")
        return simulation_result_dict
    except Exception as api_error:
        stack_trace = traceback.format_exc()
        print(stack_trace)  # Print stack trace to console
        logger.error("Error during ventilation API call. Error is %s \n%s" % (api_error, stack_trace))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=stack_trace)

# Main entry point
if __name__ == "__main__":
    config = get_configuration()
    time_format = '%Y-%m-%d %H:%M:%S%z'
    
    logger.info("[main]: app is running at port 8070")
    uvicorn.run(app, host="127.0.0.1", port=8070)
