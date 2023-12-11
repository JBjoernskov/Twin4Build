"""

This code is main code to run simulations as an API 

"""

# importing modules
import os
import sys
import datetime
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

from fastapi import FastAPI
from fastapi import FastAPI, Request,Body, APIRouter
from fastapi import Depends, HTTPException

from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# logging module 
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
        logger.info("[run_simulation] : Entered in run_simulation Function")
        input_dict_loaded = input_dict
        filename_data_model = self.config['model']['filename']
        logger.info("[temp_run_simulation] : Entered in temp_run_simulation Function")

        model = Model(id="model", saveSimulationResult=True)
        model.load_model(semantic_model_filename=filename_data_model, input_config=input_dict_loaded, infer_connections=True)

        startTime = datetime.datetime.strptime(input_dict_loaded["metadata"]["start_time"], '%Y-%m-%d %H:%M:%S')
        endTime = datetime.datetime.strptime(input_dict_loaded["metadata"]["end_time"], '%Y-%m-%d %H:%M:%S')
        stepSize = int(self.config['model']['stepsize'])

        print("startTime", startTime)
        print("endTime", endTime)

        sensor_inputs = input_dict_loaded["inputs_sensor"]
        weather_inputs = sensor_inputs["ml_inputs_dmi"]
        print("weather timestamps")
        print(weather_inputs["observed"])
        

        simulator = Simulator(model=model)
        
        simulator = Simulator(model=model,
                            do_plot=False)
        
        simulator.simulate(model=model,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)
        
        ######### THIS WAS USED FOR TESTING #########
        import twin4build.utils.plot.plot as plot
        import matplotlib.pyplot as plt
        import numpy as np
        axes = plot.plot_space_wDELTA(model, simulator, "OE20-601b-2")
        time_format = '%Y-%m-%d %H:%M:%S%z'
        time = np.array([datetime.datetime.strptime(t, time_format) for t in input_dict["inputs_sensor"]["ml_inputs"]["opcuats"]])
        float_x = [float(x) for x in input_dict["inputs_sensor"]["ml_inputs"]["temperature"]]
        x = np.array(float_x)
        epoch_timestamp = np.vectorize(lambda data:datetime.datetime.timestamp(data)) (time)
        sorted_idx = np.argsort(epoch_timestamp)
        axes[0].plot(time[sorted_idx], x[sorted_idx], color="green")

        axes = plot.plot_space_CO2(model, simulator, "OE20-601b-2")
        float_x = [float(x) for x in input_dict["inputs_sensor"]["ml_inputs"]["co2concentration"]]
        x = np.array(float_x)
        epoch_timestamp = np.vectorize(lambda data:datetime.datetime.timestamp(data)) (time)
        sorted_idx = np.argsort(epoch_timestamp)
        axes[0].plot(time[sorted_idx], x[sorted_idx], color="green")
        # x_start = endTime-datetime.timedelta(days=8)
        # x_end = endTime
        # for ax in axes:
        #     ax.set_xlim([x_start, x_end])
        #plt.show()
        ###########################################

        simulation_result_dict = self.get_simulation_result(simulator)
        #simulation_result_json = self.convert_simulation_result_to_json_response(simulation_result_dict)
        logger.info("[run_simulation] : Sucessfull Execution of API ")
        return simulation_result_dict
        """except Exception as api_error:
            print("Error during api calling, Error is %s: " %api_error)
            logger.error("Error during API call. Error is %s "%api_error)
            msg = "An error has been occured during API call please check. Error is %s"%api_error
            return(msg)"""

if __name__ == "__main__":
    app_instance = SimulatorAPI()
    #Start the FastAPI server
    uvicorn.run(app_instance.app, host="127.0.0.1", port=8070)
    logger.info("[main]: app is running at 8070 port")