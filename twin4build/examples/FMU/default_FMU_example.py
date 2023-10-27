
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import time
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)

from twin4build.utils.fmu.fmu_component import FMUComponent
from twin4build.utils.uppath import uppath

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")


from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_system import SpaceHeaterSystem
from twin4build.saref.measurement.measurement import Measurement

if __name__=="__main__":

    logger.info("[default_FMU_example] : Entered in Main Function")

    style_list = ["-"]
    step_size_list = [70]

    for style,step_size in zip(style_list, step_size_list):
        fmu_filename = "Radiator.fmu"
        file_path = os.path.join(uppath(os.path.abspath(__file__), 1), fmu_filename)

        # component = DefaultFMUComponent(r"C:\Users\jabj\AppData\Local\Temp\1\OpenModelica\OMEdit\ExampleFMU\ExampleFMU.fmu")
        print(file_path)
        component = FMUComponent(start_time=0, fmu_filename=file_path)
        print(component.parameters["Radiator.n"].variability)
        parameters = {"Q_flow_nominal": 1000}
        component.set_parameters(parameters=parameters)
        # component.initialize(start_time=0)


        component.get_paramet
        

        # n = 10

        # step_size = 1
        start_time = 0
        stop_time = 10000

        n = math.floor((stop_time-start_time)/step_size)
        stop_time = n*step_size+start_time
        time_list = list(np.linspace(start=start_time,stop=stop_time,num=n))

        m_w_list = list(np.linspace(start=start_time,stop=stop_time,num=n))

        T_z_list = list(20+273.15+np.random.randn(n)/10)

        # print(time_list)
        #     results = []

        m_w_max = 0.012
        start_time_meas = time.time()
        for i,time_ in enumerate(time_list):
            component.inputs = {"T_w_in":45+273.15, "T_z": T_z_list[i], "m_w": m_w_max if time_<5000 else 0}
            if i<n-1:
                step_size = time_list[i+1]-time_
                component.do_step(time=time_, step_size=step_size)
                for key in component.inputs.keys():
                    component.results[key].append(component.inputs[key])
                for key in component.outputs.keys():
                    component.results[key].append(component.outputs[key])
        print("FMU time: ", time.time()-start_time_meas)
        logger.info("[default_FMU_example] : FMU time: ", time.time()-start_time_meas)

        # Q_tot = -np.array(component.results["Q_con"])-np.array(component.results["Q_rad"])
        # plt.plot(time_list[:-1], component.results["Q_con"], color="red", label="Q_con", linestyle = style)
        # plt.plot(time_list[:-1], component.results["Q_rad"], color="blue", label="Q_rad", linestyle = style)
        fig, ax = plt.subplots()
        ax.plot(time_list[:-1], component.results["Q_tot"], color="red", label="Q_FMU", linestyle = style)
        ax_twin = plt.twinx()
        ax_twin.plot(time_list[:-1], np.array(component.results["T_w_out"])-273.15, color="green", label="T_w_out_FMU", linestyle = style)
        ax_twin.plot(time_list, np.array(T_z_list)-273.15, color="green", label="T_z", linestyle = style)





    style_list = ["-"]
    step_size_list = [70]
    for style,step_size in zip(style_list, step_size_list):
        space_heater = SpaceHeaterSystem(
                    outputCapacity = Measurement(hasValue=500),
                    temperatureClassification = "40/30-20",
                    thermalMassHeatCapacity = Measurement(hasValue=18000),
                    specificHeatCapacityWater = Measurement(hasValue=4180),
                    stepSize = step_size,
                    input = {"supplyWaterTemperature": 45},
                    output = {"radiatorOutletTemperature": 20,
                    "Energy":  0},
                    id = "test",
                    saveSimulationResult=True,
                    savedInput = {},
                    savedOutput = {},)

        n = math.floor((stop_time-start_time)/step_size)
        stop_time = n*step_size+start_time
        time_list = list(np.linspace(start=start_time,stop=stop_time,num=n))
        m_w_list = list(np.linspace(start=start_time,stop=stop_time,num=n))
        T_z_list = list(20+273.15+np.random.randn(n)/10)

        # print(time_list)
        #     results = []

        m_w_max = 0.012
        start_time_meas = time.time()
        for i,time_ in enumerate(time_list):
            space_heater.input = {"supplyWaterTemperature":45, "indoorTemperature": T_z_list[i]-273.15, "waterFlowRate": m_w_max if time_<5000 else 0}
            space_heater.do_step()
            space_heater.update_simulation_reult()
        print("SAREF time: ", time.time()-start_time_meas)
        logger.info("[default_FMU_example] : SAREF time: ", time.time()-start_time_meas)

        ax.plot(time_list[1:], space_heater.savedOutput["Power"][:-1], color="blue", label="Q_SAREF", linestyle = style)
    fig.legend()
    plt.show()
    logger.info("[default_FMU_example] : Exited in Main Function")


    # df = pd.DataFrame(component.result_dict)
    # df.insert(0, "time", time_list[:-1]) 
    # df.set_index("time")

    # df.plot()
    # plt.show()

    # print(df)