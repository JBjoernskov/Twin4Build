

import os
import sys
import datetime
from dateutil.tz import tzutc

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 3)
    sys.path.append(file_path)

# from twin4build.saref.date_time.date_time import DateTime
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
import twin4build.utils.building_data_collection_dict as building_data_collection_dict ########
import twin4build.utils.plot as plot
from twin4build.saref4bldg.physical_object.building_object.building_device.distribution_device.distribution_flow_device.flow_terminal.space_heater.space_heater_model import SpaceHeaterModel

def test():
    createReport = True
    do_plot = False
    timeStep = 600 #Seconds
    startPeriod = datetime.datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tzutc())
    endPeriod = datetime.datetime(year=2018, month=1, day=5, hour=0, minute=0, second=0, tzinfo=tzutc())
    model = Model(timeStep = timeStep,
                        startPeriod = startPeriod,
                        endPeriod = endPeriod,
                        createReport = createReport)
    model.load_model()
    model.draw_system_graph()
    model.get_execution_order()
    model.init_building_space_models()


    
    model.draw_system_graph_no_cycles()
    model.draw_flat_execution_graph()


    


    simulator = Simulator(timeStep = timeStep,
                            startPeriod = startPeriod,
                            endPeriod = endPeriod,
                            do_plot = do_plot)

    del building_data_collection_dict.building_data_collection_dict
    
    import time
    time_start = time.time()
    simulator.simulate(model)
    print(time.time()-time_start)



    # component = model.get_component_by_class(model.component_dict, SpaceHeaterModel)[0]
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # time_list = list(np.linspace(0,143*600,144))


    # m_w = np.array(component.savedInput["waterFlowRate"]) + np.random.randn(144)/100
    # m_w[m_w<0] = 0
    # T_z = np.array(component.savedInput["indoorTemperature"])
    # T_r = np.array(component.savedOutput["radiatorOutletTemperature"]) + np.random.randn(144)

    # input_dict = {"m_w": m_w,
    #                 "T_z": T_z}
    # output_dict = {"T_r": T_r}

    # df_input = pd.DataFrame(input_dict, index = time_list)
    # df_output = pd.DataFrame(output_dict, index = time_list)

    # df_input.plot(y="m_w", kind='line')	
    # df_input.plot(y="T_z", kind='line')	
    # df_input.to_csv("test_input.csv",index=True, index_label="time")

    # df_output.plot(y="T_r", kind='line')	
    # df_output.to_csv("test_output.csv",index=True, index_label="time")
    # plt.show()


    space_name = "Space"
    space_heater_name = "Space heater"
    temperature_controller_name = "Temperature controller"
    CO2_controller_name = "CO2 controller"
    air_to_air_heat_recovery_name = "Air to air heat recovery"
    heating_coil_name = "Heating coil"
    supply_fan_name = "Supply fan"
    damper_name = "Supply damper"

    plot.plot_space_temperature(model, simulator, space_name)
    plot.plot_space_CO2(model, simulator, space_name)
    plot.plot_weather_station(model, simulator)
    plot.plot_space_heater(model, simulator, space_heater_name)
    plot.plot_space_heater_energy(model, simulator, space_heater_name)
    plot.plot_temperature_controller(model, simulator, temperature_controller_name)
    plot.plot_CO2_controller(model, simulator, CO2_controller_name)
    plot.plot_heat_recovery_unit(model, simulator, air_to_air_heat_recovery_name)
    plot.plot_heating_coil(model, simulator, heating_coil_name)
    plot.plot_supply_fan(model, simulator, supply_fan_name)
    plot.plot_supply_fan_energy(model, simulator, supply_fan_name)
    plot.plot_space_wDELTA(model, simulator, space_name)
    plot.plot_supply_damper(model, simulator, damper_name)
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    import cProfile
    import pstats
    import io
    from line_profiler import LineProfiler

    # pr = cProfile.Profile()
    # pr.enable()


    # lp = LineProfiler()
    # lp.add_function(BuildingSpaceModel.get_temperature)
    # lp.add_function(BuildingSpaceModel.do_step)   # add additional function to profile
    # lp_wrapper = lp(test)
    # lp_wrapper()
    # lp.print_stats()

    test()

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()
    # with open('profile.txt', 'w+') as f:
    #     f.wrrite(s.getvalue())