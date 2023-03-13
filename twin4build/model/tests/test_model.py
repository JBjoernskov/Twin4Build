import os
import sys
import datetime
from dateutil.tz import tzutc
import pandas as pd
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
import twin4build.utils.plot.plot as plot

def test():
    stepSize = 600 #Seconds
    startPeriod = datetime.datetime(year=2021, month=12, day=10, hour=0, minute=0, second=0) #piecewise 20.5-23
    endPeriod = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0) #piecewise 20.5-23
    model = Model(saveSimulationResult=True)
    model.load_model()
    model.prepare_for_simulation()
    simulator = Simulator()
    simulator.simulate(model,
                        stepSize=stepSize,
                        startPeriod = startPeriod,
                        endPeriod = endPeriod)

    space_name = "Space"
    space_heater_name = "Space heater"
    temperature_controller_name = "Temperature controller"
    CO2_controller_name = "CO2 controller"
    air_to_air_heat_recovery_name = "Air to air heat recovery"
    heating_coil_name = "Heating coil"
    supply_fan_name = "Supply fan"
    damper_name = "Supply damper"

    # plot.plot_space_temperature(model, simulator, space_name)
    plot.plot_space_CO2(model, simulator, space_name)
    # plot.plot_weather_station(model, simulator)
    # plot.plot_space_heater(model, simulator, space_heater_name)
    # plot.plot_space_heater_energy(model, simulator, space_heater_name)
    # plot.plot_temperature_controller(model, simulator, temperature_controller_name)
    # plot.plot_CO2_controller(model, simulator, CO2_controller_name)
    # plot.plot_heat_recovery_unit(model, simulator, air_to_air_heat_recovery_name)
    # plot.plot_heating_coil(model, simulator, heating_coil_name)
    # plot.plot_supply_fan(model, simulator, supply_fan_name)
    # plot.plot_supply_fan_energy(model, simulator, supply_fan_name)
    # plot.plot_supply_fan_energy(model, simulator, "Exhaust fan")
    # plot.plot_space_wDELTA(model, simulator, space_name)
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