import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import json
# Only for testing before distributing package.
# If the package is installed, this is not needed.
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node
from twin4build.logger.Logging import Logging
from twin4build.model.tests.test_LBNL_model import extend_model
logger = Logging.get_logger("ai_logfile")



def test():
    logger.info("[Test Estimator] : EXited from Test Function")
    
    stepSize = 60
    

    # startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=1, minute=0, second=0) 
    # endPeriod = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0)

    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)
    estimator = Estimator(model)



    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]

    # Dictinary inputs for the estimation/optimization algorithm. 
    # Each dictionary follow the following convention:
    # {object: "attribute name"}







# array([ 7.04131076e-01,  5.00000000e+00,  1.53704878e+01,  1.04948179e+01,
#         1.92074356e+01,  2.57586300e+03,  1.21013000e+05,  6.57547285e-01,
#         2.98661582e+00,  3.14267201e-02,  3.94328405e-01, -7.03136971e-02,
#         4.04592247e-01])


    # x0 = {coil: [3.591852189575182, 5, 18.41226242542397, 10.64284456709903, 13.976768318598277, 2568.0665571292134, 121013.0],
    #     valve: [0.7675164465562464, 3.171332962392209],
    #     fan: [0.01938953650131158, 0.4266115043400285, -0.061931356081560654, 0.4148001183354297]}
    
    # x0 = {coil: [1, 5, 18.41226242542397, 10.64284456709903, 13.976768318598277, 1000, 96000],
    #     valve: [0.8, 1],
    #     fan: [0.01938953650131158, 0.4266115043400285, -0.061931356081560654, 0.4148001183354297]}
    
    # lb = {coil: [0.5, 0.5, 1, 1, 1, 0, 1000],
    #     valve: [0.5, 0],
    #     fan: [-1, -1, -1, -1]}
    
    # ub = {coil: [10, 10, 50, 50, 50, 10000, 300000],
    #     valve: [1, 5],
    #     fan: [2, 2, 2, 2]}
    
    # targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "nominalSensibleCapacity.hasValue"],
    #                         valve: ["valveAuthority", "waterFlowRateMax"],
    #                         fan: ["c1", "c2", "c3", "c4"]}
    # targetMeasuringDevices = [model.component_dict["coil outlet air temperature sensor"],
    #                             model.component_dict["coil outlet water temperature sensor"],
    #                             model.component_dict["fan power meter"]]
    
    
    # endPeriod = datetime.datetime(year=2022, month=2, day=2, hour=23, minute=0, second=0)


    n_days = 10
    startPeriod = datetime.datetime(year=2022, month=2, day=3, hour=0, minute=0, second=0)
    startPeriod_test = datetime.datetime(year=2022, month=2, day=14, hour=0, minute=0, second=0)
    endPeriod_test = datetime.datetime(year=2022, month=2, day=28, hour=0, minute=0, second=0)
    sol_list = []
    
    endPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(1, n_days, 2)]
    startPeriod_list = [startPeriod]*len(endPeriod_list)

    # endPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(1, n_days, 2)]
    # startPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(0, n_days-1, 2)]

    print(startPeriod_list)
    print(endPeriod_list)
    

    overwrite = True
    filename = 'DryCoilEffectivenessNTU_jacobian_t.json'
    # filename = 'DryCoilEffectivenessNTU.json'
    # filename = 'test.json'

    if overwrite==False and os.path.isfile(filename):
        with open(filename, 'r') as f:
            sol_dict = json.load(f)
    else:
        sol_dict = {}
        
    print(sol_dict.keys())
    for i, (startPeriod, endPeriod) in enumerate(zip(startPeriod_list, endPeriod_list)):
        if str(i) not in sol_dict.keys(): 

            # x0 = {coil: [2/3, 3, 5, 0.8],#, 45+273.15, 12+273.15],
            #     valve: [0.8, 3],
            #     fan: [0.01938953650131158, 0.4266115043400285, -0.061931356081560654, 0.4148001183354297]}
            # lb = {coil: [0.1, 0.5, 0.5, 0.1],#, 35+273.15, 10+273.15],
            #     valve: [0.5, 0.5],
            #     fan: [-1, -1, -1, -1]}
            # ub = {coil: [2, 10, 15, 1],#60+273.15, 20+273.15],
            #     valve: [1, 10],
            #     fan: [2, 2, 2, 2]}
            # targetParameters = {coil: ["r_nominal", "m1_flow_nominal", "m2_flow_nominal", "eps_nominal"],#, "T_a1_nominal", "T_a2_nominal", "nominalSensibleCapacity.hasValue"],
            #                         valve: ["valveAuthority", "waterFlowRateMax"],
            #                         fan: ["c1", "c2", "c3", "c4"]}
            # targetMeasuringDevices = [model.component_dict["coil outlet air temperature sensor"],
            #                             model.component_dict["coil outlet water temperature sensor"],
            #                             model.component_dict["fan power meter"]]



            # x0 = {coil: [1.5, 10, 300, 30, 50, 10000],
            #     valve: [0.8, 3],
            #     fan: [0.01938953650131158, 0.4266115043400285, -0.061931356081560654, 0.4148001183354297, 0.7, 0.5]}
            
            # lb = {coil: [0.5, 0.5, 1, 1, 1, 500],
            #     valve: [0.5, 0.5],
            #     fan: [-1, -1, -1, -1, 0, 0]}
            
            # ub = {coil: [10, 15, 10000, 200, 200, 100000],
            #     valve: [1, 10],
            #     fan: [2, 2, 2, 2, 1, 1]}
            
            # x_scale = {coil: [5, 5, 10, 10, 10, 1000],
            #     valve: [10, 10],
            #     fan: [1, 1, 1, 1]}
            # targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
            #                         valve: ["valveAuthority", "waterFlowRateMax"],
            #                         fan: ["c1", "c2", "c3", "c4", "eps_motor", "f_motorToAir"]}
            # targetMeasuringDevices = [model.component_dict["coil outlet air temperature sensor"],
            #                             model.component_dict["coil outlet water temperature sensor"],
            #                             model.component_dict["fan power meter"]]

            # TEST
            x0 = {coil: [2, 4, 1],#, 45+273.15, 12+273.15],
                valve: [0.8, 5],
                fan: [0.01938953650131158, 0.4266115043400285, -0.061931356081560654, 0.4148001183354297]}
            lb = {coil: [0.5, 0.5, 0.1],#, 35+273.15, 10+273.15],
                valve: [0.5, 0.5],
                fan: [-1, -1, -1, -1]}
            ub = {coil: [10, 15, 100000],#60+273.15, 20+273.15],
                valve: [1, 10],
                fan: [2, 2, 2, 2]}
            targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "K"],#, "T_a1_nominal", "T_a2_nominal", "nominalSensibleCapacity.hasValue"],
                                    valve: ["valveAuthority", "waterFlowRateMax"],
                                    fan: ["c1", "c2", "c3", "c4"]}
            targetMeasuringDevices = [model.component_dict["coil outlet air temperature sensor"],
                                        model.component_dict["coil outlet water temperature sensor"],
                                        model.component_dict["fan power meter"]]


            y_scale = [1, 1, 200]
            estimator.estimate(x0=x0,
                                lb=lb,
                                ub=ub,
                                y_scale=y_scale,
                                trackGradients=True,
                                targetParameters=targetParameters,
                                targetMeasuringDevices=targetMeasuringDevices,
                                startPeriod=startPeriod,
                                endPeriod=endPeriod,
                                startPeriod_test=startPeriod_test,
                                endPeriod_test=endPeriod_test,
                                stepSize=stepSize)

            sol_list.append(estimator.get_solution())
            sol_dict[i] = estimator.get_solution()



            with open(filename, 'w') as fp:
                json.dump(sol_dict, fp)




if __name__ == '__main__':
    # import cProfile
    # import pstats
    # import io
    # from pstats import SortKey

  
    
    # cProfile.run("test()", 'teststats')
    # p = pstats.Stats('teststats')
    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()


    test()


