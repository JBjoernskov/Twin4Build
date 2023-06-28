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
import matplotlib.patches as mpatches
# Only for testing before distributing package.
# If the package is installed, this is not needed.
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.utils.rsetattr import rsetattr  
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node
from twin4build.logger.Logging import Logging
from twin4build.model.tests.test_LBNL_model import extend_model
from twin4build.utils.plot.plot import load_params
from twin4build.monitor.monitor import Monitor
logger = Logging.get_logger("ai_logfile")

def test():
    colors = sns.color_palette("deep")
    blue = colors[0]
    orange = colors[1]
    green = colors[2]
    red = colors[3]
    purple = colors[4]
    brown = colors[5]
    pink = colors[6]
    grey = colors[7]
    beis = colors[8]
    sky_blue = colors[9]
    load_params()
    stepSize = 60
    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)
    estimator = Estimator(model)

    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]


    n_days = 10
    startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
    startPeriod_test = datetime.datetime(year=2022, month=2, day=1, hour=1, minute=0, second=0) 
    endPeriod_test = datetime.datetime(year=2022, month=2, day=28, hour=0, minute=0, second=0)
    sol_list = []
    startPeriod_list = [startPeriod]*(n_days-1)
    endPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(1, n_days)]
    
    
    color_list = [blue, red]
    filename_list = ["DryCoilEffectivenessNTU.json", "DryCoilDiscretized_jacobian_Feb_1day.json"]
    legend_name_list = ["Estimated jacobian", "Calculated jacobian"]
    width_list = [-0.4, 0.4]
    
    id_list = ["fan power meter", "coil outlet air temperature sensor", "coil outlet water temperature sensor"]
    
    for com in id_list:
        fig,ax = plt.subplots()
        patch_list = []
        ax.set_title(com)
        for color, filename, legend_name, width in zip(color_list, filename_list, legend_name_list, width_list):
            patch_list.append(mpatches.Patch(color=color, label=legend_name))
            with open(filename, 'r') as f:
                sol = json.load(f)
            
            for i,result in sol.items():
                ax.bar(i,result["RMSE"][com], color=color, align="edge", width=width)
        ax.legend(handles=patch_list)
    
    fig,ax = plt.subplots()
    patch_list = []
    ax.set_title("Objective evaluations")
    for color, filename, legend_name, width in zip(color_list, filename_list, legend_name_list, width_list):
        patch_list.append(mpatches.Patch(color=color, label=legend_name))
        with open(filename, 'r') as f:
            sol = json.load(f)
        
        for i,result in sol.items():
            ax.bar(i,result["n_obj_eval"], color=color, align="edge", width=width)
    ax.legend(handles=patch_list)
    
    plt.show()

def test_m():

    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)
    estimator = Estimator(model)

    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]

    filename = 'DryCoilDiscretized_Jacobian_Feb_1day.json'
    # filename = 'DryCoilEffectivenessNTU_Jacobian.json'

    with open(filename, 'r') as f:
        sol_dict = json.load(f)

    print(sol_dict.keys())
    n = 0
    print(sol_dict[str(n)])

    # x = {coil: sol_dict[str(n)]["coil"],
    #     valve: sol_dict[str(n)]["valve"],
    #     fan: sol_dict[str(n)]["fan"]}
    # x_flat = [val for lst in x.values() for val in lst]
    # targetParameters = {coil: ["r_nominal", "nominalSensibleCapacity.hasValue", "m1_flow_nominal", "m2_flow_nominal", "T_a1_nominal", "T_a2_nominal"],
    #                                 valve: ["valveAuthority", "waterFlowRateMax"],
    #                                 fan: ["c1", "c2", "c3", "c4"]}
    
    #234
    # sol_dict[str(n)]["coil"][0] = 5
    # sol_dict[str(n)]["coil"][5] = 50000
    x = {coil: sol_dict[str(n)]["coil"],
        valve: sol_dict[str(n)]["valve"],
        fan: sol_dict[str(n)]["fan"]}
    x_flat = [val for lst in x.values() for val in lst]
    targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
                            valve: ["valveAuthority", "waterFlowRateMax"],
                            fan: ["c1", "c2", "c3", "c4"]}

#     x_flat = [ 1.40353308e+00,  1.06362295e+00,  2.70573159e+02,  1.82713852e+00,
#   1.12615421e+01,  2.51661829e+03,  6.76750235e-01,  4.12863219e+00,
#   9.11228031e-02, -3.73065645e-02,  8.54549554e-01, -8.19300187e-02,
#   9.99919185e-01,  9.99999364e-01,]
    
#     x_flat = [ 1.40584326e+00,  5.04118780e-01,  2.71262736e+02,  3.47480993e+00,
#   1.44971271e+00,  2.51854718e+03,  5.00396431e-01,  3.60620605e+00,
#   1.05132916e-01, -1.83889218e-01,  1.31465538e+00, -5.24579640e-01,
#   9.99998881e-01,  9.99991603e-01,] # No flow dependent
    
#     x_flat = [ 5.08853472e-01,  8.29269529e+00,  2.31854242e+02,  1.00002240e+00,
#   1.28782192e+02,  3.33797557e+03,  5.05317854e-01,  3.53916841e+00,
#   1.05749371e-01, -1.87608034e-01,  1.32248188e+00, -5.30335304e-01,
#   3.58337208e-01,  9.99999660e-01] #flow dependent
    
#     x_flat = [ 6.54338088e-01,  9.59287917e+00,  1.96084884e+02,  1.02024368e+00,
#   1.09087869e+02,  3.84930042e+03,  5.04976787e-01,  3.54456981e+00,
#   1.06572615e-01, -1.96536741e-01,  1.35053465e+00, -5.57396259e-01,
#   5.04569992e-03,  9.99996673e-01,]

    flat_component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
    flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

    for i, (obj, attr) in enumerate(zip(flat_component_list, flat_attr_list)):
        rsetattr(obj, attr, x_flat[i])

    stepSize = 60
    startPeriod_test = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0)
    endPeriod_test = datetime.datetime(year=2022, month=2, day=28, hour=16, minute=0, second=0)
    monitor = Monitor(model)
    monitor.monitor(startPeriod=startPeriod_test,
                        endPeriod=endPeriod_test,
                        stepSize=stepSize,
                        do_plot=True)
    plt.show()

if __name__ == '__main__':
    # import cProfile
    # import pstats
    # import io
    # from pstats import SortKey
    # cProfile.run("test()", 'teststats')
    # p = pstats.Stats('teststats')
    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
    test_m()


