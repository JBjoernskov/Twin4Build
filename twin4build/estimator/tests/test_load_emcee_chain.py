import matplotlib.pyplot as plt
import matplotlib
import pickle
import math
import numpy as np
import os
import datetime
import sys
import corner
import seaborn as sns
from dateutil import tz
from io import StringIO
from matplotlib.colors import LinearSegmentedColormap
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.utils.uppath import uppath
from twin4build.simulator.simulator import Simulator
from twin4build.model.model import Model
from twin4build.model.tests.test_LBNL_bypass_coil_model import fcn
import twin4build.utils.plot.plot as plot

def test_load_emcee_chain():
    # flat_attr_list = ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "workingPressure.hasValue", "flowCoefficient.hasValue", "waterFlowRateMax", "c1", "c2", "c3", "c4", "eps_motor", "f_motorToAir", "kp", "Ti", "Td"]
    # flat_attr_list = [r"$\dot{m}_{w,nom}$", r"$\dot{m}_{a,nom}$", r"$\tau_1$", r"$\tau_2$", r"$\tau_m$", r"$UA_{nom}$", r"$\Delta P_{sys}$", r"$K_{v}$", r"$\dot{m}_{w,nom}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$\epsilon$", r"$f_{motorToAir}$", r"$K_p$", r"$T_i$", r"$T_d$"]
    # flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\Delta P_{fixed}$", r"$\dot{m}_{v,w,nom}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$"]
    # flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{pump,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$T_{rise}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$"]
    # flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{p,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$\Delta P_{p,nom}$", r"$\Delta P_{v,nom}$", r"$\Delta P_{sys}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$"]
    # flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{p,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$\Delta P_{p,nom}$", r"$\Delta P_{sys}$", r"$T_{rise}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$"]
    # flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{pump,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$"]
    flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{pump,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$\Delta P_{pump}$", r"$\Delta P_{sys}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$", ]
    flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{pump,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$\Delta P_{pump}$", r"$\Delta P_{sys}$", r"$T_{w,inlet}$", r"$T_{w,outlet}$", r"$T_{a,outlet}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$", "a1", "tau1", "a2", "tau2", "a3", "tau3", "a4", "tau4", "a5", "tau5"]
    flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{pump,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$\Delta P_{pump}$", r"$\Delta P_{sys}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$", r"$a_1$", r"$tau_1$", r"$a_2$", r"$tau_2$", "a3", "tau3", "a4", "tau4", "a5", "tau5"]
    flat_attr_list = [r"$\dot{m}_{c,w,nom}$", r"$\dot{m}_{c,a,nom}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$UA_{nom}$", r"$\dot{m}_{v,w,nom}$", r"$\dot{m}_{pump,w,nom}$", r"$\Delta P_{check}$", r"$\Delta P_{coil}$", r"$\Delta P_{pump}$", r"$\Delta P_{sys}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{comb}$", r"$K_p$", r"$T_i$", r"$T_d$"]
    
    


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
    plot.load_params()

    do_iac_plot = False
    do_logl_plot = True
    do_trace_plot = True
    do_swap_plot = False
    do_jump_plot = False
    do_corner_plot = True
    do_inference = False
    assume_uncorrelated_noise = False

    assert (do_iac_plot and do_inference)!=True

    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230829_155706_chain_log.pickle")
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230830_194210_chain_log.pickle")
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230902_183719_chain_log.pickle")
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230904_171903_chain_log.pickle")
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230905_092246_chain_log.pickle")
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230907_160103_chain_log.pickle")
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230908_114136_chain_log.pickle")
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230908_233103_chain_log.pickle") #No flow dependence
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230911_113906_chain_log.pickle") #No flow dependence
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230912_120849_chain_log.pickle") #No flow dependence
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230913_093046_chain_log.pickle") #No flow dependence
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230914_164406_chain_log.pickle") #No flow dependence
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230915_091654_chain_log.pickle") #No flow dependence
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230923_164550_chain_log.pickle") #T_max=1e+5
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230923_192545_chain_log.pickle") #T_max=inf
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230923_194507_chain_log.pickle") #T_max=inf, Tau=300
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230923_234059_chain_log.pickle") #T_max=inf, 8*walkers
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230925_102035_chain_log.pickle") #Tinf_fanLimits_coilFlowDependent , 8*walkers
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230926_131018_chain_log.pickle") #10 temps , 4*walkers
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230926_225917_chain_log.pickle") #10 temps , 4*walkers
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230927_135449_chain_log.pickle") #10 temps , 4*walkers, 30tau
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230927_154822_chain_log.pickle") #1 temps , 30*walkers, 30tau
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230928_102504_chain_log.pickle") #10 temps , 4*walkers, 30tau
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230928_124040_chain_log.pickle") #15 temps , 8*walkers, 30tau
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230929_101059_chain_log.pickle") #15 temps , 8*walkers, 30tau, large water massflow         CURRENT BEST
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230930_175014_chain_log.pickle") #15 temps , 8*walkers, 200tau, large water massflow
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20230930_181209_chain_log.pickle") #15 temps , 8*walkers, 200tau, large water massflow, gaussian x0
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231001_162530_chain_log.pickle") #1 temps , 500walkers, 200tau, large water massflow
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231002_141008_chain_log.pickle") #12 temps , 8*walkers, 30tau, large water massflow
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231002_160750_chain_log.pickle") #12 temps , 8*walkers, 30tau, large water massflow
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231005_134721_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231005_215753_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, large massflow
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231009_132524_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, change prior
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231009_153513_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, change prior, lower UA
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231010_120630_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, change prior, GlycolEthanol
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231011_131932_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, change prior, GlycolEthanol, valve more parameters
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231012_154701_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, change prior, GlycolEthanol, valve more parameters, lower UA
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231013_123013_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, change prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231017_074841_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, change prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231018_092240_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 2), "chain_logs", "20231018_135249_chain_log.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231205_110432_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231205_110432_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231205_164843_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231206_131318_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231206_212149_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231207_160247_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231208_160545_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231219_155600_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20231229_103204_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240102_141037_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240103_140207_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240104_094246_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240104_171830_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240105_165854_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240107_224328_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240107_224328_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240108_175437_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240109_110253_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240109_143730_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240110_093807_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240110_141839_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240110_174122_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240108_175437_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240111_114834_.pickle") #15 temps , 8*walkers, 30tau, test bypass valve, lower massflow and pressure, gaussian prior, GlycolEthanol, valve more parameters, lower UA, lower massflow, Kp
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240111_164945_.pickle") # assume_uncorrelated_noise = True, uniform prior
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240112_120101_.pickle") # assume_uncorrelated_noise = False, gaussian prior, Exp-squared
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240115_135515_.pickle") # assume_uncorrelated_noise = False, uniform prior, Exp-squared
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240116_085308_.pickle") # assume_uncorrelated_noise = False, uniform prior, Matern 3/2
    # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240117_164040_.pickle") # assume_uncorrelated_noise = False, gaussian model prior, uniform noise prior, Matern 3/2
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240121_111340_.pickle") # assume_uncorrelated_noise = False, uniform prior, Matern 5/2
    loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "model_parameters", "chain_logs", "model_20240122_084923_.pickle") # assume_uncorrelated_noise = False, gaussian model prior, uniform noise prior, Exp-squared


    with open(loaddir, 'rb') as handle:
        result = pickle.load(handle)
        result["chain.T"] = 1/result["chain.betas"] ##################################
    
    burnin = int(result["chain.x"].shape[0])-50 #800
    #########################################
    list_ = ["integratedAutoCorrelatedTime", "chain.jumps_accepted", "chain.jumps_proposed", "chain.swaps_accepted", "chain.swaps_proposed"]
    for key in list_:
        result[key] = np.array(result[key])
    #########################################

    vmin = np.min(result["chain.betas"])
    vmax = np.max(result["chain.betas"])


    # for key in result.keys():
    #     if key not in list_:
    #     result[key] = np.array(result[key])


    ndim = result["chain.x"].shape[3]
    ntemps = result["chain.x"].shape[1]
    nwalkers = result["chain.x"].shape[2] #Round up to nearest even number and multiply by 2


    # startTime = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    # endTime = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime = datetime.datetime(year=2022, month=2, day=4, hour=10, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2022, month=2, day=4, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    stepSize = 60
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False, fcn=fcn)
    simulator = Simulator(model)


    coil = model.component_dict["coil+pump+valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]


    targetParameters = {
                # coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dp1_nominal", "dpPump", "dpSystem", "tau_w_inlet", "tau_w_outlet", "tau_air_outlet"],
                coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dp1_nominal", "dpPump", "dpSystem"],
                fan: ["c1", "c2", "c3", "c4", "f_total"],
                controller: ["kp", "Ti", "Td"]}
            
    percentile = 2
    # targetMeasuringDevices = {model.component_dict["valve position sensor"]: {"standardDeviation": 0.01/percentile},
    #                         model.component_dict["coil inlet water temperature sensor"]: {"standardDeviation": 0.5/percentile},
    #                         model.component_dict["coil outlet water temperature sensor"]: {"standardDeviation": 0.5/percentile},
    #                             model.component_dict["coil outlet air temperature sensor"]: {"standardDeviation": 0.5/percentile},
    #                             model.component_dict["fan power meter"]: {"standardDeviation": 80/percentile}}
    
    targetMeasuringDevices = {model.component_dict["coil outlet air temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                model.component_dict["coil outlet water temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                model.component_dict["fan power meter"]: {"standardDeviation": 80/percentile},
                                model.component_dict["valve position sensor"]: {"standardDeviation": 0.01/percentile},
                                model.component_dict["coil inlet water temperature sensor"]: {"standardDeviation": 0.5/percentile}}

    n_par = 0
    n_par_map = {}
    # Get number of gaussian process parameters
    for j, measuring_device in enumerate(targetMeasuringDevices):
        source_component = [cp.connectsSystemThrough.connectsSystem for cp in measuring_device.connectsAt][0]
        n_par += len(source_component.input)+1
        n_par_map[measuring_device.id] = len(source_component.input)+1
    print(n_par)
    print(n_par_map)

        
    if assume_uncorrelated_noise==False:
        for j, measuring_device in enumerate(targetMeasuringDevices):
            for i in range(n_par_map[measuring_device.id]):
                if i==0:
                    s = f"$a_{str(j)}$"
                    s = r'{}'.format(s)
                    flat_attr_list.append(s)
                else:
                    s = r'$l_{%.0f,%.0f}$' % (j,i, )
                    flat_attr_list.append(s)

        # result["stepSize_train"] = stepSize
        # result["startTime_train"] = startTime
        # result["endTime_train"] = endTime

        # standardDeviation = np.array([0.01/percentile, 0.5/2, 0.5/2, 0.5/2, 80/2])


        # flat_component_list = [obj.id for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
        # flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

        # result["component_list"] = flat_component_list
        # result["attr_list"] = flat_attr_list

        # with open(loaddir, 'wb') as handle:
        #     pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if do_inference:
        model.load_chain_log(loaddir)
        startTime_train = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train = datetime.datetime(year=2022, month=2, day=3, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        model.chain_log["startTime_train"] = startTime_train
        model.chain_log["endTime_train"] = endTime_train
        # model.chain_log["stepSize_train"] = stepSize
        # model.chain_log["n_par"] = n_par
        # model.chain_log["n_par_map"] = n_par_map
        parameter_chain = result["chain.x"][burnin:,0,:,:]
        del result
        del model.chain_log["chain.x"]
        
        assert len(flat_attr_list) == ndim, f"Number of parameters in flat_attr_list ({len(flat_attr_list)}) does not match number of parameters in chain.x ({ndim})"
        # parameter_chain = result["chain.x"][-1:,0,:,:] #[-1:,0,:,:]
        parameter_chain = parameter_chain.reshape((parameter_chain.shape[0]*parameter_chain.shape[1], parameter_chain.shape[2]))
        fig, axes = simulator.run_emcee_inference(model, parameter_chain, targetParameters, targetMeasuringDevices, startTime, endTime, stepSize, assume_uncorrelated_noise=assume_uncorrelated_noise)
        ylabels = [r"$u_v [1]$", r"$T_{c,w,in} [^\circ\!C]$", r"$T_{c,w,out} [^\circ\!C]$", r"$T_{c,a,out} [^\circ\!C]$", r"$\dot{P}_f [W]$"]
        fig.subplots_adjust(hspace=0.3)
        fig.set_size_inches((15,10))
        for ax, ylabel in zip(axes, ylabels):
            # ax.legend(loc="center left", bbox_to_anchor=(1,0.5), prop={'size': 12})
            pos = ax.get_position()
            pos.x0 = 0.15       # for example 0.2, choose your value
            pos.x1 = 0.99       # for example 0.2, choose your value

            ax.set_position(pos)
            ax.tick_params(axis='y', labelsize=10)
            # ax.locator_params(axis='y', nbins=3)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.text(-0.07, 0.5, ylabel, fontsize=14, rotation="horizontal", ha="right", transform=ax.transAxes)
            ax.xaxis.label.set_color("black")
        # axes[3].plot(simulator.dateTimeSteps, model.component_dict["Supply air temperature setpoint"].savedOutput["scheduleValue"], color="blue", label="setpoint", linewidth=0.5)
        # axes[3].plot(simulator.dateTimeSteps, model.component_dict["fan inlet air temperature sensor"].get_physical_readings(startTime, endTime, stepSize)[0:-1], color="green", label="inlet air", linewidth=0.5)
        # fig.savefig(r'C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\LBNL_inference_plot.png', dpi=300)
        # ax.plot(simulator.dateTimeSteps, simulator.model.component_dict[])


    


    assert len(flat_attr_list) == ndim, f"Number of parameters in flat_attr_list ({len(flat_attr_list)}) does not match number of parameters in chain.x ({ndim})"
    
    plt.rcParams['mathtext.fontset'] = 'cm'

    nparam = len(flat_attr_list)
    ncols = 4
    nrows = math.ceil(nparam/ncols)
    
    
    
    # cm = plt.get_cmap('RdYlBu', ntemps)
    # cm_sb = sns.color_palette("vlag_r", n_colors=ntemps, center="dark") #vlag_r
    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark") #vlag_r
    cm_sb_rev = list(reversed(cm_sb))
    cm_mpl = LinearSegmentedColormap.from_list("seaborn", cm_sb)#, N=ntemps)
    cm_mpl_rev = LinearSegmentedColormap.from_list("seaborn_rev", cm_sb_rev, N=ntemps)
    

    # list_ = ["chain.logl", "chain.logP", "chain.x", "chain.betas"]
    # for key in list_:
    #     for i, arr in enumerate(result[key]):
    #         result[key][i] = arr[-nsample_checkpoint:]
        
    # for key in result.keys():
    #     result[key] = np.concatenate(result[key],axis=0)
        # result["chain.jumps_accepted"].append(chain.jumps_accepted)
        # result["chain.jumps_proposed"].append(chain.jumps_proposed)
        # result["chain.logl"].append(chain.logl)
        # result["chain.logP"].append(chain.logP)
        # result["chain.swaps_accepted"].append(chain.swaps_accepted)
        # result["chain.swaps_proposed"].append(chain.swaps_proposed)
        # result["chain.x"].append(chain.x)
        # result["chain.betas"].append(chain.betas)

    # vmin = np.min(result["chain.T"])
    # vmax = np.max(result["chain.T"])
    
    



    if do_iac_plot:
        fig_iac, axes_iac = plt.subplots(nrows=nrows, ncols=ncols, layout='compressed')
        fig_iac.set_size_inches((17, 12))
        fig_iac.suptitle("Integrated AutoCorrelated Time", fontsize=20)
        iac = result["integratedAutoCorrelatedTime"][:-1]
        n_it = iac.shape[0]
        for i in range(ntemps):
            beta = result["chain.betas"][:, i]
            for j, attr in enumerate(flat_attr_list):
                row = math.floor(j/ncols)
                col = int(j-ncols*row)
                
                if ntemps>1:
                    sc = axes_iac[row, col].plot(range(n_it), iac[:,i,j], color=cm_sb[i], alpha=1)
                else:
                    sc = axes_iac[row, col].plot(range(n_it), iac[:,i,j], color=cm_sb[i], alpha=1)
        
        # add heristic tau = N/50 line
        heuristic_line = np.arange(n_it)/50
        for j, attr in enumerate(flat_attr_list):
            row = math.floor(j/ncols)
            col = int(j-ncols*row)
            axes_iac[row, col].plot(range(n_it), heuristic_line, color="black", linestyle="dashed", alpha=1, label=r"$\tau=N/50$")
        fig_iac.legend()
        
    if do_logl_plot:
        fig_logl, ax_logl = plt.subplots(layout='compressed')
        fig_logl.set_size_inches((17/4, 12/4))
        fig_logl.suptitle("Log-likelihood", fontsize=20)
        logl = np.abs(result["chain.logl"])
        logl[logl>1e+9] = np.nan
        n_it = result["chain.logl"].shape[0]
        for i_walker in range(nwalkers):
            for i in range(ntemps):
                if i==0: #######################################################################
                    ax_logl.plot(range(n_it), logl[:,i,i_walker], color=cm_sb[i])
                    ax_logl.set_yscale('log')

    
    if do_trace_plot:
        fig_trace_beta, axes_trace = plt.subplots(nrows=nrows, ncols=ncols, layout='compressed')
        fig_trace_beta.set_size_inches((17, 12))
        chain_logl = result["chain.logl"]
        bool_ = chain_logl<-5e+9
        chain_logl[bool_] = np.nan
        chain_logl[np.isnan(chain_logl)] = np.nanmin(chain_logl)

        for nt in reversed(range(ntemps)):
            for nw in range(nwalkers):
                x = result["chain.x"][:, nt, nw, :]
                T = result["chain.T"][:, nt]
                beta = result["chain.betas"][:, nt]
                logl = chain_logl[:, nt, nw]
                # alpha = (max_alpha-min_alpha)*(logl-logl_min)/(logl_max-logl_min) + min_alpha
                # alpha = (max_alpha-min_alpha)*(T-vmin)/(vmax-vmin) + min_alpha
                # alpha = (max_alpha-min_alpha)*(beta-vmin)/(vmax-vmin) + min_alpha
                # Trace plots
                
                
                for j, attr in enumerate(flat_attr_list):
                    row = math.floor(j/ncols)
                    col = int(j-ncols*row)
                    # sc = axes_trace[row, col].scatter(range(x[:,j].shape[0]), x[:,j], c=T, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), s=0.3, cmap=cm_mpl, alpha=0.1)
                    if ntemps>1:
                        sc = axes_trace[row, col].scatter(range(x[:,j].shape[0]), x[:,j], c=beta, vmin=vmin, vmax=vmax, s=0.3, cmap=cm_mpl_rev, alpha=0.1)
                    else:
                        sc = axes_trace[row, col].scatter(range(x[:,j].shape[0]), x[:,j], s=0.3, color=cm_sb[0], alpha=0.1)
                        
                    axes_trace[row, col].axvline(burnin, color="black", linestyle="--", linewidth=2, alpha=0.8)

                    # if plotted==False:
                    #     axes_trace[row, col].text(x_left+dx/2, 0.44, 'Burnin', ha='center', va='center', rotation='horizontal', fontsize=15, transform=axes_trace[row, col].transAxes)
                    #     axes_trace[row, col].arrow(x_right, 0.5, -dx, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
                    #     axes_trace[row, col].arrow(x_left, 0.5, dx, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
                    #     axes_trace[row, col].set_ylabel(attr, fontsize=20)
                    #     plotted = True



        x_left = 0.1
        x_mid_left = 0.515
        x_right = 0.9
        x_mid_right = 0.58
        dx_left = x_mid_left-x_left
        dx_right = x_right-x_mid_right

        fontsize = 12
        for j, attr in enumerate(flat_attr_list):
            row = math.floor(j/ncols)
            col = int(j-ncols*row)
            axes_trace[row, col].axvline(burnin, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
            axes_trace[row, col].text(x_left+dx_left/2, 0.44, 'Burnin', ha='center', va='center', rotation='horizontal', fontsize=fontsize, transform=axes_trace[row, col].transAxes)
            # axes_trace[row, col].arrow(x_mid_left, 0.5, -dx_left, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
            # axes_trace[row, col].arrow(x_left, 0.5, dx_left, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)

            axes_trace[row, col].text(x_mid_right+dx_right/2, 0.44, 'Posterior', ha='center', va='center', rotation='horizontal', fontsize=fontsize, transform=axes_trace[row, col].transAxes)
            # axes_trace[row, col].arrow(x_right, 0.5, -dx_right, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
            # axes_trace[row, col].arrow(x_mid_right, 0.5, dx_right, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
            axes_trace[row, col].set_ylabel(attr, fontsize=20)

            # arrow = axes_trace[row, col].annotate('', 
            #                                     xy =(x_left, 0.5),
            #                                     xytext =(x_mid_left, 0.5), 
            #                                     arrowprops = dict(
            #                                         arrowstyle="|-|,widthA=0.7, widthB=0.7"
            #                                     ))
            
            # arrow = axes_trace[row, col].annotate('', 
            #                                     xy =(x_mid_right, 0.5),
            #                                     xytext =(x_right, 0.5), 
            #                                     arrowprops = dict(
            #                                         arrowstyle="|-|,widthA=0.7, widthB=0.7"
            #                                     ))
                                            
        # fig_trace.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.1), ncol=len(labels))#, bbox_transform=fig.transFigure)
        if ntemps>1:
            cb = fig_trace_beta.colorbar(sc, ax=axes_trace)
            cb.set_label(label=r"$T$", size=30)#, weight='bold')
            cb.solids.set(alpha=1)
            # fig_trace_beta.tight_layout()
            dist = (vmax-vmin)/(ntemps)/2
            tick_start = vmin+dist
            tick_end = vmax-dist
            tick_locs = np.linspace(tick_start, tick_end, ntemps)[::-1]
            cb.set_ticks(tick_locs)
            labels = list(result["chain.T"][0,:])
            inf_label = r"$\infty$"
            labels[-1] = inf_label
            ticklabels = [str(round(float(label), 1)) if isinstance(label, str)==False else label for label in labels] #round(x, 2)
            cb.set_ticklabels(ticklabels, size=12)

            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(12)
                txt = tick.get_text()
                if txt==inf_label:
                    tick.set_fontsize(20)
                    # tick.set_text()
                    # tick.set_ha("center")
                    # tick.set_va("center_baseline")
            # fig_trace_beta.savefig(r'C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\LBNL_trace_plot.png', dpi=300)

    if do_swap_plot and ntemps>1:
        fig_swap, ax_swap = plt.subplots(layout='compressed')
        fig_swap.set_size_inches((17, 12))
        fig_swap.suptitle("Swaps", fontsize=20)
        n = ntemps-1
        for i in range(n):
            if i==0: #######################################################################
                ax_swap.plot(range(result["chain.swaps_accepted"][:,i].shape[0]), result["chain.swaps_accepted"][:,i]/result["chain.swaps_proposed"][:,i], color=cm_sb[i])


    if do_jump_plot:
        a = result["chain.jumps_accepted"]/result["chain.jumps_proposed"]
        fig_jump, ax_jump = plt.subplots(layout='compressed')
        fig_jump.set_size_inches((17, 12))
        fig_jump.suptitle("Jumps", fontsize=20)
        # n_checkpoints = result["chain.jumps_proposed"].shape[0]
        # for i_checkpoint in range(n_checkpoints):
        #     for i in range(ntemps):
        #         ax_jump.scatter([i_checkpoint]*nwalkers, result["chain.jumps_accepted"][i_checkpoint,i,:]/result["chain.jumps_proposed"][i_checkpoint,i,:], color=cm_sb[i], s=20, alpha=1)

        n_it = result["chain.jumps_proposed"].shape[0]
        for i_walker in range(nwalkers):
            for i in range(ntemps):
                if i==0: #######################################################################
                    ax_jump.plot(range(n_it), result["chain.jumps_accepted"][:,i,i_walker]/result["chain.jumps_proposed"][:,i,i_walker], color=cm_sb[i])

    if do_corner_plot:
        # fig_corner, axes_corner = plt.subplots(nrows=ndim, ncols=ndim, layout='compressed')
        
        parameter_chain = result["chain.x"][burnin:,0,:,:]
        parameter_chain = parameter_chain.reshape(parameter_chain.shape[0]*parameter_chain.shape[1],parameter_chain.shape[2])
        fig_corner = corner.corner(parameter_chain, fig=None, labels=flat_attr_list, labelpad=-0.2, show_titles=True, color=cm_sb[0], plot_contours=True, bins=15, hist_bin_factor=5, max_n_ticks=3, quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 10, "ha": "left", "position": (0.03, 1.01)})
        fig_corner.set_size_inches((12, 12))
        pad = 0.025
        fig_corner.subplots_adjust(left=pad, bottom=pad, right=1-pad, top=1-pad, wspace=0.08, hspace=0.08)
        axes = fig_corner.get_axes()
        for ax in axes:
            ax.set_xticks([], minor=True)
            ax.set_xticks([])
            ax.set_yticks([], minor=True)
            ax.set_yticks([])

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

        median = np.median(parameter_chain, axis=0)
        corner.overplot_lines(fig_corner, median, color=red, linewidth=0.5)
        corner.overplot_points(fig_corner, median.reshape(1,median.shape[0]), marker="s", color=red)
        # fig_corner.savefig(r'C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\LBNL_corner_plot.png', dpi=300)
    # color = cm(1)
    # fig_trace_loglike, axes_trace_loglike = plt.subplots(nrows=1, ncols=1)
    # fig_trace_loglike.set_size_inches((17, 12))
    # fig_trace_loglike.suptitle("Trace plots of log likelihoods")
    # vmin = np.nanmin(-chain_logl)
    # vmax = np.nanmax(-chain_logl)
    # for nt in range(1):
    #     for nw in range(nwalkers):
    #         logl = chain_logl[:, nt, nw]
    #         axes_trace_loglike.scatter(range(logl.shape[0]), -logl, color=color, s=4, alpha=0.8)
    # axes_trace_loglike.set_yscale("log")
    # plt.show()
    
    plt.show()


if __name__=="__main__":
    test_load_emcee_chain()