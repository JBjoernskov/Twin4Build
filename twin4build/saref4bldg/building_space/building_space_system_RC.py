import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.constants import Constants
from twin4build.utils.uppath import uppath
from twin4build.logger.Logging import Logging
import do_mpc
logger = Logging.get_logger("ai_logfile")

class BuildingSpaceSystem(building_space.BuildingSpace):
    def __init__(self,
                airVolume=None,
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[Building Space Model Class] : Entered Initiate Function")

        self.densityAir = Constants.density["air"] ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###
        self.C_r = 1e+5
        self.C_w = 1e+5
        self.fSolarIrradiation = 0.1

        self.input = {'supplyAirFlowRate': None, 
                    'returnAirFlowRate': None, 
                    'heatAddedBySpaceHeater': None, 
                    'supplyAirTemperature': None,
                    'globalIrradiation': None, 
                    'outdoorTemperature': None, 
                    'numberOfPeople': None}
        self.output = {"indoorTemperature": None, 
                       "indoorCo2Concentration": None}        
        logger.info("[Building Space Model Class] : Exited from Initiate Function")

    def continous_model(self, model=None):
        model_type = 'continuous' # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type)

        for var_name in self.input.keys():
            model.set_variable(var_type='_z', var_name=var_name)

        for var_name in self.input.keys():
            model.set_variable(var_type='_z', var_name=var_name)

        model.set_variable(var_type="_x", var_name="indoorTemperature")
        model.set_variable(var_type="_x", var_name="d_indoorTemperature")

        model.set_variable(var_type="_x", var_name="indoorCo2Concentration")
        model.set_variable(var_type="_x", var_name="wallTemperature")
        model.set_variable(var_type="_z", var_name="Q_in")
        model.set_variable(var_type="_z", var_name="Q_out")

        model.set_rhs('Q_in', 
                      model.z["supplyAirFlowRate"]*model.z["supplyAirTemperature"]+
                      model.z["heatAddedBySpaceHeater"]+
                      self.fSolarIrradiation*model.z["globalIrradiation"])
        model.set_rhs("Q_out", model.z["supplyAirFlowRate"]*model.x["indoorTemperature"])

        model.z
    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        pass

