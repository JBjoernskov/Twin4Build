import sys
import os

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)

import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.constants import Constants
from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

class BuildingSpaceCo2System(building_space.BuildingSpace):

    
    def __init__(self,
                airVolume=None,
                outdoorCo2Concentration=500,
                infiltration=0.005,
                generationCo2Concentration=0.0042/1000*1.225,
                **kwargs):
        super().__init__(**kwargs)

        logger.info("[BuildingSpaceSystem] : Entered in Initialise Function")
        self.densityAir = Constants.density["air"] ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###

        # M_air = 28.9647 #g/mol
        # M_CO2 = 44.01 #g/mol
        # self.K_conversion = M_CO2/M_air
        self.outdoorCo2Concentration = outdoorCo2Concentration
        self.infiltration = infiltration
        self.generationCo2Concentration = generationCo2Concentration #kgCO2/s/person

        self.input = {'supplyAirFlowRate': None, 
                    'returnAirFlowRate': None, 
                    'numberOfPeople': None}
        self.output = {"indoorCo2Concentration": None}
        self._config = {"parameters": ["airMass",
                                        "outdoorCo2Concentration",
                                        "infiltration",
                                        "generationCo2Concentration"]}

    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["indoorCo2Concentration"] = (self.airMass*self.output["indoorCo2Concentration"] + 
                                                self.outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + self.infiltration)*stepSize + 
                                                self.generationCo2Concentration*self.input["numberOfPeople"]*stepSize)/(self.airMass + (self.input["returnAirFlowRate"]+self.infiltration)*stepSize)
                                                # self.generationCo2Concentration*self.input["numberOfPeople"]*stepSize/self.K_conversion