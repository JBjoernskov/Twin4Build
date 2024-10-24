import sys
import os
uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 4)
sys.path.append(file_path)
import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.constants import Constants
import twin4build.utils.input_output_types as tps

class BuildingSpaceOccSystem(building_space.BuildingSpace):

    
    def __init__(self,
                airVolume=None,
                outdoorCo2Concentration=400,
                infiltration=0.005,
                generationCo2Concentration=0.0042/1000*1.225,
                **kwargs):
        super().__init__(**kwargs)
        self.densityAir = Constants.density["air"] ###
        self.airVolume = airVolume ### Not currently used
        self.airMass = self.airVolume*self.densityAir ###

        # M_air = 28.9647 #g/mol
        # M_CO2 = 44.01 #g/mol
        # self.K_conversion = M_CO2/M_air
        self.outdoorCo2Concentration = outdoorCo2Concentration
        self.infiltration = infiltration
        self.generationCo2Concentration = generationCo2Concentration #kgCO2/s/person

        self.input = {'supplyAirFlowRate': tps.Scalar(), # Kg/s
                    'returnAirFlowRate': tps.Scalar(), #Not currently used # Kg/s
                    'indoorCo2Concentration': tps.Scalar()} # ppm
        self.output = {"numberOfPeople": tps.Scalar()}

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        self.output["numberOfPeople"].set(round(( self.input["supplyAirFlowRate"] * (self.input["indoorCo2Concentration"] - 
                                         self.outdoorCo2Concentration) ) / self.generationCo2Concentration))
        self.output["numberOfPeople"].set(max(0, self.output["numberOfPeople"]))
