import twin4build.saref4bldg.building_space.building_space as building_space
from twin4build.utils.constants import Constants


class BuildingSpaceModel(building_space.BuildingSpace):
    
    def __init__(self,
                airVolume=None,
                **kwargs):
        super().__init__(**kwargs)

        self.densityAir = Constants.density["air"] ###
        self.airVolume = airVolume ###
        self.airMass = self.airVolume*self.densityAir ###

        self.input = {'supplyAirFlowRate': None, 
                    'supplyDamperPosition': None, 
                    'returnAirFlowRate': None, 
                    'exhaustDamperPosition': None, 
                    'valvePosition': None, 
                    'shadePosition': None, 
                    'supplyAirTemperature': None, 
                    'supplyWaterTemperature': None, 
                    'globalIrradiation': None, 
                    'outdoorTemperature': None, 
                    'numberOfPeople': None}
        self.output = {"indoorTemperature": None, "indoorCo2Concentration": None}

    def initialize(self,
                    startPeriod=None,
                    endPeriod=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        M_air = 28.9647 #g/mol
        M_CO2 = 44.01 #g/mol
        K_conversion = M_CO2/M_air*1e-6
        outdoorCo2Concentration = 400
        infiltration = 0.07
        generationCo2Concentration = 0.000008316
        self.output["indoorCo2Concentration"] = (self.airMass*self.output["indoorCo2Concentration"] + 
                                                outdoorCo2Concentration*(self.input["supplyAirFlowRate"] + infiltration)*stepSize + 
                                                generationCo2Concentration*self.input["numberOfPeople"]*stepSize/K_conversion)/(self.airMass + (self.input["returnAirFlowRate"]+infiltration)*stepSize)